use crate::interface::components::*;
use crate::interface::select_history;
use bevy::prelude::*;

const MAX_ACC: f32 = 1.5e-3;
const MAX_EXTRAPOLATION_INTERVALS: f64 = 2.0;

fn validate_acceleration(acceleration: Vec3) -> Option<Vec3> {
    let magnitude = acceleration.length();
    if acceleration.is_finite() && magnitude.is_finite() && magnitude <= MAX_ACC {
        Some(acceleration)
    } else {
        None
    }
}

fn hermite_vector(a: Vec3, b: Vec3, tangent_a: Vec3, tangent_b: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * a
        + (t3 - 2.0 * t2 + t) * tangent_a
        + (-2.0 * t3 + 3.0 * t2) * b
        + (t3 - t2) * tangent_b
}

/// Interpolates completed body-frame GPU samples and performs only a bounded,
/// slope-limited extrapolation past the newest result. Unbounded cubic
/// extrapolation is deliberately avoided because asynchronous readback can
/// occasionally skip render frames.
fn predict_body_acceleration(
    history: &GravitySampleHistory,
    epoch: u64,
    target_time: f64,
    maximum_extrapolation_intervals: f64,
) -> Option<Vec3> {
    let samples: Vec<&GravityFieldSample> = history
        .samples
        .iter()
        .filter(|sample| sample.snapshot.epoch == epoch)
        .collect();
    let latest = *samples.last()?;
    if samples.len() == 1 {
        return Some(latest.body_acceleration);
    }

    if target_time <= latest.snapshot.simulation_time_seconds {
        let upper_index = samples
            .iter()
            .position(|sample| sample.snapshot.simulation_time_seconds >= target_time)
            .unwrap_or(samples.len() - 1);
        if upper_index == 0 {
            return Some(samples[0].body_acceleration);
        }
        let lower = samples[upper_index - 1];
        let upper = samples[upper_index];
        let interval = (upper.snapshot.simulation_time_seconds
            - lower.snapshot.simulation_time_seconds)
            .max(f64::EPSILON);
        let u = ((target_time - lower.snapshot.simulation_time_seconds) / interval).clamp(0.0, 1.0)
            as f32;

        let previous = samples
            .get(upper_index.saturating_sub(2))
            .copied()
            .unwrap_or(lower);
        let next = samples.get(upper_index + 1).copied().unwrap_or(upper);
        let lower_span = (upper.snapshot.simulation_time_seconds
            - previous.snapshot.simulation_time_seconds)
            .max(interval);
        let upper_span = (next.snapshot.simulation_time_seconds
            - lower.snapshot.simulation_time_seconds)
            .max(interval);
        let lower_value = lower.body_acceleration;
        let upper_value = upper.body_acceleration;
        let lower_tangent =
            (upper_value - previous.body_acceleration) * (interval / lower_span) as f32;
        let upper_tangent = (next.body_acceleration - lower_value) * (interval / upper_span) as f32;
        return Some(hermite_vector(
            lower_value,
            upper_value,
            lower_tangent,
            upper_tangent,
            u,
        ));
    }

    let previous = samples[samples.len() - 2];
    let interval = (latest.snapshot.simulation_time_seconds
        - previous.snapshot.simulation_time_seconds)
        .max(f64::EPSILON);
    let latest_value = latest.body_acceleration;
    let previous_value = previous.body_acceleration;
    let latest_delta = latest_value - previous_value;
    let mut slope = latest_delta / interval as f32;
    if samples.len() >= 3 {
        let older = samples[samples.len() - 3];
        let older_interval = (previous.snapshot.simulation_time_seconds
            - older.snapshot.simulation_time_seconds)
            .max(f64::EPSILON);
        let older_slope = (previous_value - older.body_acceleration) / older_interval as f32;
        // A weighted two-interval derivative suppresses readback jitter while
        // retaining the phase trend of the moving probe.
        slope = slope * 0.75 + older_slope * 0.25;
    }

    let raw_horizon = target_time - latest.snapshot.simulation_time_seconds;
    let horizon = raw_horizon.clamp(0.0, maximum_extrapolation_intervals * interval);
    let mut correction = slope * horizon as f32;
    let maximum_correction = latest_delta.length() * maximum_extrapolation_intervals as f32;
    if maximum_correction > 0.0 && correction.length() > maximum_correction {
        correction = correction.normalize_or_zero() * maximum_correction;
    }
    Some(latest_value + correction)
}

fn rotation_after(base: Quat, elapsed_seconds: f64) -> Quat {
    let angular_speed = std::f64::consts::TAU / RYUGU_ROTATION_PERIOD_SECS as f64;
    Quat::from_axis_angle(
        RYUGU_SPIN_AXIS.normalize(),
        (angular_speed * elapsed_seconds) as f32,
    ) * base
}

fn hash_benchmark_trajectory(samples: &[GravityBenchmarkSample]) -> u64 {
    samples
        .iter()
        .fold(1469598103934665603_u64, |hash, sample| {
            [
                sample.simulation_time_seconds.to_bits(),
                sample.position.x.to_bits() as u64,
                sample.position.y.to_bits() as u64,
                sample.position.z.to_bits() as u64,
                sample.velocity.x.to_bits() as u64,
                sample.velocity.y.to_bits() as u64,
                sample.velocity.z.to_bits() as u64,
            ]
            .into_iter()
            .fold(hash, |hash, value| {
                (hash ^ value).wrapping_mul(1099511628211_u64)
            })
        })
}

fn gpu_world_acceleration(
    history: &GravitySampleHistory,
    epoch: u64,
    target_time: f64,
    frame_start_time: f64,
    frame_start_rotation: Quat,
    maximum_extrapolation_intervals: f64,
) -> Option<Vec3> {
    let rotation = rotation_after(frame_start_rotation, target_time - frame_start_time);
    let body_acceleration =
        predict_body_acceleration(history, epoch, target_time, maximum_extrapolation_intervals)?;
    Some(rotation * body_acceleration)
}

pub fn physics_system(
    ryugu_query: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    radial_history: Option<Res<RadialGravityHistory>>,
    werner_history: Option<Res<WernerGravityHistory>>,
    mmfft_history: Option<Res<MmfftCompressedHistory>>,
    fmm_history: Option<Res<FmmGravityHistory>>,
    mut blend: ResMut<GravityBlendFactor>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut clock: ResMut<SimulationClock>,
    mut benchmark: ResMut<GravityBenchmarkTrajectory>,
    mut inversion: ResMut<TrajectoryInversionState>,
    time: Res<Time<Fixed>>,
    (active_method, simulation_acceleration, planning): (
        Res<ActiveGravityMethod>,
        Res<SimulationAcceleration>,
        Res<PlanningComparisonState>,
    ),
) {
    if runtime_error.is_active() || planning.blocks_realtime_gpu() {
        return;
    }
    let Some(ryugu_transform) = ryugu_query.iter().next() else {
        return;
    };
    let Some((mut probe_transform, mut probe_velocity, mut orbit_history)) =
        cassini_query.iter_mut().next()
    else {
        return;
    };
    if benchmark.epoch != clock.epoch {
        benchmark.epoch = clock.epoch;
        benchmark.samples.clear();
        benchmark.capture_id = None;
        benchmark.complete = false;
    }

    let active_history = select_history(
        *active_method,
        radial_history.as_deref(),
        werner_history.as_deref(),
        mmfft_history.as_deref(),
        fmm_history.as_deref(),
    );
    // Equation (184) is an aggregate trajectory observation operator and has
    // no instantaneous force history. For the Bevy scene only, use the
    // already-running radial GPU field as the physical display reference;
    // this preserves continuous shape-dependent motion without introducing a
    // second CPU point-source integrator or changing the benchmark scenario.
    let integration_history = if *active_method == ActiveGravityMethod::FrequencyDomain {
        radial_history.as_ref().map(|history| &history.0)
    } else {
        active_history
    };
    let maximum_extrapolation_intervals = match *active_method {
        ActiveGravityMethod::RadialAnalytic => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::HomogeneousWerner => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::FrequencyDomain => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::MmfftCompressed => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::Fmm => MAX_EXTRAPOLATION_INTERVALS,
    };
    let latest_sample =
        integration_history.and_then(|history| history.latest_for_epoch(clock.epoch));
    let gpu_ready = latest_sample.is_some();
    if gpu_ready {
        blend.0 = 1.0;
    }

    let stable_frame_dt = time.delta_secs_f64() * TIME_SCALE as f64;
    let substep_dt = stable_frame_dt / PHYSICS_SUBSTEPS as f64;
    let stable_steps = simulation_acceleration.stable_steps();
    let presented_frame_dt = stable_frame_dt * stable_steps as f64;
    let frame_start_time = clock.elapsed_seconds;
    let frame_start_rotation = ryugu_transform.rotation;
    if benchmark.samples.is_empty() && frame_start_time <= BENCHMARK_DURATION_SECONDS {
        benchmark.samples.push(GravityBenchmarkSample {
            simulation_time_seconds: frame_start_time,
            position: probe_transform.translation,
            velocity: probe_velocity.0,
        });
    }

    let acceleration_at = |sample_time: f64, _world_position: Vec3| -> Result<Vec3, &'static str> {
        let Some(history) = integration_history else {
            return Err("The selected GPU gravity evaluator is not registered.");
        };
        let Some(gpu_acceleration) = gpu_world_acceleration(
            history,
            clock.epoch,
            sample_time,
            frame_start_time,
            frame_start_rotation,
            maximum_extrapolation_intervals,
        ) else {
            // Readback latency is normal during warm-up. Pause the integrator
            // until the selected evaluator produces a snapshot; no alternate
            // force model is substituted.
            return Err("Waiting for a valid gravity readback snapshot.");
        };
        validate_acceleration(gpu_acceleration)
            .ok_or("The selected gravity evaluator returned an invalid acceleration.")
    };

    // Every pointwise evaluator uses the same 100-substep leapfrog integrator.
    // Intermediate states are retained in the orbit trail but are not presented,
    // which accelerates the visualization without enlarging the stable step size.
    for stable_step in 0..stable_steps {
        let stable_step_start = frame_start_time + stable_step as f64 * stable_frame_dt;
        for substep in 0..PHYSICS_SUBSTEPS {
            let start_time = stable_step_start + substep as f64 * substep_dt;
            let end_time = start_time + substep_dt;
            let acceleration_start = match acceleration_at(start_time, probe_transform.translation)
            {
                Ok(acceleration) => acceleration,
                Err("Waiting for a valid gravity readback snapshot.") => {
                    return;
                }
                Err(message) => {
                    runtime_error.raise(message);
                    return;
                }
            };
            probe_velocity.0 += acceleration_start * (0.5 * substep_dt as f32);
            probe_transform.translation += probe_velocity.0 * substep_dt as f32;
            let acceleration_end = match acceleration_at(end_time, probe_transform.translation) {
                Ok(acceleration) => acceleration,
                Err("Waiting for a valid gravity readback snapshot.") => {
                    return;
                }
                Err(message) => {
                    runtime_error.raise(message);
                    return;
                }
            };
            probe_velocity.0 += acceleration_end * (0.5 * substep_dt as f32);
            if *active_method == ActiveGravityMethod::RadialAnalytic
                && inversion.truth_orbit.len() < ORBIT_HISTORY_LEN
            {
                inversion.truth_orbit.push(probe_transform.translation);
            }
            if !benchmark.complete && end_time <= BENCHMARK_DURATION_SECONDS + 1.0e-9 {
                benchmark.samples.push(GravityBenchmarkSample {
                    simulation_time_seconds: end_time,
                    position: probe_transform.translation,
                    velocity: probe_velocity.0,
                });
                if end_time + 1.0e-9 >= BENCHMARK_DURATION_SECONDS {
                    benchmark.complete = true;
                    benchmark.capture_id = Some(hash_benchmark_trajectory(&benchmark.samples));
                }
            }
        }

        if orbit_history.0.len() >= ORBIT_HISTORY_LEN {
            orbit_history.0.pop_front();
        }
        orbit_history.0.push_back(probe_transform.translation);
    }

    clock.advance(presented_frame_dt);
}

pub fn ryugu_rotation_system(
    mut ryugu_query: Query<&mut Transform, With<RyuguMarker>>,
    clock: Res<SimulationClock>,
) {
    let angular_speed = std::f64::consts::TAU / RYUGU_ROTATION_PERIOD_SECS as f64;
    let rotation = Quat::from_axis_angle(
        RYUGU_SPIN_AXIS.normalize(),
        (angular_speed * clock.elapsed_seconds) as f32,
    );
    for mut transform in ryugu_query.iter_mut() {
        // Derive body attitude from the authoritative simulation clock. If
        // physics is waiting for a GPU readback, both clock and body frame now
        // remain frozen instead of silently diverging.
        transform.rotation = rotation;
    }
}
