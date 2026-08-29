use crate::cpu::curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory};
use crate::cpu::volterra::{
    VolterraConfig, VolterraError, VolterraForceInput, VolterraPropagationStatus,
    propagate_reference_line_batched,
};
use crate::interface::components::*;
use crate::interface::select_history;
use bevy::math::DVec3;
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

fn eq106_local_body_acceleration(
    history: &GravitySampleHistory,
    epoch: u64,
    target_time: f64,
    target_body_position: Vec3,
) -> Option<Vec3> {
    let (lower, upper) = history.bracketing(epoch, target_time)?;
    eq106_local_body_acceleration_between(lower, upper, target_time, target_body_position)
}

fn eq106_local_body_acceleration_between(
    lower: &GravityFieldSample,
    upper: &GravityFieldSample,
    target_time: f64,
    target_body_position: Vec3,
) -> Option<Vec3> {
    let evaluate = |sample: &GravityFieldSample| {
        let jacobian = sample.body_acceleration_jacobian?;
        let displacement = target_body_position - sample.snapshot.body_position;
        if !displacement.is_finite() || !jacobian.is_finite() {
            return None;
        }
        // The shader derives this Jacobian from the same scalar potential as
        // the returned field. Symmetrizing removes only f32 accumulation
        // asymmetry and keeps each local model conservative to first order.
        let symmetric_jacobian = (jacobian + jacobian.transpose()) * 0.5;
        Some(sample.body_acceleration + symmetric_jacobian * displacement)
    };
    let lower_acceleration = evaluate(lower)?;
    if std::ptr::eq(lower, upper) {
        return Some(lower_acceleration);
    }
    let upper_acceleration = evaluate(upper)?;
    let interval = upper.snapshot.simulation_time_seconds - lower.snapshot.simulation_time_seconds;
    if interval <= f64::EPSILON {
        return Some(lower_acceleration);
    }
    let weight =
        ((target_time - lower.snapshot.simulation_time_seconds) / interval).clamp(0.0, 1.0) as f32;
    Some(lower_acceleration.lerp(upper_acceleration, weight))
}

#[allow(clippy::too_many_arguments)]
fn fill_eq106_world_residual_batch(
    history: &GravitySampleHistory,
    epoch: u64,
    inputs: &[VolterraForceInput],
    frame_start_time: f64,
    stable_step_start: f64,
    frame_start_translation: Vec3,
    frame_start_rotation: Quat,
    outputs: &mut [DVec3],
) -> Option<()> {
    if inputs.len() != outputs.len() {
        return None;
    }
    let mut anchors = history
        .samples
        .iter()
        .filter(|sample| sample.snapshot.epoch == epoch)
        .peekable();
    let mut lower = None;
    for (input, output) in inputs.iter().zip(outputs) {
        let target_time = stable_step_start + input.elapsed_seconds;
        while anchors
            .peek()
            .is_some_and(|sample| sample.snapshot.simulation_time_seconds <= target_time + 1.0e-6)
        {
            lower = anchors.next();
        }
        let lower_sample = lower?;
        let upper = anchors
            .peek()
            .copied()
            .filter(|sample| sample.snapshot.simulation_time_seconds >= target_time - 1.0e-6)
            .unwrap_or(lower_sample);
        let rotation = rotation_after(frame_start_rotation, target_time - frame_start_time);
        let target_body_position =
            rotation.inverse() * (input.position.as_vec3() - frame_start_translation);
        let body_acceleration = eq106_local_body_acceleration_between(
            lower_sample,
            upper,
            target_time,
            target_body_position,
        )?;
        *output = validate_acceleration(rotation * body_acceleration)?.as_dvec3();
    }
    Some(())
}

fn eq106_snapshot_matches_clock(sample: &GravityFieldSample, clock: &SimulationClock) -> bool {
    sample.snapshot.epoch == clock.epoch
        && sample.snapshot.request_id == clock.request_id
        && (sample.snapshot.simulation_time_seconds - clock.elapsed_seconds).abs() <= 1.0e-6
}

fn gpu_world_residual(
    history: &GravitySampleHistory,
    epoch: u64,
    target_world_position: Vec3,
    target_time: f64,
    frame_start_time: f64,
    frame_start_translation: Vec3,
    frame_start_rotation: Quat,
    maximum_extrapolation_intervals: f64,
    use_local_potential_hessian: bool,
) -> Option<Vec3> {
    let rotation = rotation_after(frame_start_rotation, target_time - frame_start_time);
    let body_acceleration = if use_local_potential_hessian {
        let target_body_position =
            rotation.inverse() * (target_world_position - frame_start_translation);
        eq106_local_body_acceleration(history, epoch, target_time, target_body_position)?
    } else {
        predict_body_acceleration(history, epoch, target_time, maximum_extrapolation_intervals)?
    };
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
    eq106_history: Option<Res<Eq106GpuHistory>>,
    mmfft_history: Option<Res<MmfftCompressedHistory>>,
    fmm_history: Option<Res<FmmGravityHistory>>,
    mut blend: ResMut<GravityBlendFactor>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut clock: ResMut<SimulationClock>,
    mut curved_residual: ResMut<CurvedArcResidualHistory>,
    mut benchmark: ResMut<GravityBenchmarkTrajectory>,
    mut inversion: ResMut<TrajectoryInversionState>,
    time: Res<Time<Fixed>>,
    (active_method, curved_planner, mut volterra_status, simulation_acceleration, planning): (
        Res<ActiveGravityMethod>,
        Res<CurvedArcPlannerState>,
        ResMut<VolterraPropagationStatus>,
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
        eq106_history.as_deref(),
        mmfft_history.as_deref(),
        fmm_history.as_deref(),
    );
    let maximum_extrapolation_intervals = match *active_method {
        ActiveGravityMethod::RadialAnalytic => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::HomogeneousWerner => MAX_EXTRAPOLATION_INTERVALS,
        // Eq.106 readback is validated against completed GPU snapshots only.
        // During this diagnostic phase, do not extrapolate across a missing
        // asynchronous result; doing so changes the force model being tested.
        ActiveGravityMethod::CurvedArcEq106 => 0.0,
        ActiveGravityMethod::MmfftCompressed => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::Fmm => MAX_EXTRAPOLATION_INTERVALS,
    };
    let latest_sample = active_history.and_then(|history| {
        if *active_method == ActiveGravityMethod::CurvedArcEq106 {
            history.at_or_before(clock.epoch, clock.elapsed_seconds)
        } else {
            history.latest_for_epoch(clock.epoch)
        }
    });
    if *active_method == ActiveGravityMethod::CurvedArcEq106
        && !latest_sample.is_some_and(|sample| eq106_snapshot_matches_clock(sample, &clock))
    {
        // A local Eq.106 field is consumed exactly once. Advancing several
        // accelerated frames from one asynchronous readback leaves the local
        // Taylor/Hessian neighborhood and creates the large secular Jacobi
        // drift visible at 8x.
        return;
    }
    let gpu_ready = latest_sample.is_some();
    if gpu_ready {
        blend.0 = 1.0;
    }

    let stable_frame_dt = time.delta_secs_f64() * TIME_SCALE as f64;
    let substep_dt = stable_frame_dt / PHYSICS_SUBSTEPS as f64;
    let stable_steps = simulation_acceleration.stable_steps();
    let presented_frame_dt = stable_frame_dt * stable_steps as f64;
    let frame_start_time = clock.elapsed_seconds;
    let frame_start_translation = ryugu_transform.translation;
    let frame_start_rotation = ryugu_transform.rotation;
    let use_local_potential_hessian = *active_method == ActiveGravityMethod::CurvedArcEq106;
    if benchmark.samples.is_empty() && frame_start_time <= BENCHMARK_DURATION_SECONDS {
        benchmark.samples.push(GravityBenchmarkSample {
            simulation_time_seconds: frame_start_time,
            position: probe_transform.translation,
            velocity: probe_velocity.0,
            body_rotation: frame_start_rotation,
        });
    }

    let acceleration_at = |position: Vec3, sample_time: f64| -> Result<Vec3, &'static str> {
        let Some(history) = active_history else {
            return Err("The selected GPU gravity evaluator is not registered.");
        };
        let Some(gpu_acceleration) = gpu_world_residual(
            history,
            clock.epoch,
            position,
            sample_time,
            frame_start_time,
            frame_start_translation,
            frame_start_rotation,
            maximum_extrapolation_intervals,
            use_local_potential_hessian,
        ) else {
            // Readback latency is normal during warm-up. Pause the integrator
            // until the selected evaluator produces a snapshot; no alternate
            // force model is substituted.
            return Err("Waiting for a valid gravity readback snapshot.");
        };
        validate_acceleration(gpu_acceleration)
            .ok_or("The selected gravity evaluator returned an invalid acceleration.")
    };

    if *active_method == ActiveGravityMethod::CurvedArcEq106 {
        // Equations (27), (28), and (40): close position -> conservative local
        // Eq.106 Taylor field -> trajectory on each stable segment. The field
        // value and Hessian come from the completed GPU sample; every Picard
        // iterate reevaluates that first-order spatial model at all M updated
        // positions in one batch. This is self-consistent inside the certified
        // Taylor tube, not a claim of a fresh full source sum at every iterate.
        let certified_tube_radius = curved_planner
            .active_segment
            .as_ref()
            .map(|segment| 0.8 * segment.distance_lower_bound)
            .filter(|radius| radius.is_finite() && *radius > 0.0)
            .unwrap_or(f64::from(PLANNING_TRAJECTORY_TUBE_RADIUS_METERS))
            .min(f64::from(PLANNING_TRAJECTORY_TUBE_RADIUS_METERS));
        let config = VolterraConfig {
            // Keep the same authoritative 10 ms output cadence while solving
            // the segment as one waveform rather than 100 dependent kicks.
            node_count: PHYSICS_SUBSTEPS + 1,
            maximum_picard_iterations: 10,
            maximum_endpoint_iterations: 6,
            damping: 0.75,
            relative_tolerance: 2.0e-7,
            minimum_longitudinal_speed: 1.0e-5,
            maximum_transverse_distance: certified_tube_radius,
        };

        for stable_step in 0..stable_steps {
            let stable_step_start = frame_start_time + stable_step as f64 * stable_frame_dt;
            let initial_position = probe_transform.translation.as_dvec3();
            let initial_velocity = probe_velocity.0.as_dvec3();
            let solution = propagate_reference_line_batched(
                initial_position,
                initial_velocity,
                initial_velocity,
                stable_frame_dt,
                config,
                |inputs, accelerations| {
                    let history = active_history
                        .ok_or("The selected GPU gravity evaluator is not registered.")?;
                    fill_eq106_world_residual_batch(
                        history,
                        clock.epoch,
                        inputs,
                        frame_start_time,
                        stable_step_start,
                        frame_start_translation,
                        frame_start_rotation,
                        accelerations,
                    )
                    .ok_or("Waiting for a valid gravity readback snapshot.")
                },
            );
            let solution = match solution {
                Ok(solution) => solution,
                Err(VolterraError::Force("Waiting for a valid gravity readback snapshot."))
                | Err(VolterraError::Force("Waiting for Equation (106) source quadrature.")) => {
                    return;
                }
                Err(VolterraError::Force(message)) => {
                    volterra_status.rejected_segments =
                        volterra_status.rejected_segments.saturating_add(1);
                    runtime_error.raise(message);
                    return;
                }
                Err(VolterraError::NonMonotoneLongitudinalMotion) => {
                    volterra_status.rejected_segments =
                        volterra_status.rejected_segments.saturating_add(1);
                    runtime_error.raise(
                        "Eq.106 Volterra propagation reached a longitudinal turning point; split the reference arc before retrying.",
                    );
                    return;
                }
                Err(VolterraError::TaylorTubeExceeded) => {
                    volterra_status.rejected_segments =
                        volterra_status.rejected_segments.saturating_add(1);
                    runtime_error.raise(
                        "Eq.106 Volterra propagation left the certified Taylor tube; rebuild a shorter reference segment.",
                    );
                    return;
                }
                Err(VolterraError::PicardDidNotConverge)
                | Err(VolterraError::EndpointDidNotConverge) => {
                    volterra_status.rejected_segments =
                        volterra_status.rejected_segments.saturating_add(1);
                    runtime_error.raise(
                        "Eq.106 Volterra/Picard propagation did not converge on the current segment.",
                    );
                    return;
                }
                Err(VolterraError::InvalidInput) => {
                    volterra_status.rejected_segments =
                        volterra_status.rejected_segments.saturating_add(1);
                    runtime_error.raise("Eq.106 Volterra propagation received invalid state data.");
                    return;
                }
            };
            volterra_status.accepted_segments = volterra_status.accepted_segments.saturating_add(1);
            volterra_status.latest = Some(solution.diagnostics);

            let mut solution_cursor = 1;
            for substep in 0..PHYSICS_SUBSTEPS {
                let start_elapsed = substep as f64 * substep_dt;
                let end_elapsed = (substep + 1) as f64 * substep_dt;
                let Some(start) = solution.sample_at_ordered(start_elapsed, &mut solution_cursor)
                else {
                    runtime_error
                        .raise("Eq.106 Volterra output did not cover the fixed-step interval.");
                    return;
                };
                let Some(end) = solution.sample_at_ordered(end_elapsed, &mut solution_cursor)
                else {
                    runtime_error
                        .raise("Eq.106 Volterra output did not cover the fixed-step interval.");
                    return;
                };
                let start_time = stable_step_start + start_elapsed;
                let end_time = stable_step_start + end_elapsed;
                let start_rotation =
                    rotation_after(frame_start_rotation, start_time - frame_start_time);
                let end_rotation =
                    rotation_after(frame_start_rotation, end_time - frame_start_time);
                curved_residual.accumulate_curve_work(
                    start_time,
                    end_time,
                    start_rotation.inverse() * (start.position.as_vec3() - frame_start_translation),
                    end_rotation.inverse() * (end.position.as_vec3() - frame_start_translation),
                    start_rotation.inverse() * start.acceleration.as_vec3(),
                    end_rotation.inverse() * end.acceleration.as_vec3(),
                );
                if !benchmark.complete && end_time <= BENCHMARK_DURATION_SECONDS + 1.0e-9 {
                    benchmark.samples.push(GravityBenchmarkSample {
                        simulation_time_seconds: end_time,
                        position: end.position.as_vec3(),
                        velocity: end.velocity.as_vec3(),
                        body_rotation: end_rotation,
                    });
                    if end_time + 1.0e-9 >= BENCHMARK_DURATION_SECONDS {
                        benchmark.complete = true;
                        benchmark.capture_id = Some(hash_benchmark_trajectory(&benchmark.samples));
                    }
                }
            }

            let endpoint = *solution
                .samples
                .last()
                .expect("a successful Volterra solve has at least two samples");
            probe_transform.translation = endpoint.position.as_vec3();
            probe_velocity.0 = endpoint.velocity.as_vec3();
            if orbit_history.0.len() >= ORBIT_HISTORY_LEN {
                orbit_history.0.pop_front();
            }
            orbit_history.0.push_back(probe_transform.translation);
        }

        clock.advance(presented_frame_dt);
        return;
    }

    // Non-Eq.106 evaluators retain the 100-substep leapfrog frame.
    // Intermediate states are retained in the orbit trail but are not presented,
    // which accelerates the visualization without enlarging the stable step size.
    for stable_step in 0..stable_steps {
        let stable_step_start = frame_start_time + stable_step as f64 * stable_frame_dt;
        for substep in 0..PHYSICS_SUBSTEPS {
            let start_time = stable_step_start + substep as f64 * substep_dt;
            let end_time = start_time + substep_dt;
            let start_world_position = probe_transform.translation;
            let acceleration_start = match acceleration_at(start_world_position, start_time) {
                Ok(acceleration) => acceleration,
                Err("Waiting for a valid gravity readback snapshot.") => {
                    return;
                }
                Err("Waiting for Equation (106) source quadrature.") => {
                    return;
                }
                Err(message) => {
                    runtime_error.raise(message);
                    return;
                }
            };
            probe_velocity.0 += acceleration_start * (0.5 * substep_dt as f32);
            probe_transform.translation += probe_velocity.0 * substep_dt as f32;
            let acceleration_end = match acceleration_at(probe_transform.translation, end_time) {
                Ok(acceleration) => acceleration,
                Err("Waiting for a valid gravity readback snapshot.") => {
                    return;
                }
                Err("Waiting for Equation (106) source quadrature.") => {
                    return;
                }
                Err(message) => {
                    runtime_error.raise(message);
                    return;
                }
            };
            let start_rotation =
                rotation_after(frame_start_rotation, start_time - frame_start_time);
            let end_rotation = rotation_after(frame_start_rotation, end_time - frame_start_time);
            curved_residual.accumulate_curve_work(
                start_time,
                end_time,
                start_rotation.inverse() * (start_world_position - frame_start_translation),
                end_rotation.inverse() * (probe_transform.translation - frame_start_translation),
                start_rotation.inverse() * acceleration_start,
                end_rotation.inverse() * acceleration_end,
            );
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
                    body_rotation: end_rotation,
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
