use crate::interface::components::*;
// Legacy history-based frontend integrator disabled.
use bevy::prelude::*;

// const MAX_ACC: f32 = 1.5e-3;
// const MAX_EXTRAPOLATION_INTERVALS: f64 = 2.0;
//
// fn validate_acceleration(acceleration: Vec3) -> Option<Vec3> {
//     let magnitude = acceleration.length();
//     if acceleration.is_finite() && magnitude.is_finite() && magnitude <= MAX_ACC {
//         Some(acceleration)
//     } else {
//         None
//     }
// }
//
//
fn hermite_vector(a: Vec3, b: Vec3, tangent_a: Vec3, tangent_b: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * a
        + (t3 - 2.0 * t2 + t) * tangent_a
        + (-2.0 * t3 + 3.0 * t2) * b
        + (t3 - t2) * tangent_b
}

// /// Interpolates completed body-frame GPU samples and performs only a bounded,
// /// slope-limited extrapolation past the newest result. Unbounded cubic
// /// extrapolation is deliberately avoided because asynchronous readback can
// /// occasionally skip render frames.
// fn predict_body_acceleration(
//     history: &GravitySampleHistory,
//     epoch: u64,
//     target_time: f64,
//     maximum_extrapolation_intervals: f64,
// ) -> Option<Vec3> {
//     let samples: Vec<&GravityFieldSample> = history
//         .samples
//         .iter()
//         .filter(|sample| sample.snapshot.epoch == epoch)
//         .collect();
//     let latest = *samples.last()?;
//     if samples.len() == 1 {
//         return Some(latest.body_acceleration);
//     }
//
//     if target_time <= latest.snapshot.simulation_time_seconds {
//         let upper_index = samples
//             .iter()
//             .position(|sample| sample.snapshot.simulation_time_seconds >= target_time)
//             .unwrap_or(samples.len() - 1);
//         if upper_index == 0 {
//             return Some(samples[0].body_acceleration);
//         }
//         let lower = samples[upper_index - 1];
//         let upper = samples[upper_index];
//         let interval = (upper.snapshot.simulation_time_seconds
//             - lower.snapshot.simulation_time_seconds)
//             .max(f64::EPSILON);
//         let u = ((target_time - lower.snapshot.simulation_time_seconds) / interval).clamp(0.0, 1.0)
//             as f32;
//
//         let previous = samples
//             .get(upper_index.saturating_sub(2))
//             .copied()
//             .unwrap_or(lower);
//         let next = samples.get(upper_index + 1).copied().unwrap_or(upper);
//         let lower_span = (upper.snapshot.simulation_time_seconds
//             - previous.snapshot.simulation_time_seconds)
//             .max(interval);
//         let upper_span = (next.snapshot.simulation_time_seconds
//             - lower.snapshot.simulation_time_seconds)
//             .max(interval);
//         let lower_value = lower.body_acceleration;
//         let upper_value = upper.body_acceleration;
//         let lower_tangent =
//             (upper_value - previous.body_acceleration) * (interval / lower_span) as f32;
//         let upper_tangent = (next.body_acceleration - lower_value) * (interval / upper_span) as f32;
//         return Some(hermite_vector(
//             lower_value,
//             upper_value,
//             lower_tangent,
//             upper_tangent,
//             u,
//         ));
//     }
//
//     let previous = samples[samples.len() - 2];
//     let interval = (latest.snapshot.simulation_time_seconds
//         - previous.snapshot.simulation_time_seconds)
//         .max(f64::EPSILON);
//     let latest_value = latest.body_acceleration;
//     let previous_value = previous.body_acceleration;
//     let latest_delta = latest_value - previous_value;
//     let mut slope = latest_delta / interval as f32;
//     if samples.len() >= 3 {
//         let older = samples[samples.len() - 3];
//         let older_interval = (previous.snapshot.simulation_time_seconds
//             - older.snapshot.simulation_time_seconds)
//             .max(f64::EPSILON);
//         let older_slope = (previous_value - older.body_acceleration) / older_interval as f32;
//         // A weighted two-interval derivative suppresses readback jitter while
//         // retaining the phase trend of the moving probe.
//         slope = slope * 0.75 + older_slope * 0.25;
//     }
//
//     let raw_horizon = target_time - latest.snapshot.simulation_time_seconds;
//     let horizon = raw_horizon.clamp(0.0, maximum_extrapolation_intervals * interval);
//     let mut correction = slope * horizon as f32;
//     let maximum_correction = latest_delta.length() * maximum_extrapolation_intervals as f32;
//     if maximum_correction > 0.0 && correction.length() > maximum_correction {
//         correction = correction.normalize_or_zero() * maximum_correction;
//     }
//     Some(latest_value + correction)
// }
//
// fn rotation_after(base: Quat, elapsed_seconds: f64) -> Quat {
//     let angular_speed = std::f64::consts::TAU / RYUGU_ROTATION_PERIOD_SECS as f64;
//     Quat::from_axis_angle(
//         RYUGU_SPIN_AXIS.normalize(),
//         (angular_speed * elapsed_seconds) as f32,
//     ) * base
// }
//
//
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

// fn gpu_world_acceleration(
//     history: &GravitySampleHistory,
//     epoch: u64,
//     target_time: f64,
//     frame_start_time: f64,
//     frame_start_rotation: Quat,
//     maximum_extrapolation_intervals: f64,
// ) -> Option<Vec3> {
//     let rotation = rotation_after(frame_start_rotation, target_time - frame_start_time);
//     let body_acceleration =
//         predict_body_acceleration(history, epoch, target_time, maximum_extrapolation_intervals)?;
//     Some(rotation * body_acceleration)
// }
//
// pub fn physics_system(
//     cpp_state: Res<crate::cpp_backend::CppBackendState>,
//     ryugu_query: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
//     mut cassini_query: Query<
//         (&mut Transform, &mut Velocity, &mut OrbitHistory),
//         (With<CassiniMarker>, Without<RyuguMarker>),
//     >,
//     radial_history: Option<Res<RadialGravityHistory>>,
//     werner_history: Option<Res<WernerGravityHistory>>,
//     mmfft_history: Option<Res<MmfftCompressedHistory>>,
//     fmm_history: Option<Res<FmmGravityHistory>>,
//     equation106: Option<Res<crate::gpu::equation106::Equation106History>>,
//     mut blend: ResMut<GravityBlendFactor>,
//     mut runtime_error: ResMut<GravityRuntimeError>,
//     mut clock: ResMut<SimulationClock>,
//     mut benchmark: ResMut<GravityBenchmarkTrajectory>,
//     mut inversion: ResMut<TrajectoryInversionState>,
//     (active_method, simulation_acceleration, planning): (
//         Res<ActiveGravityMethod>,
//         Res<SimulationAcceleration>,
//         Res<PlanningComparisonState>,
//     ),
// ) {
//     if runtime_error.is_active() || planning.blocks_realtime_gpu() {
//         return;
//     }
//     if *active_method != ActiveGravityMethod::FrequencyDomain && !cpp_state.ready {
//         return;
//     }
//     let Some(ryugu_transform) = ryugu_query.iter().next() else {
//         return;
//     };
//     let Some((mut probe_transform, mut probe_velocity, mut orbit_history)) =
//         cassini_query.iter_mut().next()
//     else {
//         return;
//     };
//     if benchmark.epoch != clock.epoch {
//         benchmark.epoch = clock.epoch;
//         benchmark.samples.clear();
//         benchmark.capture_id = None;
//         benchmark.complete = false;
//     }
//
//     let active_history = select_history(
//         *active_method,
//         radial_history.as_deref(),
//         werner_history.as_deref(),
//         mmfft_history.as_deref(),
//         fmm_history.as_deref(),
//         equation106.as_deref(),
//     );
//     // Eq.106 drives the frequency trajectory; Eq.184 consumes its captured
//     // states afterwards and never masquerades as an instantaneous force.
//     let integration_history = active_history;
//     let maximum_extrapolation_intervals = match *active_method {
//         ActiveGravityMethod::RadialAnalytic => MAX_EXTRAPOLATION_INTERVALS,
//         ActiveGravityMethod::HomogeneousWerner => MAX_EXTRAPOLATION_INTERVALS,
//         ActiveGravityMethod::FrequencyDomain => MAX_EXTRAPOLATION_INTERVALS,
//         ActiveGravityMethod::MmfftCompressed => MAX_EXTRAPOLATION_INTERVALS,
//         ActiveGravityMethod::Fmm => MAX_EXTRAPOLATION_INTERVALS,
//     };
//     let latest_sample =
//         integration_history.and_then(|history| history.latest_for_epoch(clock.epoch));
//     let gpu_ready = latest_sample.is_some();
//     if gpu_ready {
//         blend.0 = 1.0;
//     }
//
//     // The render clock is not authoritative. FixedUpdate is configured at
//     // 60 Hz, while the physical step is explicitly carried by SimulationClock
//     // so native Basilisk adapters and browser WASM share the same interval.
//     let stable_frame_dt = clock.fixed_step_seconds;
//     let substep_dt = stable_frame_dt / PHYSICS_SUBSTEPS as f64;
//     let available_steps = integration_history
//         .and_then(|history| {
//             let mut samples = history
//                 .samples
//                 .iter()
//                 .rev()
//                 .filter(|sample| sample.snapshot.epoch == clock.epoch);
//             let latest = samples.next()?;
//             let horizon = samples.next().map_or(stable_frame_dt, |previous| {
//                 (latest.snapshot.simulation_time_seconds
//                     - previous.snapshot.simulation_time_seconds)
//                     * maximum_extrapolation_intervals
//             });
//             Some(
//                 ((latest.snapshot.simulation_time_seconds + horizon - clock.elapsed_seconds)
//                     / stable_frame_dt
//                     + 1.0e-6)
//                     .floor()
//                     .max(0.0) as u32,
//             )
//         })
//         .unwrap_or(1);
//     let stable_steps = simulation_acceleration.stable_steps().min(available_steps);
//     if stable_steps == 0 {
//         return;
//     }
//     let presented_frame_dt = stable_frame_dt * stable_steps as f64;
//     let frame_start_time = clock.elapsed_seconds;
//     let frame_start_rotation = ryugu_transform.rotation;
//     if benchmark.samples.is_empty() && frame_start_time <= BENCHMARK_DURATION_SECONDS {
//         benchmark.samples.push(GravityBenchmarkSample {
//             simulation_time_seconds: frame_start_time,
//             position: probe_transform.translation,
//             velocity: probe_velocity.0,
//         });
//     }
//
//     let acceleration_at = |sample_time: f64, world_position: Vec3| -> Result<Vec3, &'static str> {
//         if *active_method != ActiveGravityMethod::FrequencyDomain {
//             let rotation = rotation_after(frame_start_rotation, sample_time - frame_start_time);
//             let (field, _) = crate::cpp_backend::evaluate(
//                 *active_method,
//                 rotation.inverse() * (world_position - ryugu_transform.translation),
//             )
//             .map_err(|_| "The C++ gravity evaluator failed at an integration stage.")?;
//             return validate_acceleration(rotation * field)
//                 .ok_or("The C++ gravity evaluator returned an invalid acceleration.");
//         }
//         let Some(history) = integration_history else {
//             return Err("The selected GPU gravity evaluator is not registered.");
//         };
//         let Some(gpu_acceleration) = gpu_world_acceleration(
//             history,
//             clock.epoch,
//             sample_time,
//             frame_start_time,
//             frame_start_rotation,
//             maximum_extrapolation_intervals,
//         ) else {
//             // Readback latency is normal during warm-up. Pause the integrator
//             // until the selected evaluator produces a snapshot; no alternate
//             // force model is substituted.
//             return Err("Waiting for a valid gravity readback snapshot.");
//         };
//         validate_acceleration(gpu_acceleration)
//             .ok_or("The selected gravity evaluator returned an invalid acceleration.")
//     };
//
//     // Check the full requested interval before mutating any state. Otherwise
//     // an unavailable end snapshot left a half-kick/drift at the old clock time.
//     match acceleration_at(
//         frame_start_time + presented_frame_dt,
//         probe_transform.translation,
//     ) {
//         Ok(_) => {}
//         Err("Waiting for a valid gravity readback snapshot.") => return,
//         Err(message) => {
//             runtime_error.raise(message);
//             return;
//         }
//     }
//     if *active_method == ActiveGravityMethod::FrequencyDomain
//         && !inversion.ready
//         && inversion.knots.is_empty()
//     {
//         inversion.knots.push(TrajectoryInversionKnot {
//             position: probe_transform.translation,
//             velocity: probe_velocity.0,
//             simulation_time_seconds: frame_start_time,
//             baseline_acceleration: acceleration_at(frame_start_time, probe_transform.translation)
//                 .unwrap_or(Vec3::ZERO),
//             body_rotation: frame_start_rotation,
//         });
//     }
//
//     // Every pointwise evaluator uses the same 100-substep leapfrog integrator.
//     // Intermediate states are retained in the orbit trail but are not presented,
//     // which accelerates the visualization without enlarging the stable step size.
//     for stable_step in 0..stable_steps {
//         let stable_step_start = frame_start_time + stable_step as f64 * stable_frame_dt;
//         for substep in 0..PHYSICS_SUBSTEPS {
//             let start_time = stable_step_start + substep as f64 * substep_dt;
//             let end_time = start_time + substep_dt;
//             let acceleration_start = match acceleration_at(start_time, probe_transform.translation)
//             {
//                 Ok(acceleration) => acceleration,
//                 Err("Waiting for a valid gravity readback snapshot.") => {
//                     return;
//                 }
//                 Err(message) => {
//                     runtime_error.raise(message);
//                     return;
//                 }
//             };
//             let previous_position = probe_transform.translation;
//             let previous_velocity = probe_velocity.0;
//             let half_velocity = previous_velocity + acceleration_start * (0.5 * substep_dt as f32);
//             let next_position = previous_position + half_velocity * substep_dt as f32;
//             let acceleration_end = match acceleration_at(end_time, next_position) {
//                 Ok(acceleration) => acceleration,
//                 Err("Waiting for a valid gravity readback snapshot.") => {
//                     return;
//                 }
//                 Err(message) => {
//                     runtime_error.raise(message);
//                     return;
//                 }
//             };
//             probe_transform.translation = next_position;
//             probe_velocity.0 = half_velocity + acceleration_end * (0.5 * substep_dt as f32);
//             if *active_method == ActiveGravityMethod::FrequencyDomain && !inversion.ready {
//                 let interval = TRAJECTORY_INVERSION_CAPTURE_SECONDS
//                     / (TRAJECTORY_INVERSION_SAMPLE_COUNT - 1) as f64;
//                 while inversion.knots.len() < TRAJECTORY_INVERSION_SAMPLE_COUNT {
//                     let sample_time = inversion.knots.len() as f64 * interval;
//                     if sample_time > end_time + 1.0e-9 {
//                         break;
//                     }
//                     let fraction = ((sample_time - start_time) / substep_dt).clamp(0.0, 1.0) as f32;
//                     inversion.knots.push(TrajectoryInversionKnot {
//                         position: hermite_vector(
//                             previous_position,
//                             next_position,
//                             previous_velocity * substep_dt as f32,
//                             probe_velocity.0 * substep_dt as f32,
//                             fraction,
//                         ),
//                         velocity: previous_velocity.lerp(probe_velocity.0, fraction),
//                         simulation_time_seconds: sample_time,
//                         baseline_acceleration: acceleration_start.lerp(acceleration_end, fraction),
//                         body_rotation: rotation_after(
//                             frame_start_rotation,
//                             sample_time - frame_start_time,
//                         ),
//                     });
//                 }
//             }
//             if *active_method == ActiveGravityMethod::RadialAnalytic
//                 && inversion.truth_orbit.len() < ORBIT_HISTORY_LEN
//             {
//                 inversion.truth_orbit.push(probe_transform.translation);
//             }
//             if !benchmark.complete && end_time <= BENCHMARK_DURATION_SECONDS + 1.0e-9 {
//                 benchmark.samples.push(GravityBenchmarkSample {
//                     simulation_time_seconds: end_time,
//                     position: probe_transform.translation,
//                     velocity: probe_velocity.0,
//                 });
//                 if end_time + 1.0e-9 >= BENCHMARK_DURATION_SECONDS {
//                     benchmark.complete = true;
//                     benchmark.capture_id = Some(hash_benchmark_trajectory(&benchmark.samples));
//                 }
//             }
//         }
//
//         if orbit_history.0.len() >= ORBIT_HISTORY_LEN {
//             orbit_history.0.pop_front();
//         }
//         orbit_history.0.push_back(probe_transform.translation);
//     }
//
//     clock.advance(presented_frame_dt);
// }
//
//

pub fn physics_system(
    mut frame_pacer: Local<crate::cpp_backend::BackendFramePacer>,
    ready: Res<crate::cpp_backend::CppBackendState>,
    body: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut probe: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    frequency: Option<Res<crate::gpu::equation106::Equation106History>>,
    active: Res<ActiveGravityMethod>,
    planning: Res<PlanningComparisonState>,
    acceleration: Res<SimulationAcceleration>,
    mut blend: ResMut<GravityBlendFactor>,
    mut error: ResMut<GravityRuntimeError>,
    mut clock: ResMut<SimulationClock>,
    mut benchmark: ResMut<GravityBenchmarkTrajectory>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if !crate::cpp_backend::should_advance_backend(&mut frame_pacer, *active) {
        return;
    }
    if !ready.ready || error.is_active() || planning.blocks_realtime_gpu() {
        return;
    }
    let (Ok(body), Ok((mut transform, mut velocity, mut orbit))) =
        (body.single(), probe.single_mut())
    else {
        return;
    };
    let mut steps = acceleration.stable_steps();
    let mut history = Vec::new();
    if *active == ActiveGravityMethod::FrequencyDomain {
        let Some(frequency) = frequency else { return };
        for sample in &frequency.0.samples {
            if sample.snapshot.epoch != clock.epoch {
                continue;
            }
            history.push(sample.snapshot.simulation_time_seconds);
            history.extend(sample.body_acceleration.to_array().map(f64::from));
        }
        let samples = history.as_chunks::<4>().0;
        let Some(last) = samples.last() else { return };
        let horizon = if samples.len() > 1 {
            2.0 * (last[0] - samples[samples.len() - 2][0])
        } else {
            clock.fixed_step_seconds
        };
        let available = ((last[0] + horizon - clock.elapsed_seconds) / clock.fixed_step_seconds
            + 1e-6)
            .floor()
            .max(0.0) as u32;
        steps = steps.min(available);
        if steps == 0 {
            return;
        }
    }
    let initial: Vec<f64> = transform
        .translation
        .to_array()
        .into_iter()
        .chain(velocity.0.to_array())
        .chain(body.rotation.to_array())
        .map(f64::from)
        .collect();
    let trace = match crate::cpp_backend::advance(
        clock.epoch,
        *active,
        &initial,
        clock.fixed_step_seconds,
        steps,
        &history,
    ) {
        Ok(trace) => trace,
        Err(message) => {
            if message.contains("Waiting for frequency-domain samples") {
                return;
            }
            error.raise(message);
            return;
        }
    };
    if trace.len() < 28 || !trace.len().is_multiple_of(14) || !trace.iter().all(|v| v.is_finite()) {
        error.raise("Invalid simulation snapshot sequence");
        return;
    }
    let records = trace.as_chunks::<14>().0;
    let position = |r: &[f64; 14]| Vec3::new(r[1] as f32, r[2] as f32, r[3] as f32);
    let speed = |r: &[f64; 14]| Vec3::new(r[4] as f32, r[5] as f32, r[6] as f32);
    let field = |r: &[f64; 14]| Vec3::new(r[7] as f32, r[8] as f32, r[9] as f32);
    let attitude =
        |r: &[f64; 14]| Quat::from_xyzw(r[10] as f32, r[11] as f32, r[12] as f32, r[13] as f32);
    if benchmark.epoch != clock.epoch {
        benchmark.epoch = clock.epoch;
        benchmark.samples.clear();
        benchmark.capture_id = None;
        benchmark.complete = false;
    }
    if benchmark.samples.is_empty() {
        benchmark.samples.push(GravityBenchmarkSample {
            simulation_time_seconds: records[0][0],
            position: position(&records[0]),
            velocity: speed(&records[0]),
        });
    }
    if *active == ActiveGravityMethod::FrequencyDomain
        && !inversion.ready
        && inversion.knots.is_empty()
    {
        let r = &records[0];
        inversion.knots.push(TrajectoryInversionKnot {
            position: position(r),
            velocity: speed(r),
            simulation_time_seconds: r[0],
            baseline_acceleration: field(r),
            body_rotation: attitude(r),
        });
    }
    for pair in records.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if *active == ActiveGravityMethod::FrequencyDomain && !inversion.ready {
            let interval = TRAJECTORY_INVERSION_CAPTURE_SECONDS
                / (TRAJECTORY_INVERSION_SAMPLE_COUNT - 1) as f64;
            while inversion.knots.len() < TRAJECTORY_INVERSION_SAMPLE_COUNT {
                let time = inversion.knots.len() as f64 * interval;
                if time > b[0] + 1e-9 {
                    break;
                }
                let dt = (b[0] - a[0]) as f32;
                let fraction = ((time - a[0]) / (b[0] - a[0])).clamp(0.0, 1.0) as f32;
                inversion.knots.push(TrajectoryInversionKnot {
                    position: hermite_vector(
                        position(a),
                        position(b),
                        speed(a) * dt,
                        speed(b) * dt,
                        fraction,
                    ),
                    velocity: speed(a).lerp(speed(b), fraction),
                    simulation_time_seconds: time,
                    baseline_acceleration: field(a).lerp(field(b), fraction),
                    body_rotation: attitude(a).slerp(attitude(b), fraction),
                });
            }
        }
        if *active == ActiveGravityMethod::RadialAnalytic
            && inversion.truth_orbit.len() < ORBIT_HISTORY_LEN
        {
            inversion.truth_orbit.push(position(b));
        }
        if !benchmark.complete && b[0] <= BENCHMARK_DURATION_SECONDS + 1e-9 {
            benchmark.samples.push(GravityBenchmarkSample {
                simulation_time_seconds: b[0],
                position: position(b),
                velocity: speed(b),
            });
            if b[0] + 1e-9 >= BENCHMARK_DURATION_SECONDS {
                benchmark.complete = true;
                benchmark.capture_id = Some(hash_benchmark_trajectory(&benchmark.samples));
            }
        }
    }
    let last = records.last().expect("validated trace");
    // The independent backend may return fewer records than the legacy GPU
    // substep batch. Retain the latest authoritative state on every tick so
    // the visible orbit reflects real propagation rather than an empty trail.
    if orbit.0.len() >= ORBIT_HISTORY_LEN {
        orbit.0.pop_front();
    }
    orbit.0.push_back(position(last));
    transform.translation = position(last);
    velocity.0 = speed(last);
    clock.elapsed_seconds = (last[0] * 1e9).round() * 1e-9;
    clock.request_id = clock.request_id.wrapping_add(1);
    blend.0 = 1.0;
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
