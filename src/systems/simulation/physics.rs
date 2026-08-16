use crate::components::*;
use crate::systems::curved_arc::CurvedArcResidualHistory;
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

fn eq106_local_body_acceleration(
    history: &GravitySampleHistory,
    epoch: u64,
    target_time: f64,
    target_body_position: Vec3,
) -> Option<Vec3> {
    let (lower, upper) = history.bracketing(epoch, target_time)?;
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
    time: Res<Time<Fixed>>,
    active_method: Res<ActiveGravityMethod>,
    simulation_acceleration: Res<SimulationAcceleration>,
) {
    if runtime_error.is_active() {
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

    let active_history = match *active_method {
        ActiveGravityMethod::RadialAnalytic => radial_history.as_ref().map(|history| &history.0),
        ActiveGravityMethod::HomogeneousWerner => werner_history.as_ref().map(|history| &history.0),
        ActiveGravityMethod::CurvedArcEq106 => eq106_history.as_ref().map(|history| &history.0),
        // MMFFT uses its own compressed-source readback history.
        ActiveGravityMethod::MmfftCompressed => mmfft_history.as_ref().map(|history| &history.0),
        ActiveGravityMethod::Fmm => fmm_history.as_ref().map(|history| &history.0),
    };
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

    // Each acceleration step completes an unchanged 12-substep leapfrog frame.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(request_id: u64, time: f64, acceleration: Vec3) -> GravityFieldSample {
        GravityFieldSample {
            snapshot: GravityRequestSnapshot {
                request_id,
                epoch: 1,
                simulation_time_seconds: time,
                body_position: Vec3::ZERO,
                ryugu_transform: Transform::IDENTITY,
                probe_position: Vec3::ZERO,
                probe_velocity: Vec3::ZERO,
            },
            predictive: false,
            body_acceleration: acceleration,
            positive_potential: 1.0,
            independent_positive_potential: None,
            body_acceleration_jacobian: None,
            eq106_diagnostics: None,
        }
    }

    #[test]
    fn predictor_interpolates_linear_history() {
        let mut history = GravitySampleHistory::default();
        history.push(sample(1, 0.0, Vec3::ZERO));
        history.push(sample(2, 10.0, Vec3::X));
        history.push(sample(3, 20.0, Vec3::X * 2.0));
        let predicted = predict_body_acceleration(&history, 1, 15.0, 2.0).unwrap();
        assert!((predicted - Vec3::X * 1.5).length() < 1.0e-5);
    }

    #[test]
    fn completed_lookup_skips_accelerated_predictive_anchors() {
        let mut history = GravitySampleHistory::default();
        let completed = sample(10, 100.0, Vec3::X);
        let mut predictive = sample(11, 200.0, Vec3::Y);
        predictive.predictive = true;
        history.push(completed);
        history.push(predictive);

        assert_eq!(
            history
                .completed_at_or_before(1, 200.0)
                .expect("completed anchor")
                .snapshot
                .request_id,
            10
        );
    }

    #[test]
    fn maximum_eq106_batch_keeps_its_authoritative_anchor() {
        let mut history = GravitySampleHistory::default();
        for block_index in 0..=MAX_SIMULATION_ACCELERATION {
            let mut anchor = sample(100 + block_index as u64, block_index as f64 * 10.0, Vec3::X);
            anchor.predictive = block_index > 0;
            history.push(anchor);
        }

        assert_eq!(history.samples.len(), 9);
        assert_eq!(
            history
                .completed_at_or_before(1, 80.0)
                .expect("authoritative 8x batch anchor")
                .snapshot
                .request_id,
            100
        );
    }

    #[test]
    fn predictor_bounds_long_extrapolation() {
        let mut history = GravitySampleHistory::default();
        history.push(sample(1, 0.0, Vec3::ZERO));
        history.push(sample(2, 10.0, Vec3::X));
        let predicted = predict_body_acceleration(&history, 1, 1_000.0, 2.0).unwrap();
        assert!((predicted - Vec3::X * 3.0).length() < 1.0e-5);
    }

    #[test]
    fn eq106_substep_uses_the_local_potential_hessian() {
        let mut history = GravitySampleHistory::default();
        let mut field = sample(1, 0.0, Vec3::new(1.0e-4, 0.0, 0.0));
        field.body_acceleration_jacobian = Some(Mat3::from_diagonal(Vec3::splat(2.0e-7)));
        history.push(field);

        let predicted =
            eq106_local_body_acceleration(&history, 1, 0.0, Vec3::new(10.0, -5.0, 2.0)).unwrap();
        let expected = Vec3::new(1.02e-4, -1.0e-6, 4.0e-7);
        assert!((predicted - expected).length() < 1.0e-10);
    }

    #[test]
    fn eq106_snapshot_must_match_the_unadvanced_clock() {
        let sample = sample(7, 42.0, Vec3::ZERO);
        let clock = SimulationClock {
            request_id: 7,
            epoch: 1,
            elapsed_seconds: 42.0,
        };
        assert!(eq106_snapshot_matches_clock(&sample, &clock));
        assert!(!eq106_snapshot_matches_clock(
            &sample,
            &SimulationClock {
                request_id: 8,
                ..clock
            }
        ));
    }
}
