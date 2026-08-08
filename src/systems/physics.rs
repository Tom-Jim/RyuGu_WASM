use crate::components::*;
use crate::systems::curved_arc::{CurvedArcPlannerState, approximate_eq106_acceleration};
use bevy::prelude::*;

const MAX_ACC: f32 = 1.5e-3;
const MAX_EXTRAPOLATION_INTERVALS: f64 = 2.0;

fn newtonian_acceleration(source: Vec3, target: Vec3, source_mass: f32) -> Vec3 {
    let displacement = source - target;
    let distance_squared = displacement.length_squared() + GRAVITY_EPSILON * GRAVITY_EPSILON;
    displacement / distance_squared.sqrt() * (G * source_mass / distance_squared)
}

fn clamp_acceleration(acceleration: Vec3) -> Vec3 {
    let magnitude = acceleration.length();
    if magnitude > MAX_ACC {
        acceleration * (MAX_ACC / magnitude)
    } else {
        acceleration
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

fn body_gravity_residual(sample: &GravityFieldSample) -> Vec3 {
    let point_mass = newtonian_acceleration(Vec3::ZERO, sample.snapshot.body_position, RYUGU_MASS);
    sample.body_acceleration - point_mass
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
        return Some(body_gravity_residual(latest));
    }

    if target_time <= latest.snapshot.simulation_time_seconds {
        let upper_index = samples
            .iter()
            .position(|sample| sample.snapshot.simulation_time_seconds >= target_time)
            .unwrap_or(samples.len() - 1);
        if upper_index == 0 {
            return Some(body_gravity_residual(samples[0]));
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
        let lower_value = body_gravity_residual(lower);
        let upper_value = body_gravity_residual(upper);
        let lower_tangent =
            (upper_value - body_gravity_residual(previous)) * (interval / lower_span) as f32;
        let upper_tangent =
            (body_gravity_residual(next) - lower_value) * (interval / upper_span) as f32;
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
    let latest_value = body_gravity_residual(latest);
    let previous_value = body_gravity_residual(previous);
    let latest_delta = latest_value - previous_value;
    let mut slope = latest_delta / interval as f32;
    if samples.len() >= 3 {
        let older = samples[samples.len() - 3];
        let older_interval = (previous.snapshot.simulation_time_seconds
            - older.snapshot.simulation_time_seconds)
            .max(f64::EPSILON);
        let older_slope = (previous_value - body_gravity_residual(older)) / older_interval as f32;
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

fn gpu_world_residual(
    history: &GravitySampleHistory,
    epoch: u64,
    target_time: f64,
    frame_start_time: f64,
    frame_start_rotation: Quat,
    maximum_extrapolation_intervals: f64,
) -> Option<Vec3> {
    let body_residual =
        predict_body_acceleration(history, epoch, target_time, maximum_extrapolation_intervals)?;
    let rotation = rotation_after(frame_start_rotation, target_time - frame_start_time);
    Some(rotation * body_residual)
}

pub fn physics_system(
    ryugu_query: Query<(&Transform, &Mass), (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    radial_history: Option<Res<RadialGravityHistory>>,
    werner_history: Option<Res<WernerGravityHistory>>,
    mut blend: ResMut<GravityBlendFactor>,
    mut clock: ResMut<SimulationClock>,
    time: Res<Time<Fixed>>,
    active_method: Res<ActiveGravityMethod>,
    curved_planner: Res<CurvedArcPlannerState>,
    simulation_acceleration: Res<SimulationAcceleration>,
) {
    let Some((ryugu_transform, ryugu_mass)) = ryugu_query.iter().next() else {
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
        // Eq. (106) transports the radial 70-style sample along the planned
        // convergent arc; the planner controls when this branch is usable.
        ActiveGravityMethod::CurvedArcEq106 => radial_history.as_ref().map(|history| &history.0),
    };
    let maximum_extrapolation_intervals = match *active_method {
        ActiveGravityMethod::RadialAnalytic => MAX_EXTRAPOLATION_INTERVALS,
        ActiveGravityMethod::HomogeneousWerner => 0.0,
        ActiveGravityMethod::CurvedArcEq106 => 0.0,
    };
    let gpu_ready = active_history
        .and_then(|history| history.latest_for_epoch(clock.epoch))
        .is_some();
    if gpu_ready {
        blend.0 = (blend.0 + 1.0 / GRAVITY_BLEND_FRAMES).min(1.0);
    }
    let blend_weight = blend.0;

    let stable_frame_dt = time.delta_secs_f64() * TIME_SCALE as f64;
    let substep_dt = stable_frame_dt / PHYSICS_SUBSTEPS as f64;
    let stable_steps = simulation_acceleration.stable_steps();
    let presented_frame_dt = stable_frame_dt * stable_steps as f64;
    let frame_start_time = clock.elapsed_seconds;
    let frame_start_rotation = ryugu_transform.rotation;

    let acceleration_at = |position: Vec3, sample_time: f64| {
        let fallback = newtonian_acceleration(ryugu_transform.translation, position, ryugu_mass.0);
        if *active_method == ActiveGravityMethod::CurvedArcEq106 {
            let Some(history) = active_history else {
                return fallback;
            };
            let Some(curved) = approximate_eq106_acceleration(
                history,
                clock.epoch,
                position,
                *ryugu_transform,
                ryugu_mass.0,
                &curved_planner,
            ) else {
                return fallback;
            };
            return clamp_acceleration(fallback + blend_weight * (curved - fallback));
        }
        let Some(history) = active_history else {
            return fallback;
        };
        let Some(gpu_residual) = gpu_world_residual(
            history,
            clock.epoch,
            sample_time,
            frame_start_time,
            frame_start_rotation,
            maximum_extrapolation_intervals,
        ) else {
            return fallback;
        };
        if !gpu_residual.is_finite() {
            fallback
        } else {
            clamp_acceleration(fallback + blend_weight * gpu_residual)
        }
    };

    // Each acceleration step completes an unchanged 12-substep leapfrog frame.
    // Intermediate states are retained in the orbit trail but are not presented,
    // which accelerates the visualization without enlarging the stable step size.
    for stable_step in 0..stable_steps {
        let stable_step_start = frame_start_time + stable_step as f64 * stable_frame_dt;
        for substep in 0..PHYSICS_SUBSTEPS {
            let start_time = stable_step_start + substep as f64 * substep_dt;
            let end_time = start_time + substep_dt;
            let acceleration_start = acceleration_at(probe_transform.translation, start_time);
            probe_velocity.0 += acceleration_start * (0.5 * substep_dt as f32);
            probe_transform.translation += probe_velocity.0 * substep_dt as f32;
            let acceleration_end = acceleration_at(probe_transform.translation, end_time);
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
    time: Res<Time<Fixed>>,
    simulation_acceleration: Res<SimulationAcceleration>,
) {
    let dt =
        time.delta_secs_f64() * TIME_SCALE as f64 * simulation_acceleration.stable_steps() as f64;
    for mut transform in ryugu_query.iter_mut() {
        transform.rotation = rotation_after(transform.rotation, dt);
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
            body_acceleration: acceleration,
            positive_potential: 1.0,
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
    fn predictor_bounds_long_extrapolation() {
        let mut history = GravitySampleHistory::default();
        history.push(sample(1, 0.0, Vec3::ZERO));
        history.push(sample(2, 10.0, Vec3::X));
        let predicted = predict_body_acceleration(&history, 1, 1_000.0, 2.0).unwrap();
        assert!((predicted - Vec3::X * 3.0).length() < 1.0e-5);
    }
}
