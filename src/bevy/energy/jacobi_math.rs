pub fn rotating_frame_jacobi_constant(
    body_position: Vec3,
    inertial_velocity_body_frame: Vec3,
    positive_gravitational_potential: f32,
    angular_velocity_body_frame: Vec3,
) -> Option<f64> {
    if !body_position.is_finite()
        || !inertial_velocity_body_frame.is_finite()
        || !positive_gravitational_potential.is_finite()
        || positive_gravitational_potential <= 0.0
        || !angular_velocity_body_frame.is_finite()
    {
        return None;
    }

    let frame_velocity =
        inertial_velocity_body_frame - angular_velocity_body_frame.cross(body_position);
    let centrifugal_speed = angular_velocity_body_frame.cross(body_position);
    let jacobi = 2.0 * positive_gravitational_potential as f64
        + centrifugal_speed.length_squared() as f64
        - frame_velocity.length_squared() as f64;
    jacobi.is_finite().then_some(jacobi)
}

/// Potential paired with the conservative local field used by the CPU Eq.106
/// substeps. For `g(x) = g0 + H (x-x0)`, this construction guarantees
/// `grad(U_loc) = g` when `H` is symmetric.
fn eq106_local_positive_potential(sample: &GravityFieldSample, body_position: Vec3) -> Option<f32> {
    let jacobian = sample.body_acceleration_jacobian?;
    let displacement = body_position - sample.snapshot.body_position;
    if !body_position.is_finite() || !displacement.is_finite() || !jacobian.is_finite() {
        return None;
    }
    let hessian = (jacobian + jacobian.transpose()) * 0.5;
    let potential = sample.positive_potential
        + sample.body_acceleration.dot(displacement)
        + 0.5 * displacement.dot(hessian * displacement);
    (potential.is_finite() && potential > 0.0).then_some(potential)
}

fn eq106_interpolated_positive_potential(
    history: &GravitySampleHistory,
    epoch: u64,
    simulation_time_seconds: f64,
    body_position: Vec3,
) -> Option<f32> {
    let (lower, upper) = history.bracketing(epoch, simulation_time_seconds)?;
    let lower_potential = eq106_local_positive_potential(lower, body_position)?;
    if std::ptr::eq(lower, upper) {
        return Some(lower_potential);
    }
    let upper_potential = eq106_local_positive_potential(upper, body_position)?;
    let interval = upper.snapshot.simulation_time_seconds - lower.snapshot.simulation_time_seconds;
    if interval <= f64::EPSILON {
        return Some(lower_potential);
    }
    let weight = ((simulation_time_seconds - lower.snapshot.simulation_time_seconds) / interval)
        .clamp(0.0, 1.0) as f32;
    lower_potential
        .lerp(upper_potential, weight)
        .is_finite()
        .then_some(lower_potential.lerp(upper_potential, weight))
}
