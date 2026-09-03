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
