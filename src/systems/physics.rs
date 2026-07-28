use bevy::prelude::*;
use crate::components::*;

pub fn physics_system(
    ryugu_query: Query<(&Transform, &Mass), (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut cassini_query: Query<
        (&mut Transform, &Mass, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    grav_acc:    Res<GravityAcceleration>,
    grav_voxels: Option<Res<GravVoxelSource>>,
    time:        Res<Time>,
) {
    let dt = time.delta_secs() * TIME_SCALE;
    let Some((r_transform, r_mass)) = ryugu_query.iter().next() else { return };
    let Some((mut c_transform, c_mass, mut c_velocity, mut c_history)) =
        cassini_query.iter_mut().next()
    else {
        return;
    };

    let acceleration = if grav_voxels.is_some() && grav_acc.0 != Vec3::ZERO {
        // GPU result is in Ryugu body-fixed Cartesian — rotate to world frame
        r_transform.rotation * grav_acc.0
    } else {
        // Fallback: Newtonian point-mass until GPU pipeline warms up
        let diff      = r_transform.translation - c_transform.translation;
        let dist_sq   = diff.length_squared() + GRAVITY_EPSILON * GRAVITY_EPSILON;
        let force_mag = G * r_mass.0 * c_mass.0 / dist_sq;
        (diff / dist_sq.sqrt()) * force_mag / c_mass.0
    };

    c_velocity.0 += acceleration * dt;
    c_transform.translation += c_velocity.0 * dt;

    if c_history.0.len() >= ORBIT_HISTORY_LEN {
        c_history.0.pop_front();
    }
    c_history.0.push_back(c_transform.translation);
}

pub fn ryugu_rotation_system(
    mut ryugu_query: Query<&mut Transform, With<RyuguMarker>>,
    time: Res<Time>,
) {
    let dt            = time.delta_secs() * TIME_SCALE;
    let angular_speed = std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS;
    let delta_angle   = angular_speed * dt;
    let axis          = RYUGU_SPIN_AXIS.normalize();
    let rotation      = Quat::from_axis_angle(axis, delta_angle);
    for mut transform in ryugu_query.iter_mut() {
        transform.rotation = rotation * transform.rotation;
    }
}
