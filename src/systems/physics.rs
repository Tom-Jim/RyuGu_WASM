use crate::components::*;
use crate::systems::werner_pipeline::WernerAcceleration;
use bevy::prelude::*;

pub fn physics_system(
    ryugu_query: Query<(&Transform, &Mass), (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut cassini_query: Query<
        (&mut Transform, &Mass, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    grav_acc: Res<GravityAcceleration>,
    werner_acc: Option<Res<WernerAcceleration>>,
    radial_source: Option<Res<RadialGravitySource>>,
    mut blend: ResMut<GravityBlendFactor>,
    time: Res<Time<Fixed>>,
    active_method: Res<ActiveGravityMethod>,
) {
    let dt = time.delta_secs() * TIME_SCALE;

    let Some((r_transform, r_mass)) = ryugu_query.iter().next() else {
        return;
    };
    let Some((mut c_transform, c_mass, mut c_velocity, mut c_history)) =
        cassini_query.iter_mut().next()
    else {
        return;
    };
    // Newtonian point-mass fallback
    let diff = r_transform.translation - c_transform.translation;
    let dist_sq = diff.length_squared() + GRAVITY_EPSILON * GRAVITY_EPSILON;
    let force_mag = G * r_mass.0 * c_mass.0 / dist_sq;
    let newtonian_acc = (diff / dist_sq.sqrt()) * force_mag / c_mass.0;

    const MAX_ACC: f32 = 1.5e-3;

    let acceleration = if radial_source.is_some() {
        let radial_gpu_acc = r_transform.rotation * grav_acc.0;
        let werner_gpu_acc = if let Some(acc) = &werner_acc {
            r_transform.rotation * acc.0
        } else {
            Vec3::ZERO
        };
        let chosen_acc = match *active_method {
            ActiveGravityMethod::RadialAnalytic => radial_gpu_acc,
            ActiveGravityMethod::HomogeneousWerner => werner_gpu_acc,
        };

        if !chosen_acc.is_finite() || chosen_acc == Vec3::ZERO {
            newtonian_acc
        } else {
            let mag = chosen_acc.length();
            let safe_gpu = if mag > MAX_ACC {
                chosen_acc * (MAX_ACC / mag)
            } else {
                chosen_acc
            };

            blend.0 = (blend.0 + 1.0 / GRAVITY_BLEND_FRAMES).min(1.0);
            newtonian_acc.lerp(safe_gpu, blend.0)
        }
    } else {
        newtonian_acc
    };

    // Euler-Cromer (semi-implicit Euler): symplectic map, phase-space volume conserved.
    c_velocity.0 += acceleration * dt;
    c_transform.translation += c_velocity.0 * dt;

    if c_history.0.len() >= ORBIT_HISTORY_LEN {
        c_history.0.pop_front();
    }
    c_history.0.push_back(c_transform.translation);
}
pub fn ryugu_rotation_system(
    mut ryugu_query: Query<&mut Transform, With<RyuguMarker>>,
    time: Res<Time<Fixed>>,
) {
    let dt = time.delta_secs() * TIME_SCALE;
    let angular_speed = std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS;
    let delta_angle = angular_speed * dt;
    let axis = RYUGU_SPIN_AXIS.normalize();
    let rotation = Quat::from_axis_angle(axis, delta_angle);

    for mut transform in ryugu_query.iter_mut() {
        transform.rotation = rotation * transform.rotation;
    }
}
