use crate::components::*;
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

pub fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.8, 0.8, 1.0),
        brightness: 250.0,
        ..default()
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 80_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(1000.0, 2000.0, 1500.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: 100_000.0,
            near: 0.1,
            ..default()
        }),
        Transform::from_xyz(0.0, 800.0, 2500.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));

    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/ryugu.glb"))),
        TargetSize(900.0),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Mass(RYUGU_MASS),
        RyuguMarker,
    ));

    commands.spawn((
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/cassini.gltf")),
        ),
        TargetSize(6.7),
        Transform::from_translation(PROBE_R0),
        Mass(CASSINI_MASS),
        Velocity(*PROBE_V_INIT),
        OrbitHistory(std::collections::VecDeque::with_capacity(ORBIT_HISTORY_LEN)),
        CassiniMarker,
    ));
}

pub fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new(
            "Press 'S': View | Press 'F': Normals | Press 'D': Section | Mode: [Overview] | Normals: [OFF] | Section: [OFF]",
        ),
        TextFont {
            font_size: bevy::text::FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.9, 0.9)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(15.0),
            left: Val::Px(15.0),
            ..default()
        },
        UiTextMarker,
    ));
}

pub fn camera_switch_system(keyboard: Res<ButtonInput<KeyCode>>, mut mode: ResMut<CameraMode>) {
    if keyboard.just_pressed(KeyCode::KeyS) {
        *mode = match *mode {
            CameraMode::Overview => CameraMode::FollowCassini,
            CameraMode::FollowCassini => CameraMode::Overview,
        };
    }
}

pub fn camera_follow_system(
    mode: Res<CameraMode>,
    cassini_query: Query<&Transform, (With<CassiniMarker>, Without<Camera3d>)>,
    mut cam_query: Query<&mut PanOrbitCamera, With<Camera3d>>,
) {
    let Some(mut pan_orbit) = cam_query.iter_mut().next() else {
        return;
    };
    pan_orbit.target_focus = match *mode {
        CameraMode::Overview => Vec3::ZERO,
        CameraMode::FollowCassini => cassini_query
            .iter()
            .next()
            .map(|t| t.translation)
            .unwrap_or(Vec3::ZERO),
    };
}

pub fn render_gizmos_system(
    mut gizmos: Gizmos,
    camera_query: Query<&Transform, With<Camera3d>>,
    cassini_query: Query<(&Transform, &OrbitHistory), With<CassiniMarker>>,
    global_transforms: Query<&GlobalTransform>,
    show_normals: Res<ShowNormals>,
    topo: Option<Res<AsteroidTopologyGpuData>>,
    normals_data: Option<Res<AsteroidNormalsGpuData>>,
    active_method: Res<ActiveGravityMethod>,
    time: Res<Time>,
) {
    let Some(cam) = camera_query.iter().next() else {
        return;
    };
    for (ct, history) in cassini_query.iter() {
        let pts: Vec<Vec3> = history.0.iter().copied().collect();
        let orbit_color = match *active_method {
            ActiveGravityMethod::RadialAnalytic => Color::srgba(0.0, 1.0, 1.0, 0.8),
            ActiveGravityMethod::HomogeneousWerner => Color::srgba(1.0, 0.2, 0.2, 0.8),
        };
        gizmos.linestrip(pts, orbit_color);

        if cam.translation.distance(ct.translation) > VISIBILITY_THRESHOLD {
            let pos = ct.translation;
            gizmos.sphere(pos, 12.0, Color::srgb(1.0, 0.9, 0.1));
            let pulse = 20.0 + (time.elapsed_secs() * 5.0).sin() * 6.0;
            gizmos.circle(pos, pulse, Color::srgb(1.0, 0.6, 0.0));
            let d = 35.0_f32;
            gizmos.line(
                pos - Vec3::X * d,
                pos + Vec3::X * d,
                Color::srgb(1.0, 0.9, 0.1),
            );
            gizmos.line(
                pos - Vec3::Y * d,
                pos + Vec3::Y * d,
                Color::srgb(1.0, 0.9, 0.1),
            );
            gizmos.line(
                pos - Vec3::Z * d,
                pos + Vec3::Z * d,
                Color::srgb(1.0, 0.9, 0.1),
            );
        }
    }
    if show_normals.0
        && let (Some(topo), Some(normals)) = (topo, normals_data)
        && let Some(mesh_entity) = topo.mesh_entity
        && let Ok(mesh_gtf) = global_transforms.get(mesh_entity)
    {
        let rot = mesh_gtf.compute_transform().rotation;
        for (i, &local_pos) in topo.positions.iter().enumerate() {
            if i >= normals.0.len() {
                break;
            }
            let world_pos = mesh_gtf.transform_point(local_pos);
            let world_normal = (rot * normals.0[i]).normalize_or_zero();
            let tip = world_pos + world_normal * NORMAL_ARROW_LENGTH;
            gizmos.line(world_pos, tip, Color::srgb(0.2, 1.0, 0.8));
        }
    }
}

pub fn render_section_system(
    mut gizmos: Gizmos,
    ryugu_query: Query<&Transform, With<RyuguMarker>>,
    camera_query: Query<&Transform, (With<Camera3d>, Without<RyuguMarker>)>,
    show_section: Res<ShowSection>,
    active_method: Res<ActiveGravityMethod>,
    density_c: Option<Res<DensityC>>,
    werner_density: Option<Res<WernerDensity>>,
    topo: Option<Res<AsteroidTopologyGpuData>>,
) {
    if !show_section.0 {
        return;
    }
    let Some(ryugu_tf) = ryugu_query.iter().next() else {
        return;
    };
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let Some(topo) = topo else { return };
    let c = density_c.map(|r| r.0).unwrap_or(1.0);
    let uniform_density = werner_density.map(|r| r.0).unwrap_or(0.0);
    if (*active_method == ActiveGravityMethod::RadialAnalytic && c <= 0.0)
        || (*active_method == ActiveGravityMethod::HomogeneousWerner && uniform_density <= 0.0)
    {
        return;
    }

    let com = ryugu_tf.translation;
    let plane_normal = (cam_tf.translation - com).normalize_or_zero();
    if plane_normal == Vec3::ZERO {
        return;
    }

    let up = if plane_normal.abs().dot(Vec3::Y) < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent_u = plane_normal.cross(up).normalize();
    let tangent_v = plane_normal.cross(tangent_u).normalize();

    // Linear normalization for 1/r density field
    let eps = DENSITY_EPSILON;
    let max_density = c / eps;
    let min_density = c / (SECTION_CLIP_RADIUS + eps);
    let density_range = (max_density - min_density).max(1e-6);

    // Stride-sampled local vertices for mesh-boundary clipping (limits to ~2000 samples)
    let n_verts = topo.positions.len();
    let stride = (n_verts / 2000).max(1);
    let local_verts: Vec<Vec3> = topo.positions.iter().step_by(stride).copied().collect();

    // Decompose inverse transform: world → local mesh space (no Mat4::inverse needed)
    let inv_rot = ryugu_tf.rotation.inverse();
    let inv_scale = 1.0 / ryugu_tf.scale.x;

    let grid_half = 550.0_f32;
    let steps = 15_i32;
    let step_size = grid_half * 2.0 / (steps * 2) as f32;
    let dot_radius = step_size * 0.35;

    for i in -steps..=steps {
        for j in -steps..=steps {
            let u = i as f32 * step_size;
            let v = j as f32 * step_size;
            let point = com + tangent_u * u + tangent_v * v;

            // Transform sample point into asteroid local mesh space
            let local_pt = (inv_rot * (point - com)) * inv_scale;
            let dir = local_pt.normalize_or_zero();
            if dir == Vec3::ZERO {
                continue;
            }

            // Surface radius in dir: max vertex projection along dir
            // Rotates with the asteroid — boundary shape changes as Ryugu spins
            let r_surface = local_verts
                .iter()
                .map(|p| p.dot(dir))
                .fold(0.0_f32, f32::max);
            if local_pt.length() > r_surface {
                continue;
            }

            let color = match *active_method {
                ActiveGravityMethod::RadialAnalytic => {
                    // Continuous rho(r)=C/(r+epsilon) represented by equation (18).
                    let r = (point - com).length().max(0.01);
                    let density = c / (r + eps);
                    let t = ((density - min_density) / density_range).clamp(0.0, 1.0);

                    // Red center → yellow → cyan → blue/purple edge.
                    if t > 0.75 {
                        Color::srgb(1.0, (1.0 - t) * 4.0, 0.0)
                    } else if t > 0.35 {
                        Color::srgb((t - 0.35) * 2.5, 1.0, (0.75 - t) * 2.5)
                    } else {
                        Color::srgb(0.0, t * 2.85, 0.5 + 0.5 * (1.0 - t * 2.85))
                    }
                }
                // Every interior point has rho=M/V in the Werner model, so a
                // single color is the only faithful normalized visualization.
                ActiveGravityMethod::HomogeneousWerner => Color::srgb(0.15, 0.8, 1.0),
            };
            gizmos.sphere(point, dot_radius, color);
        }
    }
}

/// Toggles Ryugu's material alpha when ShowSection changes.
pub fn section_alpha_system(
    show_section: Res<ShowSection>,
    ryugu_query: Query<Entity, With<RyuguMarker>>,
    children_query: Query<&Children>,
    material_handles: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !show_section.is_changed() {
        return;
    }

    let Some(root) = ryugu_query.iter().next() else {
        return;
    };

    let mut stack = vec![root];
    while let Some(curr) = stack.pop() {
        if let Ok(handle) = material_handles.get(curr)
            && let Some(mut mat) = materials.get_mut(&handle.0)
        {
            let srgba = mat.base_color.to_srgba();
            if show_section.0 {
                mat.base_color = Color::srgba(srgba.red, srgba.green, srgba.blue, 0.25);
                mat.alpha_mode = AlphaMode::Blend;
            } else {
                mat.base_color = Color::srgba(srgba.red, srgba.green, srgba.blue, 1.0);
                mat.alpha_mode = AlphaMode::Opaque;
            }
        }
        if let Ok(children) = children_query.get(curr) {
            for child in children.iter() {
                stack.push(child);
            }
        }
    }
}
