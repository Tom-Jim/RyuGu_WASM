use crate::interface::components::*;
use crate::cpu::frequency_domain::{AggregatedGravitySource, generate_fixed_point_trajectory};
use crate::cpu::inversion::{
    quintic_knot_accelerations, quintic_segment_position_acceleration,
};
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct ScientificGizmos;

pub fn configure_scientific_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<ScientificGizmos>();
    config.line.width = 1.75;
    config.line.perspective = false;
    config.line.joints = GizmoLineJoint::Round(4);
    config.depth_bias = -0.002;
}

pub fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    probe_initial: Res<ProbeInitialConditions>,
) {
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

    let _camera = commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: 100_000.0,
            near: 0.1,
            ..default()
        }),
        Transform::from_xyz(0.0, 800.0, 2500.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    )).id();

    // Mobile Dawn/Vulkan stacks are particularly prone to failing PBR pipeline
    // creation for multisampled targets (reported as VK_ERROR_UNKNOWN). The
    // simulation's compute paths stay exactly the same; this only selects the
    // single-sampled PBR variant, which is supported by the WebGPU baseline.
    #[cfg(target_arch = "wasm32")]
    if crate::browser_is_mobile() {
        commands.entity(_camera).insert(Msaa::Off);
    }

    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/ryugu.glb"))),
        TargetSize(900.0),
        Transform::from_xyz(0.0, 0.0, 0.0),
        RyuguMarker,
    ));

    commands.spawn((
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/cassini.gltf")),
        ),
        TargetSize(6.7),
        Transform::from_translation(probe_initial.position),
        Velocity(probe_initial.velocity()),
        OrbitHistory(std::collections::VecDeque::from([probe_initial.position])),
        CassiniMarker,
    ));
}

/// Builds one deterministic finite-length equation-(185) fixed-point arc and
/// exposes sixteen uniform knots. The resulting known trajectory is shared
/// by equation-(184), FFT, and FMM; it never depends on radial readback.
pub fn capture_trajectory_inversion_system(
    clock: Res<SimulationClock>,
    active_method: Res<ActiveGravityMethod>,
    frequency_domain_source: Option<Res<AggregatedGravitySource>>,
    probe_initial: Res<ProbeInitialConditions>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if inversion.runtime_epoch != clock.epoch {
        let queued_inversion = inversion.start_requested;
        let preserve_truth_track = inversion.preserve_truth_track;
        inversion.preserve_truth_track = false;
        inversion.runtime_epoch = clock.epoch;
        inversion.capture_epoch = clock.epoch;
        inversion.last_capture_request_id = None;
        inversion.wall_elapsed_seconds = 0.0;
        inversion.knots.clear();
        if !preserve_truth_track {
            inversion.truth_knots.clear();
            inversion.truth_capture_id = None;
            inversion.truth_capture_epoch = 0;
            inversion.truth_source_hash = 0;
            inversion.truth_orbit.clear();
        }
        inversion.capture_id = None;
        inversion.capture_source_hash =
            frequency_domain_source.as_ref().map_or(0, |source| source.source_hash);
        inversion.ready = false;
        inversion.inverted = false;
        inversion.start_requested = queued_inversion;
        inversion.error = None;
        inversion.optimizer = None;
        if !preserve_truth_track {
            inversion.batch_capture_id = None;
            inversion.displayed_density = None;
            inversion.results = std::array::from_fn(|_| None);
            inversion.best_results = std::array::from_fn(|_| None);
        }
        if preserve_truth_track && !inversion.truth_knots.is_empty() {
            inversion.knots = inversion.truth_knots.clone();
            inversion.capture_id = inversion.truth_capture_id;
            inversion.capture_epoch = inversion.truth_capture_epoch;
            inversion.capture_source_hash = inversion.truth_source_hash;
            inversion.ready = true;
            inversion.displayed_density = inversion.results[active_method.performance_index()]
                .clone()
                .or_else(|| inversion.best_results[active_method.performance_index()].clone());
        }
    }
    if !inversion.ready
        && let Some(source) = frequency_domain_source.as_ref()
    {
        inversion.capture_source_hash = source.source_hash;
    }
    if inversion.ready {
        return;
    }
    // Equation (185) fixed-point trajectory: all methods receive the same
    // deterministic finite arc, independent of radial-history readback.
    let Some(source) = frequency_domain_source.as_ref() else {
        return;
    };
    let knots = generate_fixed_point_trajectory(
        source,
        probe_initial.position,
        probe_initial.velocity(),
        TRAJECTORY_INVERSION_CAPTURE_SECONDS,
        TRAJECTORY_INVERSION_SAMPLE_COUNT,
    );
    let Some(knots) = knots else { return; };
    inversion.knots = knots;
    if inversion.truth_knots.is_empty() {
        inversion.truth_knots = inversion.knots.clone();
        inversion.truth_capture_id = Some(hash_trajectory_capture(&inversion.truth_knots));
        inversion.truth_capture_epoch = inversion.capture_epoch;
        inversion.truth_source_hash = inversion.capture_source_hash;
    }
    if !inversion.truth_knots.is_empty() {
        inversion.knots = inversion.truth_knots.clone();
    }
    inversion.capture_id = inversion.truth_capture_id;
    inversion.capture_epoch = inversion.truth_capture_epoch;
    inversion.capture_source_hash = inversion.truth_source_hash;
    inversion.ready = true;
}

pub(crate) fn hash_trajectory_capture(knots: &[TrajectoryInversionKnot]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for knot in knots {
        for value in [
            knot.simulation_time_seconds.to_bits(),
            knot.position.x.to_bits() as u64,
            knot.position.y.to_bits() as u64,
            knot.position.z.to_bits() as u64,
            knot.velocity.x.to_bits() as u64,
            knot.velocity.y.to_bits() as u64,
            knot.velocity.z.to_bits() as u64,
            knot.baseline_acceleration.x.to_bits() as u64,
            knot.baseline_acceleration.y.to_bits() as u64,
            knot.baseline_acceleration.z.to_bits() as u64,
            knot.body_rotation.x.to_bits() as u64,
            knot.body_rotation.y.to_bits() as u64,
            knot.body_rotation.z.to_bits() as u64,
            knot.body_rotation.w.to_bits() as u64,
        ] {
            hasher.write_u64(value);
        }
    }
    hasher.finish()
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

/// Zoom only the Bevy camera. The HTML overlay is intentionally not involved
/// so keyboard navigation cannot resize or translate the surrounding UI.
pub fn camera_keyboard_zoom_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<&mut PanOrbitCamera, With<Camera3d>>,
) {
    let zoom_in = keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::ArrowRight);
    let zoom_out = keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::ArrowLeft);
    if !zoom_in && !zoom_out {
        return;
    }
    let factor = if zoom_in { 0.88 } else { 1.14 };
    for mut camera in &mut cameras {
        camera.target_radius = (camera.target_radius.max(1.0) * factor).clamp(200.0, 20_000.0);
    }
}

pub fn render_gizmos_system(
    mut gizmos: Gizmos<ScientificGizmos>,
    camera_query: Query<&Transform, With<Camera3d>>,
    cassini_query: Query<(&Transform, &OrbitHistory), With<CassiniMarker>>,
    global_transforms: Query<&GlobalTransform>,
    show_normals: Res<ShowNormals>,
    topo: Option<Res<AsteroidTopologyGpuData>>,
    normals_data: Option<Res<AsteroidNormalsGpuData>>,
    active_method: Res<ActiveGravityMethod>,
    time: Res<Time>,
    inversion: Res<TrajectoryInversionState>,
) {
    let Some(cam) = camera_query.iter().next() else {
        return;
    };
    for (ct, history) in cassini_query.iter() {
        let orbit_color = match *active_method {
            ActiveGravityMethod::RadialAnalytic => Color::srgba(0.0, 1.0, 1.0, 0.8),
            ActiveGravityMethod::HomogeneousWerner => Color::srgba(1.0, 0.2, 0.2, 0.8),
            ActiveGravityMethod::FrequencyDomain => Color::srgba(0.8, 0.35, 1.0, 0.9),
            ActiveGravityMethod::MmfftCompressed => Color::srgba(1.0, 0.72, 0.2, 0.9),
            ActiveGravityMethod::Fmm => Color::srgba(0.25, 0.9, 0.55, 0.9),
        };
        if history.0.len() >= 2 {
            // The main trail always follows the detector's actual integrated
            // path. Frozen inversion samples are rendered separately below and
            // must never replace or hide this bounded live history. Decimate
            // only the display polyline: retaining all 27,500 simulation
            // samples in the resource preserves the physical history while a
            // bounded gizmo stream avoids rebuilding tens of thousands of
            // transient line vertices every frame.
            const MAX_ORBIT_GIZMO_POINTS: usize = 4_096;
            let stride = history.0.len().div_ceil(MAX_ORBIT_GIZMO_POINTS).max(1);
            let last_index = history.0.len() - 1;
            let append_last = (!last_index.is_multiple_of(stride)).then_some(last_index);
            gizmos.linestrip(
                (0..history.0.len())
                    .step_by(stride)
                    .chain(append_last)
                    .map(|index| history.0[index]),
                orbit_color,
            );
        }

        if cam.translation.distance(ct.translation) > VISIBILITY_THRESHOLD {
            let pos = ct.translation;
            gizmos
                .sphere(pos, 12.0, Color::srgb(1.0, 0.9, 0.1))
                .resolution(8);
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
    let display_knots: &[TrajectoryInversionKnot] = if *active_method
        != ActiveGravityMethod::HomogeneousWerner
        && inversion.truth_knots.len() == TRAJECTORY_INVERSION_SAMPLE_COUNT
    {
        &inversion.truth_knots
    } else if inversion.inverted {
        &inversion.knots
    } else {
        &[]
    };
    if display_knots.len() >= 2 {
        let mut curve = Vec::with_capacity((display_knots.len() - 1) * 25 + 1);
        if let Some(accelerations) = quintic_knot_accelerations(display_knots) {
            for index in 0..display_knots.len() - 1 {
                let start = display_knots[index];
                let end = display_knots[index + 1];
                for substep in 0..25 {
                    if index > 0 && substep == 0 {
                        continue;
                    }
                    let Some((position, _, _)) = quintic_segment_position_acceleration(
                        start,
                        end,
                        accelerations[index],
                        accelerations[index + 1],
                        substep as f32 / 24.0,
                    ) else {
                        continue;
                    };
                    curve.push(position);
                }
            }
        } else {
            // A malformed derivative estimate must not make valid captured
            // knots disappear; retain the honest piecewise-linear trajectory.
            curve.extend(display_knots.iter().map(|knot| knot.position));
        }
        if curve.len() >= 2 {
            let denominator = curve.len().saturating_sub(1).max(1) as f32;
            gizmos.linestrip_gradient(curve.iter().enumerate().map(|(index, position)| {
                let t = index as f32 / denominator;
                (
                    *position,
                    Color::hsl(315.0 - 45.0 * t, 0.92, 0.58 + 0.12 * t),
                )
            }));
        }
    }
    if show_normals.0
        && let (Some(topo), Some(normals)) = (topo, normals_data)
        && let Some(mesh_entity) = topo.mesh_entity
        && let Ok(mesh_gtf) = global_transforms.get(mesh_entity)
    {
        let rot = mesh_gtf.compute_transform().rotation;
        const MAX_VISIBLE_NORMALS: usize = 2_048;
        let available = topo.positions.len().min(normals.0.len());
        let stride = available.div_ceil(MAX_VISIBLE_NORMALS).max(1);
        for i in (0..available).step_by(stride) {
            let local_pos = topo.positions[i];
            let world_pos = mesh_gtf.transform_point(local_pos);
            let world_normal = (rot * normals.0[i]).normalize_or_zero();
            let tip = world_pos + world_normal * NORMAL_ARROW_LENGTH;
            gizmos.line(world_pos, tip, Color::srgb(0.2, 1.0, 0.8));
        }
    }
}
