use crate::cpu::frequency_domain::{AggregatedGravitySource, generate_fixed_point_trajectory};
use crate::cpu::inversion::{quintic_knot_accelerations, quintic_segment_position_acceleration};
use crate::interface::components::*;
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
    clock: Res<SimulationClock>,
    mut probe_visual: ResMut<ProbeVisualState>,
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

    let _camera = commands
        .spawn((
            Camera3d::default(),
            Projection::Perspective(PerspectiveProjection {
                far: 100_000.0,
                near: 0.1,
                ..default()
            }),
            Transform::from_xyz(0.0, 800.0, 2500.0).looking_at(Vec3::ZERO, Vec3::Y),
            PanOrbitCamera::default(),
        ))
        .id();

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

    let initial_position = probe_initial.position;
    let initial_velocity = probe_initial.velocity();
    probe_visual.reset(
        initial_position,
        initial_velocity,
        clock.epoch,
        bevy::platform::time::Instant::now(),
    );
    let probe = commands
        .spawn((
            Transform::from_translation(initial_position),
            Velocity(initial_velocity),
            OrbitHistory(std::collections::VecDeque::from([initial_position])),
            CassiniMarker,
        ))
        .id();
    let probe_visual_entity = commands
        .spawn((
            WorldAssetRoot(
                asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/cassini.gltf")),
            ),
            TargetSize(6.7),
            Transform::default(),
            ProbeVisualTransform {
                world_translation: initial_position,
            },
        ))
        .id();
    commands.entity(probe).add_child(probe_visual_entity);
}

/// Builds one deterministic finite-length equation-(185) fixed-point arc and
/// exposes sixteen uniform knots. The resulting known trajectory is shared
/// by equation-(184), FFT, and FMM; it never depends on radial readback.
pub fn capture_trajectory_inversion_system(
    clock: Res<SimulationClock>,
    active_method: Res<ActiveGravityMethod>,
    frequency_domain_source: Option<Res<AggregatedGravitySource>>,
    density_mode: Res<DensityMode>,
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
        inversion.capture_source_hash = frequency_domain_source.as_ref().map_or(0, |source| {
            match *density_mode {
                DensityMode::Variable => source.source_hash,
                DensityMode::Constant => source.constant_hash,
            }
        });
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
        inversion.capture_source_hash = match *density_mode {
            DensityMode::Variable => source.source_hash,
            DensityMode::Constant => source.constant_hash,
        };
    }
    if inversion.ready {
        return;
    }
    // Frequency-mode knots are accumulated by the causal Eq.106 GPU-driven
    // integrator, not by the direct reference generator or Radial history.
    let knots = if *active_method == ActiveGravityMethod::FrequencyDomain {
        if inversion.knots.len() != TRAJECTORY_INVERSION_SAMPLE_COUNT {
            return;
        }
        inversion.knots.clone()
    } else {
        let Some(source) = frequency_domain_source.as_ref() else {
            return;
        };
        let Some(knots) = generate_fixed_point_trajectory(
            source,
            *density_mode,
            probe_initial.position,
            probe_initial.velocity(),
            TRAJECTORY_INVERSION_CAPTURE_SECONDS,
            TRAJECTORY_INVERSION_SAMPLE_COUNT,
        ) else {
            return;
        };
        knots
    };
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
    probe_visual: Res<ProbeVisualState>,
    mut cam_query: Query<&mut PanOrbitCamera, With<Camera3d>>,
) {
    let Some(mut pan_orbit) = cam_query.iter_mut().next() else {
        return;
    };
    pan_orbit.target_focus = match *mode {
        CameraMode::Overview => Vec3::ZERO,
        CameraMode::FollowCassini => probe_visual.rendered_position,
    };
}

const PROBE_VISUAL_BLEND_SECONDS: f32 = 0.12;

fn probe_visual_extrapolation_limit(method: ActiveGravityMethod) -> f32 {
    // Keep this presentation-only horizon at twice the current backend pacing
    // interval. Stage D will remove the long pacing intervals after requests
    // become asynchronous; this guard remains useful if a worker stalls.
    match method {
        ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::FrequencyDomain => 0.5,
        ActiveGravityMethod::MmfftCompressed => 1.5,
        ActiveGravityMethod::HomogeneousWerner => 2.0,
        ActiveGravityMethod::Fmm => 3.0,
    }
}

fn extrapolated_probe_position(
    visual: &ProbeVisualState,
    method: ActiveGravityMethod,
    now: bevy::platform::time::Instant,
) -> Vec3 {
    let Some(sampled_at) = visual.authoritative_wall_time else {
        return visual.authoritative_position;
    };
    let elapsed = now
        .saturating_duration_since(sampled_at)
        .as_secs_f32()
        .min(probe_visual_extrapolation_limit(method));
    let extrapolated = visual.authoritative_position + visual.authoritative_velocity * elapsed;
    let Some(blend_started) = visual.blend_started else {
        return extrapolated;
    };
    let blend = (now
        .saturating_duration_since(blend_started)
        .as_secs_f32()
        / PROBE_VISUAL_BLEND_SECONDS)
        .clamp(0.0, 1.0);
    visual.blend_from.lerp(extrapolated, blend)
}

/// Updates only the Cassini model child. The authoritative parent transform is
/// intentionally read-only so no extrapolated value can reach physics data.
pub fn probe_visual_extrapolation_system(
    active_method: Res<ActiveGravityMethod>,
    clock: Res<SimulationClock>,
    authoritative_probe: Query<(&Transform, &Velocity), With<CassiniMarker>>,
    mut visual_probe: Query<(&mut Transform, &mut ProbeVisualTransform), Without<CassiniMarker>>,
    mut visual: ResMut<ProbeVisualState>,
) {
    let (Ok((authoritative_transform, velocity)), Ok((mut transform, mut visual_transform))) =
        (authoritative_probe.single(), visual_probe.single_mut())
    else {
        return;
    };
    let now = bevy::platform::time::Instant::now();
    if visual.authoritative_epoch != clock.epoch || visual.authoritative_wall_time.is_none() {
        visual.reset(
            authoritative_transform.translation,
            velocity.0,
            clock.epoch,
            now,
        );
    }
    let world_translation = extrapolated_probe_position(&visual, *active_method, now);
    visual.rendered_position = world_translation;
    visual_transform.world_translation = world_translation;

    // The visual entity is a child of the authoritative entity. Preserve the
    // child scale established by normalize_model_scale_system and express only
    // the presentation offset in the parent's local frame.
    transform.translation = authoritative_transform
        .to_matrix()
        .inverse()
        .transform_point3(world_translation);
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
    cassini_query: Query<&OrbitHistory, With<CassiniMarker>>,
    global_transforms: Query<&GlobalTransform>,
    show_normals: Res<ShowNormals>,
    topo: Option<Res<AsteroidTopologyGpuData>>,
    active_method: Res<ActiveGravityMethod>,
    time: Res<Time>,
    inversion: Res<TrajectoryInversionState>,
    probe_visual: Res<ProbeVisualState>,
) {
    let Some(cam) = camera_query.iter().next() else {
        return;
    };
    for history in cassini_query.iter() {
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
            // only the display polyline: retaining all 100,000 simulation
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

        if cam.translation.distance(probe_visual.rendered_position) > VISIBILITY_THRESHOLD {
            let pos = probe_visual.rendered_position;
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
    // Equation-(184) input knots are a complete known observation trajectory,
    // not the live inertial orbit. The frequency-domain mode now displays its
    // continuous rotating-body integration above; drawing the frozen knots on
    // top of it would create a second, misleading orbit.
    let display_knots: &[TrajectoryInversionKnot] =
        if *active_method == ActiveGravityMethod::FrequencyDomain {
            &[]
        } else if *active_method != ActiveGravityMethod::HomogeneousWerner
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
        && let Some(topo) = topo
        && let Some(mesh_entity) = topo.mesh_entity
        && let Ok(mesh_gtf) = global_transforms.get(mesh_entity)
    {
        let rot = mesh_gtf.compute_transform().rotation;
        // Draw one outward face normal for every triangular face. Vertex-normal
        // sampling hid entire regions on coarse meshes and made the display
        // depend on an arbitrary visibility cap.
        for triangle in topo.triangles.as_chunks::<3>().0 {
            let Some(p0) = topo.positions.get(triangle[0] as usize).copied() else {
                continue;
            };
            let Some(p1) = topo.positions.get(triangle[1] as usize).copied() else {
                continue;
            };
            let Some(p2) = topo.positions.get(triangle[2] as usize).copied() else {
                continue;
            };
            let local_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            if local_normal == Vec3::ZERO {
                continue;
            }
            let centroid = (p0 + p1 + p2) / 3.0;
            let outward_normal = if local_normal.dot(centroid) < 0.0 {
                -local_normal
            } else {
                local_normal
            };
            let world_pos = mesh_gtf.transform_point(centroid);
            let world_normal = (rot * outward_normal).normalize_or_zero();
            gizmos.line(
                world_pos,
                world_pos + world_normal * NORMAL_ARROW_LENGTH,
                Color::srgb(0.2, 1.0, 0.8),
            );
        }
    }
}

#[cfg(test)]
mod probe_visual_tests {
    use super::*;
    use std::collections::VecDeque;
    use std::time::Duration;

    #[test]
    fn visual_extrapolation_preserves_authoritative_state_and_orbit_history() {
        let authoritative_position = Vec3::new(10.0, 20.0, 30.0);
        let authoritative_velocity = Vec3::new(2.0, -1.0, 0.5);
        let orbit = VecDeque::from([Vec3::ZERO, authoritative_position]);
        let now = bevy::platform::time::Instant::now();
        let mut visual = ProbeVisualState::default();
        visual.reset(authoritative_position, authoritative_velocity, 7, now);
        visual.authoritative_wall_time = Some(now - Duration::from_millis(100));

        let mut app = App::new();
        app.insert_resource(ActiveGravityMethod::RadialAnalytic);
        app.insert_resource(SimulationClock {
            epoch: 7,
            ..Default::default()
        });
        app.insert_resource(visual);
        let authoritative = app
            .world_mut()
            .spawn((
                Transform::from_translation(authoritative_position),
                Velocity(authoritative_velocity),
                OrbitHistory(orbit.clone()),
                CassiniMarker,
            ))
            .id();
        let rendered = app
            .world_mut()
            .spawn((Transform::default(), ProbeVisualTransform::default()))
            .id();
        app.add_systems(Update, probe_visual_extrapolation_system);

        app.update();

        let authority = app.world().entity(authoritative);
        assert_eq!(authority.get::<Transform>().unwrap().translation, authoritative_position);
        assert_eq!(authority.get::<Velocity>().unwrap().0, authoritative_velocity);
        assert_eq!(authority.get::<OrbitHistory>().unwrap().0, orbit);
        assert_ne!(
            app.world()
                .entity(rendered)
                .get::<ProbeVisualTransform>()
                .unwrap()
                .world_translation,
            authoritative_position
        );
    }
}
