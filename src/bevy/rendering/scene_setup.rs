use crate::interface::components::*;
use crate::cpu::curved_arc::AggregatedGravitySource;
use crate::cpu::inversion::{
    quintic_knot_accelerations, quintic_segment_position_acceleration,
};
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

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

/// Records only presentation states for five wall-clock seconds, then maps
/// them to sixteen uniform knots.  This deliberately runs outside
/// `FixedUpdate`: no captured value can affect force evaluation or integration.
pub fn capture_trajectory_inversion_system(
    time: Res<Time>,
    clock: Res<SimulationClock>,
    active_method: Res<ActiveGravityMethod>,
    radial_history: Option<Res<RadialGravityHistory>>,
    eq106_source: Option<Res<AggregatedGravitySource>>,
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
        inversion.raw_samples.clear();
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
            eq106_source.as_ref().map_or(0, |source| source.source_hash);
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
        && let Some(source) = eq106_source.as_ref()
    {
        inversion.capture_source_hash = source.source_hash;
    }
    if inversion.ready || clock.elapsed_seconds <= 0.0 {
        return;
    }
    // Radial is the single authoritative observation producer. Eq.106,
    // MMFFT, and FMM all invert this frozen radial trajectory.
    let sample = radial_history
        .as_ref()
        .and_then(|history| history.0.latest_for_epoch(clock.epoch));
    let Some(sample) = sample else {
        return;
    };
    if inversion.last_capture_request_id == Some(sample.snapshot.request_id) {
        // GPU readback can be visible for several presentation frames. Do not
        // duplicate one snapshot in the wall-time capture; repeated anchors
        // make the 16-knot resampling depend on browser scheduling jitter.
        inversion.wall_elapsed_seconds += time.delta_secs_f64();
        return;
    }
    inversion.last_capture_request_id = Some(sample.snapshot.request_id);
    let baseline_acceleration = sample.snapshot.ryugu_transform.rotation * sample.body_acceleration;
    if !baseline_acceleration.is_finite() {
        return;
    }

    inversion.wall_elapsed_seconds += time.delta_secs_f64();
    let elapsed_seconds = inversion.wall_elapsed_seconds;
    inversion.raw_samples.push(TrajectoryCaptureSample {
        elapsed_seconds,
        knot: TrajectoryInversionKnot {
            position: sample.snapshot.probe_position,
            velocity: sample.snapshot.probe_velocity,
            simulation_time_seconds: sample.snapshot.simulation_time_seconds,
            baseline_acceleration,
            body_rotation: sample.snapshot.ryugu_transform.rotation,
        },
    });
    if inversion.wall_elapsed_seconds < TRAJECTORY_INVERSION_CAPTURE_SECONDS
        || inversion.raw_samples.len() < 2
    {
        return;
    }

    let samples = inversion.raw_samples.clone();
    let duration = samples.last().map_or(0.0, |sample| sample.elapsed_seconds);
    let mut knots = Vec::with_capacity(TRAJECTORY_INVERSION_SAMPLE_COUNT);
    for index in 0..TRAJECTORY_INVERSION_SAMPLE_COUNT {
        let target = duration * index as f64 / (TRAJECTORY_INVERSION_SAMPLE_COUNT - 1) as f64;
        let upper = samples
            .iter()
            .position(|sample| sample.elapsed_seconds >= target)
            .unwrap_or(samples.len() - 1);
        let lower = upper.saturating_sub(1);
        let a = samples[lower];
        let b = samples[upper];
        let span = (b.elapsed_seconds - a.elapsed_seconds).max(f64::EPSILON);
        let factor = ((target - a.elapsed_seconds) / span).clamp(0.0, 1.0) as f32;
        knots.push(TrajectoryInversionKnot {
            position: a.knot.position.lerp(b.knot.position, factor),
            velocity: a.knot.velocity.lerp(b.knot.velocity, factor),
            simulation_time_seconds: a.knot.simulation_time_seconds
                + (b.knot.simulation_time_seconds - a.knot.simulation_time_seconds) * factor as f64,
            baseline_acceleration: a
                .knot
                .baseline_acceleration
                .lerp(b.knot.baseline_acceleration, factor),
            body_rotation: a.knot.body_rotation.slerp(b.knot.body_rotation, factor),
        });
    }
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
    let mut hash = 1469598103934665603_u64;
    let mut absorb = |bytes: &[u8]| {
        for byte in bytes {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(1099511628211_u64);
        }
    };
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
            absorb(&value.to_le_bytes());
        }
    }
    hash
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
    inversion: Res<TrajectoryInversionState>,
) {
    let Some(cam) = camera_query.iter().next() else {
        return;
    };
    for (ct, history) in cassini_query.iter() {
        let orbit_color = match *active_method {
            ActiveGravityMethod::RadialAnalytic => Color::srgba(0.0, 1.0, 1.0, 0.8),
            ActiveGravityMethod::HomogeneousWerner => Color::srgba(1.0, 0.2, 0.2, 0.8),
            ActiveGravityMethod::CurvedArcEq106 => Color::srgba(0.8, 0.35, 1.0, 0.9),
            ActiveGravityMethod::MmfftCompressed => Color::srgba(1.0, 0.72, 0.2, 0.9),
            ActiveGravityMethod::Fmm => Color::srgba(0.25, 0.9, 0.55, 0.9),
        };
        if history.0.len() >= 2 {
            // The main trail always follows the detector's actual integrated
            // path. Frozen inversion samples are rendered separately below and
            // must never replace or hide this bounded live history.
            gizmos.linestrip(history.0.iter().copied(), orbit_color);
        }

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
        let Some(accelerations) = quintic_knot_accelerations(display_knots) else {
            return;
        };
        let mut curve = Vec::with_capacity((display_knots.len() - 1) * 25 + 1);
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
        // A gizmo line strip can be dropped wholesale by some WebGPU/browser
        // combinations when its backing transient buffer wraps.  Submit each
        // segment explicitly: whenever two trajectory points are visible,
        // their connecting polyline segment is visible as well.
        for segment in curve.windows(2) {
            gizmos.line(segment[0], segment[1], Color::srgb(1.0, 0.16, 0.72));
        }
        for (index, knot) in display_knots.iter().enumerate() {
            let hue = index as f32 / display_knots.len() as f32;
            let color = Color::hsl(hue * 300.0 + 15.0, 0.9, 0.6);
            gizmos.sphere(knot.position, 16.0, color);
            gizmos.line(
                knot.position - Vec3::X * 24.0,
                knot.position + Vec3::X * 24.0,
                color,
            );
            gizmos.line(
                knot.position - Vec3::Y * 24.0,
                knot.position + Vec3::Y * 24.0,
                color,
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
