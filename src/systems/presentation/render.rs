use crate::components::*;
use crate::systems::inversion::{
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

/// Records only presentation states for five wall-clock seconds, then maps
/// them to sixteen uniform knots.  This deliberately runs outside
/// `FixedUpdate`: no captured value can affect force evaluation or integration.
pub fn capture_trajectory_inversion_system(
    time: Res<Time>,
    clock: Res<SimulationClock>,
    active_method: Res<ActiveGravityMethod>,
    radial_history: Option<Res<RadialGravityHistory>>,
    werner_history: Option<Res<WernerGravityHistory>>,
    eq106_history: Option<Res<Eq106GpuHistory>>,
    mmfft_history: Option<Res<MmfftCompressedHistory>>,
    fmm_history: Option<Res<FmmGravityHistory>>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if inversion.capture_epoch != clock.epoch {
        inversion.capture_epoch = clock.epoch;
        inversion.wall_elapsed_seconds = 0.0;
        inversion.raw_samples.clear();
        inversion.knots.clear();
        inversion.ready = false;
        inversion.knots_edited = false;
        inversion.inverted = false;
        inversion.selected = None;
        inversion.edit_buffer.clear();
        inversion.error = None;
        inversion.annealing = None;
        inversion.displayed_density = None;
    }
    if inversion.ready || clock.elapsed_seconds <= 0.0 {
        return;
    }
    let sample = match *active_method {
        ActiveGravityMethod::RadialAnalytic => radial_history
            .as_ref()
            .and_then(|history| history.0.latest_for_epoch(clock.epoch)),
        ActiveGravityMethod::HomogeneousWerner => werner_history
            .as_ref()
            .and_then(|history| history.0.latest_for_epoch(clock.epoch)),
        ActiveGravityMethod::CurvedArcEq106 => eq106_history.as_ref().and_then(|history| {
            history
                .0
                .completed_at_or_before(clock.epoch, clock.elapsed_seconds)
        }),
        ActiveGravityMethod::MmfftCompressed => mmfft_history
            .as_ref()
            .and_then(|history| history.0.latest_for_epoch(clock.epoch)),
        ActiveGravityMethod::Fmm => fmm_history
            .as_ref()
            .and_then(|history| history.0.latest_for_epoch(clock.epoch)),
    };
    let Some(sample) = sample else {
        return;
    };
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
    inversion.ready = true;
    inversion.knots_edited = false;
}

#[cfg(test)]
fn quintic_hermite_point(
    start: TrajectoryInversionKnot,
    end: TrajectoryInversionKnot,
    start_acceleration: Vec3,
    end_acceleration: Vec3,
    duration: f32,
    u: f32,
) -> Vec3 {
    let h = duration.max(f32::EPSILON);
    let h2 = h * h;
    let delta = end.position - start.position;
    let c0 = start.position;
    let c1 = start.velocity * h;
    let c2 = start_acceleration * (0.5 * h2);
    let c3 = delta * 10.0
        - start.velocity * (6.0 * h)
        - end.velocity * (4.0 * h)
        - start_acceleration * (1.5 * h2)
        + end_acceleration * (0.5 * h2);
    let c4 = delta * -15.0
        + start.velocity * (8.0 * h)
        + end.velocity * (7.0 * h)
        + start_acceleration * (1.5 * h2)
        - end_acceleration * h2;
    let c5 = delta * 6.0
        - (start.velocity + end.velocity) * (3.0 * h)
        - (start_acceleration - end_acceleration) * (0.5 * h2);
    ((((c5 * u + c4) * u + c3) * u + c2) * u + c1) * u + c0
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
    inversion: Res<TrajectoryInversionState>,
) {
    let Some(cam) = camera_query.iter().next() else {
        return;
    };
    for (ct, history) in cassini_query.iter() {
        let pts: Vec<Vec3> = history.0.iter().copied().collect();
        let orbit_color = match *active_method {
            ActiveGravityMethod::RadialAnalytic => Color::srgba(0.0, 1.0, 1.0, 0.8),
            ActiveGravityMethod::HomogeneousWerner => Color::srgba(1.0, 0.2, 0.2, 0.8),
            ActiveGravityMethod::CurvedArcEq106 => Color::srgba(0.8, 0.35, 1.0, 0.9),
            ActiveGravityMethod::MmfftCompressed => Color::srgba(1.0, 0.72, 0.2, 0.9),
            ActiveGravityMethod::Fmm => Color::srgba(0.25, 0.9, 0.55, 0.9),
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
    if inversion.inverted && inversion.knots.len() == TRAJECTORY_INVERSION_SAMPLE_COUNT {
        let Some(accelerations) = quintic_knot_accelerations(&inversion.knots) else {
            return;
        };
        let mut curve = Vec::with_capacity((TRAJECTORY_INVERSION_SAMPLE_COUNT - 1) * 25 + 1);
        for index in 0..TRAJECTORY_INVERSION_SAMPLE_COUNT - 1 {
            let start = inversion.knots[index];
            let end = inversion.knots[index + 1];
            for substep in 0..25 {
                if index > 0 && substep == 0 {
                    continue;
                }
                let Some((position, _)) = quintic_segment_position_acceleration(
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
        gizmos.linestrip(curve, Color::srgb(1.0, 0.16, 0.72));
        for (index, knot) in inversion.knots.iter().enumerate() {
            let hue = index as f32 / TRAJECTORY_INVERSION_SAMPLE_COUNT as f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn density_voxel(center: Vec3, density: f32) -> InvertedDensityVoxel {
        InvertedDensityVoxel {
            center,
            volume: 1.0,
            density,
            baseline_density: 1.0,
            reference_density: 1.0,
            grid: [0, 0, 0],
        }
    }

    #[test]
    fn quintic_hermite_reaches_both_knots() {
        let start = TrajectoryInversionKnot {
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::X,
            simulation_time_seconds: 0.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let end = TrajectoryInversionKnot {
            position: Vec3::new(8.0, -1.0, 4.0),
            velocity: Vec3::Y,
            simulation_time_seconds: 1.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        assert!(
            (quintic_hermite_point(start, end, Vec3::ZERO, Vec3::ZERO, 1.0, 0.0) - start.position)
                .length()
                < 1e-5
        );
        assert!(
            (quintic_hermite_point(start, end, Vec3::ZERO, Vec3::ZERO, 1.0, 1.0) - end.position)
                .length()
                < 1e-5
        );
    }

    #[test]
    fn inverted_section_interpolates_between_neighbouring_voxels() {
        let result = DensityInversionResult {
            method: ActiveGravityMethod::RadialAnalytic,
            density: 2.0,
            density_scale: 1.0,
            objective: 0.0,
            model_deviation: 0.0,
            model_fit: 1.0,
            objective_improvement: 0.0,
            trajectory_samples: 17,
            iterations: 1,
            voxel_size: 2.0,
            voxels: vec![
                density_voxel(Vec3::new(-1.0, 0.0, 0.0), 1.0),
                density_voxel(Vec3::new(1.0, 0.0, 0.0), 3.0),
            ],
        };

        let middle = interpolated_inverted_density(&result, Vec3::ZERO);
        assert!((middle - 2.0).abs() < 1.0e-5);
        assert!(interpolated_inverted_density(&result, Vec3::new(-0.9, 0.0, 0.0)) < middle);
        assert!(interpolated_inverted_density(&result, Vec3::new(0.9, 0.0, 0.0)) > middle);
    }

    #[test]
    fn marching_squares_draws_outline_and_skips_uniform_internal_levels() {
        let inside = [false, false, false, true, true, false, false, false, false];
        let outline = inside
            .iter()
            .map(|value| u8::from(*value) as f32)
            .collect::<Vec<_>>();
        assert!(!marching_squares_segments(&outline, &inside, 3, 0.5, false).is_empty());

        let uniform = vec![0.5; 9];
        assert!(marching_squares_segments(&uniform, &inside, 3, 0.5, true).is_empty());
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
    inversion: Res<TrajectoryInversionState>,
    topo: Option<Res<AsteroidTopologyGpuData>>,
) {
    // D remains an explicit view of the forward model's prior density. With D
    // off, a completed inversion samples its independently recovered 3-D
    // voxel field on the same rotating camera-facing section.
    let inferred = if show_section.0 {
        None
    } else {
        inversion.displayed_density.as_ref()
    };
    if !show_section.0 && inferred.is_none() {
        return;
    }
    let Some(ryugu_tf) = ryugu_query.iter().next() else {
        return;
    };
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let Some(topo) = topo else { return };
    let display_method = inferred.map_or(*active_method, |result| result.method);
    let c = inferred
        .filter(|result| result.method != ActiveGravityMethod::HomogeneousWerner)
        .map(|result| result.density)
        .unwrap_or_else(|| density_c.map(|r| r.0).unwrap_or(1.0));
    let uniform_density = inferred
        .filter(|result| result.method == ActiveGravityMethod::HomogeneousWerner)
        .map(|result| result.density)
        .unwrap_or_else(|| werner_density.map(|r| r.0).unwrap_or(0.0));
    if (display_method != ActiveGravityMethod::HomogeneousWerner && c <= 0.0)
        || (display_method == ActiveGravityMethod::HomogeneousWerner && uniform_density <= 0.0)
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

    // Linear normalization for the shared rho(r)=C ln(1+r/epsilon) field. The
    // radial, Eq.106, and MMFFT paths all use this same source law.
    let min_density = logarithmic_radial_density(0.0, c);
    let max_density = logarithmic_radial_density(SECTION_CLIP_RADIUS, c);
    let density_range = (max_density - min_density).max(1e-6);

    // Stride-sampled local vertices for mesh-boundary clipping (limits to ~2000 samples)
    let n_verts = topo.positions.len();
    let stride = (n_verts / 2000).max(1);
    let local_verts: Vec<Vec3> = topo.positions.iter().step_by(stride).copied().collect();

    // Decompose inverse transform: world → body metres → local mesh space.
    // The recovered voxels live in body metres, while topology vertices retain
    // the unscaled mesh coordinates.
    let inv_rot = ryugu_tf.rotation.inverse();
    let inv_scale = 1.0 / ryugu_tf.scale.x;

    let inferred_range = inferred.map(|result| {
        let (minimum, maximum) = result.voxels.iter().map(|voxel| voxel.density).fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), density| (minimum.min(density), maximum.max(density)),
        );
        // Never stretch floating-point dust over the full palette. The old
        // min/max normalization turned an effectively uniform solution into
        // red/yellow speckle that appeared to flash as the section rotated.
        let half_span = (maximum - result.density)
            .abs()
            .max((result.density - minimum).abs())
            .max(result.density.abs() * 0.05)
            .max(f32::EPSILON);
        (result.density, half_span)
    });

    let grid_half = 550.0_f32;
    let steps = 15_i32;
    let step_size = grid_half * 2.0 / (steps * 2) as f32;
    let dot_radius = step_size * 0.35;
    let grid_size = (steps * 2 + 1) as usize;
    let mut section_values = vec![0.0_f32; grid_size * grid_size];
    let mut section_inside = vec![false; grid_size * grid_size];

    for i in -steps..=steps {
        for j in -steps..=steps {
            let grid_index = ((i + steps) as usize) * grid_size + (j + steps) as usize;
            let u = i as f32 * step_size;
            let v = j as f32 * step_size;
            let point = com + tangent_u * u + tangent_v * v;

            // Transform the camera-facing plane into the rotating asteroid.
            // Sampling in body space makes the recovered density section turn
            // continuously with Ryugu instead of remaining fixed on screen.
            let body_pt = inv_rot * (point - com);
            let local_pt = body_pt * inv_scale;
            let dir = local_pt.normalize_or_zero();
            // The origin is necessarily inside the star-shaped Ryugu mesh.
            // Every other sample is clipped against the rotating radial shell.
            let is_inside = dir == Vec3::ZERO
                || local_pt.length()
                    <= local_verts
                        .iter()
                        .map(|p| p.dot(dir))
                        .fold(0.0_f32, f32::max);
            if !is_inside {
                continue;
            }

            let (normalized_density, color) =
                if let (Some(result), Some((mean, half_span))) = (inferred, inferred_range) {
                    let density = interpolated_inverted_density(result, body_pt);
                    let t = (0.5 + (density - mean) / (2.0 * half_span)).clamp(0.0, 1.0);
                    (t, inverted_density_color(t, result.method))
                } else if display_method == ActiveGravityMethod::HomogeneousWerner {
                    // Every interior point has rho=M/V in the Werner model, so a
                    // single color is the only faithful normalized visualization.
                    (0.5, Color::srgb(0.15, 0.8, 1.0))
                } else {
                    // Radial, Eq.106, MMFFT, and FMM all consume the same
                    // mass-preserving logarithmic radial source. Use the actual
                    // normalized density at this section sample for every one of
                    // those modes; only the method-specific palette changes.
                    let r = (point - com).length().max(0.01);
                    let density = logarithmic_radial_density(r, c);
                    let t = ((density - min_density) / density_range).clamp(0.0, 1.0);
                    (t, heterogeneous_density_color(t, display_method))
                };
            section_inside[grid_index] = true;
            section_values[grid_index] = normalized_density;
            gizmos.sphere(point, dot_radius, color);
        }
    }

    draw_section_contours(
        &mut gizmos,
        &section_values,
        &section_inside,
        grid_size,
        steps,
        step_size,
        com,
        tangent_u,
        tangent_v,
        plane_normal,
        display_method != ActiveGravityMethod::HomogeneousWerner,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_section_contours(
    gizmos: &mut Gizmos,
    values: &[f32],
    inside: &[bool],
    grid_size: usize,
    steps: i32,
    step_size: f32,
    center: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    plane_normal: Vec3,
    draw_internal: bool,
) {
    let point = |grid: Vec2| {
        center
            + tangent_u * ((grid.x - steps as f32) * step_size)
            + tangent_v * ((grid.y - steps as f32) * step_size)
            + plane_normal * 3.0
    };
    let outline_values = inside
        .iter()
        .map(|is_inside| u8::from(*is_inside) as f32)
        .collect::<Vec<_>>();
    for (start, end) in marching_squares_segments(&outline_values, inside, grid_size, 0.5, false) {
        gizmos.line(
            point(start),
            point(end),
            Color::srgba(1.0, 0.96, 0.35, 0.98),
        );
    }

    if !draw_internal {
        return;
    }
    let (minimum, maximum) = values
        .iter()
        .zip(inside)
        .filter_map(|(value, is_inside)| is_inside.then_some(*value))
        .fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
    if !minimum.is_finite() || maximum - minimum < 0.02 {
        return;
    }

    // Draw every useful palette band. Marching squares naturally emits no
    // segment for a level whose surrounding samples contain no matching
    // transition, so isolated/nonexistent contours are not fabricated.
    for band in 1..=9 {
        let level = minimum + (maximum - minimum) * band as f32 / 10.0;
        for (start, end) in marching_squares_segments(values, inside, grid_size, level, true) {
            gizmos.line(point(start), point(end), Color::srgba(0.96, 1.0, 1.0, 0.82));
        }
    }
}

fn marching_squares_segments(
    values: &[f32],
    inside: &[bool],
    grid_size: usize,
    level: f32,
    require_inside: bool,
) -> Vec<(Vec2, Vec2)> {
    let mut segments = Vec::new();
    if grid_size < 2 {
        return segments;
    }
    for x in 0..grid_size - 1 {
        for y in 0..grid_size - 1 {
            let indices = [
                x * grid_size + y,
                (x + 1) * grid_size + y,
                (x + 1) * grid_size + y + 1,
                x * grid_size + y + 1,
            ];
            if require_inside && indices.iter().any(|index| !inside[*index]) {
                continue;
            }
            let corners = [
                Vec2::new(x as f32, y as f32),
                Vec2::new((x + 1) as f32, y as f32),
                Vec2::new((x + 1) as f32, (y + 1) as f32),
                Vec2::new(x as f32, (y + 1) as f32),
            ];
            let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
            let mut crossings = Vec::with_capacity(4);
            for (start, end) in edges {
                let start_value = values[indices[start]];
                let end_value = values[indices[end]];
                if (start_value >= level) == (end_value >= level) {
                    continue;
                }
                let fraction = ((level - start_value) / (end_value - start_value)).clamp(0.0, 1.0);
                crossings.push(corners[start].lerp(corners[end], fraction));
            }
            match crossings.as_slice() {
                [a, b] => segments.push((*a, *b)),
                [a, b, c, d] => {
                    // Resolve the saddle consistently from the cell centre.
                    let center_above =
                        indices.iter().map(|index| values[*index]).sum::<f32>() * 0.25 >= level;
                    if center_above {
                        segments.push((*a, *d));
                        segments.push((*b, *c));
                    } else {
                        segments.push((*a, *b));
                        segments.push((*c, *d));
                    }
                }
                _ => {}
            }
        }
    }
    segments
}

/// Reconstructs a continuous section from the coarse, independently annealed
/// voxel values. A compact kernel uses only neighbouring cells, so the display
/// does not invent long-range density gradients; nearest-voxel fallback keeps
/// every interior mesh sample defined at irregular boundary cells.
fn interpolated_inverted_density(result: &DensityInversionResult, body_point: Vec3) -> f32 {
    let support = (result.voxel_size * 1.75).max(f32::MIN_POSITIVE);
    let support_squared = support * support;
    let mut weighted_density = 0.0_f32;
    let mut total_weight = 0.0_f32;
    let mut nearest = (f32::INFINITY, result.density);

    for voxel in &result.voxels {
        let distance_squared = body_point.distance_squared(voxel.center);
        if distance_squared < nearest.0 {
            nearest = (distance_squared, voxel.density);
        }
        if distance_squared < support_squared {
            let q_squared = distance_squared / support_squared;
            let weight = (1.0 - q_squared).powi(2);
            weighted_density += weight * voxel.density;
            total_weight += weight;
        }
    }

    if total_weight > f32::EPSILON {
        weighted_density / total_weight
    } else {
        nearest.1
    }
}

fn inverted_density_color(t: f32, method: ActiveGravityMethod) -> Color {
    if method == ActiveGravityMethod::HomogeneousWerner {
        let low = Vec3::new(0.08, 0.35, 0.65);
        let middle = Vec3::new(0.0, 0.75, 1.0);
        let high = Vec3::new(0.88, 1.0, 1.0);
        let rgb = if t < 0.5 {
            low.lerp(middle, t * 2.0)
        } else {
            middle.lerp(high, (t - 0.5) * 2.0)
        };
        Color::srgb(rgb.x, rgb.y, rgb.z)
    } else {
        heterogeneous_density_color(t, method)
    }
}

fn heterogeneous_density_color(t: f32, method: ActiveGravityMethod) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (outer, middle, core) = match method {
        // Cyan trajectory: warm density complement.
        ActiveGravityMethod::RadialAnalytic => (
            Vec3::new(0.45, 0.04, 0.02),
            Vec3::new(1.0, 0.18, 0.015),
            Vec3::new(1.0, 0.95, 0.12),
        ),
        // Violet trajectory: green/teal density complement.
        ActiveGravityMethod::CurvedArcEq106 => (
            Vec3::new(0.02, 0.35, 0.45),
            Vec3::new(0.02, 0.9, 0.42),
            Vec3::new(0.82, 1.0, 0.12),
        ),
        // Orange trajectory: blue/cyan density complement.
        ActiveGravityMethod::MmfftCompressed => (
            Vec3::new(0.12, 0.22, 0.65),
            Vec3::new(0.02, 0.58, 1.0),
            Vec3::new(0.72, 1.0, 1.0),
        ),
        // Green trajectory: magenta/pink density complement.
        ActiveGravityMethod::Fmm => (
            Vec3::new(0.4, 0.05, 0.5),
            Vec3::new(0.95, 0.06, 0.72),
            Vec3::new(1.0, 0.78, 0.92),
        ),
        ActiveGravityMethod::HomogeneousWerner => {
            return Color::srgb(0.15, 0.8, 1.0);
        }
    };
    let rgb = if t < 0.5 {
        outer.lerp(middle, t * 2.0)
    } else {
        middle.lerp(core, (t - 0.5) * 2.0)
    };
    Color::srgb(rgb.x, rgb.y, rgb.z)
}

/// Toggles Ryugu's material alpha when ShowSection changes.
pub fn section_alpha_system(
    show_section: Res<ShowSection>,
    inversion: Res<TrajectoryInversionState>,
    ryugu_query: Query<Entity, With<RyuguMarker>>,
    children_query: Query<&Children>,
    material_handles: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !show_section.is_changed() && !inversion.is_changed() {
        return;
    }
    let section_visible = show_section.0 || inversion.displayed_density.is_some();

    let Some(root) = ryugu_query.iter().next() else {
        return;
    };

    let mut stack = vec![root];
    while let Some(curr) = stack.pop() {
        if let Ok(handle) = material_handles.get(curr)
            && let Some(mut mat) = materials.get_mut(&handle.0)
        {
            let srgba = mat.base_color.to_srgba();
            if section_visible {
                let alpha = if show_section.0 { 0.25 } else { 0.20 };
                mat.base_color = Color::srgba(srgba.red, srgba.green, srgba.blue, alpha);
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

#[cfg(test)]
mod section_density_color_tests {
    use super::*;

    #[test]
    fn every_log_density_method_has_a_visible_gradient() {
        for method in [
            ActiveGravityMethod::RadialAnalytic,
            ActiveGravityMethod::CurvedArcEq106,
            ActiveGravityMethod::MmfftCompressed,
            ActiveGravityMethod::Fmm,
        ] {
            assert_ne!(
                heterogeneous_density_color(0.0, method),
                heterogeneous_density_color(1.0, method)
            );
        }
    }
}
