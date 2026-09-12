#[derive(Default)]
pub(crate) struct SectionClipCache {
    topology_node_count: u32,
    vertex_count: usize,
    vertices: Vec<Vec3>,
}

/// Draws isoline contours on the camera-facing section. The density heatmap
/// itself is the `DensitySliceMaterial` quad that samples the baked 3-D volume.
pub fn render_section_system(
    mut gizmos: Gizmos<ScientificGizmos>,
    ryugu_query: Query<&Transform, With<RyuguMarker>>,
    camera_query: Query<&Transform, (With<Camera3d>, Without<RyuguMarker>)>,
    show_section: Res<ShowSection>,
    active_method: Res<ActiveGravityMethod>,
    density_mode: Res<DensityMode>,
    density_c: Option<Res<DensityC>>,
    werner_density: Option<Res<WernerDensity>>,
    inversion: Res<TrajectoryInversionState>,
    topo: Option<Res<AsteroidTopologyGpuData>>,
    mut frame: Local<u8>,
    mut clip_cache: Local<SectionClipCache>,
) {
    let inferred = if show_section.0 {
        None
    } else {
        inversion
            .displayed_density
            .as_ref()
            .filter(|result| result.method == *active_method)
    };
    if !show_section.0 && inferred.is_none() {
        *frame = 0;
        return;
    }
    *frame = frame.wrapping_add(1);
    if !(*frame).is_multiple_of(2) {
        return;
    }
    let Some(ryugu_tf) = ryugu_query.iter().next() else {
        return;
    };
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let Some(topo) = topo else {
        return;
    };
    let display_method = inferred.map_or(*active_method, |result| result.method);
    let c = inferred
        .filter(|result| result.method != ActiveGravityMethod::HomogeneousWerner)
        .map(|result| result.density)
        .unwrap_or_else(|| density_c.map(|r| r.0).unwrap_or(1.0));
    let uniform_forward_density = inferred.is_none() && *density_mode == DensityMode::Constant;
    let uniform_density = inferred
        .filter(|result| result.method == ActiveGravityMethod::HomogeneousWerner)
        .map(|result| result.density)
        .unwrap_or_else(|| werner_density.map(|r| r.0).unwrap_or(0.0));
    let homogeneous_display =
        display_method == ActiveGravityMethod::HomogeneousWerner || uniform_forward_density;
    if (!homogeneous_display && c <= 0.0) || (homogeneous_display && uniform_density <= 0.0) {
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

    let min_density = logarithmic_radial_density(0.0, c);
    let max_density = logarithmic_radial_density(SECTION_CLIP_RADIUS, c);
    let density_range = (max_density - min_density).max(1e-6);

    if clip_cache.topology_node_count != topo.node_count
        || clip_cache.vertex_count != topo.positions.len()
    {
        let stride = (topo.positions.len() / 512).max(1);
        clip_cache.vertices = topo.positions.iter().step_by(stride).copied().collect();
        clip_cache.topology_node_count = topo.node_count;
        clip_cache.vertex_count = topo.positions.len();
    }
    let local_verts = &clip_cache.vertices;
    let inv_rot = ryugu_tf.rotation.inverse();
    let inv_scale = 1.0 / ryugu_tf.scale.x;

    let inferred_range = inferred.map(|result| {
        let (minimum, maximum) = result.voxels.iter().map(|voxel| voxel.density).fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), density| (minimum.min(density), maximum.max(density)),
        );
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
    let grid_size = (steps * 2 + 1) as usize;
    let mut section_values = vec![0.0_f32; grid_size * grid_size];
    let mut section_inside = vec![false; grid_size * grid_size];

    for i in -steps..=steps {
        for j in -steps..=steps {
            let grid_index = ((i + steps) as usize) * grid_size + (j + steps) as usize;
            let point = com + tangent_u * (i as f32 * step_size) + tangent_v * (j as f32 * step_size);
            let body_pt = inv_rot * (point - com);
            let local_pt = body_pt * inv_scale;
            let dir = local_pt.normalize_or_zero();
            let is_inside = dir == Vec3::ZERO
                || local_pt.length()
                    <= local_verts
                        .iter()
                        .map(|p| p.dot(dir))
                        .fold(0.0_f32, f32::max);
            if !is_inside {
                continue;
            }
            let normalized_density =
                if let (Some(result), Some((mean, half_span))) = (inferred, inferred_range) {
                    let density =
                        crate::bevy_app::render::interpolated_inverted_density(result, body_pt);
                    (0.5 + (density - mean) / (2.0 * half_span)).clamp(0.0, 1.0)
                } else if homogeneous_display {
                    0.5
                } else {
                    let r = (point - com).length().max(0.01);
                    let density = logarithmic_radial_density(r, c);
                    ((density - min_density) / density_range).clamp(0.0, 1.0)
                };
            section_inside[grid_index] = true;
            section_values[grid_index] = normalized_density;
        }
    }

    let skip_internal_isolines = inferred.is_some_and(|result| {
        let (minimum, maximum) = result.voxels.iter().map(|voxel| voxel.density).fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), density| (minimum.min(density), maximum.max(density)),
        );
        maximum - minimum <= result.density.abs() * 0.05
    });
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
        !homogeneous_display && !skip_internal_isolines,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_section_contours(
    gizmos: &mut Gizmos<ScientificGizmos>,
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

    // Boundary silhouette: any inside/outside edge.
    for y in 0..grid_size {
        for x in 0..grid_size {
            let index = y * grid_size + x;
            if !inside[index] {
                continue;
            }
            let cell = Vec2::new(x as f32, y as f32);
            if x + 1 < grid_size && !inside[index + 1] {
                gizmos.line(
                    point(cell + Vec2::new(0.5, -0.5)),
                    point(cell + Vec2::new(0.5, 0.5)),
                    Color::srgba(0.96, 1.0, 1.0, 0.9),
                );
            }
            if y + 1 < grid_size && !inside[index + grid_size] {
                gizmos.line(
                    point(cell + Vec2::new(-0.5, 0.5)),
                    point(cell + Vec2::new(0.5, 0.5)),
                    Color::srgba(0.96, 1.0, 1.0, 0.9),
                );
            }
            if x == 0 {
                gizmos.line(
                    point(cell + Vec2::new(-0.5, -0.5)),
                    point(cell + Vec2::new(-0.5, 0.5)),
                    Color::srgba(0.96, 1.0, 1.0, 0.9),
                );
            }
            if y == 0 {
                gizmos.line(
                    point(cell + Vec2::new(-0.5, -0.5)),
                    point(cell + Vec2::new(0.5, -0.5)),
                    Color::srgba(0.96, 1.0, 1.0, 0.9),
                );
            }
        }
    }

    if !draw_internal {
        return;
    }

    let (minimum, maximum) = values
        .iter()
        .copied()
        .zip(inside.iter().copied())
        .filter(|&(_, is_inside)| is_inside)
        .map(|(value, _)| value)
        .fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
    let density_scale = minimum.abs().max(maximum.abs()).max(1.0);
    if !minimum.is_finite() || maximum - minimum <= density_scale * 1.0e-4 {
        return;
    }

    for level in 1..4 {
        let iso = minimum + (maximum - minimum) * (level as f32) / 4.0;
        for segment in marching_squares_segments(values, inside, grid_size, iso) {
            let (start, end) = segment;
            gizmos.line(point(start), point(end), Color::srgba(0.96, 1.0, 1.0, 0.55));
        }
    }
}

fn marching_squares_segments(
    values: &[f32],
    inside: &[bool],
    grid_size: usize,
    iso: f32,
) -> Vec<(Vec2, Vec2)> {
    let mut segments = Vec::new();
    let sample = |x: usize, y: usize| values[y * grid_size + x];
    let occupied = |x: usize, y: usize| inside[y * grid_size + x];
    for y in 0..grid_size - 1 {
        for x in 0..grid_size - 1 {
            if !(occupied(x, y)
                && occupied(x + 1, y)
                && occupied(x, y + 1)
                && occupied(x + 1, y + 1))
            {
                continue;
            }
            let v00 = sample(x, y);
            let v10 = sample(x + 1, y);
            let v01 = sample(x, y + 1);
            let v11 = sample(x + 1, y + 1);
            let mut code = 0_u8;
            if v00 >= iso {
                code |= 1;
            }
            if v10 >= iso {
                code |= 2;
            }
            if v11 >= iso {
                code |= 4;
            }
            if v01 >= iso {
                code |= 8;
            }
            if code == 0 || code == 15 {
                continue;
            }
            let lerp = |a: f32, b: f32| {
                let denom = (b - a).abs().max(1.0e-6);
                ((iso - a) / (b - a).copysign(denom)).clamp(0.0, 1.0)
            };
            let bottom = Vec2::new(x as f32 + lerp(v00, v10), y as f32);
            let right = Vec2::new((x + 1) as f32, y as f32 + lerp(v10, v11));
            let top = Vec2::new(x as f32 + lerp(v01, v11), (y + 1) as f32);
            let left = Vec2::new(x as f32, y as f32 + lerp(v00, v01));
            let push = |segments: &mut Vec<(Vec2, Vec2)>, a: Vec2, b: Vec2| {
                segments.push((a, b));
            };
            match code {
                1 | 14 => push(&mut segments, left, bottom),
                2 | 13 => push(&mut segments, bottom, right),
                3 | 12 => push(&mut segments, left, right),
                4 | 11 => push(&mut segments, right, top),
                6 | 9 => push(&mut segments, bottom, top),
                7 | 8 => push(&mut segments, left, top),
                5 => {
                    push(&mut segments, left, bottom);
                    push(&mut segments, right, top);
                }
                10 => {
                    push(&mut segments, bottom, right);
                    push(&mut segments, left, top);
                }
                _ => {}
            }
        }
    }
    segments
}

/// Toggles Ryugu's material alpha when ShowSection changes.
pub fn section_alpha_system(
    show_section: Res<ShowSection>,
    active_method: Res<ActiveGravityMethod>,
    inversion: Res<TrajectoryInversionState>,
    ryugu_query: Query<Entity, With<RyuguMarker>>,
    children_query: Query<&Children>,
    material_handles: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !show_section.is_changed() && !inversion.is_changed() && !active_method.is_changed() {
        return;
    }
    let overlay = inversion
        .displayed_density
        .as_ref()
        .is_some_and(|result| result.method == *active_method);
    let section_visible = show_section.0 || overlay;
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
