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

    // Decompose inverse transform: world -> body metres -> local mesh space.
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
    // The forward D-section keeps its larger sample markers. In inversion
    // view use small translucent samples so the convex-QP-predicted density
    // remains visible without covering the section in overlapping spheres.
    let dot_radius = if inferred.is_some() {
        step_size * 0.10
    } else {
        step_size * 0.35
    };
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
            let marker_color = if inferred.is_some() {
                let srgba = color.to_srgba();
                Color::srgba(srgba.red, srgba.green, srgba.blue, 0.58)
            } else {
                color
            };
            gizmos.sphere(point, dot_radius, marker_color);
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
    // Interpolation of an exactly uniform voxel field still accumulates small
    // floating-point differences. Use a relative threshold so marching
    // squares does not magnify numerical dust into dozens of false loops.
    let density_scale = minimum.abs().max(maximum.abs()).max(1.0);
    if !minimum.is_finite() || maximum - minimum <= density_scale * 1.0e-4 {
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

/// Reconstructs a continuous section from the coarse, independently optimized
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
