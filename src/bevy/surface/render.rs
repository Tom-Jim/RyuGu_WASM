pub(crate) fn surface_field_render_system(
    state: Res<SurfaceFieldState>,
    geometry: Res<SurfaceFieldGeometry>,
    mut overlay_query: Query<(&Mesh3d, &mut Visibility), With<SurfaceFieldOverlay>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut last_render: Local<(u64, SurfaceFieldMetric)>,
) {
    let Some((mesh_handle, mut visibility)) = overlay_query.iter_mut().next() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) else {
        return;
    };
    let has_error = state.comparison.is_some();
    let has_scalar = state.latest.is_some();
    if !has_error && !has_scalar {
        *visibility = Visibility::Hidden;
        return;
    }
    if last_render.0 == state.revision && last_render.1 == state.metric {
        *visibility = Visibility::Visible;
        return;
    }

    let mut colors = Vec::with_capacity(geometry.render_sample_indices.len() * 3);
    let mut push_color = |color: [f32; 4]| {
        colors.extend_from_slice(&[color, color, color]);
    };
    match state.metric {
        SurfaceFieldMetric::Error => {
            let Some(comparison) = state.comparison.as_ref() else {
                *visibility = Visibility::Hidden;
                return;
            };
            let maximum = comparison
                .error_range
                .0
                .abs()
                .max(comparison.error_range.1.abs())
                .max(1.0e-6);
            for &sample_index in &geometry.render_sample_indices {
                let error = comparison
                    .signed_errors
                    .get(sample_index)
                    .copied()
                    .unwrap_or(0.0);
                push_color(diverging_error_color(error / maximum));
            }
        }
        SurfaceFieldMetric::Gravity => {
            let Some(dataset) = state.latest.as_ref() else {
                return;
            };
            for &sample_index in &geometry.render_sample_indices {
                let Some(sample) = dataset.samples.get(sample_index) else {
                    continue;
                };
                let t = normalize_range(
                    sample.effective_gravity_magnitude,
                    dataset.effective_gravity_range,
                );
                push_color(scientific_color(t, SurfaceFieldMetric::Gravity));
            }
        }
        SurfaceFieldMetric::Gradient => {
            let Some(dataset) = state.latest.as_ref() else {
                return;
            };
            for &sample_index in &geometry.render_sample_indices {
                let Some(sample) = dataset.samples.get(sample_index) else {
                    continue;
                };
                let t = normalize_range(sample.gradient_magnitude, dataset.gradient_range);
                push_color(scientific_color(t, SurfaceFieldMetric::Gradient));
            }
        }
        SurfaceFieldMetric::Slope => {
            let Some(dataset) = state.latest.as_ref() else {
                return;
            };
            for &sample_index in &geometry.render_sample_indices {
                let Some(sample) = dataset.samples.get(sample_index) else {
                    continue;
                };
                let t = normalize_range(sample.slope_degrees, dataset.slope_range);
                push_color(scientific_color(t, SurfaceFieldMetric::Slope));
            }
        }
    }
    if colors.len() != geometry.render_sample_indices.len() * 3 {
        *visibility = Visibility::Hidden;
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    *visibility = Visibility::Visible;
    *last_render = (state.revision, state.metric);
}
