pub(crate) fn build_surface_field_geometry_system(
    mut geometry: ResMut<SurfaceFieldGeometry>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    ryugu_query: Query<&Transform, With<RyuguMarker>>,
) {
    let Some(topology) = topology else { return };
    let Ok(transform) = ryugu_query.single() else {
        return;
    };
    let triangle_count = topology.triangles.len() / 3;
    let scale = transform.scale.x;
    if geometry.topology_node_count == topology.node_count
        && geometry.triangle_count == triangle_count
        && geometry.scale.to_bits() == scale.to_bits()
        && !geometry.patches.is_empty()
    {
        return;
    }

    let stride = triangle_count.div_ceil(SURFACE_PATCH_LIMIT).max(1);
    let mut patches = Vec::with_capacity(triangle_count.div_ceil(stride));
    let mut sampled_triangle_indices = Vec::with_capacity(triangle_count.div_ceil(stride));
    let mut render_triangles = Vec::with_capacity(triangle_count);
    for (triangle_index, triangle) in topology.triangles.as_chunks::<3>().0.iter().enumerate() {
        let Some((&a, rest)) = triangle.split_first() else {
            continue;
        };
        let [b, c] = [rest[0], rest[1]];
        let Some(p0) = topology.positions.get(a as usize).copied() else {
            continue;
        };
        let Some(p1) = topology.positions.get(b as usize).copied() else {
            continue;
        };
        let Some(p2) = topology.positions.get(c as usize).copied() else {
            continue;
        };
        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let local_centroid = (p0 + p1 + p2) / 3.0;
        if normal.dot(local_centroid) < 0.0 {
            normal = -normal;
        }
        if normal == Vec3::ZERO {
            continue;
        }
        let local_vertices = [p0, p1, p2];
        render_triangles.push((triangle_index, local_vertices, normal));
        if triangle_index.is_multiple_of(stride) {
            sampled_triangle_indices.push(triangle_index);
            patches.push(SurfaceFieldPatch {
                body_position: local_centroid * scale,
                normal,
            });
        }
    }
    let offset = 0.35 / scale.max(f32::MIN_POSITIVE);
    let mut render_positions = Vec::with_capacity(render_triangles.len() * 3);
    let mut render_normals = Vec::with_capacity(render_triangles.len() * 3);
    let mut render_indices = Vec::with_capacity(render_triangles.len() * 3);
    let mut render_sample_indices = Vec::with_capacity(render_triangles.len());
    for (triangle_index, local_vertices, normal) in render_triangles {
        let base = render_positions.len() as u32;
        for vertex in local_vertices {
            render_positions.push((vertex + normal * offset).to_array());
            render_normals.push(normal.to_array());
        }
        render_indices.extend_from_slice(&[base, base + 1, base + 2]);
        let sample_index = sampled_triangle_indices
            .partition_point(|&sample_triangle| sample_triangle <= triangle_index)
            .saturating_sub(1);
        render_sample_indices.push(sample_index.min(patches.len().saturating_sub(1)));
    }
    geometry.patches = patches;
    geometry.render_positions = render_positions;
    geometry.render_normals = render_normals;
    geometry.render_indices = render_indices;
    geometry.render_sample_indices = render_sample_indices;
    geometry.topology_node_count = topology.node_count;
    geometry.triangle_count = triangle_count;
    geometry.scale = scale;
}

pub(crate) fn ensure_surface_field_overlay_system(
    mut commands: Commands,
    geometry: Res<SurfaceFieldGeometry>,
    ryugu_query: Query<Entity, With<RyuguMarker>>,
    overlay_query: Query<Entity, With<SurfaceFieldOverlay>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if geometry.patches.is_empty()
        || geometry.render_sample_indices.is_empty()
        || overlay_query.iter().next().is_some()
    {
        return;
    }
    let Some(root) = ryugu_query.iter().next() else {
        return;
    };
    let positions = geometry.render_positions.clone();
    let normals = geometry.render_normals.clone();
    let colors = vec![[0.0, 0.0, 0.0, 0.0]; positions.len()];
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(geometry.render_indices.clone()));
    let mesh = meshes.add(mesh);
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        cull_mode: None,
        alpha_mode: AlphaMode::Opaque,
        depth_bias: 4.0,
        ..default()
    });
    let overlay = commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            SurfaceFieldOverlay,
            Visibility::Hidden,
        ))
        .id();
    commands.entity(root).add_child(overlay);
}
