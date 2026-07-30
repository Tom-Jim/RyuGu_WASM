use crate::components::*;
use crate::topology::build_topology;
use crate::welding::weld_mesh_vertices;
use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, VertexAttributeValues};

pub fn normalize_model_scale_system(
    mut commands: Commands,
    target_query: Query<(Entity, &TargetSize), Without<ScaleNormalized>>,
    children_query: Query<&Children>,
    aabb_query: Query<&Aabb>,
) {
    for (entity, target_size) in &target_query {
        if let Some(max_dim) = find_max_aabb_extent(entity, &children_query, &aabb_query) {
            if max_dim > 0.00001 {
                let scale_factor = target_size.0 / max_dim;
                commands.entity(entity).insert(ScaleNormalized);
                commands
                    .entity(entity)
                    .entry::<Transform>()
                    .and_modify(move |mut t| t.scale = Vec3::splat(scale_factor));
            }
        }
    }
}

fn find_max_aabb_extent(
    entity: Entity,
    children_query: &Query<&Children>,
    aabb_query: &Query<&Aabb>,
) -> Option<f32> {
    let mut max_extent = 0.0f32;
    let mut found = false;
    let mut stack = vec![entity];
    while let Some(curr) = stack.pop() {
        if let Ok(aabb) = aabb_query.get(curr) {
            let size = aabb.half_extents * 2.0;
            let m = size.x.max(size.y).max(size.z);
            if m > max_extent {
                max_extent = m;
                found = true;
            }
        }
        if let Ok(children) = children_query.get(curr) {
            for child in children.iter() {
                stack.push(child);
            }
        }
    }
    if found { Some(max_extent) } else { None }
}

pub fn build_topology_system(
    mut commands: Commands,
    ryugu_query: Query<
        (Entity, &Transform),
        (
            With<RyuguMarker>,
            With<ScaleNormalized>,
            Without<TopologyBuilt>,
        ),
    >,
    children_query: Query<&Children>,
    mesh3d_query: Query<&Mesh3d>,
    meshes: Res<Assets<Mesh>>,
) {
    let Some((entity, ryugu_transform)) = ryugu_query.iter().next() else {
        return;
    };
    let scale = ryugu_transform.scale.x;

    let Some((mesh_entity, handle)) =
        find_mesh_entity_in_children(entity, &children_query, &mesh3d_query)
    else {
        return;
    };
    let Some(mesh) = meshes.get(&handle) else {
        return;
    };
    let Some(VertexAttributeValues::Float32x3(raw)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return;
    };

    let positions: Vec<Vec3> = raw.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    let flat_indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(idx)) => idx.clone(),
        Some(Indices::U16(idx)) => idx.iter().map(|&i| i as u32).collect(),
        None => (0..positions.len() as u32).collect(),
    };

    let welded = weld_mesh_vertices(&positions, &flat_indices, 1e-4);
    let mut topo = build_topology(&welded);
    topo.mesh_entity = Some(mesh_entity);

    // Compute density constant C via 4-point Gaussian quadrature over signed tetrahedra.
    // Kernel: 1/(||r|| + ε), integrated over volume decomposed into origin-anchored tets.
    let a = (5.0 + 3.0 * 5.0_f32.sqrt()) / 20.0; // ≈ 0.5854102
    let b = (5.0 - 5.0_f32.sqrt()) / 20.0;        // ≈ 0.1381966
    let eps = DENSITY_EPSILON as f64;
    let mut total_integral = 0.0_f64;

    for chunk in topo.triangles.chunks(3) {
        let va = (topo.positions[chunk[0] as usize] * scale).as_dvec3();
        let vb = (topo.positions[chunk[1] as usize] * scale).as_dvec3();
        let vc = (topo.positions[chunk[2] as usize] * scale).as_dvec3();

        let vol = va.dot(vb.cross(vc)) / 6.0;

        let p1 = va * a as f64 + vb * b as f64 + vc * b as f64;
        let p2 = va * b as f64 + vb * a as f64 + vc * b as f64;
        let p3 = va * b as f64 + vb * b as f64 + vc * a as f64;
        let p4 = va * b as f64 + vb * b as f64 + vc * b as f64;

        let f = |p: bevy::math::DVec3| {
            1.0 / (p.length() + eps)
        };

        let sum = f(p1) + f(p2) + f(p3) + f(p4);
        total_integral += vol * 0.25 * sum;
    }

    let density_c = if total_integral.abs() > 1e-20 {
        (RYUGU_MASS as f64 / total_integral) as f32
    } else {
        1.0_f32
    };

    commands.entity(entity).insert(TopologyBuilt);
    commands.insert_resource(topo);
    commands.insert_resource(DensityC(density_c));
}

fn find_mesh_entity_in_children(
    entity: Entity,
    children_query: &Query<&Children>,
    mesh3d_query: &Query<&Mesh3d>,
) -> Option<(Entity, Handle<Mesh>)> {
    let mut stack = vec![entity];
    while let Some(curr) = stack.pop() {
        if let Ok(m) = mesh3d_query.get(curr) {
            return Some((curr, m.0.clone()));
        }
        if let Ok(children) = children_query.get(curr) {
            for child in children.iter() {
                stack.push(child);
            }
        }
    }
    None
}
