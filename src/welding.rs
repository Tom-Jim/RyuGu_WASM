use bevy::math::Vec3;
use std::collections::HashMap;

pub struct WeldedMesh {
    pub unique_positions: Vec<Vec3>,
    pub _vert_remap: Vec<u32>,
    pub welded_indices: Vec<u32>,
}

pub fn weld_mesh_vertices(positions: &[Vec3], indices: &[u32], epsilon: f32) -> WeldedMesh {
    let inv_eps = 1.0 / epsilon;
    let mut map: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut unique_positions: Vec<Vec3> = Vec::new();
    let mut vert_remap: Vec<u32> = Vec::with_capacity(positions.len());

    for &pos in positions {
        let key = (
            (pos.x * inv_eps).round() as i32,
            (pos.y * inv_eps).round() as i32,
            (pos.z * inv_eps).round() as i32,
        );
        let idx = *map.entry(key).or_insert_with(|| {
            let new_idx = unique_positions.len() as u32;
            unique_positions.push(pos);
            new_idx
        });
        vert_remap.push(idx);
    }

    let welded_indices = indices.iter().map(|&i| vert_remap[i as usize]).collect();

    WeldedMesh { unique_positions, _vert_remap: vert_remap, welded_indices }
}
