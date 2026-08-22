use super::mesh_welding::WeldedMesh;
use crate::interface::components::AsteroidTopologyGpuData;

pub fn build_topology(welded: &WeldedMesh) -> AsteroidTopologyGpuData {
    let n = welded.unique_positions.len();
    // A HashSet per vertex creates almost 100,000 small heap allocations for
    // the Ryugu mesh and stalls single-threaded WASM startup. Sorting one
    // contiguous directed-edge array yields the identical CSR adjacency.
    let mut edges = Vec::with_capacity(welded.welded_indices.len() * 2);
    for tri in welded.welded_indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        edges.extend_from_slice(&[(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)]);
    }
    edges.sort_unstable();
    edges.dedup();

    let mut offsets = vec![0_u32; n + 1];
    for &(source, _) in &edges {
        if (source as usize) < n {
            offsets[source as usize + 1] += 1;
        }
    }
    for index in 1..offsets.len() {
        offsets[index] += offsets[index - 1];
    }
    let indices = edges
        .into_iter()
        .filter_map(|(source, target)| ((source as usize) < n).then_some(target))
        .collect();

    AsteroidTopologyGpuData {
        mesh_entity: None,
        node_count: n as u32,
        positions: welded.unique_positions.clone(),
        triangles: welded.welded_indices.clone(),
        offsets,
        indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;

    #[test]
    fn csr_adjacency_is_sorted_symmetric_and_deduplicated() {
        let welded = WeldedMesh {
            unique_positions: vec![Vec3::ZERO; 4],
            _vert_remap: vec![0, 1, 2, 3],
            welded_indices: vec![0, 1, 2, 2, 1, 3, 0, 1, 2],
        };
        let topology = build_topology(&welded);
        assert_eq!(topology.offsets, vec![0, 2, 5, 8, 10]);
        assert_eq!(topology.indices, vec![1, 2, 0, 2, 3, 0, 1, 3, 1, 2]);
    }
}
