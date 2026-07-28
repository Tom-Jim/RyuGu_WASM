use crate::components::AsteroidTopologyGpuData;
use crate::welding::WeldedMesh;
use std::collections::HashSet;

pub fn build_topology(welded: &WeldedMesh) -> AsteroidTopologyGpuData {
    let n = welded.unique_positions.len();
    let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); n];

    for tri in welded.welded_indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        adj[a].insert(b as u32);
        adj[a].insert(c as u32);
        adj[b].insert(a as u32);
        adj[b].insert(c as u32);
        adj[c].insert(a as u32);
        adj[c].insert(b as u32);
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(n + 1);
    let mut indices: Vec<u32> = Vec::new();
    for neighbors in &adj {
        offsets.push(indices.len() as u32);
        let mut sorted: Vec<u32> = neighbors.iter().copied().collect();
        sorted.sort_unstable();
        indices.extend_from_slice(&sorted);
    }
    offsets.push(indices.len() as u32);

    AsteroidTopologyGpuData {
        mesh_entity: None,
        node_count: n as u32,
        positions: welded.unique_positions.clone(),
        triangles: welded.welded_indices.clone(),
        offsets,
        indices,
    }
}
