use bevy::prelude::*;
use crate::components::*;

pub fn compute_asteroid_normals_system(
    mut commands: Commands,
    topo: Option<Res<AsteroidTopologyGpuData>>,
    existing: Option<Res<AsteroidNormalsGpuData>>,
) {
    let Some(topo) = topo else { return };
    if existing.is_some() { return; }

    let n = topo.node_count as usize;
    let mut normals = vec![Vec3::ZERO; n];

    for tri in topo.triangles.chunks(3) {
        if tri.len() < 3 { continue; }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n { continue; }

        let p0 = topo.positions[i0];
        let p1 = topo.positions[i1];
        let p2 = topo.positions[i2];
        let face = (p1 - p0).cross(p2 - p0); // weighted by triangle area
        normals[i0] += face;
        normals[i1] += face;
        normals[i2] += face;
    }

    for n in normals.iter_mut() {
        let len = n.length();
        *n = if len > 1e-6 { *n / len } else { Vec3::Y };
    }

    info!("CPU normals computed: {} vertices", normals.len());
    commands.insert_resource(AsteroidNormalsGpuData(normals));
}
