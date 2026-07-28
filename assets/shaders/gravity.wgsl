struct GravParams {
    probe_x:     f32,
    probe_y:     f32,
    probe_z:     f32,
    G_const:     f32,
    voxel_count: u32,
    _pad0:       f32,
    _pad1:       f32,
    _pad2:       f32,
};

struct VoxelData {
    x:    f32,
    y:    f32,
    z:    f32,
    mass: f32,
};

@group(0) @binding(0) var<uniform>             params:     GravParams;
@group(0) @binding(1) var<storage, read>       voxels:     array<VoxelData>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn compute_voxel_acc(idx: u32) -> vec3<f32> {
    let probe = vec3<f32>(params.probe_x, params.probe_y, params.probe_z);
    let v     = voxels[idx];
    let p_i   = vec3<f32>(v.x, v.y, v.z);

    let diff    = probe - p_i;
    let dist_sq = dot(diff, diff) + 1.0;
    let inv_r3  = 1.0 / (dist_sq * sqrt(dist_sq));

    // a = -G * m / r^3 * diff  (points from voxel toward probe → attractive pull)
    return (-params.G_const * v.mass * inv_r3) * diff;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id:   vec3<u32>,
    @builtin(local_invocation_id)  local_id:    vec3<u32>,
    @builtin(workgroup_id)         workgroup_id: vec3<u32>,
) {
    let idx = global_id.x;
    let lid = local_id.x;

    var contrib = vec4<f32>(0.0);
    if idx < params.voxel_count {
        let a = compute_voxel_acc(idx);
        contrib = vec4<f32>(a, 0.0);
    }
    shared_acc[lid] = contrib;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if lid < stride {
            shared_acc[lid] += shared_acc[lid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if lid == 0u {
        output_acc[workgroup_id.x] = shared_acc[0];
    }
}
