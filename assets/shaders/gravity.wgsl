struct GravParams {
    probe_x:     f32,
    probe_y:     f32,
    probe_z:     f32,
    G_const:     f32,
    voxel_count: u32,
    stehfest_M:  u32,
    _pad0:       f32,
    _pad1:       f32,
    // WGSL strictly requires array elements to be 16-byte aligned. 
    // We pack 12 floats into 3 vec4s to prevent memory crashes!
    V:           array<vec4<f32>, 3>, 
}

struct VoxelData {
    x:    f32,
    y:    f32,
    z:    f32,
    mass: f32,
}

@group(0) @binding(0) var<uniform>             params:     GravParams;
@group(0) @binding(1) var<storage, read>       voxels:     array<VoxelData>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       lut:        array<vec2<f32>>; // S0 and S1

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn get_S_modes(z: f32) -> vec2<f32> {
    let max_z = 50.0;
    let n_samples = 4096.0;
    
    // Clamp to prevent out-of-bounds
    let idx_f = clamp((z / max_z) * (n_samples - 1.0), 0.0, n_samples - 1.0);
    
    let idx0 = u32(idx_f);
    let idx1 = min(idx0 + 1u, u32(n_samples - 1.0));
    let fract = idx_f - f32(idx0);
    
    let s0 = lut[idx0];
    let s1 = lut[idx1];
    
    return mix(s0, s1, fract);
}

fn compute_voxel_acc(idx: u32) -> vec3<f32> {
    let probe = vec3<f32>(params.probe_x, params.probe_y, params.probe_z);
    let v     = voxels[idx];
    let p_i   = vec3<f32>(v.x, v.y, v.z);
    
    let h_vec = probe - p_i;
    let h = length(h_vec);
    
    // No more artificial +1.0! Pure mathematical cutoff for singularity
    if h < 1e-6 {
        return vec3<f32>(0.0);
    }
    
    let ln2 = 0.69314718056;
    let r_i_len = length(p_i);
    var g_total = vec3<f32>(0.0);
    
    // Gaver-Stehfest inverse Laplace Transform Loop
    for (var k = 1u; k <= 12u; k = k + 1u) {
        let s_k = f32(k) * ln2 / h;
        
        // Extract 1D array index from packed vec4
        let array_idx = (k - 1u) / 4u;
        let vec_idx   = (k - 1u) % 4u;
        let v_k = params.V[array_idx][vec_idx];
        
        // Compute "as" for LUT sampling. (a is approximated as distance in this localized kernel)
        let a = h; 
        let as_val = a * s_k;
        
        let modes = get_S_modes(as_val);
        let S0 = modes.x;
        let S1 = modes.y;
        
        // Analytical Kernel formulation based on Struve-Neumann modes
        let unit_dir = h_vec / h;
        let mass_term = params.G_const * v.mass;
        
        // Derived from analytical decay mapping (simplified algebraic structure for execution)
        let s_domain_acc = -mass_term * ((S0 + S1) / (h * h)) * unit_dir;
        
        g_total += v_k * s_domain_acc;
    }
    
    return (ln2 / h) * g_total;
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