// GPU-built 56-basis FMM responses: one lane per density basis column.
struct Params {
    state_offset: u32,
    state_count: u32,
    response_start: u32,
    density_model: u32,
};

// Keep the 96-byte response record shared with planning_fmm_basis.wgsl.
struct LocalExpansion {
    center_half: vec4<f32>,
    field: vec4<f32>,
    jacobian_x: vec4<f32>,
    jacobian_y: vec4<f32>,
    jacobian_z: vec4<f32>,
    metadata: vec4<u32>,
};


@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> locals: array<LocalExpansion>;
@group(0) @binding(2) var<storage, read> densities: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> field_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_x_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_y_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_z_sum: array<vec4<f32>, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let target_index = group_id.x + group_id.y * 65535u;
    if target_index >= params.state_count { return; }
    let lane = local_id.x;
    let global_state = params.state_offset + target_index;
    field_sum[lane] = vec4(0.0);
    jacobian_x_sum[lane] = vec4(0.0);
    jacobian_y_sum[lane] = vec4(0.0);
    jacobian_z_sum[lane] = vec4(0.0);
    if lane < 56u {
        let local_index = (global_state - params.response_start) * 56u + lane;
        let basis = locals[local_index];
        let density = densities[params.density_model * 56u + lane];
        field_sum[lane] = basis.field * density;
        jacobian_x_sum[lane] = basis.jacobian_x * density;
        jacobian_y_sum[lane] = basis.jacobian_y * density;
        jacobian_z_sum[lane] = basis.jacobian_z * density;
    }
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if lane < stride {
            field_sum[lane] += field_sum[lane + stride];
            jacobian_x_sum[lane] += jacobian_x_sum[lane + stride];
            jacobian_y_sum[lane] += jacobian_y_sum[lane + stride];
            jacobian_z_sum[lane] += jacobian_z_sum[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if lane == 0u {
        let base = target_index * 4u;
        output[base] = field_sum[0];
        output[base + 1u] = jacobian_x_sum[0];
        output[base + 2u] = jacobian_y_sum[0];
        output[base + 3u] = jacobian_z_sum[0];
    }
}
