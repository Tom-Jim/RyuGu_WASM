// Batched order-two target-cell FMM.
//
// CPU P2M/M2M builds source moments once per density row.  Far source boxes
// are translated by M2L into one potential/gradient/Hessian expansion for
// every occupied target leaf.  This shader reuses that expansion for all
// trajectory states in the leaf (L2P) and accumulates only its P2P near list.

struct Params {
    state_offset: u32,
    state_count: u32,
    local_count: u32,
    particle_word_offset: u32,
    near_particle_count: u32,
    density_model: u32,
    samples_per_candidate: u32,
    _padding0: u32,
    g_const: f32,
    _theta: f32,
    _derivative_step: f32,
    _padding1: f32,
};

struct LocalExpansion {
    center_half: vec4<f32>,
    field: vec4<f32>,
    jacobian_x: vec4<f32>,
    jacobian_y: vec4<f32>,
    jacobian_z: vec4<f32>,
    metadata: vec4<u32>,
};

struct FieldResult {
    field: vec4<f32>,
    jacobian_x: vec4<f32>,
    jacobian_y: vec4<f32>,
    jacobian_z: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> locals: array<LocalExpansion>;
@group(0) @binding(2) var<storage, read> packed: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> field_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_x_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_y_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_z_sum: array<vec4<f32>, 64>;

fn zero_result() -> FieldResult {
    var result: FieldResult;
    result.field = vec4<f32>(0.0);
    result.jacobian_x = vec4<f32>(0.0);
    result.jacobian_y = vec4<f32>(0.0);
    result.jacobian_z = vec4<f32>(0.0);
    return result;
}

fn packed_particle(index: u32) -> vec4<f32> {
    let base = params.particle_word_offset + 4u * index;
    return vec4<f32>(
        bitcast<f32>(packed[base]),
        bitcast<f32>(packed[base + 1u]),
        bitcast<f32>(packed[base + 2u]),
        bitcast<f32>(packed[base + 3u]),
    );
}

fn p2p_result(particle: vec4<f32>, observer: vec3<f32>) -> FieldResult {
    let displacement = particle.xyz - observer;
    let radius2 = max(dot(displacement, displacement), 1.0e-8);
    let inverse_radius = inverseSqrt(radius2);
    let inverse_radius3 = inverse_radius / radius2;
    let inverse_radius5 = inverse_radius3 / radius2;
    let diagonal = -particle.w * inverse_radius3;
    let outer_scale = 3.0 * particle.w * inverse_radius5;
    var result: FieldResult;
    result.field = params.g_const * particle.w
        * vec4<f32>(displacement * inverse_radius3, inverse_radius);
    result.jacobian_x = params.g_const * vec4<f32>(
        vec3<f32>(diagonal, 0.0, 0.0) + outer_scale * displacement * displacement.x,
        0.0,
    );
    result.jacobian_y = params.g_const * vec4<f32>(
        vec3<f32>(0.0, diagonal, 0.0) + outer_scale * displacement * displacement.y,
        0.0,
    );
    result.jacobian_z = params.g_const * vec4<f32>(
        vec3<f32>(0.0, 0.0, diagonal) + outer_scale * displacement * displacement.z,
        0.0,
    );
    return result;
}

fn l2p_result(local: LocalExpansion, observer: vec3<f32>) -> FieldResult {
    let delta = observer - local.center_half.xyz;
    let jacobian_delta = local.jacobian_x.xyz * delta.x
        + local.jacobian_y.xyz * delta.y
        + local.jacobian_z.xyz * delta.z;
    var result: FieldResult;
    result.field = vec4<f32>(
        local.field.xyz + jacobian_delta,
        local.field.w + dot(local.field.xyz, delta) + 0.5 * dot(delta, jacobian_delta),
    );
    result.jacobian_x = local.jacobian_x;
    result.jacobian_y = local.jacobian_y;
    result.jacobian_z = local.jacobian_z;
    return result;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if group_id.x >= params.state_count { return; }
    let lane = local_id.x;
    let global_state = params.state_offset + group_id.x;
    let observer = positions[global_state].xyz;
    let local_index = packed[global_state];
    // A storage-buffer read is not considered dynamically uniform by every
    // WebGPU backend.  Never return on `local_index` before a workgroup
    // barrier: all 64 invocations must reach every barrier even when an input
    // mapping is corrupt.  The payload always contains at least one sentinel
    // local expansion, so index zero is a safe fallback.
    let valid_local = local_index < params.local_count;
    let safe_local_index = select(0u, local_index, valid_local);
    let local = locals[safe_local_index];
    var accumulated = zero_result();
    if lane == 0u && valid_local {
        accumulated = l2p_result(local, observer);
    }
    let near_start = select(0u, local.metadata.x, valid_local);
    let near_count = select(0u, local.metadata.y, valid_local);
    let near_end = min(near_start + near_count, params.near_particle_count);
    for (var particle_index = near_start + lane;
        particle_index < near_end;
        particle_index += 64u) {
        let near = p2p_result(packed_particle(particle_index), observer);
        accumulated.field += near.field;
        accumulated.jacobian_x += near.jacobian_x;
        accumulated.jacobian_y += near.jacobian_y;
        accumulated.jacobian_z += near.jacobian_z;
    }
    field_sum[lane] = accumulated.field;
    jacobian_x_sum[lane] = accumulated.jacobian_x;
    jacobian_y_sum[lane] = accumulated.jacobian_y;
    jacobian_z_sum[lane] = accumulated.jacobian_z;
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
        let base = group_id.x * 4u;
        if valid_local {
            let valid = select(0.0, 1.0, densities[params.density_model * 56u] > 0.0);
            output[base] = field_sum[0] * valid;
            output[base + 1u] = jacobian_x_sum[0] * valid;
            output[base + 2u] = jacobian_y_sum[0] * valid;
            output[base + 3u] = jacobian_z_sum[0] * valid;
        } else {
            // Tint rejects a constant-expression bitcast whose result is NaN.
            // Keep the diagnostic sentinel as a runtime value by mixing in
            // the invalid mapping index; IEEE-754 still classifies every
            // resulting payload as a quiet NaN.
            let nan_bits = 0x7fc00000u | (local_index & 0x003fffffu);
            let nan = bitcast<f32>(nan_bits);
            output[base] = vec4<f32>(nan);
            output[base + 1u] = vec4<f32>(nan);
            output[base + 2u] = vec4<f32>(nan);
            output[base + 3u] = vec4<f32>(nan);
        }
    }
}
