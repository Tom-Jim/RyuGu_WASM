// Spatial-domain forward model over the common mass-preserving point quadrature.

struct GravityParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    source_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0) var<uniform> params: GravityParams;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn source_field(index: u32) -> vec4<f32> {
    let source = sources[index];
    if source.w <= 0.0 {
        return vec4<f32>(0.0);
    }
    let displacement = source.xyz - params.probe_pos;
    let distance2 = max(dot(displacement, displacement), 1.0e-8);
    let inverse_distance = inverseSqrt(distance2);
    let inverse_distance3 = inverse_distance / distance2;
    return params.g_const * source.w * vec4<f32>(
        displacement * inverse_distance3,
        inverse_distance,
    );
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let index = global_id.x;
    let lane = local_id.x;
    var value = vec4<f32>(0.0);
    if index < params.source_count {
        value = source_field(index);
    }
    shared_acc[lane] = value;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if stride == 0u {
            break;
        }
        if lane < stride {
            shared_acc[lane] += shared_acc[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if lane == 0u {
        output_acc[workgroup_id.x] = shared_acc[0];
    }
}
