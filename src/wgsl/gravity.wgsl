// Independent radial-shell volume quadrature. No frequency-domain spectra or point residues.

struct GravityParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    source_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0) var<uniform> params: GravityParams;
struct RadialCell {
    direction_angle: vec4<f32>,
    radii_density: vec4<f32>,
};
@group(0) @binding(1) var<storage, read> sources: array<RadialCell>;
const NODES = array<f32, 8>(-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
    0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565);
const WEIGHTS = array<f32, 8>(0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
    0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363);
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn source_field(index: u32) -> vec4<f32> {
    let cell = sources[index];
    let inner = cell.radii_density.x;
    let outer = cell.radii_density.y;
    let half_width = 0.5 * (outer - inner);
    let midpoint = 0.5 * (outer + inner);
    var value = vec4<f32>(0.0);
    for (var node = 0u; node < 8u; node += 1u) {
        let radius = midpoint + half_width * NODES[node];
        let displacement = radius * cell.direction_angle.xyz - params.probe_pos;
        let distance2 = dot(displacement, displacement);
        // A true source collision is invalid, not an epsilon-softened field.
        let inverse_distance = inverseSqrt(distance2);
        let mass_weight = cell.radii_density.z * cell.direction_angle.w
            * radius * radius * half_width * WEIGHTS[node];
        value += params.g_const * mass_weight * vec4<f32>(
            displacement * (inverse_distance / distance2), inverse_distance);
    }
    return value;
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
