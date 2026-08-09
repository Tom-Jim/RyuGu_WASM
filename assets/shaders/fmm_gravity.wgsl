// Single-target GPU FMM evaluation.
//
// P2M/M2M moments are assembled once at initialization. Each invocation owns
// one octree node, applies the fixed-depth multipole acceptance criterion, and
// emits either a quadrupole far-field contribution or a finest-level near
// contribution. Parent acceptance suppresses descendants, so every branch of
// the tree contributes exactly once without a recursive GPU traversal.

struct FmmParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    node_count: u32,
    maximum_level: u32,
    theta: f32,
    _padding0: u32,
};

struct FmmNode {
    center_half: vec4<f32>,
    com_mass: vec4<f32>,
    quadrupole0: vec4<f32>,
    quadrupole1: vec4<f32>,
    metadata: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: FmmParams;
@group(0) @binding(1) var<storage, read> nodes: array<FmmNode>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn accepted(index: u32) -> bool {
    let node = nodes[index];
    let distance = length(node.com_mass.xyz - params.probe_pos);
    return distance > node.center_half.w * 1.01
        && node.center_half.w / max(distance, 1.0e-6) < params.theta;
}

fn ancestor_is_accepted(index: u32) -> bool {
    var parent = nodes[index].metadata.x;
    for (var depth = 0u; depth < 8u; depth += 1u) {
        if parent == 0xffffffffu {
            return false;
        }
        if accepted(parent) {
            return true;
        }
        parent = nodes[parent].metadata.x;
    }
    return false;
}

fn node_field(index: u32) -> vec4<f32> {
    let node = nodes[index];
    let mass = node.com_mass.w;
    if mass <= 0.0 || ancestor_is_accepted(index) {
        return vec4<f32>(0.0);
    }
    let level = node.metadata.y;
    if !accepted(index) && level < params.maximum_level {
        return vec4<f32>(0.0);
    }
    let d = node.com_mass.xyz - params.probe_pos;
    let r2 = max(dot(d, d), 1.0e-8);
    let r = sqrt(r2);
    let inv_r = 1.0 / r;
    let inv_r2 = 1.0 / r2;
    let inv_r3 = inv_r * inv_r2;
    var acceleration = mass * d * inv_r3;
    var potential = mass * inv_r;

    let qd = vec3<f32>(
        node.quadrupole0.x * d.x + node.quadrupole0.y * d.y + node.quadrupole0.z * d.z,
        node.quadrupole0.y * d.x + node.quadrupole1.x * d.y + node.quadrupole1.y * d.z,
        node.quadrupole0.z * d.x + node.quadrupole1.y * d.y + node.quadrupole1.z * d.z
    );
    let scalar = dot(d, qd);
    let inv_r5 = inv_r3 * inv_r2;
    let inv_r7 = inv_r5 * inv_r2;
    acceleration += -qd * inv_r5 + 2.5 * scalar * d * inv_r7;
    potential += 0.5 * scalar * inv_r5;
    return params.g_const * vec4<f32>(acceleration, potential);
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
    if index < params.node_count {
        value = node_field(index);
    }
    shared_acc[lane] = value;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
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
