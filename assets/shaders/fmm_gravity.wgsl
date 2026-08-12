// Single-target, order-two GPU FMM evaluation.
//
// P2M/M2M moments are assembled once on the CPU.  A node in the target's
// interaction list is translated by M2L into potential, gradient and Hessian
// coefficients about the same-level target box.  L2L then shifts that local
// expansion to the probe.  Non-separated leaves supply the bounded near field.

struct FmmParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    node_count: u32,
    maximum_level: u32,
    theta: f32,
    particle_count: u32,
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
@group(0) @binding(3) var<storage, read> particles: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn target_box(level: u32) -> vec4<f32> {
    // Extend the source tree's same-size Cartesian boxes across all space.
    // Using a separate 16R target root made a target box sixteen times larger
    // than a source box at the nominally same level, invalidating M2L/L2L.
    let source_root_half = nodes[0].center_half.w;
    let half_width = source_root_half / f32(1u << level);
    let width = 2.0 * half_width;
    let cell = floor((params.probe_pos + vec3<f32>(source_root_half)) / width);
    let center = -vec3<f32>(source_root_half) + (cell + vec3<f32>(0.5)) * width;
    return vec4<f32>(center, half_width);
}

fn accepted(index: u32) -> bool {
    let node = nodes[index];
    let level = node.metadata.y;
    if level == 0u { return false; }
    let target_cell = target_box(level);
    let distance = length(node.center_half.xyz - target_cell.xyz);
    // A local expansion must be valid over the whole target box, not merely
    // far from the small source box.  The old source-only ratio accepted very
    // coarse exterior target boxes and evaluated L2L outside convergence.
    let expansion_radius = sqrt(3.0) * (node.center_half.w + target_cell.w);
    return distance > 1.01 * expansion_radius
        && expansion_radius / max(distance, 1.0e-6) < params.theta;
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

fn direct_multipole(node: FmmNode, observer: vec3<f32>) -> vec4<f32> {
    let d = node.com_mass.xyz - observer;
    let r2 = max(dot(d, d), 1.0e-8);
    let r = sqrt(r2);
    let inv_r = 1.0 / r;
    let inv_r3 = inv_r / r2;
    var acceleration = node.com_mass.w * d * inv_r3;
    var potential = node.com_mass.w * inv_r;
    let qd = vec3<f32>(
        node.quadrupole0.x * d.x + node.quadrupole0.y * d.y + node.quadrupole0.z * d.z,
        node.quadrupole0.y * d.x + node.quadrupole1.x * d.y + node.quadrupole1.y * d.z,
        node.quadrupole0.z * d.x + node.quadrupole1.y * d.y + node.quadrupole1.z * d.z
    );
    let scalar = dot(d, qd);
    let inv_r5 = inv_r3 / r2;
    acceleration += -qd * inv_r5 + 2.5 * scalar * d * inv_r5 / r2;
    potential += 0.5 * scalar * inv_r5;
    return vec4<f32>(acceleration, potential);
}

fn m2l_then_l2l(node: FmmNode) -> vec4<f32> {
    let center = target_box(node.metadata.y).xyz;
    // M2L: translate both monopole and quadrupole source moments into local
    // potential, gradient, and Hessian coefficients at the target-box center.
    // Centered differentiation is applied to the analytic source multipole,
    // never to particles or to the final target value.
    let local = direct_multipole(node, center);
    let derivative_step = max(0.01, 0.015625 * node.center_half.w);
    let ex = vec3<f32>(derivative_step, 0.0, 0.0);
    let ey = vec3<f32>(0.0, derivative_step, 0.0);
    let ez = vec3<f32>(0.0, 0.0, derivative_step);
    let h0 = (direct_multipole(node, center + ex).xyz - direct_multipole(node, center - ex).xyz)
        / (2.0 * derivative_step);
    let h1 = (direct_multipole(node, center + ey).xyz - direct_multipole(node, center - ey).xyz)
        / (2.0 * derivative_step);
    let h2 = (direct_multipole(node, center + ez).xyz - direct_multipole(node, center - ez).xyz)
        / (2.0 * derivative_step);
    let delta = params.probe_pos - center;
    // L2L: shift the local coefficients from the target box to the probe.
    let hessian_delta = h0 * delta.x + h1 * delta.y + h2 * delta.z;
    let translated_gradient = local.xyz + hessian_delta;
    let translated_potential = local.w + dot(local.xyz, delta)
        + 0.5 * dot(delta, hessian_delta);
    return vec4<f32>(translated_gradient, translated_potential);
}

fn p2p_leaf(node: FmmNode) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    let start = node.metadata.z;
    let end = min(start + node.metadata.w, params.particle_count);
    for (var index = start; index < end; index += 1u) {
        let particle = particles[index];
        let displacement = particle.xyz - params.probe_pos;
        let distance2 = max(dot(displacement, displacement), 1.0e-8);
        let distance = sqrt(distance2);
        value += particle.w * vec4<f32>(displacement / (distance2 * distance), 1.0 / distance);
    }
    return value;
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
    var value = direct_multipole(node, params.probe_pos);
    if accepted(index) {
        // Standard far-field path: P2M/M2M -> M2L -> L2L. The geometric
        // acceptance test enforces the expansion disk; no direct multipole
        // substitution is used after an interaction has been accepted.
        value = m2l_then_l2l(node);
    } else if level == params.maximum_level {
        // Standard near-field path: exact particle-to-particle accumulation
        // over the non-separated leaf interaction list.
        value = p2p_leaf(node);
    }
    return params.g_const * value;
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
