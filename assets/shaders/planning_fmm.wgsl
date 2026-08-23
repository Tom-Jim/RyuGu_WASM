struct Params {
    state_offset: u32,
    state_count: u32,
    node_count: u32,
    maximum_level: u32,
    particle_count: u32,
    density_model: u32,
    samples_per_candidate: u32,
    _padding0: u32,
    g_const: f32,
    theta: f32,
    _derivative_step: f32,
    _padding1: f32,
};

struct FmmNode {
    center_half: vec4<f32>,
    com_mass: vec4<f32>,
    quadrupole0: vec4<f32>,
    quadrupole1: vec4<f32>,
    metadata: vec4<u32>,
};

struct FieldResult {
    field: vec4<f32>,
    jacobian_x: vec4<f32>,
    jacobian_y: vec4<f32>,
    jacobian_z: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<FmmNode>;
@group(0) @binding(2) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> observer_boxes: array<vec4<f32>, 9>;
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

fn accepted(index: u32) -> bool {
    let node = nodes[index];
    let level = node.metadata.y;
    if level == 0u { return false; }
    let observer_box = observer_boxes[min(level, 8u)];
    let distance = length(node.center_half.xyz - observer_box.xyz);
    let radius = 1.7320508075688772 * (node.center_half.w + observer_box.w);
    return distance > 1.01 * radius && radius / max(distance, 1.0e-6) < params.theta;
}

fn ancestor_is_accepted(index: u32) -> bool {
    var parent = nodes[index].metadata.x;
    for (var depth = 0u; depth < 8u; depth += 1u) {
        if parent == 0xffffffffu { return false; }
        if accepted(parent) { return true; }
        parent = nodes[parent].metadata.x;
    }
    return false;
}

fn multipole_result(node: FmmNode, observer: vec3<f32>) -> FieldResult {
    let displacement = node.com_mass.xyz - observer;
    let radius2 = max(dot(displacement, displacement), 1.0e-8);
    let inverse_radius = inverseSqrt(radius2);
    let inverse_radius3 = inverse_radius / radius2;
    let inverse_radius5 = inverse_radius3 / radius2;
    let inverse_radius7 = inverse_radius5 / radius2;
    let inverse_radius9 = inverse_radius7 / radius2;
    let qx = vec3<f32>(node.quadrupole0.x, node.quadrupole0.y, node.quadrupole0.z);
    let qy = vec3<f32>(node.quadrupole0.y, node.quadrupole1.x, node.quadrupole1.y);
    let qz = vec3<f32>(node.quadrupole0.z, node.quadrupole1.y, node.quadrupole1.z);
    let qd = qx * displacement.x + qy * displacement.y + qz * displacement.z;
    let scalar = dot(displacement, qd);
    let acceleration = node.com_mass.w * displacement * inverse_radius3
        - qd * inverse_radius5
        + 2.5 * scalar * displacement * inverse_radius7;
    let potential = node.com_mass.w * inverse_radius + 0.5 * scalar * inverse_radius5;
    let diagonal = -node.com_mass.w * inverse_radius3 - 2.5 * scalar * inverse_radius7;
    let outer_scale = 3.0 * node.com_mass.w * inverse_radius5
        + 17.5 * scalar * inverse_radius9;
    let mixed_scale = -5.0 * inverse_radius7;
    var result: FieldResult;
    result.field = vec4<f32>(acceleration, potential);
    result.jacobian_x = vec4<f32>(
        vec3<f32>(diagonal, 0.0, 0.0)
            + outer_scale * displacement * displacement.x
            + inverse_radius5 * qx
            + mixed_scale * (qd * displacement.x + displacement * qd.x),
        0.0,
    );
    result.jacobian_y = vec4<f32>(
        vec3<f32>(0.0, diagonal, 0.0)
            + outer_scale * displacement * displacement.y
            + inverse_radius5 * qy
            + mixed_scale * (qd * displacement.y + displacement * qd.y),
        0.0,
    );
    result.jacobian_z = vec4<f32>(
        vec3<f32>(0.0, 0.0, diagonal)
            + outer_scale * displacement * displacement.z
            + inverse_radius5 * qz
            + mixed_scale * (qd * displacement.z + displacement * qd.z),
        0.0,
    );
    return result;
}

fn leaf_result(node: FmmNode, observer: vec3<f32>) -> FieldResult {
    var result = zero_result();
    let end = min(node.metadata.z + node.metadata.w, params.particle_count);
    for (var index = node.metadata.z; index < end; index += 1u) {
        let particle = particles[index];
        let displacement = particle.xyz - observer;
        let radius2 = max(dot(displacement, displacement), 1.0e-8);
        let inverse_radius = inverseSqrt(radius2);
        let inverse_radius3 = inverse_radius / radius2;
        let inverse_radius5 = inverse_radius3 / radius2;
        let diagonal = -particle.w * inverse_radius3;
        let outer_scale = 3.0 * particle.w * inverse_radius5;
        result.field += particle.w * vec4<f32>(displacement * inverse_radius3, inverse_radius);
        result.jacobian_x += vec4<f32>(
            vec3<f32>(diagonal, 0.0, 0.0) + outer_scale * displacement * displacement.x,
            0.0,
        );
        result.jacobian_y += vec4<f32>(
            vec3<f32>(0.0, diagonal, 0.0) + outer_scale * displacement * displacement.y,
            0.0,
        );
        result.jacobian_z += vec4<f32>(
            vec3<f32>(0.0, 0.0, diagonal) + outer_scale * displacement * displacement.z,
            0.0,
        );
    }
    return result;
}

fn node_result(index: u32, observer: vec3<f32>) -> FieldResult {
    let node = nodes[index];
    if node.com_mass.w <= 0.0 || ancestor_is_accepted(index) { return zero_result(); }
    let is_accepted = accepted(index);
    if !is_accepted && node.metadata.y < params.maximum_level { return zero_result(); }
    var result = zero_result();
    if is_accepted {
        result = multipole_result(node, observer);
    } else {
        result = leaf_result(node, observer);
    }
    result.field *= params.g_const;
    result.jacobian_x *= params.g_const;
    result.jacobian_y *= params.g_const;
    result.jacobian_z *= params.g_const;
    return result;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let observer_index = group_id.x;
    if observer_index >= params.state_count { return; }
    let lane = local_id.x;
    let observer_position = positions[params.state_offset + observer_index].xyz;
    if lane == 0u {
        let root_half = nodes[0].center_half.w;
        for (var level = 0u; level <= min(params.maximum_level, 8u); level += 1u) {
            let half = root_half / f32(1u << level);
            let width = 2.0 * half;
            let cell = floor((observer_position + vec3<f32>(root_half)) / width);
            observer_boxes[level] = vec4<f32>(
                -vec3<f32>(root_half) + (cell + vec3<f32>(0.5)) * width,
                half,
            );
        }
    }
    workgroupBarrier();
    var accumulated = zero_result();
    for (var node_index = lane; node_index < params.node_count; node_index += 64u) {
        let value = node_result(node_index, observer_position);
        accumulated.field += value.field;
        accumulated.jacobian_x += value.jacobian_x;
        accumulated.jacobian_y += value.jacobian_y;
        accumulated.jacobian_z += value.jacobian_z;
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
        let valid = select(0.0, 1.0, densities[params.density_model * 56u] > 0.0);
        let base = observer_index * 4u;
        output[base] = field_sum[0] * valid;
        output[base + 1u] = jacobian_x_sum[0] * valid;
        output[base + 2u] = jacobian_y_sum[0] * valid;
        output[base + 3u] = jacobian_z_sum[0] * valid;
    }
}
