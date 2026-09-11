// Discrete Eq.121 spatial force (IR residual + analytic monopole).
// Finite R^3 quadrature of
//   g(q) = G/(2π)³ ∫ (4π i κ / κ²) ρ̂(κ) e^{i κ·q} d³κ
// with ρ̂ from the uploaded density cells. This is the spatial force for
// propagation; Eq.184 remains the whole-trajectory Laplace observation.
struct Params {
    position: vec4<f32>,
    source_count: u32,
    mode_count: u32,
    mass: f32,
    gm: f32,
    center: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
// Each density cell occupies two records in the source buffer:
// [direction.xyz, solid_angle], [r_inner, r_outer, density, padding].
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;
var<workgroup> density_sum: array<vec2<f32>, 64>;
var<workgroup> field_sum: array<vec4<f32>, 64>;

const PI: f32 = 3.141592653589793;
const GAUSS_NODE_0: f32 = -0.8611363116;
const GAUSS_NODE_1: f32 = -0.3399810436;
const GAUSS_NODE_2: f32 = 0.3399810436;
const GAUSS_NODE_3: f32 = 0.8611363116;
const GAUSS_WEIGHT_0: f32 = 0.3478548451;
const GAUSS_WEIGHT_1: f32 = 0.6521451549;
const GAUSS_WEIGHT_2: f32 = 0.6521451549;
const GAUSS_WEIGHT_3: f32 = 0.3478548451;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cell_spectrum(k: vec3<f32>, cell_index: u32) -> vec2<f32> {
    let angular = sources[cell_index * 2u];
    let radial = sources[cell_index * 2u + 1u];
    let inner = radial.x;
    let outer = max(radial.y, inner);
    let density = max(radial.z, 0.0);
    let half_width = 0.5 * (outer - inner);
    let midpoint = 0.5 * (outer + inner);
    if half_width <= 0.0 || density <= 0.0 || angular.w <= 0.0 {
        return vec2<f32>(0.0);
    }
    var result = vec2<f32>(0.0);
    let nodes = array<f32, 4>(
        GAUSS_NODE_0, GAUSS_NODE_1, GAUSS_NODE_2, GAUSS_NODE_3
    );
    let weights = array<f32, 4>(
        GAUSS_WEIGHT_0, GAUSS_WEIGHT_1, GAUSS_WEIGHT_2, GAUSS_WEIGHT_3
    );
    for (var sample = 0u; sample < 4u; sample += 1u) {
        let radius = midpoint + half_width * nodes[sample];
        let volume_weight = density * angular.w * radius * radius * half_width * weights[sample];
        let phase = -dot(k, angular.xyz * radius);
        result += volume_weight * vec2<f32>(cos(phase), sin(phase));
    }
    return result;
}

@compute @workgroup_size(64)
fn density(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let k = nodes[group.x].xyz;
    var value = vec2<f32>(0.0);
    for (var index = lane; index < params.source_count; index += 64u) {
        value += cell_spectrum(k, index);
    }
    density_sum[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lane < stride { density_sum[lane] += density_sum[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u { spectrum[group.x] = density_sum[0]; }
}

@compute @workgroup_size(64)
fn field(@builtin(local_invocation_index) lane: u32) {
    // Discrete Eq.121: 64-node residual plus the analytic k→0 monopole of the
    // same integral. The IR term is GM r̂/r²; the nodes carry ρ̂(κ) − M e^{-iκ·cm}.
    var value = vec4<f32>(0.0);
    let query = params.position.xyz;
    let normalization = params.position.w * 4.0 * PI / ((2.0 * PI) * (2.0 * PI) * (2.0 * PI));
    let center = params.center.xyz;
    for (var index = lane; index < params.mode_count; index += 64u) {
        let node = nodes[index];
        let k = node.xyz;
        let weight = node.w;
        let k_squared = max(dot(k, k), 1.0e-20);
        let phase_cm = -dot(k, center);
        let residual = spectrum[index] - params.mass * vec2<f32>(cos(phase_cm), sin(phase_cm));
        let phase = dot(k, query);
        let rho_hat_times_phase = complex_mul(
            residual,
            vec2<f32>(cos(phase), sin(phase)),
        );
        let coefficient = normalization * weight / k_squared;
        let acceleration = -coefficient * rho_hat_times_phase.y * k;
        let potential = coefficient * rho_hat_times_phase.x;
        value = value + vec4<f32>(acceleration, potential);
    }
    field_sum[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lane < stride { field_sum[lane] += field_sum[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u {
        var result = field_sum[0];
        let offset = query - center;
        let distance_squared = max(dot(offset, offset), 1.0e-12);
        let inverse_distance = inverseSqrt(distance_squared);
        result += vec4<f32>(
            -params.gm * offset * (inverse_distance * inverse_distance * inverse_distance),
            params.gm * inverse_distance,
        );
        output[0] = result;
    }
}
