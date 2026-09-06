// Eq.106 -> Eq.123 -> analytic Bromwich residue at s = i k.e.
// Res[e^(s h)/(s-i k.e)] = e^(i k.e h), giving Eq.121 at q=q0+h e.
// The pole inversion is exact; rho_hat and the R^3 integral below are finite
// quadratures. This is the spatial force for propagation, NOT Eq.184's
// Laplace observation. No Radial kernel, history, or fallback is used.
struct Params {
    position: vec3<f32>,
    g: f32,
    source_count: u32,
    mode_count: u32,
    padding: vec2<u32>,
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

const GAUSS_NODE_0: f32 = -0.8611363116;
const GAUSS_NODE_1: f32 = -0.3399810436;
const GAUSS_NODE_2: f32 = 0.3399810436;
const GAUSS_NODE_3: f32 = 0.8611363116;
const GAUSS_WEIGHT_0: f32 = 0.3478548451;
const GAUSS_WEIGHT_1: f32 = 0.6521451549;
const GAUSS_WEIGHT_2: f32 = 0.6521451549;
const GAUSS_WEIGHT_3: f32 = 0.3478548451;

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

fn cell_field(position: vec3<f32>, cell_index: u32, g: f32) -> vec4<f32> {
    let angular = sources[cell_index * 2u];
    let radial = sources[cell_index * 2u + 1u];
    let inner = radial.x;
    let outer = max(radial.y, inner);
    let density = max(radial.z, 0.0);
    let half_width = 0.5 * (outer - inner);
    let midpoint = 0.5 * (outer + inner);
    if half_width <= 0.0 || density <= 0.0 || angular.w <= 0.0 {
        return vec4<f32>(0.0);
    }
    let nodes = array<f32, 4>(
        GAUSS_NODE_0, GAUSS_NODE_1, GAUSS_NODE_2, GAUSS_NODE_3
    );
    let weights = array<f32, 4>(
        GAUSS_WEIGHT_0, GAUSS_WEIGHT_1, GAUSS_WEIGHT_2, GAUSS_WEIGHT_3
    );
    var result = vec4<f32>(0.0);
    for (var sample = 0u; sample < 4u; sample += 1u) {
        let radius = midpoint + half_width * nodes[sample];
        let source_position = angular.xyz * radius;
        let displacement = source_position - position;
        let radius_squared = max(dot(displacement, displacement), 1.0e-8);
        let inverse_radius = inverseSqrt(radius_squared);
        let mass_weight = density * angular.w * radius * radius * half_width * weights[sample];
        let acceleration = g * mass_weight * displacement * inverse_radius / radius_squared;
        let potential = g * mass_weight * inverse_radius;
        // WGSL does not accept compound assignment to a vector swizzle on
        // all browser backends. Update the complete vector instead.
        result = result + vec4<f32>(acceleration, potential);
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
    var value = vec4<f32>(0.0);
    // This is the inverse-pole (spatial) value of Eq.106/121. Integrate each
    // continuous density cell in radius instead of collapsing it to a point
    // mass. The spectrum pass above remains available for the Eq.184 operator;
    // the orbit force uses this convergent volume residue to avoid low-k
    // truncation precession near the body.
    for (var index = lane; index < params.source_count; index += 64u) {
        value += cell_field(params.position, index, params.g);
    }
    field_sum[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lane < stride { field_sum[lane] += field_sum[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u { output[0] = field_sum[0]; }
}
