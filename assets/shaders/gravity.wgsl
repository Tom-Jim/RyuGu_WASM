// Spatial-domain radial analytic forward model.
// One invocation evaluates one angular-cell/radial-layer pair; workgroups
// reduce their 64 contributions before asynchronous CPU readback.

struct GravityParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    layer_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct RadialLayer {
    direction_solid_angle: vec4<f32>,
    radii_density: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: GravityParams;
@group(0) @binding(1) var<storage, read> layers: array<RadialLayer>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

// Eight-point Gauss-Legendre integration evaluates U and its analytic gradient
// at the same nodes. This avoids the f32 cancellation in the closed primitive
// and the even worse cancellation from finite-differencing two nearly equal
// potential values. The xyz result is therefore the gradient of the same
// discrete positive potential stored in w.
fn numerical_field_quadrature(inner: f32, outer: f32, source_dir: vec3<f32>, probe: vec3<f32>) -> vec4<f32> {
    let nodes = array<f32, 8>(
        -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
         0.1834346425,  0.5255324099,  0.7966664774,  0.9602898565
    );
    let weights = array<f32, 8>(
        0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
        0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
    );
    let midpoint = 0.5 * (inner + outer);
    let half_width = 0.5 * (outer - inner);
    var sum = vec4<f32>(0.0);
    for (var index = 0u; index < 8u; index = index + 1u) {
        let lambda = midpoint + half_width * nodes[index];
        let displacement = lambda * source_dir - probe;
        let distance2 = max(dot(displacement, displacement), 1.0e-8);
        let distance = sqrt(distance2);
        let mass_measure = weights[index] * lambda * lambda;
        sum += mass_measure * vec4<f32>(
            displacement / (distance2 * distance),
            1.0 / distance
        );
    }
    return half_width * sum;
}

fn layer_field(index: u32) -> vec4<f32> {
    let layer = layers[index];
    let source_dir = normalize(layer.direction_solid_angle.xyz);
    let solid_angle = layer.direction_solid_angle.w;
    let inner = layer.radii_density.x;
    let outer = layer.radii_density.y;
    let density = layer.radii_density.z;

    if outer <= inner || density <= 0.0 {
        return vec4<f32>(0.0);
    }
    return params.g_const * solid_angle * density
        * numerical_field_quadrature(inner, outer, source_dir, params.probe_pos);
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
    if index < params.layer_count {
        value = layer_field(index);
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
