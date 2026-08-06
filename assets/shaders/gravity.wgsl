// Spatial-domain radial analytic forward model (equation 18).
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

fn asinh_safe(x: f32) -> f32 {
    let ax = abs(x);
    let value = log(ax + sqrt(ax * ax + 1.0));
    if x < 0.0 {
        return -value;
    }
    return value;
}

// K(lambda) = J3(lambda) n' - R J2(lambda) n.
fn radial_primitive(lambda: f32, radius: f32, cosine: f32, source_dir: vec3<f32>, probe_dir: vec3<f32>) -> vec3<f32> {
    let a = radius * cosine;
    let u = lambda - a;
    let b2 = max(radius * radius * (1.0 - cosine * cosine), 1.0e-12);
    let b = sqrt(b2);
    let distance = sqrt(u * u + b2);
    let hyperbolic = asinh_safe(u / b);

    let j2 = hyperbolic - u / distance - 2.0 * a / distance
        + a * a * u / (b2 * distance);
    let j3 = distance + b2 / distance
        + 3.0 * a * (hyperbolic - u / distance)
        - 3.0 * a * a / distance
        + a * a * a * u / (b2 * distance);
    return j3 * source_dir - radius * j2 * probe_dir;
}

// Collinear and near-collinear directions make the closed form ill-conditioned
// even though the integral is finite away from the surface. The derivation
// explicitly permits a numerical fallback, so use 8-point Gauss-Legendre here.
fn collinear_quadrature(inner: f32, outer: f32, source_dir: vec3<f32>, probe: vec3<f32>) -> vec3<f32> {
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
    var sum = vec3<f32>(0.0);
    for (var index = 0u; index < 8u; index = index + 1u) {
        let lambda = midpoint + half_width * nodes[index];
        let displacement = lambda * source_dir - probe;
        let distance2 = max(dot(displacement, displacement), 1.0e-8);
        sum += weights[index] * lambda * lambda * displacement
            / (distance2 * sqrt(distance2));
    }
    return half_width * sum;
}

fn layer_acceleration(index: u32) -> vec3<f32> {
    let layer = layers[index];
    let source_dir = normalize(layer.direction_solid_angle.xyz);
    let solid_angle = layer.direction_solid_angle.w;
    let inner = layer.radii_density.x;
    let outer = layer.radii_density.y;
    let density = layer.radii_density.z;

    let probe = params.probe_pos;
    let radius = length(probe);
    if radius < 1.0e-6 || outer <= inner || density <= 0.0 {
        return vec3<f32>(0.0);
    }
    let probe_dir = probe / radius;
    let cosine = clamp(dot(probe_dir, source_dir), -1.0, 1.0);
    let sine2 = max(0.0, 1.0 - cosine * cosine);

    var radial_integral: vec3<f32>;
    if sine2 < 1.0e-4 {
        radial_integral = collinear_quadrature(inner, outer, source_dir, probe);
    } else {
        radial_integral =
            radial_primitive(outer, radius, cosine, source_dir, probe_dir)
            - radial_primitive(inner, radius, cosine, source_dir, probe_dir);
    }
    return params.g_const * solid_angle * density * radial_integral;
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
        value = vec4<f32>(layer_acceleration(index), 0.0);
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
