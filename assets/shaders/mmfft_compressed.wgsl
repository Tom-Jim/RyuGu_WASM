// MMFFT compressed-source forward field.
//
// The source records are packed to 16 bytes: three signed-normalized direction
// components, a normalized solid angle, two normalized radii, and normalized
// density. A workgroup decodes one tile and performs the same eight-node
// moment quadrature used by the radial reference path. The tile metadata and
// packed records are the extension point for a future hierarchical FFT pass.

struct MmfftParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    record_count: u32,
    solid_angle_scale: f32,
    radius_scale: f32,
    density_scale: f32,
    tile_size: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0) var<uniform> params: MmfftParams;
@group(0) @binding(1) var<storage, read> records: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn unpack_i16(word: u32, high: bool) -> i32 {
    let shift = select(0u, 16u, high);
    let raw = i32((word >> shift) & 0xffffu);
    return select(raw, raw - 65536, raw >= 32768);
}

fn unpack_u16(word: u32, high: bool) -> f32 {
    let shift = select(0u, 16u, high);
    return f32((word >> shift) & 0xffffu) / 65535.0;
}

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

fn record_field(index: u32) -> vec4<f32> {
    let packed = records[index];
    let direction = normalize(vec3<f32>(
        f32(unpack_i16(packed.x, false)),
        f32(unpack_i16(packed.x, true)),
        f32(unpack_i16(packed.y, false))
    ) / 32767.0);
    let solid_angle = unpack_u16(packed.y, true) * params.solid_angle_scale;
    let inner = unpack_u16(packed.z, false) * params.radius_scale;
    let outer = unpack_u16(packed.z, true) * params.radius_scale;
    let density = unpack_u16(packed.w, false) * params.density_scale;
    if outer <= inner || density <= 0.0 || solid_angle <= 0.0 {
        return vec4<f32>(0.0);
    }
    return params.g_const * solid_angle * density
        * numerical_field_quadrature(inner, outer, direction, params.probe_pos);
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
    if index < params.record_count {
        value = record_field(index);
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
