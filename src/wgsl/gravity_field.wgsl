/// Method-matched exterior gravity field for discrete arrow glyphs.
///
/// mode 0 — inverted-voxel N-body only (Section off overlay; not FD)
/// mode 1 — legacy packed Eq.(121) residual + IR monopole (unused for live
///          Frequency-domain glyphs; those use Worker FLUPS instead)
/// Worker radial/Werner/FMM/FFT/FD skip this shader and use the numerical Worker.

struct GravityFieldParams {
    // x = sample count, y = source/mode count, z = mode (0 n-body, 1 eq106/121), w unused.
    counts: vec4<f32>,
    // Eq.106/121 trailer: xyz = mass centroid, w = GM. Unused for N-body.
    center_gm: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> params: GravityFieldParams;
@group(0) @binding(1)
var<storage, read> sample_positions: array<vec4<f32>>;
/// N-body: (x, y, z, mass). Eq.121: pairs (kx, ky, kz, re) + (im, 0, 0, 0).
@group(0) @binding(2)
var<storage, read> field_sources: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read_write> sample_fields: array<vec4<f32>>;

const G: f32 = 6.6743e-11;

fn nbody_acceleration(position: vec3<f32>, source_count: u32) -> vec3<f32> {
    var gravity = vec3<f32>(0.0);
    for (var index = 0u; index < source_count; index++) {
        let source = field_sources[index];
        let offset = position - source.xyz;
        let distance_squared = max(dot(offset, offset), 1.0e-4);
        let inv_distance = inverseSqrt(distance_squared);
        let inv3 = inv_distance * inv_distance * inv_distance;
        gravity += -G * source.w * offset * inv3;
    }
    return gravity;
}

fn equation121_acceleration(position: vec3<f32>, mode_count: u32) -> vec3<f32> {
    var gravity = vec3<f32>(0.0);
    for (var index = 0u; index < mode_count; index++) {
        let packed = field_sources[index * 2u];
        let imag_coeff = field_sources[index * 2u + 1u].x;
        let k = packed.xyz;
        let re_coeff = packed.w;
        let phase = dot(k, position);
        let sin_phase = sin(phase);
        let cos_phase = cos(phase);
        // Matches CPU: im = re*sin + im*cos; gravity -= im * k.
        let imag_part = re_coeff * sin_phase + imag_coeff * cos_phase;
        gravity -= imag_part * k;
    }
    let offset = position - params.center_gm.xyz;
    let distance_squared = max(dot(offset, offset), 1.0e-12);
    let inv_distance = inverseSqrt(distance_squared);
    let inv3 = inv_distance * inv_distance * inv_distance;
    gravity += -params.center_gm.w * offset * inv3;
    return gravity;
}

@compute @workgroup_size(64)
fn evaluate_gravity_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let sample_count = u32(params.counts.x);
    if index >= sample_count {
        return;
    }
    let position = sample_positions[index].xyz;
    let source_count = u32(params.counts.y);
    let mode = u32(params.counts.z);
    var acceleration = vec3<f32>(0.0);
    if mode == 1u {
        acceleration = equation121_acceleration(position, source_count);
    } else {
        acceleration = nbody_acceleration(position, source_count);
    }
    sample_fields[index] = vec4<f32>(acceleration, length(acceleration));
}
