// Periodic spherical-ring MMFFT.
//
// Each workgroup owns one polar/radial ring of 64 azimuth bins.  It performs
// forward radix-2 FFTs of the ring mass and four target-dependent Newton
// kernels, multiplies in frequency space, and performs the matching IFFTs.

struct MmfftParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    record_count: u32,
    ring_count: u32,
    azimuth_bins: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

@group(0) @binding(0) var<uniform> params: MmfftParams;
@group(0) @binding(1) var<storage, read> records: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> spectrum_mass: array<vec2<f32>, 64>;
var<workgroup> spectrum_x: array<vec2<f32>, 64>;
var<workgroup> spectrum_y: array<vec2<f32>, 64>;
var<workgroup> spectrum_z: array<vec2<f32>, 64>;
var<workgroup> spectrum_p: array<vec2<f32>, 64>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn butterfly_stage(lane: u32, size: u32, inverse: bool) {
    if lane >= 32u { return; }
    let half = size >> 1u;
    let group = lane / half;
    let j = lane - group * half;
    let first = group * size + j;
    let second = first + half;
    let sign = select(-1.0, 1.0, inverse);
    let angle = sign * 6.283185307179586 * f32(j) / f32(size);
    let twiddle = vec2<f32>(cos(angle), sin(angle));

    let m0 = spectrum_mass[first]; let m1 = complex_mul(spectrum_mass[second], twiddle);
    let x0 = spectrum_x[first]; let x1 = complex_mul(spectrum_x[second], twiddle);
    let y0 = spectrum_y[first]; let y1 = complex_mul(spectrum_y[second], twiddle);
    let z0 = spectrum_z[first]; let z1 = complex_mul(spectrum_z[second], twiddle);
    let p0 = spectrum_p[first]; let p1 = complex_mul(spectrum_p[second], twiddle);
    spectrum_mass[first] = m0 + m1; spectrum_mass[second] = m0 - m1;
    spectrum_x[first] = x0 + x1; spectrum_x[second] = x0 - x1;
    spectrum_y[first] = y0 + y1; spectrum_y[second] = y0 - y1;
    spectrum_z[first] = z0 + z1; spectrum_z[second] = z0 - z1;
    spectrum_p[first] = p0 + p1; spectrum_p[second] = p0 - p1;
}

fn transform(lane: u32, inverse: bool) {
    var size = 2u;
    loop {
        butterfly_stage(lane, size, inverse);
        workgroupBarrier();
        if size == 64u { break; }
        size = size << 1u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lane = local_id.x;
    let ring = workgroup_id.x;
    if ring >= params.ring_count { return; }
    let record = records[ring * 64u + lane];
    let mass = max(record.x, 0.0);
    let radius = max(record.y, 0.0);
    let cosine = clamp(record.z, -1.0, 1.0);

    // For c[0] = sum_j mass[j] kernel[-j], lane k holds kernel[-k].
    let kernel_angle = -6.283185307179586 * f32(lane) / 64.0;
    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    let source = radius * vec3<f32>(sine * cos(kernel_angle), sine * sin(kernel_angle), cosine);
    let displacement = source - params.probe_pos;
    let distance2 = max(dot(displacement, displacement), 1.0e-8);
    let distance = sqrt(distance2);
    let kernel = vec4<f32>(displacement / (distance2 * distance), 1.0 / distance);

    // Decimation-in-time FFT with explicit six-bit bit reversal.
    let reversed = reverseBits(lane) >> 26u;
    spectrum_mass[reversed] = vec2<f32>(mass, 0.0);
    spectrum_x[reversed] = vec2<f32>(kernel.x, 0.0);
    spectrum_y[reversed] = vec2<f32>(kernel.y, 0.0);
    spectrum_z[reversed] = vec2<f32>(kernel.z, 0.0);
    spectrum_p[reversed] = vec2<f32>(kernel.w, 0.0);
    workgroupBarrier();

    transform(lane, false);
    if lane < 64u {
        spectrum_x[lane] = complex_mul(spectrum_mass[lane], spectrum_x[lane]);
        spectrum_y[lane] = complex_mul(spectrum_mass[lane], spectrum_y[lane]);
        spectrum_z[lane] = complex_mul(spectrum_mass[lane], spectrum_z[lane]);
        spectrum_p[lane] = complex_mul(spectrum_mass[lane], spectrum_p[lane]);
        // IFFT input also needs bit-reversal for the same DIT kernel.
    }
    workgroupBarrier();

    let x = spectrum_x[lane]; let y = spectrum_y[lane];
    let z = spectrum_z[lane]; let p = spectrum_p[lane];
    workgroupBarrier();
    spectrum_x[reversed] = x; spectrum_y[reversed] = y;
    spectrum_z[reversed] = z; spectrum_p[reversed] = p;
    spectrum_mass[reversed] = vec2<f32>(0.0);
    workgroupBarrier();
    transform(lane, true);

    if lane == 0u {
        output_acc[ring] = params.g_const * vec4<f32>(
            spectrum_x[0].x, spectrum_y[0].x, spectrum_z[0].x, spectrum_p[0].x
        ) / 64.0;
    }
}
