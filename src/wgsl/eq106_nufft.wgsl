// Fixed-band GPU Type-2 NUFFT precomputation for Equation (106).
//
// The 129 signed modes are zero-padded to a 1024-point periodic grid and
// transformed with an in-workgroup radix-2 inverse FFT. Arbitrary trajectory
// abscissae are evaluated by cubic interpolation in eq106_complex.wgsl. The
// 8x oversampling over the occupied 128-mode band keeps interpolation error
// small; the evaluator folds cubic-vs-linear disagreement into its existing
// spectral certificate.

struct Eq106Params {
    probe_pos: vec3<f32>,
    g_const: f32,
    line_origin: vec3<f32>,
    sigma: f32,
    line_direction: vec3<f32>,
    omega_step: f32,
    source_count: u32,
    half_count: u32,
    quadrature_count: u32,
    taylor_order: u32,
    density_mode_count: u32,
    segment_id: u32,
    evaluate_dual_certificate: u32,
    target_count: u32,
    line_limit: f32,
    target_offset: u32,
    inversion_mode: u32,
    input_base: u32,
};

struct SpectrumSample {
    acceleration_x: vec2<f32>,
    acceleration_y: vec2<f32>,
    acceleration_z: vec2<f32>,
    potential: vec2<f32>,
};

struct DensityModeBuffer {
    records: array<vec4<f32>, 544>,
};

@group(0) @binding(0) var<storage, read> segment_params: array<Eq106Params>;
// Keep this binding mode aligned with the shared planning layout. Other
// Eq.106 planning entry points populate `spectrum` through the same binding,
// so WebGPU requires the Type-2 read pass to declare it as read-write too.
@group(0) @binding(3) var<storage, read_write> spectrum: array<SpectrumSample>;
@group(0) @binding(6) var<storage, read_write> nufft_storage: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> density_modes: DensityModeBuffer;

const PI: f32 = 3.141592653589793;
const FFT_SIZE: u32 = 1024u;
const FFT_PAIR_COUNT: u32 = 6u;
const MAX_TAYLOR_COEFFICIENT_COUNT: u32 = 45u;
const SPECTRUM_FREQUENCY_CAPACITY: u32 = 129u;

// One vec4 packs two independent complex transforms and exactly occupies the
// portable WebGPU 16 KiB workgroup-storage floor.
var<workgroup> fft_values: array<vec4<f32>, 1024>;

fn complex_mul(left: vec2<f32>, right: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        left.x * right.x - left.y * right.y,
        left.x * right.y + left.y * right.x,
    );
}

fn bit_reverse_10(value: u32) -> u32 {
    var input = value;
    var result = 0u;
    for (var bit = 0u; bit < 10u; bit += 1u) {
        result = (result << 1u) | (input & 1u);
        input >>= 1u;
    }
    return result;
}

fn selected_modes(sample: SpectrumSample, pair: u32, sigma: f32, omega: f32) -> vec4<f32> {
    if pair == 0u {
        return vec4<f32>(sample.acceleration_x, sample.acceleration_y);
    }
    if pair == 1u {
        return vec4<f32>(sample.acceleration_z, sample.potential);
    }
    let s = vec2<f32>(sigma, omega);
    if pair == 2u {
        return vec4<f32>(
            complex_mul(sample.acceleration_x, s),
            complex_mul(sample.acceleration_y, s),
        );
    }
    if pair == 3u {
        return vec4<f32>(complex_mul(sample.acceleration_z, s), vec2<f32>(0.0));
    }
    if pair == 4u {
        return vec4<f32>(sample.acceleration_x, sample.acceleration_y);
    }
    return vec4<f32>(sample.acceleration_z, vec2<f32>(0.0));
}

@compute @workgroup_size(256, 1, 1)
fn build_type2_nufft_grid(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let coefficient = workgroup_id.x;
    let segment_index = workgroup_id.y;
    let pair = workgroup_id.z;
    let params = segment_params[segment_index];
    let coefficient_count = (params.taylor_order + 1u) * (params.taylor_order + 2u) / 2u;
    if coefficient >= coefficient_count || pair >= FFT_PAIR_COUNT {
        return;
    }
    let frequency_count = 2u * params.half_count + 1u;
    for (var bin = lane; bin < FFT_SIZE; bin += 256u) {
        var value = vec4<f32>(0.0);
        var signed_frequency = 0i;
        var occupied = false;
        if bin <= params.half_count {
            signed_frequency = i32(bin);
            occupied = true;
        } else if bin >= FFT_SIZE - params.half_count {
            signed_frequency = i32(bin) - i32(FFT_SIZE);
            occupied = true;
        }
        if occupied {
            let frequency_index = u32(signed_frequency + i32(params.half_count));
            let sample = spectrum[
                (params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT
                    * SPECTRUM_FREQUENCY_CAPACITY
                + coefficient * SPECTRUM_FREQUENCY_CAPACITY
                + frequency_index
            ];
            value = selected_modes(
                sample,
                pair,
                params.sigma,
                f32(signed_frequency) * params.omega_step,
            );
            if pair >= 4u && frequency_index % 2u != 0u {
                value = vec4<f32>(0.0);
            }
        }
        fft_values[bit_reverse_10(bin)] = value;
    }
    workgroupBarrier();

    var span = 2u;
    loop {
        if span > FFT_SIZE { break; }
        let half_span = span / 2u;
        for (var butterfly = lane; butterfly < FFT_SIZE / 2u; butterfly += 256u) {
            let group = butterfly / half_span;
            let offset = butterfly % half_span;
            let lower_index = group * span + offset;
            let upper_index = lower_index + half_span;
            let angle = 2.0 * PI * f32(offset) / f32(span);
            let twiddle = vec2<f32>(cos(angle), sin(angle));
            let lower = fft_values[lower_index];
            let upper = fft_values[upper_index];
            let rotated = vec4<f32>(
                complex_mul(upper.xy, twiddle),
                complex_mul(upper.zw, twiddle),
            );
            fft_values[lower_index] = lower + rotated;
            fft_values[upper_index] = lower - rotated;
        }
        workgroupBarrier();
        span <<= 1u;
    }

    let grid_base = u32(density_modes.records[112].z)
        + ((params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT
            * FFT_PAIR_COUNT + coefficient * FFT_PAIR_COUNT + pair) * FFT_SIZE;
    for (var grid_index = lane; grid_index < FFT_SIZE; grid_index += 256u) {
        // WGSL's inverse transform is unnormalised, so restore the Fourier
        // sum scale before interpolation.
        nufft_storage[grid_base + grid_index] = fft_values[grid_index] / f32(FFT_SIZE);
    }
}
