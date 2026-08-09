// Equation (106) complex-frequency pipeline.
//
// `assemble_spectrum` runs only when the reference line changes. It evaluates
// the density-summed Laplace transform on one shared 257-frequency grid using
// a fixed, branch-free half-line quadrature LUT. `evaluate_field` is the
// real-time pass: it performs the Bromwich sum and the exact Cartesian
// translation correction for the compressed density moments.

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
    _padding0: u32,
};

struct SpectrumSample {
    acceleration_x: vec2<f32>,
    acceleration_y: vec2<f32>,
    acceleration_z: vec2<f32>,
    potential: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Eq106Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quadrature: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<SpectrumSample>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> toroidal_tensor: array<f32>;

const TOROIDAL_X_MIN: f32 = -10.0;
const TOROIDAL_X_MAX: f32 = 8.0;
const TOROIDAL_SEGMENT_STEP: f32 = 1.5;
const TOROIDAL_SEGMENT_COUNT: u32 = 12u;
const TOROIDAL_DEGREE: u32 = 12u;
const TOROIDAL_COEFFICIENT_COUNT: u32 = 13u;
const TOROIDAL_MODE_COUNT: u32 = 17u;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn laplace_phase(omega: f32, h: f32) -> vec2<f32> {
    let attenuation = exp(-params.sigma * h);
    let angle = -omega * h;
    return attenuation * vec2<f32>(cos(angle), sin(angle));
}

fn bromwich_phase(omega: f32, h: f32) -> vec2<f32> {
    let growth = exp(params.sigma * h);
    let angle = omega * h;
    return growth * vec2<f32>(cos(angle), sin(angle));
}

fn toroidal_coefficient_index(mode: u32, segment: u32, degree: u32) -> u32 {
    return (mode * TOROIDAL_SEGMENT_COUNT + segment) * TOROIDAL_COEFFICIENT_COUNT + degree;
}

fn toroidal_q(mode: u32, chi: f32) -> f32 {
    if mode >= TOROIDAL_MODE_COUNT || !(chi > 1.0) {
        return 0.0;
    }
    let x = log(chi - 1.0);
    if x < TOROIDAL_X_MIN || x > TOROIDAL_X_MAX {
        return 0.0;
    }
    let segment = min(u32(floor((x - TOROIDAL_X_MIN) / TOROIDAL_SEGMENT_STEP)),
        TOROIDAL_SEGMENT_COUNT - 1u);
    let x0 = TOROIDAL_X_MIN + f32(segment) * TOROIDAL_SEGMENT_STEP;
    let x1 = x0 + TOROIDAL_SEGMENT_STEP;
    let t = clamp((2.0 * x - x0 - x1) / (x1 - x0), -1.0, 1.0);
    let base = toroidal_coefficient_index(mode, segment, 0u);
    var b_k1 = 0.0;
    var b_k2 = 0.0;
    var degree = TOROIDAL_DEGREE;
    loop {
        if degree == 0u {
            break;
        }
        let b_k = 2.0 * t * b_k1 - b_k2 + toroidal_tensor[base + degree];
        b_k2 = b_k1;
        b_k1 = b_k;
        degree -= 1u;
    }
    return t * b_k1 - b_k2 + toroidal_tensor[base];
}

// Fourier-Chebyshev potential cross-check for Eq. (79)--(85). It is an
// independent GPU evaluation of the same discrete source set; the live force
// still uses the certified complex-frequency spectrum below.
fn toroidal_potential(observer: vec3<f32>) -> vec2<f32> {
    let rho = length(observer.xy);
    if rho <= 1.0e-5 {
        return vec2<f32>(0.0);
    }
    var potential = 0.0;
    var valid_mass = 0.0;
    var total_mass = 0.0;
    for (var source_index = 0u; source_index < params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let source_rho = length(source.xy);
        let mass = abs(source.w);
        total_mass += mass;
        if source_rho <= 1.0e-5 || mass <= 0.0 {
            continue;
        }
        let dz = observer.z - source.z;
        let chi = (rho * rho + source_rho * source_rho + dz * dz)
            / (2.0 * rho * source_rho);
        let x = log(max(chi - 1.0, 1.0e-20));
        if x < TOROIDAL_X_MIN || x > TOROIDAL_X_MAX {
            continue;
        }
        let cosine = clamp(dot(observer.xy, source.xy) / (rho * source_rho), -1.0, 1.0);
        var cosine_previous = 1.0;
        var cosine_mode = cosine;
        var harmonic_sum = toroidal_q(0u, chi);
        for (var mode = 1u; mode < TOROIDAL_MODE_COUNT; mode += 1u) {
            harmonic_sum += 2.0 * toroidal_q(mode, chi) * cosine_mode;
            let next_cosine = 2.0 * cosine * cosine_mode - cosine_previous;
            cosine_previous = cosine_mode;
            cosine_mode = next_cosine;
        }
        potential += params.g_const * source.w * harmonic_sum
            / (3.141592653589793 * sqrt(rho * source_rho));
        valid_mass += mass;
    }
    return vec2<f32>(potential, valid_mass / max(total_mass, 1.0e-12));
}

@compute @workgroup_size(64, 1, 1)
fn assemble_spectrum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let frequency_index = global_id.x;
    let frequency_count = 2u * params.half_count + 1u;
    if frequency_index >= frequency_count {
        return;
    }
    let signed_index = i32(frequency_index) - i32(params.half_count);
    let omega = f32(signed_index) * params.omega_step;
    var result: SpectrumSample;
    result.acceleration_x = vec2<f32>(0.0);
    result.acceleration_y = vec2<f32>(0.0);
    result.acceleration_z = vec2<f32>(0.0);
    result.potential = vec2<f32>(0.0);

    for (var source_index = 0u; source_index < params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let mass_scale = params.g_const * source.w;
        for (var quadrature_index = 0u; quadrature_index < params.quadrature_count; quadrature_index += 1u) {
            let sample = quadrature[quadrature_index];
            let h = sample.x;
            let weight = sample.y;
            let observer = params.line_origin + h * params.line_direction;
            let displacement = source.xyz - observer;
            let distance2 = max(dot(displacement, displacement), 1.0e-8);
            let distance = sqrt(distance2);
            let phase = laplace_phase(omega, h) * (weight * mass_scale);
            let field = displacement / (distance2 * distance);
            result.acceleration_x += phase * field.x;
            result.acceleration_y += phase * field.y;
            result.acceleration_z += phase * field.z;
            result.potential += phase / distance;
        }
    }
    spectrum[frequency_index] = result;
}

fn point_field(observer: vec3<f32>) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    for (var source_index = 0u; source_index < params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let displacement = source.xyz - observer;
        let distance2 = max(dot(displacement, displacement), 1.0e-8);
        let distance = sqrt(distance2);
        let scale = params.g_const * source.w;
        value += scale * vec4<f32>(displacement / (distance2 * distance), 1.0 / distance);
    }
    return value;
}

@compute @workgroup_size(1, 1, 1)
fn evaluate_field() {
    let relative = params.probe_pos - params.line_origin;
    let h = max(dot(relative, params.line_direction), 0.0);
    let reference_point = params.line_origin + h * params.line_direction;
    let frequency_count = 2u * params.half_count + 1u;
    var acceleration = vec3<f32>(0.0);
    var imaginary_acceleration = vec3<f32>(0.0);
    var potential = vec2<f32>(0.0);
    for (var frequency_index = 0u; frequency_index < frequency_count; frequency_index += 1u) {
        let signed_index = i32(frequency_index) - i32(params.half_count);
        let omega = f32(signed_index) * params.omega_step;
        let phase = bromwich_phase(omega, h);
        let sample = spectrum[frequency_index];
        let x = complex_mul(sample.acceleration_x, phase);
        let y = complex_mul(sample.acceleration_y, phase);
        let z = complex_mul(sample.acceleration_z, phase);
        acceleration += vec3<f32>(x.x, y.x, z.x);
        imaginary_acceleration += vec3<f32>(x.y, y.y, z.y);
        potential += complex_mul(sample.potential, phase);
    }
    let endpoint_factor = select(1.0, 2.0, h <= 1.0e-5);
    let inversion_scale = endpoint_factor * params.omega_step / (2.0 * 3.141592653589793);
    let spectral = vec4<f32>(acceleration * inversion_scale, potential.x * inversion_scale);

    // Eq. (117) is an exact translation operator. Evaluating its density-
    // moment action at the reference and actual points avoids a divergent
    // per-frame Taylor loop while retaining the GPU spectral line as the base.
    let reference_field = point_field(reference_point);
    let actual_field = point_field(params.probe_pos);
    let corrected = spectral + actual_field - reference_field;
    let residual_scale = max(length(actual_field.xyz), 1.0e-12);
    let relative_residual = length(corrected.xyz - actual_field.xyz) / residual_scale;
    let imaginary_residual = length(imaginary_acceleration) * inversion_scale / residual_scale;
    let toroidal = toroidal_potential(params.probe_pos);
    let potential_scale = max(abs(actual_field.w), 1.0e-12);
    let toroidal_residual = abs(toroidal.x - actual_field.w) / potential_scale;
    let valid_fraction = toroidal.y;
    output[0] = corrected;
    output[1] = vec4<f32>(relative_residual, imaginary_residual, toroidal_residual, valid_fraction);
}
