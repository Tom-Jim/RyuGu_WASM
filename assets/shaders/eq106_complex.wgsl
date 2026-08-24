// Equation (106) cached transverse-Taylor spectral pipeline.
//
// Source traversal is confined to `assemble_line_samples`, which runs only
// when a local reference line is created. It builds all two-dimensional
// transverse Taylor coefficients through the certified order selected per
// segment (orders 1..8). `assemble_spectrum`
// Laplace-transforms those coefficients once. Every real-time query thereafter
// uses only the cached spectra; it never reads the source buffer.

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

struct DensityModeBuffer {
    records: array<vec4<f32>, 544>,
};

struct ToroidalTensorBuffer {
    values: array<vec4<f32>, 663>,
};

struct SpectrumSample {
    acceleration_x: vec2<f32>,
    acceleration_y: vec2<f32>,
    acceleration_z: vec2<f32>,
    potential: vec2<f32>,
};

// The z/y dispatch dimensions select the segment record, so all segments
// advance through the same four kernels in one command.
@group(0) @binding(0) var<storage, read> segment_params: array<Eq106Params>;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quadrature: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<SpectrumSample>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;
// Bindings 5 and 7 implement the independent Eq. (79)-(83)
// Fourier-toroidal potential used by the runtime Eq. (157) residual.
@group(0) @binding(5) var<uniform> toroidal_tensor: ToroidalTensorBuffer;
@group(0) @binding(6) var<storage, read_write> line_samples: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> density_modes: DensityModeBuffer;
@group(0) @binding(8) var<storage, read> psi_operator: array<f32>;
@group(0) @binding(9) var<storage, read> targets: array<vec4<f32>>;

var<workgroup> eq_params: Eq106Params;

var<workgroup> evaluation_phase: array<vec2<f32>, 129>;
var<workgroup> evaluation_spectral_derivative: array<vec2<f32>, 129>;
var<workgroup> evaluation_omega: array<f32, 129>;
var<workgroup> evaluation_sum: array<vec4<f32>, 64>;
var<workgroup> evaluation_derivative: array<vec4<f32>, 64>;
var<workgroup> evaluation_imaginary: array<vec4<f32>, 64>;
var<workgroup> evaluation_tail: array<vec4<f32>, 64>;
var<workgroup> evaluation_edge: array<vec2<f32>, 64>;

const PI: f32 = 3.141592653589793;
const TAYLOR_MAX_ORDER: u32 = 8u;
const MAX_TAYLOR_COEFFICIENT_COUNT: u32 = 45u;
const QUADRATURE_CAPACITY: u32 = 64u;
const SPECTRUM_FREQUENCY_CAPACITY: u32 = 129u;
const OUTPUT_ROWS_PER_BLOCK: u32 = 9u;
const TARGET_DISPATCH_WIDTH: u32 = 65535u;
const TOROIDAL_MODE_COUNT: u32 = 17u;
const TOROIDAL_SEGMENT_COUNT: u32 = 12u;
const TOROIDAL_DEGREE: u32 = 12u;
const TOROIDAL_X_MIN: f32 = -10.0;
const TOROIDAL_X_MAX: f32 = 8.0;
const TOROIDAL_SEGMENT_STEP: f32 = 1.5;
const PSI_SEGMENT_COUNT: u32 = 16u;
const PSI_DEGREE: u32 = 8u;
const PSI_LOG_A_MIN: f32 = -3.2188758248682006;
const PSI_LOG_A_MAX: f32 = 1.791759469228055;
const PSI_LOG_A_STEP: f32 = (PSI_LOG_A_MAX - PSI_LOG_A_MIN) / f32(PSI_SEGMENT_COUNT);

const GL8_NODES = array<f32, 8>(
    -0.9602898564975363, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498,
     0.1834346424956498,  0.5255324099163290,
     0.7966664774136267,  0.9602898564975363,
);
const GL8_WEIGHTS = array<f32, 8>(
    0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
);
const GL16_NODES = array<f32, 16>(
    0.08764941047892784, 0.4626963289150808,
    1.1410577748312269, 2.1292836450983806,
    3.4370866338932066, 5.078018614549768,
    7.070338535048234, 9.438314336391938,
    12.21422336886616, 15.44152736878162,
    19.18015685675313, 23.51590569399191,
    28.57872974288214, 34.58339870228663,
    41.94045264768833, 51.70116033954332,
);
const GL16_WEIGHTS = array<f32, 16>(
    2.0615171495780099e-1, 3.310578549508842e-1,
    2.6579577764421415e-1, 1.3629693429637754e-1,
    4.732892869412522e-2, 1.1299900080339454e-2,
    1.84907094352631e-3, 2.0427191530827846e-4,
    1.4844586873981299e-5, 6.8283193308712e-7,
    1.88102484107967e-8, 2.86235024297388e-10,
    2.1270790332241e-12, 6.29796700251788e-15,
    5.05047370003551e-18, 4.16146237037285e-22,
);

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_exp(value: vec2<f32>) -> vec2<f32> {
    let amplitude = exp(value.x);
    return amplitude * vec2<f32>(cos(value.y), sin(value.y));
}

fn complex_inverse(value: vec2<f32>) -> vec2<f32> {
    let norm_squared = max(dot(value, value), 1.0e-30);
    return vec2<f32>(value.x, -value.y) / norm_squared;
}

struct PsiBase {
    value: vec2<f32>,
    derivative: vec2<f32>,
};

struct PsiPair {
    value: vec2<f32>,
    derivative: vec2<f32>,
};

fn psi_full_asymptotic(x: vec2<f32>, eta: f32) -> PsiPair {
    let polynomial = array<f32, 3>(1.0 + eta * eta, -2.0 * eta, 1.0);
    var coefficients: array<f32, 33>;
    coefficients[0] = inverseSqrt(polynomial[0]);
    let inverse_x = complex_inverse(x);
    var inverse_power = inverse_x;
    var factorial = 1.0;
    var previous_term = 3.402823e38;
    var result: PsiPair;
    result.value = vec2<f32>(0.0);
    result.derivative = vec2<f32>(0.0);
    for (var order = 0u; order < 33u; order += 1u) {
        if order > 0u {
            var numerator = 0.0;
            for (var degree = 1u; degree <= min(2u, order); degree += 1u) {
                numerator += ((0.5 * f32(degree)) - f32(order))
                    * polynomial[degree] * coefficients[order - degree];
            }
            coefficients[order] = numerator / (f32(order) * polynomial[0]);
            factorial *= f32(order);
        }
        let term = inverse_power * (factorial * coefficients[order]);
        let magnitude = length(term);
        if magnitude > 0.0 {
            if magnitude >= previous_term {
                break;
            }
            previous_term = magnitude;
        }
        result.value += term;
        result.derivative -= complex_mul(term, inverse_x) * f32(order + 1u);
        inverse_power = complex_mul(inverse_power, inverse_x);
    }
    return result;
}

fn psi_table_component(frequency: u32, segment: u32, t: f32, component: u32) -> f32 {
    let coefficient_count = PSI_DEGREE + 1u;
    let base = ((frequency * PSI_SEGMENT_COUNT + segment) * coefficient_count) * 4u + component;
    var b1 = 0.0;
    var b2 = 0.0;
    var degree = PSI_DEGREE;
    loop {
        let coefficient = psi_operator[base + degree * 4u];
        let b = 2.0 * t * b1 - b2 + coefficient;
        b2 = b1;
        b1 = b;
        if degree == 1u {
            break;
        }
        degree -= 1u;
    }
    return t * b1 - b2 + psi_operator[base];
}

// Certified complex Chebyshev lookup for the Struve--Neumann half-axis term
// L(x)=pi/2(H_0(x)-Y_0(x)) and L_x.  The table stores non-negative frequency
// rays; real source data gives the negative rays by complex conjugation.
fn psi_base(signed_frequency: i32, normalized_a: f32) -> PsiBase {
    let log_a = clamp(log(normalized_a), PSI_LOG_A_MIN, PSI_LOG_A_MAX);
    let segment = min(
        u32(floor((log_a - PSI_LOG_A_MIN) / PSI_LOG_A_STEP)),
        PSI_SEGMENT_COUNT - 1u,
    );
    let x0 = PSI_LOG_A_MIN + f32(segment) * PSI_LOG_A_STEP;
    let x1 = x0 + PSI_LOG_A_STEP;
    let t = clamp((2.0 * log_a - x0 - x1) / (x1 - x0), -1.0, 1.0);
    let frequency = u32(abs(signed_frequency));
    var result: PsiBase;
    result.value = vec2<f32>(
        psi_table_component(frequency, segment, t, 0u),
        psi_table_component(frequency, segment, t, 1u),
    );
    result.derivative = vec2<f32>(
        psi_table_component(frequency, segment, t, 2u),
        psi_table_component(frequency, segment, t, 3u),
    );
    if signed_frequency < 0 {
        result.value.y = -result.value.y;
        result.derivative.y = -result.derivative.y;
    }
    return result;
}

// Analytic finite-eta continuation of the incomplete Struve term. Inside the
// convergence disk it uses the coefficient recurrence implied by
// (1+eta^2) f'=[x(1+eta^2)-eta]f. Outside, an 8-node real-path rule avoids
// extending that Taylor series across its eta=+-i singularities.
fn finite_eta_correction(x: vec2<f32>, eta: f32) -> PsiPair {
    var result: PsiPair;
    result.value = vec2<f32>(0.0);
    result.derivative = vec2<f32>(0.0);
    if abs(eta) <= 0.72 && length(x) * abs(eta) <= 4.0 {
        var c_minus_two = vec2<f32>(0.0);
        var c_minus_one = vec2<f32>(0.0);
        var c = vec2<f32>(1.0, 0.0);
        var dc_minus_two = vec2<f32>(0.0);
        var dc_minus_one = vec2<f32>(0.0);
        var dc = vec2<f32>(0.0);
        var eta_power = eta;
        for (var order = 0u; order <= 36u; order += 1u) {
            let denominator = f32(order + 1u);
            result.value += c * (eta_power / denominator);
            result.derivative += dc * (eta_power / denominator);
            let next = (
                complex_mul(x, c) + complex_mul(x, c_minus_two)
                - f32(order) * c_minus_one
            ) / denominator;
            let next_derivative = (
                c + complex_mul(x, dc) + c_minus_two
                + complex_mul(x, dc_minus_two) - f32(order) * dc_minus_one
            ) / denominator;
            c_minus_two = c_minus_one;
            c_minus_one = c;
            c = next;
            dc_minus_two = dc_minus_one;
            dc_minus_one = dc;
            dc = next_derivative;
            eta_power *= eta;
        }
        let phase = complex_exp(-x * eta);
        let j = result.value;
        result.value = complex_mul(phase, j);
        result.derivative = complex_mul(phase, result.derivative - eta * j);
        return result;
    }

    // Return the already scaled quantities exp(-x eta)J and
    // d/dx[exp(-x eta)J], which avoids a large intermediate for eta<0.
    let segment_count = clamp(u32(ceil(length(x) * abs(eta) / 2.0)), 1u, 12u);
    for (var segment = 0u; segment < segment_count; segment += 1u) {
        let start = eta * f32(segment) / f32(segment_count);
        let end = eta * f32(segment + 1u) / f32(segment_count);
        let midpoint = 0.5 * (start + end);
        let half_width = 0.5 * (end - start);
        for (var node = 0u; node < 8u; node += 1u) {
            let v = midpoint + half_width * GL8_NODES[node];
            let weight = half_width * GL8_WEIGHTS[node] / sqrt(1.0 + v * v);
            let phase = complex_exp(-x * (eta - v));
            result.value += phase * weight;
            result.derivative += phase * (weight * (v - eta));
        }
    }
    return result;
}

// a->0, z'<0: exp(-s z') E1(-s z') = integral_0^infinity
// exp(-t)/(t-s z') dt. Gauss--Laguerre makes this an explicit rational limit.
fn scaled_e1_axis_limit(w: vec2<f32>) -> vec2<f32> {
    if length(w) < 4.0 {
        let logarithm = vec2<f32>(log(length(w)), atan2(w.y, w.x));
        var e1 = -logarithm - vec2<f32>(0.5772156649015329, 0.0);
        var power = vec2<f32>(1.0, 0.0);
        var factorial = 1.0;
        for (var order = 1u; order <= 36u; order += 1u) {
            power = complex_mul(power, -w);
            factorial *= f32(order);
            e1 -= power / (f32(order) * factorial);
        }
        return complex_mul(complex_exp(w), e1);
    }
    var result = vec2<f32>(0.0);
    for (var node = 0u; node < 16u; node += 1u) {
        result += GL16_WEIGHTS[node]
            * complex_inverse(w + vec2<f32>(GL16_NODES[node], 0.0));
    }
    return result;
}

fn laplace_phase(omega: f32, h: f32) -> vec2<f32> {
    let attenuation = exp(-eq_params.sigma * h);
    let angle = -omega * h;
    return attenuation * vec2<f32>(cos(angle), sin(angle));
}

fn coefficient_index(a: u32, b: u32) -> u32 {
    let degree = a + b;
    return degree * (degree + 1u) / 2u + b;
}

fn coefficient_degree(index: u32) -> u32 {
    return u32(floor((sqrt(8.0 * f32(index) + 1.0) - 1.0) * 0.5));
}

fn coefficient_a(index: u32) -> u32 {
    let degree = coefficient_degree(index);
    let b = index - degree * (degree + 1u) / 2u;
    return degree - b;
}

fn coefficient_b(index: u32) -> u32 {
    let degree = coefficient_degree(index);
    return index - degree * (degree + 1u) / 2u;
}

fn active_coefficient_count() -> u32 {
    let order = min(eq_params.taylor_order, TAYLOR_MAX_ORDER);
    return (order + 1u) * (order + 2u) / 2u;
}

fn transverse_basis(direction: vec3<f32>) -> mat3x3<f32> {
    let tangent = normalize(direction);
    let helper = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(tangent.z) > 0.8);
    let normal = normalize(cross(helper, tangent));
    let binormal = normalize(cross(tangent, normal));
    return mat3x3<f32>(tangent, normal, binormal);
}

fn scalar_power_coefficient(
    coefficients: ptr<function, array<f32, 45>>,
    a: u32,
    b: u32,
) -> f32 {
    if a + b > TAYLOR_MAX_ORDER {
        return 0.0;
    }
    return (*coefficients)[coefficient_index(a, b)];
}

// Build the coefficients of
// (x0 + x10*u + x01*v + u^2 + v^2)^alpha
// by grouping the bivariate series by total degree. This is the multivariate
// form of the same power-series recurrence used by the CPU reference tests.
fn build_radial_power_series(
    x0: f32,
    x10: f32,
    x01: f32,
    alpha: f32,
) -> array<f32, 45> {
    var result: array<f32, 45>;
    for (var index = 0u; index < active_coefficient_count(); index += 1u) {
        result[index] = 0.0;
    }
    result[0] = pow(x0, alpha);
    for (var degree = 1u; degree <= min(eq_params.taylor_order, TAYLOR_MAX_ORDER); degree += 1u) {
        for (var b = 0u; b <= degree; b += 1u) {
            let a = degree - b;
            var numerator = 0.0;
            let linear_factor = (alpha + 1.0) - f32(degree);
            if a >= 1u {
                numerator += linear_factor * x10 * scalar_power_coefficient(&result, a - 1u, b);
            }
            if b >= 1u {
                numerator += linear_factor * x01 * scalar_power_coefficient(&result, a, b - 1u);
            }
            let quadratic_factor = 2.0 * (alpha + 1.0) - f32(degree);
            if a >= 2u {
                numerator += quadratic_factor * scalar_power_coefficient(&result, a - 2u, b);
            }
            if b >= 2u {
                numerator += quadratic_factor * scalar_power_coefficient(&result, a, b - 2u);
            }
            result[coefficient_index(a, b)] = numerator / (f32(degree) * x0);
        }
    }
    return result;
}

@compute @workgroup_size(64, 1, 1)
fn assemble_line_samples(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let quadrature_index = global_id.x;
    let segment_index = global_id.y;
    if quadrature_index == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    if quadrature_index >= eq_params.quadrature_count {
        return;
    }

    let sample = quadrature[quadrature_index];
    let basis = transverse_basis(eq_params.line_direction);
    let tangent = basis[0];
    let normal = basis[1];
    let binormal = basis[2];
    let observer = eq_params.line_origin + sample.x * tangent;
    let coefficient_count = active_coefficient_count();
    var packed: array<vec4<f32>, 45>;
    for (var index = 0u; index < coefficient_count; index += 1u) {
        packed[index] = vec4<f32>(0.0);
    }

    // This is the only production pass that traverses sources. It runs once per
    // reference segment and produces every coefficient needed by later queries.
    for (var source_index = 0u; source_index < eq_params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let r0 = source.xyz - observer;
        let x0 = max(dot(r0, r0), 1.0e-8);
        let x10 = -2.0 * dot(r0, normal);
        let x01 = -2.0 * dot(r0, binormal);
        let inverse_r = build_radial_power_series(x0, x10, x01, -0.5);
        let inverse_r3 = build_radial_power_series(x0, x10, x01, -1.5);
        let scale = eq_params.g_const * source.w;

        for (var coefficient = 0u; coefficient < coefficient_count; coefficient += 1u) {
            let a = coefficient_a(coefficient);
            let b = coefficient_b(coefficient);
            var previous_u = 0.0;
            var previous_v = 0.0;
            if a > 0u {
                previous_u = inverse_r3[coefficient_index(a - 1u, b)];
            }
            if b > 0u {
                previous_v = inverse_r3[coefficient_index(a, b - 1u)];
            }
            packed[coefficient] += scale * vec4<f32>(
                r0 * inverse_r3[coefficient] - normal * previous_u - binormal * previous_v,
                inverse_r[coefficient],
            );
        }
    }

    for (var coefficient = 0u; coefficient < coefficient_count; coefficient += 1u) {
        let destination = (eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * QUADRATURE_CAPACITY
            + coefficient * QUADRATURE_CAPACITY + quadrature_index;
        line_samples[destination] = packed[coefficient] * sample.y;
    }
}

@compute @workgroup_size(64, 1, 1)
fn assemble_spectrum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let segment_index = global_id.y;
    if global_id.x == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let frequency_count = 2u * eq_params.half_count + 1u;
    let work_count = active_coefficient_count() * frequency_count;
    let flat_index = global_id.x;
    if flat_index >= work_count {
        return;
    }
    let coefficient = flat_index / frequency_count;
    let frequency_index = flat_index % frequency_count;
    let signed_index = i32(frequency_index) - i32(eq_params.half_count);
    let omega = f32(signed_index) * eq_params.omega_step;
    var result: SpectrumSample;
    result.acceleration_x = vec2<f32>(0.0);
    result.acceleration_y = vec2<f32>(0.0);
    result.acceleration_z = vec2<f32>(0.0);
    result.potential = vec2<f32>(0.0);

    for (var quadrature_index = 0u; quadrature_index < eq_params.quadrature_count; quadrature_index += 1u) {
        let quadrature_sample = quadrature[quadrature_index];
        let packed = line_samples[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * QUADRATURE_CAPACITY
            + coefficient * QUADRATURE_CAPACITY + quadrature_index];
        let phase = laplace_phase(omega, quadrature_sample.x);
        result.acceleration_x += phase * packed.x;
        result.acceleration_y += phase * packed.y;
        result.acceleration_z += phase * packed.z;
        result.potential += phase * packed.w;
    }
    spectrum[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY
        + flat_index] = result;
}

// Eqs. (47),(68)-(70): source-summed transformed field on the reference line.
// This pass overwrites coefficient zero after the sampled higher-order Taylor
// jet has been assembled. It performs no half-line quadrature and its cost is
// O(N_source N_frequency) table/recurrence work per new spectral element.
@compute @workgroup_size(64, 1, 1)
fn assemble_analytic_spectrum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let segment_index = global_id.y;
    if global_id.x == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let frequency_index = global_id.x;
    let frequency_count = 2u * eq_params.half_count + 1u;
    if frequency_index >= frequency_count {
        return;
    }
    let signed_index = i32(frequency_index) - i32(eq_params.half_count);
    let omega = f32(signed_index) * eq_params.omega_step;
    let s = vec2<f32>(eq_params.sigma, omega);
    let radius = 2.0 / max(eq_params.sigma, 1.0e-12);
    let basis = transverse_basis(eq_params.line_direction);
    let tangent = basis[0];
    var result: SpectrumSample;
    result.acceleration_x = vec2<f32>(0.0);
    result.acceleration_y = vec2<f32>(0.0);
    result.acceleration_z = vec2<f32>(0.0);
    result.potential = vec2<f32>(0.0);
    var valid = true;

    for (var source_index = 0u; source_index < eq_params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let relative = source.xyz - eq_params.line_origin;
        let z_prime = dot(relative, tangent);
        let transverse = relative - z_prime * tangent;
        let a = length(transverse);
        let normalized_a = a / radius;
        let scale = eq_params.g_const * source.w;

        if normalized_a < exp(PSI_LOG_A_MIN) {
            // The point-source line transform is singular when a=0 and the
            // source lies on the forward half-line. A valid exterior element
            // can only approach the regular z'<0 axis limit.
            if z_prime >= -1.0e-5 * radius {
                valid = false;
                continue;
            }
            let w = -s * z_prime;
            let psi = scaled_e1_axis_limit(w);
            let vertical = complex_mul(s, psi)
                - vec2<f32>(1.0 / abs(z_prime), 0.0);
            result.acceleration_x += vertical * (scale * tangent.x);
            result.acceleration_y += vertical * (scale * tangent.y);
            result.acceleration_z += vertical * (scale * tangent.z);
            result.potential += psi * scale;
            continue;
        }
        if normalized_a > exp(PSI_LOG_A_MAX) {
            valid = false;
            continue;
        }

        let eta = z_prime / a;
        let x = s * a;
        var psi = vec2<f32>(0.0);
        var psi_x = vec2<f32>(0.0);
        if length(x) >= 16.0 || -x.x * eta >= 6.0 {
            let full = psi_full_asymptotic(x, eta);
            psi = full.value;
            psi_x = full.derivative;
        } else {
            let base = psi_base(signed_index, normalized_a);
            let correction = finite_eta_correction(x, eta);
            let phase = complex_exp(-x * eta);
            psi = complex_mul(phase, base.value) + correction.value;
            psi_x = complex_mul(
                phase,
                base.derivative - eta * base.value,
            ) + correction.derivative;
        }
        let inverse_boundary = inverseSqrt(1.0 + eta * eta);
        let x_psi = complex_mul(x, psi);
        let k_v = x_psi - vec2<f32>(inverse_boundary, 0.0);
        let k_h = complex_mul(x, psi_x) + eta * x_psi
            - vec2<f32>(eta * inverse_boundary, 0.0);
        let horizontal_direction = transverse / a;
        let horizontal = k_h * (-scale / a);
        let vertical = k_v * (scale / a);
        result.acceleration_x += horizontal * horizontal_direction.x + vertical * tangent.x;
        result.acceleration_y += horizontal * horizontal_direction.y + vertical * tangent.y;
        result.acceleration_z += horizontal * horizontal_direction.z + vertical * tangent.z;
        result.potential += psi * scale;
    }

    if valid {
        spectrum[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY
            + frequency_index] = result;
    } else {
        // A rejected operator-domain query must not silently fall back to a
        // different force law. Chrome's WebGPU validator rejects a literal
        // NaN in WGSL, so use a finite out-of-domain sentinel; the CPU gate
        // rejects magnitudes above the physical bound and rebuilds the line.
        let invalid = 3.0e30;
        result.acceleration_x = vec2<f32>(invalid);
        result.acceleration_y = vec2<f32>(invalid);
        result.acceleration_z = vec2<f32>(invalid);
        result.potential = vec2<f32>(invalid);
        spectrum[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY
            + frequency_index] = result;
    }
}

fn integer_power(value: f32, exponent: u32) -> f32 {
    var result = 1.0;
    for (var index = 0u; index < exponent; index += 1u) {
        result *= value;
    }
    return result;
}

fn monomial(value_u: f32, value_v: f32, a: u32, b: u32) -> f32 {
    return integer_power(value_u, a) * integer_power(value_v, b);
}

fn toroidal_tensor_value(index: u32) -> f32 {
    let packed = toroidal_tensor.values[index / 4u];
    return packed[index % 4u];
}

fn toroidal_q(mode: u32, chi: f32) -> f32 {
    let x = clamp(log(max(chi - 1.0, exp(TOROIDAL_X_MIN))), TOROIDAL_X_MIN, TOROIDAL_X_MAX);
    let segment = min(u32(floor((x - TOROIDAL_X_MIN) / TOROIDAL_SEGMENT_STEP)), TOROIDAL_SEGMENT_COUNT - 1u);
    let x0 = TOROIDAL_X_MIN + f32(segment) * TOROIDAL_SEGMENT_STEP;
    let x1 = x0 + TOROIDAL_SEGMENT_STEP;
    let t = clamp((2.0 * x - x0 - x1) / (x1 - x0), -1.0, 1.0);
    let base = (mode * TOROIDAL_SEGMENT_COUNT + segment) * (TOROIDAL_DEGREE + 1u);
    var b_k1 = 0.0;
    var b_k2 = 0.0;
    var degree = TOROIDAL_DEGREE;
    loop {
        let b_k = 2.0 * t * b_k1 - b_k2 + toroidal_tensor_value(base + degree);
        b_k2 = b_k1;
        b_k1 = b_k;
        if degree == 1u {
            break;
        }
        degree -= 1u;
    }
    return t * b_k1 - b_k2 + toroidal_tensor_value(base);
}

// Eq. (79)-(83): independent Fourier-toroidal potential. The ring-major
// density buffer stores m=0..16 for every (r',z') ring. This is deliberately
// independent of the Eq. (70) Bromwich/Taylor path used for acceleration.
fn fourier_toroidal_potential(position: vec3<f32>) -> f32 {
    let rho = length(position.xy);
    if rho <= 1.0e-4 {
        return 0.0;
    }
    let phi = atan2(position.y, position.x);
    var potential = 0.0;
    for (var index = 0u; index < eq_params.density_mode_count; index += 1u) {
        let record = density_modes.records[index];
        let mode = index % TOROIDAL_MODE_COUNT;
        let ring_radius = record.x;
        if ring_radius <= 1.0e-6 {
            continue;
        }
        let dz = position.z - record.y;
        let chi = max(
            (rho * rho + ring_radius * ring_radius + dz * dz) / (2.0 * rho * ring_radius),
            1.0 + exp(TOROIDAL_X_MIN),
        );
        let q = toroidal_q(mode, chi);
        let angle = f32(mode) * phi;
        let real_mode = record.z * cos(angle) - record.w * sin(angle);
        let symmetry = select(2.0, 1.0, mode == 0u);
        potential += symmetry * q * real_mode / sqrt(rho * ring_radius);
    }
    return eq_params.g_const * potential / PI;
}

@compute @workgroup_size(64, 1, 1)
fn evaluate_field(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let segment_index = workgroup_id.z;
    let lane = local_id.x;
    if lane == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let local_target_index = workgroup_id.y * TARGET_DISPATCH_WIDTH + workgroup_id.x;
    let target_active = local_target_index < eq_params.target_count;
    // Batched segments can have different target counts. Extra workgroups must
    // still traverse every barrier, so evaluate a clamped record and discard
    // its result only after the last barrier.
    let safe_target_count = max(eq_params.target_count, 1u);
    let safe_local_target_index = min(local_target_index, safe_target_count - 1u);
    let target_index = eq_params.target_offset + safe_local_target_index;
    let target_record = targets[eq_params.input_base + target_index];
    let probe_pos = target_record.xyz;
    let elapsed = target_record.w;
    let basis = transverse_basis(eq_params.line_direction);
    let tangent = basis[0];
    let normal = basis[1];
    let binormal = basis[2];
    let relative = probe_pos - eq_params.line_origin;
    let h = max(dot(relative, tangent), 0.0);
    let u = dot(relative, normal);
    let v = dot(relative, binormal);
    let frequency_count = 2u * eq_params.half_count + 1u;
    let active_order = min(eq_params.taylor_order, TAYLOR_MAX_ORDER);
    let coefficient_count = active_coefficient_count();
    // The frequency grid already contains the complete signed band
    // [-Omega, +Omega]. Do not apply a second endpoint doubling at h=0;
    // doing so creates a force/potential jump whenever a new line starts.
    let inversion_scale = eq_params.omega_step / (2.0 * PI);

    var field = vec4<f32>(0.0);
    var derivative_h = vec3<f32>(0.0);
    var derivative_u = vec3<f32>(0.0);
    var derivative_v = vec3<f32>(0.0);
    var imaginary_field = vec3<f32>(0.0);
    var last_order_field = vec3<f32>(0.0);
    var tail_field = vec3<f32>(0.0);

    var independent_potential = 0.0;
    if lane == 0u && target_active && eq_params.evaluate_dual_certificate != 0u && target_index == 0u {
        independent_potential = fourier_toroidal_potential(probe_pos);
    }

    // Frequencies are equally spaced. Seed the negative endpoint once and
    // rotate by exp(i*delta_omega*h). Lane zero constructs the recurrence once;
    // all lanes then share it while reducing the frequency dimension.
    if lane == 0u {
        let first_angle = -f32(eq_params.half_count) * eq_params.omega_step * h;
        let growth = exp(eq_params.sigma * h);
        var phase = growth * vec2<f32>(cos(first_angle), sin(first_angle));
        let phase_step_angle = eq_params.omega_step * h;
        let phase_step = vec2<f32>(cos(phase_step_angle), sin(phase_step_angle));
        for (var frequency_index = 0u; frequency_index < frequency_count; frequency_index += 1u) {
            let signed_index = i32(frequency_index) - i32(eq_params.half_count);
            let omega = f32(signed_index) * eq_params.omega_step;
            evaluation_phase[frequency_index] = phase;
            evaluation_omega[frequency_index] = omega;
            evaluation_spectral_derivative[frequency_index] = complex_mul(
                vec2<f32>(eq_params.sigma, omega),
                phase,
            );
            phase = complex_mul(phase, phase_step);
        }
    }
    workgroupBarrier();

    var local_edge = vec2<f32>(0.0);
    for (var frequency_index = lane; frequency_index < frequency_count; frequency_index += 64u) {
        let omega = evaluation_omega[frequency_index];
        let denominator = max(eq_params.sigma * eq_params.sigma + omega * omega, 1.0e-20);
        let phase = evaluation_phase[frequency_index];
        local_edge.x += (phase.x * eq_params.sigma + phase.y * omega) / denominator;
        if eq_params.inversion_mode == 0u {
            let spectral_derivative = evaluation_spectral_derivative[frequency_index];
            local_edge.y += (
                spectral_derivative.x * eq_params.sigma
                + spectral_derivative.y * omega
            ) / denominator;
        }
    }
    evaluation_edge[lane] = local_edge;
    workgroupBarrier();
    var edge_stride = 32u;
    loop {
        if edge_stride == 0u { break; }
        if lane < edge_stride {
            evaluation_edge[lane] += evaluation_edge[lane + edge_stride];
        }
        workgroupBarrier();
        edge_stride = edge_stride >> 1u;
    }

    let raw_edge_response = evaluation_edge[0].x * inversion_scale;
    let edge_response = max(abs(raw_edge_response), 0.25);
    var edge_response_derivative = 0.0;
    if abs(raw_edge_response) > 0.25 {
        let response_sign = select(-1.0, 1.0, raw_edge_response >= 0.0);
        edge_response_derivative = response_sign * evaluation_edge[0].y * inversion_scale;
    }

    // Keep every invocation on the same statically bounded barrier path.
    // Dawn cannot prove that a function result derived from workgroup memory
    // is uniform, even though lane zero initialized `eq_params` above.
    for (var coefficient = 0u; coefficient < MAX_TAYLOR_COEFFICIENT_COUNT; coefficient += 1u) {
        var local_sum = vec4<f32>(0.0);
        var local_derivative = vec3<f32>(0.0);
        var local_imaginary = vec3<f32>(0.0);
        var local_tail = vec3<f32>(0.0);
        // The outer loop and every barrier stay statically uniform, but only
        // active Taylor coefficients perform spectrum loads and complex math.
        // Extra workgroups from a batched dispatch likewise remain on the
        // barrier path without repeating an active target's full evaluation.
        if coefficient < coefficient_count && target_active {
            for (var frequency_index = lane; frequency_index < frequency_count; frequency_index += 64u) {
                let signed_index = i32(frequency_index) - i32(eq_params.half_count);
                let phase = evaluation_phase[frequency_index];
                let sample = spectrum[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY
                    + coefficient * SPECTRUM_FREQUENCY_CAPACITY + frequency_index];
                let x = complex_mul(sample.acceleration_x, phase);
                let y = complex_mul(sample.acceleration_y, phase);
                let z = complex_mul(sample.acceleration_z, phase);
                local_sum += vec4<f32>(
                    x.x,
                    y.x,
                    z.x,
                    complex_mul(sample.potential, phase).x,
                );
                if eq_params.inversion_mode == 0u {
                    let spectral_derivative = evaluation_spectral_derivative[frequency_index];
                    local_imaginary += vec3<f32>(x.y, y.y, z.y);
                    local_derivative += vec3<f32>(
                        complex_mul(sample.acceleration_x, spectral_derivative).x,
                        complex_mul(sample.acceleration_y, spectral_derivative).x,
                        complex_mul(sample.acceleration_z, spectral_derivative).x,
                    );
                    if abs(signed_index) == i32(eq_params.half_count) {
                        local_tail += vec3<f32>(abs(x.x), abs(y.x), abs(z.x));
                    }
                }
            }
        }
        evaluation_sum[lane] = local_sum;
        evaluation_derivative[lane] = vec4<f32>(local_derivative, 0.0);
        evaluation_imaginary[lane] = vec4<f32>(local_imaginary, 0.0);
        evaluation_tail[lane] = vec4<f32>(local_tail, 0.0);
        workgroupBarrier();

        var stride = 32u;
        loop {
            if stride == 0u { break; }
            if lane < stride {
                evaluation_sum[lane] += evaluation_sum[lane + stride];
                evaluation_derivative[lane] += evaluation_derivative[lane + stride];
                evaluation_imaginary[lane] += evaluation_imaginary[lane + stride];
                evaluation_tail[lane] += evaluation_tail[lane + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if lane == 0u && coefficient < coefficient_count {
            let a = coefficient_a(coefficient);
            let b = coefficient_b(coefficient);
            let degree = a + b;
            let raw = evaluation_sum[0] * inversion_scale;
            let reconstructed = raw.xyz / edge_response;
            let reconstructed_potential = raw.w / edge_response;
            let value = monomial(u, v, a, b);
            field += vec4<f32>(reconstructed, reconstructed_potential) * value;
            if eq_params.inversion_mode == 0u {
                let raw_derivative_h = evaluation_derivative[0].xyz * inversion_scale;
                let reconstructed_derivative_h = (
                    raw_derivative_h * edge_response
                    - raw.xyz * edge_response_derivative
                ) / (edge_response * edge_response);
                imaginary_field += evaluation_imaginary[0].xyz
                    * inversion_scale / edge_response * value;
                tail_field += evaluation_tail[0].xyz
                    * inversion_scale / edge_response * abs(value);
                derivative_h += reconstructed_derivative_h * value;
                if a > 0u {
                    derivative_u += reconstructed.xyz * f32(a) * monomial(u, v, a - 1u, b);
                }
                if b > 0u {
                    derivative_v += reconstructed.xyz * f32(b) * monomial(u, v, a, b - 1u);
                }
                if degree == active_order {
                    last_order_field += reconstructed.xyz * value;
                }
            }
        }
        workgroupBarrier();
    }

    if lane != 0u {
        return;
    }
    if !target_active {
        return;
    }
    if eq_params.inversion_mode != 0u {
        output[target_index] = field;
        return;
    }

    let derivative_x = derivative_h * tangent.x + derivative_u * normal.x + derivative_v * binormal.x;
    let derivative_y = derivative_h * tangent.y + derivative_u * normal.y + derivative_v * binormal.y;
    let derivative_z = derivative_h * tangent.z + derivative_u * normal.z + derivative_v * binormal.z;
    let field_scale = max(length(field.xyz), 1.0e-12);
    let taylor_residual = length(last_order_field) / field_scale;
    let imaginary_residual = length(imaginary_field) / field_scale;
    let spectral_tail_residual = length(tail_field) / field_scale;
    let transverse_ratio = length(vec2<f32>(u, v)) / max(eq_params.line_limit, 1.0);
    let output_base = target_index * OUTPUT_ROWS_PER_BLOCK;

    output[output_base] = field;
    output[output_base + 1u] = vec4<f32>(
        taylor_residual,
        imaginary_residual,
        spectral_tail_residual,
        transverse_ratio,
    );
    output[output_base + 2u] = vec4<f32>(derivative_x, 0.0);
    output[output_base + 3u] = vec4<f32>(derivative_y, 0.0);
    output[output_base + 4u] = vec4<f32>(derivative_z, 0.0);
    output[output_base + 5u] = vec4<f32>(h, u, v, f32(eq_params.segment_id));
    output[output_base + 6u] = vec4<f32>(
        field.w,
        independent_potential,
        f32(eq_params.evaluate_dual_certificate),
        elapsed,
    );
    output[output_base + 7u] = vec4<f32>(probe_pos, f32(eq_params.target_count));
    output[output_base + 8u] = vec4<f32>(eq_params.line_origin, 0.0);
}
