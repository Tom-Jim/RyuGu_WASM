// Equation (106) complex-frequency pipeline.
//
// `assemble_spectrum` builds one cached frequency grid for the current local
// reference element using a fixed half-line quadrature LUT. `evaluate_field`
// performs the Bromwich sum and Cartesian translation correction at 1..=8
// predicted anchors along that element in parallel. This keeps the expensive
// density/Laplace assembly segment-scoped instead of rebuilding it per frame.

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
    body_velocity: vec3<f32>,
    block_dt: f32,
    batch_count: u32,
    _padding1: vec3<f32>,
};

struct BlockFrame {
    origin: vec3<f32>,
    direction: vec3<f32>,
    velocity: vec3<f32>,
};

struct SpectrumSample {
    acceleration_x: vec2<f32>,
    acceleration_y: vec2<f32>,
    acceleration_z: vec2<f32>,
    potential: vec2<f32>,
};

struct PointFieldDifferential {
    field: vec4<f32>,
    jacobian_x: vec3<f32>,
    _padding_x: f32,
    jacobian_y: vec3<f32>,
    _padding_y: f32,
    jacobian_z: vec3<f32>,
    _padding_z: f32,
};

@group(0) @binding(0) var<uniform> params: Eq106Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quadrature: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<SpectrumSample>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> toroidal_tensor: array<f32>;
@group(0) @binding(6) var<storage, read_write> line_samples: array<vec4<f32>>;

const TOROIDAL_X_MIN: f32 = -10.0;
const TOROIDAL_X_MAX: f32 = 8.0;
const TOROIDAL_SEGMENT_STEP: f32 = 1.5;
const TOROIDAL_SEGMENT_COUNT: u32 = 12u;
const TOROIDAL_DEGREE: u32 = 12u;
const TOROIDAL_COEFFICIENT_COUNT: u32 = 13u;
const TOROIDAL_MODE_COUNT: u32 = 17u;
const OUTPUT_ROWS_PER_BLOCK: u32 = 9u;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denominator = max(dot(b, b), 1.0e-20);
    return vec2<f32>(
        (a.x * b.x + a.y * b.y) / denominator,
        (a.y * b.x - a.x * b.y) / denominator,
    );
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
fn assemble_line_samples(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let quadrature_index = global_id.x;
    if quadrature_index >= params.quadrature_count {
        return;
    }
    let sample = quadrature[quadrature_index];
    let observer = params.line_origin + sample.x * params.line_direction;
    var packed = vec4<f32>(0.0);
    for (var source_index = 0u; source_index < params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let displacement = source.xyz - observer;
        let distance2 = max(dot(displacement, displacement), 1.0e-8);
        let distance = sqrt(distance2);
        let mass_scale = params.g_const * source.w;
        packed += vec4<f32>(
            mass_scale * displacement / (distance2 * distance),
            mass_scale / distance,
        );
    }
    line_samples[quadrature_index] = packed * sample.y;
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

    for (var quadrature_index = 0u; quadrature_index < params.quadrature_count; quadrature_index += 1u) {
        let sample = quadrature[quadrature_index];
        let packed = line_samples[quadrature_index];
        let phase = laplace_phase(omega, sample.x);
        result.acceleration_x += phase * packed.x;
        result.acceleration_y += phase * packed.y;
        result.acceleration_z += phase * packed.z;
        result.potential += phase * packed.w;
    }
    spectrum[frequency_index] = result;
    if frequency_index == 0u {
        // Fix the additive potential gauge at the exact direct potential of
        // the current line origin. This value persists until the reference
        // line is rebuilt, preventing truncated-spectrum constants from being
        // concatenated across curved-arc segments.
        let line_origin_field = point_field_differential(params.line_origin).field;
        // The toroidal-harmonic cross-check is certification metadata, not a
        // per-anchor force term. Compute it once when the cached spectrum is
        // assembled and carry the scalar in output[5].x. Re-evaluating all
        // modes and all sources inside every real-time anchor was the main
        // reason Eq.106 could hold the browser at 2-3 FPS.
        let toroidal = toroidal_potential(params.line_origin);
        let toroidal_scale = max(abs(line_origin_field.w), 1.0e-12);
        let toroidal_residual = abs(toroidal.x - line_origin_field.w) / toroidal_scale;
        output[5] = vec4<f32>(toroidal_residual, toroidal.y, 0.0, line_origin_field.w);
    }
}

fn point_field_differential(observer: vec3<f32>) -> PointFieldDifferential {
    var result: PointFieldDifferential;
    result.field = vec4<f32>(0.0);
    result.jacobian_x = vec3<f32>(0.0);
    result._padding_x = 0.0;
    result.jacobian_y = vec3<f32>(0.0);
    result._padding_y = 0.0;
    result.jacobian_z = vec3<f32>(0.0);
    result._padding_z = 0.0;
    for (var source_index = 0u; source_index < params.source_count; source_index += 1u) {
        let source = sources[source_index];
        let displacement = source.xyz - observer;
        let distance2 = max(dot(displacement, displacement), 1.0e-8);
        let distance = sqrt(distance2);
        let scale = params.g_const * source.w;
        let inverse_r3_scale = scale / (distance2 * distance);
        let three_inverse_r5_scale = 3.0 * inverse_r3_scale / distance2;
        result.field += vec4<f32>(inverse_r3_scale * displacement, scale / distance);
        result.jacobian_x += three_inverse_r5_scale * displacement.x * displacement
            - vec3<f32>(inverse_r3_scale, 0.0, 0.0);
        result.jacobian_y += three_inverse_r5_scale * displacement.y * displacement
            - vec3<f32>(0.0, inverse_r3_scale, 0.0);
        result.jacobian_z += three_inverse_r5_scale * displacement.z * displacement
            - vec3<f32>(0.0, 0.0, inverse_r3_scale);
    }
    return result;
}

fn rotating_acceleration(position: vec3<f32>, velocity: vec3<f32>) -> vec3<f32> {
    let spin_axis = normalize(vec3<f32>(-0.043, -0.914, 0.405));
    let angular_velocity = spin_axis * (2.0 * 3.141592653589793 / (7.63 * 3600.0));
    let gravity = point_field_differential(position).field.xyz;
    return gravity
        - 2.0 * cross(angular_velocity, velocity)
        - cross(angular_velocity, cross(angular_velocity, position));
}

fn block_frame(block_index: u32) -> BlockFrame {
    // Match the CPU's fixed 12-substep cadence instead of extrapolating all
    // accelerated anchors from one initial acceleration. The latter diverges
    // rapidly at 8x and makes the CPU consume Hessians centered on the wrong
    // trajectory. This bounded loop remains inside the existing compute pass.
    // These anchors only center the CPU's conservative Hessian evaluation;
    // the authoritative integrator still uses 12 substeps. Four predictor
    // substeps are sufficient on the multi-hour orbital time scale and avoid
    // repeating hundreds of full source traversals for an 8x batch.
    const SUBSTEPS: u32 = 4u;
    let substep_dt = params.block_dt / f32(SUBSTEPS);
    var predicted_position = params.probe_pos;
    var predicted_velocity = params.body_velocity;
    let total_substeps = block_index * SUBSTEPS;
    for (var substep = 0u; substep < total_substeps; substep += 1u) {
        let acceleration_start = rotating_acceleration(predicted_position, predicted_velocity);
        let half_velocity = predicted_velocity + 0.5 * acceleration_start * substep_dt;
        predicted_position += half_velocity * substep_dt;
        let acceleration_end = rotating_acceleration(predicted_position, half_velocity);
        predicted_velocity = half_velocity + 0.5 * acceleration_end * substep_dt;
    }
    var frame: BlockFrame;
    frame.origin = predicted_position;
    frame.direction = normalize(predicted_velocity);
    frame.velocity = predicted_velocity;
    if length(predicted_velocity) <= 1.0e-8 {
        frame.direction = params.line_direction;
    }
    return frame;
}

@compute @workgroup_size(1, 1, 1)
fn evaluate_field(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_index = global_id.x;
    if block_index >= params.batch_count {
        return;
    }
    let frame = block_frame(block_index);
    let probe_pos = frame.origin;
    let line_origin = params.line_origin;
    let line_direction = params.line_direction;
    let output_base = block_index * OUTPUT_ROWS_PER_BLOCK;
    let relative = probe_pos - line_origin;
    let h = max(dot(relative, line_direction), 0.0);
    let reference_point = line_origin + h * line_direction;
    let frequency_count = 2u * params.half_count + 1u;
    var acceleration = vec3<f32>(0.0);
    var acceleration_origin = vec3<f32>(0.0);
    var acceleration_minus = vec3<f32>(0.0);
    var acceleration_plus = vec3<f32>(0.0);
    var imaginary_acceleration = vec3<f32>(0.0);
    var integrated_longitudinal_acceleration = vec2<f32>(0.0);
    let derivative_step = max(1.0, 0.01 / max(params.sigma, 1.0e-6));
    // The inverse transform has a one-sided half-line endpoint at h=0. Do not
    // center a derivative across that boundary: the endpoint reconstruction
    // weight differs from interior samples and would appear as false curvature.
    let h_minus = select(h - derivative_step, h, h < derivative_step);
    let h_plus = h + derivative_step;
    for (var frequency_index = 0u; frequency_index < frequency_count; frequency_index += 1u) {
        let signed_index = i32(frequency_index) - i32(params.half_count);
        let omega = f32(signed_index) * params.omega_step;
        let phase = bromwich_phase(omega, h);
        let phase_origin = bromwich_phase(omega, 0.0);
        let phase_minus = bromwich_phase(omega, h_minus);
        let phase_plus = bromwich_phase(omega, h_plus);
        let laplace_rate = vec2<f32>(params.sigma, omega);
        let sample = spectrum[frequency_index];
        let x = complex_mul(sample.acceleration_x, phase);
        let y = complex_mul(sample.acceleration_y, phase);
        let z = complex_mul(sample.acceleration_z, phase);
        acceleration += vec3<f32>(x.x, y.x, z.x);
        acceleration_origin += vec3<f32>(
            complex_mul(sample.acceleration_x, phase_origin).x,
            complex_mul(sample.acceleration_y, phase_origin).x,
            complex_mul(sample.acceleration_z, phase_origin).x,
        );
        acceleration_minus += vec3<f32>(
            complex_mul(sample.acceleration_x, phase_minus).x,
            complex_mul(sample.acceleration_y, phase_minus).x,
            complex_mul(sample.acceleration_z, phase_minus).x,
        );
        acceleration_plus += vec3<f32>(
            complex_mul(sample.acceleration_x, phase_plus).x,
            complex_mul(sample.acceleration_y, phase_plus).x,
            complex_mul(sample.acceleration_z, phase_plus).x,
        );
        imaginary_acceleration += vec3<f32>(x.y, y.y, z.y);
        let longitudinal_sample = vec2<f32>(
            dot(
                vec3<f32>(
                    sample.acceleration_x.x,
                    sample.acceleration_y.x,
                    sample.acceleration_z.x,
                ),
                line_direction,
            ),
            dot(
                vec3<f32>(
                    sample.acceleration_x.y,
                    sample.acceleration_y.y,
                    sample.acceleration_z.y,
                ),
                line_direction,
            ),
        );
        let integrated_phase =
            complex_div(phase - vec2<f32>(1.0, 0.0), laplace_rate);
        integrated_longitudinal_acceleration +=
            complex_mul(longitudinal_sample, integrated_phase);
    }
    let endpoint_factor = select(1.0, 2.0, h <= 1.0e-5);
    let inversion_scale = endpoint_factor * params.omega_step / (2.0 * 3.141592653589793);
    let spectral_acceleration = acceleration * inversion_scale;
    let spectral_acceleration_origin = acceleration_origin
        * 2.0 * params.omega_step / (2.0 * 3.141592653589793);
    let minus_endpoint_factor = select(1.0, 2.0, h_minus <= 1.0e-5);
    let plus_endpoint_factor = select(1.0, 2.0, h_plus <= 1.0e-5);
    let spectral_acceleration_minus = acceleration_minus
        * minus_endpoint_factor * params.omega_step / (2.0 * 3.141592653589793);
    let spectral_acceleration_plus = acceleration_plus
        * plus_endpoint_factor * params.omega_step / (2.0 * 3.141592653589793);
    let interior_inversion_scale = params.omega_step / (2.0 * 3.141592653589793);
    let spectral_potential_change =
        integrated_longitudinal_acceleration.x * interior_inversion_scale;

    // Construct one scalar corrected potential by analytically integrating
    // the already assembled longitudinal Eq.106 acceleration spectrum:
    //
    // U(x) = U_direct(x)
    //      + integral_0^h a_106(tau).line_direction d tau
    //      + U_direct(line_origin) - U_direct(x_ref(h)).
    //
    // The division by s=(sigma+i*omega) damps truncation error. Directly
    // differentiating the truncated potential spectrum multiplies that error
    // by s and is numerically unstable.
    let reference = point_field_differential(reference_point);
    let line_origin_field = point_field_differential(line_origin);
    let reference_minus = point_field_differential(
        line_origin + h_minus * line_direction,
    );
    let reference_plus = point_field_differential(
        line_origin + h_plus * line_direction,
    );
    let actual = point_field_differential(probe_pos);
    let spectral_longitudinal_acceleration =
        dot(spectral_acceleration, line_direction);
    let origin_longitudinal_defect = dot(
        spectral_acceleration_origin - line_origin_field.field.xyz,
        line_direction,
    );
    let reference_longitudinal_acceleration =
        dot(reference.field.xyz, line_direction);
    let correction_acceleration =
        spectral_longitudinal_acceleration
        - reference_longitudinal_acceleration
        - origin_longitudinal_defect;
    let correction_potential =
        spectral_potential_change
        - origin_longitudinal_defect * h
        + output[5].w
        - reference.field.w;

    // A local straight-reference representation must not be concatenated as
    // unrelated affine gauges along a curved orbit. Fade the scalar correction
    // and its derivative to zero in an overlap region before Rust rebuilds the
    // line. Applying the product rule keeps acceleration = grad(U), so segment
    // transitions do not inject Jacobi energy or Eq.(157) residual.
    let line_limit = max(params._padding1.x, 1.0);
    // Leave a broad zero-correction overlap before Rust expires the line at
    // 0.85L. This covers several 1x authoritative frames and the full tail of
    // an 8x predicted batch, so no consumer ever crosses directly from a
    // non-zero old correction to the next line's zero-origin correction.
    let taper_start = 0.35 * line_limit;
    let taper_end = 0.55 * line_limit;
    let taper_span = max(taper_end - taper_start, 1.0);
    let taper_t = clamp((h - taper_start) / taper_span, 0.0, 1.0);
    let taper_smooth = taper_t * taper_t * (3.0 - 2.0 * taper_t);
    let taper = 1.0 - taper_smooth;
    let taper_derivative = select(
        0.0,
        -6.0 * taper_t * (1.0 - taper_t) / taper_span,
        h > taper_start && h < taper_end,
    );
    let tapered_correction_acceleration =
        taper * correction_acceleration + taper_derivative * correction_potential;
    let corrected_acceleration =
        actual.field.xyz + tapered_correction_acceleration * line_direction;
    let corrected_potential = actual.field.w
        + taper * correction_potential;

    let residual_scale = max(length(actual.field.xyz), 1.0e-12);
    // Only the longitudinal Eq.106 defect is applied to the translated field;
    // certify that actual correction rather than unused transverse spectrum
    // components.
    let relative_residual = abs(tapered_correction_acceleration) / residual_scale;
    let imaginary_residual = length(imaginary_acceleration) * inversion_scale / residual_scale;
    // `output[5].x` was written once by assemble_spectrum and is intentionally
    // reused for every anchor in this cached line element.
    let toroidal_residual = output[5].x;
    let valid_fraction = output[5].y;
    // Export both representations. Rust aligns the segmented Eq.106 potential
    // once when a new reference line is assembled and uses that field for
    // Jacobi diagnostics. The global direct potential remains independent and
    // is used only by the Eq. (157) dual-representation residual.
    output[output_base] = vec4<f32>(corrected_acceleration, actual.field.w);
    output[output_base + 1u] = vec4<f32>(relative_residual, imaginary_residual, toroidal_residual, valid_fraction);

    // Differentiate the same scalar Eq.106 correction used above. A centered
    // line derivative avoids multiplying the truncated Bromwich spectrum by
    // s, which is the unstable certification path this pipeline rejects.
    let correction_minus = dot(
        spectral_acceleration_minus - reference_minus.field.xyz,
        line_direction,
    );
    let correction_plus = dot(
        spectral_acceleration_plus - reference_plus.field.xyz,
        line_direction,
    );
    let correction_slope = (correction_plus - correction_minus)
        / max(h_plus - h_minus, 1.0e-6);
    let taper_second_derivative = select(
        0.0,
        (-6.0 + 12.0 * taper_t) / (taper_span * taper_span),
        h > taper_start && h < taper_end,
    );
    let tapered_correction_slope = taper * correction_slope
        + 2.0 * taper_derivative * correction_acceleration
        + taper_second_derivative * correction_potential;
    let correction_row_x = tapered_correction_slope
        * line_direction.x * line_direction;
    let correction_row_y = tapered_correction_slope
        * line_direction.y * line_direction;
    let correction_row_z = tapered_correction_slope
        * line_direction.z * line_direction;
    output[output_base + 2u] = vec4<f32>(actual.jacobian_x + correction_row_x, 0.0);
    output[output_base + 3u] = vec4<f32>(actual.jacobian_y + correction_row_y, 0.0);
    output[output_base + 4u] = vec4<f32>(actual.jacobian_z + correction_row_z, 0.0);
    output[output_base + 5u] = vec4<f32>(frame.velocity, output[output_base + 5u].w);
    output[output_base + 6u] = vec4<f32>(
        corrected_potential,
        actual.field.w,
        spectral_potential_change,
        f32(block_index) * params.block_dt,
    );
    output[output_base + 7u] = vec4<f32>(probe_pos, f32(params.batch_count));
    output[output_base + 8u] = vec4<f32>(line_origin, 0.0);
}
