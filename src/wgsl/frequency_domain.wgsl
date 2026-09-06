// Frequency-domain spatial kernel used by the known-trajectory evaluator.
//
// The source pass constructs rho_hat(k) from the uploaded density cells. The
// runtime forward path integrates each continuous radial cell in WGSL; the
// planning/sensitivity paths may use their own unit point records. The
// evaluator then constructs the complete Fourier-Laplace characteristic
// T_gamma(s,k) before applying the Newton multiplier and reciprocal reduction.

const PI: f32 = 3.141592653589793;
const WAVE_VECTOR_COUNT: u32 = 64u;
const SPECTRUM_STRIDE: u32 = WAVE_VECTOR_COUNT;
const DENSITY_STRIDE: u32 = WAVE_VECTOR_COUNT;
const OUTPUT_ROWS_PER_BLOCK: u32 = 11u;
const PLANNING_VOXEL_COUNT: u32 = 56u;
const PLANNING_VOXEL_BANK_COUNT: u32 = 28u;
const PLANNING_OUTPUT_PREFIX_VEC4: u32 = 90112u;

struct FrequencyDomainParams {
    g_const: f32,
    source_count: u32,
    quadrature_count: u32,
    spectrum_slot: u32,
    target_count: u32,
    target_offset: u32,
    inversion_mode: u32,
    source_layout: u32,
    input_base: u32,
    // Keep every member scalar so the storage-array stride is exactly the
    // 48-byte layout written by the Rust dispatch code. A vec3 here would be
    // aligned to 16 bytes by some WebGPU backends and move the Laplace scalar
    // to a different offset, producing zero/garbage observation frequencies.
    trajectory_origin_x: f32,
    trajectory_origin_y: f32,
    trajectory_origin_z: f32,
    laplace_frequency: f32,
};

struct DensityModeBuffer {
    records: array<vec4<f32>, 544>,
};

struct SpectrumSample {
    wave_vector_xy: vec2<f32>,
    wave_vector_z_weight: vec2<f32>,
    density_spectrum: vec2<f32>,
    reserved: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> parameters: array<FrequencyDomainParams>;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quadrature: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<SpectrumSample>;
@group(0) @binding(4) var<storage, read_write> result_buffer: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> density_spectra: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> density_metadata: DensityModeBuffer;
@group(0) @binding(9) var<storage, read> trajectory_samples: array<vec4<f32>>;

var<workgroup> complex_reduction: array<vec2<f32>, 256>;
var<workgroup> field_reduction: array<vec4<f32>, 64>;
var<workgroup> jacobian_x_reduction: array<vec4<f32>, 64>;
var<workgroup> jacobian_y_reduction: array<vec4<f32>, 64>;
var<workgroup> jacobian_z_reduction: array<vec4<f32>, 64>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

const GAUSS_NODE_0: f32 = -0.8611363116;
const GAUSS_NODE_1: f32 = -0.3399810436;
const GAUSS_NODE_2: f32 = 0.3399810436;
const GAUSS_NODE_3: f32 = 0.8611363116;
const GAUSS_WEIGHT_0: f32 = 0.3478548451;
const GAUSS_WEIGHT_1: f32 = 0.6521451549;
const GAUSS_WEIGHT_2: f32 = 0.6521451549;
const GAUSS_WEIGHT_3: f32 = 0.3478548451;

fn volume_cell_spectrum(k: vec3<f32>, cell_index: u32) -> vec2<f32> {
    let angular = sources[cell_index * 2u];
    let radial = sources[cell_index * 2u + 1u];
    let inner = radial.x;
    let outer = max(radial.y, inner);
    let density = max(radial.z, 0.0);
    let half_width = 0.5 * (outer - inner);
    let midpoint = 0.5 * (outer + inner);
    if half_width <= 0.0 || density <= 0.0 || angular.w <= 0.0 { return vec2<f32>(0.0); }
    let nodes = array<f32, 4>(GAUSS_NODE_0, GAUSS_NODE_1, GAUSS_NODE_2, GAUSS_NODE_3);
    let weights = array<f32, 4>(GAUSS_WEIGHT_0, GAUSS_WEIGHT_1, GAUSS_WEIGHT_2, GAUSS_WEIGHT_3);
    var result = vec2<f32>(0.0);
    for (var sample = 0u; sample < 4u; sample += 1u) {
        let radius = midpoint + half_width * nodes[sample];
        let weight = density * angular.w * radius * radius * half_width * weights[sample];
        let phase = -dot(k, angular.xyz * radius);
        result += weight * vec2<f32>(cos(phase), sin(phase));
    }
    return result;
}

fn source_spectrum(
    k: vec3<f32>,
    source_index: u32,
    source_layout: u32,
    inversion_mode: u32,
) -> vec2<f32> {
    // Planning records reuse the padding word at offset 28 for a global
    // output offset, so only the runtime forward operator may select the
    // volume-cell layout.
    if inversion_mode != 3u && source_layout == 1u {
        return volume_cell_spectrum(k, source_index);
    }
    let source = sources[source_index];
    let phase = -dot(k, source.xyz);
    return source.w * vec2<f32>(cos(phase), sin(phase));
}

fn wave_vector(sample_index: u32) -> vec3<f32> {
    return quadrature[sample_index].xyz;
}

fn reduce_complex(lane: u32) -> vec2<f32> {
    var stride = 64u;
    loop {
        if stride == 0u { break; }
        if lane < stride {
            complex_reduction[lane] += complex_reduction[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return complex_reduction[0];
}

// Complete discrete trajectory characteristic from equations (143) and (184).
// The w component is the absolute physical sample time. Composite-trapezoid
// weights integrate the whole known orbit; shifting by the first uploaded time
// would multiply the transform by exp(s*t0) and would no longer be equation (143).
fn trajectory_characteristic(
    k: vec3<f32>,
    p: FrequencyDomainParams,
    laplace_frequency: f32,
    time_lane: u32,
) -> vec2<f32> {
    var result = vec2<f32>(0.0);
    let count = p.target_count;
    if count == 0u { return result; }
    for (var sample_index = time_lane; sample_index < count; sample_index += 4u) {
        let current = trajectory_samples[p.input_base + p.target_offset + sample_index];
        var previous = current;
        var next = current;
        if sample_index > 0u {
            previous = trajectory_samples[p.input_base + p.target_offset + sample_index - 1u];
        }
        if sample_index + 1u < count {
            next = trajectory_samples[p.input_base + p.target_offset + sample_index + 1u];
        }
        let left_dt = max(current.w - previous.w, 0.0);
        let right_dt = max(next.w - current.w, 0.0);
        // Composite trapezoid weights for the discrete form of (143).  Both
        // endpoints carry half of their adjacent interval; using a full first
        // interval and a zero final interval biases T_gamma toward t_0.
        var weight = 0.5 * right_dt;
        if count == 1u {
            weight = 1.0;
        } else if sample_index == 0u {
            weight = 0.5 * right_dt;
        } else if sample_index + 1u == count {
            weight = 0.5 * left_dt;
        } else {
            weight = 0.5 * (left_dt + right_dt);
        }
        let phase = dot(k, current.xyz);
        let attenuation = exp(-laplace_frequency * current.w);
        result += weight * attenuation * vec2<f32>(cos(phase), sin(phase));
    }
    return result;
}

#ifdef FREQUENCY_DOMAIN_SOURCE
@compute @workgroup_size(128, 1, 1)
fn assemble_density_spectrum(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let sample_index = workgroup_id.x;
    let trajectory_index = workgroup_id.y;
    let p = parameters[trajectory_index];
    let k = wave_vector(sample_index);
    var sum = vec2<f32>(0.0);
    for (var source_index = lane; source_index < p.source_count; source_index += 128u) {
        sum += source_spectrum(k, source_index, p.source_layout, p.inversion_mode);
    }
    complex_reduction[lane] = sum;
    workgroupBarrier();
    let density = reduce_complex(lane);
    if lane == 0u {
        let destination = (p.spectrum_slot - 1u) * DENSITY_STRIDE + sample_index;
        density_spectra[destination] = vec4<f32>(density, 0.0, quadrature[sample_index].w);
    }
}

@compute @workgroup_size(128, 1, 1)
fn assemble_voxel_density_spectrum(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let sample_index = workgroup_id.x;
    let trajectory_index = workgroup_id.y;
    let voxel_index = workgroup_id.z;
    let p = parameters[trajectory_index];
    let group = density_metadata.records[voxel_index];
    let source_begin = u32(group.x);
    let source_end = source_begin + u32(group.y);
    let k = wave_vector(sample_index);
    var sum = vec2<f32>(0.0);
    for (var source_index = source_begin + lane; source_index < source_end; source_index += 128u) {
        sum += source_spectrum(k, source_index, p.source_layout, p.inversion_mode);
    }
    complex_reduction[lane] = sum;
    workgroupBarrier();
    let density = reduce_complex(lane);
    if lane == 0u {
        let destination = ((p.spectrum_slot - 1u) * PLANNING_VOXEL_COUNT + voxel_index)
            * DENSITY_STRIDE + sample_index;
        density_spectra[destination] = vec4<f32>(density, 0.0, quadrature[sample_index].w);
    }
}
#endif

#ifdef FREQUENCY_DOMAIN_SPECTRUM
@compute @workgroup_size(64, 1, 1)
fn publish_density_spectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample_index = id.x;
    let trajectory_index = id.y;
    if sample_index >= WAVE_VECTOR_COUNT { return; }
    let p = parameters[trajectory_index];
    let packed = density_spectra[(p.spectrum_slot - 1u) * DENSITY_STRIDE + sample_index];
    let k = wave_vector(sample_index);
    var result: SpectrumSample;
    result.wave_vector_xy = k.xy;
    result.wave_vector_z_weight = vec2<f32>(k.z, packed.w);
    result.density_spectrum = packed.xy;
    result.reserved = vec2<f32>(0.0);
    spectrum[(p.spectrum_slot - 1u) * SPECTRUM_STRIDE + sample_index] = result;
}

@compute @workgroup_size(64, 1, 1)
fn publish_voxel_density_spectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample_index = id.x;
    let trajectory_index = id.y;
    let voxel_index = id.z;
    if sample_index >= WAVE_VECTOR_COUNT || voxel_index >= PLANNING_VOXEL_COUNT { return; }
    let p = parameters[trajectory_index];
    let source_index = ((p.spectrum_slot - 1u) * PLANNING_VOXEL_COUNT + voxel_index)
        * DENSITY_STRIDE + sample_index;
    let packed = density_spectra[source_index];
    let bank_voxel = voxel_index % PLANNING_VOXEL_BANK_COUNT;
    let bank_index = ((p.spectrum_slot - 1u) * PLANNING_VOXEL_BANK_COUNT + bank_voxel)
        * SPECTRUM_STRIDE + sample_index;
    if voxel_index < PLANNING_VOXEL_BANK_COUNT {
        density_spectra[u32(density_metadata.records[112].x) + bank_index] = packed;
    } else {
        result_buffer[PLANNING_OUTPUT_PREFIX_VEC4 + bank_index] = packed;
    }
}

@compute @workgroup_size(64, 1, 1)
fn combine_density_spectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample_index = id.x;
    let trajectory_index = id.y;
    if sample_index >= WAVE_VECTOR_COUNT { return; }
    let p = parameters[trajectory_index];
    var density = vec2<f32>(0.0);
    for (var voxel = 0u; voxel < PLANNING_VOXEL_COUNT; voxel += 1u) {
        let bank_voxel = voxel % PLANNING_VOXEL_BANK_COUNT;
        let bank_index = ((p.spectrum_slot - 1u) * PLANNING_VOXEL_BANK_COUNT + bank_voxel)
            * SPECTRUM_STRIDE + sample_index;
        var packed: vec4<f32>;
        if voxel < PLANNING_VOXEL_BANK_COUNT {
            packed = density_spectra[u32(density_metadata.records[112].x) + bank_index];
        } else {
            packed = result_buffer[PLANNING_OUTPUT_PREFIX_VEC4 + bank_index];
        }
        density += density_metadata.records[PLANNING_VOXEL_COUNT + voxel].x * packed.xy;
    }
    let k = wave_vector(sample_index);
    var result: SpectrumSample;
    result.wave_vector_xy = k.xy;
    result.wave_vector_z_weight = vec2<f32>(k.z, quadrature[sample_index].w);
    result.density_spectrum = density;
    result.reserved = vec2<f32>(0.0);
    spectrum[(p.spectrum_slot - 1u) * SPECTRUM_STRIDE + sample_index] = result;
}
#endif

#ifdef FREQUENCY_DOMAIN_EVALUATOR
@compute @workgroup_size(256, 1, 1)
fn evaluate_trajectory_field(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let trajectory_index = workgroup_id.z;
    let p = parameters[trajectory_index];
    let observation_index = workgroup_id.y;
    // Each output row is an independent equation-(184) observation at a
    // positive Laplace frequency.  Every row integrates the complete known
    // trajectory; the row index is not a pointwise spatial target.
    var normalized_frequency = 0.0;
    if p.target_count > 1u {
        normalized_frequency = f32(observation_index) / f32(p.target_count - 1u);
    }
    let observation_frequency = p.laplace_frequency * (1.0 + 7.0 * normalized_frequency);
    // 64 wave vectors x 4 disjoint time lanes. All 256 lanes participate in
    // both barriers, including trajectories shorter than four nodes.
    let wave_index = lane % WAVE_VECTOR_COUNT;
    let time_lane = lane / WAVE_VECTOR_COUNT;
    let wave = spectrum[(p.spectrum_slot - 1u) * SPECTRUM_STRIDE + wave_index];
    let wave_k = vec3<f32>(wave.wave_vector_xy, wave.wave_vector_z_weight.x);
    complex_reduction[lane] = trajectory_characteristic(wave_k, p, observation_frequency, time_lane);
    workgroupBarrier();
    var field = vec4<f32>(0.0);
    var jacobian_x = vec3<f32>(0.0);
    var jacobian_y = vec3<f32>(0.0);
    var jacobian_z = vec3<f32>(0.0);
    // The vector field and Jacobian below are the frequency-domain
    // integral.  The scalar potential is accumulated from the real part of
    // the same reciprocal-space product; no direct point-source fallback is
    // mixed into the frequency-domain result.
    var positive_potential = 0.0;
    let normalization = p.g_const * 4.0 * PI / ((2.0 * PI) * (2.0 * PI) * (2.0 * PI));
    if lane < WAVE_VECTOR_COUNT {
        let sample_index = lane;
        let density_mode = spectrum[(p.spectrum_slot - 1u) * SPECTRUM_STRIDE + sample_index];
        let k = vec3<f32>(density_mode.wave_vector_xy, density_mode.wave_vector_z_weight.x);
        // Midpoint quadrature never samples k=0; preserve the exact multiplier.
        let k_squared = dot(k, k);
        // Compute the complete k-dependent Fourier--Laplace characteristic
        // over every uploaded trajectory sample before multiplying by rho_hat.
        let trajectory_factor = complex_reduction[lane] + complex_reduction[lane + 64u]
            + complex_reduction[lane + 128u] + complex_reduction[lane + 192u];
        // rho_hat(k) T_gamma(s,k), the two scalar factors immediately before
        // d^3k in equation (184). Keep this binding local to the evaluator so
        // no shader variant can accidentally reference a source-pass symbol.
        let rho_hat_times_trajectory = complex_mul(
            density_mode.density_spectrum,
            trajectory_factor,
        );
        let coefficient = normalization * density_mode.wave_vector_z_weight.y / k_squared;
        let acceleration = -coefficient * rho_hat_times_trajectory.y * k;
        field += vec4<f32>(acceleration, 0.0);
        positive_potential += coefficient * rho_hat_times_trajectory.x;
        let hessian_scale = -coefficient * rho_hat_times_trajectory.x;
        jacobian_x += hessian_scale * k * k.x;
        jacobian_y += hessian_scale * k * k.y;
        jacobian_z += hessian_scale * k * k.z;
    }
    if lane < WAVE_VECTOR_COUNT {
    field.w = positive_potential;
    field_reduction[lane] = field;
    jacobian_x_reduction[lane] = vec4<f32>(jacobian_x, 0.0);
    jacobian_y_reduction[lane] = vec4<f32>(jacobian_y, 0.0);
    jacobian_z_reduction[lane] = vec4<f32>(jacobian_z, 0.0);
    }
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if lane < stride {
            field_reduction[lane] += field_reduction[lane + stride];
            jacobian_x_reduction[lane] += jacobian_x_reduction[lane + stride];
            jacobian_y_reduction[lane] += jacobian_y_reduction[lane + stride];
            jacobian_z_reduction[lane] += jacobian_z_reduction[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if lane != 0u { return; }
    let output_index = p.target_offset + observation_index;
    let trajectory_record = trajectory_samples[p.input_base + output_index];
    if p.inversion_mode == 1u {
        result_buffer[output_index] = field_reduction[0];
        return;
    }
    let base = output_index * OUTPUT_ROWS_PER_BLOCK;
    result_buffer[base] = field_reduction[0];
    result_buffer[base + 1u] = vec4<f32>(0.0);
    result_buffer[base + 2u] = jacobian_x_reduction[0];
    result_buffer[base + 3u] = jacobian_y_reduction[0];
    result_buffer[base + 4u] = jacobian_z_reduction[0];
    result_buffer[base + 5u] = vec4<f32>(observation_frequency, 0.0, 0.0, f32(p.spectrum_slot));
    result_buffer[base + 6u] = vec4<f32>(field_reduction[0].w, 0.0, 0.0, 0.0);
    result_buffer[base + 7u] = vec4<f32>(trajectory_record.xyz, f32(p.target_count));
    result_buffer[base + 8u] = vec4<f32>(
        p.trajectory_origin_x,
        p.trajectory_origin_y,
        p.trajectory_origin_z,
        0.0,
    );
    result_buffer[base + 9u] = vec4<f32>(0.0);
    result_buffer[base + 10u] = vec4<f32>(0.0);
}
#endif
