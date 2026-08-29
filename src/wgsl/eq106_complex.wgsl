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
@group(0) @binding(9) var<storage, read> targets: array<vec4<f32>>;

var<workgroup> eq_params: Eq106Params;

#ifdef EQ106_EVALUATOR
var<workgroup> evaluation_phase: array<vec2<f32>, 129>;
var<workgroup> evaluation_spectral_derivative: array<vec2<f32>, 129>;
var<workgroup> evaluation_omega: array<f32, 129>;
// The evaluator deliberately uses 32 lanes.  Its five vec4 reductions, two
// vec2 reductions, and phase tables then fit under the portable 16 KiB
// workgroup-storage limit (including `eq_params`).
var<workgroup> evaluation_sum: array<vec4<f32>, 32>;
var<workgroup> evaluation_derivative: array<vec4<f32>, 32>;
var<workgroup> evaluation_imaginary: array<vec4<f32>, 32>;
var<workgroup> evaluation_tail: array<vec4<f32>, 32>;
var<workgroup> evaluation_edge: array<vec2<f32>, 32>;
var<workgroup> evaluation_coarse: array<vec4<f32>, 32>;
var<workgroup> evaluation_edge_coarse: array<vec2<f32>, 32>;
#endif
#ifdef EQ106_SOURCE
// Two 128-lane reductions occupy 4 KiB. Together with the evaluator scratch
// and `eq_params`, the module stays below WebGPU's portable 16 KiB limit while
// reducing two Taylor coefficients per barrier sequence.
var<workgroup> source_reduction: array<vec4<f32>, 256>;
#endif

const PI: f32 = 3.141592653589793;
const TAYLOR_MAX_ORDER: u32 = 8u;
const MAX_TAYLOR_COEFFICIENT_COUNT: u32 = 45u;
const QUADRATURE_CAPACITY: u32 = 64u;
const SPECTRUM_FREQUENCY_CAPACITY: u32 = 129u;
const OUTPUT_ROWS_PER_BLOCK: u32 = 11u;
const SELF_FD_DELTAS = array<f32, 5>(0.25, 0.5, 1.0, 2.0, 4.0);
const TARGET_DISPATCH_WIDTH: u32 = 65535u;
const PLANNING_VOXEL_COUNT: u32 = 56u;
const PLANNING_VOXEL_BANK_COUNT: u32 = 28u;
const SOURCE_LANE_COUNT: u32 = 128u;
const SOURCE_REDUCTION_CHUNK: u32 = 2u;
const NUFFT_GRID_SIZE: u32 = 1024u;
const NUFFT_PAIR_COUNT: u32 = 6u;
const COEFFICIENT_A = array<u32, 45>(
    0u,
    1u, 0u,
    2u, 1u, 0u,
    3u, 2u, 1u, 0u,
    4u, 3u, 2u, 1u, 0u,
    5u, 4u, 3u, 2u, 1u, 0u,
    6u, 5u, 4u, 3u, 2u, 1u, 0u,
    7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u,
    8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u,
);
const COEFFICIENT_B = array<u32, 45>(
    0u,
    0u, 1u,
    0u, 1u, 2u,
    0u, 1u, 2u, 3u,
    0u, 1u, 2u, 3u, 4u,
    0u, 1u, 2u, 3u, 4u, 5u,
    0u, 1u, 2u, 3u, 4u, 5u, 6u,
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u,
);
const COEFFICIENT_DEGREE = array<u32, 45>(
    0u,
    1u, 1u,
    2u, 2u, 2u,
    3u, 3u, 3u, 3u,
    4u, 4u, 4u, 4u, 4u,
    5u, 5u, 5u, 5u, 5u, 5u,
    6u, 6u, 6u, 6u, 6u, 6u, 6u,
    7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
    8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u,
);
// 16 candidates * 512 samples * 11 vec4 output rows. Planning reserves this
// prefix for evaluator output and stores the high voxel-spectrum bank after it.
const PLANNING_OUTPUT_PREFIX_VEC4: u32 = 90112u;
const TOROIDAL_MODE_COUNT: u32 = 17u;
const TOROIDAL_SEGMENT_COUNT: u32 = 12u;
const TOROIDAL_DEGREE: u32 = 12u;
const TOROIDAL_X_MIN: f32 = -10.0;
const TOROIDAL_X_MAX: f32 = 8.0;
const TOROIDAL_SEGMENT_STEP: f32 = 1.5;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

#ifdef EQ106_EVALUATOR
struct NufftInterpolation {
    value: vec4<f32>,
    error: vec4<f32>,
};

fn type2_nufft_interpolate(
    coefficient: u32,
    pair: u32,
    h: f32,
) -> NufftInterpolation {
    let period = 2.0 * PI / max(eq_params.omega_step, 1.0e-12);
    let coordinate = fract(max(h, 0.0) / period) * f32(NUFFT_GRID_SIZE);
    let index1 = u32(floor(coordinate)) % NUFFT_GRID_SIZE;
    let index0 = (index1 + NUFFT_GRID_SIZE - 1u) % NUFFT_GRID_SIZE;
    let index2 = (index1 + 1u) % NUFFT_GRID_SIZE;
    let index3 = (index1 + 2u) % NUFFT_GRID_SIZE;
    let fraction = coordinate - floor(coordinate);
    let base = u32(density_modes.records[112].z)
        + ((eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT
            * NUFFT_PAIR_COUNT + coefficient * NUFFT_PAIR_COUNT + pair) * NUFFT_GRID_SIZE;
    let p0 = line_samples[base + index0];
    let p1 = line_samples[base + index1];
    let p2 = line_samples[base + index2];
    let p3 = line_samples[base + index3];
    // Spell out the vector weight for browser WGSL implementations that do
    // not yet accept the vector-vector-scalar overload of `mix`.
    let linear = mix(p1, p2, vec4<f32>(fraction));
    let fraction2 = fraction * fraction;
    let fraction3 = fraction2 * fraction;
    let cubic = 0.5 * (
        2.0 * p1
        + (-p0 + p2) * fraction
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * fraction2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * fraction3
    );
    var result: NufftInterpolation;
    result.value = cubic;
    result.error = abs(cubic - linear);
    return result;
}
#endif

fn coefficient_index(a: u32, b: u32) -> u32 {
    let degree = a + b;
    return degree * (degree + 1u) / 2u + b;
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

#ifdef EQ106_SOURCE
fn radial_power_coefficient(
    coefficients: ptr<function, array<vec2<f32>, 45>>,
    a: u32,
    b: u32,
) -> vec2<f32> {
    if a + b > TAYLOR_MAX_ORDER {
        return vec2<f32>(0.0);
    }
    return (*coefficients)[coefficient_index(a, b)];
}

// Build r^-1 and r^-3 together. Both powers share the same sparse bivariate
// polynomial and predecessor walk, so a paired recurrence avoids constructing
// and traversing two 45-element function-local arrays for every volume source.
// The x/y components are exactly the alpha=-1/2 and alpha=-3/2 recurrences
// used by the independent CPU reference tests.
fn build_inverse_radial_power_series(
    x0: f32,
    x10: f32,
    x01: f32,
) -> array<vec2<f32>, 45> {
    var result: array<vec2<f32>, 45>;
    for (var index = 0u; index < active_coefficient_count(); index += 1u) {
        result[index] = vec2<f32>(0.0);
    }
    let inverse_radius = inverseSqrt(x0);
    result[0] = vec2<f32>(inverse_radius, inverse_radius / x0);
    for (var degree = 1u; degree <= min(eq_params.taylor_order, TAYLOR_MAX_ORDER); degree += 1u) {
        for (var b = 0u; b <= degree; b += 1u) {
            let a = degree - b;
            var numerator = vec2<f32>(0.0);
            let degree_f = f32(degree);
            let linear_factor = vec2<f32>(0.5 - degree_f, -0.5 - degree_f);
            if a >= 1u {
                numerator += linear_factor * x10 * radial_power_coefficient(&result, a - 1u, b);
            }
            if b >= 1u {
                numerator += linear_factor * x01 * radial_power_coefficient(&result, a, b - 1u);
            }
            let quadratic_factor = vec2<f32>(1.0 - degree_f, -1.0 - degree_f);
            if a >= 2u {
                numerator += quadratic_factor * radial_power_coefficient(&result, a - 2u, b);
            }
            if b >= 2u {
                numerator += quadratic_factor * radial_power_coefficient(&result, a, b - 2u);
            }
            result[coefficient_index(a, b)] = numerator / (degree_f * x0);
        }
    }
    return result;
}

fn accumulate_source_range(
    observer: vec3<f32>,
    normal: vec3<f32>,
    binormal: vec3<f32>,
    source_begin: u32,
    source_end: u32,
    lane: u32,
) -> array<vec4<f32>, 45> {
    let coefficient_count = active_coefficient_count();
    // Store the Taylor jet in dimensionless transverse coordinates. This
    // keeps high-order coefficients near the scale of the field instead of
    // summing metre^-8 values in f32. The source-independent coordinate scale
    // is applied once after lane reduction instead of for every source below.
    var packed: array<vec4<f32>, 45>;
    for (var index = 0u; index < MAX_TAYLOR_COEFFICIENT_COUNT; index += 1u) {
        packed[index] = vec4<f32>(0.0);
    }
    for (var source_index = source_begin + lane; source_index < source_end; source_index += SOURCE_LANE_COUNT) {
        let source = sources[source_index];
        let r0 = source.xyz - observer;
        let x0 = max(dot(r0, r0), 1.0e-8);
        let x10 = -2.0 * dot(r0, normal);
        let x01 = -2.0 * dot(r0, binormal);
        let inverse_radial_powers = build_inverse_radial_power_series(x0, x10, x01);
        let source_scale = eq_params.g_const * source.w;

        for (var coefficient = 0u; coefficient < coefficient_count; coefficient += 1u) {
            let a = COEFFICIENT_A[coefficient];
            let b = COEFFICIENT_B[coefficient];
            var previous_u = 0.0;
            var previous_v = 0.0;
            if a > 0u {
                previous_u = inverse_radial_powers[coefficient_index(a - 1u, b)].y;
            }
            if b > 0u {
                previous_v = inverse_radial_powers[coefficient_index(a, b - 1u)].y;
            }
            let inverse_r = inverse_radial_powers[coefficient];
            packed[coefficient] += source_scale * vec4<f32>(
                r0 * inverse_r.y - normal * previous_u - binormal * previous_v,
                inverse_r.x,
            );
        }
    }
    return packed;
}

fn reduce_and_store_taylor_jet(
    packed: ptr<function, array<vec4<f32>, 45>>,
    coefficient_count: u32,
    lane: u32,
    destination_base: u32,
    quadrature_weight: f32,
) {
    let source_radius = 2.0 / max(eq_params.sigma, 1.0e-12);
    let transverse_scale = max(length(eq_params.line_origin) - source_radius, 1.0);
    // Keep every workgroup on the same barrier schedule. The active
    // coefficient count must not control a loop containing barriers on
    // Dawn/Tint, even though it is identical in every lane. Inactive
    // coefficients remain zero and are skipped at the store.
    for (var chunk = 0u; chunk < MAX_TAYLOR_COEFFICIENT_COUNT; chunk += SOURCE_REDUCTION_CHUNK) {
        for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
            let coefficient = chunk + slot;
            var value = vec4<f32>(0.0);
            if coefficient < coefficient_count {
                value = (*packed)[coefficient];
            }
            source_reduction[slot * SOURCE_LANE_COUNT + lane] = value;
        }
        workgroupBarrier();
        // Keep the barrier schedule syntactically fixed. A `loop` whose
        // private stride is decremented until zero is mathematically fixed,
        // but Dawn/Tint's uniformity analysis cannot always prove that fact.
        // These are the same seven radix-2 stages, written explicitly so every
        // lane reaches every barrier on every browser backend.
        if lane < 64u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 64u];
            }
        }
        workgroupBarrier();
        if lane < 32u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 32u];
            }
        }
        workgroupBarrier();
        if lane < 16u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 16u];
            }
        }
        workgroupBarrier();
        if lane < 8u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 8u];
            }
        }
        workgroupBarrier();
        if lane < 4u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 4u];
            }
        }
        workgroupBarrier();
        if lane < 2u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 2u];
            }
        }
        workgroupBarrier();
        if lane < 1u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let base = slot * SOURCE_LANE_COUNT;
                source_reduction[base + lane] += source_reduction[base + lane + 1u];
            }
        }
        workgroupBarrier();
        if lane == 0u {
            for (var slot = 0u; slot < SOURCE_REDUCTION_CHUNK; slot += 1u) {
                let coefficient = chunk + slot;
                if coefficient < coefficient_count {
                    var coordinate_scale = 1.0;
                    for (var degree = 0u; degree < COEFFICIENT_DEGREE[coefficient]; degree += 1u) {
                        coordinate_scale *= transverse_scale;
                    }
                    line_samples[destination_base + coefficient * QUADRATURE_CAPACITY] =
                        source_reduction[slot * SOURCE_LANE_COUNT]
                        * (quadrature_weight * coordinate_scale);
                }
            }
        }
    }
}
#endif

#ifdef EQ106_SOURCE
@compute @workgroup_size(128, 1, 1)
fn assemble_line_samples(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let quadrature_index = workgroup_id.x;
    let segment_index = workgroup_id.y;
    if local_index == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let sample = quadrature[quadrature_index];
    let basis = transverse_basis(eq_params.line_direction);
    let observer = eq_params.line_origin + sample.x * basis[0];
    var packed = accumulate_source_range(
        observer, basis[1], basis[2], 0u, eq_params.source_count, local_index,
    );
    let coefficient_count = active_coefficient_count();

    let destination_base = (eq_params.segment_id - 1u)
        * MAX_TAYLOR_COEFFICIENT_COUNT * QUADRATURE_CAPACITY + quadrature_index;
    reduce_and_store_taylor_jet(
        &packed, coefficient_count, local_index, destination_base, sample.y,
    );
}

// Planning density inversion: one workgroup owns a
// (canonical segment, density voxel, quadrature point) tuple. Its 128 lanes
// traverse the voxel's source range in parallel and reduce the complete
// transverse Taylor jet without atomics.
@compute @workgroup_size(128, 1, 1)
fn assemble_voxel_line_samples(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let quadrature_index = workgroup_id.x;
    let segment_index = workgroup_id.y;
    let voxel_index = workgroup_id.z;
    if local_index == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let group = density_modes.records[voxel_index];
    let source_begin = u32(group.x);
    let source_end = source_begin + u32(group.y);
    let sample = quadrature[quadrature_index];
    let basis = transverse_basis(eq_params.line_direction);
    let observer = eq_params.line_origin + sample.x * basis[0];
    var packed = accumulate_source_range(
        observer, basis[1], basis[2], source_begin, source_end, local_index,
    );
    let coefficient_count = active_coefficient_count();
    let destination_base = ((eq_params.segment_id - 1u) * PLANNING_VOXEL_COUNT + voxel_index)
        * MAX_TAYLOR_COEFFICIENT_COUNT * QUADRATURE_CAPACITY + quadrature_index;
    reduce_and_store_taylor_jet(
        &packed, coefficient_count, local_index, destination_base, sample.y,
    );
}
#endif

#ifdef EQ106_SPECTRUM
@compute @workgroup_size(64, 1, 1)
fn assemble_spectrum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let segment_index = global_id.y;
    if local_index == 0u {
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
    var result: SpectrumSample;
    result.acceleration_x = vec2<f32>(0.0);
    result.acceleration_y = vec2<f32>(0.0);
    result.acceleration_z = vec2<f32>(0.0);
    result.potential = vec2<f32>(0.0);

    for (var quadrature_index = 0u; quadrature_index < eq_params.quadrature_count; quadrature_index += 1u) {
        let packed = line_samples[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * QUADRATURE_CAPACITY
            + coefficient * QUADRATURE_CAPACITY + quadrature_index];
        let phase = quadrature[QUADRATURE_CAPACITY
            + frequency_index * QUADRATURE_CAPACITY + quadrature_index];
        result.acceleration_x += phase * packed.x;
        result.acceleration_y += phase * packed.y;
        result.acceleration_z += phase * packed.z;
        result.potential += phase * packed.w;
    }
    spectrum[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY
        + flat_index] = result;
}
#endif

#ifdef EQ106_SPECTRUM
@compute @workgroup_size(64, 1, 1)
fn assemble_voxel_spectrum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let segment_index = global_id.y;
    let voxel_index = global_id.z;
    if local_index == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let frequency_count = 2u * eq_params.half_count + 1u;
    let work_count = active_coefficient_count() * frequency_count;
    let flat_index = global_id.x;
    if flat_index >= work_count || voxel_index >= PLANNING_VOXEL_COUNT {
        return;
    }
    let coefficient = flat_index / frequency_count;
    let frequency_index = flat_index % frequency_count;
    var result: SpectrumSample;
    result.acceleration_x = vec2<f32>(0.0);
    result.acceleration_y = vec2<f32>(0.0);
    result.acceleration_z = vec2<f32>(0.0);
    result.potential = vec2<f32>(0.0);
    for (var quadrature_index = 0u; quadrature_index < eq_params.quadrature_count; quadrature_index += 1u) {
        let packed = line_samples[(((eq_params.segment_id - 1u) * PLANNING_VOXEL_COUNT + voxel_index)
            * MAX_TAYLOR_COEFFICIENT_COUNT + coefficient) * QUADRATURE_CAPACITY
            + quadrature_index];
        let phase = quadrature[QUADRATURE_CAPACITY
            + frequency_index * QUADRATURE_CAPACITY + quadrature_index];
        result.acceleration_x += phase * packed.x;
        result.acceleration_y += phase * packed.y;
        result.acceleration_z += phase * packed.z;
        result.potential += phase * packed.w;
    }
    let bank_voxel = voxel_index % PLANNING_VOXEL_BANK_COUNT;
    let destination = (((eq_params.segment_id - 1u) * PLANNING_VOXEL_BANK_COUNT + bank_voxel)
        * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY) + flat_index;
    let packed_a = vec4<f32>(result.acceleration_x, result.acceleration_y);
    let packed_b = vec4<f32>(result.acceleration_z, result.potential);
    if voxel_index < PLANNING_VOXEL_BANK_COUNT {
        let low_offset = u32(density_modes.records[112].x) + 2u * destination;
        line_samples[low_offset] = packed_a;
        line_samples[low_offset + 1u] = packed_b;
    } else {
        let high_offset = PLANNING_OUTPUT_PREFIX_VEC4 + 2u * destination;
        output[high_offset] = packed_a;
        output[high_offset + 1u] = packed_b;
    }
}
#endif

#ifdef EQ106_SPECTRUM
@compute @workgroup_size(64, 1, 1)
fn combine_voxel_spectrum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let segment_index = global_id.y;
    if local_index == 0u {
        eq_params = segment_params[segment_index];
    }
    workgroupBarrier();
    let frequency_count = 2u * eq_params.half_count + 1u;
    let work_count = active_coefficient_count() * frequency_count;
    let flat_index = global_id.x;
    if flat_index >= work_count {
        return;
    }
    var result: SpectrumSample;
    result.acceleration_x = vec2<f32>(0.0);
    result.acceleration_y = vec2<f32>(0.0);
    result.acceleration_z = vec2<f32>(0.0);
    result.potential = vec2<f32>(0.0);
    for (var voxel = 0u; voxel < PLANNING_VOXEL_COUNT; voxel += 1u) {
        let density = density_modes.records[PLANNING_VOXEL_COUNT + voxel].x;
        let bank_voxel = voxel % PLANNING_VOXEL_BANK_COUNT;
        let source_index = (((eq_params.segment_id - 1u) * PLANNING_VOXEL_BANK_COUNT + bank_voxel)
            * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY) + flat_index;
        var packed_a: vec4<f32>;
        var packed_b: vec4<f32>;
        if voxel < PLANNING_VOXEL_BANK_COUNT {
            let low_offset = u32(density_modes.records[112].x) + 2u * source_index;
            packed_a = line_samples[low_offset];
            packed_b = line_samples[low_offset + 1u];
        } else {
            let high_offset = PLANNING_OUTPUT_PREFIX_VEC4 + 2u * source_index;
            packed_a = output[high_offset];
            packed_b = output[high_offset + 1u];
        }
        var sample: SpectrumSample;
        sample.acceleration_x = packed_a.xy;
        sample.acceleration_y = packed_a.zw;
        sample.acceleration_z = packed_b.xy;
        sample.potential = packed_b.zw;
        result.acceleration_x += density * sample.acceleration_x;
        result.acceleration_y += density * sample.acceleration_y;
        result.acceleration_z += density * sample.acceleration_z;
        result.potential += density * sample.potential;
    }
    spectrum[(eq_params.segment_id - 1u) * MAX_TAYLOR_COEFFICIENT_COUNT
        * SPECTRUM_FREQUENCY_CAPACITY + flat_index] = result;
}
#endif

#ifdef EQ106_EVALUATOR
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
// independent of the sampled Bromwich/Taylor path used for acceleration.
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

@compute @workgroup_size(32, 1, 1)
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
    let source_radius = 2.0 / max(eq_params.sigma, 1.0e-12);
    let transverse_scale = max(length(eq_params.line_origin) - source_radius, 1.0);
    let normalized_u = u / transverse_scale;
    let normalized_v = v / transverse_scale;
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
    var coarse_field = vec3<f32>(0.0);
    // A cheap, same-field diagnostic: reconstruct the Taylor polynomial at
    // symmetric transverse offsets and compare its finite difference with the
    // analytic u/v derivative assembled below. This catches coefficient
    // indexing and transverse-basis mistakes without another spectrum build.
    // One metre keeps the f32 field subtraction above cancellation noise;
    // its O(delta^2) truncation remains negligible relative to the hundreds
    // of metres source distance in the certified exterior tube.
    // The direct f64 benchmark still verifies every First target.  This
    // same-field consistency check is sampled so its Eq.106-only arithmetic
    // cannot dominate the performance comparison.
    let self_fd_active = eq_params.evaluate_dual_certificate != 0u
        && target_index % 32u == 0u;
    var self_fd_u_plus: array<vec3<f32>, 5>;
    var self_fd_u_minus: array<vec3<f32>, 5>;
    var self_fd_v_plus: array<vec3<f32>, 5>;
    var self_fd_v_minus: array<vec3<f32>, 5>;

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
    var local_edge_coarse = vec2<f32>(0.0);
    for (var frequency_index = lane; frequency_index < frequency_count; frequency_index += 32u) {
        let omega = evaluation_omega[frequency_index];
        let denominator = max(eq_params.sigma * eq_params.sigma + omega * omega, 1.0e-20);
        let phase = evaluation_phase[frequency_index];
        local_edge.x += (phase.x * eq_params.sigma + phase.y * omega) / denominator;
        if frequency_index % 2u == 0u {
            local_edge_coarse.x +=
                (phase.x * eq_params.sigma + phase.y * omega) / denominator;
        }
        if eq_params.inversion_mode != 1u {
            let spectral_derivative = evaluation_spectral_derivative[frequency_index];
            local_edge.y += (
                spectral_derivative.x * eq_params.sigma
                + spectral_derivative.y * omega
            ) / denominator;
            if frequency_index % 2u == 0u {
                local_edge_coarse.y += (
                    spectral_derivative.x * eq_params.sigma
                    + spectral_derivative.y * omega
                ) / denominator;
            }
        }
    }
    evaluation_edge[lane] = local_edge;
    evaluation_edge_coarse[lane] = local_edge_coarse;
    workgroupBarrier();
    var edge_stride = 16u;
    loop {
        if edge_stride == 0u { break; }
        if lane < edge_stride {
            evaluation_edge[lane] += evaluation_edge[lane + edge_stride];
            evaluation_edge_coarse[lane] += evaluation_edge_coarse[lane + edge_stride];
        }
        workgroupBarrier();
        edge_stride = edge_stride >> 1u;
    }

    // Discrete inverse-transform normalization: reproduce the constant
    // response exactly when the signed-band quadrature is well conditioned.
    // The previous max(abs(response), 0.25) silently changed the operator and
    // its derivative. A near-zero response is now an explicit certificate
    // failure instead of an empirical clamp.
    let raw_edge_response = evaluation_edge[0].x * inversion_scale;
    let edge_response_valid = abs(raw_edge_response) >= 1.0e-3;
    let edge_response = select(1.0, raw_edge_response, edge_response_valid);
    let edge_response_derivative = select(
        0.0,
        evaluation_edge[0].y * inversion_scale,
        edge_response_valid,
    );
    let coarse_edge_response_raw = evaluation_edge_coarse[0].x * (2.0 * inversion_scale);
    let coarse_edge_response_valid = abs(coarse_edge_response_raw) >= 1.0e-3;
    let coarse_edge_response = select(1.0, coarse_edge_response_raw, coarse_edge_response_valid);

    // Keep every invocation on the same statically bounded barrier path.
    // Dawn cannot prove that a function result derived from workgroup memory
    // is uniform, even though lane zero initialized `eq_params` above.
    for (var coefficient = 0u; coefficient < MAX_TAYLOR_COEFFICIENT_COUNT; coefficient += 1u) {
        var local_sum = vec4<f32>(0.0);
        var local_derivative = vec3<f32>(0.0);
        var local_imaginary = vec3<f32>(0.0);
        var local_tail = vec3<f32>(0.0);
        var local_coarse = vec3<f32>(0.0);
        // The outer loop and every barrier stay statically uniform, but only
        // active Taylor coefficients perform spectrum loads and complex math.
        // Extra workgroups from a batched dispatch likewise remain on the
        // barrier path without repeating an active target's full evaluation.
        if coefficient < coefficient_count && target_active {
            if eq_params.inversion_mode == 2u {
                if lane == 0u {
                    let growth = exp(eq_params.sigma * h);
                    let fine_xy = type2_nufft_interpolate(coefficient, 0u, h);
                    let fine_zp = type2_nufft_interpolate(coefficient, 1u, h);
                    let derivative_xy = type2_nufft_interpolate(coefficient, 2u, h);
                    let derivative_z = type2_nufft_interpolate(coefficient, 3u, h);
                    let coarse_xy = type2_nufft_interpolate(coefficient, 4u, h);
                    let coarse_z = type2_nufft_interpolate(coefficient, 5u, h);
                    local_sum = growth * vec4<f32>(
                        fine_xy.value.x,
                        fine_xy.value.z,
                        fine_zp.value.x,
                        fine_zp.value.z,
                    );
                    local_derivative = growth * vec3<f32>(
                        derivative_xy.value.x,
                        derivative_xy.value.z,
                        derivative_z.value.x,
                    );
                    local_coarse = growth * vec3<f32>(
                        coarse_xy.value.x,
                        coarse_xy.value.z,
                        coarse_z.value.x,
                    );
                    if eq_params.evaluate_dual_certificate != 0u {
                        local_imaginary = growth * vec3<f32>(
                            fine_xy.value.y,
                            fine_xy.value.w,
                            fine_zp.value.y,
                        );
                        local_tail = growth * vec3<f32>(
                            length(fine_xy.error.xy),
                            length(fine_xy.error.zw),
                            length(fine_zp.error.xy),
                        );
                        for (var endpoint = 0u; endpoint < 2u; endpoint += 1u) {
                            let frequency_index = select(0u, frequency_count - 1u, endpoint == 1u);
                            let phase = evaluation_phase[frequency_index];
                            let sample = spectrum[(eq_params.segment_id - 1u)
                                * MAX_TAYLOR_COEFFICIENT_COUNT * SPECTRUM_FREQUENCY_CAPACITY
                                + coefficient * SPECTRUM_FREQUENCY_CAPACITY + frequency_index];
                            let x = complex_mul(sample.acceleration_x, phase);
                            let y = complex_mul(sample.acceleration_y, phase);
                            let z = complex_mul(sample.acceleration_z, phase);
                            local_tail += vec3<f32>(abs(x.x), abs(y.x), abs(z.x));
                        }
                    }
                }
            } else {
                for (var frequency_index = lane; frequency_index < frequency_count; frequency_index += 32u) {
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
                    if frequency_index % 2u == 0u {
                        local_coarse += vec3<f32>(x.x, y.x, z.x);
                    }
                    if eq_params.inversion_mode != 1u {
                        let spectral_derivative = evaluation_spectral_derivative[frequency_index];
                        local_derivative += vec3<f32>(
                            complex_mul(sample.acceleration_x, spectral_derivative).x,
                            complex_mul(sample.acceleration_y, spectral_derivative).x,
                            complex_mul(sample.acceleration_z, spectral_derivative).x,
                        );
                        if eq_params.evaluate_dual_certificate != 0u {
                            local_imaginary += vec3<f32>(x.y, y.y, z.y);
                            if abs(signed_index) == i32(eq_params.half_count) {
                                local_tail += vec3<f32>(abs(x.x), abs(y.x), abs(z.x));
                            }
                        }
                    }
                }
            }
        }
        evaluation_sum[lane] = local_sum;
        evaluation_derivative[lane] = vec4<f32>(local_derivative, 0.0);
        evaluation_imaginary[lane] = vec4<f32>(local_imaginary, 0.0);
        evaluation_tail[lane] = vec4<f32>(local_tail, 0.0);
        evaluation_coarse[lane] = vec4<f32>(local_coarse, 0.0);
        workgroupBarrier();

        var stride = 16u;
        loop {
            if stride == 0u { break; }
            if lane < stride {
                evaluation_sum[lane] += evaluation_sum[lane + stride];
                evaluation_derivative[lane] += evaluation_derivative[lane + stride];
                evaluation_imaginary[lane] += evaluation_imaginary[lane + stride];
                evaluation_tail[lane] += evaluation_tail[lane + stride];
                evaluation_coarse[lane] += evaluation_coarse[lane + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if lane == 0u && coefficient < coefficient_count {
            let a = COEFFICIENT_A[coefficient];
            let b = COEFFICIENT_B[coefficient];
            let degree = COEFFICIENT_DEGREE[coefficient];
            let raw = evaluation_sum[0] * inversion_scale;
            let reconstructed = raw.xyz / edge_response;
            let reconstructed_potential = raw.w / edge_response;
            let value = monomial(normalized_u, normalized_v, a, b);
            field += vec4<f32>(reconstructed, reconstructed_potential) * value;
            let reconstructed_coarse = evaluation_coarse[0].xyz
                * (2.0 * inversion_scale) / coarse_edge_response;
            coarse_field += reconstructed_coarse * value;
            if eq_params.inversion_mode != 1u {
                let raw_derivative_h = evaluation_derivative[0].xyz * inversion_scale;
                let reconstructed_derivative_h = (
                    raw_derivative_h * edge_response
                    - raw.xyz * edge_response_derivative
                ) / (edge_response * edge_response);
                derivative_h += reconstructed_derivative_h * value;
                if a > 0u {
                    derivative_u += reconstructed.xyz * f32(a)
                        * monomial(normalized_u, normalized_v, a - 1u, b)
                        / transverse_scale;
                }
                if b > 0u {
                    derivative_v += reconstructed.xyz * f32(b)
                        * monomial(normalized_u, normalized_v, a, b - 1u)
                        / transverse_scale;
                }
                if self_fd_active {
                    for (var fd_index = 0u; fd_index < 5u; fd_index += 1u) {
                        let delta = SELF_FD_DELTAS[fd_index];
                        let normalized_delta = delta / transverse_scale;
                        self_fd_u_plus[fd_index] += reconstructed.xyz
                            * monomial(normalized_u + normalized_delta, normalized_v, a, b);
                        self_fd_u_minus[fd_index] += reconstructed.xyz
                            * monomial(normalized_u - normalized_delta, normalized_v, a, b);
                        self_fd_v_plus[fd_index] += reconstructed.xyz
                            * monomial(normalized_u, normalized_v + normalized_delta, a, b);
                        self_fd_v_minus[fd_index] += reconstructed.xyz
                            * monomial(normalized_u, normalized_v - normalized_delta, a, b);
                    }
                }
                if eq_params.evaluate_dual_certificate != 0u {
                    imaginary_field += evaluation_imaginary[0].xyz
                        * inversion_scale / edge_response * value;
                    tail_field += evaluation_tail[0].xyz
                        * inversion_scale / edge_response * abs(value);
                    if degree == active_order {
                        last_order_field += reconstructed.xyz * value;
                    }
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
    if eq_params.inversion_mode == 1u {
        output[target_index] = field;
        return;
    }

    let derivative_x = derivative_h * tangent.x + derivative_u * normal.x + derivative_v * binormal.x;
    let derivative_y = derivative_h * tangent.y + derivative_u * normal.y + derivative_v * binormal.y;
    let derivative_z = derivative_h * tangent.z + derivative_u * normal.z + derivative_v * binormal.z;
    let field_scale = max(length(field.xyz), 1.0e-12);
    let certificate_active = eq_params.evaluate_dual_certificate != 0u;
    let field_taylor_residual = select(0.0, length(last_order_field) / field_scale, certificate_active);
    let source_compression_absolute_bound = 64.0 * eq_params.g_const
        * density_modes.records[112].y;
    let source_compression_residual = select(
        0.0,
        source_compression_absolute_bound / field_scale,
        certificate_active,
    );
    // The highest retained derivative is part of the solution, not the first
    // omitted term. In particular at A=1,u=v=0 it is usually the dominant
    // transverse gradient and must not be treated as a unit-sized remainder.
    // Gradient-tail admission is controlled by the CPU geometric bound
    // (A+1)*epsilon^A/(1-epsilon)^2; the shader certificate reports only the
    // observable field tail plus independent spectral diagnostics.
    let taylor_residual = select(
        1.0,
        max(field_taylor_residual, source_compression_residual),
        edge_response_valid,
    );
    let imaginary_residual = select(0.0, length(imaginary_field) / field_scale, certificate_active);
    let frequency_convergence_residual = select(
        1.0,
        length(field.xyz - coarse_field) / field_scale,
        coarse_edge_response_valid,
    );
    let spectral_tail_residual = select(
        0.0,
        max(length(tail_field) / field_scale, frequency_convergence_residual),
        certificate_active,
    );
    let transverse_ratio = select(
        0.0,
        length(vec2<f32>(u, v)) / max(eq_params.line_limit, 1.0),
        certificate_active,
    );
    var self_fd_errors: array<f32, 5>;
    if self_fd_active {
        let self_fd_scale = max(sqrt(
            dot(derivative_u, derivative_u) + dot(derivative_v, derivative_v)
        ), 1.0e-12);
        for (var fd_index = 0u; fd_index < 5u; fd_index += 1u) {
            let delta = SELF_FD_DELTAS[fd_index];
            let self_fd_u = (self_fd_u_plus[fd_index] - self_fd_u_minus[fd_index])
                / (2.0 * delta);
            let self_fd_v = (self_fd_v_plus[fd_index] - self_fd_v_minus[fd_index])
                / (2.0 * delta);
            let self_fd_error = sqrt(
                dot(self_fd_u - derivative_u, self_fd_u - derivative_u)
                + dot(self_fd_v - derivative_v, self_fd_v - derivative_v)
            );
            self_fd_errors[fd_index] = self_fd_error / self_fd_scale;
        }
    }
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
    output[output_base + 4u] = vec4<f32>(derivative_z, self_fd_errors[2]);
    output[output_base + 5u] = vec4<f32>(h, u, v, f32(eq_params.segment_id));
    output[output_base + 6u] = vec4<f32>(
        field.w,
        independent_potential,
        f32(eq_params.evaluate_dual_certificate),
        elapsed,
    );
    output[output_base + 7u] = vec4<f32>(probe_pos, f32(eq_params.target_count));
    output[output_base + 8u] = vec4<f32>(eq_params.line_origin, 0.0);
    output[output_base + 9u] = vec4<f32>(
        self_fd_errors[0], self_fd_errors[1], self_fd_errors[2], self_fd_errors[3]
    );
    output[output_base + 10u] = vec4<f32>(self_fd_errors[4], 0.0, 0.0, 0.0);
}
#endif
