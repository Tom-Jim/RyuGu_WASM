// Runtime evaluation of the two-level 3-D MMFFT hierarchy. The stored field
// was produced by a conservative deposit, zero-padded forward FFT, Newton
// kernel multiplication, and inverse FFT for every level.

struct MmfftParams {
    probe_pos: vec3<f32>,
    g_const: f32,
    grid_sizes: vec2<u32>,
    level_count: u32,
    _padding0: u32,
    half_extents: vec2<f32>,
    total_mass: f32,
};

@group(0) @binding(0) var<uniform> params: MmfftParams;
@group(0) @binding(1) var<storage, read> hierarchy: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> sample_sum: array<vec4<f32>, 64>;
var<workgroup> sample_base: vec3<u32>;
var<workgroup> sample_wx: vec4<f32>;
var<workgroup> sample_wy: vec4<f32>;
var<workgroup> sample_wz: vec4<f32>;
var<workgroup> sample_dx: vec4<f32>;
var<workgroup> sample_dy: vec4<f32>;
var<workgroup> sample_dz: vec4<f32>;
var<workgroup> sample_inverse_spacing: f32;

fn linear_index(cell: vec3<u32>, level: u32) -> u32 {
    let n = params.grid_sizes[level];
    let offset = select(params.grid_sizes.x * params.grid_sizes.x * params.grid_sizes.x, 0u, level == 0u);
    return offset + (cell.z * n + cell.y) * n + cell.x;
}

fn cubic_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    );
}

fn cubic_derivatives(t: f32) -> vec4<f32> {
    let t2 = t * t;
    return vec4<f32>(
        -0.5 + 2.0 * t - 1.5 * t2,
        -5.0 * t + 4.5 * t2,
        0.5 + 4.0 * t - 4.5 * t2,
        -t + 1.5 * t2,
    );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var selected_level = 0xffffffffu;
    for (var level = 0u; level < params.level_count; level += 1u) {
        let half_extent = params.half_extents[level];
        let spacing = 2.0 * half_extent / f32(params.grid_sizes[level]);
        if all(abs(params.probe_pos) <= vec3<f32>(half_extent - spacing)) {
            selected_level = level;
            break;
        }
    }
    if lane == 0u && selected_level != 0xffffffffu {
        let half_extent = params.half_extents[selected_level];
        let spacing = 2.0 * half_extent / f32(params.grid_sizes[selected_level]);
        let coordinate = (params.probe_pos + vec3<f32>(half_extent)) / spacing
            - vec3<f32>(0.5);
        let base_f = clamp(
            floor(coordinate),
            vec3<f32>(1.0),
            vec3<f32>(f32(params.grid_sizes[selected_level] - 3u)),
        );
        let fraction = clamp(coordinate - base_f, vec3<f32>(0.0), vec3<f32>(1.0));
        sample_base = vec3<u32>(base_f) - vec3<u32>(1u);
        sample_wx = cubic_weights(fraction.x);
        sample_wy = cubic_weights(fraction.y);
        sample_wz = cubic_weights(fraction.z);
        sample_dx = cubic_derivatives(fraction.x);
        sample_dy = cubic_derivatives(fraction.y);
        sample_dz = cubic_derivatives(fraction.z);
        sample_inverse_spacing = 1.0 / spacing;
    }
    workgroupBarrier();

    if selected_level == 0xffffffffu {
        if lane == 0u {
            let distance2 = max(dot(params.probe_pos, params.probe_pos), 1.0e-8);
            let distance = sqrt(distance2);
            output_acc[0] = params.g_const * vec4<f32>(
                -params.total_mass * params.probe_pos / (distance2 * distance),
                params.total_mass / distance,
            );
        }
        return;
    }

    let dx = lane & 3u;
    let dy = (lane >> 2u) & 3u;
    let dz = lane >> 4u;
    let corner_potential = hierarchy[
        linear_index(sample_base + vec3<u32>(dx, dy, dz), selected_level)
    ].w;
    let wx = sample_wx[dx];
    let wy = sample_wy[dy];
    let wz = sample_wz[dz];
    sample_sum[lane] = corner_potential * vec4<f32>(
        sample_dx[dx] * wy * wz * sample_inverse_spacing,
        sample_dy[dy] * wx * wz * sample_inverse_spacing,
        sample_dz[dz] * wx * wy * sample_inverse_spacing,
        wx * wy * wz,
    );
    workgroupBarrier();

    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if lane < stride {
            sample_sum[lane] += sample_sum[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if lane == 0u {
        // Acceleration remains the analytic gradient of the exact same
        // tricubic potential returned in w.
        output_acc[0] = sample_sum[0];
    }
}
