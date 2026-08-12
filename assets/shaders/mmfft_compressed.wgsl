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

fn sample_grid(level: u32, position: vec3<f32>) -> vec4<f32> {
    let n = params.grid_sizes[level];
    let half_extent = params.half_extents[level];
    let spacing = 2.0 * half_extent / f32(n);
    let coordinate = (position + vec3<f32>(half_extent)) / spacing - vec3<f32>(0.5);
    let base_f = clamp(floor(coordinate), vec3<f32>(1.0), vec3<f32>(f32(n - 3u)));
    let fraction = clamp(coordinate - base_f, vec3<f32>(0.0), vec3<f32>(1.0));
    let base = vec3<u32>(base_f) - vec3<u32>(1u);
    let wx = cubic_weights(fraction.x);
    let wy = cubic_weights(fraction.y);
    let wz = cubic_weights(fraction.z);
    let dxw = cubic_derivatives(fraction.x);
    let dyw = cubic_derivatives(fraction.y);
    let dzw = cubic_derivatives(fraction.z);
    var potential = 0.0;
    var gradient = vec3<f32>(0.0);
    for (var dz = 0u; dz < 4u; dz += 1u) {
        for (var dy = 0u; dy < 4u; dy += 1u) {
            for (var dx = 0u; dx < 4u; dx += 1u) {
                let corner_potential = hierarchy[linear_index(base + vec3<u32>(dx, dy, dz), level)].w;
                potential += wx[dx] * wy[dy] * wz[dz] * corner_potential;
                gradient += corner_potential / spacing * vec3<f32>(
                    dxw[dx] * wy[dy] * wz[dz],
                    dyw[dy] * wx[dx] * wz[dz],
                    dzw[dz] * wx[dx] * wy[dy],
                );
            }
        }
    }
    // The acceleration is the analytic gradient of the exact same tricubic
    // potential returned in w. This discrete identity prevents an interpolant
    // mismatch from injecting Jacobi energy at accelerated time scales.
    return vec4<f32>(gradient, potential);
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    var value = vec4<f32>(0.0);
    var found = false;
    for (var level = 0u; level < params.level_count; level += 1u) {
        let half_extent = params.half_extents[level];
        // Leave one interpolation cell at the edge to avoid clamping a target
        // onto a constant boundary value.
        let margin = 2.0 * half_extent / f32(params.grid_sizes[level]);
        if all(abs(params.probe_pos) <= vec3<f32>(half_extent - margin)) {
            value = sample_grid(level, params.probe_pos);
            found = true;
            break;
        }
    }
    if !found {
        let distance2 = max(dot(params.probe_pos, params.probe_pos), 1.0e-8);
        let distance = sqrt(distance2);
        value = params.g_const * vec4<f32>(
            -params.total_mass * params.probe_pos / (distance2 * distance),
            params.total_mass / distance,
        );
    }
    output_acc[0] = value;
}
