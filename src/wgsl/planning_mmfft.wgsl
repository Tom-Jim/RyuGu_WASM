struct Params {
    state_offset: u32,
    state_count: u32,
    density_model: u32,
    level_count: u32,
    grid_sizes: vec2<u32>,
    _padding0: vec2<u32>,
    half_extents: vec2<f32>,
    total_mass: f32,
    _derivative_step: f32,
    grid_scales: vec2<f32>,
    gravity_constant: f32,
    _padding1: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> hierarchy: array<u32>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> field_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_x_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_y_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_z_sum: array<vec4<f32>, 64>;
var<workgroup> sample_base: vec3<u32>;
var<workgroup> weights_x: array<f32, 6>;
var<workgroup> weights_y: array<f32, 6>;
var<workgroup> weights_z: array<f32, 6>;
var<workgroup> derivatives_x: array<f32, 6>;
var<workgroup> derivatives_y: array<f32, 6>;
var<workgroup> derivatives_z: array<f32, 6>;
var<workgroup> second_derivatives_x: array<f32, 6>;
var<workgroup> second_derivatives_y: array<f32, 6>;
var<workgroup> second_derivatives_z: array<f32, 6>;
var<workgroup> inverse_spacing: f32;
var<workgroup> selected_level: u32;

fn linear_index(cell: vec3<u32>, level: u32) -> u32 {
    let n = params.grid_sizes[level];
    let offset = select(
        params.grid_sizes.x * params.grid_sizes.x * params.grid_sizes.x,
        0u,
        level == 0u,
    );
    return offset + (cell.z * n + cell.y) * n + cell.x;
}

fn potential_at(index: u32, level: u32) -> f32 {
    // The GPU basis combiner stores one f32 potential per word, not two f16
    // samples. Differentiate the same potential without half-precision noise.
    return bitcast<f32>(hierarchy[index]) * params.grid_scales[level];
}

// A single quintic interpolant supplies potential, gravity and Hessian.
// Catmull-Rom second derivatives retain large cell-scale errors even when its
// values look smooth. Six nodes reproduce degree-five polynomials exactly.
struct InterpolationWeights {
    value: array<f32, 6>,
    first: array<f32, 6>,
    second: array<f32, 6>,
};
fn quintic_weights(t: f32) -> InterpolationWeights {
    var weights: InterpolationWeights;
    for (var i = 0u; i < 6u; i += 1u) {
        var value = 1.0;
        var first = 0.0;
        var second = 0.0;
        for (var j = 0u; j < 6u; j += 1u) {
            if i == j { continue; }
            let inverse = 1.0 / (f32(i) - f32(j));
            let factor = (t - (f32(j) - 2.0)) * inverse;
            second = second * factor + 2.0 * first * inverse;
            first = first * factor + value * inverse;
            value *= factor;
        }
        weights.value[i] = value;
        weights.first[i] = first;
        weights.second[i] = second;
    }
    return weights;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let target_index = group_id.x + group_id.y * 65535u;
    if target_index >= params.state_count { return; }
    let lane = local_id.x;
    let observer_position = positions[params.state_offset + target_index].xyz;
    if lane == 0u {
        selected_level = 0xffffffffu;
        for (var level = 0u; level < params.level_count; level += 1u) {
            let spacing = 2.0 * params.half_extents[level] / f32(params.grid_sizes[level]);
            if all(abs(observer_position) <= vec3<f32>(params.half_extents[level] - 4.0 * spacing)) {
                selected_level = level;
                break;
            }
        }
        if selected_level != 0xffffffffu {
            let half = params.half_extents[selected_level];
            let spacing = 2.0 * half / f32(params.grid_sizes[selected_level]);
            let coordinate = (observer_position + vec3<f32>(half)) / spacing - vec3<f32>(0.5);
            let base = clamp(
                floor(coordinate),
                vec3<f32>(2.0),
                vec3<f32>(f32(params.grid_sizes[selected_level] - 4u)),
            );
            let fraction = clamp(coordinate - base, vec3<f32>(0.0), vec3<f32>(1.0));
            sample_base = vec3<u32>(base) - vec3<u32>(2u);
            let x = quintic_weights(fraction.x);
            let y = quintic_weights(fraction.y);
            let z = quintic_weights(fraction.z);
            weights_x = x.value;
            weights_y = y.value;
            weights_z = z.value;
            derivatives_x = x.first;
            derivatives_y = y.first;
            derivatives_z = z.first;
            second_derivatives_x = x.second;
            second_derivatives_y = y.second;
            second_derivatives_z = z.second;
            inverse_spacing = 1.0 / spacing;
        }
    }
    workgroupBarrier();
    if selected_level == 0xffffffffu {
        if lane == 0u {
            let radius2 = max(dot(observer_position, observer_position), 1.0e-8);
            let inverse_radius = inverseSqrt(radius2);
            let inverse_radius3 = inverse_radius / radius2;
            let inverse_radius5 = inverse_radius3 / radius2;
            let scale = params.gravity_constant * params.total_mass;
            field_sum[lane] = vec4<f32>(
                -scale * observer_position * inverse_radius3,
                scale * inverse_radius,
            );
            let diagonal = -scale * inverse_radius3;
            let outer_scale = 3.0 * scale * inverse_radius5;
            jacobian_x_sum[lane] = vec4<f32>(
                vec3<f32>(diagonal, 0.0, 0.0) + outer_scale * observer_position * observer_position.x,
                0.0,
            );
            jacobian_y_sum[lane] = vec4<f32>(
                vec3<f32>(0.0, diagonal, 0.0) + outer_scale * observer_position * observer_position.y,
                0.0,
            );
            jacobian_z_sum[lane] = vec4<f32>(
                vec3<f32>(0.0, 0.0, diagonal) + outer_scale * observer_position * observer_position.z,
                0.0,
            );
        } else {
            field_sum[lane] = vec4<f32>(0.0);
            jacobian_x_sum[lane] = vec4<f32>(0.0);
            jacobian_y_sum[lane] = vec4<f32>(0.0);
            jacobian_z_sum[lane] = vec4<f32>(0.0);
        }
    } else {
        var field = vec4<f32>(0.0);
        var jacobian_x = vec4<f32>(0.0);
        var jacobian_y = vec4<f32>(0.0);
        var jacobian_z = vec4<f32>(0.0);
        // Same 64-lane group; at most four grid loads per lane.
        for (var stencil = lane; stencil < 216u; stencil += 64u) {
            let dx = stencil % 6u;
            let dy = (stencil / 6u) % 6u;
            let dz = stencil / 36u;
            let potential = potential_at(
                linear_index(sample_base + vec3<u32>(dx, dy, dz), selected_level),
                selected_level,
            );
            let wx = weights_x[dx];
            let wy = weights_y[dy];
            let wz = weights_z[dz];
            let first_scale = inverse_spacing;
            let second_scale = inverse_spacing * inverse_spacing;
            let dxx = second_derivatives_x[dx] * wy * wz * second_scale;
            let dyy = second_derivatives_y[dy] * wx * wz * second_scale;
            let dzz = second_derivatives_z[dz] * wx * wy * second_scale;
            let dxy = derivatives_x[dx] * derivatives_y[dy] * wz * second_scale;
            let dxz = derivatives_x[dx] * wy * derivatives_z[dz] * second_scale;
            let dyz = wx * derivatives_y[dy] * derivatives_z[dz] * second_scale;
            field += potential * vec4<f32>(
                derivatives_x[dx] * wy * wz * first_scale,
                derivatives_y[dy] * wx * wz * first_scale,
                derivatives_z[dz] * wx * wy * first_scale,
                wx * wy * wz,
            );
            jacobian_x += potential * vec4<f32>(dxx, dxy, dxz, 0.0);
            jacobian_y += potential * vec4<f32>(dxy, dyy, dyz, 0.0);
            jacobian_z += potential * vec4<f32>(dxz, dyz, dzz, 0.0);
        }
        field_sum[lane] = field;
        jacobian_x_sum[lane] = jacobian_x;
        jacobian_y_sum[lane] = jacobian_y;
        jacobian_z_sum[lane] = jacobian_z;
    }
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if lane < stride {
            field_sum[lane] += field_sum[lane + stride];
            jacobian_x_sum[lane] += jacobian_x_sum[lane + stride];
            jacobian_y_sum[lane] += jacobian_y_sum[lane + stride];
            jacobian_z_sum[lane] += jacobian_z_sum[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if lane == 0u {
        let valid = 1.0; // The complete density row is validated by the host; voxel 0 may be empty.
        let base = target_index * 4u;
        output[base] = field_sum[0] * valid;
        output[base + 1u] = jacobian_x_sum[0] * valid;
        output[base + 2u] = jacobian_y_sum[0] * valid;
        output[base + 3u] = jacobian_z_sum[0] * valid;
    }
}
