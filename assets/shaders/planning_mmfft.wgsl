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
    gravity_constant: f32,
    _padding1: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> hierarchy: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> field_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_x_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_y_sum: array<vec4<f32>, 64>;
var<workgroup> jacobian_z_sum: array<vec4<f32>, 64>;
var<workgroup> sample_base: vec3<u32>;
var<workgroup> weights_x: vec4<f32>;
var<workgroup> weights_y: vec4<f32>;
var<workgroup> weights_z: vec4<f32>;
var<workgroup> derivatives_x: vec4<f32>;
var<workgroup> derivatives_y: vec4<f32>;
var<workgroup> derivatives_z: vec4<f32>;
var<workgroup> second_derivatives_x: vec4<f32>;
var<workgroup> second_derivatives_y: vec4<f32>;
var<workgroup> second_derivatives_z: vec4<f32>;
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

fn cubic_second_derivatives(t: f32) -> vec4<f32> {
    return vec4<f32>(2.0 - 3.0 * t, -5.0 + 9.0 * t, 4.0 - 9.0 * t, -1.0 + 3.0 * t);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if group_id.x >= params.state_count { return; }
    let lane = local_id.x;
    let observer_position = positions[params.state_offset + group_id.x].xyz;
    if lane == 0u {
        selected_level = 0xffffffffu;
        for (var level = 0u; level < params.level_count; level += 1u) {
            let spacing = 2.0 * params.half_extents[level] / f32(params.grid_sizes[level]);
            if all(abs(observer_position) <= vec3<f32>(params.half_extents[level] - spacing)) {
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
                vec3<f32>(1.0),
                vec3<f32>(f32(params.grid_sizes[selected_level] - 3u)),
            );
            let fraction = clamp(coordinate - base, vec3<f32>(0.0), vec3<f32>(1.0));
            sample_base = vec3<u32>(base) - vec3<u32>(1u);
            weights_x = cubic_weights(fraction.x);
            weights_y = cubic_weights(fraction.y);
            weights_z = cubic_weights(fraction.z);
            derivatives_x = cubic_derivatives(fraction.x);
            derivatives_y = cubic_derivatives(fraction.y);
            derivatives_z = cubic_derivatives(fraction.z);
            second_derivatives_x = cubic_second_derivatives(fraction.x);
            second_derivatives_y = cubic_second_derivatives(fraction.y);
            second_derivatives_z = cubic_second_derivatives(fraction.z);
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
        let dx = lane & 3u;
        let dy = (lane >> 2u) & 3u;
        let dz = lane >> 4u;
        let potential = hierarchy[
            linear_index(sample_base + vec3<u32>(dx, dy, dz), selected_level)
        ].w;
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
        field_sum[lane] = potential * vec4<f32>(
            derivatives_x[dx] * wy * wz * first_scale,
            derivatives_y[dy] * wx * wz * first_scale,
            derivatives_z[dz] * wx * wy * first_scale,
            wx * wy * wz,
        );
        jacobian_x_sum[lane] = potential * vec4<f32>(dxx, dxy, dxz, 0.0);
        jacobian_y_sum[lane] = potential * vec4<f32>(dxy, dyy, dyz, 0.0);
        jacobian_z_sum[lane] = potential * vec4<f32>(dxz, dyz, dzz, 0.0);
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
        let valid = select(0.0, 1.0, densities[params.density_model * 56u] > 0.0);
        let base = group_id.x * 4u;
        output[base] = field_sum[0] * valid;
        output[base + 1u] = jacobian_x_sum[0] * valid;
        output[base + 2u] = jacobian_y_sum[0] * valid;
        output[base + 3u] = jacobian_z_sum[0] * valid;
    }
}
