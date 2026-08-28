struct PlanningMetricsParams {
    pl_state_offset: u32,
    pl_state_count: u32,
    pl_samples_per_candidate: u32,
    pl_density_model: u32,
    pl_row_stride: u32,
    pl_candidate_count: u32,
    pl_padding0: vec2<u32>,
    pl_body_radius: f32,
    pl_certificate_tolerance: f32,
    pl_transverse_limit: f32,
    pl_padding1: f32,
};

@group(0) @binding(0) var<uniform> pl_params: PlanningMetricsParams;
@group(0) @binding(1) var<storage, read> pl_fields: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pl_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pl_baseline: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> pl_metrics: array<vec4<f32>>;

var<workgroup> pl_difference_sum: array<f32, 64>;
var<workgroup> pl_baseline_sum: array<f32, 64>;
var<workgroup> pl_gradient_sum: array<f32, 64>;
var<workgroup> pl_minimum_altitude: array<f32, 64>;
var<workgroup> pl_invalid_sum: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let pl_candidate_index = group_id.x;
    let pl_lane_index = local_id.x;
    if pl_candidate_index >= pl_params.pl_candidate_count { return; }
    var pl_difference_energy = 0.0;
    var pl_reference_energy = 0.0;
    var pl_jacobian_energy = 0.0;
    var pl_altitude = 1.0e30;
    var pl_invalid_count = 0u;
    for (
        var pl_sample_index = pl_lane_index;
        pl_sample_index < pl_params.pl_samples_per_candidate;
        pl_sample_index += 64u
    ) {
        let pl_local_index = pl_candidate_index * pl_params.pl_samples_per_candidate + pl_sample_index;
        let pl_global_index = pl_params.pl_state_offset + pl_local_index;
        let pl_row_index = pl_local_index * pl_params.pl_row_stride;
        let pl_field_value = pl_fields[pl_row_index];
        if pl_params.pl_row_stride == 9u {
            let pl_certificate_value = pl_fields[pl_row_index + 1u];
            if pl_certificate_value.x > pl_params.pl_certificate_tolerance
                || pl_certificate_value.y > pl_params.pl_certificate_tolerance
                || pl_certificate_value.z > pl_params.pl_certificate_tolerance
                || pl_certificate_value.w > pl_params.pl_transverse_limit
            {
                pl_invalid_count = 1u;
            }
        }
        var pl_baseline_value = pl_field_value;
        if pl_params.pl_density_model == 0u {
            pl_baseline[pl_global_index] = pl_field_value;
        } else {
            pl_baseline_value = pl_baseline[pl_global_index];
        }
        let pl_delta = pl_field_value.xyz - pl_baseline_value.xyz;
        pl_difference_energy += dot(pl_delta, pl_delta);
        pl_reference_energy += dot(pl_baseline_value.xyz, pl_baseline_value.xyz);
        var pl_gradient_row = 1u;
        if pl_params.pl_row_stride == 9u { pl_gradient_row = 2u; }
        pl_jacobian_energy += dot(
            pl_fields[pl_row_index + pl_gradient_row].xyz,
            pl_fields[pl_row_index + pl_gradient_row].xyz,
        ) + dot(
            pl_fields[pl_row_index + pl_gradient_row + 1u].xyz,
            pl_fields[pl_row_index + pl_gradient_row + 1u].xyz,
        ) + dot(
            pl_fields[pl_row_index + pl_gradient_row + 2u].xyz,
            pl_fields[pl_row_index + pl_gradient_row + 2u].xyz,
        );
        pl_altitude = min(
            pl_altitude,
            length(pl_positions[pl_global_index].xyz) - pl_params.pl_body_radius,
        );
    }
    pl_difference_sum[pl_lane_index] = pl_difference_energy;
    pl_baseline_sum[pl_lane_index] = pl_reference_energy;
    pl_gradient_sum[pl_lane_index] = pl_jacobian_energy;
    pl_minimum_altitude[pl_lane_index] = pl_altitude;
    pl_invalid_sum[pl_lane_index] = pl_invalid_count;
    workgroupBarrier();
    var pl_stride = 32u;
    loop {
        if pl_stride == 0u { break; }
        if pl_lane_index < pl_stride {
            pl_difference_sum[pl_lane_index] += pl_difference_sum[pl_lane_index + pl_stride];
            pl_baseline_sum[pl_lane_index] += pl_baseline_sum[pl_lane_index + pl_stride];
            pl_gradient_sum[pl_lane_index] += pl_gradient_sum[pl_lane_index + pl_stride];
            pl_minimum_altitude[pl_lane_index] = min(
                pl_minimum_altitude[pl_lane_index],
                pl_minimum_altitude[pl_lane_index + pl_stride],
            );
            pl_invalid_sum[pl_lane_index] += pl_invalid_sum[pl_lane_index + pl_stride];
        }
        workgroupBarrier();
        pl_stride >>= 1u;
    }
    if pl_lane_index == 0u {
        var pl_reduced_difference = pl_difference_sum[0];
        if pl_invalid_sum[0] != 0u { pl_reduced_difference = -1.0; }
        pl_metrics[pl_candidate_index] = vec4<f32>(
            pl_reduced_difference,
            pl_baseline_sum[0],
            pl_minimum_altitude[0],
            pl_gradient_sum[0],
        );
    }
}
