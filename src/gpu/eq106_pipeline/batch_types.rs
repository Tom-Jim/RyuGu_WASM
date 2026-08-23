// GPU-resident Equation (106) engineering evaluator.
//
// Expensive source traversal and transverse Taylor assembly are performed only
// when the reference line changes. Real-time frames reuse the active-order
// coefficient spectra and never revisit the source buffer.
//
// The pass is deliberately documented as an engineering approximation to the
// derivation in `docs/mathtidy.md`: it uses fixed quadrature over the
// mass-preserving source representation and a complete two-dimensional
// transverse Taylor jet. Runtime guards shorten and rebuild a segment when its
// spectral or truncation certificate is rejected.

use crate::interface::components::*;
use crate::cpu::curved_arc::{AggregatedGravitySource, CurvedArcPlannerState};
use crate::cpu::eq106_operator::Eq106OperatorTensorResource;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor,
        BufferInitDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
        CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
        PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bevy::shader::ShaderCacheError;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use wgpu29::{ComputePassTimestampWrites, QuerySet};

const HALF_COUNT: u32 = 64;
const FREQUENCY_COUNT: u32 = 2 * HALF_COUNT + 1;
const QUADRATURE_COUNT: u32 = 64;
const TAYLOR_MAX_ORDER: u32 = 4;
const MAX_TAYLOR_COEFFICIENT_COUNT: u32 = 15;
const DUAL_CERTIFICATE_CADENCE: u32 = 30;
const OUTPUT_ROWS_PER_BLOCK: u64 = 9;
const OUTPUT_BYTES: u64 = OUTPUT_ROWS_PER_BLOCK * 16;
const TAYLOR_REMAINDER_TARGET: f32 = 1.0e-3;
const TIMESTAMP_BYTES: u64 = 8;
const TARGET_DISPATCH_WIDTH: u32 = 65_535;

fn taylor_coefficient_count(order: u32) -> u32 {
    let order = order.clamp(1, TAYLOR_MAX_ORDER);
    ((order + 1) * (order + 2) / 2).min(MAX_TAYLOR_COEFFICIENT_COUNT)
}

fn target_dispatch_grid(target_count: u32) -> (u32, u32) {
    (
        target_count.min(TARGET_DISPATCH_WIDTH).max(1),
        target_count.div_ceil(TARGET_DISPATCH_WIDTH).max(1),
    )
}

#[derive(Clone, Debug, Default)]
struct Eq106TimingLayout {
    build_pairs: Vec<(u32, u32)>,
    evaluation_pairs: Vec<(u32, u32)>,
    readback_pair: Option<(u32, u32)>,
    query_count: u32,
}

fn timestamp_writes<'a>(
    query_set: Option<&'a QuerySet>,
    beginning: Option<u32>,
    end: Option<u32>,
) -> Option<ComputePassTimestampWrites<'a>> {
    query_set.map(|query_set| ComputePassTimestampWrites {
        query_set,
        beginning_of_pass_write_index: beginning,
        end_of_pass_write_index: end,
    })
}

fn decode_gpu_timings(
    bytes: &[u8],
    timestamp_period_ns: f32,
    layout: &Eq106TimingLayout,
    cpu_readback_wait_ms: f64,
    target_count: u32,
    spectral_element_count: u32,
) -> Eq106TimingSample {
    let timestamps = bytes
        .chunks_exact(TIMESTAMP_BYTES as usize)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let elapsed_ms = |pairs: &[(u32, u32)]| {
        (!pairs.is_empty())
            .then(|| {
                pairs.iter().try_fold(0.0, |sum, &(begin, end)| {
                    let begin = *timestamps.get(begin as usize)?;
                    let end = *timestamps.get(end as usize)?;
                    (end >= begin).then_some(sum + (end - begin) as f64)
                })
            })
            .flatten()
            .map(|ticks| ticks * timestamp_period_ns as f64 / 1.0e6)
    };
    let readback = layout
        .readback_pair
        .and_then(|(begin, end)| elapsed_ms(&[(begin, end)]));
    Eq106TimingSample {
        spectrum_build_ms: elapsed_ms(&layout.build_pairs),
        target_evaluation_ms: elapsed_ms(&layout.evaluation_pairs),
        gpu_readback_copy_ms: readback,
        cpu_readback_wait_ms,
        target_count,
        spectral_element_count,
    }
}

#[derive(Clone, Copy, Debug)]
struct Eq106BatchElement {
    target_offset: u32,
    target_count: u32,
    line_origin: Vec3,
    line_direction: Vec3,
    line_limit: f32,
    taylor_order: u32,
}

fn select_batch_taylor_order(epsilon: f32) -> Option<u32> {
    if !epsilon.is_finite() || !(0.0..1.0).contains(&epsilon) {
        return None;
    }
    (1..=TAYLOR_MAX_ORDER)
        .find(|order| epsilon.powi(*order as i32 + 1) / (1.0 - epsilon) <= TAYLOR_REMAINDER_TARGET)
}

fn build_trajectory_batch_elements(
    positions: &[Vec3],
    velocities: &[Vec3],
    source_radius: f32,
    certified_line_limit: f32,
) -> Vec<Eq106BatchElement> {
    if positions.is_empty() || positions.len() != velocities.len() {
        return Vec::new();
    }
    let maximum_line_limit = certified_line_limit
        .min(4.0 * source_radius)
        .max(0.35 * source_radius);
    let mut elements = Vec::new();
    let mut start = 0;
    while start < positions.len() {
        let origin = positions[start];
        let mut direction = velocities[start].normalize_or_zero();
        if direction == Vec3::ZERO
            && let Some(next) = positions.get(start + 1)
        {
            direction = (*next - origin).normalize_or_zero();
        }
        if direction == Vec3::ZERO {
            break;
        }
        let mut maximum_h = 0.0_f32;
        let mut maximum_offset = 0.0_f32;
        let mut minimum_line_radius = f32::INFINITY;
        let mut best_count = 0_u32;
        let mut best_order = 1_u32;
        let mut end = start;
        while end < positions.len() {
            let position = positions[end];
            let relative = position - origin;
            let h = relative.dot(direction);
            if h < -1.0e-3 || h > maximum_line_limit {
                break;
            }
            let line_point = origin + h.max(0.0) * direction;
            let next_maximum_offset = maximum_offset.max(position.distance(line_point));
            let next_minimum_line_radius = minimum_line_radius.min(line_point.length());
            let distance_lower_bound = next_minimum_line_radius - source_radius;
            if distance_lower_bound <= 0.0 {
                break;
            }
            let epsilon = next_maximum_offset / distance_lower_bound;
            let Some(order) = select_batch_taylor_order(epsilon) else {
                break;
            };
            maximum_h = maximum_h.max(h);
            maximum_offset = next_maximum_offset;
            minimum_line_radius = next_minimum_line_radius;
            best_count = (end - start + 1) as u32;
            best_order = order;
            end += 1;
        }
        if best_count == 0 {
            break;
        }
        elements.push(Eq106BatchElement {
            target_offset: start as u32,
            target_count: best_count,
            line_origin: origin,
            line_direction: direction,
            line_limit: (maximum_h / 0.85)
                .max(0.35 * source_radius)
                .min(maximum_line_limit)
                .max(1.0),
            taylor_order: best_order,
        });
        start += best_count as usize;
    }
    elements
}

#[derive(Resource, Default)]
struct ExtractedEq106Input {
    enabled: bool,
    probe: Vec3,
    velocity: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    target_bytes: Vec<u8>,
    target_snapshots: Vec<GravityRequestSnapshot>,
    batch_elements: Vec<Eq106BatchElement>,
    batch_capture_id: Option<u64>,
    sensitivity_column: Option<u32>,
    sources: Option<Vec<u8>>,
    fourier_modes: Option<Vec<u8>>,
    operator_tensor: Option<Vec<u8>>,
    psi_operator: Option<Vec<u8>>,
    source_count: u32,
    density_mode_count: u32,
    radius: f32,
    source_hash: u64,
    certified_line_limit: f32,
    taylor_order: u32,
}

#[derive(Resource, Default)]
struct Eq106GpuBuffers(Option<Eq106GpuBuffersInner>);

struct Eq106GpuBuffersInner {
    uniform: Buffer,
    targets: Buffer,
    output: Buffer,
    staging: Buffer,
    output_size: u64,
    bind_group: BindGroup,
    layout: BindGroupLayout,
    sources: Buffer,
    quadrature: Buffer,
    spectrum: Buffer,
    operator_tensor: Buffer,
    line_samples: Buffer,
    density_modes: Buffer,
    psi_operator: Buffer,
    timing_query_set: Option<QuerySet>,
    timing_resolve: Option<Buffer>,
    source_count: u32,
    density_mode_count: u32,
    element_capacity: u32,
    line_origin: Vec3,
    line_direction: Vec3,
    segment_id: u32,
    source_hash: u64,
    spectrum_ready: bool,
    line_scale: f32,
    taylor_order: u32,
    target_count: u32,
    dual_certificate_frame: u32,
    last_submitted: Option<(u64, u64)>,
}

#[derive(Resource)]
struct Eq106ComputePipeline {
    line_samples_id: CachedComputePipelineId,
    assemble_id: CachedComputePipelineId,
    analytic_id: CachedComputePipelineId,
    evaluate_id: CachedComputePipelineId,
}

pub struct Eq106GpuComputePlugin;

impl Plugin for Eq106GpuComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Eq106GpuReadbackChannel>();
        app.init_resource::<Eq106GpuHistory>();
        app.init_resource::<Eq106TrajectoryBatchResult>();
        app.init_resource::<Eq106SensitivityMatrix>();
        app.init_resource::<Eq106PerformanceMetrics>();
        app.add_systems(PreUpdate, poll_eq106_readback);
        app.add_systems(Update, clear_eq106_history_on_probe_reset);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedEq106Input>();
        render_app.init_resource::<Eq106GpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_eq106_input);
        render_app.add_systems(
            Render,
            (initialize_eq106_pipeline, dispatch_eq106)
                .chain()
                .in_set(RenderSystems::Render),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<Eq106GpuReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
    }
}

impl FromWorld for Eq106ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            storage_rw_entry(4),
            storage_ro_entry(5),
            storage_rw_entry(6),
            storage_ro_entry(7),
            storage_ro_entry(8),
            storage_ro_entry(9),
        ];
        let layout = BindGroupLayoutDescriptor::new("eq106_complex_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/eq106_complex.wgsl");
        let cache = world.resource::<PipelineCache>();
        let line_samples_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_line_samples".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("assemble_line_samples".into()),
            zero_initialize_workgroup_memory: false,
        });
        let assemble_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_spectrum".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("assemble_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let analytic_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_analytic_spectrum".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("assemble_analytic_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let evaluate_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_evaluate_field".into()),
            layout: vec![layout],
            immediate_size: 0,
            shader,
            shader_defs: vec![],
            entry_point: Some("evaluate_field".into()),
            zero_initialize_workgroup_memory: false,
        });
        Self {
            line_samples_id,
            assemble_id,
            analytic_id,
            evaluate_id,
        }
    }
}

fn clear_eq106_history_on_probe_reset(
    probe: Res<ProbeInitialConditions>,
    mut history: ResMut<Eq106GpuHistory>,
) {
    if probe.is_changed() {
        history.0.clear();
    }
}

fn poll_eq106_readback(
    channel: Res<Eq106GpuReadbackChannel>,
    mut history: ResMut<Eq106GpuHistory>,
    mut batch_result: ResMut<Eq106TrajectoryBatchResult>,
    mut sensitivity: ResMut<Eq106SensitivityMatrix>,
    mut performance: ResMut<Eq106PerformanceMetrics>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if let Ok(mut error) = channel.pipeline_error.try_lock()
        && let Some(message) = error.take()
    {
        runtime_error.raise(message);
        return;
    }
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    let Some(packet) = guard.take() else { return };
    performance.latest = Some(packet.timings);
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let decoded = match decode_eq106_packet(&packet, angular_velocity_world) {
        Ok(decoded) => decoded,
        Err(Eq106DecodeError::Incomplete) => {
            runtime_error.raise("Equation (106) batch readback is incomplete.");
            return;
        }
        Err(Eq106DecodeError::InvalidSample) => {
            channel.rebuild_requested.store(true, Ordering::Release);
            return;
        }
    };

    if let (Some(capture_id), Some(column)) =
        (packet.batch_capture_id, packet.sensitivity_column)
    {
        if sensitivity.capture_id != Some(capture_id) {
            sensitivity.capture_id = Some(capture_id);
            sensitivity.columns.clear();
            sensitivity.sample_count = decoded.len();
        }
        // `start_density_inversion_system` installs the capture identity before
        // the first GPU column returns. Initialize the row count from that
        // first column even when the capture identity already matches.
        if column == 0 && sensitivity.columns.is_empty() {
            sensitivity.sample_count = decoded.len();
        }
        if sensitivity.sample_count != decoded.len() {
            runtime_error.raise(format!(
                "Equation (106) sensitivity column {column} returned {} samples; expected {}.",
                decoded.len(),
                sensitivity.sample_count,
            ));
            sensitivity.columns.clear();
            sensitivity.sample_count = 0;
            return;
        }
        if sensitivity.columns.len() == column as usize {
            sensitivity.columns.push(
                decoded
                    .into_iter()
                    .map(|sample| {
                        sample.snapshot.ryugu_transform.rotation * sample.body_acceleration
                    })
                    .collect(),
            );
        }
    } else if let Some(capture_id) = packet.batch_capture_id {
        batch_result.capture_id = Some(capture_id);
        batch_result.samples = decoded;
    } else if let Some(sample) = decoded.into_iter().next() {
        history.0.push(sample);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Eq106DecodeError {
    Incomplete,
    InvalidSample,
}

fn decode_eq106_packet(
    packet: &Eq106ReadbackPacket,
    angular_velocity_world: Vec3,
) -> Result<Vec<GravityFieldSample>, Eq106DecodeError> {
    let expected_rows = packet.snapshots.len() * OUTPUT_ROWS_PER_BLOCK as usize;
    if packet.partial_sums.len() != expected_rows {
        return Err(Eq106DecodeError::Incomplete);
    }
    packet
        .snapshots
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            let start = index * OUTPUT_ROWS_PER_BLOCK as usize;
            decode_eq106_sample(
                &packet.partial_sums[start..start + OUTPUT_ROWS_PER_BLOCK as usize],
                snapshot,
                angular_velocity_world,
            )
            .ok_or(Eq106DecodeError::InvalidSample)
        })
        .collect()
}

fn decode_eq106_sample(
    rows: &[[f32; 4]],
    base_snapshot: &GravityRequestSnapshot,
    angular_velocity_world: Vec3,
) -> Option<GravityFieldSample> {
    let field = rows[0];
    let certificate = rows[1];
    if certificate[0] > GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
        || certificate[1] > GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
        || certificate[2] > GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
        || certificate[3] > 0.30
    {
        return None;
    }
    let potentials = rows[6];
    let anchor_row = rows[7];
    let origin_row = rows[8];
    let acceleration = Vec3::new(field[0], field[1], field[2]);
    let positive_potential = potentials[0];
    let independent_potential = (potentials[2] > 0.5).then_some(potentials[1]);
    let elapsed = potentials[3] as f64;
    let anchor_position = Vec3::new(anchor_row[0], anchor_row[1], anchor_row[2]);
    let line_origin = Vec3::new(origin_row[0], origin_row[1], origin_row[2]);
    let local_coordinates = rows[5];
    let jacobian = Mat3::from_cols(
        Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
        Vec3::new(rows[3][0], rows[3][1], rows[3][2]),
        Vec3::new(rows[4][0], rows[4][1], rows[4][2]),
    );
    if !acceleration.is_finite()
        || acceleration.abs().max_element() > 1.0e12
        || !positive_potential.is_finite()
        || positive_potential <= 0.0
        || positive_potential > 1.0e20
        || independent_potential.is_some_and(|potential| !potential.is_finite() || potential <= 0.0)
        || !anchor_position.is_finite()
        || !line_origin.is_finite()
        || !local_coordinates.iter().all(|value| value.is_finite())
        || !jacobian.is_finite()
    {
        return None;
    }

    let mut snapshot = base_snapshot.clone();
    snapshot.simulation_time_seconds += elapsed;
    snapshot.body_position = anchor_position;
    let future_rotation = Quat::from_axis_angle(
        RYUGU_SPIN_AXIS.normalize(),
        std::f32::consts::TAU * elapsed as f32 / RYUGU_ROTATION_PERIOD_SECS,
    ) * base_snapshot.ryugu_transform.rotation;
    snapshot.ryugu_transform.rotation = future_rotation;
    snapshot.probe_position =
        snapshot.ryugu_transform.translation + future_rotation * anchor_position;

    Some(GravityFieldSample {
        snapshot,
        predictive: false,
        body_acceleration: acceleration,
        positive_potential,
        #[cfg(feature = "eq106-dual-certificate")]
        independent_positive_potential: independent_potential,
        body_acceleration_jacobian: Some(jacobian),
        eq106_diagnostics: Some(Eq106SampleDiagnostics {
            segment_id: local_coordinates[3].round().max(0.0) as u64,
            line_origin,
            line_direction: (base_snapshot.ryugu_transform.rotation.inverse()
                * (base_snapshot.probe_velocity
                    - angular_velocity_world.cross(
                        base_snapshot.probe_position - base_snapshot.ryugu_transform.translation,
                    )))
            .normalize_or_zero(),
            h: local_coordinates[0],
            u: local_coordinates[1],
            v: local_coordinates[2],
            certificates: certificate,
        }),
    })
}
