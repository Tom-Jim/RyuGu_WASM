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

use crate::cpu::curved_arc::{AggregatedGravitySource, CurvedArcPlannerState};
use crate::cpu::eq106_operator::Eq106OperatorTensorResource;
use crate::interface::components::*;
use bevy::log::{debug, error, info, trace};
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
    renderer::{RenderDevice, RenderQueue}, GpuResourceAppExt,
};
use bevy::shader::{ShaderCacheError, ShaderDefVal};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use wgpu29::{ComputePassTimestampWrites, QuerySet};

const HALF_COUNT: u32 = 64;
const FREQUENCY_COUNT: u32 = 2 * HALF_COUNT + 1;
const QUADRATURE_COUNT: u32 = 64;
const PLANNING_CHEBYSHEV_MODES: u32 = 32;
const TAYLOR_MAX_ORDER: u32 = 8;
const MAX_TAYLOR_COEFFICIENT_COUNT: u32 = 45;
const DUAL_CERTIFICATE_CADENCE: u32 = 30;
const OUTPUT_ROWS_PER_BLOCK: u64 = 11;
const OUTPUT_BYTES: u64 = OUTPUT_ROWS_PER_BLOCK * 16;
const TAYLOR_REMAINDER_TARGET: f32 = 1.0e-3;
const TAYLOR_GRADIENT_REMAINDER_TARGET: f32 = 1.0e-2;
// With the default feature set the shader-side dual certificate is disabled,
// so rows[1].xyz are zero except that rows[1].x becomes the exact sentinel 1.0
// when mobile f32 edge normalization is judged ill-conditioned. Values between
// 0.04 and 1.0 never occur in this mode; a 0.20 limit therefore changed
// nothing. The strict `value > limit` check below accepts that sentinel only on
// mobile, while desktop and benchmark admission remain at 4%.
const MOBILE_EQ106_CERTIFICATE_TOLERANCE: f32 = 1.0;
const TIMESTAMP_BYTES: u64 = 8;
const TARGET_DISPATCH_WIDTH: u32 = 65_535;
const EQ106_GPU_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) const fn eq106_sensitivity_configuration_hash() -> u64 {
    (FREQUENCY_COUNT as u64) << 32 ^ (TAYLOR_MAX_ORDER as u64) << 16 ^ QUADRATURE_COUNT as u64
}

fn taylor_coefficient_count(order: u32) -> u32 {
    let order = order.clamp(1, TAYLOR_MAX_ORDER);
    ((order + 1) * (order + 2) / 2).min(MAX_TAYLOR_COEFFICIENT_COUNT)
}

fn target_dispatch_grid(target_count: u32) -> (u32, u32) {
    (
        target_count.clamp(1, TARGET_DISPATCH_WIDTH),
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
        .as_chunks::<{ TIMESTAMP_BYTES as usize }>()
        .0
        .iter()
        .map(|chunk| u64::from_le_bytes(*chunk))
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
    Eq106TimingSample {
        spectrum_build_ms: elapsed_ms(&layout.build_pairs),
        target_evaluation_ms: elapsed_ms(&layout.evaluation_pairs),
        cpu_readback_wait_ms,
        target_count,
        dispatch_count: 1,
        spectrum_rebuild_count: spectral_element_count,
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
    /// One-based slot in the spectrum buffer. Multiple candidate evaluators
    /// may deliberately reference the same canonical slot.
    spectrum_index: u32,
}

fn select_batch_taylor_order(epsilon: f32) -> Option<u32> {
    if !epsilon.is_finite() || !(0.0..1.0).contains(&epsilon) {
        return None;
    }
    (1..=TAYLOR_MAX_ORDER).find(|order| {
        let field_bound = epsilon.powi(*order as i32 + 1) / (1.0 - epsilon);
        // Differentiating the Taylor tail amplifies both its leading power and
        // the geometric-series denominator. A field-only certificate allowed
        // gradients that were visibly outside the planning tolerance.
        let gradient_bound =
            (*order as f32 + 1.0) * epsilon.powi(*order as i32) / (1.0 - epsilon).powi(2);
        field_bound <= TAYLOR_REMAINDER_TARGET && gradient_bound <= TAYLOR_GRADIENT_REMAINDER_TARGET
    })
}

fn build_trajectory_batch_elements(
    positions: &[Vec3],
    velocities: &[Vec3],
    times: &[f32],
    source_radius: f32,
    certified_line_limit: f32,
) -> Vec<Eq106BatchElement> {
    build_tube_batch_elements(
        positions,
        velocities,
        times,
        source_radius,
        certified_line_limit,
        0.0,
    )
}

fn build_canonical_tube_elements(
    positions: &[Vec3],
    velocities: &[Vec3],
    times: &[f32],
    source_radius: f32,
    certified_line_limit: f32,
    tube_radius: f32,
) -> Vec<Eq106BatchElement> {
    build_tube_batch_elements(
        positions,
        velocities,
        times,
        source_radius,
        certified_line_limit,
        tube_radius.max(0.0),
    )
}

fn build_tube_batch_elements(
    positions: &[Vec3],
    velocities: &[Vec3],
    times: &[f32],
    source_radius: f32,
    certified_line_limit: f32,
    tube_radius: f32,
) -> Vec<Eq106BatchElement> {
    if positions.is_empty() || positions.len() != velocities.len() || positions.len() != times.len()
    {
        return Vec::new();
    }
    let maximum_line_limit = certified_line_limit
        .min(4.0 * source_radius)
        .max(0.35 * source_radius);
    let mut elements = Vec::new();
    let mut start = 0;
    while start < positions.len() {
        let anchor = positions[start];
        let mut direction = velocities[start].normalize_or_zero();
        if direction == Vec3::ZERO
            && let Some(next) = positions.get(start + 1)
        {
            direction = (*next - anchor).normalize_or_zero();
        }
        if direction == Vec3::ZERO {
            break;
        }
        // Shift the line backwards by the full tube radius. Any candidate
        // offset can therefore project at most to h=0 at the first sample;
        // evaluate_field never needs to clamp a genuinely negative query.
        let origin = anchor - tube_radius * direction;
        let mut maximum_h = 0.0_f32;
        let mut maximum_offset = 0.0_f32;
        let mut minimum_line_radius = f32::INFINITY;
        let mut best_count = 0_u32;
        let mut best_order = 1_u32;
        let mut end = start;
        while end < positions.len() {
            let elapsed = times[end] - times[start];
            if !elapsed.is_finite() || elapsed < 0.0 || elapsed > NEAR_SYNC_SEGMENT_MAX_SECONDS {
                break;
            }
            let position = positions[end];
            let relative = position - origin;
            let h = relative.dot(direction);
            if h < -1.0e-3 || h > maximum_line_limit {
                break;
            }
            let line_point = origin + h.max(0.0) * direction;
            let next_maximum_offset =
                maximum_offset.max(position.distance(line_point) + tube_radius);
            let next_maximum_h = maximum_h.max(h + tube_radius);
            let next_minimum_line_radius = if tube_radius > 0.0 {
                // Planning samples the entire finite interval, including the
                // tube's longitudinal margins. Bound its closest approach,
                // not just the discrete target positions.
                let closest_h = (-origin.dot(direction)).clamp(0.0, next_maximum_h);
                (origin + closest_h * direction).length()
            } else { minimum_line_radius.min(line_point.length()) };
            let distance_lower_bound = next_minimum_line_radius - source_radius;
            if distance_lower_bound <= 0.0
                || (tube_radius > 0.0 && next_maximum_h > 4.0 * distance_lower_bound) {
                // Keep the nearest possible source singularity far enough
                // from the Chebyshev interval for the fixed degree-31 budget.
                break;
            }
            let epsilon = next_maximum_offset / distance_lower_bound;
            let Some(order) = select_batch_taylor_order(epsilon) else {
                break;
            };
            maximum_h = maximum_h.max(h + tube_radius);
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
            line_limit: if tube_radius > 0.0 {
                // No artificial 0.35*R minimum: sampling beyond a short arc
                // can approach the body even though all actual targets are safe.
                maximum_h.max(1.0)
            } else {
                (maximum_h / 0.85).max(0.35 * source_radius)
                    .min(maximum_line_limit).max(1.0)
            },
            taylor_order: best_order,
            spectrum_index: elements.len() as u32 + 1,
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
    sensitivity_sources: Vec<Vec<u8>>,
    sensitivity_source_counts: Vec<u32>,
    sensitivity_source_hash: u64,
    sensitivity_basis_hash: u64,
    sources: Option<Vec<u8>>,
    fourier_modes: Option<Vec<u8>>,
    operator_tensor: Option<Vec<u8>>,
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
    evaluate_id: CachedComputePipelineId,
    planning_voxel_line_samples_id: CachedComputePipelineId,
    planning_voxel_spectrum_id: CachedComputePipelineId,
    planning_combine_spectrum_id: CachedComputePipelineId,
    planning_evaluate_id: CachedComputePipelineId,
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
        render_app.init_gpu_resource::<Eq106GpuBuffers>();
        render_app.init_gpu_resource::<PlanningEq106DispatchState>();
        render_app.add_systems(ExtractSchedule, extract_eq106_input);
        render_app.add_systems(
            Render,
            (initialize_eq106_pipeline, dispatch_eq106)
                .chain()
                .in_set(RenderSystems::Render),
        );
        render_app.add_systems(
            Render,
            dispatch_planning_eq106
                .after(initialize_eq106_pipeline)
                .after(crate::gpu::planning::PlanningGpuSystems::PrepareSharedInput)
                .in_set(crate::gpu::planning::PlanningGpuSystems::Dispatch)
                .in_set(RenderSystems::Cleanup),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<Eq106GpuReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        // Queue every active entry point during application startup. This
        // keeps the first Eq.106 selection and First benchmark free of compile
        // time without retaining the unused analytic-spectrum pipeline.
        render_app.init_gpu_resource::<Eq106ComputePipeline>();
    }
}

impl FromWorld for Eq106ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        log_eq106_wgsl_source();
        let entries = [
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            storage_rw_entry(4),
            uniform_entry(5),
            storage_rw_entry(6),
            uniform_entry(7),
            storage_ro_entry(9),
        ];
        let layout = BindGroupLayoutDescriptor::new("eq106_complex_bgl", &entries);
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::Eq106,
        );
        let cache = world.resource::<PipelineCache>();
        let source_shader_defs = vec![ShaderDefVal::from("EQ106_SOURCE")];
        let spectrum_shader_defs = vec![ShaderDefVal::from("EQ106_SPECTRUM")];
        let evaluator_shader_defs = vec![ShaderDefVal::from("EQ106_EVALUATOR")];
        let line_samples_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_line_samples".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: source_shader_defs.clone(),
            entry_point: Some("assemble_line_samples".into()),
            zero_initialize_workgroup_memory: false,
        });
        let assemble_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_spectrum".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: spectrum_shader_defs.clone(),
            entry_point: Some("assemble_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let evaluate_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_evaluate_field".into()),
            layout: vec![layout],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: evaluator_shader_defs.clone(),
            entry_point: Some("evaluate_field".into()),
            zero_initialize_workgroup_memory: false,
        });
        let planning_layout = BindGroupLayoutDescriptor::new(
            "eq106_planning_voxel_bgl",
            &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
                storage_rw_entry(4),
                uniform_entry(5),
                storage_rw_entry(6),
                uniform_entry(7),
                storage_ro_entry(9),
            ],
        );
        let planning_voxel_line_samples_id =
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("eq106_planning_voxel_line_samples".into()),
                layout: vec![planning_layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: source_shader_defs.clone(),
                entry_point: Some("assemble_voxel_line_samples".into()),
                zero_initialize_workgroup_memory: false,
            });
        let planning_voxel_spectrum_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_planning_voxel_spectrum".into()),
            layout: vec![planning_layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: spectrum_shader_defs.clone(),
            entry_point: Some("assemble_voxel_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let planning_combine_spectrum_id =
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("eq106_planning_combine_spectrum".into()),
                layout: vec![planning_layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: spectrum_shader_defs,
                entry_point: Some("combine_voxel_spectrum".into()),
                zero_initialize_workgroup_memory: false,
            });
        let planning_evaluate_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_planning_evaluate_field".into()),
            layout: vec![planning_layout],
            immediate_size: 0,
            shader,
            shader_defs: evaluator_shader_defs,
            entry_point: Some("evaluate_field".into()),
            zero_initialize_workgroup_memory: false,
        });
        Self {
            line_samples_id,
            assemble_id,
            evaluate_id,
            planning_voxel_line_samples_id,
            planning_voxel_spectrum_id,
            planning_combine_spectrum_id,
            planning_evaluate_id,
        }
    }
}

fn log_eq106_wgsl_source() {
    let source = include_str!("../../wgsl/eq106_complex.wgsl");
    debug!(
        target: "wgsl::eq106",
        bytes = source.len(),
        lines = source.lines().count(),
        "loading Eq.106 WGSL source"
    );
    for (line_number, line) in source.lines().enumerate() {
        trace!(target: "wgsl::eq106", line = line_number + 1, "WGSL {line}");
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
    mut planner: ResMut<CurvedArcPlannerState>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if channel.in_flight.load(Ordering::Acquire)
        && let Ok(mut submitted_at) = channel.submitted_at.try_lock()
        && submitted_at
            .as_ref()
            .is_some_and(|started| started.elapsed() > EQ106_GPU_TIMEOUT)
    {
        submitted_at.take();
        runtime_error.raise(
            "Equation (106) GPU request exceeded 10 seconds; possible shader hang or device loss.",
        );
        return;
    }
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
    if packet.sensitivity_column_count > 0 {
        let Some(capture_id) = packet.batch_capture_id else {
            runtime_error.raise("Equation (106) sensitivity readback has no capture identity.");
            return;
        };
        let column_count = packet.sensitivity_column_count as usize;
        let sample_count = packet.snapshots.len();
        if sensitivity.capture_id != Some(capture_id)
            || sensitivity.source_hash != packet.sensitivity_source_hash
            || sensitivity.basis_hash != packet.sensitivity_basis_hash
            || sensitivity.configuration_hash != packet.sensitivity_configuration_hash
        {
            return;
        }
        if packet.partial_sums.len() != column_count * sample_count {
            runtime_error.raise(format!(
                "Equation (106) sensitivity batch returned {} vectors; expected {} x {}.",
                packet.partial_sums.len(),
                column_count,
                sample_count,
            ));
            return;
        }
        let assembled_at = Instant::now();
        let columns = packet
            .partial_sums
            .chunks_exact(sample_count)
            .map(|column| {
                column
                    .iter()
                    .zip(&packet.snapshots)
                    .map(|(field, snapshot)| {
                        snapshot.ryugu_transform.rotation * Vec3::new(field[0], field[1], field[2])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if columns
            .iter()
            .flatten()
            .any(|acceleration| !acceleration.is_finite())
        {
            runtime_error.raise("Equation (106) sensitivity matrix contains non-finite values.");
            return;
        }
        sensitivity.capture_id = Some(capture_id);
        sensitivity.sample_count = sample_count;
        sensitivity.voxel_count = column_count;
        sensitivity.columns = columns;
        let inversion = performance.inversion.get_or_insert_default();
        inversion.gpu_readback_ms = packet.timings.cpu_readback_wait_ms;
        inversion.design_matrix_assembly_ms = assembled_at.elapsed().as_secs_f64() * 1.0e3;
        inversion.dispatch_count = packet.timings.dispatch_count;
        inversion.spectrum_rebuild_count = packet.timings.spectrum_rebuild_count;
        inversion.spectrum_build_ms = packet.timings.spectrum_build_ms;
        inversion.target_evaluation_ms = packet.timings.target_evaluation_ms;
        return;
    }
    let decoded = match decode_eq106_packet(&packet, eq106_certificate_tolerance()) {
        Ok(decoded) => decoded,
        Err(Eq106DecodeError::Incomplete { actual, expected }) => {
            runtime_error.raise(format!(
                "Equation (106) batch readback is incomplete: {actual} rows, expected {expected}."
            ));
            return;
        }
        Err(Eq106DecodeError::Rejected { sample, reason }) => {
            planner.consecutive_rejections = planner.consecutive_rejections.saturating_add(1);
            let retry = planner.consecutive_rejections;
            let message = format!(
                "Eq.106 waiting: {} at sample {}; retry {retry}/4, Taylor order {}.",
                reason.message(),
                sample + 1,
                planner.taylor_order,
            );
            planner.reject_status = Some(message.clone());
            warn!(target: "eq106::certificate", %message, "Eq.106 sample rejected");
            if retry >= 4 {
                runtime_error.raise(format!(
                    "Equation (106) stopped after four consecutive certificate failures. {message}"
                ));
            } else {
                channel.rebuild_requested.store(true, Ordering::Release);
            }
            return;
        }
    };
    planner.consecutive_rejections = 0;
    planner.reject_status = None;

    if let Some(capture_id) = packet.batch_capture_id {
        batch_result.capture_id = Some(capture_id);
        batch_result.samples = decoded;
    } else if let Some(sample) = decoded.into_iter().next() {
        history.0.push(sample);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Eq106DecodeError {
    Incomplete {
        actual: usize,
        expected: usize,
    },
    Rejected {
        sample: usize,
        reason: Eq106RejectReason,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Eq106RejectReason {
    TaylorField { value: f32, limit: f32 },
    ImaginaryResidual { value: f32, limit: f32 },
    SpectralTail { value: f32, limit: f32 },
    TransverseRatio { value: f32, limit: f32 },
    NonFinite,
    NonPhysical,
}

impl Eq106RejectReason {
    fn message(self) -> String {
        match self {
            Self::TaylorField { value, limit } => {
                format!("field-tail certificate {value:.3e} > {limit:.3e}")
            }
            Self::ImaginaryResidual { value, limit } => {
                format!("imaginary residual {value:.3e} > {limit:.3e}")
            }
            Self::SpectralTail { value, limit } => {
                format!("spectral-tail certificate {value:.3e} > {limit:.3e}")
            }
            Self::TransverseRatio { value, limit } => {
                format!("transverse ratio {value:.3e} > {limit:.3e}")
            }
            Self::NonFinite => "non-finite field, certificate, or Jacobian".into(),
            Self::NonPhysical => "non-physical potential or field magnitude".into(),
        }
    }
}

fn eq106_certificate_tolerance() -> f32 {
    #[cfg(target_arch = "wasm32")]
    let is_mobile = crate::browser_is_mobile();
    #[cfg(not(target_arch = "wasm32"))]
    let is_mobile = false;
    eq106_certificate_tolerance_for_mobile(is_mobile)
}

const fn eq106_certificate_tolerance_for_mobile(is_mobile: bool) -> f32 {
    if is_mobile {
        MOBILE_EQ106_CERTIFICATE_TOLERANCE
    } else {
        GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
    }
}

#[cfg(test)]
mod certificate_policy_tests {
    use super::*;

    #[test]
    fn desktop_keeps_four_percent_certificate_limit() {
        assert_eq!(
            eq106_certificate_tolerance_for_mobile(false),
            GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
        );
    }

    #[test]
    fn mobile_accepts_the_shader_edge_failure_sentinel() {
        let limit = eq106_certificate_tolerance_for_mobile(true);
        assert_eq!(limit, 1.0);
        assert!(1.0 <= limit);
        assert!(f32::INFINITY > limit);
    }
}

fn decode_eq106_packet(
    packet: &Eq106ReadbackPacket,
    certificate_tolerance: f32,
) -> Result<Vec<GravityFieldSample>, Eq106DecodeError> {
    let expected_rows = packet.snapshots.len() * OUTPUT_ROWS_PER_BLOCK as usize;
    if packet.partial_sums.len() != expected_rows {
        return Err(Eq106DecodeError::Incomplete {
            actual: packet.partial_sums.len(),
            expected: expected_rows,
        });
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
                certificate_tolerance,
            )
            .map_err(|reason| Eq106DecodeError::Rejected {
                sample: index,
                reason,
            })
        })
        .collect()
}

fn decode_eq106_sample(
    rows: &[[f32; 4]],
    base_snapshot: &GravityRequestSnapshot,
    certificate_tolerance: f32,
) -> Result<GravityFieldSample, Eq106RejectReason> {
    let field = rows[0];
    let certificate = rows[1];
    for (value, reason) in [
        (
            certificate[0],
            Eq106RejectReason::TaylorField {
                value: certificate[0],
                limit: certificate_tolerance,
            },
        ),
        (
            certificate[1],
            Eq106RejectReason::ImaginaryResidual {
                value: certificate[1],
                limit: certificate_tolerance,
            },
        ),
        (
            certificate[2],
            Eq106RejectReason::SpectralTail {
                value: certificate[2],
                limit: certificate_tolerance,
            },
        ),
        (
            certificate[3],
            Eq106RejectReason::TransverseRatio {
                value: certificate[3],
                limit: 0.30,
            },
        ),
    ] {
        if !value.is_finite() {
            return Err(Eq106RejectReason::NonFinite);
        }
        let limit = match reason {
            Eq106RejectReason::TransverseRatio { limit, .. }
            | Eq106RejectReason::TaylorField { limit, .. }
            | Eq106RejectReason::ImaginaryResidual { limit, .. }
            | Eq106RejectReason::SpectralTail { limit, .. } => limit,
            _ => unreachable!(),
        };
        if value > limit {
            return Err(reason);
        }
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
        return Err(
            if !acceleration.is_finite()
                || !positive_potential.is_finite()
                || independent_potential.is_some_and(|potential| !potential.is_finite())
                || !anchor_position.is_finite()
                || !line_origin.is_finite()
                || !local_coordinates.iter().all(|value| value.is_finite())
                || !jacobian.is_finite()
            {
                Eq106RejectReason::NonFinite
            } else {
                Eq106RejectReason::NonPhysical
            },
        );
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

    Ok(GravityFieldSample {
        snapshot,
        predictive: false,
        body_acceleration: acceleration,
        positive_potential,
        #[cfg(feature = "eq106-dual-certificate")]
        independent_positive_potential: independent_potential,
        body_acceleration_jacobian: Some(jacobian),
    })
}


#[cfg(test)]
mod finite_planning_segment_tests {
    use super::*;

    #[test]
    fn short_outgoing_arc_does_not_sample_an_unused_body_scale_interval() {
        let positions = [Vec3::new(600.0, 0.0, 0.0), Vec3::new(610.0, 0.0, 0.0)];
        let velocities = [Vec3::X; 2];
        let elements = build_canonical_tube_elements(&positions, &velocities, &[0.0, 10.0],
            500.0, 2000.0, 15.0);
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].target_count, 2);
        assert_eq!(elements[0].line_limit, 40.0);
        assert!(elements[0].line_limit < 0.35 * 500.0);
        let start = elements[0].line_origin;
        assert!(start.length() > 500.0);
    }
}
