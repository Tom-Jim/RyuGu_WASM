// GPU implementation of the complete frequency-domain forward operator.
// Density spectra and the Fourier-Laplace trajectory characteristic are
// evaluated on the device before reciprocal-space reduction.

use crate::cpu::frequency_domain::{
    AggregatedGravitySource, EQ184_BASE_LAPLACE_SIGMA, EQ184_QUADRATURE_COUNT,
    eq184_quadrature_node,
};
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

const QUADRATURE_COUNT: u32 = EQ184_QUADRATURE_COUNT as u32;
const FREQUENCY_COUNT: u32 = QUADRATURE_COUNT;
const OUTPUT_ROWS_PER_BLOCK: u64 = 11;
const OUTPUT_BYTES: u64 = OUTPUT_ROWS_PER_BLOCK * 16;
const FREQUENCY_DOMAIN_GPU_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) const fn frequency_domain_sensitivity_configuration_hash() -> u64 {
    (FREQUENCY_COUNT as u64) << 32 ^ QUADRATURE_COUNT as u64
}

#[derive(Clone, Copy, Debug)]
struct FrequencyDomainBatchElement {
    target_offset: u32,
    target_count: u32,
    trajectory_origin: Vec3,
    /// One-based slot in the spectrum buffer. Multiple candidate evaluators
    /// may deliberately reference the same canonical slot.
    spectrum_index: u32,
}

fn build_trajectory_batch_elements(
    positions: &[Vec3],
    times: &[f32],
) -> Vec<FrequencyDomainBatchElement> {
    build_trajectory_elements(positions, times)
}

fn build_known_trajectory_elements(
    positions: &[Vec3],
    times: &[f32],
) -> Vec<FrequencyDomainBatchElement> {
    build_trajectory_elements(positions, times)
}

fn build_trajectory_elements(positions: &[Vec3], times: &[f32]) -> Vec<FrequencyDomainBatchElement> {
    if positions.is_empty()
        || positions.len() != times.len()
        || positions.iter().any(|position| !position.is_finite())
        || times.iter().any(|time| !time.is_finite() || *time < 0.0)
    {
        return Vec::new();
    }
    // Equation (184) integrates one known trajectory.  Splitting on a time
    // reversal silently changes that trajectory into several unrelated
    // integrals, so reject malformed captures instead of producing biased
    // segment-wise characteristics.
    if times.windows(2).any(|window| window[1] < window[0]) {
        return Vec::new();
    }
    vec![FrequencyDomainBatchElement {
        target_offset: 0,
        target_count: positions.len() as u32,
        trajectory_origin: positions[0],
        spectrum_index: 1,
    }]
}

#[derive(Resource, Default)]
struct ExtractedFrequencyDomainInput {
    enabled: bool,
    /// First trajectory sample, retained only for epoch/capture identity and
    /// never as a frequency-domain output observation.
    snapshot: Option<GravityRequestSnapshot>,
    target_bytes: Vec<u8>,
    observation_count: u32,
    batch_elements: Vec<FrequencyDomainBatchElement>,
    batch_capture_id: Option<u64>,
    runtime_revision: u64,
    sensitivity_sources: Vec<Vec<u8>>,
    sensitivity_source_counts: Vec<u32>,
    sensitivity_source_hash: u64,
    sensitivity_basis_hash: u64,
    sources: Option<Vec<u8>>,
    source_count: u32,
    radius: f32,
    source_hash: u64,
}

#[derive(Resource, Default)]
struct FrequencyDomainGpuBuffers(Option<FrequencyDomainGpuBuffersInner>);

struct FrequencyDomainGpuBuffersInner {
    targets: Buffer,
    output: Buffer,
    staging: Buffer,
    output_size: u64,
    layout: BindGroupLayout,
    sources: Buffer,
    quadrature: Buffer,
    spectrum: Buffer,
    density_spectra: Buffer,
    density_modes: Buffer,
    source_count: u32,
    source_radius: f32,
    element_capacity: u32,
    source_hash: u64,
    target_count: u32,
    last_submitted: Option<(u64, u64)>,
}

#[derive(Resource)]
struct FrequencyDomainComputePipeline {
    density_spectrum_id: CachedComputePipelineId,
    assemble_id: CachedComputePipelineId,
    evaluate_id: CachedComputePipelineId,
    planning_voxel_density_spectrum_id: CachedComputePipelineId,
    planning_voxel_spectrum_id: CachedComputePipelineId,
    planning_combine_spectrum_id: CachedComputePipelineId,
    planning_evaluate_id: CachedComputePipelineId,
}

pub struct FrequencyDomainGpuComputePlugin;

impl Plugin for FrequencyDomainGpuComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrequencyDomainGpuReadbackChannel>();
        app.init_resource::<FrequencyDomainTrajectoryBatchResult>();
        app.init_resource::<FrequencyDomainSensitivityMatrix>();
        app.init_resource::<FrequencyDomainPerformanceMetrics>();
        app.add_systems(PreUpdate, poll_frequency_domain_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedFrequencyDomainInput>();
        render_app.init_gpu_resource::<FrequencyDomainGpuBuffers>();
        render_app.init_gpu_resource::<PlanningFrequencyDomainDispatchState>();
        render_app.add_systems(ExtractSchedule, extract_frequency_domain_input);
        render_app.add_systems(
            Render,
            (initialize_frequency_domain_pipeline, dispatch_frequency_domain)
                .chain()
                .in_set(RenderSystems::Render),
        );
        render_app.add_systems(
            Render,
            dispatch_planning_frequency_domain
                .after(initialize_frequency_domain_pipeline)
                .after(crate::gpu::planning::PlanningGpuSystems::PrepareSharedInput)
                .in_set(crate::gpu::planning::PlanningGpuSystems::Dispatch)
                .in_set(RenderSystems::Cleanup),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<FrequencyDomainGpuReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        // Queue every active entry point during application startup. This
        // keeps the first Frequency-domain algorithm selection and First benchmark free of compile
        // time without retaining the unused analytic-spectrum pipeline.
        render_app.init_gpu_resource::<FrequencyDomainComputePipeline>();
    }
}

impl FromWorld for FrequencyDomainComputePipeline {
    fn from_world(world: &mut World) -> Self {
        log_frequency_domain_wgsl_source();
        let entries = [
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            storage_rw_entry(4),
            storage_rw_entry(6),
            uniform_entry(7),
            storage_ro_entry(9),
        ];
        let layout = BindGroupLayoutDescriptor::new("frequency_domain_bgl", &entries);
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::FrequencyDomain,
        );
        let cache = world.resource::<PipelineCache>();
        let source_shader_defs = vec![ShaderDefVal::from("FREQUENCY_DOMAIN_SOURCE")];
        let spectrum_shader_defs = vec![ShaderDefVal::from("FREQUENCY_DOMAIN_SPECTRUM")];
        let evaluator_shader_defs = vec![ShaderDefVal::from("FREQUENCY_DOMAIN_EVALUATOR")];
        let density_spectrum_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("frequency_domain_assemble_density_spectra".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: source_shader_defs.clone(),
            entry_point: Some("assemble_density_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let assemble_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("frequency_domain_assemble_spectrum".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: spectrum_shader_defs.clone(),
            entry_point: Some("publish_density_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let evaluate_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("frequency_domain_evaluate_field".into()),
            layout: vec![layout],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: evaluator_shader_defs.clone(),
            entry_point: Some("evaluate_trajectory_field".into()),
            zero_initialize_workgroup_memory: false,
        });
        let planning_layout = BindGroupLayoutDescriptor::new(
            "frequency_domain_planning_voxel_bgl",
            &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
                storage_rw_entry(4),
                storage_rw_entry(6),
                uniform_entry(7),
                storage_ro_entry(9),
            ],
        );
        let planning_voxel_density_spectrum_id =
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("frequency_domain_planning_voxel_density_spectra".into()),
                layout: vec![planning_layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: source_shader_defs.clone(),
                entry_point: Some("assemble_voxel_density_spectrum".into()),
                zero_initialize_workgroup_memory: false,
            });
        let planning_voxel_spectrum_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("frequency_domain_planning_voxel_spectrum".into()),
            layout: vec![planning_layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: spectrum_shader_defs.clone(),
            entry_point: Some("publish_voxel_density_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let planning_combine_spectrum_id =
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("frequency_domain_planning_combine_spectrum".into()),
                layout: vec![planning_layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: spectrum_shader_defs,
                entry_point: Some("combine_density_spectrum".into()),
                zero_initialize_workgroup_memory: false,
            });
        let planning_evaluate_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("frequency_domain_planning_evaluate_field".into()),
            layout: vec![planning_layout],
            immediate_size: 0,
            shader,
            shader_defs: evaluator_shader_defs,
            entry_point: Some("evaluate_trajectory_field".into()),
            zero_initialize_workgroup_memory: false,
        });
        Self {
            density_spectrum_id,
            assemble_id,
            evaluate_id,
            planning_voxel_density_spectrum_id,
            planning_voxel_spectrum_id,
            planning_combine_spectrum_id,
            planning_evaluate_id,
        }
    }
}

fn log_frequency_domain_wgsl_source() {
    let source = include_str!("../../wgsl/frequency_domain.wgsl");
    debug!(
        target: "wgsl::frequency_domain",
        bytes = source.len(),
        lines = source.lines().count(),
        "loading Frequency-domain algorithm WGSL source"
    );
    for (line_number, line) in source.lines().enumerate() {
        trace!(target: "wgsl::frequency_domain", line = line_number + 1, "WGSL {line}");
    }
}

fn poll_frequency_domain_readback(
    channel: Res<FrequencyDomainGpuReadbackChannel>,
    mut batch_result: ResMut<FrequencyDomainTrajectoryBatchResult>,
    mut sensitivity: ResMut<FrequencyDomainSensitivityMatrix>,
    mut performance: ResMut<FrequencyDomainPerformanceMetrics>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if channel.in_flight.load(Ordering::Acquire)
        && let Ok(mut submitted_at) = channel.submitted_at.try_lock()
        && submitted_at
            .as_ref()
            .is_some_and(|started| started.elapsed() > FREQUENCY_DOMAIN_GPU_TIMEOUT)
    {
        submitted_at.take();
        runtime_error.raise(
            "Frequency-domain algorithm GPU request exceeded 10 seconds; possible shader hang or device loss.",
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
            runtime_error.raise("Frequency-domain algorithm sensitivity readback has no capture identity.");
            return;
        };
        let column_count = packet.sensitivity_column_count as usize;
        let sample_count = packet.observation_count as usize;
        if sensitivity.capture_id != Some(capture_id)
            || sensitivity.source_hash != packet.sensitivity_source_hash
            || sensitivity.basis_hash != packet.sensitivity_basis_hash
            || sensitivity.configuration_hash != packet.sensitivity_configuration_hash
        {
            return;
        }
        if packet.partial_sums.len() != column_count * sample_count {
            runtime_error.raise(format!(
                "Frequency-domain algorithm sensitivity batch returned {} vectors; expected {} x {}.",
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
                    .map(|field| Vec3::new(field[0], field[1], field[2]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if columns
            .iter()
            .flatten()
            .any(|acceleration| !acceleration.is_finite())
        {
            runtime_error.raise("Frequency-domain algorithm sensitivity matrix contains non-finite values.");
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
    let decoded = match decode_frequency_domain_packet(&packet) {
        Ok(decoded) => decoded,
        Err(FrequencyDomainDecodeError::Incomplete { actual, expected }) => {
            runtime_error.raise(format!(
                "Frequency-domain algorithm batch readback is incomplete: {actual} rows, expected {expected}."
            ));
            return;
        }
        Err(FrequencyDomainDecodeError::Invalid { sample, reason }) => {
            runtime_error.raise(format!(
                "Frequency-domain algorithm returned an invalid field at sample {} ({reason}).",
                sample + 1,
            ));
            return;
        }
    };

    let Some(capture_id) = packet.batch_capture_id else {
        runtime_error.raise(
            "Frequency-domain algorithm aggregate readback has no trajectory capture identity.",
        );
        return;
    };
    // Publish the complete batch atomically. Clearing the displayed series
    // before decode made a transient/incomplete GPU packet leave the UI with
    // an empty frequency response even though the preceding valid response
    // was still the correct result for the active capture.
    batch_result.capture_id = Some(capture_id);
    batch_result.observations = decoded;
    batch_result.revision = batch_result.revision.wrapping_add(1);
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FrequencyDomainDecodeError {
    Incomplete {
        actual: usize,
        expected: usize,
    },
    Invalid { sample: usize, reason: &'static str },
}

fn decode_frequency_domain_packet(
    packet: &FrequencyDomainReadbackPacket,
) -> Result<Vec<FrequencyDomainObservation>, FrequencyDomainDecodeError> {
    let expected_rows = packet.observation_count as usize * OUTPUT_ROWS_PER_BLOCK as usize;
    if packet.partial_sums.len() != expected_rows {
        return Err(FrequencyDomainDecodeError::Incomplete {
            actual: packet.partial_sums.len(),
            expected: expected_rows,
        });
    }
    (0..packet.observation_count as usize)
        .map(|index| {
            let start = index * OUTPUT_ROWS_PER_BLOCK as usize;
            decode_frequency_domain_sample(
                &packet.partial_sums[start..start + OUTPUT_ROWS_PER_BLOCK as usize],
            )
            .map_err(|reason| FrequencyDomainDecodeError::Invalid {
                sample: index,
                reason,
            })
        })
        .collect()
}

fn decode_frequency_domain_sample(
    rows: &[[f32; 4]],
) -> Result<FrequencyDomainObservation, &'static str> {
    let field = rows[0];
    let potentials = rows[6];
    let transformed_field = Vec3::new(field[0], field[1], field[2]);
    let laplace_frequency = rows[5][0];
    let transformed_potential = potentials[0];
    let jacobian = Mat3::from_cols(
        Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
        Vec3::new(rows[3][0], rows[3][1], rows[3][2]),
        Vec3::new(rows[4][0], rows[4][1], rows[4][2]),
    );
    if !transformed_field.is_finite() {
        return Err("transformed field is non-finite");
    }
    if transformed_field.abs().max_element() > 1.0e12 {
        return Err("transformed field exceeds the numerical bound");
    }
    if !laplace_frequency.is_finite() || laplace_frequency <= 0.0 {
        return Err("Laplace frequency is non-finite or non-positive");
    }
    if !transformed_potential.is_finite() || transformed_potential.abs() > 1.0e20 {
        return Err("transformed potential is non-finite or out of range");
    }
    if !jacobian.is_finite() {
        return Err("transformed Jacobian is non-finite");
    }
    Ok(FrequencyDomainObservation {
        laplace_frequency,
        transformed_field,
        transformed_jacobian: jacobian,
        transformed_potential,
    })
}
