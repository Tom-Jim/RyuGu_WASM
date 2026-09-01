use crate::interface::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
        BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CachedPipelineState, CommandEncoderDescriptor,
        ComputePassDescriptor, ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
    GpuResourceAppExt,
};
use bevy::shader::ShaderCacheError;
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[derive(Resource, Default)]
pub(crate) struct ExtractedPlanningInput {
    pub batch: Option<PlanningCandidateBatch>,
    pub request: PlanningGpuRequest,
    pub payload: PlanningMethodPayload,
    pub eq106_operator: Arc<[u8]>,
    pub source_radius: f32,
}

#[derive(Resource, Default)]
pub(crate) struct PlanningSharedGpuBuffers(pub Option<PlanningSharedGpuBuffersInner>);

pub(crate) struct PlanningSharedGpuBuffersInner {
    pub batch_id: u64,
    pub positions: Buffer,
    pub densities: Buffer,
    pub uploaded_position_bytes: usize,
}

#[derive(Resource, Default)]
struct PlanningDispatchBuffers {
    request_id: u64,
    output_size: u64,
    staging_size: u64,
    baseline_size: u64,
    metric_size: u64,
    output: Option<Buffer>,
    baseline: Option<Buffer>,
    metrics: Option<Buffer>,
    staging: Option<Buffer>,
}

#[derive(Resource)]
struct PlanningMethodPipelines {
    fmm: CachedComputePipelineId,
    mmfft: CachedComputePipelineId,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PlanningGpuSystems {
    PrepareSharedInput,
    Dispatch,
}

pub struct PlanningGpuComputePlugin;

impl Plugin for PlanningGpuComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlanningCandidateBatch>();
        app.init_resource::<PlanningGpuRequest>();
        app.init_resource::<PlanningGpuResult>();
        app.init_resource::<PlanningGpuReadbackChannel>();
        app.init_resource::<PlanningMethodPayload>();
        app.add_systems(PreUpdate, poll_planning_gpu_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedPlanningInput>();
        render_app.init_gpu_resource::<PlanningSharedGpuBuffers>();
        render_app.init_gpu_resource::<PlanningDispatchBuffers>();
        render_app.init_gpu_resource::<crate::gpu::planning_timestamps::PlanningTimestampPool>();
        render_app.init_gpu_resource::<PlanningFftGpu>();
        render_app.init_gpu_resource::<PlanningFmmGpu>();
        render_app.add_systems(ExtractSchedule, extract_planning_input);
        render_app.add_systems(
            Render,
            prepare_planning_shared_buffers
                .in_set(PlanningGpuSystems::PrepareSharedInput)
                .in_set(RenderSystems::PrepareResources),
        );
        render_app.add_systems(
            Render,
            dispatch_planning_method
                .after(PlanningGpuSystems::PrepareSharedInput)
                .in_set(PlanningGpuSystems::Dispatch)
                .in_set(RenderSystems::Cleanup),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<PlanningGpuReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_gpu_resource::<PlanningMethodPipelines>();
        render_app.init_gpu_resource::<PlanningFftPipelines>();
        render_app.init_gpu_resource::<PlanningFmmPipelines>();
        render_app.init_gpu_resource::<crate::gpu::planning_reduction::PlanningReductionPipeline>();
    }
}

impl FromWorld for PlanningMethodPipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let server = world.resource::<AssetServer>();
        let fmm = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("planning_fmm_batch".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "planning_fmm_batch_bgl",
                &planning_method_layout_entries(),
            )],
            immediate_size: 0,
            shader: crate::wgsl::load(server, crate::wgsl::EmbeddedShader::PlanningFmm),
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });
        let mmfft = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("planning_mmfft_batch".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "planning_mmfft_batch_bgl",
                &planning_method_layout_entries(),
            )],
            immediate_size: 0,
            shader: crate::wgsl::load(server, crate::wgsl::EmbeddedShader::PlanningMmfft),
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });
        Self { fmm, mmfft }
    }
}

fn poll_planning_gpu_readback(
    channel: Res<PlanningGpuReadbackChannel>,
    mut result: ResMut<PlanningGpuResult>,
) {
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    if let Some(packet) = guard.take() {
        result.0 = Some(packet);
    }
}

fn extract_planning_input(
    mut extracted: ResMut<ExtractedPlanningInput>,
    batch: Extract<Res<PlanningCandidateBatch>>,
    request: Extract<Res<PlanningGpuRequest>>,
    payload: Extract<Res<PlanningMethodPayload>>,
    operator: Extract<Option<Res<crate::cpu::eq106_operator::Eq106OperatorTensorResource>>>,
) {
    if batch.batch_id == 0 || request.batch_id != batch.batch_id {
        extracted.batch = None;
        extracted.request = PlanningGpuRequest::default();
        extracted.payload = PlanningMethodPayload::default();
        return;
    }
    if extracted
        .batch
        .as_ref()
        .is_none_or(|current| current.batch_id != batch.batch_id)
    {
        extracted.batch = Some(batch.clone());
    }
    extracted.request = request.clone();
    extracted.payload = payload.clone();
    if extracted.eq106_operator.is_empty()
        && let Some(operator) = operator.as_ref()
    {
        extracted.eq106_operator = Arc::from(operator.tensor.as_le_bytes());
    }
    extracted.source_radius = batch.eq106_source_radius;
}

fn prepare_planning_shared_buffers(
    mut buffers: ResMut<PlanningSharedGpuBuffers>,
    extracted: Res<ExtractedPlanningInput>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(batch) = extracted.batch.as_ref() else {
        return;
    };
    if buffers
        .0
        .as_ref()
        .is_none_or(|inner| inner.batch_id != batch.batch_id)
    {
        let densities = f32_bytes(batch.density_models.iter().copied());
        buffers.0 = Some(PlanningSharedGpuBuffersInner {
            batch_id: batch.batch_id,
            positions: render_device.create_buffer(&BufferDescriptor {
                label: Some("planning_candidate_positions"),
                size: batch.gpu_position_bytes.len().max(16) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            densities: render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("planning_density_models"),
                contents: &densities,
                usage: BufferUsages::STORAGE,
            }),
            uploaded_position_bytes: 0,
        });
    }
    let inner = buffers.0.as_mut().expect("planning buffers initialized");
    if inner.uploaded_position_bytes < batch.gpu_position_bytes.len() {
        let start = inner.uploaded_position_bytes;
        let end = (start + PLANNING_GPU_UPLOAD_BYTES_PER_FRAME).min(batch.gpu_position_bytes.len());
        render_queue.write_buffer(
            &inner.positions,
            start as u64,
            &batch.gpu_position_bytes[start..end],
        );
        inner.uploaded_position_bytes = end;
    }
}

fn dispatch_planning_method(
    extracted: Res<ExtractedPlanningInput>,
    shared: Res<PlanningSharedGpuBuffers>,
    pipelines: Res<PlanningMethodPipelines>,
    reduction: Res<crate::gpu::planning_reduction::PlanningReductionPipeline>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut buffers: ResMut<PlanningDispatchBuffers>,
    channel: Res<PlanningGpuReadbackChannel>,
    fft_pipelines: Res<PlanningFftPipelines>,
    mut fft_gpu: ResMut<PlanningFftGpu>,
    fmm_pipelines: Res<PlanningFmmPipelines>,
    mut fmm_gpu: ResMut<PlanningFmmGpu>,
    timestamp_pool: Res<crate::gpu::planning_timestamps::PlanningTimestampPool>,
) {
    let request = &extracted.request;
    let Some(method) = request.method else {
        // Drop cancelled cache ownership; at most one bounded submission may
        // still be running. Its callback owns no UI/job state.
        fft_gpu.0 = None;
        fmm_gpu.0 = None;
        // Mapping callbacks retain their own handles. Releasing these cached
        // handles must not retain a large response bank after cancellation.
        *buffers = PlanningDispatchBuffers::default();
        return;
    };
    if method != ActiveGravityMethod::MmfftCompressed { fft_gpu.0 = None; }
    if method != ActiveGravityMethod::Fmm { fmm_gpu.0 = None; }
    if !matches!(
        method,
        ActiveGravityMethod::MmfftCompressed | ActiveGravityMethod::Fmm
    ) || extracted.payload.method != Some(method)
        || extracted.payload.density_model != request.density_model
    {
        return;
    }
    let Some(batch) = extracted.batch.as_ref() else {
        return;
    };
    let Some(shared) = shared.0.as_ref().filter(|shared| shared.matches(batch)) else {
        return;
    };
    let pipeline_id = if method == ActiveGravityMethod::Fmm {
        pipelines.fmm
    } else {
        pipelines.mmfft
    };
    let Some(pipeline) = cache.get_compute_pipeline(pipeline_id) else {
        report_planning_pipeline_failure(
            &cache,
            pipeline_id,
            request.request_id,
            method.planning_label(),
            &channel,
        );
        return;
    };
    let Some(reduction_pipeline) = cache.get_compute_pipeline(reduction.0) else {
        report_planning_pipeline_failure(
            &cache,
            reduction.0,
            request.request_id,
            "planning reduction",
            &channel,
        );
        return;
    };
    if buffers.request_id == request.request_id {
        return;
    }
    // Preparation has its own bounded asynchronous fence. Do not claim the
    // evaluator's in-flight flag while the CPU/UI is waiting for a GPU stage.
    let (primary, response_start) = if method == ActiveGravityMethod::Fmm {
        let Some(prepared) = fmm_gpu.prepare(&extracted, shared, &fmm_pipelines, &cache,
            &render_device, &render_queue, &channel, &timestamp_pool) else { return; };
        prepared
    } else {
        let Some(primary) = fft_gpu.prepare(&extracted, shared, &fft_pipelines, &cache,
            &render_device, &render_queue, &channel, &timestamp_pool) else { return; };
        (primary, 0)
    };
    let std::task::Poll::Ready(timestamps) = timestamp_pool.acquire(&render_device, &render_queue, 1)
        else { return; };
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let method_preprocess_started = bevy::platform::time::Instant::now();
    let target_count = request.candidate_count * batch.samples_per_candidate;
    let output_size = u64::from(target_count) * 4 * 16;
    let verification_targets =
        crate::gpu::planning_reduction::planning_verification_targets(request, batch);
    let metric_size = u64::from(request.candidate_count) * 16;
    let data_size = metric_size + verification_targets.len() as u64 * 4 * 16;
    let staging_size = data_size + 16;
    let preparation_cost = if method == ActiveGravityMethod::Fmm {
        fmm_gpu.take_cost()
    } else {
        fft_gpu.take_cost()
    };
    let output_changed = buffers.output_size != output_size || buffers.output.is_none();
    let output = if output_changed {
        render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_method_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    } else {
        buffers
            .output
            .as_ref()
            .expect("checked output buffer")
            .clone()
    };
    let baseline_size = batch.state_count() as u64 * 16;
    let baseline = match (&buffers.baseline, buffers.baseline_size == baseline_size) {
        (Some(buffer), true) => buffer.clone(),
        _ => render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_method_baseline"),
            size: baseline_size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
    };
    let metrics = match (&buffers.metrics, buffers.metric_size == metric_size) {
        (Some(buffer), true) => buffer.clone(),
        _ => render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_method_metrics"),
            size: metric_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
    };
    let staging_changed = buffers.staging_size != staging_size || buffers.staging.is_none();
    let staging = if staging_changed {
        render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_method_staging"),
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    } else {
        buffers
            .staging
            .as_ref()
            .expect("checked staging buffer")
            .clone()
    };
    let uniform_bytes = if method == ActiveGravityMethod::Fmm {
        fmm_planning_uniform(request, batch, response_start)
    } else {
        mmfft_planning_uniform(request, batch, &extracted.payload)
    };
    let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planning_method_uniform"),
        contents: &uniform_bytes,
        usage: BufferUsages::UNIFORM,
    });
    let layout = render_device.create_bind_group_layout(
        "planning_method_runtime_bgl",
        &planning_method_layout_entries(),
    );
    // FFT interpolates the GPU potential at these positions; FMM combines
    // its GPU response bank with this density row. Neither uploads CPU fields.
    let evaluation_input = if method == ActiveGravityMethod::Fmm {
        &shared.densities
    } else {
        &shared.positions
    };
    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 1,
            resource: primary.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 2,
            resource: evaluation_input.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 3,
            resource: output.as_entire_binding(),
        },
    ];
    let bind_group = render_device.create_bind_group("planning_method_bg", &layout, &entries);
    let method_preprocess_ms = method_preprocess_started.elapsed().as_secs_f64() * 1.0e3;
    let encode_started = bevy::platform::time::Instant::now();
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("planning_method_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("planning_method_pass"),
            timestamp_writes: timestamps.as_ref().map(|queries| queries.writes(0)),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(target_count.min(65_535), target_count.div_ceil(65_535), 1);
    }
    let (_reduction_uniform, _reduction_bind_group) =
        crate::gpu::planning_reduction::encode_planning_reduction(
            &render_device,
            &mut encoder,
            reduction_pipeline,
            request,
            batch,
            4,
            &output,
            &shared.positions,
            &baseline,
            &metrics,
        );
    encoder.copy_buffer_to_buffer(&metrics, 0, &staging, 0, metric_size);
    for (compact_index, target_index) in verification_targets.iter().copied().enumerate() {
        encoder.copy_buffer_to_buffer(
            &output,
            u64::from(target_index) * 4 * 16,
            &staging,
            metric_size + compact_index as u64 * 4 * 16,
            4 * 16,
        );
    }
    if let Some(queries) = &timestamps {
        queries.resolve_into(&mut encoder, &staging, data_size);
    }
    render_queue.submit([encoder.finish()]);
    let command_submission_ms = encode_started.elapsed().as_secs_f64() * 1.0e3;
    buffers.request_id = request.request_id;
    buffers.output_size = output_size;
    buffers.staging_size = staging_size;
    buffers.baseline_size = baseline_size;
    buffers.metric_size = metric_size;
    buffers.output = Some(output);
    buffers.baseline = Some(baseline);
    buffers.metrics = Some(metrics);
    buffers.staging = Some(staging.clone());

    let packet_request = request.clone();
    let shared_data = Arc::clone(&channel.data);
    let in_flight = Arc::clone(&channel.in_flight);
    let mapped = staging.clone();
    let backend = if method == ActiveGravityMethod::Fmm {
        PlanningExecutionBackend::GpuFmm
    } else {
        PlanningExecutionBackend::GpuMmfft
    };
    let average_fmm_interactions = extracted.payload.maximum_level;
    let state_indices = verification_targets;
    let request_candidate_count = request.candidate_count;
    let gpu_completion_started = bevy::platform::time::Instant::now();
    mapped
        .slice(..staging_size)
        .map_async(MapMode::Read, move |result| {
            let gpu_completion_map_ms = gpu_completion_started.elapsed().as_secs_f64() * 1.0e3;
            let decode_started = bevy::platform::time::Instant::now();
            let mut kernel_ms = None;
            let rows = if result.is_ok() {
                let view = staging.slice(..staging_size).get_mapped_range();
                kernel_ms = timestamps.as_ref()
                    .and_then(|queries| queries.decode(&view[data_size as usize..]))
                    .and_then(|values| values.first().copied());
                let decoded = bytes_to_f32x4(&view[..data_size as usize]);
                let candidate_metrics = decoded[..request_candidate_count as usize].to_vec();
                let rows = decoded[request_candidate_count as usize..].to_vec();
                drop(view);
                staging.unmap();
                (rows, candidate_metrics)
            } else {
                (Vec::new(), Vec::new())
            };
            let kernel_evaluations = if method == ActiveGravityMethod::Fmm {
                u64::from(target_count) * u64::from(average_fmm_interactions.max(1))
            } else {
                u64::from(target_count) * 216
            };
            if let Ok(mut guard) = shared_data.lock() {
                *guard = Some(PlanningGpuPacket {
                    request: packet_request,
                    state_indices,
                    raw_rows: rows.0.clone(),
                    rows: rows.0,
                    rejection_counts: [0; 6],
                    rejected_sample_count: 0,
                    self_fd_step_maxima: [0.0; 5],
                    first_rejection: None,
                    candidate_metrics: rows.1,
                    readback_valid: result.is_ok(),
                    timing: PlanningGpuTiming {
                        method_preprocess_ms: method_preprocess_ms + preparation_cost.cpu_ms,
                        command_submission_ms: command_submission_ms + preparation_cost.submission_ms,
                        gpu_completion_map_ms: gpu_completion_map_ms + preparation_cost.completion_ms,
                        readback_decode_ms: decode_started.elapsed().as_secs_f64() * 1.0e3 + preparation_cost.decode_ms,
                        kernel_ms: kernel_ms.zip(preparation_cost.all_ms).map(|(eval, prep)| eval + prep),
                        evaluation_kernel_ms: kernel_ms,
                        basis_kernel_ms: kernel_ms.and(preparation_cost.basis_ms),
                        dispatch_count: 2 + preparation_cost.dispatches,
                        forward_kernel_evaluations: kernel_evaluations,
                        spectral_element_count: 0,
                        gradient_self_fd_relative_error: 0.0,
                    },
                    backend,
                });
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn report_planning_pipeline_failure(
    cache: &PipelineCache,
    pipeline_id: CachedComputePipelineId,
    request_id: u64,
    label: &str,
    channel: &PlanningGpuReadbackChannel,
) {
    let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(pipeline_id) else {
        return;
    };
    if matches!(
        error,
        ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable
    ) {
        return;
    }
    if let Ok(mut slot) = channel.error.try_lock()
        && slot.is_none()
    {
        *slot = Some((
            request_id,
            format!("{label} GPU pipeline failed to compile: {error}"),
        ));
    }
}

fn fmm_planning_uniform(
    request: &PlanningGpuRequest,
    batch: &PlanningCandidateBatch,
    response_start: u32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for value in [
        request.candidate_start * batch.samples_per_candidate,
        request.candidate_count * batch.samples_per_candidate,
        response_start,
        request.density_model,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn mmfft_planning_uniform(
    request: &PlanningGpuRequest,
    batch: &PlanningCandidateBatch,
    payload: &PlanningMethodPayload,
) -> Vec<u8> {
    // WGSL uniform layout gives the trailing vec3 a 16-byte alignment, so the
    // Params structure occupies 80 bytes even though its scalar payload ends
    // at byte 64. Keep the shader offsets unchanged and provide the required
    // tail padding explicitly.
    let mut bytes = Vec::with_capacity(80);
    for value in [
        request.candidate_start * batch.samples_per_candidate,
        request.candidate_count * batch.samples_per_candidate,
        request.density_model,
        2,
        payload.grid_sizes[0],
        payload.grid_sizes[1],
        0,
        0,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [
        payload.half_extents[0],
        payload.half_extents[1],
        payload.total_mass,
        0.25,
        payload.grid_scales[0],
        payload.grid_scales[1],
        G,
        0.0,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.resize(80, 0);
    bytes
}

// Both evaluator pipelines share this binding shape; their uniform structures
// and read-only input at binding 2 remain method-specific.
fn planning_method_layout_entries() -> [BindGroupLayoutEntry; 4] {
    [uniform_entry(0), storage_ro_entry(1), storage_ro_entry(2), storage_rw_entry(3)]
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Uniform)
}

fn storage_ro_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Storage { read_only: true })
}

fn storage_rw_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Storage { read_only: false })
}

fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    let iterator = values.into_iter();
    let (lower, _) = iterator.size_hint();
    let mut bytes = Vec::with_capacity(lower * 4);
    for value in iterator {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

include!("planning_fft_gpu.rs");

include!("planning_fmm_gpu.rs");
