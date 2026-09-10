use crate::interface::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
    render_resource::{Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
};

#[derive(Resource, Default)]
pub(crate) struct ExtractedPlanningInput {
    pub batch: Option<PlanningCandidateBatch>,
    pub request: PlanningGpuRequest,
    pub payload: PlanningMethodPayload,
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
        render_app.init_gpu_resource::<crate::gpu::planning_timestamps::PlanningTimestampPool>();
        render_app.add_systems(ExtractSchedule, extract_planning_input);
        render_app.add_systems(
            Render,
            prepare_planning_shared_buffers
                .in_set(PlanningGpuSystems::PrepareSharedInput)
                .in_set(RenderSystems::PrepareResources),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<PlanningGpuReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_gpu_resource::<crate::gpu::planning_reduction::PlanningReductionPipeline>();
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
    extracted.source_radius = batch.frequency_domain_source_radius;
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

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    let iterator = values.into_iter();
    let (lower, _) = iterator.size_hint();
    let mut bytes = Vec::with_capacity(lower * 4);
    for value in iterator {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
