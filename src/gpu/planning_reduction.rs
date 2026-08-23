use crate::interface::components::*;
use bevy::prelude::*;
use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
        CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
        ShaderStages,
    },
    renderer::RenderDevice,
};

impl crate::gpu::planning::PlanningSharedGpuBuffersInner {
    pub(crate) fn matches(&self, batch: &PlanningCandidateBatch) -> bool {
        let state_count = batch.state_count() as u64;
        self.batch_id == batch.batch_id
            && self.positions.size() == state_count * 16
            && self.uploaded_position_bytes == batch.gpu_position_bytes.len()
            && self.densities.size() == batch.density_models.len() as u64 * 4
    }
}

pub(crate) fn planning_verification_targets(
    request: &PlanningGpuRequest,
    batch: &PlanningCandidateBatch,
) -> Vec<u32> {
    let model_stride = PLANNING_REFERENCE_MODEL_STRIDE.min((batch.density_model_count / 4).max(1));
    if !request.density_model.is_multiple_of(model_stride) {
        return Vec::new();
    }
    let candidate_stride =
        PLANNING_REFERENCE_CANDIDATE_STRIDE.min((batch.candidate_count / 8).max(1));
    let mut targets = Vec::new();
    for local_candidate in 0..request.candidate_count {
        let candidate = request.candidate_start + local_candidate;
        if !candidate.is_multiple_of(candidate_stride) {
            continue;
        }
        for sample in 0..batch.samples_per_candidate {
            if sample.is_multiple_of(PLANNING_REFERENCE_STRIDE)
                || sample.abs_diff(batch.samples_per_candidate / 2) <= 1
            {
                targets.push(local_candidate * batch.samples_per_candidate + sample);
            }
        }
    }
    targets
}

#[derive(Resource)]
pub(crate) struct PlanningReductionPipeline(pub CachedComputePipelineId);

impl FromWorld for PlanningReductionPipeline {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<bevy::render::render_resource::PipelineCache>();
        let server = world.resource::<AssetServer>();
        let layout = BindGroupLayoutDescriptor::new(
            "planning_reduction_bgl",
            &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        );
        Self(cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("planning_reduction".into()),
            layout: vec![layout],
            immediate_size: 0,
            shader: server.load("shaders/planning_metrics.wgsl"),
            shader_defs: vec![],
            entry_point: None,
            zero_initialize_workgroup_memory: false,
        }))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_planning_reduction(
    render_device: &RenderDevice,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    request: &PlanningGpuRequest,
    batch: &PlanningCandidateBatch,
    row_stride: u32,
    fields: &Buffer,
    positions: &Buffer,
    baseline: &Buffer,
    metrics: &Buffer,
) -> (Buffer, BindGroup) {
    let mut bytes = Vec::with_capacity(48);
    for value in [
        request.candidate_start * batch.samples_per_candidate,
        request.candidate_count * batch.samples_per_candidate,
        batch.samples_per_candidate,
        request.density_model,
        row_stride,
        request.candidate_count,
        0,
        0,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [
        batch.body_radius,
        GRAVITY_BENCHMARK_RELATIVE_TOLERANCE,
        0.30,
        0.0,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planning_reduction_uniform"),
        contents: &bytes,
        usage: BufferUsages::UNIFORM,
    });
    let layout = render_device.create_bind_group_layout(
        "planning_reduction_runtime_bgl",
        &[
            buffer_entry(0, BufferBindingType::Uniform),
            buffer_entry(1, BufferBindingType::Storage { read_only: true }),
            buffer_entry(2, BufferBindingType::Storage { read_only: true }),
            buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            buffer_entry(4, BufferBindingType::Storage { read_only: false }),
        ],
    );
    let bind_group = render_device.create_bind_group(
        "planning_reduction_bg",
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: fields.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: positions.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: baseline.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: metrics.as_entire_binding(),
            },
        ],
    );
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("planning_reduction_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(request.candidate_count, 1, 1);
    drop(pass);
    (uniform, bind_group)
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
