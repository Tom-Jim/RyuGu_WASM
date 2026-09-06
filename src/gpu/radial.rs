use crate::interface::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::Arc;
use std::sync::atomic::Ordering;

const WORKGROUP_SIZE: u32 = 64;

#[derive(Resource, Default)]
struct ExtractedGravityInput {
    enabled: bool,
    probe: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    source_bytes: Option<Vec<u8>>,
    source_count: u32,
    source_hash: u64,
}

#[derive(Resource, Default)]
struct GravityGpuBuffers(Option<GravityGpuBuffersInner>);

struct GravityGpuBuffersInner {
    uniform: bevy::render::render_resource::Buffer,
    output: bevy::render::render_resource::Buffer,
    staging: bevy::render::render_resource::Buffer,
    bind_group: BindGroup,
    source_count: u32,
    workgroup_count: u32,
    output_size: u64,
    last_submitted: Option<(u64, u64)>,
    source_hash: u64,
}

#[derive(Resource)]
struct GravityComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

pub struct GravityComputePlugin;

impl Plugin for GravityComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GravityReadbackChannel>();
        app.init_resource::<GravityPotential>();
        app.init_resource::<RadialGravityHistory>();
        app.add_systems(PreUpdate, poll_gravity_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedGravityInput>();
        render_app.init_gpu_resource::<GravityGpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_gravity_input_system);
        render_app.add_systems(
            Render,
            dispatch_gravity_system.in_set(RenderSystems::Render),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<GravityReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_gpu_resource::<GravityComputePipeline>();
    }
}

impl FromWorld for GravityComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)];
        let layout = BindGroupLayoutDescriptor::new("radial_gravity_bgl", &entries);
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::Gravity,
        );
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("radial_gravity_compute".into()),
                    layout: vec![layout],
                    immediate_size: 0,
                    shader,
                    shader_defs: vec![],
                    entry_point: None,
                    zero_initialize_workgroup_memory: false,
                });
        Self { pipeline_id }
    }
}

fn poll_gravity_readback(
    channel: Res<GravityReadbackChannel>,
    mut acceleration: ResMut<GravityAcceleration>,
    mut potential: ResMut<GravityPotential>,
    mut history: ResMut<RadialGravityHistory>,
) {
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    let Some(packet) = guard.take() else {
        return;
    };
    let total_f64 = packet
        .partial_sums
        .iter()
        .fold([0.0_f64; 4], |mut sum, value| {
            for index in 0..4 {
                sum[index] += value[index] as f64;
            }
            sum
        });
    let total = Vec4::from_array(total_f64.map(|value| value as f32));
    let acceleration_is_valid = total.xyz().is_finite();
    let potential_is_valid = total.w.is_finite() && total.w > 0.0;
    if acceleration_is_valid {
        acceleration.0 = total.xyz();
    } else {
        warn!("[gravity] discarded non-finite radial-model GPU result");
    }
    if potential_is_valid {
        potential.0 = Some(total.w);
    } else {
        potential.0 = None;
        warn!("[gravity] discarded invalid radial-model potential");
    }
    if acceleration_is_valid && potential_is_valid {
        history.0.push(GravityFieldSample {
            snapshot: packet.snapshot,
            body_acceleration: total.xyz(),
            positive_potential: total.w,
        });
    }
}

fn extract_gravity_input_system(
    mut extracted: ResMut<ExtractedGravityInput>,
    source: Extract<Option<Res<DensityQuadratureSource>>>,
    active: Extract<Res<ActiveGravityMethod>>,
    clock: Extract<Res<SimulationClock>>,
    planning: Extract<Res<PlanningComparisonState>>,
    density_mode: Extract<Res<DensityMode>>,
    cassini: Extract<Query<&Transform, With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    extracted.enabled =
        **active == ActiveGravityMethod::RadialAnalytic && !planning.blocks_realtime_gpu();
    extracted.snapshot = None;
    if !extracted.enabled {
        return;
    }
    let (Some(source), Ok(cassini), Ok(ryugu)) =
        (source.as_ref(), cassini.single(), ryugu.single())
    else {
        return;
    };

    extracted.probe = ryugu.rotation.inverse() * (cassini.translation - ryugu.translation);
    extracted.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
    });
    let (bytes, hash) = match **density_mode {
        DensityMode::Variable => (&source.bytes, source.source_hash),
        DensityMode::Constant => (&source.constant_bytes, source.constant_hash),
    };
    extracted.source_count = (bytes.len() / 32) as u32;
    if extracted.source_bytes.is_none() || extracted.source_hash != hash {
        extracted.source_bytes = Some(bytes.clone());
        extracted.source_hash = hash;
    }
}

fn dispatch_gravity_system(
    mut buffers: ResMut<GravityGpuBuffers>,
    pipeline_resource: Option<Res<GravityComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedGravityInput>,
    channel: Res<GravityReadbackChannel>,
) {
    let Some(pipeline_resource) = pipeline_resource else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_resource.pipeline_id) else {
        return;
    };
    if !extracted.enabled || extracted.source_count == 0 {
        return;
    }

    if buffers
        .0
        .as_ref()
        .is_some_and(|inner| inner.source_hash != extracted.source_hash)
    {
        buffers.0 = None;
    }
    if buffers.0.is_none() {
        let Some(source_bytes) = extracted.source_bytes.as_ref() else {
            return;
        };
        let workgroup_count = extracted.source_count.div_ceil(WORKGROUP_SIZE);
        let output_size = workgroup_count as u64 * 16;
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("radial_gravity_uniform"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sources = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("radial_gravity_sources"),
            contents: source_bytes,
            usage: BufferUsages::STORAGE,
        });
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("radial_gravity_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("radial_gravity_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "radial_gravity_bgl_runtime",
            &[uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)],
        );
        let bind_group = render_device.create_bind_group(
            "radial_gravity_bg",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sources.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(GravityGpuBuffersInner {
            uniform,
            output,
            staging,
            bind_group,
            source_count: extracted.source_count,
            workgroup_count,
            output_size,
            last_submitted: None,
            source_hash: extracted.source_hash,
        });
    }

    let inner = buffers.0.as_mut().expect("gravity buffers initialized");
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        return;
    };
    let submission_key = (snapshot.epoch, snapshot.request_id);
    if inner.last_submitted == Some(submission_key) {
        return;
    }
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    inner.last_submitted = Some(submission_key);

    let uniform_bytes = gravity_uniform_bytes(extracted.probe, inner.source_count);
    render_queue.write_buffer(&inner.uniform, 0, &uniform_bytes);

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("radial_gravity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("radial_gravity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(inner.workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&inner.output, 0, &inner.staging, 0, inner.output_size);
    render_queue.submit([encoder.finish()]);

    let shared = Arc::clone(&channel.data);
    let in_flight = Arc::clone(&channel.in_flight);
    let staging = inner.staging.clone();
    let map_staging = staging.clone();
    let snapshot = snapshot.clone();
    map_staging
        .slice(..)
        .map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let view = staging.slice(..).get_mapped_range();
                let partial_sums = bytes_to_f32x4(&view);
                if let Ok(mut guard) = shared.lock() {
                    *guard = Some(GravityReadbackPacket {
                        partial_sums,
                        snapshot,
                    });
                }
                drop(view);
                staging.unmap();
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn gravity_uniform_bytes(probe: Vec3, source_count: u32) -> [u8; 32] {
    let mut bytes = [0_u8; 32];
    for (offset, value) in [(0, probe.x), (4, probe.y), (8, probe.z), (12, G)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[16..20].copy_from_slice(&source_count.to_le_bytes());
    bytes
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

fn buffer_entry(binding: u32, buffer_type: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: buffer_type,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
