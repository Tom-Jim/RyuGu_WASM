use crate::interface::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::Arc;

// ── Render-world: extracted bytes (Option → avoid Commands in render world) ──

#[derive(Resource, Default)]
struct ExtractedTopologyRaw(Option<ExtractedTopologyInner>);

struct ExtractedTopologyInner {
    node_count: u32,
    pos_bytes: Vec<u8>,
    off_bytes: Vec<u8>,
    idx_bytes: Vec<u8>,
}

// ── Render-world: GPU buffers ─────────────────────────────────────────────────

#[derive(Resource, Default)]
struct NormalsGpuBuffers(Option<NormalsGpuBuffersInner>);

struct NormalsGpuBuffersInner {
    _pos: bevy::render::render_resource::Buffer,
    _off: bevy::render::render_resource::Buffer,
    _idx: bevy::render::render_resource::Buffer,
    _out: bevy::render::render_resource::Buffer,
    _staging: bevy::render::render_resource::Buffer,
}

// ── Render-world: pipeline + dispatch state ───────────────────────────────────

#[derive(Resource)]
struct NormalsComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource, Default, PartialEq, Eq)]
enum DispatchState {
    #[default]
    Pending,
    Done,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct NormalsComputePlugin;

impl Plugin for NormalsComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NormalsReadbackChannel>();
        app.add_systems(Update, poll_normals_readback);

        let render_app = app.sub_app_mut(RenderApp);
        // Pre-insert Option resources so render-world systems use ResMut, not Commands
        render_app.init_resource::<ExtractedTopologyRaw>();
        render_app.init_resource::<NormalsGpuBuffers>();
        render_app.init_resource::<DispatchState>();
        render_app.add_systems(ExtractSchedule, extract_topology_system);
        render_app.add_systems(
            Render,
            dispatch_normals_system.in_set(RenderSystems::Render),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<NormalsReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<NormalsComputePipeline>();
    }
}

// ── FromWorld ─────────────────────────────────────────────────────────────────

impl FromWorld for NormalsComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ];
        let bgl = BindGroupLayoutDescriptor::new("normals_bgl", &entries);
        let shader = world.resource::<AssetServer>().load("shaders/normals.wgsl");
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("normals_compute".into()),
                    layout: vec![bgl],
                    immediate_size: 0,
                    shader,
                    shader_defs: vec![],
                    entry_point: None,
                    zero_initialize_workgroup_memory: false,
                });
        NormalsComputePipeline { pipeline_id }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── Extract system — writes into pre-inserted Option, no Commands ─────────────

fn extract_topology_system(
    mut extracted: ResMut<ExtractedTopologyRaw>,
    topology: Extract<Option<Res<AsteroidTopologyGpuData>>>,
) {
    if extracted.0.is_some() {
        return;
    }
    let Some(topo) = topology.as_ref() else {
        return;
    };
    if topo.node_count == 0 {
        return;
    }

    let pos_bytes: Vec<u8> = topo
        .positions
        .iter()
        .flat_map(|p| [p.x, p.y, p.z, 0.0_f32])
        .flat_map(|f: f32| f.to_le_bytes())
        .collect();
    let off_bytes: Vec<u8> = topo.offsets.iter().flat_map(|u| u.to_le_bytes()).collect();
    let idx_bytes: Vec<u8> = topo.indices.iter().flat_map(|u| u.to_le_bytes()).collect();

    extracted.0 = Some(ExtractedTopologyInner {
        node_count: topo.node_count,
        pos_bytes,
        off_bytes,
        idx_bytes,
    });
}

// ── Dispatch system — writes into pre-inserted Option, no Commands ────────────

fn dispatch_normals_system(
    mut state: ResMut<DispatchState>,
    mut buffers: ResMut<NormalsGpuBuffers>,
    pipeline_res: Option<Res<NormalsComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedTopologyRaw>,
    channel: Res<NormalsReadbackChannel>,
) {
    if *state == DispatchState::Done || buffers.0.is_some() {
        return;
    }
    let Some(pl) = pipeline_res else { return };
    let Some(topo) = extracted.0.as_ref() else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pl.pipeline_id) else {
        return;
    };
    if topo.node_count == 0 {
        return;
    }

    let out_size = (topo.node_count as u64) * 16;

    let pos_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("n_pos"),
        contents: &topo.pos_bytes,
        usage: BufferUsages::STORAGE,
    });
    let off_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("n_off"),
        contents: &topo.off_bytes,
        usage: BufferUsages::STORAGE,
    });
    let idx_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("n_idx"),
        contents: &topo.idx_bytes,
        usage: BufferUsages::STORAGE,
    });
    let out_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("n_out"),
        size: out_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("n_staging"),
        size: out_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = render_device.create_bind_group_layout(
        "normals_bgl_rt",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ],
    );
    let bind_group = render_device.create_bind_group(
        "normals_bg",
        &bgl,
        &[
            BindGroupEntry {
                binding: 0,
                resource: pos_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: off_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: idx_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    );

    let workgroups = topo.node_count.div_ceil(64);
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("normals_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("normals_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
    render_queue.submit([encoder.finish()]);

    let shared = Arc::clone(&channel.0);
    let staging_ref = staging.clone();
    staging.slice(..).map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            let view = staging_ref.slice(..).get_mapped_range();
            let normals = bytes_to_f32x4(&view);
            if let Ok(mut lock) = shared.lock() {
                *lock = Some(normals);
            }
            drop(view);
            staging_ref.unmap();
        }
    });

    buffers.0 = Some(NormalsGpuBuffersInner {
        _pos: pos_buf,
        _off: off_buf,
        _idx: idx_buf,
        _out: out_buf,
        _staging: staging,
    });
    *state = DispatchState::Done;
}

// ── Main-world poll ───────────────────────────────────────────────────────────

fn poll_normals_readback(
    mut commands: Commands,
    channel: Res<NormalsReadbackChannel>,
    existing: Option<Res<AsteroidNormalsGpuData>>,
) {
    if existing.is_some() {
        return;
    }
    let Ok(mut guard) = channel.0.try_lock() else {
        return;
    };
    let Some(raw) = guard.take() else { return };
    let normals: Vec<Vec3> = raw.iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect();
    info!(
        "Successfully readback {} normals from WebGPU!",
        normals.len()
    );
    commands.insert_resource(AsteroidNormalsGpuData(normals));
}
