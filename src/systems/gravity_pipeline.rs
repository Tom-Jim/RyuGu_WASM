use crate::components::*;
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

// ── Render-world extracted frame ──────────────────────────────────────────────

#[derive(Resource, Default)]
struct ExtractedGravInput {
    probe_x:     f32,
    probe_y:     f32,
    probe_z:     f32,
    voxel_bytes: Option<Vec<u8>>, // populated only once
    voxel_count: u32,
}

// ── Render-world GPU buffers ──────────────────────────────────────────────────

#[derive(Resource, Default)]
struct GravGpuBuffers(Option<GravGpuBuffersInner>);

struct GravGpuBuffersInner {
    uniform_buf:  bevy::render::render_resource::Buffer,
    voxel_buf:    bevy::render::render_resource::Buffer,
    output_buf:   bevy::render::render_resource::Buffer,
    voxel_count:  u32,
    n_workgroups: u32,
}

#[derive(Resource)]
struct GravComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource, Default, PartialEq, Eq)]
enum GravDispatchState {
    #[default]
    NeedsBuild,
    Ready,
    Dispatched,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct GravityComputePlugin;

impl Plugin for GravityComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GravityReadbackChannel>();
        app.add_systems(
            Update,
            (build_gravity_voxels_system, poll_gravity_readback).chain(),
        );

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedGravInput>();
        render_app.init_resource::<GravGpuBuffers>();
        render_app.init_resource::<GravDispatchState>();
        render_app.add_systems(ExtractSchedule, extract_grav_input_system);
        render_app.add_systems(
            Render,
            dispatch_grav_system.in_set(RenderSystems::Render),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<GravityReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<GravComputePipeline>();
    }
}

// ── FromWorld ─────────────────────────────────────────────────────────────────

impl FromWorld for GravComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
        ];
        let bgl = BindGroupLayoutDescriptor::new("grav_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/gravity.wgsl");
        let pipeline_id = world
            .resource::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label:    Some("grav_compute".into()),
                layout:   vec![bgl],
                immediate_size: 0,
                shader,
                shader_defs: vec![],
                entry_point: None,
                zero_initialize_workgroup_memory: false,
            });
        GravComputePipeline { pipeline_id }
    }
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_ro_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── Main-world: build voxel buffer once from asteroid topology ────────────────

pub fn build_gravity_voxels_system(
    mut commands: Commands,
    topo:     Option<Res<AsteroidTopologyGpuData>>,
    ryugu_q:  Query<&Transform, With<RyuguMarker>>,
    existing: Option<Res<GravVoxelSource>>,
) {
    if existing.is_some() { return; }
    let Some(topo) = topo else { return };
    let Ok(ryugu_tf) = ryugu_q.single() else { return };
    if topo.positions.is_empty() { return; }

    let scale = ryugu_tf.scale.x;
    let n     = topo.positions.len() as u32;
    let eps   = DENSITY_EPSILON;

    // 1. 根据 1/r 非均匀密度计算每个顶点的原始权重
    let raw_weights: Vec<f32> = topo.positions.iter().map(|p| {
        let r_v = (*p * scale).length().max(1e-3);
        1.0 / (r_v + eps)
    }).collect();

    // 2. 对权重求和，计算全局归一化系数，确保总质量绝对守恒
    let sum_weights: f32 = raw_weights.iter().sum();
    let mass_norm = RYUGU_MASS / sum_weights;

    // 3. 注入数据
    let bytes: Vec<u8> = topo
        .positions
        .iter()
        .zip(raw_weights.iter())
        .flat_map(|(p, w)| {
            let pw = *p * scale;
            let mass = w * mass_norm;
            [pw.x, pw.y, pw.z, mass]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect();

    commands.insert_resource(GravVoxelSource { bytes, count: n });
}

// ── Main-world: receive GPU partial sums and reduce to Vec3 ───────────────────

pub fn poll_gravity_readback(
    channel:      Res<GravityReadbackChannel>,
    mut grav_acc: ResMut<GravityAcceleration>,
) {
    let Ok(mut guard) = channel.0.try_lock() else { return };
    let Some(partial_sums) = guard.take() else { return };

    let total = partial_sums.iter().fold(Vec3::ZERO, |acc, v| {
        acc + Vec3::new(v[0], v[1], v[2])
    });
    grav_acc.0 = total;
}

// ── Extract system ────────────────────────────────────────────────────────────

fn extract_grav_input_system(
    mut extracted: ResMut<ExtractedGravInput>,
    mut dispatch:  ResMut<GravDispatchState>,
    voxels:        Extract<Option<Res<GravVoxelSource>>>,
    cassini_q:     Extract<Query<&Transform, With<CassiniMarker>>>,
    ryugu_q:       Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    if *dispatch == GravDispatchState::Dispatched {
        *dispatch = GravDispatchState::Ready;
    }
    let Some(vox_src)   = voxels.as_ref()          else { return };
    let Ok(cassini_tf)  = cassini_q.single()        else { return };
    let Ok(ryugu_tf)    = ryugu_q.single()          else { return };

    // Cassini position in Ryugu body-fixed Cartesian frame
    let inv_rot   = ryugu_tf.rotation.inverse();
    let local_pos = inv_rot * (cassini_tf.translation - ryugu_tf.translation);

    extracted.probe_x    = local_pos.x;
    extracted.probe_y    = local_pos.y;
    extracted.probe_z    = local_pos.z;
    extracted.voxel_count = vox_src.count;

    if extracted.voxel_bytes.is_none() {
        extracted.voxel_bytes = Some(vox_src.bytes.clone());
    }
}

// ── Render-world dispatch ─────────────────────────────────────────────────────

fn dispatch_grav_system(
    mut state:      ResMut<GravDispatchState>,
    mut buffers:    ResMut<GravGpuBuffers>,
    pipeline_res:   Option<Res<GravComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device:  Res<RenderDevice>,
    render_queue:   Res<RenderQueue>,
    extracted:      Res<ExtractedGravInput>,
    channel:        Res<GravityReadbackChannel>,
) {
    if *state == GravDispatchState::Dispatched {
        return;
    }
    let Some(pl)       = pipeline_res else { return };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pl.pipeline_id) else { return };
    if extracted.voxel_count == 0 { return; }

    // Build persistent voxel + output buffers on first use
    if buffers.0.is_none() {
        let Some(voxel_bytes) = extracted.voxel_bytes.as_ref() else { return };
        let n_wg   = extracted.voxel_count.div_ceil(64);
        let out_sz = (n_wg as u64) * 16;

        let uniform_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("grav_uniform"),
            size:  32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let voxel_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label:    Some("grav_voxels"),
            contents: voxel_bytes,
            usage:    BufferUsages::STORAGE,
        });
        let output_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("grav_output"),
            size:  out_sz,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        buffers.0 = Some(GravGpuBuffersInner {
            uniform_buf,
            voxel_buf,
            output_buf,
            voxel_count:  extracted.voxel_count,
            n_workgroups: n_wg,
        });
        *state = GravDispatchState::Ready;
    }

    if *state != GravDispatchState::Ready { return; }

    let bufs   = buffers.0.as_ref().unwrap();
    let n_wg   = bufs.n_workgroups;
    let out_sz = (n_wg as u64) * 16;

    // GravParams uniform layout (32 bytes):
    //   probe_x/y/z: f32  (offsets 0-11)
    //   G_const: f32      (offset 12)
    //   voxel_count: u32  (offset 16)
    //   _pad0/1/2: u32    (offsets 20-28)
    let mut ub = Vec::<u8>::with_capacity(32);
    for f in [extracted.probe_x, extracted.probe_y, extracted.probe_z, G] {
        ub.extend_from_slice(&f.to_le_bytes());
    }
    ub.extend_from_slice(&bufs.voxel_count.to_le_bytes());
    ub.extend_from_slice(&0u32.to_le_bytes());
    ub.extend_from_slice(&0u32.to_le_bytes());
    ub.extend_from_slice(&0u32.to_le_bytes());
    render_queue.write_buffer(&bufs.uniform_buf, 0, &ub);

    // Fresh staging buffer each frame (tiny: ceil(N/64)*16 bytes)
    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("grav_staging"),
        size:  out_sz,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = render_device.create_bind_group_layout(
        "grav_bgl_rt",
        &[uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)],
    );
    let bind_group = render_device.create_bind_group(
        "grav_bg",
        &bgl,
        &[
            BindGroupEntry { binding: 0, resource: bufs.uniform_buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: bufs.voxel_buf.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: bufs.output_buf.as_entire_binding() },
        ],
    );

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("grav_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label:            Some("grav_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&bufs.output_buf, 0, &staging, 0, out_sz);
    render_queue.submit([encoder.finish()]);

    let shared      = Arc::clone(&channel.0);
    let staging_ref = staging.clone();
    staging.slice(..).map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            let view    = staging_ref.slice(..).get_mapped_range();
            let partial = bytes_to_f32x4(&view);
            if let Ok(mut lock) = shared.lock() {
                *lock = Some(partial);
            }
            drop(view);
            staging_ref.unmap();
        }
    });

    *state = GravDispatchState::Dispatched;
}

fn bytes_to_f32x4(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .chunks_exact(16)
        .map(|c| {
            let mut v = [0f32; 4];
            for (i, b4) in c.chunks_exact(4).enumerate() {
                v[i] = f32::from_le_bytes(b4.try_into().unwrap());
            }
            v
        })
        .collect()
}
