// GPU fast multipole method for the fifth gravity slot.
//
// The common 1024-source aggregation is compressed into a six-level linear
// octree once. P2M/M2M mass, center-of-mass, and traceless quadrupole moments
// are stored in breadth-first order. The real-time WGSL pass applies a
// fixed-depth multipole acceptance criterion in parallel and asynchronously
// reads back only workgroup reductions.

use crate::interface::components::*;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        Buffer, BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue}, GpuResourceAppExt,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

const WORKGROUP_SIZE: u32 = 64;
const MAXIMUM_LEVEL: u32 = 5;
// Bounds the complete source-box plus target-box radius used by M2L/L2L.
// 0.20 keeps the order-two local expansion well inside its disk; the shader
// independently certifies each translated value against its node multipole.
const THETA: f32 = 0.10;
const INVALID_PARENT: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default)]
struct MomentAccumulator {
    mass: f64,
    first: DVec3,
    second: [f64; 6],
}

impl MomentAccumulator {
    fn add(&mut self, position: DVec3, mass: f64) {
        self.mass += mass;
        self.first += position * mass;
        let [x, y, z] = position.to_array();
        for (slot, value) in self
            .second
            .iter_mut()
            .zip([x * x, x * y, x * z, y * y, y * z, z * z])
        {
            *slot += mass * value;
        }
    }

    /// Exact raw-moment M2M translation. Raw moments are expressed in the
    /// common body frame, so summing children is algebraically identical to
    /// translating their central moments to the parent center, without the
    /// cancellation introduced by repeated f32 translations.
    fn merge(&mut self, child: Self) {
        self.mass += child.mass;
        self.first += child.first;
        for (parent, child) in self.second.iter_mut().zip(child.second) {
            *parent += child;
        }
    }
}

#[derive(Resource, Default)]
struct ExtractedFmmInput {
    enabled: bool,
    probe: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    node_bytes: Option<Vec<u8>>,
    particle_bytes: Option<Vec<u8>>,
    node_count: u32,
    particle_count: u32,
    maximum_level: u32,
}

#[derive(Resource, Default)]
struct FmmGpuBuffers(Option<FmmGpuBuffersInner>);

struct FmmGpuBuffersInner {
    uniform: Buffer,
    output: Buffer,
    staging: Buffer,
    bind_group: BindGroup,
    workgroup_count: u32,
    output_size: u64,
    last_submitted: Option<(u64, u64)>,
}

#[derive(Resource)]
struct FmmComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

pub struct FmmComputePlugin;

impl Plugin for FmmComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FmmReadbackChannel>();
        app.init_resource::<FmmGravityHistory>();
        app.add_systems(Update, build_fmm_source_system);
        app.add_systems(Update, clear_fmm_history_on_probe_reset);
        app.add_systems(PreUpdate, poll_fmm_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedFmmInput>();
        render_app.init_gpu_resource::<FmmGpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_fmm_input);
        render_app.add_systems(Render, dispatch_fmm.in_set(RenderSystems::Render));
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<FmmReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_gpu_resource::<FmmComputePipeline>();
    }
}

impl FromWorld for FmmComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            storage_ro_entry(3),
        ];
        let layout = BindGroupLayoutDescriptor::new("fmm_gravity_bgl", &entries);
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::Fmm,
        );
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("fmm_gravity_compute".into()),
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

pub fn build_fmm_source_system(
    mut commands: Commands,
    aggregated: Option<Res<crate::cpu::curved_arc::AggregatedGravitySource>>,
    existing: Option<Res<FmmSource>>,
    active_method: Res<ActiveGravityMethod>,
    planning: Res<PlanningComparisonState>,
) {
    if planning.blocks_realtime_gpu() || existing.is_some() || *active_method != ActiveGravityMethod::Fmm {
        return;
    }
    let Some(aggregated) = aggregated else {
        return;
    };
    let records = aggregated
        .sources
        .iter()
        .map(|source| (source.position, source.mass))
        .collect::<Vec<_>>();
    let radius = aggregated.radius;
    if records.is_empty() || radius <= 0.0 {
        return;
    }

    // P2M is performed only at the leaves. Every coarser level is then built
    // exclusively through M2M aggregation, which makes the hierarchy itself
    // (rather than a repeated particle scan) the authoritative source.
    let leaf_grid = 1u32 << MAXIMUM_LEVEL;
    let mut level_maps = vec![HashMap::new(); MAXIMUM_LEVEL as usize + 1];
    let mut leaf_particles: HashMap<(u32, u32, u32), Vec<(DVec3, f64)>> = HashMap::new();
    for &(position, mass) in &records {
        let normalized = (position / radius + DVec3::ONE) * 0.5;
        let key = (
            ((normalized.x.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32)
                .min(leaf_grid - 1),
            ((normalized.y.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32)
                .min(leaf_grid - 1),
            ((normalized.z.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32)
                .min(leaf_grid - 1),
        );
        level_maps[MAXIMUM_LEVEL as usize]
            .entry(key)
            .or_insert_with(MomentAccumulator::default)
            .add(position, mass);
        leaf_particles
            .entry(key)
            .or_default()
            .push((position, mass));
    }
    for level in (1..=MAXIMUM_LEVEL as usize).rev() {
        let children = level_maps[level]
            .iter()
            .map(|(key, moment)| (*key, *moment))
            .collect::<Vec<_>>();
        for (key, child) in children {
            level_maps[level - 1]
                .entry((key.0 / 2, key.1 / 2, key.2 / 2))
                .or_insert_with(MomentAccumulator::default)
                .merge(child);
        }
    }

    let mut levels: Vec<Vec<((u32, u32, u32), MomentAccumulator)>> = Vec::new();
    for cells in level_maps {
        let mut sorted = cells.into_iter().collect::<Vec<_>>();
        sorted.sort_by_key(|(key, _)| *key);
        levels.push(sorted);
    }

    let mut level_offsets = Vec::with_capacity(levels.len());
    let mut offset = 0u32;
    for level in &levels {
        level_offsets.push(offset);
        offset += level.len() as u32;
    }
    let mut index_maps = Vec::with_capacity(levels.len());
    for (level_index, level) in levels.iter().enumerate() {
        let map = level
            .iter()
            .enumerate()
            .map(|(index, (key, _))| (*key, level_offsets[level_index] + index as u32))
            .collect::<HashMap<_, _>>();
        index_maps.push(map);
    }

    let mut particle_bytes = Vec::with_capacity(records.len() * 16);
    let mut leaf_ranges = HashMap::new();
    let mut sorted_leaf_keys = leaf_particles.keys().copied().collect::<Vec<_>>();
    sorted_leaf_keys.sort_unstable();
    for key in sorted_leaf_keys {
        let particles = &leaf_particles[&key];
        let start = (particle_bytes.len() / 16) as u32;
        for &(position, mass) in particles {
            for value in [
                position.x as f32,
                position.y as f32,
                position.z as f32,
                mass as f32,
            ] {
                particle_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        leaf_ranges.insert(key, (start, particles.len() as u32));
    }

    let mut bytes = Vec::with_capacity(offset as usize * 80);
    for (level_index, level) in levels.iter().enumerate() {
        let grid = 1u32 << level_index;
        let cell_width = 2.0 * radius / grid as f64;
        for (key, moment) in level {
            let center = DVec3::new(
                -radius + (key.0 as f64 + 0.5) * cell_width,
                -radius + (key.1 as f64 + 0.5) * cell_width,
                -radius + (key.2 as f64 + 0.5) * cell_width,
            );
            let com = moment.first / moment.mass.max(f64::MIN_POSITIVE);
            let [x, y, z] = com.to_array();
            let central = [
                moment.second[0] - moment.mass * x * x,
                moment.second[1] - moment.mass * x * y,
                moment.second[2] - moment.mass * x * z,
                moment.second[3] - moment.mass * y * y,
                moment.second[4] - moment.mass * y * z,
                moment.second[5] - moment.mass * z * z,
            ];
            let trace = central[0] + central[3] + central[5];
            let quadrupole = [
                3.0 * central[0] - trace,
                3.0 * central[1],
                3.0 * central[2],
                3.0 * central[3] - trace,
                3.0 * central[4],
                3.0 * central[5] - trace,
            ];
            for value in [
                center.x as f32,
                center.y as f32,
                center.z as f32,
                (0.5 * cell_width) as f32,
                com.x as f32,
                com.y as f32,
                com.z as f32,
                moment.mass as f32,
                quadrupole[0] as f32,
                quadrupole[1] as f32,
                quadrupole[2] as f32,
                0.0,
                quadrupole[3] as f32,
                quadrupole[4] as f32,
                quadrupole[5] as f32,
                0.0,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let parent = if level_index == 0 {
                INVALID_PARENT
            } else {
                index_maps[level_index - 1][&(key.0 / 2, key.1 / 2, key.2 / 2)]
            };
            let (particle_start, particle_count) = if level_index == MAXIMUM_LEVEL as usize {
                leaf_ranges.get(key).copied().unwrap_or((0, 0))
            } else {
                (0, 0)
            };
            for value in [parent, level_index as u32, particle_start, particle_count] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    info!(
        "[fmm] built {} octree nodes from {} common sources through level {} (P2M/M2M + M2L/L2L + P2P)",
        offset,
        records.len(),
        MAXIMUM_LEVEL
    );
    commands.insert_resource(FmmSource {
        bytes,
        particle_bytes,
        node_count: offset,
        particle_count: records.len() as u32,
        maximum_level: MAXIMUM_LEVEL,
    });
}

fn clear_fmm_history_on_probe_reset(
    probe: Res<ProbeInitialConditions>,
    mut history: ResMut<FmmGravityHistory>,
) {
    if probe.is_changed() {
        history.0.clear();
    }
}

fn poll_fmm_readback(channel: Res<FmmReadbackChannel>, mut history: ResMut<FmmGravityHistory>) {
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    let Some(packet) = guard.take() else { return };
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
    if total.xyz().is_finite() && total.w.is_finite() && total.w > 0.0 {
        history.0.push(GravityFieldSample {
            snapshot: packet.snapshot,
            predictive: false,
            body_acceleration: total.xyz(),
            positive_potential: total.w,
            #[cfg(feature = "eq106-dual-certificate")]
            independent_positive_potential: None,
            body_acceleration_jacobian: None,
        });
    }
}

fn extract_fmm_input(
    mut extracted: ResMut<ExtractedFmmInput>,
    source: Extract<Option<Res<FmmSource>>>,
    active: Extract<Res<ActiveGravityMethod>>,
    planning: Extract<Res<PlanningComparisonState>>,
    clock: Extract<Res<SimulationClock>>,
    cassini: Extract<Query<(&Transform, &Velocity), With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    extracted.enabled = **active == ActiveGravityMethod::Fmm && !planning.blocks_realtime_gpu();
    if !extracted.enabled {
        return;
    }
    let (Some(source), Ok((probe, velocity)), Ok(ryugu)) =
        (source.as_ref(), cassini.single(), ryugu.single())
    else {
        return;
    };
    extracted.probe = ryugu.rotation.inverse() * (probe.translation - ryugu.translation);
    extracted.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
        body_position: extracted.probe,
        ryugu_transform: *ryugu,
        probe_position: probe.translation,
        probe_velocity: velocity.0,
    });
    extracted.node_count = source.node_count;
    extracted.particle_count = source.particle_count;
    extracted.maximum_level = source.maximum_level;
    if extracted.node_bytes.is_none() {
        extracted.node_bytes = Some(source.bytes.clone());
    }
    if extracted.particle_bytes.is_none() {
        extracted.particle_bytes = Some(source.particle_bytes.clone());
    }
}

fn dispatch_fmm(
    mut buffers: ResMut<FmmGpuBuffers>,
    pipeline: Option<Res<FmmComputePipeline>>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedFmmInput>,
    channel: Res<FmmReadbackChannel>,
) {
    let Some(pipeline) =
        pipeline.and_then(|pipeline| cache.get_compute_pipeline(pipeline.pipeline_id))
    else {
        return;
    };
    if !extracted.enabled || extracted.node_count == 0 {
        return;
    }
    if buffers.0.is_none() {
        let (Some(node_bytes), Some(particle_bytes)) = (
            extracted.node_bytes.as_ref(),
            extracted.particle_bytes.as_ref(),
        ) else {
            return;
        };
        let workgroup_count = extracted.node_count.div_ceil(WORKGROUP_SIZE);
        let output_size = workgroup_count as u64 * 16;
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("fmm_uniform"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nodes = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fmm_nodes"),
            contents: node_bytes,
            usage: BufferUsages::STORAGE,
        });
        let particles = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fmm_leaf_particles"),
            contents: particle_bytes,
            usage: BufferUsages::STORAGE,
        });
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("fmm_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("fmm_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "fmm_gravity_bgl_runtime",
            &[
                uniform_entry(0),
                storage_ro_entry(1),
                storage_rw_entry(2),
                storage_ro_entry(3),
            ],
        );
        let bind_group = render_device.create_bind_group(
            "fmm_gravity_bg",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: nodes.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: particles.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(FmmGpuBuffersInner {
            uniform,
            output,
            staging,
            bind_group,
            workgroup_count,
            output_size,
            last_submitted: None,
        });
    }
    let inner = buffers.0.as_mut().expect("FMM buffers initialized");
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        return;
    };
    let key = (snapshot.epoch, snapshot.request_id);
    if inner.last_submitted == Some(key) {
        return;
    }
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    inner.last_submitted = Some(key);
    render_queue.write_buffer(
        &inner.uniform,
        0,
        &uniform_bytes(
            extracted.probe,
            extracted.node_count,
            extracted.maximum_level,
            extracted.particle_count,
        ),
    );
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("fmm_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fmm_pass"),
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
                if let Ok(mut guard) = shared.lock() {
                    *guard = Some(GravityReadbackPacket {
                        partial_sums: bytes_to_f32x4(&view),
                        snapshot,
                    });
                }
                drop(view);
                staging.unmap();
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn uniform_bytes(
    probe: Vec3,
    node_count: u32,
    maximum_level: u32,
    particle_count: u32,
) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (offset, value) in [
        (0, probe.x),
        (4, probe.y),
        (8, probe.z),
        (12, G),
        (24, THETA),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[16..20].copy_from_slice(&node_count.to_le_bytes());
    bytes[20..24].copy_from_slice(&maximum_level.to_le_bytes());
    bytes[28..32].copy_from_slice(&particle_count.to_le_bytes());
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
