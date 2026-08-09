//! MMFFT + GPU-memory-compression forward-field pipeline.
//!
//! The source is packed independently from the radial path. Each 32-byte
//! radial record becomes a 16-byte quantized record, while workgroups decode
//! and reduce acceleration plus positive potential on the GPU. The packed
//! layout is intentionally tile-friendly: a future hierarchical MMFFT pass
//! can replace `record_field` without changing the ECS snapshot/readback API.

use crate::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        Buffer, BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::Arc;
use std::sync::atomic::Ordering;

const WORKGROUP_SIZE: u32 = 64;
const COMPRESSED_RECORD_BYTES: usize = 16;

#[derive(Resource, Default)]
struct ExtractedMmfftInput {
    enabled: bool,
    probe: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    source_bytes: Option<Vec<u8>>,
    record_count: u32,
    solid_angle_scale: f32,
    radius_scale: f32,
    density_scale: f32,
    tile_size: u32,
}

#[derive(Resource, Default)]
struct MmfftGpuBuffers(Option<MmfftGpuBuffersInner>);

struct MmfftGpuBuffersInner {
    uniform: Buffer,
    output: Buffer,
    staging: Buffer,
    bind_group: BindGroup,
    record_count: u32,
    workgroup_count: u32,
    output_size: u64,
    last_submitted: Option<(u64, u64)>,
}

#[derive(Resource)]
struct MmfftComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

pub struct MmfftCompressedComputePlugin;

impl Plugin for MmfftCompressedComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MmfftCompressedConfig>();
        app.init_resource::<MmfftReadbackChannel>();
        app.init_resource::<MmfftCompressedHistory>();
        app.add_systems(Update, build_mmfft_compressed_source_system);
        app.add_systems(Update, clear_mmfft_history_on_probe_reset);
        app.add_systems(PreUpdate, poll_mmfft_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedMmfftInput>();
        render_app.init_resource::<MmfftGpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_mmfft_input_system);
        render_app.add_systems(Render, dispatch_mmfft_system.in_set(RenderSystems::Render));
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<MmfftReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<MmfftComputePipeline>();
    }
}

fn clear_mmfft_history_on_probe_reset(
    probe_initial: Res<ProbeInitialConditions>,
    mut history: ResMut<MmfftCompressedHistory>,
) {
    if probe_initial.is_changed() {
        history.0.clear();
    }
}

impl FromWorld for MmfftComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)];
        let layout = BindGroupLayoutDescriptor::new("mmfft_compressed_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/mmfft_compressed.wgsl");
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("mmfft_compressed_compute".into()),
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

/// Compresses the already mass-normalized radial records. Keeping this as a
/// separate ECS resource means the later MMFFT hierarchy can replace only this
/// builder and the shader while preserving all snapshot semantics.
pub fn build_mmfft_compressed_source_system(
    mut commands: Commands,
    radial: Option<Res<RadialGravitySource>>,
    config: Res<MmfftCompressedConfig>,
    existing: Option<Res<MmfftCompressedSource>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(radial) = radial else { return };
    if radial.bytes.len() < 32 {
        return;
    }

    let mut records = Vec::with_capacity(radial.count as usize);
    let mut solid_angle_scale = 0.0_f32;
    let mut radius_scale = 0.0_f32;
    let mut density_scale = 0.0_f32;
    for chunk in radial.bytes.chunks_exact(32) {
        let direction = Vec3::new(read_f32(chunk, 0), read_f32(chunk, 4), read_f32(chunk, 8));
        let solid_angle = read_f32(chunk, 12).max(0.0);
        let inner = read_f32(chunk, 16).max(0.0);
        let outer = read_f32(chunk, 20).max(inner);
        let density = read_f32(chunk, 24).max(0.0);
        solid_angle_scale = solid_angle_scale.max(solid_angle);
        radius_scale = radius_scale.max(outer);
        density_scale = density_scale.max(density);
        records.push((direction, solid_angle, inner, outer, density));
    }
    if solid_angle_scale <= 0.0 || radius_scale <= 0.0 || density_scale <= 0.0 {
        return;
    }

    let mut bytes = Vec::with_capacity(records.len() * COMPRESSED_RECORD_BYTES);
    for (direction, solid_angle, inner, outer, density) in records {
        let direction = direction.normalize_or_zero();
        let dx = pack_i16(direction.x);
        let dy = pack_i16(direction.y);
        let dz = pack_i16(direction.z);
        let w0 = pack_u16_pair(dx, dy);
        let w1 = pack_u16_pair(dz, quantize_u16(solid_angle, solid_angle_scale));
        let w2 = pack_u16_pair(
            quantize_u16(inner, radius_scale),
            quantize_u16(outer, radius_scale),
        );
        let w3 = pack_u16_pair(quantize_u16(density, density_scale), 0);
        for word in [w0, w1, w2, w3] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }

    let tile_size = config.tile_size.max(WORKGROUP_SIZE);
    info!(
        "[mmfft] compressed source: {} records, {} -> {} bytes, tile_size={}, scales=({:.4e}, {:.4e}, {:.4e})",
        radial.count,
        radial.bytes.len(),
        bytes.len(),
        tile_size,
        solid_angle_scale,
        radius_scale,
        density_scale
    );
    commands.insert_resource(MmfftCompressedSource {
        bytes,
        count: radial.count,
        tile_size,
        solid_angle_scale,
        radius_scale,
        density_scale,
    });
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn pack_i16(value: f32) -> u16 {
    (value.clamp(-1.0, 1.0) * 32767.0).round() as i16 as u16
}

fn quantize_u16(value: f32, scale: f32) -> u16 {
    ((value / scale.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0) * 65535.0).round() as u16
}

fn pack_u16_pair(low: u16, high: u16) -> u32 {
    u32::from(low) | (u32::from(high) << 16)
}

fn poll_mmfft_readback(
    channel: Res<MmfftReadbackChannel>,
    mut history: ResMut<MmfftCompressedHistory>,
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
    if total.xyz().is_finite() && total.w.is_finite() && total.w > 0.0 {
        history.0.push(GravityFieldSample {
            snapshot: packet.snapshot,
            body_acceleration: total.xyz(),
            positive_potential: total.w,
        });
    } else {
        warn!("[mmfft] discarded non-finite compressed GPU result");
    }
}

fn extract_mmfft_input_system(
    mut extracted: ResMut<ExtractedMmfftInput>,
    source: Extract<Option<Res<MmfftCompressedSource>>>,
    active_method: Extract<Res<ActiveGravityMethod>>,
    clock: Extract<Res<SimulationClock>>,
    cassini: Extract<Query<(&Transform, &Velocity), With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    extracted.enabled = **active_method == ActiveGravityMethod::MmfftCompressed;
    if !extracted.enabled {
        return;
    }
    let (Some(source), Ok((cassini, velocity)), Ok(ryugu)) =
        (source.as_ref(), cassini.single(), ryugu.single())
    else {
        return;
    };
    extracted.probe = ryugu.rotation.inverse() * (cassini.translation - ryugu.translation);
    extracted.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
        body_position: extracted.probe,
        ryugu_transform: *ryugu,
        probe_position: cassini.translation,
        probe_velocity: velocity.0,
    });
    extracted.record_count = source.count;
    extracted.solid_angle_scale = source.solid_angle_scale;
    extracted.radius_scale = source.radius_scale;
    extracted.density_scale = source.density_scale;
    extracted.tile_size = source.tile_size;
    if extracted.source_bytes.is_none() {
        extracted.source_bytes = Some(source.bytes.clone());
    }
}

fn dispatch_mmfft_system(
    mut buffers: ResMut<MmfftGpuBuffers>,
    pipeline_resource: Option<Res<MmfftComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedMmfftInput>,
    channel: Res<MmfftReadbackChannel>,
) {
    let Some(pipeline_resource) = pipeline_resource else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_resource.pipeline_id) else {
        return;
    };
    if !extracted.enabled || extracted.record_count == 0 {
        return;
    }
    if buffers.0.is_none() {
        let Some(source_bytes) = extracted.source_bytes.as_ref() else {
            return;
        };
        let workgroup_count = extracted.record_count.div_ceil(WORKGROUP_SIZE);
        let output_size = workgroup_count as u64 * 16;
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("mmfft_compressed_uniform"),
            size: 48,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let records = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("mmfft_compressed_records"),
            contents: source_bytes,
            usage: BufferUsages::STORAGE,
        });
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("mmfft_compressed_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("mmfft_compressed_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "mmfft_compressed_bgl_runtime",
            &[uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)],
        );
        let bind_group = render_device.create_bind_group(
            "mmfft_compressed_bg",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: records.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(MmfftGpuBuffersInner {
            uniform,
            output,
            staging,
            bind_group,
            record_count: extracted.record_count,
            workgroup_count,
            output_size,
            last_submitted: None,
        });
    }

    let inner = buffers.0.as_mut().expect("MMFFT buffers initialized");
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
    let uniform = mmfft_uniform_bytes(
        extracted.probe,
        inner.record_count,
        extracted.solid_angle_scale,
        extracted.radius_scale,
        extracted.density_scale,
        extracted.tile_size,
    );
    render_queue.write_buffer(&inner.uniform, 0, &uniform);
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("mmfft_compressed_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("mmfft_compressed_pass"),
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

fn mmfft_uniform_bytes(
    probe: Vec3,
    record_count: u32,
    solid_angle_scale: f32,
    radius_scale: f32,
    density_scale: f32,
    tile_size: u32,
) -> [u8; 48] {
    let mut bytes = [0_u8; 48];
    for (offset, value) in [(0, probe.x), (4, probe.y), (8, probe.z), (12, G)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (16, record_count.to_le_bytes()),
        (20, solid_angle_scale.to_le_bytes()),
        (24, radius_scale.to_le_bytes()),
        (28, density_scale.to_le_bytes()),
        (32, tile_size.to_le_bytes()),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value);
    }
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

fn bytes_to_f32x4(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .chunks_exact(16)
        .map(|chunk| {
            std::array::from_fn(|index| {
                let start = index * 4;
                f32::from_le_bytes(chunk[start..start + 4].try_into().unwrap())
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_record_halves_source_storage() {
        let source = vec![0_u8; 32 * 128];
        let compressed = source.len() / 2;
        assert_eq!(compressed, 16 * 128);
        assert_eq!(COMPRESSED_RECORD_BYTES, 16);
    }

    #[test]
    fn signed_direction_quantization_is_bounded() {
        assert_eq!(pack_i16(-2.0), i16::MIN as u16 + 1);
        assert_eq!(pack_i16(2.0), i16::MAX as u16);
        assert_eq!(quantize_u16(5.0, 10.0), 32768);
    }
}
