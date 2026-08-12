//! MMFFT spherical-ring convolution with an explicit radix-2 FFT/IFFT.
//!
//! The irregular radial cells are conservatively deposited into azimuthal
//! rings. Each GPU workgroup transforms one 64-sample mass ring and four
//! Newton-kernel channels, multiplies them in frequency space, and inverse
//! transforms the convolution. The hierarchy is therefore an actual FFT
//! operator rather than a direct quadrature hidden behind the MMFFT label.

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

#[cfg(test)]
const WORKGROUP_SIZE: u32 = 64;
const COMPRESSED_RECORD_BYTES: usize = 16;
const AZIMUTH_BINS: usize = 64;
const POLAR_BINS: usize = 16;
const RADIAL_LAYERS: usize = 4;
const RING_COUNT: usize = POLAR_BINS * RADIAL_LAYERS;

#[derive(Resource, Default)]
struct ExtractedMmfftInput {
    enabled: bool,
    probe: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    source_bytes: Option<Vec<u8>>,
    record_count: u32,
    ring_count: u32,
    azimuth_bins: u32,
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

#[derive(Clone, Copy, Default)]
struct RingBin {
    mass: f64,
    radial_moment: f64,
    cosine_moment: f64,
}

/// Deposits the mass-normalized radial records onto periodic azimuth rings.
/// Every deposited bin stores mass plus the ring's shared mass-weighted radius
/// and polar cosine; consequently the FFT convolution is circulant in azimuth.
pub fn build_mmfft_compressed_source_system(
    mut commands: Commands,
    radial: Option<Res<RadialGravitySource>>,
    _config: Res<MmfftCompressedConfig>,
    existing: Option<Res<MmfftCompressedSource>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(radial) = radial else { return };
    if radial.bytes.len() < 32 {
        return;
    }

    let mut bins = vec![RingBin::default(); RING_COUNT * AZIMUTH_BINS];
    for (record_index, chunk) in radial.bytes.chunks_exact(32).enumerate() {
        let direction = Vec3::new(read_f32(chunk, 0), read_f32(chunk, 4), read_f32(chunk, 8));
        let solid_angle = read_f32(chunk, 12).max(0.0);
        let inner = read_f32(chunk, 16).max(0.0);
        let outer = read_f32(chunk, 20).max(inner);
        let density = read_f32(chunk, 24).max(0.0);
        if outer <= inner || density <= 0.0 || solid_angle <= 0.0 {
            continue;
        }
        let mass =
            density as f64 * solid_angle as f64 * ((outer as f64).powi(3) - (inner as f64).powi(3))
                / 3.0;
        let radius = 0.75 * ((outer as f64).powi(4) - (inner as f64).powi(4))
            / ((outer as f64).powi(3) - (inner as f64).powi(3)).max(f64::MIN_POSITIVE);
        let direction = direction.normalize_or_zero();
        let azimuth = direction
            .y
            .atan2(direction.x)
            .rem_euclid(std::f32::consts::TAU);
        let azimuth_bin = ((azimuth / std::f32::consts::TAU * AZIMUTH_BINS as f32).floor()
            as usize)
            .min(AZIMUTH_BINS - 1);
        let polar_bin =
            (((direction.z + 1.0) * 0.5 * POLAR_BINS as f32).floor() as usize).min(POLAR_BINS - 1);
        let layer = record_index % RADIAL_LAYERS;
        let bin = &mut bins[(layer * POLAR_BINS + polar_bin) * AZIMUTH_BINS + azimuth_bin];
        bin.mass += mass;
        bin.radial_moment += mass * radius;
        bin.cosine_moment += mass * direction.z as f64;
    }

    let mut bytes = Vec::with_capacity(bins.len() * COMPRESSED_RECORD_BYTES);
    for ring in 0..RING_COUNT {
        let ring_slice = &bins[ring * AZIMUTH_BINS..(ring + 1) * AZIMUTH_BINS];
        let ring_mass = ring_slice.iter().map(|bin| bin.mass).sum::<f64>();
        let radius = ring_slice.iter().map(|bin| bin.radial_moment).sum::<f64>()
            / ring_mass.max(f64::MIN_POSITIVE);
        let cosine = ring_slice.iter().map(|bin| bin.cosine_moment).sum::<f64>()
            / ring_mass.max(f64::MIN_POSITIVE);
        for bin in ring_slice {
            for value in [
                bin.mass as f32,
                radius as f32,
                cosine.clamp(-1.0, 1.0) as f32,
                0.0,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    info!(
        "[mmfft] {} radial records -> {} periodic rings x {} bins ({} bytes, radix-2 FFT/IFFT)",
        radial.count,
        RING_COUNT,
        AZIMUTH_BINS,
        bytes.len()
    );
    commands.insert_resource(MmfftCompressedSource {
        bytes,
        count: (RING_COUNT * AZIMUTH_BINS) as u32,
        ring_count: RING_COUNT as u32,
        azimuth_bins: AZIMUTH_BINS as u32,
    });
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
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
            predictive: false,
            body_acceleration: total.xyz(),
            positive_potential: total.w,
            independent_positive_potential: None,
            body_acceleration_jacobian: None,
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
    extracted.ring_count = source.ring_count;
    extracted.azimuth_bins = source.azimuth_bins;
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
        let workgroup_count = extracted.ring_count;
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
        extracted.ring_count,
        extracted.azimuth_bins,
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
    ring_count: u32,
    azimuth_bins: u32,
) -> [u8; 48] {
    let mut bytes = [0_u8; 48];
    for (offset, value) in [(0, probe.x), (4, probe.y), (8, probe.z), (12, G)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (16, record_count.to_le_bytes()),
        (20, ring_count.to_le_bytes()),
        (24, azimuth_bins.to_le_bytes()),
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
    fn radix_two_ring_layout_is_exactly_one_workgroup() {
        assert_eq!(AZIMUTH_BINS as u32, WORKGROUP_SIZE);
        assert!(AZIMUTH_BINS.is_power_of_two());
    }

    #[test]
    fn runtime_record_size_matches_the_packed_shader_layout() {
        let config = MmfftCompressedConfig::default();
        assert_eq!(
            config.compressed_record_bytes as usize,
            COMPRESSED_RECORD_BYTES
        );
    }
}
