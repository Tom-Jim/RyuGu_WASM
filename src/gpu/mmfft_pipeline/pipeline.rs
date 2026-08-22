// Two-level three-dimensional MMFFT gravity operator.
//
// Each level conservatively deposits the irregular density cells on a
// Cartesian mesh, zero pads it to twice the physical extent, performs a real
// 3-D FFT convolution with the Newton kernel, and applies the matching IFFT.
// The runtime GPU pass only interpolates the finest containing level; no
// direct quadrature is hidden behind the MMFFT name.

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
    renderer::{RenderDevice, RenderQueue},
};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use std::sync::atomic::Ordering;

const LEVEL_GRID_SIZES: [usize; 2] = [64, 16];
const LEVEL_HALF_EXTENTS: [f64; 2] = [4096.0, 16384.0];

#[derive(Resource, Default)]
struct ExtractedMmfftInput {
    enabled: bool,
    probe: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    source_bytes: Option<Vec<u8>>,
    grid_sizes: [u32; 2],
    level_count: u32,
    half_extents: [f32; 2],
    total_mass: f32,
}

#[derive(Resource, Default)]
struct MmfftGpuBuffers(Option<MmfftGpuBuffersInner>);

struct MmfftGpuBuffersInner {
    uniform: Buffer,
    output: Buffer,
    staging: Buffer,
    bind_group: BindGroup,
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

#[cfg(test)]
fn fft_1d(values: &mut [Complex64], inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let transform = if inverse {
        planner.plan_fft_inverse(values.len())
    } else {
        planner.plan_fft_forward(values.len())
    };
    process_fft_line(values, transform.as_ref(), inverse);
}

fn fft_3d(values: &mut [Complex64], n: usize, inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let transform = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    let mut line = vec![Complex64::default(); n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                line[x] = values[(z * n + y) * n + x];
            }
            process_fft_line(&mut line, transform.as_ref(), inverse);
            for x in 0..n {
                values[(z * n + y) * n + x] = line[x];
            }
        }
    }
    for z in 0..n {
        for x in 0..n {
            for y in 0..n {
                line[y] = values[(z * n + y) * n + x];
            }
            process_fft_line(&mut line, transform.as_ref(), inverse);
            for y in 0..n {
                values[(z * n + y) * n + x] = line[y];
            }
        }
    }
    for y in 0..n {
        for x in 0..n {
            for z in 0..n {
                line[z] = values[(z * n + y) * n + x];
            }
            process_fft_line(&mut line, transform.as_ref(), inverse);
            for z in 0..n {
                values[(z * n + y) * n + x] = line[z];
            }
        }
    }
}

fn process_fft_line(values: &mut [Complex64], transform: &dyn Fft<f64>, inverse: bool) {
    transform.process(values);
    if inverse {
        let scale = (values.len() as f64).recip();
        for value in values {
            *value *= scale;
        }
    }
}

fn grid_index(x: usize, y: usize, z: usize, side: usize) -> usize {
    (z * side + y) * side + x
}

/// Full zero-padded 3-D FFT/IFFT convolution for one hierarchy level.
fn build_level(records: &[(DVec3, f64)], half_extent: f64, n: usize) -> Vec<[f32; 4]> {
    let p = 2 * n;
    let spacing = 2.0 * half_extent / n as f64;
    let mut density = vec![Complex64::default(); p * p * p];
    // Cloud-in-cell deposition is conservative and avoids nearest-cell phase
    // jumps when the irregular radial quadrature is mapped to the FFT mesh.
    for &(position, mass) in records {
        let grid = (position + DVec3::splat(half_extent)) / spacing - DVec3::splat(0.5);
        let base = grid.floor();
        let fraction = grid - base;
        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    let ix = base.x as isize + dx;
                    let iy = base.y as isize + dy;
                    let iz = base.z as isize + dz;
                    if ix < 0
                        || iy < 0
                        || iz < 0
                        || ix >= n as isize
                        || iy >= n as isize
                        || iz >= n as isize
                    {
                        continue;
                    }
                    let weight = if dx == 0 {
                        1.0 - fraction.x
                    } else {
                        fraction.x
                    } * if dy == 0 {
                        1.0 - fraction.y
                    } else {
                        fraction.y
                    } * if dz == 0 {
                        1.0 - fraction.z
                    } else {
                        fraction.z
                    };
                    density[grid_index(ix as usize, iy as usize, iz as usize, p)].re +=
                        mass * weight;
                }
            }
        }
    }
    fft_3d(&mut density, p, false);
    let mass_spectrum = density;
    let mut field = vec![[0.0_f32; 4]; n * n * n];
    // Convolve the Newton potential. The runtime differentiates the same
    // trilinear interpolant analytically, so acceleration and potential remain
    // a discrete conservative pair without three redundant inverse FFTs.
    let mut kernel = vec![Complex64::default(); p * p * p];
    for z in 0..p {
        for y in 0..p {
            for x in 0..p {
                let signed = |index: usize| {
                    if index < n {
                        index as isize
                    } else {
                        index as isize - p as isize
                    }
                };
                let displacement =
                    DVec3::new(signed(x) as f64, signed(y) as f64, signed(z) as f64) * spacing;
                kernel[grid_index(x, y, z, p)].re = 1.0 / displacement.length().max(0.5 * spacing);
            }
        }
    }
    fft_3d(&mut kernel, p, false);
    for (value, mass) in kernel.iter_mut().zip(&mass_spectrum) {
        *value = *value * *mass;
    }
    fft_3d(&mut kernel, p, true);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                field[grid_index(x, y, z, n)][3] =
                    (G as f64 * kernel[grid_index(x, y, z, p)].re) as f32;
            }
        }
    }
    field
}

pub fn build_mmfft_compressed_source_system(
    mut commands: Commands,
    radial: Option<Res<RadialGravitySource>>,
    existing: Option<Res<MmfftCompressedSource>>,
    active_method: Res<ActiveGravityMethod>,
) {
    if existing.is_some() || *active_method != ActiveGravityMethod::MmfftCompressed {
        return;
    }
    let Some(radial) = radial else { return };
    if radial.bytes.len() < 32 {
        return;
    }

    let mut records = Vec::with_capacity(radial.count as usize);
    let mut total_mass = 0.0_f64;
    for chunk in radial.bytes.chunks_exact(32) {
        let direction = DVec3::new(
            read_f32(chunk, 0) as f64,
            read_f32(chunk, 4) as f64,
            read_f32(chunk, 8) as f64,
        )
        .normalize_or_zero();
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
        records.push((direction * radius, mass));
        total_mass += mass;
    }
    let mut bytes =
        Vec::with_capacity(LEVEL_GRID_SIZES.iter().map(|n| n.pow(3)).sum::<usize>() * 16);
    for (grid_size, half_extent) in LEVEL_GRID_SIZES.into_iter().zip(LEVEL_HALF_EXTENTS) {
        for sample in build_level(&records, half_extent, grid_size) {
            for value in sample {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    info!(
        "[mmfft] {} radial records -> nested {:?} grids (zero-padded 3D FFT/IFFT, {} bytes)",
        radial.count,
        LEVEL_GRID_SIZES,
        bytes.len()
    );
    commands.insert_resource(MmfftCompressedSource {
        bytes,
        grid_sizes: LEVEL_GRID_SIZES.map(|value| value as u32),
        level_count: LEVEL_HALF_EXTENTS.len() as u32,
        half_extents: LEVEL_HALF_EXTENTS.map(|value| value as f32),
        total_mass: total_mass as f32,
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
            #[cfg(feature = "eq106-dual-certificate")]
            independent_positive_potential: None,
            body_acceleration_jacobian: None,
            eq106_diagnostics: None,
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
    extracted.grid_sizes = source.grid_sizes;
    extracted.level_count = source.level_count;
    extracted.half_extents = source.half_extents;
    extracted.total_mass = source.total_mass;
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
    if !extracted.enabled || extracted.grid_sizes[0] == 0 || extracted.level_count == 0 {
        return;
    }
    if buffers.0.is_none() {
        let Some(source_bytes) = extracted.source_bytes.as_ref() else {
            return;
        };
        let output_size = 16;
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
        extracted.grid_sizes,
        extracted.level_count,
        extracted.half_extents,
        extracted.total_mass,
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
        pass.dispatch_workgroups(1, 1, 1);
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
    grid_sizes: [u32; 2],
    level_count: u32,
    half_extents: [f32; 2],
    total_mass: f32,
) -> [u8; 48] {
    let mut bytes = [0_u8; 48];
    for (offset, value) in [(0, probe.x), (4, probe.y), (8, probe.z), (12, G)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (16, grid_sizes[0].to_le_bytes()),
        (20, grid_sizes[1].to_le_bytes()),
        (24, level_count.to_le_bytes()),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value);
    }
    for (offset, value) in [
        (32, half_extents[0]),
        (36, half_extents[1]),
        (40, total_mass),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
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
