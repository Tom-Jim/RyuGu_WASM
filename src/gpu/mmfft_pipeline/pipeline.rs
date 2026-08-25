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

fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

fn cubic_derivatives(t: f32) -> [f32; 4] {
    let t2 = t * t;
    [
        -0.5 + 2.0 * t - 1.5 * t2,
        -5.0 * t + 4.5 * t2,
        0.5 + 4.0 * t - 4.5 * t2,
        -t + 1.5 * t2,
    ]
}

fn sample_mmfft_grid(field: &[[f32; 4]], position: Vec3, half_extent: f32, n: usize) -> Vec3 {
    let spacing = 2.0 * half_extent / n as f32;
    let inverse_spacing = spacing.recip();
    let coordinate = (position + Vec3::splat(half_extent)) / spacing - Vec3::splat(0.5);
    let base_floor = coordinate.floor().clamp(Vec3::ONE, Vec3::splat((n - 3) as f32));
    let fraction = (coordinate - base_floor).clamp(Vec3::ZERO, Vec3::ONE);
    let base = base_floor.as_uvec3() - UVec3::ONE;
    let wx = cubic_weights(fraction.x);
    let wy = cubic_weights(fraction.y);
    let wz = cubic_weights(fraction.z);
    let dx = cubic_derivatives(fraction.x);
    let dy = cubic_derivatives(fraction.y);
    let dz = cubic_derivatives(fraction.z);
    let mut gradient = Vec3::ZERO;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let p = field[grid_index(
                    base.x as usize + x,
                    base.y as usize + y,
                    base.z as usize + z,
                    n,
                )][3];
                gradient += p * inverse_spacing
                    * Vec3::new(
                        dx[x] * wy[y] * wz[z],
                        dy[y] * wx[x] * wz[z],
                        dz[z] * wx[x] * wy[y],
                    );
            }
        }
    }
    gradient
}

/// Builds method-consistent MMFFT unit-density voxel columns using the same
/// CIC deposition, zero-padded convolution, and tricubic derivative as WGSL.
pub(crate) fn voxel_basis_sensitivities(
    basis: &VoxelBasisSources,
    samples: &[TrajectoryInversionKnot],
) -> Vec<Vec3> {
    let body_positions = samples
        .iter()
        .map(|sample| sample.body_rotation.inverse() * sample.position)
        .collect::<Vec<_>>();
    let sample_levels = body_positions
        .iter()
        .map(|position| {
            LEVEL_GRID_SIZES
                .into_iter()
                .zip(LEVEL_HALF_EXTENTS)
                .position(|(n, half)| {
                    let margin = 2.0 * half as f32 / n as f32;
                    position.abs().max_element() <= half as f32 - margin
                })
        })
        .collect::<Vec<_>>();
    let used_levels = std::array::from_fn::<_, 2, _>(|level| {
        sample_levels.iter().any(|sample| *sample == Some(level))
    });
    let mut workspaces = LEVEL_GRID_SIZES
        .into_iter()
        .zip(LEVEL_HALF_EXTENTS)
        .enumerate()
        .map(|(level, (n, half))| used_levels[level].then(|| MmfftLevelWorkspace::new(n, half)))
        .collect::<Vec<_>>();
    let column_count = basis.columns.len();
    let mut row_major = vec![Vec3::ZERO; samples.len() * column_count];
    for (column_index, sources) in basis.columns.iter().enumerate() {
        let records = sources
            .iter()
            .map(|source| (source.position, source.volume))
            .collect::<Vec<_>>();
        for (level, workspace) in workspaces.iter_mut().enumerate() {
            let Some(workspace) = workspace.as_mut() else {
                continue;
            };
            let n = workspace.n;
            let half_extent = workspace.half_extent as f32;
            let field = workspace.build(&records);
            for (sample_index, sample) in samples.iter().enumerate() {
                if sample_levels[sample_index] == Some(level) {
                    let acceleration = sample_mmfft_grid(
                        field,
                        body_positions[sample_index],
                        half_extent,
                        n,
                    );
                    row_major[sample_index * column_count + column_index] =
                        sample.body_rotation * acceleration;
                }
            }
        }
        for (sample_index, sample) in samples.iter().enumerate() {
            if sample_levels[sample_index].is_some() {
                continue;
            }
            let body_acceleration = sources.iter().fold(Vec3::ZERO, |sum, source| {
                let displacement = source.position.as_vec3() - body_positions[sample_index];
                let distance_squared = displacement.length_squared().max(1.0e-12);
                sum + displacement
                    * (G * source.volume as f32 / (distance_squared * distance_squared.sqrt()))
            });
            row_major[sample_index * column_count + column_index] =
                sample.body_rotation * body_acceleration;
        }
    }
    row_major
}

pub fn build_mmfft_compressed_source_system(
    mut commands: Commands,
    aggregated: Option<Res<crate::cpu::curved_arc::AggregatedGravitySource>>,
    existing: Option<Res<MmfftCompressedSource>>,
    active_method: Res<ActiveGravityMethod>,
) {
    if existing.is_some() || *active_method != ActiveGravityMethod::MmfftCompressed {
        return;
    }
    let Some(aggregated) = aggregated else {
        return;
    };
    if aggregated.sources.is_empty() {
        return;
    }

    let records = aggregated
        .sources
        .iter()
        .map(|source| (source.position, source.mass))
        .collect::<Vec<_>>();
    let mut workspaces = LEVEL_GRID_SIZES
        .into_iter()
        .zip(LEVEL_HALF_EXTENTS)
        .map(|(n, half)| MmfftLevelWorkspace::new(n, half))
        .collect::<Vec<_>>();
    let mut bytes =
        Vec::with_capacity(LEVEL_GRID_SIZES.iter().map(|n| n.pow(3)).sum::<usize>() * 16);
    for workspace in &mut workspaces {
        for sample in workspace.build(&records) {
            for value in *sample {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    info!(
        "[mmfft] {} common sources -> nested {:?} grids (zero-padded 3D FFT/IFFT, {} bytes)",
        records.len(),
        LEVEL_GRID_SIZES,
        bytes.len()
    );
    commands.insert_resource(MmfftCompressedSource {
        bytes,
        grid_sizes: LEVEL_GRID_SIZES.map(|value| value as u32),
        level_count: LEVEL_HALF_EXTENTS.len() as u32,
        half_extents: LEVEL_HALF_EXTENTS.map(|value| value as f32),
        total_mass: aggregated.total_mass as f32,
    });
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
    planning: Extract<Res<PlanningComparisonState>>,
    clock: Extract<Res<SimulationClock>>,
    cassini: Extract<Query<(&Transform, &Velocity), With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    extracted.enabled = **active_method == ActiveGravityMethod::MmfftCompressed
        && !planning.blocks_realtime_gpu();
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
