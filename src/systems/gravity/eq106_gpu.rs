//! GPU-resident Equation (106) engineering evaluator.
//!
//! Expensive half-line kernel assembly is performed only when the reference
//! line changes. Real-time frames reuse the assembled 257-frequency buffer and
//! dispatch a single lightweight Bromwich/translation pass every six render
//! frames. The main WASM thread only polls the asynchronous readback.
//!
//! The pass is deliberately documented as an engineering approximation to the
//! derivation in `docs/mathtidy_EN.md`: it uses fixed quadrature over the
//! point-mass source representation, an analytic Cartesian field correction,
//! and a certified toroidal-harmonic cross-check. CPU Taylor/Padé certificates
//! remain separate from this real-time WGSL evaluator.

use crate::components::*;
use crate::systems::curved_arc::Eq106SourceData;
use crate::systems::eq106_operator::Eq106OperatorTensorResource;
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

const HALF_COUNT: u32 = 128;
const FREQUENCY_COUNT: u32 = 2 * HALF_COUNT + 1;
const QUADRATURE_COUNT: u32 = 256;
const EVALUATION_CADENCE_FRAMES: u32 = 6;
const SPECTRUM_BYTES: u64 = FREQUENCY_COUNT as u64 * 32;
const OUTPUT_BYTES: u64 = 32;

#[derive(Resource, Default)]
struct ExtractedEq106Input {
    enabled: bool,
    probe: Vec3,
    velocity: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    sources: Option<Vec<u8>>,
    operator_tensor: Option<Vec<u8>>,
    source_count: u32,
    radius: f32,
    source_hash: u64,
}

#[derive(Resource, Default)]
struct Eq106GpuBuffers(Option<Eq106GpuBuffersInner>);

struct Eq106GpuBuffersInner {
    uniform: Buffer,
    output: Buffer,
    staging: Buffer,
    bind_group: BindGroup,
    line_origin: Vec3,
    line_direction: Vec3,
    source_hash: u64,
    spectrum_ready: bool,
    render_frame: u32,
    last_submitted: Option<(u64, u64)>,
}

#[derive(Resource)]
struct Eq106ComputePipeline {
    assemble_id: CachedComputePipelineId,
    evaluate_id: CachedComputePipelineId,
}

pub struct Eq106GpuComputePlugin;

impl Plugin for Eq106GpuComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Eq106GpuReadbackChannel>();
        app.init_resource::<Eq106GpuHistory>();
        app.add_systems(PreUpdate, poll_eq106_readback);
        app.add_systems(Update, clear_eq106_history_on_probe_reset);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedEq106Input>();
        render_app.init_resource::<Eq106GpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_eq106_input);
        render_app.add_systems(Render, dispatch_eq106.in_set(RenderSystems::Render));
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<Eq106GpuReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<Eq106ComputePipeline>();
    }
}

impl FromWorld for Eq106ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            storage_rw_entry(4),
            storage_ro_entry(5),
        ];
        let layout = BindGroupLayoutDescriptor::new("eq106_complex_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/eq106_complex.wgsl");
        let cache = world.resource::<PipelineCache>();
        let assemble_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_spectrum".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("assemble_spectrum".into()),
            zero_initialize_workgroup_memory: false,
        });
        let evaluate_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_evaluate_field".into()),
            layout: vec![layout],
            immediate_size: 0,
            shader,
            shader_defs: vec![],
            entry_point: Some("evaluate_field".into()),
            zero_initialize_workgroup_memory: false,
        });
        Self {
            assemble_id,
            evaluate_id,
        }
    }
}

fn clear_eq106_history_on_probe_reset(
    probe: Res<ProbeInitialConditions>,
    mut history: ResMut<Eq106GpuHistory>,
) {
    if probe.is_changed() {
        history.0.clear();
    }
}

fn poll_eq106_readback(
    channel: Res<Eq106GpuReadbackChannel>,
    mut history: ResMut<Eq106GpuHistory>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    let Some(packet) = guard.take() else { return };
    let Some(field) = packet.partial_sums.first().copied() else {
        return;
    };
    let certificate = packet
        .partial_sums
        .get(1)
        .copied()
        .unwrap_or([f32::INFINITY; 4]);
    if certificate[0] > 0.25
        || certificate[1] > 0.05
        || (certificate[3] > 0.99 && certificate[2] > 0.25)
    {
        runtime_error.raise(format!(
            "Equation (106) GPU certification failed (field={:.3e}, imaginary={:.3e}, toroidal={:.3e}, coverage={:.3e}).",
            certificate[0], certificate[1], certificate[2], certificate[3]
        ));
        return;
    }
    let acceleration = Vec3::new(field[0], field[1], field[2]);
    if acceleration.is_finite() && field[3].is_finite() && field[3] > 0.0 {
        history.0.push(GravityFieldSample {
            snapshot: packet.snapshot,
            body_acceleration: acceleration,
            positive_potential: field[3],
        });
    } else {
        runtime_error.raise("Equation (106) GPU returned a non-finite field sample.");
    }
}

fn extract_eq106_input(
    mut extracted: ResMut<ExtractedEq106Input>,
    source: Extract<Option<Res<Eq106SourceData>>>,
    operator_tensor: Extract<Option<Res<Eq106OperatorTensorResource>>>,
    active: Extract<Res<ActiveGravityMethod>>,
    clock: Extract<Res<SimulationClock>>,
    cassini: Extract<Query<(&Transform, &Velocity), With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    extracted.enabled = **active == ActiveGravityMethod::CurvedArcEq106;
    if !extracted.enabled {
        return;
    }
    let (Some(source), Ok((probe, velocity)), Ok(ryugu)) =
        (source.as_ref(), cassini.single(), ryugu.single())
    else {
        return;
    };
    extracted.probe = ryugu.rotation.inverse() * (probe.translation - ryugu.translation);
    extracted.velocity = ryugu.rotation.inverse() * velocity.0;
    extracted.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
        body_position: extracted.probe,
        ryugu_transform: *ryugu,
        probe_position: probe.translation,
        probe_velocity: velocity.0,
    });
    extracted.source_count = source.sources.len() as u32;
    extracted.radius = source.radius as f32;
    let source_hash = source.source_hash;
    if extracted.sources.is_none() || extracted.source_hash != source_hash {
        let mut bytes = Vec::with_capacity(source.sources.len() * 16);
        for item in &source.sources {
            for value in [
                item.position.x as f32,
                item.position.y as f32,
                item.position.z as f32,
                item.mass as f32,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        extracted.sources = Some(bytes);
    }
    extracted.source_hash = source_hash;
    if extracted.operator_tensor.is_none() {
        extracted.operator_tensor = operator_tensor
            .as_ref()
            .map(|resource| resource.tensor.as_le_bytes());
    }
}

fn dispatch_eq106(
    mut buffers: ResMut<Eq106GpuBuffers>,
    pipelines: Option<Res<Eq106ComputePipeline>>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedEq106Input>,
    channel: Res<Eq106GpuReadbackChannel>,
) {
    let Some(pipelines) = pipelines else { return };
    let (Some(assemble), Some(evaluate)) = (
        cache.get_compute_pipeline(pipelines.assemble_id),
        cache.get_compute_pipeline(pipelines.evaluate_id),
    ) else {
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
        // The source buffer is immutable in the render world. Rebuild the
        // bind group when the mass-preserving radial source hash changes.
        buffers.0 = None;
    }
    if buffers.0.is_none() {
        let (Some(source_bytes), Some(operator_bytes)) = (
            extracted.sources.as_ref(),
            extracted.operator_tensor.as_ref(),
        ) else {
            return;
        };
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_uniform"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sources = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_sources"),
            contents: source_bytes,
            usage: BufferUsages::STORAGE,
        });
        let quadrature_bytes = half_line_quadrature_bytes();
        let quadrature = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_quadrature_lut"),
            contents: &quadrature_bytes,
            usage: BufferUsages::STORAGE,
        });
        let operator_tensor = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_toroidal_operator_tensor"),
            contents: operator_bytes,
            usage: BufferUsages::STORAGE,
        });
        let spectrum = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_spectrum"),
            size: SPECTRUM_BYTES,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_output"),
            size: OUTPUT_BYTES,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_staging"),
            size: OUTPUT_BYTES,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "eq106_complex_bgl_runtime",
            &[
                uniform_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
                storage_rw_entry(4),
                storage_ro_entry(5),
            ],
        );
        let bind_group = render_device.create_bind_group(
            "eq106_complex_bg",
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
                    resource: quadrature.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: spectrum.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: operator_tensor.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(Eq106GpuBuffersInner {
            uniform,
            output,
            staging,
            bind_group,
            line_origin: extracted.probe,
            line_direction: extracted.velocity.normalize_or_zero(),
            source_hash: extracted.source_hash,
            spectrum_ready: false,
            render_frame: 0,
            last_submitted: None,
        });
    }

    let inner = buffers.0.as_mut().expect("Eq106 GPU buffers initialized");
    inner.render_frame = inner.render_frame.wrapping_add(1);
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        return;
    };
    let relative = extracted.probe - inner.line_origin;
    let h = relative.dot(inner.line_direction);
    let transverse = (relative - h * inner.line_direction).length();
    let spectral_horizon = 0.75 * std::f32::consts::TAU / 0.002;
    let line_expired = inner.source_hash != extracted.source_hash
        || h < 0.0
        || h > spectral_horizon
        || transverse > (0.25 * extracted.radius).max(20.0);
    if line_expired {
        inner.line_origin = extracted.probe;
        inner.line_direction = extracted.velocity.normalize_or_zero();
        inner.source_hash = extracted.source_hash;
        inner.spectrum_ready = false;
    }
    if inner.line_direction == Vec3::ZERO {
        return;
    }
    let key = (snapshot.epoch, snapshot.request_id);
    if inner.spectrum_ready
        && (!inner.render_frame.is_multiple_of(EVALUATION_CADENCE_FRAMES)
            || inner.last_submitted == Some(key))
    {
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
    let uniform = uniform_bytes(
        extracted.probe,
        inner.line_origin,
        inner.line_direction,
        extracted.source_count,
        extracted.radius,
    );
    render_queue.write_buffer(&inner.uniform, 0, &uniform);
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("eq106_complex_encoder"),
    });
    if !inner.spectrum_ready {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_assemble_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(assemble);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(FREQUENCY_COUNT.div_ceil(64), 1, 1);
        drop(pass);
        inner.spectrum_ready = true;
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_evaluate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(evaluate);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&inner.output, 0, &inner.staging, 0, OUTPUT_BYTES);
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
                let values = bytes_to_f32x4(&view);
                if let Ok(mut guard) = shared.lock() {
                    *guard = Some(GravityReadbackPacket {
                        partial_sums: values,
                        snapshot,
                    });
                }
                drop(view);
                staging.unmap();
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn half_line_quadrature_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(QUADRATURE_COUNT as usize * 8);
    let du = 1.0 / QUADRATURE_COUNT as f32;
    for index in 0..QUADRATURE_COUNT {
        let u = (index as f32 + 0.5) * du;
        let denominator = 1.0 - u;
        let h = u / denominator;
        let weight = du / (denominator * denominator);
        bytes.extend_from_slice(&h.to_le_bytes());
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    bytes
}

fn uniform_bytes(
    probe: Vec3,
    origin: Vec3,
    direction: Vec3,
    source_count: u32,
    radius: f32,
) -> [u8; 64] {
    let mut bytes = [0_u8; 64];
    for (offset, value) in [
        (0, probe.x),
        (4, probe.y),
        (8, probe.z),
        (12, G),
        (16, origin.x),
        (20, origin.y),
        (24, origin.z),
        (28, 2.0 / radius.max(1.0)),
        (32, direction.x),
        (36, direction.y),
        (40, direction.z),
        (44, 0.002),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (48, source_count),
        (52, HALF_COUNT),
        (56, QUADRATURE_COUNT),
        (60, 0),
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
