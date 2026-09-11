//! Independent Eq.106/121 GPU diagnostic stamps.
//! Live Verlet force is Worker FLUPS Eq.(121) at `q_B` (inverse Laplace of
//! Eq.(106)); do not feed these stamps into propagation. Eq.(184) is the
//! known-curve Fourier–Laplace observation used by density inversion and the
//! spectral chart, not the live Verlet force.
use crate::cpu::frequency_domain::{
    EQ184_QUADRATURE_COUNT, EQ184_QUADRATURE_LAYOUT, eq184_quadrature_node,
};
use crate::interface::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};
use bevy::shader::ShaderCacheError;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Resource, Default)]
pub(crate) struct Equation106History(pub GravitySampleHistory);

#[derive(Resource, Clone, Default)]
pub(crate) struct Equation106Channel {
    data: Arc<Mutex<Option<GravityReadbackPacket>>>,
    error: Arc<Mutex<Option<String>>>,
    in_flight: Arc<AtomicBool>,
}
impl Equation106Channel {
    pub(crate) fn reset_after_device_loss(&self) {
        if let Ok(mut slot) = self.data.lock() {
            *slot = None;
        }
        if let Ok(mut slot) = self.error.lock() {
            *slot = None;
        }
        self.in_flight.store(false, Ordering::Release);
    }
}
#[derive(Resource, Default)]
struct Input {
    enabled: bool,
    position: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    source_bytes: Vec<u8>,
    source_count: u32,
    source_hash: u64,
    radius: f64,
    mass: f32,
    gm: f32,
    center: Vec3,
}
#[derive(Resource, Default)]
struct Buffers(Option<BufferState>);
struct BufferState {
    uniform: Buffer,
    output: Buffer,
    staging: Buffer,
    bind_group: BindGroup,
    source_hash: u64,
    layout: u64,
    last: Option<(u64, u64)>,
    spectrum_ready: bool,
}
#[derive(Resource)]
struct Pipelines {
    density: CachedComputePipelineId,
    field: CachedComputePipelineId,
}
pub(crate) struct Equation106Plugin;
impl Plugin for Equation106Plugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Equation106History>()
            .init_resource::<Equation106Channel>()
            .add_systems(PreUpdate, poll);
        let render = app.sub_app_mut(RenderApp);
        render
            .init_resource::<Input>()
            .init_gpu_resource::<Buffers>()
            .add_systems(ExtractSchedule, extract)
            .add_systems(Render, dispatch.in_set(RenderSystems::Render));
    }
    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<Equation106Channel>().clone();
        let render = app.sub_app_mut(RenderApp);
        render.insert_resource(channel);
        render.init_gpu_resource::<Pipelines>();
    }
}
fn entries() -> [BindGroupLayoutEntry; 5] {
    std::array::from_fn(|binding| BindGroupLayoutEntry {
        binding: binding as u32,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: if binding == 0 {
                BufferBindingType::Uniform
            } else {
                BufferBindingType::Storage {
                    read_only: binding < 3,
                }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    })
}
impl FromWorld for Pipelines {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new("eq106_layout", &entries());
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::Equation106,
        );
        let cache = world.resource::<PipelineCache>();
        let queue = |entry: &'static str| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("eq106_{entry}").into()),
                layout: vec![layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
            })
        };
        let density = queue("density");
        let field = queue("field");
        Self { density, field }
    }
}
fn extract(
    mut input: ResMut<Input>,
    active: Extract<Res<ActiveGravityMethod>>,
    source: Extract<Option<Res<DensityQuadratureSource>>>,
    clock: Extract<Res<SimulationClock>>,
    planning: Extract<Res<PlanningComparisonState>>,
    density_mode: Extract<Res<DensityMode>>,
    probe: Extract<Query<&Transform, With<CassiniMarker>>>,
    body: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    input.enabled =
        **active == ActiveGravityMethod::FrequencyDomain && !planning.blocks_realtime_gpu();
    input.snapshot = None;
    if !input.enabled {
        return;
    }
    let (Some(source), Ok(probe), Ok(body)) = (source.as_ref(), probe.single(), body.single())
    else {
        return;
    };
    let (bytes, hash) = match **density_mode {
        DensityMode::Variable => (&source.bytes, source.source_hash),
        DensityMode::Constant => (&source.constant_bytes, source.constant_hash),
    };
    if input.source_bytes.is_empty() || input.source_hash != hash {
        input.source_bytes = bytes.clone();
        input.source_count = (bytes.len() / 32) as u32;
        input.source_hash = hash;
        input.radius = source.radius as f64;
        if let Some((mass, center)) =
            crate::cpu::frequency_domain::quadrature_mass_centroid(bytes.as_chunks::<32>().0)
        {
            input.mass = mass as f32;
            input.center = Vec3::new(center.x as f32, center.y as f32, center.z as f32);
            input.gm = G * input.mass;
        }
    }
    input.position = body.rotation.inverse() * (probe.translation - body.translation);
    input.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
    });
}
fn poll(
    channel: Res<Equation106Channel>,
    mut history: ResMut<Equation106History>,
    clock: Res<SimulationClock>,
    active: Res<ActiveGravityMethod>,
) {
    if let Ok(mut slot) = channel.error.try_lock()
        && let Some(message) = slot.take()
        && *active == ActiveGravityMethod::FrequencyDomain
    {
        // GPU Eq.(106) is the independent diagnostic stamp. Live Verlet is
        // Worker FLUPS Eq.(121); a failed diagnostic must not freeze the orbit.
        bevy::log::warn!("Equation (106) GPU diagnostic: {message}");
    }
    let Ok(mut slot) = channel.data.try_lock() else {
        return;
    };
    let Some(packet) = slot.take() else {
        return;
    };
    if packet.snapshot.epoch != clock.epoch || *active != ActiveGravityMethod::FrequencyDomain {
        return;
    }
    let Some(value) = packet.partial_sums.first() else {
        return;
    };
    let value = Vec4::from_array(*value);
    if !value.is_finite() {
        bevy::log::warn!(
            "Equation (106) GPU diagnostic returned a non-finite field; live Verlet continues."
        );
        return;
    }
    history.0.push(GravityFieldSample {
        snapshot: packet.snapshot,
        body_acceleration: value.xyz(),
    });
}
fn dispatch(
    mut buffers: ResMut<Buffers>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    input: Res<Input>,
    channel: Res<Equation106Channel>,
) {
    if !input.enabled || input.source_bytes.is_empty() || channel.in_flight.load(Ordering::Acquire)
    {
        return;
    }
    let Some(snapshot) = input.snapshot.as_ref() else {
        return;
    };
    for id in [pipelines.density, pipelines.field] {
        if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(id)
            && !matches!(
                error,
                ShaderCacheError::ShaderNotLoaded(_)
                    | ShaderCacheError::ShaderImportNotYetAvailable
            )
        {
            if let Ok(mut slot) = channel.error.lock() {
                *slot = Some(format!("Equation (106) GPU pipeline failed: {error}"));
            }
            return;
        }
    }
    let (Some(density), Some(field)) = (
        cache.get_compute_pipeline(pipelines.density),
        cache.get_compute_pipeline(pipelines.field),
    ) else {
        return;
    };
    if buffers.0.as_ref().is_some_and(|state| {
        state.source_hash != input.source_hash || state.layout != EQ184_QUADRATURE_LAYOUT
    }) {
        buffers.0 = None;
    }
    if buffers.0.is_none() {
        let nodes: Vec<[f32; 4]> = (0..EQ184_QUADRATURE_COUNT)
            .filter_map(|index| {
                eq184_quadrature_node(index, input.radius)
                    .map(|(k, w)| [k.x as f32, k.y as f32, k.z as f32, w as f32])
            })
            .collect();
        if nodes.len() != EQ184_QUADRATURE_COUNT {
            return;
        }
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("eq106_params"),
            size: 48,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sources = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_sources"),
            contents: &input.source_bytes,
            usage: BufferUsages::STORAGE,
        });
        let nodes = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_wave_nodes"),
            contents: bytemuck::cast_slice(&nodes),
            usage: BufferUsages::STORAGE,
        });
        let spectrum = device.create_buffer(&BufferDescriptor {
            label: Some("eq106_density_spectrum"),
            size: EQ184_QUADRATURE_COUNT as u64 * 8,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("eq106_field"),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("eq106_readback"),
            size: 16,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout("eq106_runtime_layout", &entries());
        let bind_group = device.create_bind_group(
            "eq106_runtime_bindings",
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
                    resource: nodes.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: spectrum.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(BufferState {
            uniform,
            output,
            staging,
            bind_group,
            source_hash: input.source_hash,
            layout: EQ184_QUADRATURE_LAYOUT,
            last: None,
            spectrum_ready: false,
        });
    }
    let state = buffers.0.as_mut().expect("initialized eq106 buffers");
    let identity = (snapshot.epoch, snapshot.request_id);
    if state.last == Some(identity) {
        return;
    }
    if channel.in_flight.swap(true, Ordering::AcqRel) {
        return;
    }
    let mut params = [0u8; 48];
    for (index, value) in [input.position.x, input.position.y, input.position.z, G]
        .iter()
        .enumerate()
    {
        params[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    params[16..20].copy_from_slice(&input.source_count.to_le_bytes());
    params[20..24].copy_from_slice(&(EQ184_QUADRATURE_COUNT as u32).to_le_bytes());
    params[24..28].copy_from_slice(&input.mass.to_le_bytes());
    params[28..32].copy_from_slice(&input.gm.to_le_bytes());
    for (index, value) in [input.center.x, input.center.y, input.center.z, 0.0]
        .iter()
        .enumerate()
    {
        let offset = 32 + index * 4;
        params[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    queue.write_buffer(&state.uniform, 0, &params);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("eq106_encoder"),
    });
    if !state.spectrum_ready {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_density"),
            timestamp_writes: None,
        });
        pass.set_pipeline(density);
        pass.set_bind_group(0, &state.bind_group, &[]);
        pass.dispatch_workgroups(EQ184_QUADRATURE_COUNT as u32, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_inverse_residue"),
            timestamp_writes: None,
        });
        pass.set_pipeline(field);
        pass.set_bind_group(0, &state.bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&state.output, 0, &state.staging, 0, 16);
    queue.submit([encoder.finish()]);
    state.spectrum_ready = true;
    state.last = Some(identity);
    let staging = state.staging.clone();
    let mapped = staging.clone();
    let channel = channel.clone();
    let snapshot = snapshot.clone();
    mapped.slice(..).map_async(MapMode::Read, move |result| {
        match result {
            Ok(()) => {
                let view = staging.slice(..).get_mapped_range();
                let packet = GravityReadbackPacket {
                    snapshot,
                    partial_sums: bytes_to_f32x4(&view),
                };
                if let Ok(mut slot) = channel.data.lock() {
                    *slot = Some(packet);
                }
                drop(view);
                staging.unmap();
            }
            Err(error) => {
                if let Ok(mut slot) = channel.error.lock() {
                    *slot = Some(format!("Equation (106) readback failed: {error:?}"));
                }
            }
        }
        channel.in_flight.store(false, Ordering::Release);
    });
}
