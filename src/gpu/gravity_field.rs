//! Discrete gravity-arrow glyphs via WGSL compute + Bevy gizmos + Worker.
//!
//! - Section forward model (all methods → Worker pointwise; never mass-point
//!   N-body):
//!   - Frequency-domain → Worker FLUPS Eq.(121) = inv-Laplace of Eq.(106) at a
//!     point (same operator as live `evaluate("frequency_domain")`)
//!   - Radial / Werner / FMM / FFT → Worker pointwise operators on configured
//!     geometry
//! - Inversion overlay (Section off): N-body on displayed inverted voxels
//!   (gravity of the recovered density field), including Frequency-domain
//!
//! Sample positions and accelerations are body-frame. Rendering rotates both
//! with the live Ryugu attitude so the field tracks spin in real time.

use crate::cpu::frequency_domain::{AggregatedGravitySource, FrequencyDomainPointSource};
use crate::interface::components::*;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages, ShaderType,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bevy::platform::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const GRAVITY_ARROW_SHELLS: u32 = 6;
pub const GRAVITY_ARROW_PER_SHELL: u32 = 24;
pub const GRAVITY_ARROW_COUNT: usize = (GRAVITY_ARROW_SHELLS * GRAVITY_ARROW_PER_SHELL) as usize;
/// First shell sits clear of the irregular mesh (sphere bound underestimates lobes).
const SURFACE_PADDING: f32 = 2.4;
const OUTER_RADIUS_FACTOR: f32 = 8.0;
const GPU_STALL_TIMEOUT: Duration = Duration::from_millis(2500);
/// Keep inverted-voxel N-body cheap enough for interactive WebGPU.
const MAX_NBODY_SOURCES: usize = 4096;
/// Soft hemisphere cull when Section is on (body_pos · view_body).
const SECTION_HEMISPHERE_EPS: f32 = 0.05;

const EVAL_MODE_NBODY: u32 = 0;
/// Worker pointwise radial / Werner / FMM / FFT / frequency_domain (no GPU payload).
const EVAL_MODE_WORKER: u32 = 2;

#[derive(Resource, Clone, Default)]
pub struct GravityFieldChannel {
    pub in_flight: Arc<AtomicBool>,
    pub result: Arc<std::sync::Mutex<Option<(u64, Vec<[f32; 4]>)>>>,
}

impl GravityFieldChannel {
    fn new() -> Self {
        Self {
            in_flight: Arc::new(AtomicBool::new(false)),
            result: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub fn reset_after_device_loss(&self) {
        self.in_flight.store(false, Ordering::Release);
        if let Ok(mut slot) = self.result.try_lock() {
            slot.take();
        }
    }
}

#[derive(Resource, Default)]
pub struct GravityFieldGlyphs {
    pub positions: Vec<Vec3>,
    /// Body-frame acceleration `(gx, gy, gz, |g|)`.
    pub fields: Vec<Vec4>,
    pub enabled: bool,
    pub method: Option<ActiveGravityMethod>,
    pub source_hash: u64,
    pub request_revision: u64,
    pub eval_mode: u32,
    pub source_bytes: Vec<u8>,
    pub source_count: u32,
    pub center_gm: Vec4,
    /// Body-frame radius of the innermost exterior sample shell.
    pub surface_radius: f32,
    pub worker_pending: Option<crate::cpp_backend::BackendGravityFieldSnapshot>,
    pub worker_request_id: u64,
    /// Revision last handed to the Worker; prevents per-frame FMM glyph spam.
    pub worker_dispatched_revision: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct GravityFieldUniform {
    counts: Vec4,
    center_gm: Vec4,
}

#[derive(Resource, Default)]
struct ExtractedGravityField(Option<ExtractedGravityFieldInner>);

struct ExtractedGravityFieldInner {
    revision: u64,
    position_bytes: Vec<u8>,
    source_bytes: Vec<u8>,
    sample_count: u32,
    source_count: u32,
    eval_mode: u32,
    center_gm: Vec4,
}

#[derive(Resource, Default)]
struct GravityFieldGpu(Option<GravityFieldGpuInner>);

struct GravityFieldGpuInner {
    _uniform: bevy::render::render_resource::Buffer,
    _positions: bevy::render::render_resource::Buffer,
    _sources: bevy::render::render_resource::Buffer,
    _fields: bevy::render::render_resource::Buffer,
    _staging: bevy::render::render_resource::Buffer,
    revision: u64,
    submitted_at: Instant,
}

#[derive(Resource)]
struct GravityFieldPipeline {
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource, Default, PartialEq, Eq)]
enum GravityFieldDispatchState {
    #[default]
    Idle,
    Mapped,
}

pub struct GravityFieldComputePlugin;

impl Plugin for GravityFieldComputePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GravityFieldChannel::new())
            .init_resource::<GravityFieldGlyphs>()
            .add_systems(
                Update,
                (
                    prepare_gravity_field_samples_system,
                    seed_gravity_field_cpu_system,
                    dispatch_worker_gravity_field_system,
                    poll_gravity_field_readback_system,
                    poll_worker_gravity_field_system,
                    render_gravity_field_gizmos_system,
                )
                    .chain(),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<ExtractedGravityField>()
            .init_gpu_resource::<GravityFieldGpu>()
            .init_gpu_resource::<GravityFieldDispatchState>()
            .add_systems(ExtractSchedule, extract_gravity_field_system)
            .add_systems(
                Render,
                dispatch_gravity_field_system.in_set(RenderSystems::Render),
            );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<GravityFieldChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_gpu_resource::<GravityFieldPipeline>();
    }
}

impl FromWorld for GravityFieldPipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ];
        let bgl = BindGroupLayoutDescriptor::new("gravity_field_bgl", &entries);
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::GravityField,
        );
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("gravity_field_evaluate".into()),
                    layout: vec![bgl],
                    immediate_size: 0,
                    shader,
                    shader_defs: vec![],
                    entry_point: Some("evaluate_gravity_field".into()),
                    zero_initialize_workgroup_memory: false,
                });
        Self { pipeline_id }
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

fn prepare_gravity_field_samples_system(
    show_section: Res<ShowSection>,
    active_method: Res<ActiveGravityMethod>,
    density_mode: Res<DensityMode>,
    inversion: Res<TrajectoryInversionState>,
    aggregated: Option<Res<AggregatedGravitySource>>,
    quadrature: Option<Res<DensityQuadratureSource>>,
    worker_channel: Res<crate::cpp_backend::BackendGravityFieldChannel>,
    gpu_channel: Res<GravityFieldChannel>,
    mut glyphs: ResMut<GravityFieldGlyphs>,
) {
    // Match density_slice: Section always shows the forward model; inversion
    // glyphs use N-body on the recovered density when Section is off (including
    // Frequency-domain — post-invert gravity of the inverted field).
    let inverted = if show_section.0 {
        None
    } else {
        inversion
            .displayed_density
            .as_ref()
            .filter(|result| result.method == *active_method)
    };
    let overlay = inverted.is_some();
    let enabled = show_section.0 || overlay;
    let source_hash = if let Some(result) = inverted.filter(|_| overlay) {
        // Include mean density so arrows refresh when the slice rebakes.
        result.source_hash ^ result.voxels.len() as u64 ^ result.density.to_bits() as u64
    } else {
        match *active_method {
            ActiveGravityMethod::FrequencyDomain => {
                quadrature.as_ref().map_or(0, |source| match *density_mode {
                    DensityMode::Variable => source.source_hash,
                    DensityMode::Constant => source.constant_hash,
                })
            }
            ActiveGravityMethod::HomogeneousWerner => aggregated
                .as_ref()
                .map_or(0, |source| source.constant_hash),
            _ => aggregated.as_ref().map_or(0, |source| match *density_mode {
                DensityMode::Variable => source.source_hash,
                DensityMode::Constant => source.constant_hash,
            }),
        }
    };

    let surface_radius = exterior_surface_radius(aggregated.as_deref(), quadrature.as_deref());
    let identity_changed = glyphs.method != Some(*active_method)
        || glyphs.source_hash != source_hash
        || glyphs.enabled != enabled;
    if identity_changed {
        glyphs.method = enabled.then_some(*active_method);
        glyphs.source_hash = source_hash;
        glyphs.enabled = enabled;
        glyphs.request_revision = glyphs.request_revision.wrapping_add(1);
        glyphs.eval_mode = 0;
        glyphs.fields.clear();
        glyphs.source_bytes.clear();
        glyphs.source_count = 0;
        glyphs.center_gm = Vec4::ZERO;
        glyphs.worker_pending = None;
        glyphs.worker_dispatched_revision = None;
        worker_channel.reset();
        gpu_channel.reset_after_device_loss();
    }
    if !enabled {
        glyphs.positions.clear();
        return;
    }
    if glyphs.positions.len() != GRAVITY_ARROW_COUNT
        || (glyphs.surface_radius - surface_radius).abs() > 1.0
    {
        glyphs.surface_radius = surface_radius;
        glyphs.positions = build_shell_sample_positions(surface_radius);
        glyphs.request_revision = glyphs.request_revision.wrapping_add(1);
        glyphs.fields.clear();
        glyphs.worker_dispatched_revision = None;
    }

    // Worker glyphs intentionally carry an empty GPU payload; do not treat that
    // as "needs rebuild". Never bump request_revision on a timer alone — body
    // rotation is applied at draw time, and payload identity already gates refresh.
    let needs_payload = identity_changed
        || (glyphs.eval_mode != EVAL_MODE_WORKER && glyphs.source_bytes.is_empty());
    if !needs_payload {
        return;
    }

    let assembled = if overlay {
        let Some(result) = inverted else {
            return;
        };
        pack_inverted_voxels(result)
            .map(|(bytes, count)| (EVAL_MODE_NBODY, bytes, count, Vec4::ZERO))
    } else {
        // Section forward: Worker pointwise for every method, including FD FLUPS.
        match *active_method {
            ActiveGravityMethod::FrequencyDomain
            | ActiveGravityMethod::RadialAnalytic
            | ActiveGravityMethod::HomogeneousWerner
            | ActiveGravityMethod::Fmm
            | ActiveGravityMethod::MmfftCompressed => {
                Some((EVAL_MODE_WORKER, Vec::new(), glyphs.positions.len() as u32, Vec4::ZERO))
            }
        }
    };

    let Some((eval_mode, source_bytes, source_count, center_gm)) = assembled else {
        return;
    };
    if eval_mode != EVAL_MODE_WORKER && (source_bytes.is_empty() || source_count == 0) {
        return;
    }

    glyphs.eval_mode = eval_mode;
    glyphs.source_bytes = source_bytes;
    glyphs.source_count = source_count;
    glyphs.center_gm = center_gm;
}

/// Immediate host evaluation so N-body invert arrows appear before WGSL readback.
fn seed_gravity_field_cpu_system(mut glyphs: ResMut<GravityFieldGlyphs>) {
    if !glyphs.enabled
        || glyphs.positions.is_empty()
        || glyphs.eval_mode == EVAL_MODE_WORKER
        || glyphs.source_bytes.is_empty()
        || glyphs.source_count == 0
        || !glyphs.fields.is_empty()
    {
        return;
    }
    let Some(fields) = evaluate_payload_cpu(
        glyphs.source_count,
        &glyphs.source_bytes,
        &glyphs.positions,
    ) else {
        return;
    };
    glyphs.fields = fields;
}

fn dispatch_worker_gravity_field_system(
    channel: Res<crate::cpp_backend::BackendGravityFieldChannel>,
    mut glyphs: ResMut<GravityFieldGlyphs>,
) {
    if !glyphs.enabled
        || glyphs.eval_mode != EVAL_MODE_WORKER
        || glyphs.positions.is_empty()
        || glyphs.worker_pending.is_some()
        || !channel.is_idle()
    {
        return;
    }
    // One Worker submission per revision. Without this, idle completion re-arms
    // every frame and 144 FMM pointwise evals hitch live orbit. The Worker
    // drains live `advance` inside gravity_field tiles, so orbit stays smooth
    // while Section glyphs fill.
    if glyphs.worker_dispatched_revision == Some(glyphs.request_revision) {
        return;
    }
    let Some(method) = glyphs.method else {
        return;
    };
    glyphs.worker_request_id = glyphs.worker_request_id.wrapping_add(1).max(1);
    let snapshot = crate::cpp_backend::BackendGravityFieldSnapshot {
        request_id: glyphs.worker_request_id,
        epoch: glyphs.request_revision,
    };
    let targets: Vec<DVec3> = glyphs
        .positions
        .iter()
        .map(|position| DVec3::new(position.x as f64, position.y as f64, position.z as f64))
        .collect();
    match crate::cpp_backend::request_gravity_field(&channel, snapshot, method, &targets) {
        Ok(true) => {
            glyphs.worker_pending = Some(snapshot);
            glyphs.worker_dispatched_revision = Some(glyphs.request_revision);
        }
        Ok(false) => {}
        Err(message) => {
            bevy::log::warn!("Gravity-field Worker request failed: {message}");
        }
    }
}

fn poll_worker_gravity_field_system(
    channel: Res<crate::cpp_backend::BackendGravityFieldChannel>,
    mut glyphs: ResMut<GravityFieldGlyphs>,
) {
    let Some(packet) = channel.take() else {
        return;
    };
    let pending = glyphs.worker_pending.take();
    if pending != Some(packet.snapshot) || packet.snapshot.epoch != glyphs.request_revision {
        return;
    }
    let Ok(values) = packet.result else {
        // Allow an immediate retry on the next idle channel tick.
        glyphs.worker_dispatched_revision = None;
        return;
    };
    if values.len() != glyphs.positions.len() * 4 {
        return;
    }
    let mut fields = Vec::with_capacity(glyphs.positions.len());
    for chunk in values.as_chunks::<4>().0 {
        let acceleration = Vec3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32);
        if !acceleration.is_finite() {
            fields.push(Vec4::ZERO);
            continue;
        }
        fields.push(acceleration.extend(acceleration.length()));
    }
    glyphs.fields = fields;
    glyphs.source_count = glyphs.positions.len() as u32;
}

fn poll_gravity_field_readback_system(
    channel: Res<GravityFieldChannel>,
    mut glyphs: ResMut<GravityFieldGlyphs>,
) {
    let Ok(mut slot) = channel.result.try_lock() else {
        return;
    };
    let Some((revision, fields)) = slot.take() else {
        return;
    };
    channel.in_flight.store(false, Ordering::Release);
    if revision != glyphs.request_revision || !glyphs.enabled {
        return;
    }
    if fields.len() != glyphs.positions.len() {
        return;
    }
    glyphs.fields = fields.into_iter().map(Vec4::from_array).collect();
}

fn render_gravity_field_gizmos_system(
    mut gizmos: Gizmos<crate::bevy_app::render::ScientificGizmos>,
    ryugu_query: Query<&Transform, (With<RyuguMarker>, Without<Camera3d>)>,
    camera_query: Query<&Transform, (With<Camera3d>, Without<RyuguMarker>)>,
    show_section: Res<ShowSection>,
    glyphs: Res<GravityFieldGlyphs>,
) {
    if !glyphs.enabled || glyphs.positions.is_empty() || glyphs.fields.is_empty() {
        return;
    }
    let Some(ryugu_tf) = ryugu_query.iter().next() else {
        return;
    };
    let view_body = camera_query.iter().next().and_then(|cam_tf| {
        let view_world = (cam_tf.translation - ryugu_tf.translation).normalize_or_zero();
        (view_world != Vec3::ZERO).then(|| ryugu_tf.rotation.inverse() * view_world)
    });
    let count = glyphs.positions.len().min(glyphs.fields.len());
    let max_magnitude = glyphs.fields[..count]
        .iter()
        .map(|field| field.w.max(0.0))
        .fold(0.0_f32, f32::max)
        .max(1.0e-12);
    let min_magnitude = glyphs.fields[..count]
        .iter()
        .map(|field| field.w)
        .filter(|value| *value > 0.0)
        .fold(f32::INFINITY, f32::min)
        .min(max_magnitude);
    let log_min = min_magnitude.ln();
    let log_span = (max_magnitude.ln() - log_min).max(1.0e-6);
    let surface = glyphs.surface_radius.max(SECTION_CLIP_RADIUS);
    // Body clearance for shaft endpoints (surface_radius already includes padding).
    let body_clearance = (surface / SURFACE_PADDING).max(SECTION_CLIP_RADIUS) * 1.08;
    let max_length_factor = if show_section.0 { 0.15 } else { 0.32 };

    for index in 0..count {
        let field = glyphs.fields[index];
        let acceleration = field.truncate();
        if !acceleration.is_finite() || field.w <= 0.0 {
            continue;
        }
        let direction = acceleration.normalize_or_zero();
        if direction == Vec3::ZERO {
            continue;
        }
        let body_pos = glyphs.positions[index];
        if body_pos.length() < body_clearance {
            continue;
        }
        // Section view: drop far-hemisphere samples so translucent mesh + cut
        // do not stack opposing arrows into visual chaos.
        if show_section.0
            && let Some(view_body) = view_body
            && body_pos.dot(view_body) < SECTION_HEMISPHERE_EPS
        {
            continue;
        }
        // Body-frame samples + accelerations; rotate both with live attitude.
        let world_direction = ryugu_tf.rotation * direction;
        let tip = ryugu_tf.translation + ryugu_tf.rotation * body_pos;
        let sample_radius = body_pos.length().max(surface);
        // Shaft stays outside the body: head at the exterior sample, tail further out.
        let strength = ((field.w.ln() - log_min) / log_span).clamp(0.0, 1.0);
        let outward_budget = (sample_radius - body_clearance).max(surface * 0.12);
        let max_length = outward_budget.min(sample_radius * max_length_factor);
        let length = (0.35 + 0.65 * strength) * max_length;
        let start = tip - world_direction * length;
        let start_body = ryugu_tf.rotation.inverse() * (start - ryugu_tf.translation);
        if start_body.length() < body_clearance {
            continue;
        }
        gizmos.arrow(start, tip, gravity_strength_color(strength));
    }
}

fn extract_gravity_field_system(
    glyphs: Extract<Res<GravityFieldGlyphs>>,
    channel: Extract<Res<GravityFieldChannel>>,
    mut extracted: ResMut<ExtractedGravityField>,
    mut gpu: ResMut<GravityFieldGpu>,
    mut state: ResMut<GravityFieldDispatchState>,
) {
    if !glyphs.enabled
        || glyphs.eval_mode == EVAL_MODE_WORKER
        || glyphs.positions.is_empty()
        || glyphs.source_bytes.is_empty()
        || glyphs.source_count == 0
    {
        return;
    }

    // Recover from a stalled map_async so a later revision can dispatch again.
    if let Some(inner) = gpu.0.as_ref()
        && channel.in_flight.load(Ordering::Acquire)
        && inner.submitted_at.elapsed() > GPU_STALL_TIMEOUT
    {
        channel.in_flight.store(false, Ordering::Release);
        gpu.0 = None;
        *state = GravityFieldDispatchState::Idle;
    }

    if channel.in_flight.load(Ordering::Acquire) {
        return;
    }
    if gpu
        .0
        .as_ref()
        .is_some_and(|inner| inner.revision == glyphs.request_revision)
    {
        return;
    }
    if *state == GravityFieldDispatchState::Mapped || gpu.0.is_some() {
        gpu.0 = None;
        *state = GravityFieldDispatchState::Idle;
    }
    if extracted.0.is_some() {
        return;
    }
    let position_bytes: Vec<u8> = glyphs
        .positions
        .iter()
        .flat_map(|position| [position.x, position.y, position.z, 0.0_f32])
        .flat_map(f32::to_le_bytes)
        .collect();
    extracted.0 = Some(ExtractedGravityFieldInner {
        revision: glyphs.request_revision,
        position_bytes,
        source_bytes: glyphs.source_bytes.clone(),
        sample_count: glyphs.positions.len() as u32,
        source_count: glyphs.source_count,
        eval_mode: glyphs.eval_mode,
        center_gm: glyphs.center_gm,
    });
}

fn dispatch_gravity_field_system(
    mut state: ResMut<GravityFieldDispatchState>,
    mut buffers: ResMut<GravityFieldGpu>,
    pipeline_res: Option<Res<GravityFieldPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut extracted: ResMut<ExtractedGravityField>,
    channel: Res<GravityFieldChannel>,
) {
    if *state == GravityFieldDispatchState::Mapped || buffers.0.is_some() {
        return;
    }
    let Some(pl) = pipeline_res else {
        return;
    };
    let Some(payload) = extracted.0.take() else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pl.pipeline_id) else {
        extracted.0 = Some(payload);
        return;
    };
    if channel.in_flight.load(Ordering::Acquire) {
        extracted.0 = Some(payload);
        return;
    }
    if payload.source_bytes.is_empty() || payload.source_count == 0 {
        return;
    }

    let out_size = payload.sample_count as u64 * 16;
    let uniform = GravityFieldUniform {
        counts: Vec4::new(
            payload.sample_count as f32,
            payload.source_count as f32,
            payload.eval_mode as f32,
            0.0,
        ),
        center_gm: payload.center_gm,
    };
    let mut uniform_bytes = [0_u8; 32];
    uniform_bytes[..16].copy_from_slice(bytemuck_vec4(uniform.counts).as_slice());
    uniform_bytes[16..32].copy_from_slice(bytemuck_vec4(uniform.center_gm).as_slice());

    let uniform_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gravity_field_uniform"),
        contents: &uniform_bytes,
        usage: BufferUsages::UNIFORM,
    });
    let positions = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gravity_field_positions"),
        contents: &payload.position_bytes,
        usage: BufferUsages::STORAGE,
    });
    let sources = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gravity_field_sources"),
        contents: &payload.source_bytes,
        usage: BufferUsages::STORAGE,
    });
    let fields = render_device.create_buffer(&BufferDescriptor {
        label: Some("gravity_field_fields"),
        size: out_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("gravity_field_staging"),
        size: out_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = render_device.create_bind_group_layout(
        "gravity_field_bgl_rt",
        &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ],
    );
    let bind_group = render_device.create_bind_group(
        "gravity_field_bg",
        &bgl,
        &[
            BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: positions.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: sources.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: fields.as_entire_binding(),
            },
        ],
    );

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("gravity_field_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("gravity_field_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(payload.sample_count.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&fields, 0, &staging, 0, out_size);
    render_queue.submit([encoder.finish()]);

    channel.in_flight.store(true, Ordering::Release);
    let shared = Arc::clone(&channel.result);
    let in_flight = Arc::clone(&channel.in_flight);
    let staging_ref = staging.clone();
    let revision = payload.revision;
    staging.slice(..).map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            let view = staging_ref.slice(..).get_mapped_range();
            let mut values = Vec::with_capacity(view.len() / 16);
            for chunk in view.as_chunks::<16>().0 {
                values.push([
                    f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                    f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                    f32::from_le_bytes(chunk[12..16].try_into().unwrap()),
                ]);
            }
            drop(view);
            staging_ref.unmap();
            if let Ok(mut lock) = shared.lock() {
                *lock = Some((revision, values));
            } else {
                in_flight.store(false, Ordering::Release);
            }
        } else {
            in_flight.store(false, Ordering::Release);
        }
    });

    buffers.0 = Some(GravityFieldGpuInner {
        _uniform: uniform_buf,
        _positions: positions,
        _sources: sources,
        _fields: fields,
        _staging: staging,
        revision: payload.revision,
        submitted_at: Instant::now(),
    });
    *state = GravityFieldDispatchState::Mapped;
}

fn pack_point_masses(sources: &[FrequencyDomainPointSource]) -> (Vec<u8>, u32) {
    let selected = subsample_point_masses(sources, MAX_NBODY_SOURCES);
    let bytes = selected
        .iter()
        .flat_map(|source| {
            [
                source.position.x as f32,
                source.position.y as f32,
                source.position.z as f32,
                source.mass as f32,
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    (bytes, selected.len() as u32)
}

fn pack_inverted_voxels(result: &DensityInversionResult) -> Option<(Vec<u8>, u32)> {
    if result.voxels.is_empty() {
        return None;
    }
    // Use each voxel's mass-preserving volume (not the full cube voxel_size³),
    // matching the Eq.(184) / FMM basis columns that produced ρ̂.
    let mut sources: Vec<FrequencyDomainPointSource> = result
        .voxels
        .iter()
        .filter_map(|voxel| {
            let density = f64::from(voxel.density);
            let volume = f64::from(voxel.volume);
            if !density.is_finite() || density <= 0.0 || !volume.is_finite() || volume <= 0.0 {
                return None;
            }
            let mass = density * volume;
            if !mass.is_finite() || mass <= 0.0 {
                return None;
            }
            Some(FrequencyDomainPointSource {
                position: DVec3::new(
                    f64::from(voxel.center.x),
                    f64::from(voxel.center.y),
                    f64::from(voxel.center.z),
                ),
                mass,
            })
        })
        .collect();
    if sources.is_empty() {
        return None;
    }
    sources = subsample_point_masses(&sources, MAX_NBODY_SOURCES);
    let (bytes, count) = pack_point_masses(&sources);
    Some((bytes, count))
}

fn subsample_point_masses(
    sources: &[FrequencyDomainPointSource],
    limit: usize,
) -> Vec<FrequencyDomainPointSource> {
    if sources.len() <= limit {
        return sources.to_vec();
    }
    let total_mass: f64 = sources.iter().map(|source| source.mass).sum();
    let mut ranked: Vec<(usize, f64)> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (index, source.mass))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(limit);
    ranked.sort_by_key(|(index, _)| *index);
    let mut selected: Vec<FrequencyDomainPointSource> = ranked
        .into_iter()
        .map(|(index, _)| sources[index])
        .collect();
    // Keep |g| ≈ GM/r²: dropping light cells without renormalization under-reads
    // the field by ~50–100× vs Eq.(121) and makes Radial/FMM/FFT disagree with FD.
    let selected_mass: f64 = selected.iter().map(|source| source.mass).sum();
    if total_mass.is_finite()
        && selected_mass.is_finite()
        && total_mass > 0.0
        && selected_mass > 0.0
    {
        let scale = total_mass / selected_mass;
        for source in &mut selected {
            source.mass *= scale;
        }
    }
    selected
}

fn evaluate_payload_cpu(
    source_count: u32,
    source_bytes: &[u8],
    positions: &[Vec3],
) -> Option<Vec<Vec4>> {
    let sources = unpack_point_masses(source_count, source_bytes)?;
    let mut fields = Vec::with_capacity(positions.len());
    for position in positions {
        let mut gravity = Vec3::ZERO;
        for source in &sources {
            let offset = *position - source.0;
            let distance_squared = offset.length_squared().max(1.0e-4);
            let inv_distance = distance_squared.sqrt().recip();
            let inv3 = inv_distance * inv_distance * inv_distance;
            gravity += -G * source.1 * offset * inv3;
        }
        fields.push(gravity.extend(gravity.length()));
    }
    Some(fields)
}

fn unpack_point_masses(source_count: u32, source_bytes: &[u8]) -> Option<Vec<(Vec3, f32)>> {
    let expected = source_count as usize * 16;
    if source_bytes.len() < expected || source_count == 0 {
        return None;
    }
    let mut sources = Vec::with_capacity(source_count as usize);
    for chunk in source_bytes[..expected].as_chunks::<16>().0 {
        sources.push((
            Vec3::new(
                f32::from_le_bytes(chunk[0..4].try_into().ok()?),
                f32::from_le_bytes(chunk[4..8].try_into().ok()?),
                f32::from_le_bytes(chunk[8..12].try_into().ok()?),
            ),
            f32::from_le_bytes(chunk[12..16].try_into().ok()?),
        ));
    }
    Some(sources)
}

fn bytemuck_vec4(value: Vec4) -> [u8; 16] {
    let mut out = [0_u8; 16];
    out[0..4].copy_from_slice(&value.x.to_le_bytes());
    out[4..8].copy_from_slice(&value.y.to_le_bytes());
    out[8..12].copy_from_slice(&value.z.to_le_bytes());
    out[12..16].copy_from_slice(&value.w.to_le_bytes());
    out
}

fn exterior_surface_radius(
    aggregated: Option<&AggregatedGravitySource>,
    quadrature: Option<&DensityQuadratureSource>,
) -> f32 {
    let model_radius = aggregated
        .map(|source| source.radius as f32)
        .or_else(|| quadrature.map(|source| source.radius))
        .unwrap_or(SECTION_CLIP_RADIUS)
        .max(SECTION_CLIP_RADIUS);
    model_radius * SURFACE_PADDING
}

fn build_shell_sample_positions(surface_radius: f32) -> Vec<Vec3> {
    let surface = surface_radius.max(SECTION_CLIP_RADIUS * SURFACE_PADDING);
    let outer = surface * OUTER_RADIUS_FACTOR;
    let mut positions = Vec::with_capacity(GRAVITY_ARROW_COUNT);
    for shell in 0..GRAVITY_ARROW_SHELLS {
        let t = (shell as f32 + 0.5) / GRAVITY_ARROW_SHELLS as f32;
        let radius = surface * (outer / surface).powf(t);
        for index in 0..GRAVITY_ARROW_PER_SHELL {
            let (x, y, z) = fibonacci_sphere(index, GRAVITY_ARROW_PER_SHELL);
            positions.push(Vec3::new(x, y, z) * radius);
        }
    }
    positions
}

fn fibonacci_sphere(index: u32, count: u32) -> (f32, f32, f32) {
    let offset = 2.0 / count as f32;
    let y = index as f32 * offset - 1.0 + offset * 0.5;
    let radius = (1.0 - y * y).max(0.0).sqrt();
    let phi = index as f32 * std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    (phi.cos() * radius, y, phi.sin() * radius)
}

fn gravity_strength_color(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let cold = Vec3::new(0.15, 0.45, 1.0);
    let mid = Vec3::new(0.2, 0.95, 0.35);
    let hot = Vec3::new(1.0, 0.18, 0.05);
    let rgb = if t < 0.5 {
        cold.lerp(mid, t * 2.0)
    } else {
        mid.lerp(hot, (t - 0.5) * 2.0)
    };
    Color::srgb(rgb.x, rgb.y, rgb.z)
}
