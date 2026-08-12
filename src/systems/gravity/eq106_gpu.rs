//! GPU-resident Equation (106) engineering evaluator.
//!
//! Expensive half-line kernel assembly is performed only when the reference
//! line changes. Real-time frames reuse the assembled 129-frequency buffer and
//! dispatch a Bromwich/high-order Taylor translation pass each render frame.
//! The main WASM thread only polls the asynchronous readback.
//!
//! The pass is deliberately documented as an engineering approximation to the
//! derivation in `docs/mathtidy.md`: it uses fixed quadrature over the
//! mass-preserving source representation, an independently transformed density
//! Fourier representation, and the complete planner-selected Eq. (118)
//! directional Taylor jet. Runtime guards forbid leaving its convergence disk.

use crate::components::*;
use crate::systems::curved_arc::{
    CurvedArcPlannerState, CurvedArcResidualHistory, Eq106SourceData,
};
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

const HALF_COUNT: u32 = 64;
const FREQUENCY_COUNT: u32 = 2 * HALF_COUNT + 1;
const QUADRATURE_COUNT: u32 = 64;
const MAX_BATCH_COUNT: u32 = MAX_SIMULATION_ACCELERATION + 1;
// Spectrum assembly remains segment-scoped. Evaluate often enough that the
// CPU leapfrog stays inside the certified local Hessian neighborhood without
// forcing a synchronous GPU readback on every rendered frame.
const EVALUATION_CADENCE_FRAMES: u32 = 1;
const OUTPUT_ROWS_PER_BLOCK: u64 = 9;
const SPECTRUM_BYTES: u64 = FREQUENCY_COUNT as u64 * 32;
const OUTPUT_BYTES: u64 = MAX_BATCH_COUNT as u64 * OUTPUT_ROWS_PER_BLOCK * 16;

#[derive(Resource, Default)]
struct ExtractedEq106Input {
    enabled: bool,
    probe: Vec3,
    velocity: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    sources: Option<Vec<u8>>,
    fourier_modes: Option<Vec<u8>>,
    operator_tensor: Option<Vec<u8>>,
    source_count: u32,
    density_mode_count: u32,
    radius: f32,
    source_hash: u64,
    batch_count: u32,
    block_dt: f32,
    certified_line_limit: f32,
    taylor_order: u32,
}

#[derive(Resource, Default)]
struct Eq106GpuBuffers(Option<Eq106GpuBuffersInner>);

/// Additive potential gauge shared by consecutive local Eq.106 spectral
/// elements. The GPU correction is tapered to zero before a reference-line
/// transition, so this is only a defensive C0 alignment for f32 roundoff. It
/// must never add a force bias: doing that recursively changes the physical
/// field every time an accelerated batch crosses a segment boundary.
#[derive(Resource, Default)]
struct Eq106PotentialGauge {
    epoch: Option<u64>,
    line_origin: Option<Vec3>,
    offset: f32,
    anchor_potential: Option<f32>,
    anchor_curve_work: Option<f64>,
}

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
    line_samples_id: CachedComputePipelineId,
    assemble_id: CachedComputePipelineId,
    evaluate_id: CachedComputePipelineId,
}

pub struct Eq106GpuComputePlugin;

impl Plugin for Eq106GpuComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Eq106GpuReadbackChannel>();
        app.init_resource::<Eq106GpuHistory>();
        app.init_resource::<Eq106PotentialGauge>();
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
            storage_rw_entry(6),
            storage_ro_entry(7),
        ];
        let layout = BindGroupLayoutDescriptor::new("eq106_complex_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/eq106_complex.wgsl");
        let cache = world.resource::<PipelineCache>();
        let line_samples_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("eq106_assemble_line_samples".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("assemble_line_samples".into()),
            zero_initialize_workgroup_memory: false,
        });
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
            line_samples_id,
            assemble_id,
            evaluate_id,
        }
    }
}

fn clear_eq106_history_on_probe_reset(
    probe: Res<ProbeInitialConditions>,
    mut history: ResMut<Eq106GpuHistory>,
    mut gauge: ResMut<Eq106PotentialGauge>,
) {
    if probe.is_changed() {
        history.0.clear();
        *gauge = Eq106PotentialGauge::default();
    }
}

fn poll_eq106_readback(
    channel: Res<Eq106GpuReadbackChannel>,
    mut history: ResMut<Eq106GpuHistory>,
    mut gauge: ResMut<Eq106PotentialGauge>,
    curved_residual: Res<CurvedArcResidualHistory>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    let Some(packet) = guard.take() else { return };
    let Some(first_anchor) = packet.partial_sums.get(7).copied() else {
        return;
    };
    let batch_count = (first_anchor[3].round() as usize).clamp(1, MAX_BATCH_COUNT as usize);
    let base_snapshot = packet.snapshot;
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);

    for block_index in 0..batch_count {
        let output_base = block_index * OUTPUT_ROWS_PER_BLOCK as usize;
        let Some(rows) = packet
            .partial_sums
            .get(output_base..output_base + OUTPUT_ROWS_PER_BLOCK as usize)
        else {
            runtime_error.raise("Equation (106) GPU readback omitted a batch spectral element.");
            return;
        };
        let field = rows[0];
        let certificate = rows[1];
        let potentials = rows[6];
        let anchor_row = rows[7];
        let origin_row = rows[8];
        if certificate[0] > 0.25
            || certificate[1] > 0.05
            || (certificate[3] > 0.99 && certificate[2] > 0.25)
        {
            runtime_error.raise(format!(
                "Equation (106) GPU batch element {} certification failed (field={:.3e}, imaginary={:.3e}, toroidal={:.3e}, coverage={:.3e}).",
                block_index + 1,
                certificate[0], certificate[1], certificate[2], certificate[3]
            ));
            return;
        }
        let acceleration = Vec3::new(field[0], field[1], field[2]);
        let segmented_potential = potentials[0];
        let independent_potential = potentials[1];
        let elapsed = potentials[3] as f64;
        let anchor_position = Vec3::new(anchor_row[0], anchor_row[1], anchor_row[2]);
        let line_origin = Vec3::new(origin_row[0], origin_row[1], origin_row[2]);
        let predicted_body_velocity = Vec3::new(rows[5][0], rows[5][1], rows[5][2]);
        let jacobian = Mat3::from_cols(
            Vec3::new(rows[2][0], rows[3][0], rows[4][0]),
            Vec3::new(rows[2][1], rows[3][1], rows[4][1]),
            Vec3::new(rows[2][2], rows[3][2], rows[4][2]),
        );
        if !acceleration.is_finite()
            || !segmented_potential.is_finite()
            || segmented_potential <= 0.0
            || !independent_potential.is_finite()
            || independent_potential <= 0.0
            || !anchor_position.is_finite()
            || !line_origin.is_finite()
            || !predicted_body_velocity.is_finite()
            || !jacobian.is_finite()
        {
            runtime_error.raise(
                "Equation (106) GPU returned a non-finite batched field sample or local potential Hessian.",
            );
            return;
        }

        let mut snapshot = base_snapshot.clone();
        if block_index > 0 {
            snapshot.request_id = snapshot
                .request_id
                .wrapping_mul(MAX_BATCH_COUNT as u64)
                .wrapping_add(block_index as u64);
        }
        snapshot.simulation_time_seconds += elapsed;
        snapshot.body_position = anchor_position;
        let future_rotation = Quat::from_axis_angle(
            RYUGU_SPIN_AXIS.normalize(),
            std::f32::consts::TAU * elapsed as f32 / RYUGU_ROTATION_PERIOD_SECS,
        ) * base_snapshot.ryugu_transform.rotation;
        snapshot.ryugu_transform.rotation = future_rotation;
        snapshot.probe_position =
            snapshot.ryugu_transform.translation + future_rotation * anchor_position;
        let angular_velocity_body = future_rotation.inverse() * angular_velocity_world;
        snapshot.probe_velocity = future_rotation
            * (predicted_body_velocity + angular_velocity_body.cross(anchor_position));

        if gauge.epoch != Some(base_snapshot.epoch) {
            *gauge = Eq106PotentialGauge {
                epoch: Some(snapshot.epoch),
                line_origin: Some(line_origin),
                offset: 0.0,
                anchor_potential: None,
                anchor_curve_work: None,
            };
        } else if gauge
            .line_origin
            .is_none_or(|previous| previous.distance_squared(line_origin) > 1.0e-6)
        {
            // Continue the scalar potential from the last authoritative anchor
            // using the same curved-path work used by Eq.(157). The endpoint
            // anchor makes this a first-order accurate overlap continuation;
            // no acceleration bias is introduced.
            if let Some(current_work) =
                curved_residual.curve_work_at(snapshot.simulation_time_seconds)
            {
                gauge.offset = gauge
                    .anchor_potential
                    .zip(gauge.anchor_curve_work)
                    .map(|(anchor_potential, anchor_work)| {
                        (anchor_potential as f64 + current_work - anchor_work) as f32
                            - segmented_potential
                    })
                    .filter(|offset| offset.is_finite())
                    .unwrap_or(0.0);
            }
            gauge.line_origin = Some(line_origin);
        }

        let positive_potential = segmented_potential + gauge.offset;
        if !positive_potential.is_finite() || positive_potential <= 0.0 {
            runtime_error
                .raise("Equation (106) potential gauge alignment produced an invalid value.");
            return;
        }
        if block_index == 0
            && let Some(curve_work) =
                curved_residual.curve_work_at(snapshot.simulation_time_seconds)
        {
            gauge.anchor_potential = Some(positive_potential);
            gauge.anchor_curve_work = Some(curve_work);
        }
        history.0.push(GravityFieldSample {
            snapshot,
            predictive: block_index > 0,
            body_acceleration: acceleration,
            // This segmented potential is generated by the same Eq.106 field
            // that drives integration. The independent direct potential below
            // is reserved for the Eq. (157) dual-representation residual.
            positive_potential,
            // Eq. (157) must remain an actual dual-representation check.  The
            // shader evaluates this direct full-space point potential
            // independently of the segmented spectral potential above.
            independent_positive_potential: Some(independent_potential),
            body_acceleration_jacobian: Some(jacobian),
        });
    }
}

fn extract_eq106_input(
    mut extracted: ResMut<ExtractedEq106Input>,
    source: Extract<Option<Res<Eq106SourceData>>>,
    operator_tensor: Extract<Option<Res<Eq106OperatorTensorResource>>>,
    active: Extract<Res<ActiveGravityMethod>>,
    clock: Extract<Res<SimulationClock>>,
    fixed_time: Extract<Res<Time<Fixed>>>,
    simulation_acceleration: Extract<Res<SimulationAcceleration>>,
    planner: Extract<Res<CurvedArcPlannerState>>,
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
    let relative_world_position = probe.translation - ryugu.translation;
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    extracted.probe = ryugu.rotation.inverse() * relative_world_position;
    // Eq.106's reference line lives in the body-fixed density coordinates, so
    // its tangent must be dq_body/dt, not the inertial velocity merely rotated
    // into body axes. Omitting omega x r tilts the spectral line by a large
    // fraction of the orbital velocity and causes secular trajectory error.
    extracted.velocity = ryugu.rotation.inverse()
        * (velocity.0 - angular_velocity_world.cross(relative_world_position));
    extracted.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
        body_position: extracted.probe,
        ryugu_transform: *ryugu,
        probe_position: probe.translation,
        probe_velocity: velocity.0,
    });
    // Include the endpoint of the final stable interval. Physics blends the
    // two surrounding local Hessian models at every substep, eliminating the
    // one-sided field extrapolation that produced Jacobi steps.
    extracted.batch_count = simulation_acceleration.stable_steps() + 1;
    extracted.block_dt = fixed_time.delta_secs() * TIME_SCALE;
    extracted.source_count = source.sources.len() as u32;
    extracted.density_mode_count = source.fourier_modes.len() as u32;
    extracted.radius = source.radius as f32;
    extracted.certified_line_limit = planner
        .active_segment
        .as_ref()
        .filter(|segment| segment.taylor_order.is_some() && segment.epsilon_max < 1.0)
        .map(|segment| {
            let curvature_limit = if segment.maximum_curvature > f64::MIN_POSITIVE {
                (8.0 * 0.25 * segment.distance_lower_bound / segment.maximum_curvature).sqrt()
            } else {
                f64::INFINITY
            };
            segment.arc_length.max(1.0).min(curvature_limit) as f32
        })
        .unwrap_or(f32::INFINITY);
    extracted.taylor_order = planner.taylor_order.clamp(1, 8);
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
        let mut mode_bytes = Vec::with_capacity(source.fourier_modes.len() * 16);
        for record in &source.fourier_modes {
            for value in record {
                mode_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        extracted.fourier_modes = Some(mode_bytes);
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
    let (Some(line_samples), Some(assemble), Some(evaluate)) = (
        cache.get_compute_pipeline(pipelines.line_samples_id),
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
        let (Some(source_bytes), Some(operator_bytes), Some(mode_bytes)) = (
            extracted.sources.as_ref(),
            extracted.operator_tensor.as_ref(),
            extracted.fourier_modes.as_ref(),
        ) else {
            return;
        };
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_uniform"),
            // WGSL's final vec3<u32> member is rounded to the uniform
            // structure's 16-byte alignment. Keep the allocation at the
            // validated 112-byte size even though the populated fields end
            // at offset 96.
            size: 112,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sources = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_sources"),
            contents: source_bytes,
            usage: BufferUsages::STORAGE,
        });
        let quadrature_bytes = half_line_quadrature_bytes(0.5 * extracted.radius.max(1.0));
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
        let density_modes = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_density_fourier_modes"),
            contents: mode_bytes,
            usage: BufferUsages::STORAGE,
        });
        let spectrum = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_spectrum"),
            size: SPECTRUM_BYTES,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let line_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_line_samples"),
            size: QUADRATURE_COUNT as u64 * 16,
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
                storage_rw_entry(6),
                storage_ro_entry(7),
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
                BindGroupEntry {
                    binding: 6,
                    resource: line_samples.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: density_modes.as_entire_binding(),
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
    // Eq. (106) is a straight-reference operator; the curved trajectory must
    // be covered by local spectral elements. The frequency-grid Nyquist range
    // is not a convergence radius, so do not reuse one line for kilometres.
    let predicted_batch_travel = extracted.velocity.length()
        * extracted.block_dt
        * extracted.batch_count.saturating_sub(1) as f32;
    // At 8x the probe advances roughly one old 0.15R segment per presented
    // frame. Size the cached line for two accelerated batches so spectrum
    // assembly is not repeated every frame, while keeping the operator local.
    let longitudinal_limit = (0.35 * extracted.radius)
        .max(2.25 * predicted_batch_travel)
        // Keep at least ~18 authoritative 1x frames in one element. This
        // gives the previous sample time to enter the zero-correction overlap
        // before the next reference line is installed, avoiding a visible
        // potential step without extending the spectral work per frame.
        .max(160.0)
        // The curved-arc planner supplies the docs/mathtidy.md curvature
        // bound.  Unlike the throughput heuristic above, this is a hard cap:
        // spectral correction is tapered to zero before leaving this disk.
        .min(extracted.certified_line_limit)
        .max(1.0);
    let transverse_limit = (0.10 * extracted.radius).max(20.0);
    let line_expired = inner.source_hash != extracted.source_hash
        || h < 0.0
        || h > 0.85 * longitudinal_limit
        || transverse > transverse_limit;
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
    if !inner.render_frame.is_multiple_of(EVALUATION_CADENCE_FRAMES)
        || inner.last_submitted == Some(key)
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
        extracted.velocity,
        extracted.block_dt,
        extracted.batch_count,
        longitudinal_limit,
        extracted.taylor_order,
        extracted.density_mode_count,
    );
    render_queue.write_buffer(&inner.uniform, 0, &uniform);
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("eq106_complex_encoder"),
    });
    if !inner.spectrum_ready {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_line_samples_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(line_samples);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(QUADRATURE_COUNT.div_ceil(64), 1, 1);
        drop(pass);
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
        pass.dispatch_workgroups(extracted.batch_count.clamp(1, MAX_BATCH_COUNT), 1, 1);
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

fn half_line_quadrature_bytes(length_scale: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(QUADRATURE_COUNT as usize * 8);
    let du = 1.0 / QUADRATURE_COUNT as f32;
    let length_scale = length_scale.max(1.0);
    for index in 0..QUADRATURE_COUNT {
        let u = (index as f32 + 0.5) * du;
        let denominator = 1.0 - u;
        // h = L u/(1-u), dh/du = L/(1-u)^2. Choosing L=1/sigma
        // resolves the physical Laplace-decay length instead of concentrating
        // nearly every node inside the first few metres.
        let h = length_scale * u / denominator;
        let weight = length_scale * du / (denominator * denominator);
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
    body_velocity: Vec3,
    block_dt: f32,
    batch_count: u32,
    longitudinal_limit: f32,
    taylor_order: u32,
    density_mode_count: u32,
) -> [u8; 112] {
    let mut bytes = [0_u8; 112];
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
        (60, taylor_order.clamp(1, 8)),
        (80, batch_count.clamp(1, MAX_BATCH_COUNT)),
        (84, density_mode_count),
        (88, 0),
        (92, 0),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (64, body_velocity.x),
        (68, body_velocity.y),
        (72, body_velocity.z),
        (76, block_dt.max(1.0e-3)),
        (96, longitudinal_limit.max(1.0)),
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
