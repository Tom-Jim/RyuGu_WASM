use crate::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::Arc;
use std::sync::atomic::Ordering;

const WORKGROUP_SIZE: u32 = 64;
const RADIAL_LAYER_COUNT: u32 = 4;

#[derive(Resource, Default)]
struct ExtractedGravityInput {
    probe: Vec3,
    layer_bytes: Option<Vec<u8>>,
    layer_count: u32,
}

#[derive(Resource, Default)]
struct GravityGpuBuffers(Option<GravityGpuBuffersInner>);

struct GravityGpuBuffersInner {
    uniform: bevy::render::render_resource::Buffer,
    output: bevy::render::render_resource::Buffer,
    staging: bevy::render::render_resource::Buffer,
    bind_group: BindGroup,
    layer_count: u32,
    workgroup_count: u32,
    output_size: u64,
}

#[derive(Resource)]
struct GravityComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

pub struct GravityComputePlugin;

impl Plugin for GravityComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GravityReadbackChannel>();
        app.add_systems(
            Update,
            (build_radial_gravity_source_system, poll_gravity_readback).chain(),
        );

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedGravityInput>();
        render_app.init_resource::<GravityGpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_gravity_input_system);
        render_app.add_systems(
            Render,
            dispatch_gravity_system.in_set(RenderSystems::Render),
        );
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<GravityReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<GravityComputePipeline>();
    }
}

impl FromWorld for GravityComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)];
        let layout = BindGroupLayoutDescriptor::new("radial_gravity_bgl", &entries);
        let shader = world.resource::<AssetServer>().load("shaders/gravity.wgsl");
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("radial_gravity_compute".into()),
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

/// Builds the angular-cell/radial-layer discretization used by equation (18).
/// The mesh is assumed star-shaped with respect to its model origin, which is
/// true for the Ryugu asset. A non-star-shaped mesh would require multiple
/// radial intervals per angular cell.
pub fn build_radial_gravity_source_system(
    mut commands: Commands,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    ryugu: Query<&Transform, With<RyuguMarker>>,
    existing: Option<Res<RadialGravitySource>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(topology) = topology else { return };
    let Ok(transform) = ryugu.single() else {
        return;
    };
    if topology.triangles.is_empty() {
        return;
    }

    let scale = transform.scale.x;
    let mut cells = Vec::with_capacity(topology.triangles.len() / 3);
    let mut density_integral = 0.0_f64;
    let mut solid_angle_sum = 0.0_f64;

    for tri in topology.triangles.chunks_exact(3) {
        let p0 = topology.positions[tri[0] as usize] * scale;
        let p1 = topology.positions[tri[1] as usize] * scale;
        let p2 = topology.positions[tri[2] as usize] * scale;
        let Some((direction, radius, solid_angle)) = angular_cell(p0, p1, p2) else {
            continue;
        };
        cells.push((direction, radius, solid_angle));
        solid_angle_sum += solid_angle as f64;
        density_integral += solid_angle as f64
            * radial_density_integral(0.0, radius as f64, DENSITY_EPSILON as f64);
    }

    if cells.is_empty() || density_integral <= f64::EPSILON {
        error!("[gravity] failed to build equation (18) angular cells");
        return;
    }

    let density_c = (RYUGU_MASS as f64 / density_integral) as f32;
    let mut bytes = Vec::with_capacity(cells.len() * RADIAL_LAYER_COUNT as usize * 32);

    for (direction, radius, solid_angle) in cells {
        for layer in 0..RADIAL_LAYER_COUNT {
            // Equal-volume shells avoid over-resolving the small central region.
            let inner_fraction = (layer as f32 / RADIAL_LAYER_COUNT as f32).cbrt();
            let outer_fraction = ((layer + 1) as f32 / RADIAL_LAYER_COUNT as f32).cbrt();
            let r_inner = radius * inner_fraction;
            let r_outer = radius * outer_fraction;
            let shell_measure = (r_outer.powi(3) - r_inner.powi(3)) / 3.0;
            let shell_integral =
                radial_density_integral(r_inner as f64, r_outer as f64, DENSITY_EPSILON as f64)
                    as f32;
            let density = density_c * shell_integral / shell_measure.max(f32::MIN_POSITIVE);

            push_f32s(
                &mut bytes,
                [direction.x, direction.y, direction.z, solid_angle],
            );
            push_f32s(&mut bytes, [r_inner, r_outer, density, 0.0]);
        }
    }

    let count = (bytes.len() / 32) as u32;
    info!(
        "[gravity] equation (18): {} angular cells, {} radial layers, solid-angle sum={:.6}, C={:.6e}",
        count / RADIAL_LAYER_COUNT,
        count,
        solid_angle_sum,
        density_c
    );
    if (solid_angle_sum - std::f64::consts::TAU * 2.0).abs() > 0.05 {
        warn!(
            "[gravity] mesh subtends {:.6} sr instead of 4π; check that it is closed and star-shaped",
            solid_angle_sum
        );
    }

    commands.insert_resource(DensityC(density_c));
    commands.insert_resource(RadialGravitySource { bytes, count });
}

fn angular_cell(p0: Vec3, p1: Vec3, p2: Vec3) -> Option<(Vec3, f32, f32)> {
    let n0 = p0.try_normalize()?;
    let n1 = p1.try_normalize()?;
    let n2 = p2.try_normalize()?;
    let direction = (n0 + n1 + n2).try_normalize()?;

    let numerator = n0.dot(n1.cross(n2)).abs();
    let denominator = 1.0 + n0.dot(n1) + n1.dot(n2) + n2.dot(n0);
    let solid_angle = 2.0 * numerator.atan2(denominator);
    if !solid_angle.is_finite() || solid_angle <= 0.0 {
        return None;
    }

    let face_normal = (p1 - p0).cross(p2 - p0);
    let divisor = face_normal.dot(direction);
    let plane_radius = face_normal.dot(p0) / divisor;
    let centroid_radius = ((p0 + p1 + p2) / 3.0).length();
    let radius = if plane_radius.is_finite() && plane_radius > 0.0 {
        plane_radius
    } else {
        centroid_radius
    };
    (radius > 0.0).then_some((direction, radius, solid_angle))
}

/// Integral of r²/(r+epsilon), evaluated in f64 to avoid cancellation near 0.
fn radial_density_integral(inner: f64, outer: f64, epsilon: f64) -> f64 {
    fn primitive(r: f64, epsilon: f64) -> f64 {
        0.5 * r * r - epsilon * r + epsilon * epsilon * (1.0 + r / epsilon).ln()
    }
    primitive(outer, epsilon) - primitive(inner, epsilon)
}

fn push_f32s(bytes: &mut Vec<u8>, values: [f32; 4]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn poll_gravity_readback(
    channel: Res<GravityReadbackChannel>,
    mut acceleration: ResMut<GravityAcceleration>,
) {
    let Ok(mut guard) = channel.data.try_lock() else {
        return;
    };
    let Some(partial_sums) = guard.take() else {
        return;
    };
    let total = partial_sums.iter().fold(Vec3::ZERO, |sum, value| {
        sum + Vec3::new(value[0], value[1], value[2])
    });
    if total.is_finite() {
        acceleration.0 = total;
    } else {
        warn!("[gravity] discarded non-finite equation (18) GPU result");
    }
}

fn extract_gravity_input_system(
    mut extracted: ResMut<ExtractedGravityInput>,
    source: Extract<Option<Res<RadialGravitySource>>>,
    cassini: Extract<Query<&Transform, With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    let (Some(source), Ok(cassini), Ok(ryugu)) =
        (source.as_ref(), cassini.single(), ryugu.single())
    else {
        return;
    };

    extracted.probe = ryugu.rotation.inverse() * (cassini.translation - ryugu.translation);
    extracted.layer_count = source.count;
    if extracted.layer_bytes.is_none() {
        extracted.layer_bytes = Some(source.bytes.clone());
    }
}

fn dispatch_gravity_system(
    mut buffers: ResMut<GravityGpuBuffers>,
    pipeline_resource: Option<Res<GravityComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedGravityInput>,
    channel: Res<GravityReadbackChannel>,
) {
    let Some(pipeline_resource) = pipeline_resource else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_resource.pipeline_id) else {
        return;
    };
    if extracted.layer_count == 0 {
        return;
    }

    if buffers.0.is_none() {
        let Some(layer_bytes) = extracted.layer_bytes.as_ref() else {
            return;
        };
        let workgroup_count = extracted.layer_count.div_ceil(WORKGROUP_SIZE);
        let output_size = workgroup_count as u64 * 16;
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("radial_gravity_uniform"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layers = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("radial_gravity_layers"),
            contents: layer_bytes,
            usage: BufferUsages::STORAGE,
        });
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("radial_gravity_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("radial_gravity_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "radial_gravity_bgl_runtime",
            &[uniform_entry(0), storage_ro_entry(1), storage_rw_entry(2)],
        );
        let bind_group = render_device.create_bind_group(
            "radial_gravity_bg",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layers.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(GravityGpuBuffersInner {
            uniform,
            output,
            staging,
            bind_group,
            layer_count: extracted.layer_count,
            workgroup_count,
            output_size,
        });
    }

    let inner = buffers.0.as_ref().expect("gravity buffers initialized");
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }

    let uniform_bytes = gravity_uniform_bytes(extracted.probe, inner.layer_count);
    render_queue.write_buffer(&inner.uniform, 0, &uniform_bytes);

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("radial_gravity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("radial_gravity_pass"),
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
    map_staging
        .slice(..)
        .map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let view = staging.slice(..).get_mapped_range();
                let partial = bytes_to_f32x4(&view);
                if let Ok(mut guard) = shared.lock() {
                    *guard = Some(partial);
                }
                drop(view);
                staging.unmap();
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn gravity_uniform_bytes(probe: Vec3, layer_count: u32) -> [u8; 32] {
    let mut bytes = [0_u8; 32];
    for (offset, value) in [(0, probe.x), (4, probe.y), (8, probe.z), (12, G)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[16..20].copy_from_slice(&layer_count.to_le_bytes());
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
    use bevy::math::DVec3;

    fn primitive_cpu(lambda: f64, probe: DVec3, source_direction: DVec3) -> DVec3 {
        let radius = probe.length();
        let probe_direction = probe / radius;
        let cosine = probe_direction.dot(source_direction);
        let a = radius * cosine;
        let u = lambda - a;
        let b2 = radius * radius * (1.0 - cosine * cosine);
        let distance = (u * u + b2).sqrt();
        let hyperbolic = (u / b2.sqrt()).asinh();
        let j2 = hyperbolic - u / distance - 2.0 * a / distance + a * a * u / (b2 * distance);
        let j3 = distance + b2 / distance + 3.0 * a * (hyperbolic - u / distance)
            - 3.0 * a * a / distance
            + a * a * a * u / (b2 * distance);
        j3 * source_direction - radius * j2 * probe_direction
    }

    #[test]
    fn density_integral_matches_numerical_quadrature() {
        let analytic = radial_density_integral(3.0, 27.0, 10.0);
        let steps = 100_000;
        let h = 24.0 / steps as f64;
        let numerical = (0..steps)
            .map(|i| {
                let r = 3.0 + (i as f64 + 0.5) * h;
                r * r / (r + 10.0) * h
            })
            .sum::<f64>();
        assert!((analytic - numerical).abs() / analytic < 1.0e-9);
    }

    #[test]
    fn triangle_solid_angle_is_positive() {
        let cell = angular_cell(Vec3::X, Vec3::Y, Vec3::Z).unwrap();
        assert!((cell.2 - std::f32::consts::FRAC_PI_2).abs() < 1.0e-6);
    }

    #[test]
    fn gravity_uniform_matches_wgsl_layout() {
        let bytes = gravity_uniform_bytes(Vec3::new(1.0, 2.0, 3.0), 42);
        assert_eq!(f32::from_le_bytes(bytes[12..16].try_into().unwrap()), G);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 42);
    }

    #[test]
    fn equation_18_primitive_matches_direct_radial_integral() {
        let probe = DVec3::new(900.0, 130.0, 80.0);
        let source_direction = DVec3::new(0.2, 0.7, 0.6).normalize();
        let (inner, outer) = (30.0, 420.0);
        let analytic = primitive_cpu(outer, probe, source_direction)
            - primitive_cpu(inner, probe, source_direction);

        let steps = 200_000;
        let width = (outer - inner) / steps as f64;
        let numerical = (0..steps).fold(DVec3::ZERO, |sum, index| {
            let lambda = inner + (index as f64 + 0.5) * width;
            let displacement = lambda * source_direction - probe;
            sum + lambda * lambda * displacement / displacement.length().powi(3) * width
        });
        assert!((analytic - numerical).length() / numerical.length() < 1.0e-9);
    }
}
