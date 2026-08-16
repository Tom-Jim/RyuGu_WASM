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
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const WORKGROUP_SIZE: u32 = 64;

#[derive(Resource, Default)]
pub struct WernerAcceleration(pub Vec3);

#[derive(Resource, Default)]
pub struct WernerPotential(pub Option<f32>);

#[derive(Resource, Clone)]
pub struct WernerReadbackChannel {
    data: Arc<Mutex<Option<GravityReadbackPacket>>>,
    in_flight: Arc<AtomicBool>,
}

impl Default for WernerReadbackChannel {
    fn default() -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Correct closed-polyhedron Werner data. Each edge record is five vec4 values
/// (`p0`, `p1`, and three rows of E); each face is four vec4 values
/// (`p0`, `p1`, `p2`, outward normal).
#[derive(Resource)]
struct WernerSource {
    edge_bytes: Vec<u8>,
    face_bytes: Vec<u8>,
    edge_count: u32,
    face_count: u32,
    g_density: f32,
}

#[derive(Resource, Default)]
struct ExtractedWernerInput {
    probe: Vec3,
    snapshot: Option<GravityRequestSnapshot>,
    edge_bytes: Option<Vec<u8>>,
    face_bytes: Option<Vec<u8>>,
    edge_count: u32,
    face_count: u32,
    g_density: f32,
}

#[derive(Resource, Default)]
struct WernerGpuBuffers(Option<WernerGpuBuffersInner>);

struct WernerGpuBuffersInner {
    uniform: bevy::render::render_resource::Buffer,
    output: bevy::render::render_resource::Buffer,
    staging: bevy::render::render_resource::Buffer,
    bind_group: BindGroup,
    edge_count: u32,
    face_count: u32,
    item_count: u32,
    workgroup_count: u32,
    output_size: u64,
    last_submitted: Option<(u64, u64)>,
}

#[derive(Resource)]
struct WernerComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

pub struct WernerComputePlugin;

impl Plugin for WernerComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WernerReadbackChannel>();
        app.init_resource::<WernerAcceleration>();
        app.init_resource::<WernerPotential>();
        app.init_resource::<WernerGravityHistory>();
        app.add_systems(Update, build_werner_source_system);
        app.add_systems(PreUpdate, poll_werner_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedWernerInput>();
        render_app.init_resource::<WernerGpuBuffers>();
        render_app.add_systems(ExtractSchedule, extract_werner_input_system);
        render_app.add_systems(Render, dispatch_werner_system.in_set(RenderSystems::Render));
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<WernerReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<WernerComputePipeline>();
    }
}

impl FromWorld for WernerComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
        ];
        let layout = BindGroupLayoutDescriptor::new("werner_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/werner_gravity.wgsl");
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("werner_compute".into()),
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

#[derive(Clone, Copy)]
struct EdgeSide {
    face_normal: Vec3,
    edge_outward: Vec3,
}

fn build_werner_source_system(
    mut commands: Commands,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    ryugu: Query<&Transform, With<RyuguMarker>>,
    existing: Option<Res<WernerSource>>,
    active_method: Res<ActiveGravityMethod>,
) {
    if existing.is_some() || *active_method != ActiveGravityMethod::HomogeneousWerner {
        return;
    }
    let Some(topology) = topology else { return };
    let Ok(transform) = ryugu.single() else {
        return;
    };

    let scale = transform.scale.x;
    let mut edge_sides: HashMap<(u32, u32), Vec<EdgeSide>> = HashMap::new();
    let mut face_bytes = Vec::with_capacity(topology.triangles.len() / 3 * 64);
    let mut volume = 0.0_f64;
    let mut face_count = 0_u32;

    for triangle in topology.triangles.chunks_exact(3) {
        let mut indices = [triangle[0], triangle[1], triangle[2]];
        let mut points = [
            topology.positions[indices[0] as usize] * scale,
            topology.positions[indices[1] as usize] * scale,
            topology.positions[indices[2] as usize] * scale,
        ];
        let centroid = (points[0] + points[1] + points[2]) / 3.0;
        let raw_normal = (points[1] - points[0]).cross(points[2] - points[0]);
        if raw_normal.dot(centroid) < 0.0 {
            indices.swap(1, 2);
            points.swap(1, 2);
        }
        let Some(normal) = (points[1] - points[0])
            .cross(points[2] - points[0])
            .try_normalize()
        else {
            continue;
        };

        volume += points[0]
            .as_dvec3()
            .dot(points[1].as_dvec3().cross(points[2].as_dvec3()))
            / 6.0;
        for point in points {
            push_vec4(&mut face_bytes, point.extend(0.0));
        }
        push_vec4(&mut face_bytes, normal.extend(0.0));
        face_count += 1;

        for edge_index in 0..3 {
            let next = (edge_index + 1) % 3;
            let from_index = indices[edge_index];
            let to_index = indices[next];
            let edge_direction = (points[next] - points[edge_index]).normalize();
            let edge_outward = edge_direction.cross(normal);
            let key = if from_index < to_index {
                (from_index, to_index)
            } else {
                (to_index, from_index)
            };
            edge_sides.entry(key).or_default().push(EdgeSide {
                face_normal: normal,
                edge_outward,
            });
        }
    }

    if volume <= f64::EPSILON || face_count == 0 {
        error!("[werner] cannot build a positive-volume closed polyhedron");
        return;
    }

    let mut edge_bytes = Vec::with_capacity(edge_sides.len() * 80);
    let mut edge_count = 0_u32;
    let mut invalid_edges = 0_usize;
    for ((first, second), sides) in edge_sides {
        if sides.len() != 2 {
            invalid_edges += 1;
            continue;
        }
        let p0 = topology.positions[first as usize] * scale;
        let p1 = topology.positions[second as usize] * scale;
        let tensor = outer_product(sides[0].face_normal, sides[0].edge_outward)
            + outer_product(sides[1].face_normal, sides[1].edge_outward);
        let rows = tensor.transpose();
        push_vec4(&mut edge_bytes, p0.extend(0.0));
        push_vec4(&mut edge_bytes, p1.extend(0.0));
        push_vec4(&mut edge_bytes, rows.x_axis.extend(0.0));
        push_vec4(&mut edge_bytes, rows.y_axis.extend(0.0));
        push_vec4(&mut edge_bytes, rows.z_axis.extend(0.0));
        edge_count += 1;
    }

    if invalid_edges != 0 {
        warn!(
            "[werner] skipped {} boundary/non-manifold edges; source mesh should be watertight",
            invalid_edges
        );
    }
    let density = RYUGU_MASS / volume as f32;
    info!(
        "[werner] closed polyhedron: {} faces, {} shared edges, volume={:.6e} m³, rho={:.6e} kg/m³",
        face_count, edge_count, volume, density
    );
    commands.insert_resource(WernerSource {
        edge_bytes,
        face_bytes,
        edge_count,
        face_count,
        g_density: G * density,
    });
    commands.insert_resource(WernerDensity(density));
}

/// Returns the dyad `left * right^T`.
fn outer_product(left: Vec3, right: Vec3) -> Mat3 {
    Mat3::from_cols(left * right.x, left * right.y, left * right.z)
}

fn push_vec4(bytes: &mut Vec<u8>, value: Vec4) {
    for component in value.to_array() {
        bytes.extend_from_slice(&component.to_le_bytes());
    }
}

fn extract_werner_input_system(
    mut extracted: ResMut<ExtractedWernerInput>,
    source: Extract<Option<Res<WernerSource>>>,
    clock: Extract<Res<SimulationClock>>,
    cassini: Extract<Query<(&Transform, &Velocity), With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
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
    extracted.edge_count = source.edge_count;
    extracted.face_count = source.face_count;
    extracted.g_density = source.g_density;
    if extracted.edge_bytes.is_none() {
        extracted.edge_bytes = Some(source.edge_bytes.clone());
        extracted.face_bytes = Some(source.face_bytes.clone());
    }
}

fn dispatch_werner_system(
    mut buffers: ResMut<WernerGpuBuffers>,
    pipeline_resource: Option<Res<WernerComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedWernerInput>,
    channel: Res<WernerReadbackChannel>,
) {
    let Some(pipeline_resource) = pipeline_resource else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_resource.pipeline_id) else {
        return;
    };
    let item_count = extracted.edge_count.max(extracted.face_count);
    if item_count == 0 {
        return;
    }

    if buffers.0.is_none() {
        let (Some(edge_bytes), Some(face_bytes)) =
            (extracted.edge_bytes.as_ref(), extracted.face_bytes.as_ref())
        else {
            return;
        };
        let workgroup_count = item_count.div_ceil(WORKGROUP_SIZE);
        let output_size = workgroup_count as u64 * 16;
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("werner_uniform"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edges = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("werner_edges"),
            contents: edge_bytes,
            usage: BufferUsages::STORAGE,
        });
        let faces = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("werner_faces"),
            contents: face_bytes,
            usage: BufferUsages::STORAGE,
        });
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("werner_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("werner_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "werner_bgl_runtime",
            &[
                uniform_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
            ],
        );
        let bind_group = render_device.create_bind_group(
            "werner_bg",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: edges.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: faces.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(WernerGpuBuffersInner {
            uniform,
            output,
            staging,
            bind_group,
            edge_count: extracted.edge_count,
            face_count: extracted.face_count,
            item_count,
            workgroup_count,
            output_size,
            last_submitted: None,
        });
    }

    let inner = buffers.0.as_mut().expect("Werner buffers initialized");
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        return;
    };
    let submission_key = (snapshot.epoch, snapshot.request_id);
    if inner.last_submitted == Some(submission_key) {
        return;
    }
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    inner.last_submitted = Some(submission_key);
    let uniform_bytes = werner_uniform_bytes(
        extracted.probe,
        extracted.g_density,
        inner.edge_count,
        inner.face_count,
        inner.item_count,
    );
    render_queue.write_buffer(&inner.uniform, 0, &uniform_bytes);

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("werner_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("werner_pass"),
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

fn poll_werner_readback(
    channel: Res<WernerReadbackChannel>,
    mut acceleration: ResMut<WernerAcceleration>,
    mut potential: ResMut<WernerPotential>,
    mut history: ResMut<WernerGravityHistory>,
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
    let acceleration_is_valid = total.xyz().is_finite();
    let potential_is_valid = total.w.is_finite() && total.w > 0.0;
    if acceleration_is_valid {
        acceleration.0 = total.xyz();
    } else {
        warn!("[werner] discarded non-finite GPU result");
    }
    if potential_is_valid {
        potential.0 = Some(total.w);
    } else {
        potential.0 = None;
        warn!("[werner] discarded invalid GPU potential");
    }
    if acceleration_is_valid && potential_is_valid {
        history.0.push(GravityFieldSample {
            snapshot: packet.snapshot,
            predictive: false,
            body_acceleration: total.xyz(),
            positive_potential: total.w,
            independent_positive_potential: None,
            body_acceleration_jacobian: None,
            eq106_diagnostics: None,
        });
    }
}

fn werner_uniform_bytes(
    probe: Vec3,
    g_density: f32,
    edge_count: u32,
    face_count: u32,
    item_count: u32,
) -> [u8; 32] {
    let mut bytes = [0_u8; 32];
    for (offset, value) in [(0, probe.x), (4, probe.y), (8, probe.z), (12, g_density)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[16..20].copy_from_slice(&edge_count.to_le_bytes());
    bytes[20..24].copy_from_slice(&face_count.to_le_bytes());
    bytes[24..28].copy_from_slice(&item_count.to_le_bytes());
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

    fn polyhedron_field(
        vertices: &[Vec3],
        faces: &[([Vec3; 3], Vec3)],
        sides: &HashMap<(usize, usize), Vec<EdgeSide>>,
        probe: Vec3,
    ) -> (Vec3, f32) {
        let mut edge_sum = Vec3::ZERO;
        let mut edge_potential_sum = 0.0;
        for ((first, second), adjacent) in sides {
            let r0 = vertices[*first] - probe;
            let r1 = vertices[*second] - probe;
            let edge_length = (vertices[*second] - vertices[*first]).length();
            let logarithm = ((r0.length() + r1.length() + edge_length)
                / (r0.length() + r1.length() - edge_length))
                .ln();
            let tensor = outer_product(adjacent[0].face_normal, adjacent[0].edge_outward)
                + outer_product(adjacent[1].face_normal, adjacent[1].edge_outward);
            let tensor_r = tensor * r0;
            edge_sum += tensor_r * logarithm;
            edge_potential_sum += r0.dot(tensor_r) * logarithm;
        }
        let mut face_sum = Vec3::ZERO;
        let mut face_potential_sum = 0.0;
        for (points, normal) in faces {
            let r0 = points[0] - probe;
            let r1 = points[1] - probe;
            let r2 = points[2] - probe;
            let denominator = r0.length() * r1.length() * r2.length()
                + r0.length() * r1.dot(r2)
                + r1.length() * r2.dot(r0)
                + r2.length() * r0.dot(r1);
            let solid_angle = 2.0 * r0.dot(r1.cross(r2)).atan2(denominator);
            let normal_distance = normal.dot(r0);
            face_sum += *normal * normal_distance * solid_angle;
            face_potential_sum += normal_distance * normal_distance * solid_angle;
        }
        (
            -edge_sum + face_sum,
            0.5 * (edge_potential_sum - face_potential_sum),
        )
    }

    #[test]
    fn outer_product_applies_to_vector() {
        let left = Vec3::new(1.0, 2.0, 3.0);
        let right = Vec3::new(4.0, 5.0, 6.0);
        let vector = Vec3::new(0.5, -1.0, 2.0);
        let matrix = outer_product(left, right);
        assert!((matrix * vector - left * right.dot(vector)).length() < 1.0e-5);
    }

    #[test]
    fn uniform_matches_wgsl_layout() {
        let bytes = werner_uniform_bytes(Vec3::new(1.0, 2.0, 3.0), 4.0, 5, 6, 7);
        assert_eq!(f32::from_le_bytes(bytes[12..16].try_into().unwrap()), 4.0);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 6);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 7);
    }

    #[test]
    fn closed_tetrahedron_has_correct_far_field() {
        let vertices = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
        ];
        let triangles = [[0_usize, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        let mut faces = Vec::new();
        let mut sides: HashMap<(usize, usize), Vec<EdgeSide>> = HashMap::new();
        let mut volume = 0.0;

        for mut indices in triangles {
            let mut points = [
                vertices[indices[0]],
                vertices[indices[1]],
                vertices[indices[2]],
            ];
            let raw = (points[1] - points[0]).cross(points[2] - points[0]);
            if raw.dot((points[0] + points[1] + points[2]) / 3.0) < 0.0 {
                indices.swap(1, 2);
                points.swap(1, 2);
            }
            let normal = (points[1] - points[0])
                .cross(points[2] - points[0])
                .normalize();
            volume += points[0].dot(points[1].cross(points[2])) / 6.0;
            faces.push((points, normal));
            for edge_index in 0..3 {
                let next = (edge_index + 1) % 3;
                let key = if indices[edge_index] < indices[next] {
                    (indices[edge_index], indices[next])
                } else {
                    (indices[next], indices[edge_index])
                };
                let edge_direction = (points[next] - points[edge_index]).normalize();
                sides.entry(key).or_default().push(EdgeSide {
                    face_normal: normal,
                    edge_outward: edge_direction.cross(normal),
                });
            }
        }

        let probe = Vec3::new(20.0, 3.0, -2.0);
        let (actual, actual_potential) = polyhedron_field(&vertices, &faces, &sides, probe);
        let expected = -probe.normalize() * volume / probe.length_squared();
        assert!(actual.normalize().dot(expected.normalize()) > 0.999);
        assert!((actual.length() / expected.length() - 1.0).abs() < 0.02);

        let expected_potential = volume / probe.length();
        assert!(actual_potential > 0.0);
        assert!((actual_potential / expected_potential - 1.0).abs() < 0.01);

        let step = 0.2;
        let potential_at = |point| polyhedron_field(&vertices, &faces, &sides, point).1;
        let finite_difference = Vec3::new(
            (potential_at(probe + step * Vec3::X) - potential_at(probe - step * Vec3::X))
                / (2.0 * step),
            (potential_at(probe + step * Vec3::Y) - potential_at(probe - step * Vec3::Y))
                / (2.0 * step),
            (potential_at(probe + step * Vec3::Z) - potential_at(probe - step * Vec3::Z))
                / (2.0 * step),
        );
        let gradient_error = (actual - finite_difference).length() / actual.length();
        assert!(
            gradient_error < 0.02,
            "Werner acceleration/potential gradient mismatch: {gradient_error:.3e}; a={actual:?}, fd={finite_difference:?}"
        );
    }
}
