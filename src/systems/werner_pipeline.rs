use crate::components::*;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    render_resource::{
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::Arc;

// =====================================================================
// Shared resources: the "mailbox" between pipeline and physics
// =====================================================================

/// Final Werner acceleration produced by the WGSL kernel, consumed by the
/// main-world physics system.
#[derive(Resource, Default)]
pub struct WernerAcceleration(pub Vec3);

/// Async CPU/GPU readback channel: GPU partial-sums land here for the poll
/// system to reduce into `WernerAcceleration`.
#[derive(Resource, Clone)]
pub struct WernerReadbackChannel(pub Arc<std::sync::Mutex<Option<Vec<[f32; 4]>>>>);

impl Default for WernerReadbackChannel {
    fn default() -> Self {
        Self(Arc::new(std::sync::Mutex::new(None)))
    }
}

// =====================================================================
// Render-world state and extraction
// =====================================================================

#[derive(Resource, Default)]
struct WernerGpuBuffers(Option<WernerGpuBuffersInner>);

struct WernerGpuBuffersInner {
    uniform_buf: bevy::render::render_resource::Buffer,
    vertex_buf: bevy::render::render_resource::Buffer,
    index_buf: bevy::render::render_resource::Buffer,
    output_buf: bevy::render::render_resource::Buffer,
    density_buf: bevy::render::render_resource::Buffer,
    n_workgroups: u32,
}

#[derive(Resource)]
struct WernerComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource, Default, PartialEq, Eq)]
enum WernerDispatchState {
    #[default]
    NeedsBuild,
    Ready,
    Dispatched,
}

// =====================================================================
// Plugin assembly
// =====================================================================

pub struct WernerComputePlugin;

impl Plugin for WernerComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WernerReadbackChannel>();
        app.init_resource::<WernerAcceleration>();

        app.add_systems(Update, poll_werner_readback);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedWernerInput>();
        render_app.init_resource::<WernerGpuBuffers>();
        render_app.init_resource::<WernerDispatchState>();

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

// WGSL bind-group layout.
impl FromWorld for WernerComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = [
            uniform_entry(0),    // binding(0): params (GravParams)
            storage_ro_entry(1), // binding(1): vertices
            storage_ro_entry(2), // binding(2): indices
            storage_rw_entry(3), // binding(3): partial_sums
            storage_ro_entry(4), // binding(4): face_densities
        ];

        let bgl = BindGroupLayoutDescriptor::new("werner_bgl", &entries);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/werner_gravity.wgsl");
        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("werner_compute".into()),
                    layout: vec![bgl],
                    immediate_size: 0,
                    shader,
                    shader_defs: vec![],
                    entry_point: None,
                    zero_initialize_workgroup_memory: false,
                });
        WernerComputePipeline { pipeline_id }
    }
}

// =====================================================================
// Extract: push the probe position into GPU memory
// =====================================================================

// Per-face local density `G * rho` is recomputed here (rather than reused from
// the voxel source) so the GPU path remains a fair apples-to-apples comparison
// with the CPU reference.
#[derive(Resource, Default)]
struct ExtractedWernerInput {
    probe_x: f32,
    probe_y: f32,
    probe_z: f32,
    num_faces: u32,
    vertices_bytes: Option<Vec<u8>>,
    indices_bytes: Option<Vec<u8>>,
    densities_bytes: Option<Vec<u8>>,
}

// =====================================================================
// Core split: each surface triangle becomes 4 outward Weyl faces
// (1 surface triangle + 3 inner triangles against the origin vertex).
// =====================================================================
fn extract_werner_input_system(
    mut extracted: ResMut<ExtractedWernerInput>,
    mut dispatch: ResMut<WernerDispatchState>,
    topo: Extract<Option<Res<AsteroidTopologyGpuData>>>,
    density_c: Extract<Option<Res<DensityC>>>,
    cassini_q: Extract<Query<&Transform, With<CassiniWernerMarker>>>,
    ryugu_q: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    if *dispatch == WernerDispatchState::Dispatched {
        *dispatch = WernerDispatchState::Ready;
    }
    let Some(topo_data) = topo.as_ref() else {
        return;
    };
    let Ok(cassini_tf) = cassini_q.single() else {
        return;
    };
    let Ok(ryugu_tf) = ryugu_q.single() else {
        return;
    };

    let scale = ryugu_tf.scale.x;
    let inv_rot = ryugu_tf.rotation.inverse();
    let local_pos = inv_rot * (cassini_tf.translation - ryugu_tf.translation);

    extracted.probe_x = local_pos.x;
    extracted.probe_y = local_pos.y;
    extracted.probe_z = local_pos.z;
    // Each surface triangle is split into 4 faces (1 surface + 3 inner), so the
    // GPU sees 4× the triangle count of the raw mesh.
    extracted.num_faces = (topo_data.triangles.len() / 3) as u32 * 4;

    if extracted.vertices_bytes.is_none() {
        let c_val = density_c.as_ref().map(|r| r.0).unwrap_or(1.0);
        let eps = DENSITY_EPSILON;

        let mut pos_bytes = Vec::with_capacity((topo_data.positions.len() + 1) * 16);
        for p in topo_data.positions.iter() {
            let sp = *p * scale;
            pos_bytes.extend_from_slice(&sp.x.to_le_bytes());
            pos_bytes.extend_from_slice(&sp.y.to_le_bytes());
            pos_bytes.extend_from_slice(&sp.z.to_le_bytes());
            pos_bytes.extend_from_slice(&0f32.to_le_bytes());
        }
        let origin_idx = topo_data.positions.len() as u32;
        // Append the origin as an extra vertex so inner faces can reference it.
        pos_bytes.extend_from_slice(&0f32.to_le_bytes());
        pos_bytes.extend_from_slice(&0f32.to_le_bytes());
        pos_bytes.extend_from_slice(&0f32.to_le_bytes());
        pos_bytes.extend_from_slice(&0f32.to_le_bytes());

        let mut idx_bytes = Vec::with_capacity(extracted.num_faces as usize * 12);
        let mut density_bytes = Vec::with_capacity(extracted.num_faces as usize * 4);
        let mut total_mass = 0.0;

        // Pass 1: precompute per-tet mass and density (used only to derive
        // RYUGU_MASS-aligned scaling — not stored on the GPU path).
        struct Tet {
            v0: u32,
            v1: u32,
            v2: u32,
            density: f32,
        }
        let mut tets = Vec::new();

        for chunk in topo_data.triangles.chunks(3) {
            let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
            let v0 = topo_data.positions[i0 as usize] * scale;
            let v1 = topo_data.positions[i1 as usize] * scale;
            let v2 = topo_data.positions[i2 as usize] * scale;

            let centroid = (v0 + v1 + v2) * 0.25; // Tet centroid = (v0+v1+v2+origin)/4
            let r = centroid.length().max(1e-3);
            let local_density = c_val / (r + eps);

            let vol = (v0.dot(v1.cross(v2)) / 6.0).abs();
            total_mass += vol * local_density;

            tets.push(Tet {
                v0: i0,
                v1: i1,
                v2: i2,
                density: local_density,
            });
        }

        // Pass 2: scale densities so total mass matches RYUGU_MASS exactly,
        // then emit the 4 outward-facing Weyl faces per tet.
        let mass_scale = RYUGU_MASS / total_mass;

        for t in tets {
            let g_rho = G * t.density * mass_scale;
            let g_rho_bytes = g_rho.to_le_bytes();

            // 1. Outward surface triangle.
            idx_bytes.extend_from_slice(&t.v0.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v1.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v2.to_le_bytes());
            density_bytes.extend_from_slice(&g_rho_bytes);

            // 2-4. Three inner triangles anchored at the origin. Winding order
            // chosen so the outward normal points away from the surface.
            idx_bytes.extend_from_slice(&origin_idx.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v1.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v0.to_le_bytes());
            density_bytes.extend_from_slice(&g_rho_bytes);

            idx_bytes.extend_from_slice(&origin_idx.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v2.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v1.to_le_bytes());
            density_bytes.extend_from_slice(&g_rho_bytes);

            idx_bytes.extend_from_slice(&origin_idx.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v0.to_le_bytes());
            idx_bytes.extend_from_slice(&t.v2.to_le_bytes());
            density_bytes.extend_from_slice(&g_rho_bytes);
        }

        extracted.vertices_bytes = Some(pos_bytes);
        extracted.indices_bytes = Some(idx_bytes);
        extracted.densities_bytes = Some(density_bytes);
    }
}
fn dispatch_werner_system(
    mut state: ResMut<WernerDispatchState>,
    mut buffers: ResMut<WernerGpuBuffers>,
    pipeline_res: Option<Res<WernerComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedWernerInput>,
    channel: Res<WernerReadbackChannel>,
) {
    if *state == WernerDispatchState::Dispatched {
        return;
    }
    let Some(pl) = pipeline_res else { return };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pl.pipeline_id) else {
        return;
    };

    if extracted.num_faces == 0 {
        return;
    }

    if buffers.0.is_none() {
        let n_wg = extracted.num_faces.div_ceil(64);
        let out_sz = (extracted.num_faces as u64) * 16;

        let uniform_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("werner_uniform"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("werner_vertices"),
            contents: extracted.vertices_bytes.as_ref().unwrap(),
            usage: BufferUsages::STORAGE,
        });

        let index_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("werner_indices"),
            contents: extracted.indices_bytes.as_ref().unwrap(),
            usage: BufferUsages::STORAGE,
        });
        let density_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("werner_densities"),
            contents: extracted.densities_bytes.as_ref().unwrap(),
            usage: BufferUsages::STORAGE,
        });
        let output_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("werner_output"),
            size: out_sz,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        buffers.0 = Some(WernerGpuBuffersInner {
            uniform_buf,
            vertex_buf,
            index_buf,
            output_buf,
            density_buf,
            n_workgroups: n_wg,
        });
        *state = WernerDispatchState::Ready;
    }

    if *state != WernerDispatchState::Ready {
        return;
    }

    let bufs = buffers.0.as_ref().unwrap();
    let out_sz = (extracted.num_faces as u64) * 16;

    let mut ub = Vec::<u8>::with_capacity(32);
    ub.extend_from_slice(&extracted.probe_x.to_le_bytes());
    ub.extend_from_slice(&extracted.probe_y.to_le_bytes());
    ub.extend_from_slice(&extracted.probe_z.to_le_bytes());
    ub.extend_from_slice(&0f32.to_le_bytes());
    ub.extend_from_slice(&0f32.to_le_bytes());
    ub.extend_from_slice(&extracted.num_faces.to_le_bytes());
    ub.extend_from_slice(&0f32.to_le_bytes());
    ub.extend_from_slice(&0f32.to_le_bytes());
    render_queue.write_buffer(&bufs.uniform_buf, 0, &ub);

    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("werner_staging"),
        size: out_sz,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = render_device.create_bind_group_layout(
        "werner_bgl_rt",
        &[
            uniform_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            storage_ro_entry(4),
        ],
    );
    let bind_group = render_device.create_bind_group(
        "werner_bg",
        &bgl,
        &[
            BindGroupEntry {
                binding: 0,
                resource: bufs.uniform_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: bufs.vertex_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: bufs.index_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: bufs.output_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: bufs.density_buf.as_entire_binding(),
            },
        ],
    );

    let mut encoder =
        render_device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(bufs.n_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&bufs.output_buf, 0, &staging, 0, out_sz);
    render_queue.submit([encoder.finish()]);

    let shared = Arc::clone(&channel.0);
    let staging_ref = staging.clone();
    staging.slice(..).map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            let view = staging_ref.slice(..).get_mapped_range();
            let partial = bytes_to_f32x4(&view);
            if let Ok(mut lock) = shared.lock() {
                *lock = Some(partial);
            }
            drop(view);
            staging_ref.unmap();
        }
    });
    *state = WernerDispatchState::Dispatched;
}
// =====================================================================
// Readback: CPU reduces GPU face contributions into one acceleration
// =====================================================================
fn poll_werner_readback(
    channel: Res<WernerReadbackChannel>,
    mut werner_acc: ResMut<WernerAcceleration>,
) {
    let Ok(mut guard) = channel.0.try_lock() else {
        return;
    };
    let Some(partial_sums) = guard.take() else {
        return;
    };

    // Sum the per-face GPU contributions (xyz channels; w is unused) into the
    // single acceleration that physics_system consumes.
    let total = partial_sums
        .iter()
        .fold(Vec3::ZERO, |acc, v| acc + Vec3::new(v[0], v[1], v[2]));

    if total.is_finite() {
        werner_acc.0 = total;
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
fn storage_ro_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn storage_rw_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn bytes_to_f32x4(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .chunks_exact(16)
        .map(|c| {
            let mut v = [0f32; 4];
            for (i, b4) in c.chunks_exact(4).enumerate() {
                v[i] = f32::from_le_bytes(b4.try_into().unwrap());
            }
            v
        })
        .collect()
}
