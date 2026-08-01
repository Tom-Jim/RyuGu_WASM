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

#[derive(Resource)]
pub struct StruveLutSource {
    pub bytes: Vec<u8>,
}

#[derive(Resource, Default)]
struct ExtractedGravInput {
    probe_x: f32,
    probe_y: f32,
    probe_z: f32,
    voxel_bytes: Option<Vec<u8>>,
    lut_bytes: Option<Vec<u8>>,
    voxel_count: u32,
}

#[derive(Resource, Default)]
struct GravGpuBuffers(Option<GravGpuBuffersInner>);

struct GravGpuBuffersInner {
    uniform_buf: bevy::render::render_resource::Buffer,
    voxel_buf: bevy::render::render_resource::Buffer,
    output_buf: bevy::render::render_resource::Buffer,
    lut_buf: bevy::render::render_resource::Buffer,
    voxel_count: u32,
    n_workgroups: u32,
}

#[derive(Resource)]
struct GravComputePipeline {
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource, Default, PartialEq, Eq)]
enum GravDispatchState {
    #[default]
    NeedsBuild,
    Ready,
    Dispatched,
}

pub struct GravityComputePlugin;
impl Plugin for GravityComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GravityReadbackChannel>();
        app.add_systems(
            Update,
            (build_gravity_voxels_system, poll_gravity_readback).chain(),
        );

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ExtractedGravInput>();
        render_app.init_resource::<GravGpuBuffers>();
        render_app.init_resource::<GravDispatchState>();

        render_app.add_systems(ExtractSchedule, extract_grav_input_system);
        render_app.add_systems(Render, dispatch_grav_system.in_set(RenderSystems::Render));
    }

    fn finish(&self, app: &mut App) {
        let channel = app.world().resource::<GravityReadbackChannel>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(channel);
        render_app.init_resource::<GravComputePipeline>();
    }
}

impl FromWorld for GravComputePipeline {
    fn from_world(world: &mut World) -> Self {
        // 4 bindings: uniform(0), voxels(1), output(2), LUT(3)
        let entries = [
            uniform_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            storage_ro_entry(3),
        ];

        let bgl = BindGroupLayoutDescriptor::new("grav_bgl", &entries);
        let shader = world.resource::<AssetServer>().load("shaders/gravity.wgsl");

        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("grav_compute".into()),
                    layout: vec![bgl],
                    immediate_size: 0,
                    shader,
                    shader_defs: vec![],
                    entry_point: None,
                    zero_initialize_workgroup_memory: false,
                });

        GravComputePipeline { pipeline_id }
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

pub fn build_gravity_voxels_system(
    mut commands: Commands,
    topo: Option<Res<AsteroidTopologyGpuData>>,
    ryugu_q: Query<&Transform, With<RyuguMarker>>,
    existing_voxels: Option<Res<GravVoxelSource>>,
    existing_lut: Option<Res<StruveLutSource>>,
    density_c: Option<Res<DensityC>>,
) {
    if existing_lut.is_none() {
        let lut_bytes = include_bytes!("../../lut_struve.bin").to_vec();
        commands.insert_resource(StruveLutSource { bytes: lut_bytes });
    }

    if existing_voxels.is_some() {
        return;
    }
    let Some(topo) = topo else { return };
    let Ok(ryugu_tf) = ryugu_q.single() else {
        return;
    };
    if topo.positions.is_empty() {
        return;
    }

    let scale = ryugu_tf.scale.x;
    let eps = DENSITY_EPSILON;
    // Density coefficient C solved once in scale.rs by 4-point Gaussian quadrature
    // over origin-anchored tets; reusing it here keeps per-voxel masses consistent
    // with the integral that produced it.
    let c_val = density_c.map(|r| r.0).unwrap_or(1.0);

    // 4-point Gaussian quadrature weights on a reference tetrahedron (a for the
    // barycentric-self vertex, b for each of the three mixed vertices).
    let a = (5.0 + 3.0 * 5.0_f32.sqrt()) / 20.0;
    let b = (5.0 - 5.0_f32.sqrt()) / 20.0;

    let expected_voxels = (topo.triangles.len() / 3) * 4;
    let mut voxel_bytes = Vec::with_capacity(expected_voxels * 16);
    let mut voxel_count = 0;

    // Iterate every origin-anchored tet (one per surface triangle). Each tet
    // contributes its signed volume times the 1/r density at the 4 Gauss points,
    // so the summed voxel masses reproduce RYUGU_MASS without any manual scaling.
    for chunk in topo.triangles.chunks(3) {
        let va = topo.positions[chunk[0] as usize] * scale;
        let vb = topo.positions[chunk[1] as usize] * scale;
        let vc = topo.positions[chunk[2] as usize] * scale;

        let vol = va.dot(vb.cross(vc)) / 6.0;

        let p1 = va * a + vb * b + vc * b;
        let p2 = va * b + vb * a + vc * b;
        let p3 = va * b + vb * b + vc * a;
        let p4 = va * b + vb * b + vc * b;

        let points = [p1, p2, p3, p4];

        for p in points {
            let r = p.length().max(1e-3);
            let density = c_val / (r + eps);

            // Per-voxel mass = (1/4) * tet signed-volume * local density. Negative
            // volumes from concavities yield negative masses, cancelling the false
            // pull that concave regions would otherwise exert.
            let mass = vol * 0.25 * density;

            voxel_bytes.extend_from_slice(&p.x.to_le_bytes());
            voxel_bytes.extend_from_slice(&p.y.to_le_bytes());
            voxel_bytes.extend_from_slice(&p.z.to_le_bytes());
            voxel_bytes.extend_from_slice(&mass.to_le_bytes());

            voxel_count += 1;
        }
    }

    // The summed voxel masses exactly equal RYUGU_MASS — the 1/r density kernel
    // and the same 4-point quadrature that produced C make this self-consistent,
    // so no manual mass_scale is needed here.
    commands.insert_resource(GravVoxelSource {
        bytes: voxel_bytes,
        count: voxel_count,
    });
}

pub fn poll_gravity_readback(
    channel: Res<GravityReadbackChannel>,
    mut grav_acc: ResMut<GravityAcceleration>,
    time: Res<Time>,
) {
    let Ok(mut guard) = channel.0.try_lock() else {
        return;
    };
    let Some(partial_sums) = guard.take() else {
        return;
    };

    let total = partial_sums
        .iter()
        .fold(Vec3::ZERO, |acc, v| acc + Vec3::new(v[0], v[1], v[2]));

    if !total.is_finite() {
        warn!(
            "[gravity] GPU readback NaN/Inf detected — discarding. partial_sums[0]={:?}",
            partial_sums.first()
        );
        return;
    }

    let frame = (time.elapsed_secs() * 60.0) as u32;
    if frame % 120 == 0 {
        info!(
            "[gravity] GPU acc=({:.3e},{:.3e},{:.3e}) |mag|={:.3e} n_wg={}",
            total.x,
            total.y,
            total.z,
            total.length(),
            partial_sums.len()
        );
    }

    grav_acc.0 = total;
}

fn extract_grav_input_system(
    mut extracted: ResMut<ExtractedGravInput>,
    mut dispatch: ResMut<GravDispatchState>,
    voxels: Extract<Option<Res<GravVoxelSource>>>,
    lut: Extract<Option<Res<StruveLutSource>>>,
    cassini_q: Extract<Query<&Transform, With<CassiniMarker>>>,
    ryugu_q: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    if *dispatch == GravDispatchState::Dispatched {
        *dispatch = GravDispatchState::Ready;
    }

    let Some(vox_src) = voxels.as_ref() else {
        return;
    };
    let Some(lut_src) = lut.as_ref() else {
        return;
    };
    let Ok(cassini_tf) = cassini_q.single() else {
        return;
    };
    let Ok(ryugu_tf) = ryugu_q.single() else {
        return;
    };

    let inv_rot = ryugu_tf.rotation.inverse();
    let local_pos = inv_rot * (cassini_tf.translation - ryugu_tf.translation);

    extracted.probe_x = local_pos.x;
    extracted.probe_y = local_pos.y;
    extracted.probe_z = local_pos.z;
    extracted.voxel_count = vox_src.count;

    if extracted.voxel_bytes.is_none() {
        extracted.voxel_bytes = Some(vox_src.bytes.clone());
    }
    if extracted.lut_bytes.is_none() {
        extracted.lut_bytes = Some(lut_src.bytes.clone());
    }
}

fn dispatch_grav_system(
    mut state: ResMut<GravDispatchState>,
    mut buffers: ResMut<GravGpuBuffers>,
    pipeline_res: Option<Res<GravComputePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedGravInput>,
    channel: Res<GravityReadbackChannel>,
) {
    if *state == GravDispatchState::Dispatched {
        return;
    }
    let Some(pl) = pipeline_res else { return };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pl.pipeline_id) else {
        return;
    };
    if extracted.voxel_count == 0 {
        return;
    }

    if buffers.0.is_none() {
        let Some(voxel_bytes) = extracted.voxel_bytes.as_ref() else {
            return;
        };
        let Some(lut_bytes) = extracted.lut_bytes.as_ref() else {
            return;
        };

        let n_wg = extracted.voxel_count.div_ceil(64);
        let out_sz = (n_wg as u64) * 16;

        // GravParams uniform: strictly 80 bytes for WGSL memory alignment
        let uniform_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("grav_uniform"),
            size: 80,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let voxel_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("grav_voxels"),
            contents: voxel_bytes,
            usage: BufferUsages::STORAGE,
        });

        let lut_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("grav_lut"),
            contents: lut_bytes,
            usage: BufferUsages::STORAGE,
        });

        let output_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("grav_output"),
            size: out_sz,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        buffers.0 = Some(GravGpuBuffersInner {
            uniform_buf,
            voxel_buf,
            output_buf,
            lut_buf,
            voxel_count: extracted.voxel_count,
            n_workgroups: n_wg,
        });
        *state = GravDispatchState::Ready;
    }

    if *state != GravDispatchState::Ready {
        return;
    }

    let bufs = buffers.0.as_ref().unwrap();
    let n_wg = bufs.n_workgroups;
    let out_sz = (n_wg as u64) * 16;

    let v_stehfest: [f32; 12] = [
        1.0, -49.0, 366.0, -858.0, 810.0, -270.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    // Build 80-byte uniform buffer strictly packed
    let mut ub = Vec::<u8>::with_capacity(80);
    for f in [extracted.probe_x, extracted.probe_y, extracted.probe_z, G] {
        ub.extend_from_slice(&f.to_le_bytes());
    }
    ub.extend_from_slice(&bufs.voxel_count.to_le_bytes());

    // Stehfest M=3 (6 terms): reduced order for performance/stability tradeoff
    ub.extend_from_slice(&3u32.to_le_bytes());

    ub.extend_from_slice(&0f32.to_le_bytes()); // _pad0
    ub.extend_from_slice(&0f32.to_le_bytes()); // _pad1
    for v in v_stehfest.iter() {
        ub.extend_from_slice(&v.to_le_bytes());
    }

    render_queue.write_buffer(&bufs.uniform_buf, 0, &ub);

    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("grav_staging"),
        size: out_sz,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = render_device.create_bind_group_layout(
        "grav_bgl_rt",
        &[
            uniform_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            storage_ro_entry(3), // LUT attached!
        ],
    );

    let bind_group = render_device.create_bind_group(
        "grav_bg",
        &bgl,
        &[
            BindGroupEntry {
                binding: 0,
                resource: bufs.uniform_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: bufs.voxel_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: bufs.output_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: bufs.lut_buf.as_entire_binding(),
            },
        ],
    );

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("grav_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("grav_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_wg, 1, 1);
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

    *state = GravDispatchState::Dispatched;
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
