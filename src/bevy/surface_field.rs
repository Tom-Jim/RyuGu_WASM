//! Surface gravity, effective-gravity slope, and method-comparison products.
//!
//! The real-time GPU paths remain responsible for the probe trajectory. This
//! module owns an explicit, reproducible surface product: common surface
//! patches are evaluated with method-specific CPU reference operators, then
//! uploaded as vertex colors on a thin overlay mesh. That keeps the display
//! useful for validation without making a pointwise claim for equation (184).

use crate::cpu::frequency_domain::{
    AggregatedGravitySource, EQ184_QUADRATURE_COUNT, eq184_quadrature_node,
};
use crate::gpu::mmfft::{MmfftLevelWorkspace, sample_mmfft_grid};
use crate::interface::components::*;
use bevy::asset::RenderAssetUsages;
use bevy::math::{DMat3, DVec3};
use bevy::mesh::PrimitiveTopology;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::mesh::Indices;
use num_complex::Complex64;
use std::collections::HashMap;

const SURFACE_PATCH_LIMIT: usize = 1_024;
const CONSTANT_SOURCE_LIMIT: usize = 8_192;
const SURFACE_COMPUTE_CHUNK: usize = 96;
const SURFACE_COMPUTE_BUDGET_MS: f32 = 3.0;
const SURFACE_EXPENSIVE_CHUNK: usize = 8;
const FFT_GRID_SIZE: usize = 64;
const FFT_HALF_EXTENT: f64 = 4_096.0;
const FMM_MAX_LEVEL: u32 = 5;
const FMM_LEAF_LIMIT: usize = 16;
const FMM_THETA: f64 = 0.10;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceFieldPatch {
    pub body_position: Vec3,
    pub normal: Vec3,
}

#[derive(Resource, Default)]
pub(crate) struct SurfaceFieldGeometry {
    pub patches: Vec<SurfaceFieldPatch>,
    pub render_positions: Vec<[f32; 3]>,
    pub render_normals: Vec<[f32; 3]>,
    pub render_indices: Vec<u32>,
    pub render_sample_indices: Vec<usize>,
    pub topology_node_count: u32,
    pub triangle_count: usize,
    pub scale: f32,
}

#[derive(Component)]
pub(crate) struct SurfaceFieldOverlay;

#[derive(Resource, Default)]
pub(crate) struct SurfaceFieldComputeState {
    job: Option<SurfaceFieldJob>,
}

struct SurfaceFieldJob {
    methods: Vec<ActiveGravityMethod>,
    method_index: usize,
    patch_index: usize,
    density_mode: DensityMode,
    evaluator: Option<SurfaceEvaluator>,
    datasets: Vec<SurfaceFieldDataset>,
}

enum SurfaceEvaluator {
    Point(PointMassEvaluator),
    Fft(FftSurfaceEvaluator),
    Fmm(FmmSurfaceEvaluator),
    Werner(WernerSurfaceEvaluator),
    Equation106(Vec<(DVec3, Complex64)>),
}

#[derive(Clone, Copy, Debug)]
struct FieldValue {
    gravity: Vec3,
    potential: f32,
    jacobian: DMat3,
}

impl Default for FieldValue {
    fn default() -> Self {
        Self {
            gravity: Vec3::ZERO,
            potential: 0.0,
            jacobian: DMat3::ZERO,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PointMass {
    position: DVec3,
    mass: f64,
}

struct PointMassEvaluator {
    sources: Vec<PointMass>,
}

struct FftSurfaceEvaluator {
    field: Vec<[f32; 4]>,
    n: usize,
    half_extent: f32,
}

struct FmmSurfaceEvaluator {
    tree: CpuFmmTree,
}

struct WernerSurfaceEvaluator {
    edges: Vec<WernerSurfaceEdge>,
    faces: Vec<WernerSurfaceFace>,
    g_density: f64,
}

#[derive(Clone, Copy)]
struct WernerSurfaceEdge {
    p0: DVec3,
    p1: DVec3,
    tensor_rows: DMat3,
    length: f64,
}

#[derive(Clone, Copy)]
struct WernerSurfaceFace {
    p0: DVec3,
    p1: DVec3,
    p2: DVec3,
    normal: DVec3,
}

#[derive(Clone, Copy)]
struct WernerSurfaceEdgeSide {
    normal: DVec3,
    edge_outward: DVec3,
}

struct CpuFmmTree {
    nodes: Vec<CpuFmmNode>,
    sources: Vec<PointMass>,
}

struct CpuFmmNode {
    half: f64,
    mass: f64,
    center_of_mass: DVec3,
    quadrupole: DMat3,
    level: u32,
    children: [Option<usize>; 8],
    particle_indices: Vec<usize>,
}

pub(crate) fn queue_surface_field(
    state: &mut SurfaceFieldState,
    compute: &mut SurfaceFieldComputeState,
    method: ActiveGravityMethod,
) {
    compute.job = None;
    state.latest = None;
    state.comparison = None;
    state.selected_patch = None;
    state.revision = state.revision.wrapping_add(1);
    state.computing = true;
    state.status = format!(
        "Computing {} surface gravity, gradient, and slope...",
        method.as_str()
    );
    compute.job = Some(SurfaceFieldJob {
        methods: vec![method],
        method_index: 0,
        patch_index: 0,
        density_mode: state.density_mode,
        evaluator: None,
        datasets: Vec::new(),
    });
}

pub(crate) fn cancel_surface_field(
    state: &mut SurfaceFieldState,
    compute: &mut SurfaceFieldComputeState,
    status: &str,
) {
    compute.job = None;
    state.computing = false;
    state.latest = None;
    state.comparison = None;
    state.selected_patch = None;
    state.revision = state.revision.wrapping_add(1);
    state.status = status.into();
}

pub(crate) fn queue_surface_comparison(
    state: &mut SurfaceFieldState,
    compute: &mut SurfaceFieldComputeState,
) {
    if state.computing {
        state.status = "A surface calculation is already running.".into();
        return;
    }
    if state.baseline_method == state.comparison_method {
        state.status = "Choose two different algorithms for an error map.".into();
        return;
    }
    state.latest = None;
    state.comparison = None;
    state.selected_patch = None;
    state.revision = state.revision.wrapping_add(1);
    state.computing = true;
    state.status = format!(
        "Comparing {} against {} on the same surface patches...",
        state.comparison_method.as_str(),
        state.baseline_method.as_str()
    );
    compute.job = Some(SurfaceFieldJob {
        methods: vec![state.baseline_method, state.comparison_method],
        method_index: 0,
        patch_index: 0,
        density_mode: state.density_mode,
        evaluator: None,
        datasets: Vec::new(),
    });
}

pub(crate) fn build_surface_field_geometry_system(
    mut geometry: ResMut<SurfaceFieldGeometry>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    ryugu_query: Query<&Transform, With<RyuguMarker>>,
) {
    let Some(topology) = topology else { return };
    let Ok(transform) = ryugu_query.single() else {
        return;
    };
    let triangle_count = topology.triangles.len() / 3;
    let scale = transform.scale.x;
    if geometry.topology_node_count == topology.node_count
        && geometry.triangle_count == triangle_count
        && geometry.scale.to_bits() == scale.to_bits()
        && !geometry.patches.is_empty()
    {
        return;
    }

    let stride = triangle_count.div_ceil(SURFACE_PATCH_LIMIT).max(1);
    let mut patches = Vec::with_capacity(triangle_count.div_ceil(stride));
    let mut sampled_triangle_indices = Vec::with_capacity(triangle_count.div_ceil(stride));
    let mut render_triangles = Vec::with_capacity(triangle_count);
    for (triangle_index, triangle) in topology.triangles.chunks_exact(3).enumerate() {
        let Some((&a, rest)) = triangle.split_first() else {
            continue;
        };
        let [b, c] = [rest[0], rest[1]];
        let Some(p0) = topology.positions.get(a as usize).copied() else {
            continue;
        };
        let Some(p1) = topology.positions.get(b as usize).copied() else {
            continue;
        };
        let Some(p2) = topology.positions.get(c as usize).copied() else {
            continue;
        };
        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let local_centroid = (p0 + p1 + p2) / 3.0;
        if normal.dot(local_centroid) < 0.0 {
            normal = -normal;
        }
        if normal == Vec3::ZERO {
            continue;
        }
        let local_vertices = [p0, p1, p2];
        render_triangles.push((triangle_index, local_vertices, normal));
        if triangle_index.is_multiple_of(stride) {
            sampled_triangle_indices.push(triangle_index);
            patches.push(SurfaceFieldPatch {
                body_position: local_centroid * scale,
                normal,
            });
        }
    }
    let offset = 0.35 / scale.max(f32::MIN_POSITIVE);
    let mut render_positions = Vec::with_capacity(render_triangles.len() * 3);
    let mut render_normals = Vec::with_capacity(render_triangles.len() * 3);
    let mut render_indices = Vec::with_capacity(render_triangles.len() * 3);
    let mut render_sample_indices = Vec::with_capacity(render_triangles.len());
    for (triangle_index, local_vertices, normal) in render_triangles {
        let base = render_positions.len() as u32;
        for vertex in local_vertices {
            render_positions.push((vertex + normal * offset).to_array());
            render_normals.push(normal.to_array());
        }
        render_indices.extend_from_slice(&[base, base + 1, base + 2]);
        let sample_index = sampled_triangle_indices
            .partition_point(|&sample_triangle| sample_triangle <= triangle_index)
            .saturating_sub(1);
        render_sample_indices.push(sample_index.min(patches.len().saturating_sub(1)));
    }
    geometry.patches = patches;
    geometry.render_positions = render_positions;
    geometry.render_normals = render_normals;
    geometry.render_indices = render_indices;
    geometry.render_sample_indices = render_sample_indices;
    geometry.topology_node_count = topology.node_count;
    geometry.triangle_count = triangle_count;
    geometry.scale = scale;
}

pub(crate) fn ensure_surface_field_overlay_system(
    mut commands: Commands,
    geometry: Res<SurfaceFieldGeometry>,
    ryugu_query: Query<Entity, With<RyuguMarker>>,
    overlay_query: Query<Entity, With<SurfaceFieldOverlay>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if geometry.patches.is_empty()
        || geometry.render_sample_indices.is_empty()
        || overlay_query.iter().next().is_some()
    {
        return;
    }
    let Some(root) = ryugu_query.iter().next() else {
        return;
    };
    let positions = geometry.render_positions.clone();
    let normals = geometry.render_normals.clone();
    let colors = vec![[0.0, 0.0, 0.0, 0.0]; positions.len()];
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(geometry.render_indices.clone()));
    let mesh = meshes.add(mesh);
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        cull_mode: None,
        alpha_mode: AlphaMode::Opaque,
        depth_bias: 4.0,
        ..default()
    });
    let overlay = commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            SurfaceFieldOverlay,
            Visibility::Hidden,
        ))
        .id();
    commands.entity(root).add_child(overlay);
}

pub(crate) fn surface_field_compute_system(
    geometry: Res<SurfaceFieldGeometry>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    aggregated: Option<Res<AggregatedGravitySource>>,
    mut state: ResMut<SurfaceFieldState>,
    mut compute: ResMut<SurfaceFieldComputeState>,
) {
    if geometry.patches.is_empty() || !state.computing {
        return;
    }
    let Some(topology) = topology else {
        state.status = "Waiting for the Ryugu topology to finish loading.".into();
        return;
    };
    let Some(job) = compute.job.as_mut() else {
        state.computing = false;
        state.status = "Surface calculation state was cleared; press Calculate again.".into();
        return;
    };

    if job.evaluator.is_none() {
        let method = job.methods[job.method_index];
        if job.density_mode == DensityMode::Variable && aggregated.is_none() {
            state.status = "Waiting for the variable-density source distribution...".into();
            return;
        }
        let sources = build_surface_sources(
            job.density_mode,
            &topology,
            aggregated.as_deref(),
            geometry.scale,
        );
        if sources.is_empty() {
            state.computing = false;
            state.status = "Could not build a finite surface source distribution.".into();
            compute.job = None;
            return;
        }
        job.evaluator = Some(build_evaluator(
            method,
            sources,
            &topology,
            geometry.scale,
            job.density_mode,
        ));
        job.datasets.push(SurfaceFieldDataset {
            method,
            density_mode: job.density_mode,
            samples: Vec::with_capacity(geometry.patches.len()),
            gravity_range: (f32::INFINITY, f32::NEG_INFINITY),
            effective_gravity_range: (f32::INFINITY, f32::NEG_INFINITY),
            gradient_range: (f32::INFINITY, f32::NEG_INFINITY),
            slope_range: (f32::INFINITY, f32::NEG_INFINITY),
        });
    }

    let start = job.patch_index;
    let method = job.methods[job.method_index];
    let chunk = matches!(
        method,
        ActiveGravityMethod::Fmm | ActiveGravityMethod::HomogeneousWerner
    )
    .then_some(SURFACE_EXPENSIVE_CHUNK)
    .unwrap_or(SURFACE_COMPUTE_CHUNK);
    let end = (start + chunk).min(geometry.patches.len());
    let evaluator = job
        .evaluator
        .as_ref()
        .expect("surface evaluator initialized");
    let dataset = job
        .datasets
        .last_mut()
        .expect("surface dataset initialized");
    let budget_start = Instant::now();
    let mut processed_end = start;
    for patch in &geometry.patches[start..end] {
        if processed_end > start
            && budget_start.elapsed().as_secs_f32() * 1_000.0 >= SURFACE_COMPUTE_BUDGET_MS
        {
            break;
        }
        let sample = evaluate_patch(evaluator, *patch);
        if !sample.gravity.is_finite()
            || !sample.effective_gravity.is_finite()
            || !sample.gravity_magnitude.is_finite()
            || !sample.effective_gravity_magnitude.is_finite()
            || !sample.gradient_magnitude.is_finite()
            || !sample.slope_degrees.is_finite()
        {
            state.computing = false;
            state.status = "Surface evaluator returned a non-finite field; calculation stopped. Adjust the sampling offset and calculate again.".into();
            compute.job = None;
            return;
        }
        dataset.gravity_range.0 = dataset.gravity_range.0.min(sample.gravity_magnitude);
        dataset.gravity_range.1 = dataset.gravity_range.1.max(sample.gravity_magnitude);
        dataset.effective_gravity_range.0 = dataset
            .effective_gravity_range
            .0
            .min(sample.effective_gravity_magnitude);
        dataset.effective_gravity_range.1 = dataset
            .effective_gravity_range
            .1
            .max(sample.effective_gravity_magnitude);
        dataset.gradient_range.0 = dataset.gradient_range.0.min(sample.gradient_magnitude);
        dataset.gradient_range.1 = dataset.gradient_range.1.max(sample.gradient_magnitude);
        dataset.slope_range.0 = dataset.slope_range.0.min(sample.slope_degrees);
        dataset.slope_range.1 = dataset.slope_range.1.max(sample.slope_degrees);
        dataset.samples.push(sample);
        processed_end += 1;
    }
    job.patch_index = processed_end;
    if processed_end < geometry.patches.len() {
        state.status = format!(
            "{}: {}/{} surface patches evaluated...",
            job.methods[job.method_index].as_str(),
            processed_end,
            geometry.patches.len()
        );
        return;
    }

    job.evaluator = None;
    job.patch_index = 0;
    job.method_index += 1;
    if job.method_index < job.methods.len() {
        state.status = format!(
            "{} complete; evaluating {}...",
            job.datasets
                .last()
                .expect("completed dataset")
                .method
                .as_str(),
            job.methods[job.method_index].as_str()
        );
        return;
    }

    let finished = compute.job.take().expect("surface job exists");
    state.computing = false;
    state.revision = state.revision.wrapping_add(1);
    if finished.datasets.len() == 1 {
        let dataset = finished.datasets.into_iter().next().expect("one dataset");
        state.latest = Some(dataset);
        state.comparison = None;
        state.selected_patch = state
            .latest
            .as_ref()
            .and_then(|dataset| (!dataset.samples.is_empty()).then_some(0));
        state.status = format!(
            "{} surface product ready: gravity, gradient, effective slope.",
            state
                .latest
                .as_ref()
                .expect("latest dataset")
                .method
                .as_str()
        );
    } else {
        let mut datasets = finished.datasets.into_iter();
        let baseline = datasets.next().expect("baseline dataset");
        let comparison = datasets.next().expect("comparison dataset");
        let signed_errors = baseline
            .samples
            .iter()
            .zip(&comparison.samples)
            .map(|(base, candidate)| {
                (candidate.effective_gravity_magnitude - base.effective_gravity_magnitude)
                    / base.effective_gravity_magnitude.max(1.0e-12)
            })
            .collect::<Vec<_>>();
        let error_range = signed_errors.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
        state.latest = Some(comparison.clone());
        state.selected_patch = (!comparison.samples.is_empty()).then_some(0);
        state.comparison = Some(SurfaceFieldComparison {
            baseline,
            comparison,
            signed_errors,
            error_range: finite_range(error_range),
        });
        state.status = format!(
            "Error map ready: {} compared with {}. Positive is overestimation; negative is underestimation.",
            state.comparison_method.as_str(),
            state.baseline_method.as_str()
        );
    }
}

pub(crate) fn surface_field_render_system(
    state: Res<SurfaceFieldState>,
    geometry: Res<SurfaceFieldGeometry>,
    mut overlay_query: Query<(&Mesh3d, &mut Visibility), With<SurfaceFieldOverlay>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut last_render: Local<(u64, SurfaceFieldMetric)>,
) {
    let Some((mesh_handle, mut visibility)) = overlay_query.iter_mut().next() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) else {
        return;
    };
    let has_error = state.comparison.is_some();
    let has_scalar = state.latest.is_some();
    if !has_error && !has_scalar {
        *visibility = Visibility::Hidden;
        return;
    }
    if last_render.0 == state.revision && last_render.1 == state.metric {
        *visibility = Visibility::Visible;
        return;
    }

    let mut colors = Vec::with_capacity(geometry.render_sample_indices.len() * 3);
    let mut push_color = |color: [f32; 4]| {
        colors.extend_from_slice(&[color, color, color]);
    };
    match state.metric {
        SurfaceFieldMetric::Error => {
            let Some(comparison) = state.comparison.as_ref() else {
                *visibility = Visibility::Hidden;
                return;
            };
            let maximum = comparison
                .error_range
                .0
                .abs()
                .max(comparison.error_range.1.abs())
                .max(1.0e-6);
            for &sample_index in &geometry.render_sample_indices {
                let error = comparison
                    .signed_errors
                    .get(sample_index)
                    .copied()
                    .unwrap_or(0.0);
                push_color(diverging_error_color(error / maximum));
            }
        }
        SurfaceFieldMetric::Gravity => {
            let Some(dataset) = state.latest.as_ref() else {
                return;
            };
            for &sample_index in &geometry.render_sample_indices {
                let Some(sample) = dataset.samples.get(sample_index) else {
                    continue;
                };
                let t = normalize_range(
                    sample.effective_gravity_magnitude,
                    dataset.effective_gravity_range,
                );
                push_color(scientific_color(t, SurfaceFieldMetric::Gravity));
            }
        }
        SurfaceFieldMetric::Gradient => {
            let Some(dataset) = state.latest.as_ref() else {
                return;
            };
            for &sample_index in &geometry.render_sample_indices {
                let Some(sample) = dataset.samples.get(sample_index) else {
                    continue;
                };
                let t = normalize_range(sample.gradient_magnitude, dataset.gradient_range);
                push_color(scientific_color(t, SurfaceFieldMetric::Gradient));
            }
        }
        SurfaceFieldMetric::Slope => {
            let Some(dataset) = state.latest.as_ref() else {
                return;
            };
            for &sample_index in &geometry.render_sample_indices {
                let Some(sample) = dataset.samples.get(sample_index) else {
                    continue;
                };
                let t = normalize_range(sample.slope_degrees, dataset.slope_range);
                push_color(scientific_color(t, SurfaceFieldMetric::Slope));
            }
        }
    }
    if colors.len() != geometry.render_sample_indices.len() * 3 {
        *visibility = Visibility::Hidden;
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    *visibility = Visibility::Visible;
    *last_render = (state.revision, state.metric);
}

fn build_surface_sources(
    density_mode: DensityMode,
    topology: &AsteroidTopologyGpuData,
    aggregated: Option<&AggregatedGravitySource>,
    scale: f32,
) -> Vec<PointMass> {
    if let Some(source) = aggregated {
        let selected = match density_mode {
            DensityMode::Variable => &source.sources,
            DensityMode::Constant => &source.constant_sources,
        };
        if !selected.is_empty() {
            return selected
                .iter()
                .filter_map(|source| {
                    (source.position.is_finite() && source.mass.is_finite() && source.mass > 0.0)
                        .then_some(PointMass {
                            position: source.position,
                            mass: source.mass,
                        })
                })
                .collect();
        }
    }
    build_constant_sources(topology, scale)
}

fn build_constant_sources(topology: &AsteroidTopologyGpuData, scale: f32) -> Vec<PointMass> {
    let faces = topology.triangles.chunks_exact(3).collect::<Vec<_>>();
    if faces.is_empty() {
        return Vec::new();
    }
    let stride = faces.len().div_ceil(CONSTANT_SOURCE_LIMIT).max(1);
    let mut raw = Vec::with_capacity(faces.len().div_ceil(stride));
    let mut selected_volume = 0.0_f64;
    for (face_index, face) in faces.iter().enumerate() {
        if !face_index.is_multiple_of(stride) {
            continue;
        }
        let Some(p0) = topology.positions.get(face[0] as usize).copied() else {
            continue;
        };
        let Some(p1) = topology.positions.get(face[1] as usize).copied() else {
            continue;
        };
        let Some(p2) = topology.positions.get(face[2] as usize).copied() else {
            continue;
        };
        let p0 = p0 * scale;
        let p1 = p1 * scale;
        let p2 = p2 * scale;
        let volume = (p0.dot(p1.cross(p2)).abs() / 6.0) as f64;
        let position = ((p0 + p1 + p2) / 4.0).as_dvec3();
        if volume.is_finite() && volume > 0.0 && position.is_finite() {
            selected_volume += volume;
            raw.push((position, volume));
        }
    }
    if selected_volume <= f64::EPSILON {
        return Vec::new();
    }
    raw.into_iter()
        .map(|(position, volume)| PointMass {
            position,
            mass: RYUGU_MASS as f64 * volume / selected_volume,
        })
        .collect()
}

fn build_radial_surface_sources(
    topology: &AsteroidTopologyGpuData,
    scale: f32,
) -> Option<Vec<PointMass>> {
    const LAYERS: usize = 4;
    let mut raw = Vec::new();
    let mut total_volume = 0.0_f64;
    for triangle in topology.triangles.chunks_exact(3) {
        let p0 = topology.positions.get(triangle[0] as usize)?.to_owned() * scale;
        let p1 = topology.positions.get(triangle[1] as usize)?.to_owned() * scale;
        let p2 = topology.positions.get(triangle[2] as usize)?.to_owned() * scale;
        let n0 = p0.try_normalize()?;
        let n1 = p1.try_normalize()?;
        let n2 = p2.try_normalize()?;
        let direction = (n0 + n1 + n2).try_normalize()?;
        let numerator = n0.dot(n1.cross(n2)).abs();
        let denominator = 1.0 + n0.dot(n1) + n1.dot(n2) + n2.dot(n0);
        let solid_angle = 2.0 * numerator.atan2(denominator);
        if !solid_angle.is_finite() || solid_angle <= 0.0 {
            continue;
        }
        let face_normal = (p1 - p0).cross(p2 - p0);
        let plane_radius = face_normal.dot(direction).abs();
        let radius = if plane_radius > f32::EPSILON {
            face_normal.dot(p0).abs() / plane_radius
        } else {
            ((p0 + p1 + p2) / 3.0).length()
        };
        if !radius.is_finite() || radius <= 0.0 {
            continue;
        }
        for layer in 0..LAYERS {
            let inner = radius * (layer as f32 / LAYERS as f32).cbrt();
            let outer = radius * ((layer + 1) as f32 / LAYERS as f32).cbrt();
            let volume = solid_angle as f64 * (outer.powi(3) - inner.powi(3)) as f64 / 3.0;
            let radial_centroid = 0.75 * (outer.powi(4) - inner.powi(4)) as f64
                / (outer.powi(3) - inner.powi(3)).max(f32::MIN_POSITIVE) as f64;
            let position = direction.as_dvec3() * radial_centroid;
            if volume.is_finite() && volume > 0.0 && position.is_finite() {
                total_volume += volume;
                raw.push((position, volume));
            }
        }
    }
    if raw.is_empty() || total_volume <= f64::EPSILON {
        return None;
    }
    Some(
        raw.into_iter()
            .map(|(position, volume)| PointMass {
                position,
                mass: RYUGU_MASS as f64 * volume / total_volume,
            })
            .collect(),
    )
}

fn build_werner_evaluator(
    topology: &AsteroidTopologyGpuData,
    scale: f32,
) -> Option<WernerSurfaceEvaluator> {
    let mut edge_sides: HashMap<(u32, u32), Vec<WernerSurfaceEdgeSide>> = HashMap::new();
    let mut faces = Vec::new();
    let mut volume = 0.0_f64;
    for triangle in topology.triangles.chunks_exact(3) {
        let mut indices = [triangle[0], triangle[1], triangle[2]];
        let mut points = [
            *topology.positions.get(indices[0] as usize)? * scale,
            *topology.positions.get(indices[1] as usize)? * scale,
            *topology.positions.get(indices[2] as usize)? * scale,
        ];
        let centroid = (points[0] + points[1] + points[2]) / 3.0;
        let raw_normal = (points[1] - points[0]).cross(points[2] - points[0]);
        if raw_normal.dot(centroid) < 0.0 {
            indices.swap(1, 2);
            points.swap(1, 2);
        }
        let normal = (points[1] - points[0])
            .cross(points[2] - points[0])
            .try_normalize()?;
        volume += points[0]
            .as_dvec3()
            .dot(points[1].as_dvec3().cross(points[2].as_dvec3()))
            / 6.0;
        let points = points.map(Vec3::as_dvec3);
        let normal = normal.as_dvec3();
        faces.push(WernerSurfaceFace {
            p0: points[0],
            p1: points[1],
            p2: points[2],
            normal,
        });
        for edge_index in 0..3 {
            let next = (edge_index + 1) % 3;
            let edge_direction = (points[next] - points[edge_index]).normalize();
            let key = if indices[edge_index] < indices[next] {
                (indices[edge_index], indices[next])
            } else {
                (indices[next], indices[edge_index])
            };
            edge_sides
                .entry(key)
                .or_default()
                .push(WernerSurfaceEdgeSide {
                    normal,
                    edge_outward: edge_direction.cross(normal),
                });
        }
    }
    if faces.is_empty() || volume <= f64::EPSILON {
        return None;
    }
    let mut edges = Vec::new();
    for ((first, second), sides) in edge_sides {
        if sides.len() != 2 {
            continue;
        }
        let p0 = (*topology.positions.get(first as usize)? * scale).as_dvec3();
        let p1 = (*topology.positions.get(second as usize)? * scale).as_dvec3();
        let tensor = outer_product_d(sides[0].normal, sides[0].edge_outward)
            + outer_product_d(sides[1].normal, sides[1].edge_outward);
        edges.push(WernerSurfaceEdge {
            p0,
            p1,
            tensor_rows: tensor.transpose(),
            length: (p1 - p0).length(),
        });
    }
    (!edges.is_empty()).then_some(WernerSurfaceEvaluator {
        edges,
        faces,
        g_density: G as f64 * RYUGU_MASS as f64 / volume,
    })
}

fn outer_product_d(left: DVec3, right: DVec3) -> DMat3 {
    DMat3::from_cols(left * right.x, left * right.y, left * right.z)
}

fn build_evaluator(
    method: ActiveGravityMethod,
    sources: Vec<PointMass>,
    topology: &AsteroidTopologyGpuData,
    scale: f32,
    density_mode: DensityMode,
) -> SurfaceEvaluator {
    match method {
        ActiveGravityMethod::MmfftCompressed => build_fft_evaluator(sources, FFT_GRID_SIZE),
        ActiveGravityMethod::Fmm => SurfaceEvaluator::Fmm(FmmSurfaceEvaluator {
            tree: CpuFmmTree::new(sources),
        }),
        ActiveGravityMethod::HomogeneousWerner if density_mode == DensityMode::Constant => {
            build_werner_evaluator(topology, scale)
                .map(SurfaceEvaluator::Werner)
                .unwrap_or_else(|| SurfaceEvaluator::Point(PointMassEvaluator { sources }))
        }
        ActiveGravityMethod::HomogeneousWerner => {
            SurfaceEvaluator::Point(PointMassEvaluator { sources })
        }
        ActiveGravityMethod::RadialAnalytic => {
            let radial_sources = build_radial_surface_sources(topology, scale)
                .filter(|sources| !sources.is_empty())
                .unwrap_or(sources);
            SurfaceEvaluator::Point(PointMassEvaluator {
                sources: radial_sources,
            })
        }
        // Surface fields use the Eq.106 inverse-pole spatial operator. Eq.184
        // remains exclusively a trajectory transform, never a coarse FFT alias.
        ActiveGravityMethod::FrequencyDomain => {
            let radius = topology
                .positions
                .iter()
                .map(|p| f64::from(p.length() * scale))
                .fold(1.0_f64, f64::max);
            let modes = (0..EQ184_QUADRATURE_COUNT)
                .filter_map(|index| {
                    let (k, weight) = eq184_quadrature_node(index, radius)?;
                    let density = sources
                        .iter()
                        .fold(Complex64::new(0.0, 0.0), |sum, source| {
                            sum + Complex64::from_polar(source.mass, -k.dot(source.position))
                        });
                    let coefficient = f64::from(G) * weight
                        / (2.0 * std::f64::consts::PI.powi(2) * k.length_squared());
                    Some((k, density * coefficient))
                })
                .collect();
            SurfaceEvaluator::Equation106(modes)
        }
    }
}

fn build_fft_evaluator(sources: Vec<PointMass>, n: usize) -> SurfaceEvaluator {
    let records = sources
        .iter()
        .map(|source| (source.position, source.mass))
        .collect::<Vec<_>>();
    let mut workspace = MmfftLevelWorkspace::new(n, FFT_HALF_EXTENT);
    let field = workspace.build(&records).to_vec();
    SurfaceEvaluator::Fft(FftSurfaceEvaluator {
        field,
        n,
        half_extent: FFT_HALF_EXTENT as f32,
    })
}

fn evaluate_patch(evaluator: &SurfaceEvaluator, patch: SurfaceFieldPatch) -> SurfaceFieldSample {
    let value = evaluator.field_at(patch.body_position);
    let jacobian = if matches!(
        evaluator,
        SurfaceEvaluator::Point(_) | SurfaceEvaluator::Equation106(_)
    ) {
        value.jacobian
    } else {
        finite_difference_jacobian(evaluator, patch.body_position)
    };
    let effective = value.gravity + centrifugal_acceleration(patch.body_position);
    let outward_force = (-effective).normalize_or_zero();
    let alignment = outward_force.dot(patch.normal).clamp(-1.0, 1.0);
    SurfaceFieldSample {
        position: patch.body_position,
        normal: patch.normal,
        gravity: value.gravity,
        effective_gravity: effective,
        gravity_magnitude: value.gravity.length(),
        effective_gravity_magnitude: effective.length(),
        gradient_magnitude: frobenius_norm(jacobian),
        slope_degrees: alignment.acos().to_degrees(),
    }
}

fn centrifugal_acceleration(position: Vec3) -> Vec3 {
    let omega =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    -omega.cross(omega.cross(position))
}

fn finite_difference_jacobian(evaluator: &SurfaceEvaluator, position: Vec3) -> DMat3 {
    let h = evaluator.derivative_step();
    let mut columns = [DVec3::ZERO; 3];
    for (axis, column) in columns.iter_mut().enumerate() {
        let direction = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };
        let plus = evaluator
            .field_at(position + direction * h)
            .gravity
            .as_dvec3();
        let minus = evaluator
            .field_at(position - direction * h)
            .gravity
            .as_dvec3();
        *column = (plus - minus) / (2.0 * h as f64);
    }
    DMat3::from_cols(columns[0], columns[1], columns[2])
}

impl SurfaceEvaluator {
    fn field_at(&self, position: Vec3) -> FieldValue {
        match self {
            Self::Point(evaluator) => evaluator.field_at(position),
            Self::Fft(evaluator) => evaluator.field_at(position),
            Self::Fmm(evaluator) => evaluator.field_at(position),
            Self::Werner(evaluator) => evaluator.field_at(position),
            Self::Equation106(modes) => {
                let mut field = FieldValue::default();
                let mut gravity = DVec3::ZERO;
                let mut potential = 0.0;
                for (k, density) in modes {
                    let value = *density * Complex64::from_polar(1.0, k.dot(position.as_dvec3()));
                    gravity -= value.im * *k;
                    potential += value.re;
                    field.jacobian -= DMat3::from_cols(*k * k.x, *k * k.y, *k * k.z) * value.re;
                }
                field.gravity = gravity.as_vec3();
                field.potential = potential as f32;
                field
            }
        }
    }

    fn derivative_step(&self) -> f32 {
        match self {
            Self::Point(_) | Self::Fmm(_) | Self::Werner(_) | Self::Equation106(_) => 0.5,
            Self::Fft(evaluator) => 0.5 * 2.0 * evaluator.half_extent / evaluator.n as f32,
        }
    }
}

impl PointMassEvaluator {
    fn field_at(&self, position: Vec3) -> FieldValue {
        let observer = position.as_dvec3();
        let mut result = FieldValue::default();
        for source in &self.sources {
            let displacement = source.position - observer;
            let radius_squared = displacement.length_squared().max(1.0e-12);
            let inverse_radius = radius_squared.sqrt().recip();
            let inverse_radius3 = inverse_radius / radius_squared;
            let factor = G as f64 * source.mass;
            let gravity = factor * displacement * inverse_radius3;
            let identity = DMat3::IDENTITY * inverse_radius3;
            let outer = DMat3::from_cols(
                displacement * displacement.x,
                displacement * displacement.y,
                displacement * displacement.z,
            );
            result.gravity += gravity.as_vec3();
            result.potential += (factor * inverse_radius) as f32;
            let contribution =
                factor * (outer * (3.0 * inverse_radius3 / radius_squared) - identity);
            result.jacobian += contribution;
        }
        result
    }
}

impl FftSurfaceEvaluator {
    fn field_at(&self, position: Vec3) -> FieldValue {
        let gravity = sample_mmfft_grid(&self.field, position, self.half_extent, self.n);
        let potential = sample_mmfft_potential(&self.field, position, self.half_extent, self.n);
        FieldValue {
            gravity,
            potential,
            ..default()
        }
    }
}

impl WernerSurfaceEvaluator {
    fn field_at(&self, position: Vec3) -> FieldValue {
        let observer = position.as_dvec3();
        let mut gravity = DVec3::ZERO;
        let mut potential = 0.0_f64;
        for edge in &self.edges {
            let r0 = edge.p0 - observer;
            let r1 = edge.p1 - observer;
            let length0 = r0.length();
            let length1 = r1.length();
            let denominator =
                (length0 + length1 - edge.length).max(1.0e-6 * (length0 + length1).max(1.0));
            let logarithm = ((length0 + length1 + edge.length) / denominator)
                .max(1.0)
                .ln();
            let tensor_r = edge.tensor_rows * r0;
            gravity -= tensor_r * logarithm;
            potential += 0.5 * r0.dot(tensor_r) * logarithm;
        }
        for face in &self.faces {
            let r0 = face.p0 - observer;
            let r1 = face.p1 - observer;
            let r2 = face.p2 - observer;
            let length0 = r0.length();
            let length1 = r1.length();
            let length2 = r2.length();
            let numerator = r0.dot(r1.cross(r2));
            let denominator = length0 * length1 * length2
                + length0 * r1.dot(r2)
                + length1 * r2.dot(r0)
                + length2 * r0.dot(r1);
            let solid_angle = 2.0 * numerator.atan2(denominator);
            let normal_distance = face.normal.dot(r0);
            gravity += face.normal * normal_distance * solid_angle;
            potential -= 0.5 * normal_distance * normal_distance * solid_angle;
        }
        let scale = self.g_density;
        FieldValue {
            gravity: (gravity * scale).as_vec3(),
            potential: (potential * scale) as f32,
            ..default()
        }
    }
}

impl CpuFmmTree {
    fn new(sources: Vec<PointMass>) -> Self {
        let indices = (0..sources.len()).collect::<Vec<_>>();
        let half = sources
            .iter()
            .map(|source| source.position.abs().max_element())
            .fold(1.0_f64, f64::max)
            * 1.01;
        let mut tree = Self {
            nodes: Vec::new(),
            sources,
        };
        tree.build_node(&indices, DVec3::ZERO, half, 0);
        tree
    }

    fn build_node(&mut self, indices: &[usize], center: DVec3, half: f64, level: u32) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(CpuFmmNode {
            half,
            mass: 0.0,
            center_of_mass: center,
            quadrupole: DMat3::ZERO,
            level,
            children: [None; 8],
            particle_indices: Vec::new(),
        });
        let (mass, center_of_mass, quadrupole) = self.moments(indices);
        self.nodes[node_index].mass = mass;
        self.nodes[node_index].center_of_mass = center_of_mass;
        self.nodes[node_index].quadrupole = quadrupole;
        if level >= FMM_MAX_LEVEL || indices.len() <= FMM_LEAF_LIMIT {
            self.nodes[node_index].particle_indices = indices.to_vec();
            return node_index;
        }
        let mut groups: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
        for &index in indices {
            let point = self.sources[index].position;
            let child = usize::from(point.x >= center.x)
                | (usize::from(point.y >= center.y) << 1)
                | (usize::from(point.z >= center.z) << 2);
            groups[child].push(index);
        }
        let child_half = half * 0.5;
        for (child_index, group) in groups.into_iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let offset = DVec3::new(
                if child_index & 1 == 0 {
                    -child_half
                } else {
                    child_half
                },
                if child_index & 2 == 0 {
                    -child_half
                } else {
                    child_half
                },
                if child_index & 4 == 0 {
                    -child_half
                } else {
                    child_half
                },
            );
            let child = self.build_node(&group, center + offset, child_half, level + 1);
            self.nodes[node_index].children[child_index] = Some(child);
        }
        node_index
    }

    fn moments(&self, indices: &[usize]) -> (f64, DVec3, DMat3) {
        let mut mass = 0.0;
        let mut first = DVec3::ZERO;
        let mut second = DMat3::ZERO;
        for &index in indices {
            let source = self.sources[index];
            mass += source.mass;
            first += source.position * source.mass;
            let position = source.position;
            second += DMat3::from_cols(
                position * position.x,
                position * position.y,
                position * position.z,
            ) * source.mass;
        }
        if mass <= f64::EPSILON {
            return (0.0, DVec3::ZERO, DMat3::ZERO);
        }
        let center_of_mass = first / mass;
        let central = second
            - DMat3::from_cols(
                center_of_mass * center_of_mass.x,
                center_of_mass * center_of_mass.y,
                center_of_mass * center_of_mass.z,
            ) * mass;
        let trace = central.x_axis.x + central.y_axis.y + central.z_axis.z;
        let quadrupole = central * 3.0 - DMat3::IDENTITY * trace;
        (mass, center_of_mass, quadrupole)
    }

    fn field_at(&self, position: Vec3) -> FieldValue {
        self.node_field(0, position.as_dvec3())
    }

    fn node_field(&self, index: usize, observer: DVec3) -> FieldValue {
        let node = &self.nodes[index];
        if node.mass <= 0.0 {
            return FieldValue::default();
        }
        let displacement = node.center_of_mass - observer;
        let distance = displacement.length().max(1.0e-9);
        let expansion_radius = 3.0_f64.sqrt() * node.half;
        let has_children = node.children.iter().any(Option::is_some);
        if has_children && node.level > 0 && expansion_radius / distance < FMM_THETA {
            return multipole_field(node, observer);
        }
        if !has_children {
            return node
                .particle_indices
                .iter()
                .map(|&source_index| point_mass_field(self.sources[source_index], observer))
                .fold(FieldValue::default(), add_field_values);
        }
        node.children
            .iter()
            .flatten()
            .map(|&child| self.node_field(child, observer))
            .fold(FieldValue::default(), add_field_values)
    }
}

impl FmmSurfaceEvaluator {
    fn field_at(&self, position: Vec3) -> FieldValue {
        self.tree.field_at(position)
    }
}

fn point_mass_field(source: PointMass, observer: DVec3) -> FieldValue {
    PointMassEvaluator {
        sources: vec![source],
    }
    .field_at(observer.as_vec3())
}

fn multipole_field(node: &CpuFmmNode, observer: DVec3) -> FieldValue {
    let displacement = node.center_of_mass - observer;
    let radius_squared = displacement.length_squared().max(1.0e-12);
    let inverse_radius = radius_squared.sqrt().recip();
    let inverse_radius3 = inverse_radius / radius_squared;
    let inverse_radius5 = inverse_radius3 / radius_squared;
    let qd = node.quadrupole * displacement;
    let scalar = displacement.dot(qd);
    let factor = G as f64;
    let gravity = factor
        * (node.mass * displacement * inverse_radius3 - qd * inverse_radius5
            + 2.5 * scalar * displacement * inverse_radius5 / radius_squared);
    FieldValue {
        gravity: gravity.as_vec3(),
        potential: (factor * (node.mass * inverse_radius + 0.5 * scalar * inverse_radius5)) as f32,
        ..default()
    }
}

fn add_field_values(left: FieldValue, right: FieldValue) -> FieldValue {
    FieldValue {
        gravity: left.gravity + right.gravity,
        potential: left.potential + right.potential,
        jacobian: left.jacobian + right.jacobian,
    }
}

fn sample_mmfft_potential(field: &[[f32; 4]], position: Vec3, half_extent: f32, n: usize) -> f32 {
    let spacing = 2.0 * half_extent / n as f32;
    let coordinate = (position + Vec3::splat(half_extent)) / spacing - Vec3::splat(0.5);
    let base_floor = coordinate
        .floor()
        .clamp(Vec3::ONE, Vec3::splat((n - 3) as f32));
    let fraction = (coordinate - base_floor).clamp(Vec3::ZERO, Vec3::ONE);
    let base = base_floor.as_uvec3() - UVec3::ONE;
    let weights = |t: f32| {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ]
    };
    let wx = weights(fraction.x);
    let wy = weights(fraction.y);
    let wz = weights(fraction.z);
    let mut potential = 0.0;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let index =
                    ((base.z as usize + z) * n + base.y as usize + y) * n + base.x as usize + x;
                potential += field[index][3] * wx[x] * wy[y] * wz[z];
            }
        }
    }
    potential
}

fn normalize_range(value: f32, range: (f32, f32)) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let span = range.1 - range.0;
    if !span.is_finite() || span.abs() <= f32::EPSILON {
        0.5
    } else {
        ((value - range.0) / span).clamp(0.0, 1.0)
    }
}

fn finite_range(range: (f32, f32)) -> (f32, f32) {
    if range.0.is_finite() && range.1.is_finite() {
        range
    } else {
        (0.0, 0.0)
    }
}

fn frobenius_norm(matrix: DMat3) -> f32 {
    (matrix.x_axis.length_squared()
        + matrix.y_axis.length_squared()
        + matrix.z_axis.length_squared())
    .sqrt() as f32
}

fn scientific_color(t: f32, metric: SurfaceFieldMetric) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let (low, middle, high) = match metric {
        SurfaceFieldMetric::Gravity => (
            Vec3::new(0.05, 0.12, 0.62),
            Vec3::new(0.04, 0.78, 0.92),
            Vec3::new(1.0, 0.82, 0.10),
        ),
        SurfaceFieldMetric::Gradient => (
            Vec3::new(0.15, 0.05, 0.52),
            Vec3::new(0.64, 0.22, 0.84),
            Vec3::new(1.0, 0.62, 0.08),
        ),
        SurfaceFieldMetric::Slope | SurfaceFieldMetric::Error => (
            Vec3::new(0.02, 0.26, 0.55),
            Vec3::new(0.05, 0.82, 0.72),
            Vec3::new(1.0, 0.45, 0.08),
        ),
    };
    let rgb = if t < 0.5 {
        low.lerp(middle, t * 2.0)
    } else {
        middle.lerp(high, (t - 0.5) * 2.0)
    };
    [rgb.x, rgb.y, rgb.z, 0.92]
}

fn diverging_error_color(value: f32) -> [f32; 4] {
    let value = value.clamp(-1.0, 1.0);
    let neutral = Vec3::new(0.96, 0.96, 0.92);
    let rgb = if value < 0.0 {
        Vec3::new(0.08, 0.28, 0.92).lerp(neutral, value + 1.0)
    } else {
        neutral.lerp(Vec3::new(0.92, 0.12, 0.08), value)
    };
    [rgb.x, rgb.y, rgb.z, 0.94]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_mass_gradient_has_expected_trace_free_structure() {
        let evaluator = PointMassEvaluator {
            sources: vec![PointMass {
                position: DVec3::ZERO,
                mass: 1.0,
            }],
        };
        let value = evaluator.field_at(Vec3::new(2.0, 0.0, 0.0));
        let trace = value.jacobian.x_axis.x + value.jacobian.y_axis.y + value.jacobian.z_axis.z;
        assert!(value.gravity.is_finite());
        assert!(trace.abs() < 1.0e-18);
    }

    #[test]
    fn error_palette_separates_positive_and_negative() {
        assert_ne!(diverging_error_color(-0.8), diverging_error_color(0.8));
    }
}
