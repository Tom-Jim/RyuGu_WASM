
use crate::cpu::frequency_domain::{
    AggregatedGravitySource, EQ184_QUADRATURE_COUNT, eq184_quadrature_node,
};
use crate::interface::components::*;
use bevy::asset::RenderAssetUsages;
use bevy::math::{DMat3, DVec3};
use bevy::mesh::PrimitiveTopology;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::mesh::Indices;
use num_complex::Complex64;

const SURFACE_PATCH_LIMIT: usize = 1_024;
const CONSTANT_SOURCE_LIMIT: usize = 8_192;
const SURFACE_COMPUTE_CHUNK: usize = 96;
const SURFACE_COMPUTE_BUDGET_MS: f32 = 3.0;
const SURFACE_EXPENSIVE_CHUNK: usize = 8;

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
    evaluator: Option<SurfaceEvaluator>,
    datasets: Vec<SurfaceFieldDataset>,
}

enum SurfaceEvaluator {
    Cpp(ActiveGravityMethod),
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
        evaluator: None,
        datasets: Vec::new(),
    });
}
