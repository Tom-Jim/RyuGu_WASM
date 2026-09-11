
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
    /// Identity of the current job; Worker requests carry it as their epoch so
    /// a chunk answered for a cancelled or replaced job is never applied.
    job_id: u64,
    next_request_id: u64,
    pending: Option<PendingSurfaceChunk>,
}

struct SurfaceFieldJob {
    methods: Vec<ActiveGravityMethod>,
    method_index: usize,
    patch_index: usize,
    evaluator: Option<SurfaceEvaluator>,
    datasets: Vec<SurfaceFieldDataset>,
}

/// One chunk of surface patches whose stencil is being evaluated by the
/// numerical Worker.
struct PendingSurfaceChunk {
    snapshot: crate::cpp_backend::BackendSurfaceSnapshot,
    method_index: usize,
    start: usize,
    end: usize,
}

enum SurfaceEvaluator {
    /// Pointwise field of the geometry configured in the numerical Worker.
    Cpp(ActiveGravityMethod),
    /// Discrete Eq.121 IR/UV operator (residual modes + analytic IR monopole).
    Equation121 {
        modes: Vec<(DVec3, Complex64)>,
        center: DVec3,
        gravitational_parameter: f64,
    },
}

#[derive(Clone, Copy, Debug)]
struct PointMass {
    position: DVec3,
    mass: f64,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn queue_surface_field(
    state: &mut SurfaceFieldState,
    compute: &mut SurfaceFieldComputeState,
    method: ActiveGravityMethod,
) {
    compute.job = None;
    compute.job_id = compute.job_id.wrapping_add(1);
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

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn cancel_surface_field(
    state: &mut SurfaceFieldState,
    compute: &mut SurfaceFieldComputeState,
    status: &str,
) {
    compute.job = None;
    compute.job_id = compute.job_id.wrapping_add(1);
    state.computing = false;
    state.latest = None;
    state.comparison = None;
    state.selected_patch = None;
    state.revision = state.revision.wrapping_add(1);
    state.status = status.into();
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
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
    compute.job_id = compute.job_id.wrapping_add(1);
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
