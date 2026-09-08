//! Bevy client for the independent Rust backend and its C++ numerical module.
use crate::interface::components::*;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::time::Duration;

// Live propagation prioritizes browser responsiveness. Batch validation and
// surface products retain their requested source resolution.
const MAX_LIVE_ANGULAR_CELLS: usize = 32;
const MIN_BACKEND_FRAME_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Resource, Default)]
pub struct WernerAcceleration(pub Vec3);
#[derive(Resource, Default)]
pub struct WernerPotential(pub Option<f32>);
pub type WernerReadbackChannel = GravityReadbackChannel;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = r#"
export function cpp_configure(cells, vertices, facets, mass) {
    if (!globalThis.ryuguRustBackend) throw new Error('Rust WASM backend is not loaded');
    globalThis.ryuguRustBackend.configure(cells, vertices, facets, mass);
}
export function cpp_evaluate(method, x, y, z) {
    return globalThis.ryuguRustBackend.evaluate(method, x, y, z);
}
export function cpp_sources(method, xyz, masses, targets) {
    return globalThis.ryuguRustBackend.evaluate_sources(method, xyz, masses, targets);
}
export function backend_advance(epoch, method, initial, step, steps, history) {
    return globalThis.ryuguRustBackend.advance_frame(epoch, method, initial, step, steps, history);
}
export function backend_solve_density(data) {
    return globalThis.ryuguRustBackend.solve_density(data);
}
export function backend_candidate(data) {
    return globalThis.ryuguRustBackend.propagate_candidate(data);
}
export function backend_ready() {
    return Boolean(globalThis.ryuguRustBackend && globalThis.ryuguCpp && globalThis.ryuguScheduler);
}
"#)]
extern "C" {
    #[wasm_bindgen]
    fn backend_ready() -> bool;
    #[wasm_bindgen(catch)]
    fn backend_candidate(data: &str) -> Result<Vec<f64>, JsValue>;
    #[wasm_bindgen(catch)]
    fn backend_solve_density(data: &str) -> Result<Vec<f32>, JsValue>;
    #[wasm_bindgen(catch)]
    fn cpp_configure(
        cells: &[f64],
        vertices: &[f64],
        facets: &[u32],
        mass: f64,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn cpp_evaluate(method: &str, x: f64, y: f64, z: f64) -> Result<Vec<f64>, JsValue>;
    #[wasm_bindgen(catch)]
    fn cpp_sources(
        method: &str,
        xyz: &[f64],
        masses: &[f64],
        targets: &[f64],
    ) -> Result<Vec<f64>, JsValue>;
    #[wasm_bindgen(catch)]
    fn backend_advance(
        epoch: u64,
        method: &str,
        initial: &[f64],
        step: f64,
        steps: u32,
        history: &[f64],
    ) -> Result<Vec<f64>, JsValue>;
}

pub fn propagate_candidate(data: &str) -> Result<Vec<f64>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        backend_candidate(data).map_err(|e| format!("Candidate backend: {e:?}"))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = data;
        Err("Candidate backend requires WASM".into())
    }
}

#[cfg(target_arch = "wasm32")]
pub fn solve_density(data: &str) -> Result<Vec<f32>, String> {
    backend_solve_density(data).map_err(|e| format!("Rust density backend: {e:?}"))
}

pub fn advance(
    epoch: u64,
    method: ActiveGravityMethod,
    initial: &[f64],
    step: f64,
    steps: u32,
    history: &[f64],
) -> Result<Vec<f64>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        backend_advance(
            epoch,
            crate::basilisk::BasiliskAlgorithm::from_active(method).key(),
            initial,
            step,
            steps,
            history,
        )
        .map_err(|e| format!("Simulation backend: {e:?}"))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (epoch, method, initial, step, steps, history);
        Err("The numerical backend requires WASM".into())
    }
}

pub fn evaluate_sources(
    method: &str,
    sources: &[(bevy::math::DVec3, f64)],
    targets: &[bevy::math::DVec3],
) -> Result<Vec<[f64; 4]>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        let xyz: Vec<f64> = sources.iter().flat_map(|(p, _)| p.to_array()).collect();
        let masses: Vec<f64> = sources.iter().map(|(_, m)| *m).collect();
        let positions: Vec<f64> = targets.iter().flat_map(|p| p.to_array()).collect();
        let values = cpp_sources(method, &xyz, &masses, &positions)
            .map_err(|e| format!("C++ source evaluation: {e:?}"))?;
        if values.len() != targets.len() * 4 || !values.iter().all(|v| v.is_finite()) {
            return Err("Invalid C++ batch response".into());
        }
        Ok(values.as_chunks::<4>().0.to_vec())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (method, sources, targets);
        Err("C++ browser host is unavailable".into())
    }
}

pub fn voxel_basis_sensitivities(
    method: &str,
    basis: &VoxelBasisSources,
    samples: &[TrajectoryInversionKnot],
) -> Result<Vec<Vec3>, String> {
    let targets: Vec<_> = samples
        .iter()
        .map(|s| (s.body_rotation.inverse() * s.position).as_dvec3())
        .collect();
    let mut result = vec![Vec3::ZERO; samples.len() * basis.columns.len()];
    for (column_index, column) in basis.columns.iter().enumerate() {
        let sources: Vec<_> = column.iter().map(|s| (s.position, s.volume)).collect();
        let values = evaluate_sources(method, &sources, &targets)?;
        for (i, value) in values.iter().enumerate() {
            result[i * basis.columns.len() + column_index] = samples[i].body_rotation
                * Vec3::new(value[0] as f32, value[1] as f32, value[2] as f32);
        }
    }
    Ok(result)
}

pub fn evaluate(method: ActiveGravityMethod, position: Vec3) -> Result<(Vec3, f32), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let key = match method {
            ActiveGravityMethod::RadialAnalytic => "radial",
            ActiveGravityMethod::HomogeneousWerner => "werner",
            ActiveGravityMethod::Fmm => "fmm",
            ActiveGravityMethod::MmfftCompressed => "fft",
            ActiveGravityMethod::FrequencyDomain => {
                return Err("Frequency-domain uses its dedicated operator".into());
            }
        };
        let value = cpp_evaluate(key, position.x as f64, position.y as f64, position.z as f64)
            .map_err(|error| format!("C++ {key}: {error:?}"))?;
        if value.len() != 4 || !value.iter().all(|v| v.is_finite()) {
            return Err("Invalid C++ gravity response".into());
        }
        Ok((
            Vec3::new(value[0] as f32, value[1] as f32, value[2] as f32),
            value[3] as f32,
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (method, position);
        Err("The C++ numerical backend requires the browser WASM host".into())
    }
}

#[derive(Resource, Default)]
pub struct CppBackendState {
    pub ready: bool,
    source_key: Option<(u64, DensityMode)>,
}

pub struct BackendFramePacer {
    last_update: Instant,
}

impl Default for BackendFramePacer {
    fn default() -> Self {
        Self {
            last_update: Instant::now() - MIN_BACKEND_FRAME_INTERVAL,
        }
    }
}

pub struct CppBackendPlugin;

impl Plugin for CppBackendPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CppBackendState>()
            .init_resource::<RadialGravityHistory>()
            .init_resource::<WernerGravityHistory>()
            .init_resource::<FmmGravityHistory>()
            .init_resource::<MmfftCompressedHistory>()
            .init_resource::<GravityReadbackChannel>()
            .init_resource::<WernerAcceleration>()
            .init_resource::<WernerPotential>()
            .init_resource::<FmmReadbackChannel>()
            .init_resource::<MmfftReadbackChannel>()
            .add_systems(
                Update,
                configure_backend.after(crate::cpu::density::build_density_quadrature_system),
            );
        app.add_systems(
            Update,
            crate::cpp_planning::dispatch
                .after(crate::bevy_app::backend::planning_batch_evaluator_system),
        );
    }
}

fn configure_backend(
    source: Option<Res<DensityQuadratureSource>>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    mode: Res<DensityMode>,
    ryugu: Query<&Transform, With<RyuguMarker>>,
    mut state: ResMut<CppBackendState>,
    mut error: ResMut<GravityRuntimeError>,
) {
    #[cfg(target_arch = "wasm32")]
    if !backend_ready() {
        return;
    }
    let (Some(source), Some(topology), Ok(transform)) = (source, topology, ryugu.single()) else {
        return;
    };
    let key = (source.source_hash, *mode);
    if state.source_key == Some(key) {
        return;
    }
    state.ready = false;
    let bytes = if *mode == DensityMode::Constant {
        &source.constant_bytes
    } else {
        &source.bytes
    };
    let cells: Vec<f64> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    let live_cells = reduce_live_cells(&cells);
    let vertices: Vec<f64> = topology
        .positions
        .iter()
        .flat_map(|p| (*p * transform.scale.x).to_array().map(f64::from))
        .collect();
    #[cfg(target_arch = "wasm32")]
    let result = cpp_configure(
        &live_cells,
        &vertices,
        &topology.triangles,
        RYUGU_MASS as f64,
    )
    .map_err(|e| format!("C++ geometry configuration failed: {e:?}"));
    #[cfg(not(target_arch = "wasm32"))]
    let result: Result<(), String> = {
        let _ = (live_cells, vertices);
        Err("C++ browser host is unavailable".into())
    };
    match result {
        Ok(()) => {
            state.source_key = Some(key);
            state.ready = true;
        }
        Err(message) => error.raise(message),
    }
}

fn reduce_live_cells(cells: &[f64]) -> Vec<f64> {
    let shells = cells.as_chunks::<8>().0;
    if shells.len() <= MAX_LIVE_ANGULAR_CELLS * 4 {
        return cells.to_vec();
    }
    let layers = 4;
    let angular = shells.len() / layers;
    let target = MAX_LIVE_ANGULAR_CELLS.min(angular);
    let stride = angular.div_ceil(target);
    let mut result = Vec::with_capacity(target * layers * 8);
    for angular_index in (0..angular).step_by(stride) {
        for layer in 0..layers {
            let mut shell = shells[angular_index * layers + layer];
            let represented = (angular - angular_index).min(stride);
            shell[3] *= represented as f64;
            result.extend(shell);
        }
    }
    result
}

pub fn should_advance_backend(pacer: &mut BackendFramePacer, method: ActiveGravityMethod) -> bool {
    let interval = match method {
        ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::FrequencyDomain => {
            MIN_BACKEND_FRAME_INTERVAL
        }
        ActiveGravityMethod::MmfftCompressed => Duration::from_millis(750),
        ActiveGravityMethod::HomogeneousWerner => Duration::from_secs(1),
        ActiveGravityMethod::Fmm => Duration::from_millis(1500),
    };
    let now = Instant::now();
    if now.duration_since(pacer.last_update) < interval {
        return false;
    }
    pacer.last_update = now;
    true
}
