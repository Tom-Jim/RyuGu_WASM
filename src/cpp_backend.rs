//! Bevy client for the independent Rust backend and its C++ numerical module.
use crate::interface::components::*;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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

macro_rules! backend_channel {
    ($snapshot:ident, $packet:ident, $channel:ident, $value:ty) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $snapshot {
            pub request_id: u64,
            pub epoch: u64,
        }

        // C2 installs every semantic channel before C3 migrates its consumer.
        // Keep intermediate commits warning-free while preserving that order.
        #[allow(dead_code)]
        #[derive(Clone, Debug)]
        pub struct $packet {
            pub snapshot: $snapshot,
            pub result: Result<$value, String>,
        }

        #[allow(dead_code)]
        #[derive(Resource, Clone)]
        pub struct $channel {
            pub data: Arc<Mutex<Option<$packet>>>,
            pub in_flight: Arc<AtomicBool>,
            pub snapshot: Arc<Mutex<Option<$snapshot>>>,
        }

        impl Default for $channel {
            fn default() -> Self {
                Self {
                    data: Arc::new(Mutex::new(None)),
                    in_flight: Arc::new(AtomicBool::new(false)),
                    snapshot: Arc::new(Mutex::new(None)),
                }
            }
        }

        #[allow(dead_code)]
        impl $channel {
            pub fn begin(&self, snapshot: $snapshot) -> bool {
                if self.in_flight.swap(true, Ordering::AcqRel) {
                    return false;
                }
                *self.data.lock().expect("backend result channel poisoned") = None;
                *self
                    .snapshot
                    .lock()
                    .expect("backend snapshot channel poisoned") = Some(snapshot);
                true
            }

            pub fn complete(&self, packet: $packet) -> bool {
                let expected = *self
                    .snapshot
                    .lock()
                    .expect("backend snapshot channel poisoned");
                if expected != Some(packet.snapshot) {
                    return false;
                }
                *self.data.lock().expect("backend result channel poisoned") = Some(packet);
                self.in_flight.store(false, Ordering::Release);
                true
            }

            pub fn reset(&self) {
                *self.data.lock().expect("backend result channel poisoned") = None;
                *self
                    .snapshot
                    .lock()
                    .expect("backend snapshot channel poisoned") = None;
                self.in_flight.store(false, Ordering::Release);
            }
        }
    };
}

backend_channel!(
    BackendAdvanceSnapshot,
    BackendAdvancePacket,
    BackendAdvanceChannel,
    Vec<f64>
);
backend_channel!(
    BackendEvaluateSourcesSnapshot,
    BackendEvaluateSourcesPacket,
    BackendEvaluateSourcesChannel,
    Vec<f64>
);
backend_channel!(
    BackendCandidatesSnapshot,
    BackendCandidatesPacket,
    BackendCandidatesChannel,
    Vec<f64>
);
backend_channel!(
    BackendDensitySnapshot,
    BackendDensityPacket,
    BackendDensityChannel,
    Vec<f32>
);
backend_channel!(
    BackendEvaluateSnapshot,
    BackendEvaluatePacket,
    BackendEvaluateChannel,
    Vec<f64>
);
backend_channel!(
    BackendConfigureSnapshot,
    BackendConfigurePacket,
    BackendConfigureChannel,
    ()
);

/// Every numerical-worker channel in one system parameter.
///
/// Cancelling an experiment (method switch, probe edit, crash reset) must never
/// leave a channel stuck with `in_flight` set, otherwise no later request can
/// ever be submitted on it.
#[derive(bevy::ecs::system::SystemParam)]
pub struct BackendChannels<'w> {
    pub advance: Res<'w, BackendAdvanceChannel>,
    pub evaluate_sources: Res<'w, BackendEvaluateSourcesChannel>,
    pub candidates: Res<'w, BackendCandidatesChannel>,
    pub density: Res<'w, BackendDensityChannel>,
    pub evaluate: Res<'w, BackendEvaluateChannel>,
    pub configure: Res<'w, BackendConfigureChannel>,
}

impl BackendChannels<'_> {
    pub fn reset_all(&self) {
        self.advance.reset();
        self.evaluate_sources.reset();
        self.candidates.reset();
        self.density.reset();
        self.evaluate.reset();
        self.configure.reset();
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct BackendDeliveryChannels {
    advance: BackendAdvanceChannel,
    evaluate_sources: BackendEvaluateSourcesChannel,
    candidates: BackendCandidatesChannel,
    density: BackendDensityChannel,
    evaluate: BackendEvaluateChannel,
    configure: BackendConfigureChannel,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static BACKEND_DELIVERY_CHANNELS: std::cell::RefCell<Option<BackendDeliveryChannels>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
fn install_delivery_channels(channels: BackendDeliveryChannels) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| *slot.borrow_mut() = Some(channels));
}

#[cfg(target_arch = "wasm32")]
fn delivered_result<T>(value: T, error: Option<String>) -> Result<T, String> {
    error.map_or_else(|| Ok(value), Err)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_advance_result(
    request_id: u64,
    epoch: u64,
    value: Vec<f64>,
    error: Option<String>,
) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| {
        if let Some(channels) = slot.borrow().as_ref() {
            channels.advance.complete(BackendAdvancePacket {
                snapshot: BackendAdvanceSnapshot { request_id, epoch },
                result: delivered_result(value, error),
            });
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_evaluate_sources_result(
    request_id: u64,
    epoch: u64,
    value: Vec<f64>,
    error: Option<String>,
) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| {
        if let Some(channels) = slot.borrow().as_ref() {
            channels
                .evaluate_sources
                .complete(BackendEvaluateSourcesPacket {
                    snapshot: BackendEvaluateSourcesSnapshot { request_id, epoch },
                    result: delivered_result(value, error),
                });
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_candidates_result(
    request_id: u64,
    epoch: u64,
    value: Vec<f64>,
    error: Option<String>,
) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| {
        if let Some(channels) = slot.borrow().as_ref() {
            channels.candidates.complete(BackendCandidatesPacket {
                snapshot: BackendCandidatesSnapshot { request_id, epoch },
                result: delivered_result(value, error),
            });
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_density_result(
    request_id: u64,
    epoch: u64,
    value: Vec<f32>,
    error: Option<String>,
) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| {
        if let Some(channels) = slot.borrow().as_ref() {
            channels.density.complete(BackendDensityPacket {
                snapshot: BackendDensitySnapshot { request_id, epoch },
                result: delivered_result(value, error),
            });
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_evaluate_result(
    request_id: u64,
    epoch: u64,
    value: Vec<f64>,
    error: Option<String>,
) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| {
        if let Some(channels) = slot.borrow().as_ref() {
            channels.evaluate.complete(BackendEvaluatePacket {
                snapshot: BackendEvaluateSnapshot { request_id, epoch },
                result: delivered_result(value, error),
            });
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_configure_result(request_id: u64, epoch: u64, error: Option<String>) {
    BACKEND_DELIVERY_CHANNELS.with(|slot| {
        if let Some(channels) = slot.borrow().as_ref() {
            channels.configure.complete(BackendConfigurePacket {
                snapshot: BackendConfigureSnapshot { request_id, epoch },
                result: delivered_result((), error),
            });
        }
    });
}

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
export function backend_candidates(data) {
    return globalThis.ryuguRustBackend.propagate_candidates(data);
}
export function backend_ready() {
    return Boolean(globalThis.ryuguRustBackend && globalThis.ryuguCpp && globalThis.ryuguScheduler);
}
export function backend_worker_ready() {
    return Boolean(globalThis.ryuguNumericalWorkerReady && globalThis.ryuguBackendClient?.isReady());
}
function post_backend_request(kind, requestId, epoch, payload) {
    if (!globalThis.ryuguBackendClient?.request(kind, requestId, epoch, payload)) {
        throw new Error('Numerical Worker is not ready');
    }
}
export function request_backend_advance(requestId, epoch, method, initial, step, steps, history) {
    post_backend_request('advance', requestId, epoch,
        { epoch, method, initial, step, steps, history });
}
export function request_backend_sources(requestId, epoch, method, xyz, masses, targets) {
    post_backend_request('evaluate_sources', requestId, epoch,
        { method, xyz, masses, targets });
}
export function request_backend_candidates(requestId, epoch, data) {
    post_backend_request('propagate_candidates', requestId, epoch, { data });
}
export function request_backend_density(requestId, epoch, data) {
    post_backend_request('solve_density', requestId, epoch, { data });
}
export function request_backend_evaluate(requestId, epoch, method, x, y, z) {
    post_backend_request('evaluate', requestId, epoch, { method, x, y, z });
}
export function request_backend_configure(requestId, epoch, cells, vertices, facets, mass) {
    post_backend_request('configure', requestId, epoch, { cells, vertices, facets, mass });
}
"#)]
extern "C" {
    #[wasm_bindgen]
    fn backend_ready() -> bool;
    #[wasm_bindgen]
    fn backend_worker_ready() -> bool;
    #[wasm_bindgen(catch)]
    fn backend_candidates(data: &str) -> Result<Vec<f64>, JsValue>;
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
    fn request_backend_advance(
        request_id: u64,
        epoch: u64,
        method: &str,
        initial: &[f64],
        step: f64,
        steps: u32,
        history: &[f64],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_sources(
        request_id: u64,
        epoch: u64,
        method: &str,
        xyz: &[f64],
        masses: &[f64],
        targets: &[f64],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_candidates(request_id: u64, epoch: u64, data: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_density(request_id: u64, epoch: u64, data: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_evaluate(
        request_id: u64,
        epoch: u64,
        method: &str,
        x: f64,
        y: f64,
        z: f64,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_configure(
        request_id: u64,
        epoch: u64,
        cells: &[f64],
        vertices: &[f64],
        facets: &[u32],
        mass: f64,
    ) -> Result<(), JsValue>;
}

#[allow(dead_code)]
pub fn request_advance(
    channel: &BackendAdvanceChannel,
    snapshot: BackendAdvanceSnapshot,
    method: ActiveGravityMethod,
    initial: &[f64],
    step: f64,
    steps: u32,
    history: &[f64],
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        if let Err(error) = request_backend_advance(
            snapshot.request_id,
            snapshot.epoch,
            crate::basilisk::BasiliskAlgorithm::from_active(method).key(),
            initial,
            step,
            steps,
            history,
        ) {
            channel.reset();
            return Err(format!("Simulation Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, method, initial, step, steps, history);
        Ok(false)
    }
}

#[allow(dead_code)]
pub fn request_evaluate_sources(
    channel: &BackendEvaluateSourcesChannel,
    snapshot: BackendEvaluateSourcesSnapshot,
    method: &str,
    sources: &[(bevy::math::DVec3, f64)],
    targets: &[bevy::math::DVec3],
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        let xyz: Vec<f64> = sources.iter().flat_map(|(p, _)| p.to_array()).collect();
        let masses: Vec<f64> = sources.iter().map(|(_, mass)| *mass).collect();
        let positions: Vec<f64> = targets
            .iter()
            .flat_map(|position| position.to_array())
            .collect();
        if let Err(error) = request_backend_sources(
            snapshot.request_id,
            snapshot.epoch,
            method,
            &xyz,
            &masses,
            &positions,
        ) {
            channel.reset();
            return Err(format!("Source-evaluation Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, method, sources, targets);
        Ok(false)
    }
}

#[allow(dead_code)]
pub fn request_candidates(
    channel: &BackendCandidatesChannel,
    snapshot: BackendCandidatesSnapshot,
    data: &str,
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        if let Err(error) = request_backend_candidates(snapshot.request_id, snapshot.epoch, data) {
            channel.reset();
            return Err(format!("Candidate Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, data);
        Ok(false)
    }
}

#[allow(dead_code)]
pub fn request_density(
    channel: &BackendDensityChannel,
    snapshot: BackendDensitySnapshot,
    data: &str,
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        if let Err(error) = request_backend_density(snapshot.request_id, snapshot.epoch, data) {
            channel.reset();
            return Err(format!("Density Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, data);
        Ok(false)
    }
}

#[allow(dead_code)]
pub fn request_evaluate(
    channel: &BackendEvaluateChannel,
    snapshot: BackendEvaluateSnapshot,
    method: ActiveGravityMethod,
    position: Vec3,
) -> Result<bool, String> {
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
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        if let Err(error) = request_backend_evaluate(
            snapshot.request_id,
            snapshot.epoch,
            key,
            position.x as f64,
            position.y as f64,
            position.z as f64,
        ) {
            channel.reset();
            return Err(format!("Field-evaluation Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, method, position);
        Ok(false)
    }
}

#[allow(dead_code)]
pub fn request_configure(
    channel: &BackendConfigureChannel,
    snapshot: BackendConfigureSnapshot,
    cells: &[f64],
    vertices: &[f64],
    facets: &[u32],
    mass: f64,
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        if let Err(error) = request_backend_configure(
            snapshot.request_id,
            snapshot.epoch,
            cells,
            vertices,
            facets,
            mass,
        ) {
            channel.reset();
            return Err(format!("Geometry Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, cells, vertices, facets, mass);
        Ok(false)
    }
}

#[allow(dead_code)]
pub fn propagate_candidates(data: &str) -> Result<Vec<f64>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        backend_candidates(data).map_err(|e| format!("Candidate batch backend: {e:?}"))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = data;
        Err("Candidate backend requires WASM".into())
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

/// Validates the four-value `[gravity, potential]` field response.
///
/// Both the synchronous adapter and the asynchronous Worker delivery path must
/// accept exactly the same payload, so the contract lives in one place.
pub fn decode_field_response(values: Vec<f64>) -> Result<(Vec3, f32), String> {
    if values.len() != 4 || !values.iter().all(|value| value.is_finite()) {
        return Err("Invalid C++ gravity response".into());
    }
    Ok((
        Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32),
        values[3] as f32,
    ))
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
        decode_field_response(value)
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
    pub worker_ready: bool,
    source_key: Option<(u64, DensityMode)>,
    worker_source_key: Option<(u64, DensityMode)>,
    worker_configuration_request_id: u64,
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
        let advance = BackendAdvanceChannel::default();
        let evaluate_sources = BackendEvaluateSourcesChannel::default();
        let candidates = BackendCandidatesChannel::default();
        let density = BackendDensityChannel::default();
        let evaluate = BackendEvaluateChannel::default();
        let configure = BackendConfigureChannel::default();
        #[cfg(target_arch = "wasm32")]
        install_delivery_channels(BackendDeliveryChannels {
            advance: advance.clone(),
            evaluate_sources: evaluate_sources.clone(),
            candidates: candidates.clone(),
            density: density.clone(),
            evaluate: evaluate.clone(),
            configure: configure.clone(),
        });
        app.init_resource::<CppBackendState>()
            .insert_resource(advance)
            .insert_resource(evaluate_sources)
            .insert_resource(candidates)
            .insert_resource(density)
            .insert_resource(evaluate)
            .insert_resource(configure)
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
    worker_channel: Res<BackendConfigureChannel>,
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
    if state.source_key != Some(key) {
        state.ready = false;
    }
    if state.worker_source_key != Some(key) {
        state.worker_ready = false;
    }
    let completed_configuration = worker_channel
        .data
        .lock()
        .expect("backend configure result channel poisoned")
        .take()
        .filter(|packet| {
            packet.snapshot.epoch == source.source_hash
                && packet.snapshot.request_id == state.worker_configuration_request_id
        });
    if let Some(packet) = completed_configuration {
        match packet.result {
            Ok(()) => {
                state.worker_source_key = Some(key);
                state.worker_ready = true;
            }
            Err(message) => error.raise(format!("Worker geometry configuration failed: {message}")),
        }
    }
    if error.is_active() || (state.source_key == Some(key) && state.worker_source_key == Some(key))
    {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let worker_can_accept = backend_worker_ready()
        && !worker_channel.in_flight.load(Ordering::Acquire)
        && state.worker_source_key != Some(key);
    #[cfg(not(target_arch = "wasm32"))]
    let worker_can_accept = false;
    if state.source_key == Some(key) && !worker_can_accept {
        return;
    }
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
    if state.source_key != Some(key) {
        #[cfg(target_arch = "wasm32")]
        let result = cpp_configure(
            &live_cells,
            &vertices,
            &topology.triangles,
            RYUGU_MASS as f64,
        )
        .map_err(|e| format!("C++ geometry configuration failed: {e:?}"));
        #[cfg(not(target_arch = "wasm32"))]
        let result: Result<(), String> = Err("C++ browser host is unavailable".into());
        match result {
            Ok(()) => {
                state.source_key = Some(key);
                state.ready = true;
            }
            Err(message) => {
                error.raise(message);
                return;
            }
        }
    }
    if worker_can_accept {
        state.worker_configuration_request_id =
            state.worker_configuration_request_id.wrapping_add(1);
        let snapshot = BackendConfigureSnapshot {
            request_id: state.worker_configuration_request_id,
            epoch: source.source_hash,
        };
        match request_configure(
            &worker_channel,
            snapshot,
            &live_cells,
            &vertices,
            &topology.triangles,
            RYUGU_MASS as f64,
        ) {
            Ok(true) => {}
            Ok(false) => state.worker_ready = false,
            Err(message) => error.raise(message),
        }
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

#[cfg(test)]
mod backend_channel_tests {
    use super::*;

    #[test]
    fn reset_rejects_old_completion_without_unlocking_new_request() {
        let channel = BackendAdvanceChannel::default();
        let old = BackendAdvanceSnapshot {
            request_id: 41,
            epoch: 7,
        };
        let current = BackendAdvanceSnapshot {
            request_id: 42,
            epoch: 8,
        };
        assert!(channel.begin(old));
        channel.reset();
        assert!(channel.begin(current));

        assert!(!channel.complete(BackendAdvancePacket {
            snapshot: old,
            result: Ok(vec![1.0]),
        }));
        assert!(channel.in_flight.load(Ordering::Acquire));
        assert!(channel.data.lock().unwrap().is_none());

        assert!(channel.complete(BackendAdvancePacket {
            snapshot: current,
            result: Ok(vec![2.0]),
        }));
        assert!(!channel.in_flight.load(Ordering::Acquire));
        let packet = channel.data.lock().unwrap().take().unwrap();
        assert_eq!(packet.snapshot, current);
        assert_eq!(packet.result.unwrap(), vec![2.0]);
    }

    #[test]
    fn cancelling_an_experiment_clears_every_worker_channel() {
        let mut app = App::new();
        app.init_resource::<BackendAdvanceChannel>()
            .init_resource::<BackendEvaluateSourcesChannel>()
            .init_resource::<BackendCandidatesChannel>()
            .init_resource::<BackendDensityChannel>()
            .init_resource::<BackendEvaluateChannel>()
            .init_resource::<BackendConfigureChannel>()
            .add_systems(Update, |channels: BackendChannels| channels.reset_all());

        {
            let world = app.world_mut();
            for in_flight in [
                world.resource::<BackendAdvanceChannel>().in_flight.clone(),
                world
                    .resource::<BackendEvaluateSourcesChannel>()
                    .in_flight
                    .clone(),
                world
                    .resource::<BackendCandidatesChannel>()
                    .in_flight
                    .clone(),
                world.resource::<BackendDensityChannel>().in_flight.clone(),
                world.resource::<BackendEvaluateChannel>().in_flight.clone(),
                world
                    .resource::<BackendConfigureChannel>()
                    .in_flight
                    .clone(),
            ] {
                in_flight.store(true, Ordering::Release);
            }
        }

        app.update();

        let world = app.world();
        for in_flight in [
            world
                .resource::<BackendAdvanceChannel>()
                .in_flight
                .load(Ordering::Acquire),
            world
                .resource::<BackendEvaluateSourcesChannel>()
                .in_flight
                .load(Ordering::Acquire),
            world
                .resource::<BackendCandidatesChannel>()
                .in_flight
                .load(Ordering::Acquire),
            world
                .resource::<BackendDensityChannel>()
                .in_flight
                .load(Ordering::Acquire),
            world
                .resource::<BackendEvaluateChannel>()
                .in_flight
                .load(Ordering::Acquire),
            world
                .resource::<BackendConfigureChannel>()
                .in_flight
                .load(Ordering::Acquire),
        ] {
            assert!(!in_flight);
        }
        assert!(
            world
                .resource::<BackendAdvanceChannel>()
                .snapshot
                .lock()
                .unwrap()
                .is_none()
        );
    }
}
