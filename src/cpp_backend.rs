//! Bevy client for the independent Rust backend and its C++ numerical module.
//!
//! Every numeric WASM call leaves the main thread through one of the request →
//! poll channels below; the dedicated numerical Worker owns the only backend
//! instances in the page.
use crate::interface::components::*;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

// Live propagation prioritizes browser responsiveness. Batch validation and
// surface products retain their requested source resolution.
const MAX_LIVE_ANGULAR_CELLS: usize = 32;
// The numerical Worker is the rate limit for live propagation: a new advance
// request goes out as soon as the previous answer has been consumed. This floor
// only stops a very fast worker from turning the submit path into a busy poll.
const MIN_BACKEND_SUBMIT_INTERVAL: Duration = Duration::from_millis(16);

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

        // Native builds compile the channel plumbing but never call it: only
        // the wasm request/delivery path runs `begin`/`complete` and constructs
        // packets. Silence the host-only dead-code lint so the wasm target
        // still reports genuinely unused items.
        #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
        #[derive(Clone, Debug)]
        pub struct $packet {
            pub snapshot: $snapshot,
            pub result: Result<$value, String>,
        }

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

        #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
        impl $channel {
            // The slots hold plain `Option`s with no invariant that a panic
            // could break, so a poisoned lock is recovered instead of turning
            // one failure into a second panic.
            fn data_slot(&self) -> MutexGuard<'_, Option<$packet>> {
                self.data.lock().unwrap_or_else(PoisonError::into_inner)
            }

            fn snapshot_slot(&self) -> MutexGuard<'_, Option<$snapshot>> {
                self.snapshot.lock().unwrap_or_else(PoisonError::into_inner)
            }

            pub fn begin(&self, snapshot: $snapshot) -> bool {
                if self.in_flight.swap(true, Ordering::AcqRel) {
                    return false;
                }
                *self.data_slot() = None;
                *self.snapshot_slot() = Some(snapshot);
                true
            }

            pub fn complete(&self, packet: $packet) -> bool {
                if *self.snapshot_slot() != Some(packet.snapshot) {
                    return false;
                }
                *self.data_slot() = Some(packet);
                self.in_flight.store(false, Ordering::Release);
                true
            }

            pub fn reset(&self) {
                *self.data_slot() = None;
                *self.snapshot_slot() = None;
                self.in_flight.store(false, Ordering::Release);
            }

            /// No request is running and no answer is waiting.
            ///
            /// A consumer that recorded an outstanding request and then still
            /// sees an idle channel knows the request was cancelled by an
            /// experiment reset: a delivered answer always stores `data`
            /// before `in_flight` drops.
            pub fn is_idle(&self) -> bool {
                !self.in_flight.load(Ordering::Acquire) && self.data_slot().is_none()
            }

            /// Removes the delivered answer, if any, without touching an
            /// outstanding request.
            pub fn take(&self) -> Option<$packet> {
                self.data_slot().take()
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
backend_channel!(
    BackendSurfaceSnapshot,
    BackendSurfacePacket,
    BackendSurfaceChannel,
    Vec<f64>
);
backend_channel!(
    BackendComparisonSnapshot,
    BackendComparisonPacket,
    BackendComparisonChannel,
    Vec<f64>
);
backend_channel!(
    BackendReferenceSnapshot,
    BackendReferencePacket,
    BackendReferenceChannel,
    Vec<f64>
);
backend_channel!(
    BackendSensitivitySnapshot,
    BackendSensitivityPacket,
    BackendSensitivityChannel,
    Vec<f64>
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
    pub surface: Res<'w, BackendSurfaceChannel>,
    pub comparison: Res<'w, BackendComparisonChannel>,
    pub reference: Res<'w, BackendReferenceChannel>,
    pub sensitivity: Res<'w, BackendSensitivityChannel>,
}

impl BackendChannels<'_> {
    pub fn reset_all(&self) {
        self.advance.reset();
        self.evaluate_sources.reset();
        self.candidates.reset();
        self.density.reset();
        self.evaluate.reset();
        self.configure.reset();
        self.surface.reset();
        self.comparison.reset();
        self.reference.reset();
        self.sensitivity.reset();
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
    surface: BackendSurfaceChannel,
    comparison: BackendComparisonChannel,
    reference: BackendReferenceChannel,
    sensitivity: BackendSensitivityChannel,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static BACKEND_DELIVERY_CHANNELS: std::cell::RefCell<Option<BackendDeliveryChannels>> =
        const { std::cell::RefCell::new(None) };
    /// Worker ACK for Eq.121 mode uploads (Ok = applied, Err = message).
    static FREQUENCY_DOMAIN_MODES_RESULT: std::cell::RefCell<Option<Result<(), String>>> =
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

/// One `deliver_backend_*_result` export per channel. The page's Worker client
/// calls these with the request identity echoed by the Worker; `complete`
/// rejects anything that does not match the outstanding snapshot.
#[cfg(target_arch = "wasm32")]
macro_rules! backend_delivery {
    ($name:ident, $field:ident, $packet:ident, $snapshot:ident, $value:ty) => {
        #[wasm_bindgen]
        pub fn $name(request_id: u64, epoch: u64, value: $value, error: Option<String>) {
            BACKEND_DELIVERY_CHANNELS.with(|slot| {
                if let Some(channels) = slot.borrow().as_ref() {
                    channels.$field.complete($packet {
                        snapshot: $snapshot { request_id, epoch },
                        result: delivered_result(value, error),
                    });
                }
            });
        }
    };
}

#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_advance_result,
    advance,
    BackendAdvancePacket,
    BackendAdvanceSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_evaluate_sources_result,
    evaluate_sources,
    BackendEvaluateSourcesPacket,
    BackendEvaluateSourcesSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_candidates_result,
    candidates,
    BackendCandidatesPacket,
    BackendCandidatesSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_density_result,
    density,
    BackendDensityPacket,
    BackendDensitySnapshot,
    Vec<f32>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_evaluate_result,
    evaluate,
    BackendEvaluatePacket,
    BackendEvaluateSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_surface_field_result,
    surface,
    BackendSurfacePacket,
    BackendSurfaceSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_comparison_result,
    comparison,
    BackendComparisonPacket,
    BackendComparisonSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_reference_result,
    reference,
    BackendReferencePacket,
    BackendReferenceSnapshot,
    Vec<f64>
);
#[cfg(target_arch = "wasm32")]
backend_delivery!(
    deliver_backend_sensitivity_result,
    sensitivity,
    BackendSensitivityPacket,
    BackendSensitivitySnapshot,
    Vec<f64>
);

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

/// Worker finished applying (or failed to apply) the Eq.121 mode upload.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn deliver_backend_frequency_domain_modes_result(error: Option<String>) {
    FREQUENCY_DOMAIN_MODES_RESULT.with(|slot| {
        *slot.borrow_mut() = Some(match error {
            Some(message) if !message.is_empty() => Err(message),
            _ => Ok(()),
        });
    });
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = r#"
export function backend_worker_ready() {
    return Boolean(globalThis.ryuguNumericalWorkerReady && globalThis.ryuguBackendClient?.isReady());
}
// wasm-bindgen hands slices over as views into the frontend's linear memory.
// Copy them into standalone buffers so the Worker message carries only the
// payload (the client transfers those buffers instead of cloning them).
function detachedPayload(payload) {
    const copy = {};
    for (const [key, value] of Object.entries(payload)) {
        copy[key] = ArrayBuffer.isView(value) ? new value.constructor(value) : value;
    }
    return copy;
}
function post_backend_request(kind, requestId, epoch, payload) {
    if (!globalThis.ryuguBackendClient?.request(kind, requestId, epoch, detachedPayload(payload))) {
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
export function request_backend_prepare_candidate_sources(requestId, epoch, xyz, masses) {
    post_backend_request('prepare_candidate_sources', requestId, epoch, { xyz, masses });
}
export function clear_backend_candidate_sources() {
    globalThis.ryuguBackendClient?.request('clear_candidate_sources', 0, 0, {});
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
export function request_backend_frequency_domain_modes(requestId, epoch, modes) {
    post_backend_request('frequency_domain_modes', requestId, epoch, { modes });
}
export function request_backend_surface_field(requestId, epoch, method, targets) {
    post_backend_request('surface_field', requestId, epoch, { method, targets });
}
export function request_backend_comparison_sources(requestId, epoch, xyz, masses, targets) {
    post_backend_request('comparison_sources', requestId, epoch,
        { method: 'direct', xyz, masses, targets });
}
export function request_backend_reference_sources(requestId, epoch, xyz, masses, targets, chunk) {
    post_backend_request('reference_sources', requestId, epoch,
        { method: 'direct', xyz, masses, targets, chunk });
}
export function request_backend_source_sets(requestId, epoch, method, setOffsets, xyz, masses, targets) {
    post_backend_request('source_sets', requestId, epoch,
        { method, setOffsets, xyz, masses, targets });
}
"#)]
extern "C" {
    #[wasm_bindgen]
    fn backend_worker_ready() -> bool;
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
    fn request_backend_prepare_candidate_sources(
        request_id: u64,
        epoch: u64,
        xyz: &[f64],
        masses: &[f64],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen]
    fn clear_backend_candidate_sources();
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
    #[wasm_bindgen(catch)]
    fn request_backend_frequency_domain_modes(
        request_id: u64,
        epoch: u64,
        modes: &[f64],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_surface_field(
        request_id: u64,
        epoch: u64,
        method: &str,
        targets: &[f64],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_comparison_sources(
        request_id: u64,
        epoch: u64,
        xyz: &[f64],
        masses: &[f64],
        targets: &[f64],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_reference_sources(
        request_id: u64,
        epoch: u64,
        xyz: &[f64],
        masses: &[f64],
        targets: &[f64],
        chunk: u32,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn request_backend_source_sets(
        request_id: u64,
        epoch: u64,
        method: &str,
        set_offsets: &[u32],
        xyz: &[f64],
        masses: &[f64],
        targets: &[f64],
    ) -> Result<(), JsValue>;
}

/// Backend key of a method whose field is evaluated pointwise against the
/// configured geometry. Frequency-domain has its own GPU operator.
fn pointwise_method_key(method: ActiveGravityMethod) -> Result<&'static str, String> {
    match method {
        ActiveGravityMethod::RadialAnalytic => Ok("radial"),
        ActiveGravityMethod::HomogeneousWerner => Ok("werner"),
        ActiveGravityMethod::Fmm => Ok("fmm"),
        ActiveGravityMethod::MmfftCompressed => Ok("fft"),
        ActiveGravityMethod::FrequencyDomain => {
            Err("Frequency-domain uses its dedicated operator".into())
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn flatten_sources(sources: &[(bevy::math::DVec3, f64)]) -> (Vec<f64>, Vec<f64>) {
    (
        sources.iter().flat_map(|(p, _)| p.to_array()).collect(),
        sources.iter().map(|(_, mass)| *mass).collect(),
    )
}

#[cfg(target_arch = "wasm32")]
fn flatten_targets(targets: &[bevy::math::DVec3]) -> Vec<f64> {
    targets
        .iter()
        .flat_map(|position| position.to_array())
        .collect()
}

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
        let (xyz, masses) = flatten_sources(sources);
        let positions = flatten_targets(targets);
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

/// Uploads the canonical planning dynamics sources once per planning run.
///
/// The answer arrives on the candidates channel (an empty value) because the
/// upload and the slices that depend on it belong to the same consumer and
/// must be strictly ordered: the builder never sends a slice while this
/// request is outstanding, and the Worker executes requests in arrival order.
pub fn request_prepare_candidate_sources(
    channel: &BackendCandidatesChannel,
    snapshot: BackendCandidatesSnapshot,
    sources: &[(bevy::math::DVec3, f64)],
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        let (xyz, masses) = flatten_sources(sources);
        if let Err(error) = request_backend_prepare_candidate_sources(
            snapshot.request_id,
            snapshot.epoch,
            &xyz,
            &masses,
        ) {
            channel.reset();
            return Err(format!("Candidate source upload Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, sources);
        Ok(false)
    }
}

/// Releases the uploaded planning sources in the Worker. Fire-and-forget: the
/// Worker queue orders it after every earlier request of the finished run, and
/// the acknowledgement carries no channel identity so nothing can consume it.
pub fn clear_candidate_sources() {
    #[cfg(target_arch = "wasm32")]
    clear_backend_candidate_sources();
}

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

pub fn request_evaluate(
    channel: &BackendEvaluateChannel,
    snapshot: BackendEvaluateSnapshot,
    method: ActiveGravityMethod,
    position: Vec3,
) -> Result<bool, String> {
    let key = pointwise_method_key(method)?;
    #[cfg(target_arch = "wasm32")]
    {
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
        let _ = (channel, snapshot, key, position);
        Ok(false)
    }
}

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

/// Upload the finite Eq.121 operator used by frequency-domain live integration.
pub fn request_frequency_domain_modes(modes: &[f64]) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() {
            return Ok(false);
        }
        if let Err(error) = request_backend_frequency_domain_modes(0, 0, modes) {
            return Err(format!("Frequency-domain mode upload failed: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = modes;
        Ok(false)
    }
}

/// Pointwise field of the configured geometry at many body-frame targets.
/// The answer holds four values `[gx, gy, gz, potential]` per target.
pub fn request_surface_field(
    channel: &BackendSurfaceChannel,
    snapshot: BackendSurfaceSnapshot,
    method: ActiveGravityMethod,
    targets: &[bevy::math::DVec3],
) -> Result<bool, String> {
    let key = pointwise_method_key(method)?;
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        let positions = flatten_targets(targets);
        if let Err(error) =
            request_backend_surface_field(snapshot.request_id, snapshot.epoch, key, &positions)
        {
            channel.reset();
            return Err(format!("Surface-field Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, key, targets);
        Ok(false)
    }
}

/// Direct f64 point-source oracle for the Basilisk protocol comparison.
pub fn request_comparison_sources(
    channel: &BackendComparisonChannel,
    snapshot: BackendComparisonSnapshot,
    sources: &[(bevy::math::DVec3, f64)],
    target: bevy::math::DVec3,
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        let (xyz, masses) = flatten_sources(sources);
        if let Err(error) = request_backend_comparison_sources(
            snapshot.request_id,
            snapshot.epoch,
            &xyz,
            &masses,
            &target.to_array(),
        ) {
            channel.reset();
            return Err(format!("Comparison Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, sources, target);
        Ok(false)
    }
}

/// Direct f64 planning reference, evaluated in fixed source chunks.
///
/// The Worker answers with one `[targets × 4]` block per consecutive chunk of
/// `chunk` sources so the consumer can accumulate the chunk fields in the same
/// order the former time-sliced main-thread loop did.
pub fn request_reference_sources(
    channel: &BackendReferenceChannel,
    snapshot: BackendReferenceSnapshot,
    sources: &[(bevy::math::DVec3, f64)],
    targets: &[bevy::math::DVec3],
    chunk: u32,
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        let (xyz, masses) = flatten_sources(sources);
        let positions = flatten_targets(targets);
        if let Err(error) = request_backend_reference_sources(
            snapshot.request_id,
            snapshot.epoch,
            &xyz,
            &masses,
            &positions,
            chunk,
        ) {
            channel.reset();
            return Err(format!("Planning reference Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, sources, targets, chunk);
        Ok(false)
    }
}

/// Evaluates several independent source sets against one common target list.
///
/// `sets[i]` is evaluated on its own, exactly like one `evaluate_sources`
/// call per set; the answer is laid out `[set][target][4]`. This serves the
/// inversion reference trees and the voxel-basis sensitivity matrices.
pub fn request_source_sets(
    channel: &BackendSensitivityChannel,
    snapshot: BackendSensitivitySnapshot,
    method: &str,
    sets: &[Vec<(bevy::math::DVec3, f64)>],
    targets: &[bevy::math::DVec3],
) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if !backend_worker_ready() || !channel.begin(snapshot) {
            return Ok(false);
        }
        let mut set_offsets = Vec::with_capacity(sets.len() + 1);
        let mut xyz = Vec::new();
        let mut masses = Vec::new();
        set_offsets.push(0_u32);
        for set in sets {
            for (position, mass) in set {
                xyz.extend(position.to_array());
                masses.push(*mass);
            }
            set_offsets.push(masses.len() as u32);
        }
        let positions = flatten_targets(targets);
        if let Err(error) = request_backend_source_sets(
            snapshot.request_id,
            snapshot.epoch,
            method,
            &set_offsets,
            &xyz,
            &masses,
            &positions,
        ) {
            channel.reset();
            return Err(format!("Sensitivity Worker request: {error:?}"));
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (channel, snapshot, method, sets, targets);
        Ok(false)
    }
}

/// Validates the four-value `[gravity, potential]` field response.
pub fn decode_field_response(values: Vec<f64>) -> Result<(Vec3, f32), String> {
    if values.len() != 4 || !values.iter().all(|value| value.is_finite()) {
        return Err("Invalid C++ gravity response".into());
    }
    Ok((
        Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32),
        values[3] as f32,
    ))
}

/// Validates a batched field response of `count` four-value records.
pub fn decode_field_batch(values: &[f64], count: usize) -> Result<&[[f64; 4]], String> {
    if values.len() != count * 4 || !values.iter().all(|value| value.is_finite()) {
        return Err("Invalid C++ batch response".into());
    }
    Ok(values.as_chunks::<4>().0)
}

/// Readiness of the numerical Worker for the currently selected geometry.
#[derive(Resource, Default)]
pub struct CppBackendState {
    /// The Worker's backend holds the configured geometry for `source_key`.
    pub ready: bool,
    source_key: Option<(u64, DensityMode)>,
    configuration_request_id: u64,
}

/// Lower bound on how often the live advance path may submit a request.
///
/// `None` means nothing has been submitted yet, so the first request goes out
/// immediately. On the web `Instant` is `performance.now()`, which starts near
/// zero, so subtracting an interval from `Instant::now()` to express "long
/// enough ago" would overflow and panic; the option avoids the subtraction.
#[derive(Default)]
pub struct BackendFramePacer {
    last_submit: Option<Instant>,
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
        let surface = BackendSurfaceChannel::default();
        let comparison = BackendComparisonChannel::default();
        let reference = BackendReferenceChannel::default();
        let sensitivity = BackendSensitivityChannel::default();
        #[cfg(target_arch = "wasm32")]
        install_delivery_channels(BackendDeliveryChannels {
            advance: advance.clone(),
            evaluate_sources: evaluate_sources.clone(),
            candidates: candidates.clone(),
            density: density.clone(),
            evaluate: evaluate.clone(),
            configure: configure.clone(),
            surface: surface.clone(),
            comparison: comparison.clone(),
            reference: reference.clone(),
            sensitivity: sensitivity.clone(),
        });
        app.init_resource::<CppBackendState>()
            .insert_resource(advance)
            .insert_resource(evaluate_sources)
            .insert_resource(candidates)
            .insert_resource(density)
            .insert_resource(evaluate)
            .insert_resource(configure)
            .insert_resource(surface)
            .insert_resource(comparison)
            .insert_resource(reference)
            .insert_resource(sensitivity)
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
                (
                    configure_backend.after(crate::cpu::density::build_density_quadrature_system),
                    upload_frequency_domain_modes.after(configure_backend),
                ),
            );
        app.add_systems(
            Update,
            crate::cpp_planning::dispatch
                .after(crate::bevy_app::backend::planning_batch_evaluator_system),
        );
    }
}

/// Keeps the Worker's backend configured for the selected source geometry.
///
/// The configure channel is keyed by `source_hash` (epoch) and a running
/// request id, so an answer for a geometry that was replaced while the request
/// was outstanding is dropped and the current geometry is requested instead.
fn configure_backend(
    source: Option<Res<DensityQuadratureSource>>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    mode: Res<DensityMode>,
    ryugu: Query<&Transform, With<RyuguMarker>>,
    channel: Res<BackendConfigureChannel>,
    mut state: ResMut<CppBackendState>,
    mut error: ResMut<GravityRuntimeError>,
) {
    let (Some(source), Some(topology), Ok(transform)) = (source, topology, ryugu.single()) else {
        return;
    };
    let key = (source.source_hash, *mode);
    if state.source_key != Some(key) {
        state.ready = false;
    }
    let completed_configuration = channel.take().filter(|packet| {
        packet.snapshot.epoch == source.source_hash
            && packet.snapshot.request_id == state.configuration_request_id
    });
    if let Some(packet) = completed_configuration {
        match packet.result {
            Ok(()) => {
                state.source_key = Some(key);
                state.ready = true;
            }
            Err(message) => error.raise(format!("Worker geometry configuration failed: {message}")),
        }
    }
    if error.is_active() || state.source_key == Some(key) || !channel.is_idle() {
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
    let request_id = state.configuration_request_id.wrapping_add(1);
    let snapshot = BackendConfigureSnapshot {
        request_id,
        epoch: source.source_hash,
    };
    match request_configure(
        &channel,
        snapshot,
        &live_cells,
        &vertices,
        &topology.triangles,
        RYUGU_MASS as f64,
    ) {
        Ok(true) => state.configuration_request_id = request_id,
        Ok(false) => {}
        Err(message) => error.raise(message),
    }
}

/// Builds and uploads the discrete Eq.121 operator once per density geometry.
/// The Worker evaluates it at every integration substep for frequency-domain.
fn upload_frequency_domain_modes(
    source: Option<Res<DensityQuadratureSource>>,
    mode: Res<DensityMode>,
    state: Res<CppBackendState>,
    mut error: ResMut<GravityRuntimeError>,
    mut uploaded: Local<Option<(u64, DensityMode)>>,
    mut pending: Local<Option<(u64, DensityMode)>>,
) {
    #[cfg(target_arch = "wasm32")]
    if let Some(result) = FREQUENCY_DOMAIN_MODES_RESULT.with(|slot| slot.borrow_mut().take()) {
        let key = pending.take();
        match result {
            Ok(()) => {
                if let Some(key) = key {
                    *uploaded = Some(key);
                }
            }
            Err(message) => {
                *uploaded = None;
                error.raise(format!(
                    "Frequency-domain Eq.121 mode upload failed: {message}"
                ));
                return;
            }
        }
    }
    let Some(source) = source else {
        return;
    };
    if error.is_active() || !state.ready {
        return;
    }
    let (hash, bytes) = match *mode {
        DensityMode::Constant => (source.constant_hash, &source.constant_bytes),
        DensityMode::Variable => (source.source_hash, &source.bytes),
    };
    let key = (hash, *mode);
    if *uploaded == Some(key) || *pending == Some(key) {
        return;
    }
    let Some(modes) =
        crate::cpu::frequency_domain::build_equation121_modes(bytes, f64::from(source.radius))
    else {
        // Do not leave live FD spinning on silent "Waiting for … modes" forever.
        error.raise(
            "Frequency-domain Eq.121 modes could not be assembled from the density quadrature."
                .to_string(),
        );
        return;
    };
    match request_frequency_domain_modes(&modes) {
        Ok(true) => *pending = Some(key),
        Ok(false) => {}
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

/// Decides whether the live simulation may submit its next advance request.
///
/// The per-method wall-clock intervals (250 ms to 1.5 s) only existed because
/// the advance call blocked the main thread. Requests are asynchronous now, so
/// the outstanding request itself is the rate limit and the simulation advances
/// as fast as the numerical Worker can answer. Nothing here changes how many
/// integration steps a request carries.
pub fn should_advance_backend(
    pacer: &mut BackendFramePacer,
    channel: &BackendAdvanceChannel,
) -> bool {
    // Never submit while a request is running or an answer is still waiting
    // to be consumed by `physics_system`.
    if !channel.is_idle() {
        return false;
    }
    let now = Instant::now();
    if pacer.last_submit.is_some_and(|last_submit| {
        now.saturating_duration_since(last_submit) < MIN_BACKEND_SUBMIT_INTERVAL
    }) {
        return false;
    }
    pacer.last_submit = Some(now);
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
        assert!(channel.take().is_none());

        assert!(channel.complete(BackendAdvancePacket {
            snapshot: current,
            result: Ok(vec![2.0]),
        }));
        assert!(!channel.in_flight.load(Ordering::Acquire));
        let packet = channel.take().unwrap();
        assert_eq!(packet.snapshot, current);
        assert_eq!(packet.result.unwrap(), vec![2.0]);
    }

    #[test]
    fn a_cancelled_request_leaves_the_channel_idle() {
        let channel = BackendAdvanceChannel::default();
        let snapshot = BackendAdvanceSnapshot {
            request_id: 3,
            epoch: 1,
        };
        assert!(channel.is_idle());

        assert!(channel.begin(snapshot));
        assert!(!channel.is_idle());
        channel.reset();
        assert!(channel.is_idle());

        assert!(channel.begin(snapshot));
        assert!(channel.complete(BackendAdvancePacket {
            snapshot,
            result: Ok(vec![1.0]),
        }));
        assert!(!channel.is_idle());
    }

    #[test]
    fn advance_pacing_follows_the_outstanding_request() {
        let channel = BackendAdvanceChannel::default();
        let mut pacer = BackendFramePacer::default();
        assert!(should_advance_backend(&mut pacer, &channel));
        // The floor still holds back an immediate resubmission.
        assert!(!should_advance_backend(&mut pacer, &channel));

        pacer.last_submit = None;
        assert!(channel.begin(BackendAdvanceSnapshot {
            request_id: 1,
            epoch: 0,
        }));
        assert!(!should_advance_backend(&mut pacer, &channel));

        channel.reset();
        assert!(should_advance_backend(&mut pacer, &channel));
    }

    #[test]
    fn decode_field_batch_rejects_wrong_length_and_non_finite_values() {
        assert_eq!(
            decode_field_batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2).unwrap(),
            &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        );
        assert!(decode_field_batch(&[1.0, 2.0, 3.0, 4.0], 2).is_err());
        assert!(decode_field_batch(&[1.0, 2.0, 3.0, f64::NAN], 1).is_err());
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
            .init_resource::<BackendSurfaceChannel>()
            .init_resource::<BackendComparisonChannel>()
            .init_resource::<BackendReferenceChannel>()
            .init_resource::<BackendSensitivityChannel>()
            .add_systems(Update, |channels: BackendChannels| channels.reset_all());

        fn flags(world: &World) -> [Arc<AtomicBool>; 10] {
            [
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
                world.resource::<BackendSurfaceChannel>().in_flight.clone(),
                world
                    .resource::<BackendComparisonChannel>()
                    .in_flight
                    .clone(),
                world
                    .resource::<BackendReferenceChannel>()
                    .in_flight
                    .clone(),
                world
                    .resource::<BackendSensitivityChannel>()
                    .in_flight
                    .clone(),
            ]
        }

        for in_flight in flags(app.world()) {
            in_flight.store(true, Ordering::Release);
        }

        app.update();

        for in_flight in flags(app.world()) {
            assert!(!in_flight.load(Ordering::Acquire));
        }
        assert!(
            app.world()
                .resource::<BackendAdvanceChannel>()
                .snapshot
                .lock()
                .unwrap()
                .is_none()
        );
    }
}
