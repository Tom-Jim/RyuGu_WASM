//! Independent numerical WASM. No Bevy, renderer, window, or DOM dependencies.
use glam::{DMat3, DQuat, DVec3};
use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
mod density;
mod planning;

#[wasm_bindgen(inline_js = r#"
export function field_configure(cells, vertices, facets, mass) {
  globalThis.ryuguCpp.configure(cells, vertices, facets, mass);
}
export function field_evaluate(method, x, y, z) {
  return new Float64Array(globalThis.ryuguCpp.evaluate(method, [x,y,z]));
}
export function field_sources(method, xyz, masses, targets) {
  return globalThis.ryuguCpp.evaluateSources(method, xyz, masses, targets);
}
export function field_prepare(xyz, masses) {
  return globalThis.ryuguCpp.prepareSources(xyz, masses);
}
export function field_prepared(id, targets) {
  return globalThis.ryuguCpp.evaluatePreparedSources(id, targets);
}
export function field_release(id) {
  globalThis.ryuguCpp.releaseSources(id);
}
export function scheduler_reset(period) {
  const status = globalThis.ryuguScheduler.ryugu_scheduler_reset(period);
  if (status) throw new Error(`Basilisk reset failed: ${status}`);
}
export function scheduler_advance(stop) {
  const status = globalThis.ryuguScheduler.ryugu_scheduler_advance(stop);
  if (status) throw new Error(`Basilisk advance failed: ${status}`);
}
"#)]
extern "C" {
    #[wasm_bindgen(catch)]
    fn field_prepare(xyz: &[f64], masses: &[f64]) -> Result<f64, JsValue>;
    #[wasm_bindgen(catch)]
    fn field_prepared(id: f64, targets: &[f64]) -> Result<Vec<f64>, JsValue>;
    fn field_release(id: f64);
    #[wasm_bindgen(catch)]
    fn field_configure(
        cells: &[f64],
        vertices: &[f64],
        facets: &[u32],
        mass: f64,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn field_evaluate(method: &str, x: f64, y: f64, z: f64) -> Result<Vec<f64>, JsValue>;
    #[wasm_bindgen(catch)]
    fn field_sources(
        method: &str,
        xyz: &[f64],
        masses: &[f64],
        targets: &[f64],
    ) -> Result<Vec<f64>, JsValue>;
    #[wasm_bindgen(catch)]
    fn scheduler_reset(period: u64) -> Result<(), JsValue>;
    #[wasm_bindgen(catch)]
    fn scheduler_advance(stop: u64) -> Result<(), JsValue>;
}

/// Owns a C++ source upload for a complete integration slice, including errors.
struct PreparedSources(f64);

impl PreparedSources {
    fn new(xyz: &[f64], masses: &[f64]) -> Result<Self, JsValue> {
        field_prepare(xyz, masses).map(Self)
    }

    fn evaluate(&self, targets: &[f64]) -> Result<Vec<f64>, JsValue> {
        field_prepared(self.0, targets)
    }
}

impl Drop for PreparedSources {
    fn drop(&mut self) {
        field_release(self.0);
    }
}

thread_local! {
    // The numerically expensive source geometry is reused across planning
    // integration slices. Keep the thread-local payload copy-only so WASI
    // does not need to register a nontrivial TLS destructor.
    static CACHED_SOURCE: Cell<Option<f64>> = const { Cell::new(None) };
}

fn cached_source() -> Option<f64> {
    CACHED_SOURCE.with(Cell::get)
}

/// Uploads source geometry once for a sequence of candidate trajectory slices.
/// The caller must call `clear_candidate_sources` before the C++ module is
/// dropped; the browser Worker owns the lifetime in normal operation.
#[wasm_bindgen]
pub fn prepare_candidate_sources(xyz: &[f64], masses: &[f64]) -> Result<(), JsValue> {
    if masses.is_empty() || xyz.len() != masses.len() * 3 {
        return Err("Invalid candidate source geometry".into());
    }
    let source = PreparedSources::new(xyz, masses)?;
    let id = source.0;
    std::mem::forget(source);
    CACHED_SOURCE.with(|slot| {
        if let Some(old) = slot.replace(Some(id)) {
            field_release(old);
        }
    });
    Ok(())
}

#[wasm_bindgen]
pub fn clear_candidate_sources() {
    CACHED_SOURCE.with(|slot| {
        if let Some(id) = slot.replace(None) {
            field_release(id);
        }
    });
}

pub(crate) fn evaluate_cached_sources(targets: &[f64]) -> Result<Vec<f64>, JsValue> {
    let id = cached_source().ok_or("Candidate source cache is empty")?;
    field_prepared(id, targets)
}

#[wasm_bindgen]
pub fn protocol_version() -> u32 {
    1
}

#[wasm_bindgen]
pub fn configure(
    cells: &[f64],
    vertices: &[f64],
    facets: &[u32],
    mass: f64,
) -> Result<(), JsValue> {
    field_configure(cells, vertices, facets, mass)
}

#[wasm_bindgen]
pub fn evaluate(method: &str, x: f64, y: f64, z: f64) -> Result<Vec<f64>, JsValue> {
    let position = DVec3::new(x, y, z);
    if method == "fmm" {
        let cached = STATE.with(|cell| {
            let state = cell.borrow();
            (state.method == "fmm")
                .then(|| cached_body_values(&state, position, None))
                .flatten()
        });
        if let Some(values) = cached {
            return Ok(values);
        }
    }
    if method == "frequency_domain" {
        // Inverse Laplace of Eq.(106) on a point is Eq.(121). The continuous
        // ∫d³κ is FLUPS/FFTW (Zig-bound C++), not the 64/256-node spherical sum.
        let (gravity, potential) = equation121_flups_field(position)?;
        return Ok(vec![gravity.x, gravity.y, gravity.z, potential]);
    }
    field_evaluate(method, x, y, z)
}

#[wasm_bindgen]
pub fn evaluate_sources(
    method: &str,
    xyz: &[f64],
    masses: &[f64],
    targets: &[f64],
) -> Result<Vec<f64>, JsValue> {
    field_sources(method, xyz, masses, targets)
}

#[wasm_bindgen]
pub fn propagate_candidates(data: &str) -> Result<Vec<f64>, JsValue> {
    planning::propagate_candidates(data)
}

#[derive(Clone, Default)]
struct State {
    epoch: Option<u64>,
    method: String,
    time_ns: u64,
    period_ns: u64,
    position: DVec3,
    velocity: DVec3,
    initial_rotation: DQuat,
    history: Vec<[f64; 4]>,
    /// Packed Eq.121 modes: `[kx, ky, kz, coeff_re, coeff_im]` × N.
    frequency_domain_modes: Vec<f64>,
    trace: Vec<f64>,
    cached_body_field: Option<DVec3>,
    cached_body_potential: f64,
    cached_body_position: Option<DVec3>,
    cached_field_time: f64,
    /// Body-frame Eq.(106) reference line `q_c + h e` (`mathtidy.md` §2).
    reference_origin: DVec3,
    reference_axis: DVec3,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }

const EQUATION121_MODE_STRIDE: usize = 5;
const EQUATION121_NEWTON_SENTINEL: f64 = 2.0;

/// Upload the finite inverse-Laplace/Eq.121 modes used by live Eq.(106)+Taylor.
/// Modes are independent of probe state and must be refreshed when density changes.
#[wasm_bindgen]
pub fn set_frequency_domain_modes(modes: &[f64]) -> Result<(), JsValue> {
    if modes.is_empty()
        || !modes.len().is_multiple_of(EQUATION121_MODE_STRIDE)
        || !modes.iter().all(|value| value.is_finite())
    {
        return Err("Invalid frequency-domain Eq.121 modes".into());
    }
    STATE.with(|cell| {
        cell.borrow_mut().frequency_domain_modes = modes.to_vec();
    });
    Ok(())
}

fn evaluate_equation121_with_jacobian(
    modes: &[f64],
    position: DVec3,
) -> Result<(DVec3, f64, DMat3), JsValue> {
    if modes.is_empty() || !modes.len().is_multiple_of(EQUATION121_MODE_STRIDE) {
        return Err("Waiting for frequency-domain Eq.121 modes".into());
    }
    let (fourier, newton) = if modes.len() >= 2 * EQUATION121_MODE_STRIDE {
        let trailer_start = modes.len() - EQUATION121_MODE_STRIDE;
        if modes[trailer_start + 4] == EQUATION121_NEWTON_SENTINEL && modes[trailer_start + 3] > 0.0 {
            (
                &modes[..trailer_start],
                Some((
                    DVec3::new(
                        modes[trailer_start],
                        modes[trailer_start + 1],
                        modes[trailer_start + 2],
                    ),
                    modes[trailer_start + 3],
                )),
            )
        } else {
            (modes, None)
        }
    } else {
        (modes, None)
    };
    let mut gravity = DVec3::ZERO;
    let mut potential = 0.0;
    let mut jacobian = DMat3::ZERO;
    for mode in fourier.as_chunks::<EQUATION121_MODE_STRIDE>().0 {
        let k = DVec3::new(mode[0], mode[1], mode[2]);
        let phase = k.dot(position);
        let (sin_phase, cos_phase) = phase.sin_cos();
        let re = mode[3] * cos_phase - mode[4] * sin_phase;
        let im = mode[3] * sin_phase + mode[4] * cos_phase;
        gravity -= im * k;
        potential += re;
        jacobian -= DMat3::from_cols(k * k.x, k * k.y, k * k.z) * re;
    }
    if let Some((center, gravitational_parameter)) = newton {
        let offset = position - center;
        let distance_squared = offset.length_squared();
        if distance_squared > 0.0 {
            let inverse_distance = distance_squared.sqrt().recip();
            let inv3 = inverse_distance.powi(3);
            let inv5 = inverse_distance.powi(5);
            gravity += -gravitational_parameter * offset * inv3;
            potential += gravitational_parameter * inverse_distance;
            jacobian += DMat3::from_cols(
                DVec3::new(
                    -gravitational_parameter * (inv3 - 3.0 * offset.x * offset.x * inv5),
                    gravitational_parameter * 3.0 * offset.x * offset.y * inv5,
                    gravitational_parameter * 3.0 * offset.x * offset.z * inv5,
                ),
                DVec3::new(
                    gravitational_parameter * 3.0 * offset.y * offset.x * inv5,
                    -gravitational_parameter * (inv3 - 3.0 * offset.y * offset.y * inv5),
                    gravitational_parameter * 3.0 * offset.y * offset.z * inv5,
                ),
                DVec3::new(
                    gravitational_parameter * 3.0 * offset.z * offset.x * inv5,
                    gravitational_parameter * 3.0 * offset.z * offset.y * inv5,
                    -gravitational_parameter * (inv3 - 3.0 * offset.z * offset.z * inv5),
                ),
            );
        }
    }
    if !gravity.is_finite() || !potential.is_finite() || !jacobian.is_finite() {
        return Err("Equation (121) returned a non-finite field".into());
    }
    Ok((gravity, potential, jacobian))
}

/// Cartesian Fourier Poisson solve of Eq.(121): `g = G/(2π)³ ∫ (4π i κ/κ²) ρ̂ e^{iκ·q} d³κ`.
/// FLUPS+FFTW on the existing 32³ free-space grid (Zig `ryugu_flups_free_space_eval`).
/// The 64-node GPU layout stays on Eq.(184); it is not this live integral.
fn equation121_flups_field(position: DVec3) -> Result<(DVec3, f64), JsValue> {
    match field_evaluate("fft", position.x, position.y, position.z) {
        Ok(values) => {
            if values.len() != 4 {
                return Err("FLUPS Eq.121 response must be [gx, gy, gz, U]".into());
            }
            let gravity = DVec3::new(values[0], values[1], values[2]);
            let potential = values[3];
            if !gravity.is_finite() || !potential.is_finite() {
                return Err("FLUPS Eq.121 field is non-finite".into());
            }
            Ok((gravity, potential))
        }
        Err(error) => {
            let message = error.as_string().unwrap_or_default();
            if message.contains("outside FLUPS") || message.contains("Mass source outside FLUPS") {
                return STATE.with(|cell| {
                    let modes = cell.borrow().frequency_domain_modes.clone();
                    evaluate_equation121_with_jacobian(&modes, position).map(|(g, u, _)| (g, u))
                });
            }
            Err(error)
        }
    }
}

fn rotation(state: &State, time: f64) -> DQuat {
    DQuat::from_axis_angle(
        DVec3::new(-0.043, -0.914, 0.405).normalize(),
        std::f64::consts::TAU * time / f64::from(7.63_f32 * 3600.0),
    ) * state.initial_rotation
}

fn body_frame_path_velocity(state: &State, q: DQuat, inertial_position: DVec3) -> DVec3 {
    // Tangent of q_B(t) in the rotating frame, not R^T v_I. Ryugu's spin
    // ωr is comparable to orbital speed, so the missing −ω×q term would
    // point the Eq.(106) line the wrong way and blow the Taylor tube.
    let omega = DVec3::new(-0.043, -0.914, 0.405).normalize()
        * (std::f64::consts::TAU / f64::from(7.63_f32 * 3600.0));
    q.inverse() * (state.velocity - omega.cross(inertial_position))
}

fn field_refresh_interval(method: &str) -> f64 {
    match method {
        "werner" => 2.0,
        // Body-frame FMM depends only on q_B. A sim-time gate rebuilds the
        // ExaFMM tree every few ticks at high TIME_SCALE and hitchs vs FFT.
        // Planning/surface/inversion never consult this live cache.
        "fmm" => f64::INFINITY,
        // Live Eq.(106)/121 is the same FLUPS grid as Packed FFT.
        "fft" | "frequency_domain" => 1.0,
        _ => 0.0,
    }
}

fn cache_position_ok(cached: DVec3, position: DVec3, fraction: f64) -> bool {
    let scale = position.length().max(1.0);
    (position - cached).length() <= fraction * scale
}

fn cached_body_values(state: &State, position: DVec3, time: Option<f64>) -> Option<Vec<f64>> {
    let field = state.cached_body_field?;
    let cached = state.cached_body_position?;
    let fraction = if time.is_some() { 0.02 } else { 0.05 };
    if !cache_position_ok(cached, position, fraction) {
        return None;
    }
    if let Some(time) = time {
        let interval = field_refresh_interval(&state.method);
        if interval <= 0.0 || time - state.cached_field_time > interval {
            return None;
        }
    }
    Some(vec![
        field.x,
        field.y,
        field.z,
        state.cached_body_potential,
    ])
}

fn store_cached_body_field(
    state: &mut State,
    position: DVec3,
    field: DVec3,
    potential: f64,
    time: f64,
) {
    state.cached_body_field = Some(field);
    state.cached_body_potential = potential;
    state.cached_body_position = Some(position);
    state.cached_field_time = time;
}

fn rebase_reference_line(state: &mut State, body_position: DVec3, body_velocity: DVec3) {
    state.reference_origin = body_position;
    state.reference_axis = body_velocity
        .try_normalize()
        .unwrap_or_else(|| body_position.try_normalize().unwrap_or(DVec3::Z));
}

fn acceleration(state: &mut State, position: DVec3, time: f64) -> Result<DVec3, JsValue> {
    let q = rotation(state, time);
    let body_position = q.inverse() * position;
    if state.method == "frequency_domain" {
        // Eq.(106) line `q_c + h e` is segmented here (`mathtidy.md` §8.3).
        // Verlet force is the inverse Laplace at the spacecraft: Eq.(121) at
        // `q_B`, the same continuous ∫d³κ FLUPS evaluates for Packed FFT.
        let body_velocity = body_frame_path_velocity(state, q, position);
        if state.reference_axis.length_squared() < 1.0e-12 {
            rebase_reference_line(state, body_position, body_velocity);
        } else {
            let along = (body_position - state.reference_origin).dot(state.reference_axis);
            let reference = state.reference_origin + state.reference_axis * along;
            let transverse = body_position - reference;
            let tube = ((reference.length() - 450.0).max(25.0) * 0.2).min(40.0);
            if transverse.length() > tube || along.abs() > 80.0 {
                rebase_reference_line(state, body_position, body_velocity);
            }
        }
    }
    let eval_method = if state.method == "frequency_domain" {
        "fft"
    } else {
        state.method.as_str()
    };
    let body = if let Some(values) = cached_body_values(state, body_position, Some(time)) {
        DVec3::new(values[0], values[1], values[2])
    } else {
        let values = field_evaluate(eval_method, body_position.x, body_position.y, body_position.z)?;
        if values.len() != 4 {
            return Err("Invalid gravity response".into());
        }
        let field = DVec3::new(values[0], values[1], values[2]);
        store_cached_body_field(state, body_position, field, values[3], time);
        field
    };
    let field = q * body;
    // Near-surface / dense IR+UV fields routinely exceed any fixed 1.5e-3
    // engineering gate; reject only non-finite accelerations.
    if !field.is_finite() {
        return Err("Invalid gravity acceleration".into());
    }
    Ok(field)
}

fn append_trace(state: &mut State, time: f64, field: DVec3) {
    state.trace.push(time);
    state.trace.extend(state.position.to_array());
    state.trace.extend(state.velocity.to_array());
    state.trace.extend(field.to_array());
    state.trace.extend(rotation(state, time).to_array());
}

fn integration_substeps(method: &str) -> usize {
    match method {
        // FMM tree construction is the hitch relative to Packed FFT. One
        // Verlet step still uses two force samples, but they share the
        // body-field cache. FFT/FD/Radial stay at 2.
        "werner" | "fmm" => 1,
        "fft" | "radial" | "frequency_domain" => 2,
        _ => 1,
    }
}

/// Called exclusively by the Basilisk task, synchronously across the WASM ABI.
#[wasm_bindgen]
pub fn tick(time_ns: u64) -> Result<Vec<f64>, JsValue> {
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        if time_ns < state.time_ns || time_ns - state.time_ns > state.period_ns {
            return Err("Basilisk task timestamp is out of sequence".into());
        }
        if time_ns > state.time_ns {
            let mut next = state.clone();
            let start = next.time_ns as f64 * 1e-9;
            let substeps = integration_substeps(&next.method);
            let dt = (time_ns - next.time_ns) as f64 * 1e-9 / substeps as f64;
            for i in 0..substeps {
                let t = start + i as f64 * dt;
                let position = next.position;
                let a = acceleration(&mut next, position, t)?;
                let half_velocity = next.velocity + a * (dt * 0.5);
                next.position += half_velocity * dt;
                let position = next.position;
                let b = acceleration(&mut next, position, t + dt)?;
                next.velocity = half_velocity + b * (dt * 0.5);
                append_trace(&mut next, t + dt, b);
            }
            next.time_ns = time_ns;
            *state = next;
        }
        Ok(state
            .position
            .to_array()
            .into_iter()
            .chain(state.velocity.to_array())
            .collect())
    })
}

/// Input state is adopted only on an epoch reset; subsequent state is backend-owned.
#[wasm_bindgen]
pub fn advance_frame(
    epoch: u64,
    method: &str,
    initial: &[f64],
    step_seconds: f64,
    steps: u32,
    history: &[f64],
) -> Result<Vec<f64>, JsValue> {
    if initial.len() != 10
        || !initial.iter().all(|v| v.is_finite())
        || !step_seconds.is_finite()
        || step_seconds <= 0.0
        || steps == 0
        // Must stay aligned with the frontend `MAX_SIMULATION_ACCELERATION`.
        // Each step still uses the caller's fixed `step_seconds`; a larger
        // `steps` only batches more identical-`dt` leaps into one Worker trip.
        || steps > 64
        || !history.len().is_multiple_of(4)
        || !history.iter().all(|v| v.is_finite())
        || history
            .as_chunks::<4>()
            .0
            .windows(2)
            .any(|w| w[1][0] <= w[0][0])
        || initial[6..].iter().map(|v| v * v).sum::<f64>() < 1e-12
        || !matches!(
            method,
            "radial" | "werner" | "fmm" | "fft" | "frequency_domain"
        )
    {
        return Err("Invalid simulation request".into());
    }
    let period = (step_seconds * 1e9).round() as u64;
    if period == 0 {
        return Err("Simulation period is below one nanosecond".into());
    }
    let reset = STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let reset =
            state.epoch != Some(epoch) || state.method != method || state.period_ns != period;
        if reset {
            let modes = state.frequency_domain_modes.clone();
            *state = State {
                epoch: Some(epoch),
                method: method.into(),
                period_ns: period,
                position: DVec3::from_slice(initial),
                velocity: DVec3::from_slice(&initial[3..]),
                initial_rotation: DQuat::from_slice(&initial[6..]).normalize(),
                frequency_domain_modes: modes,
                ..Default::default()
            };
        }
        state.history = history.as_chunks::<4>().0.to_vec();
        state.trace.clear();
        reset
    });
    if reset {
        scheduler_reset(period)?;
    }
    let stop = STATE.with(|cell| -> Result<u64, JsValue> {
        let mut state = cell.borrow_mut();
        let time = state.time_ns as f64 * 1e-9;
        let position = state.position;
        let a = acceleration(&mut state, position, time)?;
        append_trace(&mut state, time, a);
        state
            .time_ns
            .checked_add(period.checked_mul(steps as u64).ok_or("Time overflow")?)
            .ok_or_else(|| "Time overflow".into())
    })?;
    scheduler_advance(stop)?;
    STATE.with(|cell| Ok(std::mem::take(&mut cell.borrow_mut().trace)))
}
