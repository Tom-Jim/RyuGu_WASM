//! Versioned state boundary for the three browser WASM modules.
//!
//! Basilisk's native message bus is an in-process C++ API.  The browser cannot
//! consume that bus directly, so this module defines the small, stable wire
//! contract shared by the Rust WASM backend, the HTML shell, and the
//! Zig/C++ WASM adapter. The contract deliberately uses SI units and an
//! integer nanosecond clock so display timing never becomes simulation timing.
//!
//! The wire constants and `#[repr(C)]` payload structs are the contract itself:
//! the browser only publishes the spacecraft-state half, so the native-side
//! items are intentionally unused here.
#![allow(dead_code)]

use crate::cpp_backend::{BackendComparisonChannel, BackendComparisonSnapshot};
use crate::cpu::frequency_domain::{AggregatedGravitySource, FrequencyDomainPointSource};
use crate::interface::components::{
    ActiveGravityMethod, BasiliskBridgeState, BasiliskSnapshot, CassiniMarker, DensityMode,
    GravitySampleHistory, RyuguMarker, SimulationClock, Velocity,
};
use bevy::math::{DVec3, Vec3};
use bevy::prelude::*;

/// Outstanding oracle request of the protocol comparison.
///
/// The measured acceleration and the algorithm are captured when the request
/// is issued, so the delivered reference is only ever paired with the physics
/// sample it was computed for.
struct PendingComparison {
    snapshot: BackendComparisonSnapshot,
    sample: (u64, u64),
    algorithm: BasiliskAlgorithm,
    measured: Vec3,
}

/// Request → poll state of the comparison oracle. At most one reference is in
/// flight; a new one is only requested for a physics sample that has not been
/// compared yet, so the outstanding request is the rate limit.
#[derive(Default)]
struct ComparisonOracle {
    pending: Option<PendingComparison>,
    last_sample: Option<(u64, u64)>,
    next_request_id: u64,
}

pub const PROTOCOL_NAME: &str = "ryugu-basilisk-v1";
pub const PROTOCOL_VERSION: u16 = 1;
pub const PAYLOAD_SPACECRAFT_STATE: u16 = 1;
pub const PAYLOAD_COMPARISON: u16 = 2;
/// The browser/native wall tick is 1/60 s, while this project intentionally
/// advances one physical second per fixed tick (`TIME_SCALE == 60`).
pub const BASILISK_WALL_TICK_SECONDS: f64 = 1.0 / 60.0;
pub const BASILISK_FIXED_STEP_SECONDS: f64 = 1.0;
pub const BASILISK_FIXED_STEP_NANOS: u64 = 1_000_000_000;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasiliskAlgorithm {
    #[default]
    Radial = 0,
    Werner = 1,
    Fmm = 2,
    Fft = 3,
    FrequencyDomain = 4,
}

impl BasiliskAlgorithm {
    pub const ALL: [Self; 5] = [
        Self::Radial,
        Self::Werner,
        Self::Fmm,
        Self::Fft,
        Self::FrequencyDomain,
    ];

    pub const fn key(self) -> &'static str {
        match self {
            Self::Radial => "radial",
            Self::Werner => "werner",
            Self::Fmm => "fmm",
            Self::Fft => "fft",
            Self::FrequencyDomain => "frequency_domain",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Radial => "Radial",
            Self::Werner => "Werner",
            Self::Fmm => "FMM",
            Self::Fft => "FFT",
            Self::FrequencyDomain => "Frequency-domain",
        }
    }

    pub const fn observable(self) -> &'static str {
        match self {
            Self::FrequencyDomain => {
                "trajectory Laplace transform (Eq.184) + Eq.121 pointwise propagation"
            }
            Self::Radial | Self::Werner | Self::Fmm | Self::Fft => "point acceleration",
        }
    }

    pub const fn from_active(method: ActiveGravityMethod) -> Self {
        match method {
            ActiveGravityMethod::RadialAnalytic => Self::Radial,
            ActiveGravityMethod::HomogeneousWerner => Self::Werner,
            ActiveGravityMethod::FrequencyDomain => Self::FrequencyDomain,
            ActiveGravityMethod::MmfftCompressed => Self::Fft,
            ActiveGravityMethod::Fmm => Self::Fmm,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BasiliskProtocolHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub payload_kind: u16,
    pub sequence: u64,
    pub simulation_time_ns: u64,
    pub epoch: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BasiliskScStatesMsgPayload {
    /// Inertial position, metres.
    pub r_bn_n_m: [f64; 3],
    /// Inertial velocity, metres per second.
    pub v_bn_n_mps: [f64; 3],
    /// Modified Rodrigues parameters, matching Basilisk SCStatesMsgPayload.
    pub sigma_bn: [f64; 3],
    pub header: BasiliskProtocolHeader,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BasiliskForceMsgPayload {
    pub a_bn_n_mps2: [f64; 3],
    pub potential_m2ps2: f64,
    pub algorithm: BasiliskAlgorithm,
    pub reserved: [u8; 7],
    pub header: BasiliskProtocolHeader,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BasiliskComparison {
    pub algorithm: BasiliskAlgorithm,
    pub sample_count: u64,
    pub relative_acceleration_error: f64,
    pub reference_acceleration_mps2: [f64; 3],
    pub measured_acceleration_mps2: [f64; 3],
}

pub struct BasiliskPlugin;

impl Plugin for BasiliskPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BasiliskBridgeState>();
        app.add_systems(PostUpdate, publish_basilisk_snapshot_system);
    }
}

/// Publish one protocol snapshot after physics and GPU readbacks have settled.
/// This is the browser equivalent of a Basilisk state message; the native
/// adapter consumes the same fields through the C ABI described in `src/backend/zig/`.
fn publish_basilisk_snapshot_system(
    clock: Res<SimulationClock>,
    active: Res<ActiveGravityMethod>,
    density_mode: Res<DensityMode>,
    source: Option<Res<AggregatedGravitySource>>,
    radial: Option<Res<crate::interface::components::RadialGravityHistory>>,
    werner: Option<Res<crate::interface::components::WernerGravityHistory>>,
    fft: Option<Res<crate::interface::components::MmfftCompressedHistory>>,
    fmm: Option<Res<crate::interface::components::FmmGravityHistory>>,
    frequency: Option<Res<crate::gpu::equation106::Equation106History>>,
    mut state: ResMut<BasiliskBridgeState>,
    cassini: Query<(&Transform, &Velocity), With<CassiniMarker>>,
    ryugu: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    channel: Res<BackendComparisonChannel>,
    mut oracle: Local<ComparisonOracle>,
) {
    let Ok((transform, velocity)) = cassini.single() else {
        return;
    };
    let Ok(ryugu_transform) = ryugu.single() else {
        return;
    };
    let algorithm = BasiliskAlgorithm::from_active(*active);
    let sequence = clock.request_id;
    let header = BasiliskProtocolHeader {
        magic: *b"RYGU",
        version: PROTOCOL_VERSION,
        payload_kind: PAYLOAD_SPACECRAFT_STATE,
        sequence,
        simulation_time_ns: simulation_time_nanos(clock.elapsed_seconds),
        epoch: clock.epoch,
    };
    state.fixed_step_seconds = clock.fixed_step_seconds;
    state.snapshot = Some(BasiliskSnapshot {
        algorithm: algorithm as u8,
        sequence,
        protocol_version: header.version,
        simulation_time_ns: header.simulation_time_ns,
        simulation_time_seconds: clock.elapsed_seconds,
        epoch: clock.epoch,
        position_m: transform.translation.to_array().map(f64::from),
        velocity_mps: velocity.0.to_array().map(f64::from),
    });

    let history = crate::interface::select_history(
        *active,
        radial.as_deref(),
        werner.as_deref(),
        fft.as_deref(),
        fmm.as_deref(),
        frequency.as_deref(),
    );

    // Consume a delivered reference first. It is only accepted for the exact
    // request (and epoch) it was issued for; anything else is dropped and the
    // comparison keeps its last completed value.
    if let Some(packet) = channel.take() {
        let Some(pending) = oracle.pending.take() else {
            return;
        };
        if packet.snapshot != pending.snapshot || pending.snapshot.epoch != clock.epoch {
            return;
        }
        let Some(reference) = packet
            .result
            .ok()
            .and_then(|values| point_source_acceleration_from_values(&values))
        else {
            return;
        };
        oracle.last_sample = Some(pending.sample);
        let comparison = &mut state.comparisons[algorithm_index(pending.algorithm)];
        comparison.sample_count = comparison.sample_count.saturating_add(1);
        comparison.relative_acceleration_error = relative_vector_error(reference, pending.measured);
        comparison.reference_acceleration_mps2 = reference.to_array().map(f64::from);
        comparison.measured_acceleration_mps2 = pending.measured.to_array().map(f64::from);
        return;
    }
    if oracle.pending.is_some() {
        if !channel.is_idle() {
            // Still computing: skip this frame rather than block on it.
            return;
        }
        // An experiment reset cleared the channel; the request is gone.
        oracle.pending = None;
    }
    let Some((sample, measured)) = history_sample(history, clock.epoch) else {
        return;
    };
    if oracle.last_sample == Some(sample) {
        return;
    }
    let Some(source) = source.as_deref() else {
        return;
    };
    let body_position = ryugu_transform.rotation.inverse() * transform.translation;
    let Some((sources, target)) = point_source_request(source, *density_mode, body_position) else {
        return;
    };
    oracle.next_request_id = oracle.next_request_id.wrapping_add(1).max(1);
    let snapshot = BackendComparisonSnapshot {
        request_id: sample.1.rotate_left(29) ^ oracle.next_request_id,
        epoch: clock.epoch,
    };
    if crate::cpp_backend::request_comparison_sources(&channel, snapshot, &sources, target)
        == Ok(true)
    {
        oracle.pending = Some(PendingComparison {
            snapshot,
            sample,
            algorithm,
            measured,
        });
    }
}

pub const fn simulation_time_nanos(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1_000_000_000.0 + 0.5) as u64
}

pub fn algorithm_index(algorithm: BasiliskAlgorithm) -> usize {
    algorithm as usize
}

/// Independent f64 point-source oracle used by the protocol comparison.  It
/// is intentionally separate from the frequency-domain GPU pipeline and uses
/// the same mass-preserving residues already used by the CPU reference path.
/// The direct sum itself runs in the numerical Worker; this builds the request.
fn point_source_request(
    source: &AggregatedGravitySource,
    density_mode: DensityMode,
    position: Vec3,
) -> Option<(Vec<(DVec3, f64)>, DVec3)> {
    let points: &[FrequencyDomainPointSource] = match density_mode {
        DensityMode::Variable => &source.sources,
        DensityMode::Constant => &source.constant_sources,
    };
    let total_mass = match density_mode {
        DensityMode::Variable => source.total_mass,
        DensityMode::Constant => source.constant_total_mass,
    };
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return None;
    }
    if points.is_empty() || !position.is_finite() {
        return None;
    }
    let sources = points.iter().map(|p| (p.position, p.mass)).collect();
    Some((sources, position.as_dvec3()))
}

/// Decodes the single `[gx, gy, gz, potential]` record of the oracle answer.
fn point_source_acceleration_from_values(values: &[f64]) -> Option<Vec3> {
    let record = crate::cpp_backend::decode_field_batch(values, 1).ok()?[0];
    let acceleration = DVec3::new(record[0], record[1], record[2]);
    acceleration.is_finite().then_some(acceleration.as_vec3())
}

pub fn relative_vector_error(reference: Vec3, measured: Vec3) -> f64 {
    let denominator = f64::from(reference.length().max(1.0e-20));
    f64::from(reference.distance(measured)) / denominator
}

/// Latest accepted physics sample of the current epoch: its identity
/// `(epoch, request_id)` and the measured body-frame acceleration.
pub fn history_sample(
    history: Option<&GravitySampleHistory>,
    epoch: u64,
) -> Option<((u64, u64), Vec3)> {
    history
        .and_then(|history| history.latest_for_epoch(epoch))
        .map(|sample| {
            (
                (sample.snapshot.epoch, sample.snapshot.request_id),
                sample.body_acceleration,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_time_is_integer_nanoseconds() {
        assert_eq!(simulation_time_nanos(0.0), 0);
        assert_eq!(simulation_time_nanos(1.0), 1_000_000_000);
        assert_eq!(simulation_time_nanos(1.000_000_001), 1_000_000_001);
    }

    #[test]
    fn algorithm_order_matches_browser_contract() {
        assert_eq!(
            BasiliskAlgorithm::ALL.map(|algorithm| algorithm.key()),
            ["radial", "werner", "fmm", "fft", "frequency_domain",]
        );
    }

    #[test]
    fn relative_error_is_zero_for_matching_vectors() {
        let acceleration = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(relative_vector_error(acceleration, acceleration), 0.0);
    }

    #[test]
    fn oracle_answer_decodes_one_finite_record() {
        assert_eq!(
            point_source_acceleration_from_values(&[1.0, -2.0, 3.0, 4.0]),
            Some(Vec3::new(1.0, -2.0, 3.0))
        );
        assert_eq!(
            point_source_acceleration_from_values(&[1.0, 2.0, 3.0]),
            None
        );
        assert_eq!(
            point_source_acceleration_from_values(&[1.0, f64::NAN, 3.0, 4.0]),
            None
        );
    }
}
