use bevy::prelude::*;
use bevy::platform::time::Instant;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
pub const G: f32 = 6.6743e-11;
pub const RYUGU_MASS: f32 = 4.5e11;
pub const TIME_SCALE: f32 = 60.0;
pub const BENCHMARK_DURATION_SECONDS: f64 = 901.66;
pub const BENCHMARK_SAMPLE_INTERVAL_SECONDS: f64 = 0.01;
pub const GRAVITY_BENCHMARK_RELATIVE_TOLERANCE: f32 = 0.04;
pub const ORBIT_HISTORY_LEN: usize = 27500;
pub const JACOBI_HISTORY_CAPACITY: usize = 256;
/// Keep at least two complete maximum-acceleration Eq.106 batches.  A batch
/// contains the authoritative anchor plus one endpoint for every accelerated
/// stable step (9 samples at 8x).  A capacity smaller than that silently
/// evicts the authoritative sample before the integrator can consume it.
pub const GRAVITY_SAMPLE_HISTORY_CAPACITY: usize = 2 * (MAX_SIMULATION_ACCELERATION as usize + 1);
pub const PHYSICS_SUBSTEPS: usize = 100;
pub const MIN_SIMULATION_ACCELERATION: u32 = 1;
pub const MAX_SIMULATION_ACCELERATION: u32 = 8;
pub const VISIBILITY_THRESHOLD: f32 = 250.0;
pub const NORMAL_ARROW_LENGTH: f32 = 35.0;

pub fn bytes_to_f32x4(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<{ size_of::<[f32; 4]>() }>()
        .0
        .iter()
        .map(|chunk| bytemuck::pod_read_unaligned(&chunk[..]))
        .collect()
}

pub const RYUGU_ROTATION_PERIOD_SECS: f32 = 7.63 * 3600.0;
pub const RYUGU_SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);

pub const DENSITY_EPSILON: f32 = 10.0;
pub const SECTION_CLIP_RADIUS: f32 = 450.0;
/// Shared outward-increasing logarithmic density law used by the radial,
/// Equation (106), MMFFT, and FMM modes:
/// `rho(r) = C ln(1 + r / epsilon)`.
pub fn logarithmic_radial_density(radius: f32, density_c: f32) -> f32 {
    density_c * (1.0 + radius.max(0.0) / DENSITY_EPSILON).ln()
}

#[derive(Component)]
pub struct TargetSize(pub f32);
#[derive(Component)]
pub struct ScaleNormalized;
#[derive(Component)]
pub struct TopologyBuilt;
#[derive(Component)]
pub struct RyuguMarker;
#[derive(Component)]
pub struct CassiniMarker;
#[derive(Component)]
pub struct UiTextMarker;
#[derive(Component)]
pub struct FpsTextMarker;
#[derive(Component)]
pub struct VramTextMarker;
#[derive(Component)]
pub struct Velocity(pub Vec3);
#[derive(Component)]
pub struct OrbitHistory(pub VecDeque<Vec3>);

/// Number of uniformly resampled detector states exposed by the trajectory
/// inversion controls.  The capture is presentation-only and never feeds the
/// gravity evaluators or the fixed-step integrator.
pub const TRAJECTORY_INVERSION_SAMPLE_COUNT: usize = 16;
pub const TRAJECTORY_INVERSION_CAPTURE_SECONDS: f64 = 5.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct TrajectoryInversionKnot {
    pub position: Vec3,
    pub velocity: Vec3,
    pub simulation_time_seconds: f64,
    /// World-frame acceleration returned by the selected forward evaluator at
    /// the baseline density. Gravity is linear in density, so the optimizer can
    /// evaluate density candidates without substituting another field model.
    pub baseline_acceleration: Vec3,
    pub body_rotation: Quat,
}

#[derive(Clone, Copy, Debug)]
pub struct TrajectoryCaptureSample {
    pub elapsed_seconds: f64,
    pub knot: TrajectoryInversionKnot,
}

#[derive(Clone, Copy, Debug)]
pub struct GravityBenchmarkSample {
    pub simulation_time_seconds: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub body_rotation: Quat,
}

#[derive(Resource)]
pub struct GravityBenchmarkTrajectory {
    pub epoch: u64,
    pub samples: Vec<GravityBenchmarkSample>,
    pub capture_id: Option<u64>,
    pub complete: bool,
}

impl Default for GravityBenchmarkTrajectory {
    fn default() -> Self {
        Self {
            epoch: u64::MAX,
            samples: Vec::with_capacity(
                (BENCHMARK_DURATION_SECONDS / BENCHMARK_SAMPLE_INTERVAL_SECONDS) as usize + 2,
            ),
            capture_id: None,
            complete: false,
        }
    }
}
#[derive(Clone, Debug)]
pub struct InvertedDensityVoxel {
    /// Body-fixed centre in metres (the gravity source coordinate system).
    pub center: Vec3,
    pub volume: f32,
    pub density: f32,
    /// Geometry/total-mass-only optimization prior. This must never contain the
    /// original density law used later for validation.
    pub baseline_density: f32,
    /// Original forward-model density used only after optimization to score the
    /// recovered field (uniform Werner, logarithmic for all other methods).
    pub reference_density: f32,
    pub grid: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct DensityInversionResult {
    pub method: ActiveGravityMethod,
    /// Immutable identity of the sixteen trajectory states used by this run.
    pub capture_id: u64,
    pub source_hash: u64,
    pub capture_epoch: u64,
    pub problem_id: u64,
    pub initial_objective: f64,
    pub data_error_scale: f64,
    pub density: f32,
    pub density_scale: f32,
    pub objective: f64,
    /// Volume-weighted relative RMSE against the density law assumed by the
    /// selected forward model (uniform Werner, logarithmic for the others).
    pub model_deviation: f32,
    /// `1 - model_deviation`, clamped to [0, 1], for direct UI comparison.
    pub model_fit: f32,
    /// Relative decrease of the trajectory objective from the uniform start.
    pub objective_improvement: f32,
    /// Normalized acceleration residual on the same frozen trajectory used to
    /// assemble the QP. This is distinct from density model RMSE.
    pub training_rmse: f32,
    /// Relative acceleration residual on a deterministic held-out set of
    /// trajectory states evaluated with the independent reference operator.
    pub holdout_rmse: f32,
    /// Relative diagonal observation-noise model used by the QP weighting.
    pub observation_noise_fraction: f32,
    pub observation_noise_realizations: usize,
    /// CPU time spent assembling and solving this convex QP.
    pub inversion_time_ms: f64,
    pub timing: InversionTimingBreakdown,
    /// Number of acceleration observations sampled along the complete dense
    /// Quintic Hermite trajectory.
    pub trajectory_samples: usize,
    pub iterations: u32,
    pub voxel_size: f32,
    pub voxels: Vec<InvertedDensityVoxel>,
}
#[derive(Debug)]
pub struct ConvexOptimizationJob {
    pub method: ActiveGravityMethod,
    pub capture_id: u64,
    pub source_hash: u64,
    pub capture_epoch: u64,
    pub problem_id: u64,
    pub voxels: Vec<InvertedDensityVoxel>,
    pub basis_sources: VoxelBasisSources,
    pub frozen_samples: Vec<TrajectoryInversionKnot>,
    pub sensitivities: Vec<Vec3>,
    pub observed_accelerations: Vec<Vec3>,
    pub holdout_observations: Vec<Vec3>,
    pub holdout_sensitivities: Vec<Vec3>,
    pub neighbours: Vec<(usize, usize)>,
    pub current_densities: Vec<f32>,
    pub best_densities: Vec<f32>,
    pub initial_objective: f64,
    /// Raw trajectory mismatch of the uniform start. Dividing by this value
    /// prevents the regularizers from overwhelming the very small exterior
    /// gravity signature of an internal mass redistribution.
    pub data_error_scale: f64,
    pub iterations: u32,
    pub voxel_size: f32,
    /// Wall-clock origin of the complete inversion, including method-specific
    /// sensitivity construction/readback and the final Clarabel solve.
    pub started_at: bevy::platform::time::Instant,
    pub source_preparation_ms: f64,
    pub timing: InversionTimingBreakdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryVectorField {
    Position,
    Velocity,
}

/// UI and rendering state for the user-provided Quintic Hermite trajectory.
/// `capture_epoch` keeps defaults tied to the currently running simulation;
/// changing a probe parameter starts a fresh five-second capture automatically.
#[derive(Resource)]
pub struct TrajectoryInversionState {
    /// Current simulation epoch observed by the capture system. This is
    /// separate from `capture_epoch` so method changes can retain one frozen
    /// trajectory without pretending it was sampled again.
    pub runtime_epoch: u64,
    pub capture_epoch: u64,
    pub last_capture_request_id: Option<u64>,
    pub wall_elapsed_seconds: f64,
    pub raw_samples: Vec<TrajectoryCaptureSample>,
    pub knots: Vec<TrajectoryInversionKnot>,
    /// Frozen synthetic truth track generated from the logarithmic-density
    /// radial source. Non-Werner inverse methods reuse this exact track.
    pub truth_knots: Vec<TrajectoryInversionKnot>,
    pub truth_capture_id: Option<u64>,
    pub truth_capture_epoch: u64,
    /// Source identity paired with `truth_capture_id`. Method switches must
    /// restore both values or the next inversion looks like a different
    /// physical problem and invalidates the accumulated comparison results.
    pub truth_source_hash: u64,
    /// Long radial truth path used for the common non-Werner display.
    pub truth_orbit: Vec<Vec3>,
    pub preserve_truth_track: bool,
    pub capture_id: Option<u64>,
    pub capture_source_hash: u64,
    /// Eq.106 capture does not begin until consecutive certified readbacks
    /// have arrived; adaptive segment boundaries do not break the streak.
    pub certified_sample_streak: u32,
    pub certified_segment_id: Option<u64>,
    pub ready: bool,
    /// True after a user edits any captured position or velocity. Captured
    /// knots can use the forward evaluator's acceleration directly; edited
    /// knots must derive observations from their user-provided velocities.
    pub knots_edited: bool,
    pub inverted: bool,
    pub selected: Option<(usize, TrajectoryVectorField)>,
    pub edit_buffer: String,
    pub error: Option<String>,
    pub optimizer: Option<ConvexOptimizationJob>,
    pub batch_capture_id: Option<u64>,
    /// Method-independent high-resolution truth observations cached for one
    /// immutable trajectory/source identity and reused across inverse methods.
    pub reference_cache_capture_id: Option<u64>,
    pub reference_cache_source_hash: u64,
    pub reference_training_observations: Vec<Vec3>,
    pub reference_training_sensitivities: Vec<Vec3>,
    pub reference_holdout_observations: Vec<Vec3>,
    pub reference_holdout_sensitivities: Vec<Vec3>,
    pub results: [Option<DensityInversionResult>; 5],
    /// Best fit seen for each method across method switches. Historical only;
    /// current-trajectory comparisons continue to use `results`.
    pub best_results: [Option<DensityInversionResult>; 5],
    pub displayed_density: Option<DensityInversionResult>,
}

impl Default for TrajectoryInversionState {
    fn default() -> Self {
        Self {
            runtime_epoch: 0,
            capture_epoch: 0,
            last_capture_request_id: None,
            wall_elapsed_seconds: 0.0,
            raw_samples: Vec::with_capacity(384),
            knots: Vec::with_capacity(TRAJECTORY_INVERSION_SAMPLE_COUNT),
            truth_knots: Vec::with_capacity(TRAJECTORY_INVERSION_SAMPLE_COUNT),
            truth_capture_id: None,
            truth_capture_epoch: 0,
            truth_source_hash: 0,
            truth_orbit: Vec::with_capacity(ORBIT_HISTORY_LEN),
            preserve_truth_track: false,
            capture_id: None,
            capture_source_hash: 0,
            certified_sample_streak: 0,
            certified_segment_id: None,
            ready: false,
            knots_edited: false,
            inverted: false,
            selected: None,
            edit_buffer: String::new(),
            error: None,
            optimizer: None,
            batch_capture_id: None,
            reference_cache_capture_id: None,
            reference_cache_source_hash: 0,
            reference_training_observations: Vec::new(),
            reference_training_sensitivities: Vec::new(),
            reference_holdout_observations: Vec::new(),
            reference_holdout_sensitivities: Vec::new(),
            results: std::array::from_fn(|_| None),
            best_results: std::array::from_fn(|_| None),
            displayed_density: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JacobiSample {
    pub simulation_time_seconds: f64,
    pub jacobi_constant: f64,
    /// Eq.106 segment/certificate state associated with this physical sample.
    /// Other gravity methods leave this empty.
    pub eq106_diagnostics: Option<Eq106SampleDiagnostics>,
}

#[derive(Resource)]
pub struct JacobiHistory {
    pub samples: VecDeque<JacobiSample>,
    pub elapsed_simulation_seconds: f64,
    pub origin_simulation_seconds: Option<f64>,
    pub eq106_origin_potential: Option<f64>,
    pub eq106_origin_curve_work: Option<f64>,
    pub last_request_id: Option<u64>,
    pub last_sample_method: Option<ActiveGravityMethod>,
}

impl Default for JacobiHistory {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(JACOBI_HISTORY_CAPACITY),
            elapsed_simulation_seconds: 0.0,
            origin_simulation_seconds: None,
            eq106_origin_potential: None,
            eq106_origin_curve_work: None,
            last_request_id: None,
            last_sample_method: None,
        }
    }
}

impl JacobiHistory {
    pub fn reset(&mut self) {
        self.samples.clear();
        self.elapsed_simulation_seconds = 0.0;
        self.origin_simulation_seconds = None;
        self.eq106_origin_potential = None;
        self.eq106_origin_curve_work = None;
        self.last_request_id = None;
        self.last_sample_method = None;
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SimulationClock {
    pub request_id: u64,
    pub epoch: u64,
    pub elapsed_seconds: f64,
}

impl SimulationClock {
    pub fn advance(&mut self, seconds: f64) {
        self.elapsed_seconds += seconds;
        self.request_id = self.request_id.wrapping_add(1);
    }

    pub fn reset_state(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.request_id = self.request_id.wrapping_add(1);
        self.elapsed_seconds = 0.0;
    }
}

/// Number of fully integrated stable physics frames advanced before presenting
/// the next visual state. This changes throughput, never the integration step.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimulationAcceleration(pub u32);

impl Default for SimulationAcceleration {
    fn default() -> Self {
        Self(MIN_SIMULATION_ACCELERATION)
    }
}

impl SimulationAcceleration {
    pub fn stable_steps(self) -> u32 {
        self.0
            .clamp(MIN_SIMULATION_ACCELERATION, MAX_SIMULATION_ACCELERATION)
    }
}

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub enum CameraMode {
    #[default]
    Overview,
    FollowCassini,
}

#[derive(Resource, Default)]
pub struct ShowNormals(pub bool);

#[derive(Resource, Default)]
pub struct ShowSection(pub bool);

/// Solved density constant for `rho(r)=C ln(1+r/epsilon)`.
#[derive(Resource)]
pub struct DensityC(pub f32);

impl Default for DensityC {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Constant density used by the homogeneous Werner polyhedron model.
#[derive(Resource, Default)]
pub struct WernerDensity(pub f32);

#[derive(Resource)]
pub struct AsteroidTopologyGpuData {
    pub mesh_entity: Option<Entity>,
    pub node_count: u32,
    pub positions: Vec<Vec3>,
    pub triangles: Vec<u32>,
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
}

#[derive(Resource)]
pub struct AsteroidNormalsGpuData(pub Vec<Vec3>);

#[derive(Resource, Clone)]
pub struct NormalsReadbackChannel(pub Arc<Mutex<Option<Vec<[f32; 4]>>>>);

impl Default for NormalsReadbackChannel {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

/// Radial-analytic discretization. Each 32-byte record stores one angular cell
/// and one radial layer as `[direction.xyz, solid_angle]` followed by
/// `[r_inner, r_outer, density, padding]`.
#[derive(Resource)]
pub struct RadialGravitySource {
    pub bytes: Vec<u8>,
}

/// Latest GPU-computed gravity acceleration for Cassini (Ryugu body frame).
#[derive(Resource, Default)]
pub struct GravityAcceleration(pub Vec3);

/// Latest positive gravitational potential U returned by the radial GPU model.
#[derive(Resource, Default)]
pub struct GravityPotential(pub Option<f32>);

/// Browser-visible GPU memory accounting. WebGPU does not expose portable
/// driver VRAM counters, so these values are the exact sizes of the buffers
/// allocated by each project pipeline, reported as an auditable estimate.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemoryEstimate {
    pub bytes: [u64; 5],
}

impl GpuMemoryEstimate {
    pub fn total_bytes(self) -> u64 {
        self.bytes.iter().sum()
    }
}

/// Main-world state captured when a render-world gravity dispatch is submitted.
/// The returned acceleration and potential are only valid for this snapshot.
#[derive(Clone, Debug)]
pub struct GravityRequestSnapshot {
    pub request_id: u64,
    pub epoch: u64,
    pub simulation_time_seconds: f64,
    pub body_position: Vec3,
    pub ryugu_transform: Transform,
    pub probe_position: Vec3,
    pub probe_velocity: Vec3,
}

#[derive(Clone, Debug)]
pub struct GravityReadbackPacket {
    pub partial_sums: Vec<[f32; 4]>,
    pub snapshot: GravityRequestSnapshot,
}

#[derive(Clone, Debug)]
pub struct Eq106ReadbackPacket {
    pub partial_sums: Vec<[f32; 4]>,
    pub snapshots: Vec<GravityRequestSnapshot>,
    pub batch_capture_id: Option<u64>,
    /// Number of compact, column-major sensitivity blocks in `partial_sums`.
    /// Zero denotes the ordinary nine-row runtime/trajectory output layout.
    pub sensitivity_column_count: u32,
    pub sensitivity_source_hash: u64,
    pub sensitivity_basis_hash: u64,
    pub sensitivity_configuration_hash: u64,
    pub timings: Eq106TimingSample,
}

#[derive(Clone, Debug)]
pub struct GravityFieldSample {
    pub snapshot: GravityRequestSnapshot,
    /// Batch elements after the first are predicted GPU anchors used only by
    /// the integrator; diagnostics must never treat them as observed states.
    pub predictive: bool,
    pub body_acceleration: Vec3,
    pub positive_potential: f32,
    /// Optional independent full-space potential used by Eq. (157). It must
    /// never replace `positive_potential` in Jacobi calculations because that
    /// field is paired with the acceleration actually driving the trajectory.
    #[cfg(feature = "eq106-dual-certificate")]
    pub independent_positive_potential: Option<f32>,
    /// Optional body-frame Jacobian d(acceleration)/d(position). Eq.106
    /// supplies a symmetric potential Hessian so the Volterra/Picard waveform
    /// can close its position-field loop between GPU readbacks.
    pub body_acceleration_jacobian: Option<Mat3>,
    /// Runtime evidence needed to correlate Jacobi spikes with Eq.106 segment
    /// rebuilds and the four independent truncation/spectral certificates.
    pub eq106_diagnostics: Option<Eq106SampleDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Eq106SampleDiagnostics {
    pub segment_id: u64,
    pub line_origin: Vec3,
    pub line_direction: Vec3,
    pub h: f32,
    pub u: f32,
    pub v: f32,
    pub certificates: [f32; 4],
}

#[derive(Default)]
pub struct GravitySampleHistory {
    pub samples: VecDeque<GravityFieldSample>,
}

impl GravitySampleHistory {
    pub fn push(&mut self, sample: GravityFieldSample) {
        if self
            .samples
            .back()
            .is_some_and(|latest| latest.snapshot.request_id == sample.snapshot.request_id)
        {
            self.samples.pop_back();
        }
        if self.samples.len() == GRAVITY_SAMPLE_HISTORY_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn latest_for_epoch(&self, epoch: u64) -> Option<&GravityFieldSample> {
        self.samples
            .iter()
            .rev()
            .find(|sample| sample.snapshot.epoch == epoch)
    }

    pub fn at_or_before(
        &self,
        epoch: u64,
        simulation_time_seconds: f64,
    ) -> Option<&GravityFieldSample> {
        self.samples.iter().rev().find(|sample| {
            sample.snapshot.epoch == epoch
                && sample.snapshot.simulation_time_seconds <= simulation_time_seconds + 1.0e-6
        })
    }

    /// Returns the temporal anchors surrounding a fixed-update interval. Eq.106
    /// uses them while evaluating each Volterra/Picard waveform iterate.
    pub fn bracketing(
        &self,
        epoch: u64,
        simulation_time_seconds: f64,
    ) -> Option<(&GravityFieldSample, &GravityFieldSample)> {
        let lower = self.samples.iter().rev().find(|sample| {
            sample.snapshot.epoch == epoch
                && sample.snapshot.simulation_time_seconds <= simulation_time_seconds + 1.0e-6
        })?;
        let upper = self
            .samples
            .iter()
            .find(|sample| {
                sample.snapshot.epoch == epoch
                    && sample.snapshot.simulation_time_seconds >= simulation_time_seconds - 1.0e-6
            })
            .unwrap_or(lower);
        Some((lower, upper))
    }

    /// Returns the newest completed GPU anchor at or before the requested
    /// simulation time. Accelerated Eq.106 batches also contain predictive
    /// anchors; diagnostics must skip over them instead of selecting one and
    /// then abandoning the whole update.
    pub fn completed_at_or_before(
        &self,
        epoch: u64,
        simulation_time_seconds: f64,
    ) -> Option<&GravityFieldSample> {
        self.samples.iter().rev().find(|sample| {
            !sample.predictive
                && sample.snapshot.epoch == epoch
                && sample.snapshot.simulation_time_seconds <= simulation_time_seconds + 1.0e-6
        })
    }
}

#[derive(Resource, Default)]
pub struct RadialGravityHistory(pub GravitySampleHistory);

#[derive(Resource, Default)]
pub struct WernerGravityHistory(pub GravitySampleHistory);

/// Snapshot-aligned samples produced by the GPU Equation (106) pipeline.
#[derive(Resource, Default)]
pub struct Eq106GpuHistory(pub GravitySampleHistory);

#[derive(Resource, Default)]
pub struct Eq106TrajectoryBatchResult {
    pub capture_id: Option<u64>,
    pub samples: Vec<GravityFieldSample>,
}

#[derive(Resource, Default)]
pub struct Eq106SensitivityMatrix {
    pub capture_id: Option<u64>,
    pub source_hash: u64,
    pub basis_hash: u64,
    /// Compile-time Eq.106 frequency, quadrature, and Taylor configuration.
    pub configuration_hash: u64,
    pub voxel_count: usize,
    pub sample_count: usize,
    /// Columns are stored in voxel order; each column contains all frozen
    /// trajectory acceleration responses for unit voxel density.
    pub columns: Vec<Vec<Vec3>>,
}

#[derive(Resource, Clone)]
pub struct Eq106GpuReadbackChannel {
    pub data: Arc<Mutex<Option<Eq106ReadbackPacket>>>,
    pub pipeline_error: Arc<Mutex<Option<String>>>,
    pub in_flight: Arc<AtomicBool>,
    /// Wall-clock start of the active command submission. The main world uses
    /// this to turn a lost device or hung shader into an explicit failure.
    pub submitted_at: Arc<Mutex<Option<Instant>>>,
    /// Set by the main-world certificate check when the cached local spectral
    /// element must be shortened and rebuilt. The render world consumes it
    /// before submitting another query for the same simulation snapshot.
    pub rebuild_requested: Arc<AtomicBool>,
}
