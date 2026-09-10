use bevy::prelude::*;

use bevy::platform::time::Instant;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
pub const G: f32 = 6.6743e-11;
pub const RYUGU_MASS: f32 = 4.5e11;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub const TIME_SCALE: f32 = 60.0;
pub const BENCHMARK_DURATION_SECONDS: f64 = 901.66;
pub const BENCHMARK_SAMPLE_INTERVAL_SECONDS: f64 = 0.01;
/// Number of real detector positions retained for the visible trajectory.
/// This is deliberately large enough to preserve several long orbital arcs;
/// the renderer still decimates the history to a bounded gizmo stream.
pub const ORBIT_HISTORY_LEN: usize = 100_000;
pub const JACOBI_HISTORY_CAPACITY: usize = 256;
/// Keep at least two complete maximum-acceleration pointwise-field batches.
/// A batch contains the authoritative anchor plus one endpoint for every
/// accelerated stable step (9 samples at 8x). A smaller capacity silently
/// evicts the authoritative sample before the integrator can consume it.
pub const GRAVITY_SAMPLE_HISTORY_CAPACITY: usize = 2 * (MAX_SIMULATION_ACCELERATION as usize + 1);
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
/// Frequency-domain algorithm, MMFFT, and FMM modes:
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
/// Presentation-only transform carried by the Cassini model child.
///
/// The parent entity with [`CassiniMarker`] remains the authoritative
/// simulation state used by physics, collision detection, Jacobi sampling,
/// and protocol publication. This component must never be queried by those
/// systems.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ProbeVisualTransform {
    pub world_translation: Vec3,
}
#[derive(Component)]
pub struct Velocity(pub Vec3);
#[derive(Component)]
pub struct OrbitHistory(pub VecDeque<Vec3>);

/// Dead-reckoned presentation state between authoritative backend samples.
#[derive(Resource, Clone, Debug)]
pub struct ProbeVisualState {
    pub authoritative_position: Vec3,
    pub authoritative_velocity: Vec3,
    pub authoritative_wall_time: Option<Instant>,
    pub authoritative_epoch: u64,
    pub blend_from: Vec3,
    pub blend_started: Option<Instant>,
    pub rendered_position: Vec3,
}

impl Default for ProbeVisualState {
    fn default() -> Self {
        Self {
            authoritative_position: Vec3::ZERO,
            authoritative_velocity: Vec3::ZERO,
            authoritative_wall_time: None,
            authoritative_epoch: u64::MAX,
            blend_from: Vec3::ZERO,
            blend_started: None,
            rendered_position: Vec3::ZERO,
        }
    }
}

impl ProbeVisualState {
    pub fn reset(&mut self, position: Vec3, velocity: Vec3, epoch: u64, now: Instant) {
        self.authoritative_position = position;
        self.authoritative_velocity = velocity;
        self.authoritative_wall_time = Some(now);
        self.authoritative_epoch = epoch;
        self.blend_from = position;
        self.blend_started = None;
        self.rendered_position = position;
    }

    pub fn accept_authoritative_sample(
        &mut self,
        position: Vec3,
        velocity: Vec3,
        epoch: u64,
        now: Instant,
    ) {
        self.blend_from = self.rendered_position;
        self.authoritative_position = position;
        self.authoritative_velocity = velocity;
        self.authoritative_wall_time = Some(now);
        self.authoritative_epoch = epoch;
        self.blend_started = Some(now);
    }
}

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
pub struct GravityBenchmarkSample {
    pub simulation_time_seconds: f64,
    pub position: Vec3,
    pub velocity: Vec3,
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

/// Browser/native state message produced by the Basilisk compatibility layer.
/// Positions and velocities are SI values; time is kept as seconds here for
/// the UI and is also exported as integer nanoseconds in the wire header.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default)]
pub struct BasiliskSnapshot {
    pub algorithm: u8,
    pub sequence: u64,
    pub protocol_version: u16,
    pub simulation_time_ns: u64,
    pub simulation_time_seconds: f64,
    pub epoch: u64,
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BasiliskComparisonSample {
    pub sample_count: u64,
    pub relative_acceleration_error: f64,
    pub reference_acceleration_mps2: [f64; 3],
    pub measured_acceleration_mps2: [f64; 3],
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Resource, Clone, Debug)]
pub struct BasiliskBridgeState {
    pub protocol: &'static str,
    pub version: u16,
    pub fixed_step_seconds: f64,
    pub snapshot: Option<BasiliskSnapshot>,
    pub comparisons: [BasiliskComparisonSample; 5],
}

impl Default for BasiliskBridgeState {
    fn default() -> Self {
        Self {
            protocol: "ryugu-basilisk-v1",
            version: 1,
            fixed_step_seconds: f64::from(TIME_SCALE) / 60.0,
            snapshot: None,
            comparisons: [BasiliskComparisonSample::default(); 5],
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

// Browser/JSON-facing result contract: several fields are consumed only by
// the WASM snapshot serializer and density overlay.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct DensityInversionResult {
    pub method: ActiveGravityMethod,
    /// Source identity is retained to reject stale results after a mesh rebuild.
    pub source_hash: u64,
    pub density: f32,
    pub density_scale: f32,
    /// Volume-weighted relative RMSE against the density law assumed by the
    /// selected forward model (uniform Werner, logarithmic for the others).
    pub model_deviation: f32,
    /// `1 - model_deviation`, clamped to [0, 1], for direct UI comparison.
    pub model_fit: f32,
    /// Normalized acceleration residual on the same frozen trajectory used to
    /// assemble the QP. This is distinct from density model RMSE.
    pub training_rmse: f32,
    /// Relative acceleration residual on a deterministic held-out set of
    /// trajectory states evaluated with the CPU frequency-domain reference.
    pub holdout_rmse: f32,
    /// CPU time spent assembling and solving this convex QP.
    pub inversion_time_ms: f64,
    pub voxel_size: f32,
    pub voxels: Vec<InvertedDensityVoxel>,
}
#[derive(Debug)]
pub struct ConvexOptimizationJob {
    pub method: ActiveGravityMethod,
    pub capture_id: u64,
    pub source_hash: u64,
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
    pub voxel_size: f32,
    /// Wall-clock origin of the complete inversion, including method-specific
    /// sensitivity construction/readback and the final Clarabel solve.
    pub started_at: bevy::platform::time::Instant,
    pub source_preparation_ms: f64,
    pub timing: InversionTimingBreakdown,
}

/// Backend state for the frozen Quintic Hermite trajectory.
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
    pub ready: bool,
    /// The browser frontend may request inversion before the five-second capture
    /// is complete. Keep the request in the state machine until validation can
    /// actually start the optimizer.
    pub start_requested: bool,
    pub inverted: bool,
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
            knots: Vec::with_capacity(TRAJECTORY_INVERSION_SAMPLE_COUNT),
            truth_knots: Vec::with_capacity(TRAJECTORY_INVERSION_SAMPLE_COUNT),
            truth_capture_id: None,
            truth_capture_epoch: 0,
            truth_source_hash: 0,
            truth_orbit: Vec::with_capacity(ORBIT_HISTORY_LEN),
            preserve_truth_track: false,
            capture_id: None,
            capture_source_hash: 0,
            ready: false,
            start_requested: false,
            inverted: false,
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

// Browser chart contract; native test builds do not compile the HTML adapter.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct JacobiSample {
    pub simulation_time_seconds: f64,
    pub jacobi_constant: f64,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct PerformanceDiagnosticSample {
    pub simulation_time_seconds: f64,
    pub value: f64,
}

#[derive(Resource)]
pub struct JacobiHistory {
    pub samples: VecDeque<JacobiSample>,
    pub elapsed_simulation_seconds: f64,
    pub origin_simulation_seconds: Option<f64>,
    pub last_request_id: Option<u64>,
    pub last_sample_method: Option<ActiveGravityMethod>,
}

impl Default for JacobiHistory {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(JACOBI_HISTORY_CAPACITY),
            elapsed_simulation_seconds: 0.0,
            origin_simulation_seconds: None,
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
        self.last_request_id = None;
        self.last_sample_method = None;
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct SimulationClock {
    pub request_id: u64,
    pub epoch: u64,
    pub elapsed_seconds: f64,
    /// The physical integration interval represented by one FixedUpdate.
    /// Keeping this in the clock makes the WASM and native adapters consume
    /// the same step instead of deriving it from render-frame timing.
    pub fixed_step_seconds: f64,
}

impl Default for SimulationClock {
    fn default() -> Self {
        Self {
            request_id: 0,
            epoch: 0,
            elapsed_seconds: 0.0,
            fixed_step_seconds: 1.0,
        }
    }
}

impl SimulationClock {

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

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
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

/// Density profile used by the surface-field product. The real-time Werner
/// trajectory path remains homogeneous; this switch controls the explicit
/// surface validation product and its comparison maps.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DensityMode {
    #[default]
    Variable,
    Constant,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl DensityMode {
    pub fn key(self) -> &'static str {
        match self {
            Self::Variable => "variable",
            Self::Constant => "constant",
        }
    }
}

/// Quantity used to color the latest surface-field overlay.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SurfaceFieldMetric {
    #[default]
    Gravity,
    Gradient,
    Slope,
    Error,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl SurfaceFieldMetric {
    pub fn key(self) -> &'static str {
        match self {
            Self::Gravity => "gravity",
            Self::Gradient => "gradient",
            Self::Slope => "slope",
            Self::Error => "error",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Gravity => "Effective gravity",
            Self::Gradient => "Gravity gradient",
            Self::Slope => "Effective slope",
            Self::Error => "Relative error",
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceFieldSample {
    pub position: Vec3,
    pub normal: Vec3,
    pub gravity: Vec3,
    pub effective_gravity: Vec3,
    pub gravity_magnitude: f32,
    pub effective_gravity_magnitude: f32,
    pub gradient_magnitude: f32,
    pub slope_degrees: f32,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct SurfaceFieldDataset {
    pub method: ActiveGravityMethod,
    pub density_mode: DensityMode,
    pub samples: Vec<SurfaceFieldSample>,
    pub gravity_range: (f32, f32),
    pub effective_gravity_range: (f32, f32),
    pub gradient_range: (f32, f32),
    pub slope_range: (f32, f32),
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct SurfaceFieldComparison {
    pub baseline: SurfaceFieldDataset,
    pub comparison: SurfaceFieldDataset,
    /// Signed relative effective-gravity error at each common surface sample.
    /// Positive means comparison > baseline; negative means comparison < baseline.
    pub signed_errors: Vec<f32>,
    pub error_range: (f32, f32),
}

#[derive(Resource, Debug)]
pub struct SurfaceFieldState {
    pub computing: bool,
    pub status: String,
    pub revision: u64,
    pub metric: SurfaceFieldMetric,
    pub baseline_method: ActiveGravityMethod,
    pub comparison_method: ActiveGravityMethod,
    pub latest: Option<SurfaceFieldDataset>,
    pub comparison: Option<SurfaceFieldComparison>,
    pub selected_patch: Option<usize>,
}

impl Default for SurfaceFieldState {
    fn default() -> Self {
        Self {
            computing: false,
            status: "Surface field is ready to compute.".into(),
            revision: 0,
            metric: SurfaceFieldMetric::Gravity,
            baseline_method: ActiveGravityMethod::Fmm,
            comparison_method: ActiveGravityMethod::MmfftCompressed,
            latest: None,
            comparison: None,
            selected_patch: None,
        }
    }
}

#[derive(Resource)]
pub struct AsteroidTopologyGpuData {
    pub mesh_entity: Option<Entity>,
    pub node_count: u32,
    pub positions: Vec<Vec3>,
    pub triangles: Vec<u32>,
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
}

/// GPU-produced vertex normals are retained for the normals compute/readback
/// path. The scene overlay now derives one authoritative normal per triangle
/// directly from topology, so this cache is intentionally not read by the
/// gizmo renderer.
#[allow(dead_code)]
#[derive(Resource)]
pub struct AsteroidNormalsGpuData(pub Vec<Vec3>);

#[derive(Resource, Clone)]
pub struct NormalsReadbackChannel(pub Arc<Mutex<Option<Vec<[f32; 4]>>>>);

impl Default for NormalsReadbackChannel {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

impl NormalsReadbackChannel {
    pub fn reset_after_device_loss(&self) {
        if let Ok(mut data) = self.0.try_lock() {
            data.take();
        }
    }
}

/// Radial-analytic discretization. Each 32-byte record stores one angular cell
/// and one radial layer as `[direction.xyz, solid_angle]` followed by
/// `[r_inner, r_outer, density, padding]`.
#[derive(Resource)]
pub struct DensityQuadratureSource {
    /// Two vec4 records per volume cell:
    /// `[direction.xyz, solid_angle]` and `[r_inner, r_outer, density, _]`.
    pub bytes: Vec<u8>,
    pub constant_bytes: Vec<u8>,
    pub radius: f32,
    pub source_hash: u64,
    pub constant_hash: u64,
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

/// Main-world state captured when a render-world gravity dispatch is submitted.
/// The returned acceleration and potential are only valid for this snapshot.
#[derive(Clone, Debug)]
pub struct GravityRequestSnapshot {
    pub request_id: u64,
    pub epoch: u64,
    pub simulation_time_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct GravityReadbackPacket {
    pub partial_sums: Vec<[f32; 4]>,
    pub snapshot: GravityRequestSnapshot,
}

#[derive(Clone, Debug)]
pub struct FrequencyDomainReadbackPacket {
    pub partial_sums: Vec<[f32; 4]>,
    /// Number of independent equation-(184) observations. Each observation
    /// integrates the complete uploaded trajectory at one Laplace frequency.
    pub observation_count: u32,
    pub batch_capture_id: Option<u64>,
    /// Number of compact, column-major sensitivity blocks in `partial_sums`.
    /// Zero denotes the ordinary eleven-row trajectory-observation layout.
    pub sensitivity_column_count: u32,
    pub sensitivity_source_hash: u64,
    pub sensitivity_basis_hash: u64,
    pub sensitivity_configuration_hash: u64,
    pub timings: FrequencyDomainTimingSample,
}

#[derive(Clone, Debug)]
pub struct GravityFieldSample {
    pub snapshot: GravityRequestSnapshot,
    pub body_acceleration: Vec3,
    pub positive_potential: f32,
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
}

#[derive(Resource, Default)]
pub struct RadialGravityHistory(pub GravitySampleHistory);

#[derive(Resource, Default)]
pub struct WernerGravityHistory(pub GravitySampleHistory);

/// One complete equation-(184) transform evaluated at a single positive
/// Laplace frequency. This is deliberately not a `GravityFieldSample`: it is
/// an aggregate over the whole known trajectory and cannot drive pointwise
/// dynamics or snapshot-based diagnostics.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct FrequencyDomainObservation {
    pub laplace_frequency: f32,
    pub transformed_field: Vec3,
    pub transformed_jacobian: Mat3,
    pub transformed_potential: f32,
}

#[derive(Resource, Default)]
pub struct FrequencyDomainTrajectoryBatchResult {
    pub capture_id: Option<u64>,
    pub observations: Vec<FrequencyDomainObservation>,
    pub revision: u64,
}

#[derive(Resource, Default)]
pub struct FrequencyDomainSensitivityMatrix {
    pub capture_id: Option<u64>,
    pub source_hash: u64,
    pub basis_hash: u64,
    /// Compile-time Frequency-domain algorithm reciprocal-space configuration.
    pub configuration_hash: u64,
    pub voxel_count: usize,
    pub sample_count: usize,
    /// Columns are stored in voxel order; each entry is an independent
    /// full-trajectory equation-(184) response at one Laplace frequency.
    pub columns: Vec<Vec<Vec3>>,
}

#[derive(Resource, Clone)]
pub struct FrequencyDomainGpuReadbackChannel {
    pub data: Arc<Mutex<Option<FrequencyDomainReadbackPacket>>>,
    pub pipeline_error: Arc<Mutex<Option<String>>>,
    pub in_flight: Arc<AtomicBool>,
    /// Wall-clock start of the active command submission. The main world uses
    /// this to turn a lost device or hung shader into an explicit failure.
    pub submitted_at: Arc<Mutex<Option<Instant>>>,
    /// Requests reconstruction after a readback or device error.
    pub rebuild_requested: Arc<AtomicBool>,
}

impl FrequencyDomainGpuReadbackChannel {
    pub fn reset_after_device_loss(&self) {
        if let Ok(mut data) = self.data.try_lock() {
            data.take();
        }
        if let Ok(mut error) = self.pipeline_error.try_lock() {
            error.take();
        }
        if let Ok(mut submitted) = self.submitted_at.try_lock() {
            submitted.take();
        }
        self.rebuild_requested.store(false, Ordering::Release);
        self.in_flight.store(false, Ordering::Release);
    }
}
