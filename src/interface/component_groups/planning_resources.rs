pub const PROBE_R0: Vec3 = Vec3::new(-616.535, 0.0, -65.459);
pub const PROBE_SPEED_FACTOR: f32 = 1.07;
pub const PROBE_ORBIT_NORMAL: Vec3 = Vec3::new(0.037_806, -0.933_691, -0.356_079);

pub const NEAR_SYNC_SEGMENT_MAX_SECONDS: f32 = 300.0;

pub const PLANNING_GRAVITY_ERROR_LIMIT: f32 = 2.0e-2;
pub const PLANNING_GRADIENT_ERROR_LIMIT: f32 = 2.5e-1;
pub const PLANNING_PERICENTER_ERROR_LIMIT_METERS: f32 = 1.0;

/// Reporting policy only: changing it never changes numerical outputs or
/// reference samples. Screening is explicitly unsuitable for strict claims.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanningAccuracyProfile {
    #[default]
    Strict,
    Screening,
}

#[derive(Clone, Copy)]
pub struct PlanningAccuracyLimits {
    pub gravity: f32,
    pub gradient: f32,
    pub gravity_p99: f32,
    pub gradient_p99: f32,
    pub gravity_max: f32,
    pub gradient_max: f32,
    pub pericenter_m: f32,
}

impl PlanningAccuracyProfile {
    pub fn key(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Screening => "screening",
        }
    }

    pub fn limits(self) -> PlanningAccuracyLimits {
        match self {
            Self::Strict => PlanningAccuracyLimits {
                // Equation (184) is evaluated with a finite reciprocal-space
                // quadrature.  Keep strict gates meaningful but account for
                // the declared spectral truncation and interpolation error.
                gravity: PLANNING_GRAVITY_ERROR_LIMIT,
                gradient: PLANNING_GRADIENT_ERROR_LIMIT,
                gravity_p99: 2.5 * PLANNING_GRAVITY_ERROR_LIMIT,
                gradient_p99: 2.0 * PLANNING_GRADIENT_ERROR_LIMIT,
                gravity_max: 5.0 * PLANNING_GRAVITY_ERROR_LIMIT,
                gradient_max: 4.0 * PLANNING_GRADIENT_ERROR_LIMIT,
                pericenter_m: PLANNING_PERICENTER_ERROR_LIMIT_METERS,
            },
            Self::Screening => PlanningAccuracyLimits {
                gravity: 0.02,
                gradient: 0.25,
                gravity_p99: 0.05,
                gradient_p99: 0.50,
                gravity_max: 0.10,
                gradient_max: 1.0,
                pericenter_m: 10.0,
            },
        }
    }
}

pub fn planning_accuracy_failure_labels(mask: u32) -> Vec<&'static str> {
    [
        "GPU/workload verification",
        "gravity RMS",
        "gradient RMS",
        "gravity p99/max",
        "gradient p99/max",
        "pericenter drift",
        "candidate coverage/score",
        "invalid timing",
        "missing/common reference samples",
        "external validation rejection",
    ]
    .into_iter()
    .enumerate()
    .filter_map(|(bit, reason)| (mask & (1 << bit) != 0).then_some(reason))
    .collect()
}
pub const PLANNING_SOURCE_COUNTS: [u32; 9] = [
    32_000, 64_000, 128_000, 256_000, 512_000, 1_024_000, 2_048_000, 4_096_000, 8_192_000,
];
pub const PLANNING_SOURCE_REPEATS: u32 = 7;
pub const PLANNING_DENSITY_MODEL_COUNTS: [u32; 7] = [1, 4, 16, 64, 256, 512, 1024];
pub const PLANNING_TARGET_COUNTS: [u32; 5] = [8, 64, 241, 1024, 8192];
pub const RYUGU_COLLISION_RADIUS_METERS: f32 = 464.765;
pub const PROBE_COLLISION_RADIUS_METERS: f32 = 3.35;

pub fn probe_initial_velocity_for_normal(
    position: Vec3,
    speed_factor: f32,
    orbit_normal: Vec3,
) -> Vec3 {
    let radius = position.length();
    if !radius.is_finite() || radius <= f32::EPSILON {
        return Vec3::ZERO;
    }
    let radial = position / radius;
    let tangent = orbit_normal
        .normalize_or_zero()
        .cross(radial)
        .normalize_or_zero();
    let speed = speed_factor.clamp(0.0, 2.0) * (G * RYUGU_MASS / radius).sqrt();
    tangent * speed
}

pub fn probe_initial_velocity(position: Vec3, speed_factor: f32) -> Vec3 {
    probe_initial_velocity_for_normal(position, speed_factor, PROBE_ORBIT_NORMAL)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProbeOrbitPreset {
    #[default]
    CurrentBenchmark,
    Custom,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ProbeInitialConditions {
    pub position: Vec3,
    pub speed_factor: f32,
    pub orbit_normal: Vec3,
    pub preset: ProbeOrbitPreset,
}

impl Default for ProbeInitialConditions {
    fn default() -> Self {
        Self {
            position: PROBE_R0,
            speed_factor: PROBE_SPEED_FACTOR,
            orbit_normal: PROBE_ORBIT_NORMAL,
            preset: ProbeOrbitPreset::CurrentBenchmark,
        }
    }
}

impl ProbeInitialConditions {
    pub fn velocity(self) -> Vec3 {
        if self.preset == ProbeOrbitPreset::CurrentBenchmark
            && self.orbit_normal == PROBE_ORBIT_NORMAL
        {
            probe_initial_velocity(self.position, self.speed_factor)
        } else {
            probe_initial_velocity_for_normal(self.position, self.speed_factor, self.orbit_normal)
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanningWorkloadProfile {
    #[default]
    First,
    /// Large, interruptible workload for exercising the planning pipelines
    /// without taking ownership of the browser frame loop.
    InteractiveStress,
    /// Fixed geometry crossover over source count, density RHS count and
    /// target count, with independent repeated timings in every sweep cell.
    SourceCrossover,
}

impl PlanningWorkloadProfile {
    pub fn is_compute_benchmark(self) -> bool {
        matches!(self, Self::First | Self::SourceCrossover)
    }

    pub fn dimensions(self) -> (u32, u32, u32) {
        match self {
            Self::First => (PLANNING_FIRST_CANDIDATE_COUNT, 4, 241),
            Self::InteractiveStress => (PLANNING_CANDIDATE_COUNT, 32, 512),
            // Initial sweep cell. PlanningComparisonState supplies the current
            // K_rho x N_t dimensions for every subsequent cell.
            Self::SourceCrossover => (
                1,
                PLANNING_DENSITY_MODEL_COUNTS[0],
                PLANNING_TARGET_COUNTS[0],
            ),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::First => "First 32x4x241",
            Self::InteractiveStress => "Stress 2048x32x512",
            Self::SourceCrossover => "Quadrature-source/density/target crossover",
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComparisonMetric {
    #[default]
    DensityFit,
    InversionTime,
    GravityRelativeError,
    GradientRelativeError,
    PericenterError,
    MinimumAltitude,
    ModelDiscrimination,
    PlanningObjective,
    SegmentCount,
    SpeedupVsGpuFmm,
    ColdStartAmortization,
}

impl ComparisonMetric {
    pub fn is_inversion(self) -> bool {
        matches!(self, Self::DensityFit | Self::InversionTime)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanningWorkloadIdentity {
    pub reference_capture_id: u64,
    pub reference_capture_epoch: u64,
    pub source_hash: u64,
    pub source_count: u32,
    pub basis_hash: u64,
    pub reference_arc_hash: u64,
    pub candidate_hash: u64,
    pub density_model_hash: u64,
    pub sample_hash: u64,
    pub tolerance_hash: u64,
    pub candidate_count: u32,
    pub density_model_count: u32,
    pub samples_per_candidate: u32,
    pub outputs: u8,
}

impl PlanningWorkloadIdentity {
    pub const GRAVITY: u8 = 1;
    pub const GRADIENT: u8 = 2;
    pub const MINIMUM_ALTITUDE: u8 = 4;
    pub const OBJECTIVE: u8 = 8;
    pub const REQUIRED_OUTPUTS: u8 =
        Self::GRAVITY | Self::GRADIENT | Self::MINIMUM_ALTITUDE | Self::OBJECTIVE;

    pub fn is_complete(self) -> bool {
        self.reference_capture_id != 0
            && self.source_hash != 0
            && self.basis_hash != 0
            && self.reference_arc_hash != 0
            && self.candidate_hash != 0
            && self.density_model_hash != 0
            && self.sample_hash != 0
            && self.tolerance_hash != 0
            && self.outputs & Self::REQUIRED_OUTPUTS == Self::REQUIRED_OUTPUTS
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum PlanningExecutionBackend {
    GpuFrequencyDomain,
    GpuMmfft,
    GpuFmm,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanningCandidateScore {
    pub objective: f32,
}

impl Default for PlanningCandidateScore {
    fn default() -> Self {
        Self {
            objective: f32::NAN,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct PlanningMethodMetrics {
    pub method: ActiveGravityMethod,
    pub backend: PlanningExecutionBackend,
    /// True only when this row came from the named method's GPU batch path.
    /// Shared validation runs may populate diagnostics, but cannot unlock a
    /// GPU fairness verdict.
    pub gpu_batch_verified: bool,
    pub workload: PlanningWorkloadIdentity,
    /// Measured certified hot pass over the complete BxKxH workload. The
    /// immutable geometry/basis build is reused and reported separately.
    pub certified_full_pass_ms: f64,
    /// Full raw cost plus the measured additional checked pass, with the
    /// immutable basis charged exactly once. Shared f64 references are separate.
    pub certified_estimated_total_ms: f64,
    pub raw_kernels: PlanningKernelTotals,
    pub checked_kernels: PlanningKernelTotals,
    pub external_validation_ms: f64,
    pub total_ms: f64,
    /// CPU geometry/basis wall time. GPU basis timestamps are in raw_kernels.
    pub geometry_basis_build_ms: f64,
    /// CPU RHS preparation per density model, excluding immutable basis.
    pub density_model_ms: f64,
    /// GPU request wall time per output, including amortized setup and readback.
    pub target_point_ms: f64,
    pub relative_gravity_error: f32,
    pub gradient_relative_error: f32,
    pub certified_relative_gravity_error: f32,
    pub certified_gradient_relative_error: f32,
    /// Pointwise relative-error strata. Unlike the global L2 metric these
    /// expose weak-field and boundary outliers.
    pub gravity_error_p99: f32,
    pub gravity_error_max: f32,
    pub gradient_error_p99: f32,
    pub gradient_error_max: f32,
    pub certified_gravity_error_p99: f32,
    pub certified_gravity_error_max: f32,
    pub certified_gradient_error_p99: f32,
    pub certified_gradient_error_max: f32,
    pub pericenter_error_m: f32,
    pub minimum_altitude_m: f32,
    pub model_discrimination: f32,
    pub planning_objective: f32,
    pub segment_count: u32,
    pub valid_candidate_count: u32,
    /// Number of common f64 reference states used in the error denominator.
    pub verification_sample_count: u64,
    pub certified_verification_sample_count: u64,
    pub certified_rejected_sample_count: u64,
    pub certified_valid_candidate_count: u32,
    pub cold_amortization_candidates: u32,
    pub top_candidates: [PlanningCandidateScore; 5],
}

pub struct PlanningBatchJob {
    pub run_id: u64,
    pub profile: PlanningWorkloadProfile,
    pub method: ActiveGravityMethod,
    pub method_order: [ActiveGravityMethod; 3],
    pub method_order_index: usize,
    pub batch_id: u64,
    pub candidate_count: u32,
    pub density_model_count: u32,
    pub samples_per_candidate: u32,
    pub density_seed: u64,
    pub maximum_density_mass_relative_error: f64,
    pub request_id: u64,
    pub density_model: u32,
    pub candidate_start: u32,
    pub candidate_tile_size: u32,
    pub minimum_tile_size_used: u32,
    pub maximum_tile_size_used: u32,
    pub gpu_request_count: u32,
    pub raw_gpu_request_count: u32,
    pub last_request_candidate_count: u32,
    pub awaiting_gpu: bool,
    /// Active seconds waiting for the packet belonging to the current request.
    /// A lost WebGPU map/readback must never leave the UI at 0% forever.
    pub awaiting_gpu_seconds: f64,
    pub awaiting_gpu_last_poll: Option<bevy::platform::time::Instant>,
    pub gpu_basis_progress: f64,
    pub reference_inflight_fraction: f64,
    pub gpu_preparation_submission: u32,
    pub warm_repetition: bool,
    pub certified_repetition: bool,
    pub total_evaluations: u64,
    pub gravity_error_sum: f64,
    pub gravity_reference_sum: f64,
    pub gravity_samples: u64,
    pub gradient_error_sum: f64,
    pub gradient_reference_sum: f64,
    pub gradient_samples: u64,
    pub verification_sample_count: u64,
    pub raw_gravity_error_sum: f64,
    pub raw_gradient_error_sum: f64,
    pub pointwise_gravity_errors: Vec<f32>,
    pub pointwise_gradient_errors: Vec<f32>,
    pub certified_pointwise_gravity_errors: Vec<f32>,
    pub certified_pointwise_gradient_errors: Vec<f32>,
    pub certified_gravity_error_sum: f64,
    pub certified_gravity_reference_sum: f64,
    pub certified_gradient_error_sum: f64,
    pub certified_gradient_reference_sum: f64,
    pub certified_gravity_samples: u64,
    pub certified_gradient_samples: u64,
    pub certified_verification_sample_count: u64,
    pub certified_rejected_sample_count: u64,
    pub certified_candidate_valid: Vec<bool>,
    pub rejected_sample_count: u64,
    pub pericenter_error_m: f32,
    pub minimum_altitude_m: f32,
    pub discrimination_sum: f64,
    pub discrimination_reference_sum: f64,
    pub discrimination_samples: u64,
    pub gradient_information_sum: f64,
    pub candidate_discrimination_sum: Vec<f64>,
    pub candidate_reference_sum: Vec<f64>,
    pub candidate_gradient_sum: Vec<f64>,
    pub candidate_minimum_altitude_m: Vec<f32>,
    pub candidate_valid: Vec<bool>,
    pub common_geometry_basis_ms: f64,
    pub method_geometry_basis_ms: f64,
    pub density_payload_preparation_ms: f64,
    pub certified_density_payload_preparation_ms: f64,
    pub raw_kernels: PlanningKernelTotals,
    pub certified_kernels: PlanningKernelTotals,
    pub gpu_preprocessing_ms: f64,
    pub command_submission_ms: f64,
    pub reduction_ms: f64,
    pub certified_reduction_ms: f64,
    pub verification_ms: f64,
    pub gpu_completion_map_ms: f64,
    pub readback_decode_ms: f64,
    pub warm_evaluation_ms: f64,
    pub certified_warm_evaluation_ms: f64,
    pub certified_full_pass_ms: f64,
    pub dispatch_count: u32,
    pub forward_kernel_evaluations: u64,
    pub trajectory_block_count: u32,
}

impl PlanningMethodMetrics {
    pub fn accuracy_eligible(self) -> bool {
        self.accuracy_failure_mask(PlanningAccuracyProfile::Strict, false) == 0
    }

    #[cfg(test)]
    pub fn certified_accuracy_eligible(self) -> bool {
        self.accuracy_failure_mask(PlanningAccuracyProfile::Strict, true) == 0
    }

    /// Keep numerical validity, coverage and external validation failures as
    /// hard gates in every profile. No blanket "show failed timings" option.
    pub fn accuracy_failure_mask(self, profile: PlanningAccuracyProfile, certified: bool) -> u32 {
        let limits = profile.limits();
        let within = |value: f32, limit: f32| value.is_finite() && value >= 0.0 && value <= limit;
        let gravity = if certified {
            self.certified_relative_gravity_error
        } else {
            self.relative_gravity_error
        };
        let gradient = if certified {
            self.certified_gradient_relative_error
        } else {
            self.gradient_relative_error
        };
        let total_ms = if certified {
            self.certified_estimated_total_ms
        } else {
            self.total_ms
        };
        let valid_candidates = if certified {
            self.certified_valid_candidate_count
        } else {
            self.valid_candidate_count
        };
        let (gravity_p99, gravity_max, gradient_p99, gradient_max) = if certified {
            (self.certified_gravity_error_p99, self.certified_gravity_error_max,
                self.certified_gradient_error_p99, self.certified_gradient_error_max)
        } else {
            (self.gravity_error_p99, self.gravity_error_max, self.gradient_error_p99, self.gradient_error_max)
        };
        let failures = [
            !self.gpu_batch_verified || !self.workload.is_complete(),
            !within(gravity, limits.gravity),
            !within(gradient, limits.gradient),
            !within(gravity_p99, limits.gravity_p99)
                || !within(gravity_max, limits.gravity_max),
            !within(gradient_p99, limits.gradient_p99)
                || !within(gradient_max, limits.gradient_max),
            self.method != ActiveGravityMethod::FrequencyDomain
                && !within(self.pericenter_error_m, limits.pericenter_m),
            valid_candidates != self.workload.candidate_count
                || !self.top_candidates[0].objective.is_finite(),
            !total_ms.is_finite()
                || total_ms <= 0.0
                || (certified
                    && (!self.certified_full_pass_ms.is_finite()
                        || self.certified_full_pass_ms <= 0.0
                        || self.certified_estimated_total_ms < self.total_ms)),
            self.verification_sample_count == 0
                || (certified
                    && self.certified_verification_sample_count != self.verification_sample_count),
            certified && self.certified_rejected_sample_count != 0,
        ];
        failures
            .into_iter()
            .enumerate()
            .fold(0, |mask, (bit, failed)| {
                mask | if failed { 1 << bit } else { 0 }
            })
    }
}

// Planning results are also serialized directly into the browser snapshot;
// native-only builds do not read every presentation field.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Resource)]
pub struct PlanningComparisonState {
    pub selected_metric: ComparisonMetric,
    pub accuracy_profile: PlanningAccuracyProfile,
    pub workload_profile: PlanningWorkloadProfile,
    pub results: [Option<PlanningMethodMetrics>; 5],
    pub run_requested: bool,
    pub run_id: u64,
    /// Only the final reduced/read-back result may set this, never a percentage.
    pub computation_complete: bool,
    /// Scope is fixed at launch; display selections do not mutate an active run.
    pub source_curve_all_parameters: bool,
    pub source_curve_run_id: u64,
    pub stopped_operation_work: f64,
    pub reference_duration_seconds: f32,
    pub status: String,
    pub batch_job: Option<PlanningBatchJob>,
    /// Completed fraction of the CPU candidate-preparation stage for the
    /// current workload/source count.
    pub preparation_progress: f64,
    pub requested_source_count: u32,
    pub source_curve_active: bool,
    pub source_curve_visible: bool,
    pub source_curve_index: usize,
    pub source_curve_repeat: u32,
    pub source_curve_density_index: usize,
    pub source_curve_target_index: usize,
    /// Reproducible random method order; independent of density generation.
    pub source_curve_order_seed: u64,
    pub source_curve_samples: Vec<PlanningSourceCurveSample>,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct PlanningSourceCurveSample {
    pub source_count: u32,
    pub density_model_count: u32,
    pub target_count: u32,
    pub repeat: u32,
    pub order_seed: u64,
    pub method_order: [usize; 3],
    /// Frequency-domain algorithm raw/certified, packed FFT raw/certified, FMM raw/certified.
    pub times_ms: [f64; 6],
    pub kernel_times_ms: [Option<f64>; 6],
    pub evaluation_kernel_times_ms: [Option<f64>; 6],
    pub basis_kernel_times_ms: [Option<f64>; 3],
    /// Frequency-domain algorithm, packed FFT, FMM geometry/basis build costs.
    pub geometry_basis_build_ms: [f64; 3],
    /// Frequency-domain algorithm, packed FFT, FMM average cost for adding one density model.
    pub density_model_ms: [f64; 3],
    /// Frequency-domain algorithm, packed FFT, FMM average cost for one density-model target point.
    pub target_point_ms: [f64; 3],
    pub eligible: [bool; 6],
    pub strict_failures: [u32; 6],
    pub screening_failures: [u32; 6],
    pub gravity_errors: [f32; 6],
    pub gradient_errors: [f32; 6],
}

#[derive(Resource, Debug, Default)]
pub struct ProbeCrashState {
    pub active: bool,
    pub elapsed_seconds: f32,
}

#[derive(Resource, Debug, Default)]
pub struct ProbeCrashResetRequest(pub bool);

impl ProbeCrashState {
    pub const DISPLAY_SECONDS: f32 = 3.0;

    pub fn trigger(&mut self) {
        self.active = true;
        self.elapsed_seconds = 0.0;
    }

    pub fn clear(&mut self) {
        self.active = false;
        self.elapsed_seconds = 0.0;
    }
}

impl Default for PlanningComparisonState {
    fn default() -> Self {
        Self {
            selected_metric: ComparisonMetric::DensityFit,
            accuracy_profile: PlanningAccuracyProfile::default(),
            workload_profile: PlanningWorkloadProfile::First,
            results: std::array::from_fn(|_| None),
            run_requested: false,
            run_id: 0,
            computation_complete: false,
            source_curve_all_parameters: false,
            source_curve_run_id: 0,
            stopped_operation_work: 0.0,
            reference_duration_seconds: 0.0,
            status: "Choose a planning metric to run First, or use inversion metrics with the inversion button.".into(),
            batch_job: None,
            preparation_progress: 0.0,
            requested_source_count: PLANNING_SOURCE_COUNTS[0],
            source_curve_active: false,
            source_curve_visible: false,
            source_curve_index: 0,
            source_curve_repeat: 0,
            source_curve_density_index: 0,
            source_curve_target_index: 0,
            source_curve_order_seed: 0,
            source_curve_samples: Vec::new(),
        }
    }
}

impl PlanningComparisonState {
    pub fn dimensions(&self) -> (u32, u32, u32) {
        if self.workload_profile == PlanningWorkloadProfile::SourceCrossover {
            (
                1,
                PLANNING_DENSITY_MODEL_COUNTS[self.source_curve_density_index],
                PLANNING_TARGET_COUNTS[self.source_curve_target_index],
            )
        } else {
            self.workload_profile.dimensions()
        }
    }

    /// Seven fresh batches per cell; source count varies fastest, then K, then N_t.
    pub fn advance_source_curve(&mut self) -> bool {
        self.source_curve_repeat += 1;
        if self.source_curve_repeat == PLANNING_SOURCE_REPEATS {
            self.source_curve_repeat = 0;
            self.source_curve_index += 1;
            if self.source_curve_index == PLANNING_SOURCE_COUNTS.len() {
                if !self.source_curve_all_parameters {
                    self.source_curve_index = PLANNING_SOURCE_COUNTS.len() - 1;
                    self.source_curve_repeat = PLANNING_SOURCE_REPEATS - 1;
                    return false;
                }
                self.source_curve_index = 0;
                self.source_curve_density_index += 1;
                if self.source_curve_density_index == PLANNING_DENSITY_MODEL_COUNTS.len() {
                    self.source_curve_density_index = 0;
                    self.source_curve_target_index += 1;
                    if self.source_curve_target_index == PLANNING_TARGET_COUNTS.len() {
                        // Keep dimensions addressable for the final verdict/UI.
                        self.source_curve_index = PLANNING_SOURCE_COUNTS.len() - 1;
                        self.source_curve_repeat = PLANNING_SOURCE_REPEATS - 1;
                        self.source_curve_density_index = PLANNING_DENSITY_MODEL_COUNTS.len() - 1;
                        self.source_curve_target_index = PLANNING_TARGET_COUNTS.len() - 1;
                        return false;
                    }
                }
            }
        }
        self.requested_source_count = PLANNING_SOURCE_COUNTS[self.source_curve_index];
        true
    }

    /// Estimated arithmetic work with source traversal, basis construction,
    /// FFT butterflies, density combinations, targets and reference validation.
    /// This is not a GPU FLOP counter or an ETA. Only completion can yield 100%.
    pub fn operation_work(&self) -> (f64, f64) {
        let source_curve = self.workload_profile == PlanningWorkloadProfile::SourceCrossover;
        let (b, k, nt) = self.dimensions();
        let ns = self.requested_source_count;
        let one_batch = planning_batch_work(ns, b, k, nt);
        let total = if source_curve {
            PLANNING_SOURCE_COUNTS.iter().map(|&sources| {
                if self.source_curve_all_parameters {
                    PLANNING_TARGET_COUNTS.iter().map(|&targets| {
                        PLANNING_DENSITY_MODEL_COUNTS.iter().map(|&density| {
                            planning_source_cell_work(sources, density, targets)
                        }).sum::<f64>()
                    }).sum::<f64>()
                } else { planning_source_cell_work(sources, k, nt) }
            }).sum::<f64>()
        } else { one_batch };
        if self.computation_complete { return (total, total); }
        let finished = if source_curve {
            self.source_curve_samples.iter().map(|sample| {
                planning_repeat_work(sample.source_count, 1, sample.density_model_count, sample.target_count, sample.repeat)
            }).sum::<f64>()
        } else { 0.0 };
        let preparation = planning_preparation_work(ns, b, k, nt);
        let current = self.batch_job.as_ref().map_or(
            preparation * self.preparation_progress.clamp(0.0, 1.0), |job| {
                let budget = PlanningOperationBudget::for_method(job.method, ns, nt, b);
                let done = job.method_order[..job.method_order_index].iter().map(|&method| {
                    PlanningOperationBudget::for_method(method, ns, nt, b)
                        .total(b, k, job.candidate_tile_size.min(b))
                }).sum::<f64>();
                // Reference generation is shared. Credit it during the first
                // method's raw pass only, after the reference results exist.
                let reference_fraction = if job.method_order_index > 0 || job.warm_repetition { 1.0 }
                    else { (f64::from(job.density_model) * f64::from(b)
                        + f64::from(job.candidate_start) + job.reference_inflight_fraction)
                        / (f64::from(k) * f64::from(b)).max(1.0) };
                preparation + done + budget.completed(job)
                    + reference_fraction * planning_validation_work(
                        if source_curve && self.source_curve_repeat > 0 { 0 } else { ns }, b, k, nt)
            });
        ((finished + current).max(self.stopped_operation_work).min(total), total)
    }

    pub fn progress_fraction(&self) -> f64 {
        if self.computation_complete { return 1.0; }
        let (completed, total) = self.operation_work();
        // The UI also floors the displayed percentage; only an explicit final
        // completion flag can produce 100%, including when a run is cancelled.
        (completed / total.max(1.0)).clamp(0.0, 0.999_999)
    }

    pub fn blocks_realtime_gpu(&self) -> bool {
        // Keep rendering / input responsive while a prepared benchmark owns
        // the compute queue. First/Stress must not compete with real-time
        // kernels or trigger a probe collision halfway through validation.
        // Before a First/Stress capture is ready, live integration still runs.
        self.run_requested && (self.batch_job.is_some()
            || self.workload_profile == PlanningWorkloadProfile::SourceCrossover)
    }

    pub fn completed_workload(&self) -> Option<PlanningWorkloadIdentity> {
        let frequency_domain = self.results[ActiveGravityMethod::FrequencyDomain.performance_index()]?;
        let mmfft = self.results[ActiveGravityMethod::MmfftCompressed.performance_index()]?;
        let fmm = self.results[ActiveGravityMethod::Fmm.performance_index()]?;
        let dimensions = self.dimensions();
        (frequency_domain.workload == mmfft.workload
            && frequency_domain.workload == fmm.workload
            && (
                frequency_domain.workload.candidate_count,
                frequency_domain.workload.density_model_count,
                frequency_domain.workload.samples_per_candidate,
            ) == dimensions
            && frequency_domain.workload.is_complete()
            && frequency_domain.backend == PlanningExecutionBackend::GpuFrequencyDomain
            && mmfft.backend == PlanningExecutionBackend::GpuMmfft
            && fmm.backend == PlanningExecutionBackend::GpuFmm)
            .then_some(frequency_domain.workload)
    }

    pub fn fair_verdict(&self) -> Option<String> {
        self.completed_workload()?;
        let methods = [
            ("Frequency-domain algorithm", self.results[2]?),
            ("FFT", self.results[3]?),
            ("FMM", self.results[4]?),
        ];
        let common_samples = methods[0].1.verification_sample_count > 0
            && methods.iter().all(|(_, result)| {
                result.verification_sample_count == methods[0].1.verification_sample_count
            });
        let mut eligible = Vec::new();
        let mut disqualified = Vec::new();
        for (name, result) in methods {
            let mask = result.accuracy_failure_mask(self.accuracy_profile, false)
                | if common_samples { 0 } else { 1 << 8 };
            if mask == 0 {
                eligible.push((name, result.total_ms));
            } else {
                disqualified.push(format!(
                    "{name}: {}",
                    planning_accuracy_failure_labels(mask).join(", ")
                ));
            }
        }
        eligible.sort_by(|left, right| left.1.total_cmp(&right.1));
        let verdict = eligible.first().map_or_else(
            || "No eligible winner".to_string(),
            |(name, milliseconds)| {
                format!("Fastest eligible method: {name} ({milliseconds:.2} ms)")
            },
        );
        Some(format!(
            "{} profile — {verdict}{}",
            self.accuracy_profile.key(),
            if disqualified.is_empty() {
                String::new()
            } else {
                format!("; {}", disqualified.join("; "))
            }
        ))
    }
}

#[cfg(test)]
mod planning_sweep_tests {
    use super::*;

    fn measured_row() -> PlanningMethodMetrics {
        PlanningMethodMetrics {
            method: ActiveGravityMethod::Fmm,
            backend: PlanningExecutionBackend::GpuFmm,
            gpu_batch_verified: true,
            workload: PlanningWorkloadIdentity {
                reference_capture_id: 1,
                reference_capture_epoch: 1,
                source_hash: 1,
                source_count: 32000,
                basis_hash: 1,
                reference_arc_hash: 1,
                candidate_hash: 1,
                density_model_hash: 1,
                sample_hash: 1,
                tolerance_hash: 1,
                candidate_count: 1,
                density_model_count: 1,
                samples_per_candidate: 8,
                outputs: PlanningWorkloadIdentity::REQUIRED_OUTPUTS,
            },
            certified_full_pass_ms: 10.0,
            certified_estimated_total_ms: 21.0,
            raw_kernels: PlanningKernelTotals::default(),
            checked_kernels: PlanningKernelTotals::default(),
            external_validation_ms: 0.0,
            total_ms: 11.0,
            geometry_basis_build_ms: 1.0,
            density_model_ms: 1.0,
            target_point_ms: 1.0,
            relative_gravity_error: 0.00395,
            gradient_relative_error: 0.189,
            certified_relative_gravity_error: 0.00395,
            certified_gradient_relative_error: 0.189,
            gravity_error_p99: 0.005,
            gravity_error_max: 0.006,
            gradient_error_p99: 0.2,
            gradient_error_max: 0.25,
            certified_gravity_error_p99: 0.005,
            certified_gravity_error_max: 0.006,
            certified_gradient_error_p99: 0.2,
            certified_gradient_error_max: 0.25,
            pericenter_error_m: 0.5,
            minimum_altitude_m: 150.0,
            model_discrimination: 0.0,
            planning_objective: 0.0,
            segment_count: 1,
            valid_candidate_count: 1,
            verification_sample_count: 8,
            certified_verification_sample_count: 8,
            certified_rejected_sample_count: 0,
            certified_valid_candidate_count: 1,
            cold_amortization_candidates: 1,
            top_candidates: [PlanningCandidateScore { objective: 0.0 }; 5],
        }
    }

    #[test]
    fn screening_does_not_relabel_a_strict_failure() {
        let row = measured_row();
        assert!(!row.accuracy_eligible());
        assert!(!row.certified_accuracy_eligible());
        assert_eq!(
            row.accuracy_failure_mask(PlanningAccuracyProfile::Screening, false),
            0
        );
        assert_eq!(
            row.accuracy_failure_mask(PlanningAccuracyProfile::Screening, true),
            0
        );
        let reasons = planning_accuracy_failure_labels(
            row.accuracy_failure_mask(PlanningAccuracyProfile::Strict, false),
        );
        assert!(reasons.contains(&"gradient RMS"));
        assert_eq!(row.gradient_relative_error, 0.189); // no mutation or calibration
    }

    #[test]
    fn screening_keeps_nonfinite_outlier_coverage_and_validation_gates() {
        let profile = PlanningAccuracyProfile::Screening;
        for invalid in [f32::NAN, f32::INFINITY, -0.1] {
            let mut row = measured_row();
            row.relative_gravity_error = invalid;
            assert_ne!(row.accuracy_failure_mask(profile, false) & (1 << 1), 0);
        }
        let mut row = measured_row();
        row.gradient_error_max = 1.1;
        assert_ne!(row.accuracy_failure_mask(profile, false) & (1 << 4), 0);
        row = measured_row();
        row.valid_candidate_count = 0;
        assert_ne!(row.accuracy_failure_mask(profile, false) & (1 << 6), 0);
        row = measured_row();
        row.certified_rejected_sample_count = 1;
        assert_ne!(row.accuracy_failure_mask(profile, true) & (1 << 9), 0);
        row = measured_row();
        row.certified_verification_sample_count = 0;
        assert_ne!(row.accuracy_failure_mask(profile, true) & (1 << 8), 0);
    }

    #[test]
    fn checked_time_and_outlier_gates_are_independent_of_raw() {
        let mut row = measured_row();
        let profile = PlanningAccuracyProfile::Screening;
        assert_eq!(row.accuracy_failure_mask(profile, true), 0);
        row.certified_gradient_error_max = 1.1;
        assert_eq!(row.accuracy_failure_mask(profile, false), 0);
        assert_ne!(row.accuracy_failure_mask(profile, true) & (1 << 4), 0);
        row = measured_row();
        row.certified_estimated_total_ms = row.total_ms - 1.0;
        assert_ne!(row.accuracy_failure_mask(profile, true) & (1 << 7), 0);
    }

    #[test]
    fn selected_scope_visits_only_requested_parameters_and_never_finishes_early() {
        for density_index in 0..PLANNING_DENSITY_MODEL_COUNTS.len() {
            for target_index in 0..PLANNING_TARGET_COUNTS.len() {
                let mut state = PlanningComparisonState {
                    workload_profile: PlanningWorkloadProfile::SourceCrossover,
                    source_curve_density_index: density_index,
                    source_curve_target_index: target_index,
                    run_requested: true,
                    ..Default::default()
                };
                let dimensions = state.dimensions();
                let repeats = PLANNING_SOURCE_COUNTS.len() * PLANNING_SOURCE_REPEATS as usize;
                let expected_work = PLANNING_SOURCE_COUNTS.iter().map(|&sources|
                    planning_source_cell_work(sources, dimensions.1, dimensions.2))
                    .sum::<f64>();
                assert_eq!(state.operation_work(), (0.0, expected_work));
                let mut visited = 0;
                loop {
                    assert_eq!(state.dimensions(), dimensions);
                    visited += 1;
                    if !state.advance_source_curve() { break; }
                }
                assert_eq!(visited, repeats);
                // Exhausting the workload is insufficient: final readback and
                // verification must have committed before announcing 100%.
                state.stopped_operation_work = expected_work;
                assert!(state.progress_fraction() < 1.0);
                state.run_requested = false; // cancellation must not pass
                assert!(state.progress_fraction() < 1.0);
                state.computation_complete = true;
                assert_eq!(state.progress_fraction(), 1.0);
                assert_eq!(state.operation_work(), (expected_work, expected_work));
            }
        }
    }

    #[test]
    fn sweep_visits_every_source_density_target_repeat_once() {
        let mut state = PlanningComparisonState {
            source_curve_all_parameters: true,
            workload_profile: PlanningWorkloadProfile::SourceCrossover,
            ..Default::default()
        };
        let mut visited = std::collections::HashSet::new();
        loop {
            let (candidates, density, targets) = state.dimensions();
            assert_eq!(candidates, 1);
            assert!(PLANNING_DENSITY_MODEL_COUNTS.contains(&density));
            assert!(PLANNING_TARGET_COUNTS.contains(&targets));
            assert!(state.source_curve_repeat < PLANNING_SOURCE_REPEATS);
            assert!(visited.insert((
                state.requested_source_count,
                density,
                targets,
                state.source_curve_repeat
            )));
            if !state.advance_source_curve() {
                break;
            }
        }
        assert_eq!(
            visited.len(),
            PLANNING_SOURCE_COUNTS.len()
                * PLANNING_DENSITY_MODEL_COUNTS.len()
                * PLANNING_TARGET_COUNTS.len()
                * PLANNING_SOURCE_REPEATS as usize
        );
        assert_eq!(state.dimensions(), (1, 1024, 8192));
    }
}
