pub const PROBE_R0: Vec3 = Vec3::new(-616.535, 0.0, -65.459);
pub const PROBE_SPEED_FACTOR: f32 = 1.07;
pub const PROBE_ORBIT_NORMAL: Vec3 = Vec3::new(0.037_806, -0.933_691, -0.356_079);

pub const NEAR_SYNC_SEGMENT_MAX_SECONDS: f32 = 300.0;

pub const PLANNING_GRAVITY_ERROR_LIMIT: f32 = 1.0e-3;
pub const PLANNING_GRADIENT_ERROR_LIMIT: f32 = 1.0e-2;
pub const PLANNING_PERICENTER_ERROR_LIMIT_METERS: f32 = 1.0;
pub const PLANNING_EQ106_MAX_SEGMENTS: u32 = 16;
pub const PLANNING_SOURCE_COUNTS: [u32; 9] = [
    32_000, 64_000, 128_000, 256_000, 512_000, 1_024_000, 2_048_000, 4_096_000,
    8_192_000,
];
pub const PLANNING_SOURCE_REPEATS: u32 = 1;
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanningWorkloadProfile {
    #[default]
    First,
    /// Large, interruptible workload for exercising the planning pipelines
    /// without taking ownership of the browser frame loop.
    InteractiveStress,
    /// Internal source-crossover workload: enough hot targets for stable GPU
    /// timing across the requested quadrature-source range.
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
            Self::SourceCrossover => (64, 4, 96),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::First => "First 32x4x241",
            Self::InteractiveStress => "Stress 2048x32x512",
            Self::SourceCrossover => "Quadrature-source crossover 64x4x96",
        }
    }
}

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
pub enum PlanningExecutionBackend {
    GpuEq106,
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
    pub certified_estimated_total_ms: f64,
    pub total_ms: f64,
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
    /// Frames spent waiting for the packet belonging to the current request.
    /// A lost WebGPU map/readback must never leave the UI at 0% forever.
    pub awaiting_gpu_frames: u32,
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
    pub rejection_counts: [u64; 6],
    pub self_fd_step_maxima: [f32; 5],
    pub first_rejection: Option<[u32; 5]>,
    pub maximum_gradient_self_fd_relative_error: f32,
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
    pub one_time_preparation_ms: f64,
    pub preprocessing_ms: f64,
    pub command_submission_ms: f64,
    pub reduction_ms: f64,
    pub verification_ms: f64,
    pub gpu_completion_map_ms: f64,
    pub warm_evaluation_ms: f64,
    pub certified_warm_evaluation_ms: f64,
    pub certified_full_pass_ms: f64,
    pub first_tile_ms: f64,
    pub dispatch_count: u32,
    pub forward_kernel_evaluations: u64,
    pub spectral_element_count: u32,
}

impl PlanningBatchJob {
    /// Fraction of GPU evaluation work that has actually completed for this
    /// method. The denominator covers the full raw pass, its measured warm
    /// tile, and the full independently certified pass.
    pub fn completion_fraction(&self) -> f64 {
        let pass_work = u64::from(self.candidate_count) * u64::from(self.density_model_count);
        let warm_work = u64::from(self.candidate_tile_size.min(self.candidate_count));
        let total_work = pass_work.saturating_mul(2).saturating_add(warm_work).max(1);
        let current_pass_work = u64::from(self.density_model)
            .saturating_mul(u64::from(self.candidate_count))
            .saturating_add(u64::from(self.candidate_start))
            .min(pass_work);
        let completed = if !self.warm_repetition {
            current_pass_work
        } else if !self.certified_repetition {
            pass_work
        } else {
            pass_work
                .saturating_add(warm_work)
                .saturating_add(current_pass_work)
        };
        completed.min(total_work) as f64 / total_work as f64
    }
}

impl PlanningMethodMetrics {
    pub fn accuracy_eligible(self) -> bool {
        self.gpu_batch_verified
            && self.workload.is_complete()
            && self.relative_gravity_error.is_finite()
            && self.relative_gravity_error <= PLANNING_GRAVITY_ERROR_LIMIT
            && self.gradient_relative_error.is_finite()
            && self.gradient_relative_error <= PLANNING_GRADIENT_ERROR_LIMIT
            && self.gravity_error_p99.is_finite()
            && self.gravity_error_p99 <= 5.0 * PLANNING_GRAVITY_ERROR_LIMIT
            && self.gravity_error_max.is_finite()
            && self.gravity_error_max <= 10.0 * PLANNING_GRAVITY_ERROR_LIMIT
            && self.gradient_error_p99.is_finite()
            && self.gradient_error_p99 <= 5.0 * PLANNING_GRADIENT_ERROR_LIMIT
            && self.gradient_error_max.is_finite()
            && self.gradient_error_max <= 10.0 * PLANNING_GRADIENT_ERROR_LIMIT
            && self.pericenter_error_m.is_finite()
            && self.pericenter_error_m <= PLANNING_PERICENTER_ERROR_LIMIT_METERS
            && self.top_candidates[0].objective.is_finite()
            && self.valid_candidate_count == self.workload.candidate_count
            && self.total_ms.is_finite()
            && self.total_ms > 0.0
    }

    pub fn certified_accuracy_eligible(self) -> bool {
        self.gpu_batch_verified
            && self.workload.is_complete()
            && self.certified_relative_gravity_error.is_finite()
            && self.certified_relative_gravity_error <= PLANNING_GRAVITY_ERROR_LIMIT
            && self.certified_gradient_relative_error.is_finite()
            && self.certified_gradient_relative_error <= PLANNING_GRADIENT_ERROR_LIMIT
            // The certified pass must satisfy the same pointwise outlier
            // policy as the ordinary pass.  Until certified pointwise strata
            // are stored separately, the raw strata are the conservative
            // common gate for both reported raw/certified timings.
            && self.gravity_error_p99.is_finite()
            && self.gravity_error_p99 <= 5.0 * PLANNING_GRAVITY_ERROR_LIMIT
            && self.gravity_error_max.is_finite()
            && self.gravity_error_max <= 10.0 * PLANNING_GRAVITY_ERROR_LIMIT
            && self.gradient_error_p99.is_finite()
            && self.gradient_error_p99 <= 5.0 * PLANNING_GRADIENT_ERROR_LIMIT
            && self.gradient_error_max.is_finite()
            && self.gradient_error_max <= 10.0 * PLANNING_GRADIENT_ERROR_LIMIT
            && self.certified_verification_sample_count == self.verification_sample_count
            && self.certified_rejected_sample_count == 0
            && self.certified_valid_candidate_count == self.workload.candidate_count
            && self.certified_full_pass_ms.is_finite()
            && self.certified_full_pass_ms > 0.0
    }
}

#[derive(Resource)]
pub struct PlanningComparisonState {
    pub selected_metric: ComparisonMetric,
    pub workload_profile: PlanningWorkloadProfile,
    pub results: [Option<PlanningMethodMetrics>; 5],
    pub run_requested: bool,
    pub run_id: u64,
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
    pub source_curve_samples: Vec<PlanningSourceCurveSample>,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanningSourceCurveSample {
    pub source_count: u32,
    /// Eq.106 raw/certified, packed FFT raw/certified, FMM raw/certified.
    pub times_ms: [f64; 6],
    pub eligible: [bool; 6],
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
            workload_profile: PlanningWorkloadProfile::First,
            results: std::array::from_fn(|_| None),
            run_requested: false,
            run_id: 0,
            reference_duration_seconds: 0.0,
            status: "Choose a planning metric to run First, or use inversion metrics with the inversion button.".into(),
            batch_job: None,
            preparation_progress: 0.0,
            requested_source_count: PLANNING_SOURCE_COUNTS[0],
            source_curve_active: false,
            source_curve_visible: false,
            source_curve_index: 0,
            source_curve_repeat: 0,
            source_curve_samples: Vec::new(),
        }
    }
}

impl PlanningComparisonState {
    /// Monotonic end-to-end progress. Each source point consists of one CPU
    /// preparation stage followed by the three GPU method stages.
    pub fn progress_fraction(&self) -> f64 {
        const STAGES_PER_SOURCE: f64 = 4.0;
        let source_curve = self.workload_profile == PlanningWorkloadProfile::SourceCrossover
            && (self.source_curve_active || self.source_curve_visible);
        let source_count = if source_curve {
            PLANNING_SOURCE_COUNTS.len()
        } else {
            1
        };
        let source_index = if source_curve {
            self.source_curve_index.min(source_count)
        } else {
            0
        };
        if source_index == source_count
            || (!source_curve && !self.run_requested && self.completed_workload().is_some())
        {
            return 1.0;
        }
        let within_source = self.batch_job.as_ref().map_or_else(
            || self.preparation_progress.clamp(0.0, 1.0) / STAGES_PER_SOURCE,
            |job| {
                (1.0 + job.method_order_index as f64 + job.completion_fraction())
                    / STAGES_PER_SOURCE
            },
        );
        ((source_index as f64 + within_source) / source_count as f64).clamp(0.0, 1.0)
    }

    pub fn blocks_realtime_gpu(&self) -> bool {
        // All planning modes share the frame budget with the running scene.
        // GPU submission/readback is asynchronous; freezing physics or UI for
        // a benchmark makes the application appear hung.
        false
    }

    pub fn completed_workload(&self) -> Option<PlanningWorkloadIdentity> {
        let eq106 = self.results[ActiveGravityMethod::CurvedArcEq106.performance_index()]?;
        let mmfft = self.results[ActiveGravityMethod::MmfftCompressed.performance_index()]?;
        let fmm = self.results[ActiveGravityMethod::Fmm.performance_index()]?;
        let dimensions = self.workload_profile.dimensions();
        (eq106.workload == mmfft.workload
            && eq106.workload == fmm.workload
            && (
                eq106.workload.candidate_count,
                eq106.workload.density_model_count,
                eq106.workload.samples_per_candidate,
            ) == dimensions
            && eq106.workload.is_complete()
            && eq106.backend == PlanningExecutionBackend::GpuEq106
            && mmfft.backend == PlanningExecutionBackend::GpuMmfft
            && fmm.backend == PlanningExecutionBackend::GpuFmm)
            .then_some(eq106.workload)
    }

    pub fn fair_verdict(&self) -> Option<String> {
        self.completed_workload()?;
        let eq106 = self.results[2]?;
        let mmfft = self.results[3]?;
        let fmm = self.results[4]?;
        let methods = [
            ("Eq.106 full forward", eq106),
            ("packed FFT", mmfft),
            ("target-cell FMM", fmm),
        ];
        let common_samples = eq106.verification_sample_count > 0
            && eq106.verification_sample_count == mmfft.verification_sample_count
            && eq106.verification_sample_count == fmm.verification_sample_count;
        let mut eligible = Vec::new();
        let mut disqualified = Vec::new();
        for (name, result) in methods {
            let mut reasons = Vec::new();
            if !result.gpu_batch_verified {
                reasons.push("GPU verification failed".to_string());
            }
            if !result.relative_gravity_error.is_finite()
                || result.relative_gravity_error > PLANNING_GRAVITY_ERROR_LIMIT
            {
                reasons.push(format!(
                    "gravity {:.3e} > {:.1e}",
                    result.relative_gravity_error, PLANNING_GRAVITY_ERROR_LIMIT
                ));
            }
            if !result.gradient_relative_error.is_finite()
                || result.gradient_relative_error > PLANNING_GRADIENT_ERROR_LIMIT
            {
                reasons.push(format!(
                    "gradient {:.3e} > {:.1e}",
                    result.gradient_relative_error, PLANNING_GRADIENT_ERROR_LIMIT
                ));
            }
            if !result.gravity_error_p99.is_finite()
                || result.gravity_error_p99 > 5.0 * PLANNING_GRAVITY_ERROR_LIMIT
                || !result.gravity_error_max.is_finite()
                || result.gravity_error_max > 10.0 * PLANNING_GRAVITY_ERROR_LIMIT
            {
                reasons.push(format!(
                    "gravity point p99/max {:.3e}/{:.3e}",
                    result.gravity_error_p99, result.gravity_error_max
                ));
            }
            if !result.gradient_error_p99.is_finite()
                || result.gradient_error_p99 > 5.0 * PLANNING_GRADIENT_ERROR_LIMIT
                || !result.gradient_error_max.is_finite()
                || result.gradient_error_max > 10.0 * PLANNING_GRADIENT_ERROR_LIMIT
            {
                reasons.push(format!(
                    "gradient point p99/max {:.3e}/{:.3e}",
                    result.gradient_error_p99, result.gradient_error_max
                ));
            }
            if !result.pericenter_error_m.is_finite()
                || result.pericenter_error_m > PLANNING_PERICENTER_ERROR_LIMIT_METERS
            {
                reasons.push(format!(
                    "pericenter {:.3e}m > {:.1}m",
                    result.pericenter_error_m, PLANNING_PERICENTER_ERROR_LIMIT_METERS
                ));
            }
            if result.valid_candidate_count != result.workload.candidate_count {
                reasons.push(format!(
                    "coverage {}/{}",
                    result.valid_candidate_count, result.workload.candidate_count
                ));
            }
            if name.starts_with("Eq.106") && result.segment_count > PLANNING_EQ106_MAX_SEGMENTS {
                reasons.push(format!(
                    "segments {} > {}",
                    result.segment_count, PLANNING_EQ106_MAX_SEGMENTS
                ));
            }
            if !common_samples {
                reasons.push(format!(
                    "non-common f64 sample count {}",
                    result.verification_sample_count
                ));
            }
            if reasons.is_empty() && result.accuracy_eligible() {
                eligible.push((name, result.total_ms));
            } else {
                if reasons.is_empty() {
                    reasons.push("incomplete score or timing data".into());
                }
                disqualified.push(format!("{name} disqualified: {}", reasons.join(", ")));
            }
        }
        eligible.sort_by(|left, right| left.1.total_cmp(&right.1));
        let verdict = eligible.first().map_or_else(
            || "No eligible winner".to_string(),
            |(name, milliseconds)| format!("Eligible winner: {name} ({milliseconds:.2} ms)"),
        );
        Some(if disqualified.is_empty() {
            verdict
        } else {
            format!("{verdict}; {}", disqualified.join("; "))
        })
    }
}

#[cfg(test)]
mod planning_profile_tests {
    use super::{PLANNING_SOURCE_COUNTS, PlanningComparisonState, PlanningWorkloadProfile};

    #[test]
    fn visible_profile_uses_fixed_batch_scheduling() {
        assert!(PlanningWorkloadProfile::First.is_compute_benchmark());
    }
    #[test]
    fn quadrature_curve_spans_32k_through_8192k() {
        assert_eq!(PLANNING_SOURCE_COUNTS[0], 32_000);
        assert_eq!(*PLANNING_SOURCE_COUNTS.last().unwrap(), 8_192_000);
        assert!(PLANNING_SOURCE_COUNTS.windows(2).all(|pair| pair[1] == 2 * pair[0]));
    }

    #[test]
    fn source_curve_progress_never_resets_between_source_counts() {
        let mut state = PlanningComparisonState {
            workload_profile: PlanningWorkloadProfile::SourceCrossover,
            source_curve_active: true,
            source_curve_visible: true,
            run_requested: true,
            preparation_progress: 1.0,
            ..Default::default()
        };
        let first_source_prepared = state.progress_fraction();
        state.source_curve_index = 1;
        state.preparation_progress = 0.0;
        let second_source_started = state.progress_fraction();
        assert!((first_source_prepared - 1.0 / 36.0).abs() < f64::EPSILON);
        assert!(second_source_started > first_source_prepared);

        state.source_curve_index = PLANNING_SOURCE_COUNTS.len();
        state.source_curve_active = false;
        assert_eq!(state.progress_fraction(), 1.0);
    }
}
