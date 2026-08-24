pub const PROBE_R0: Vec3 = Vec3::new(-616.535, 0.0, -65.459);
pub const PROBE_SPEED_FACTOR: f32 = 1.07;
pub const PROBE_ORBIT_NORMAL: Vec3 = Vec3::new(0.037_806, -0.933_691, -0.356_079);

pub const NEAR_SYNC_POSITION: Vec3 = Vec3::new(-1097.269, 51.622, 0.0);
pub const NEAR_SYNC_SPEED_FACTOR: f32 = 0.824_08;
pub const NEAR_SYNC_ORBIT_PERIOD_SECONDS: f64 = 27_495.468;
pub const NEAR_SYNC_SEMIMAJOR_AXIS_METERS: f32 = 831.624;
pub const NEAR_SYNC_PERICENTER_RADIUS_METERS: f32 = 564.765;
pub const NEAR_SYNC_APOCENTER_RADIUS_METERS: f32 = 1_098.483;
pub const NEAR_SYNC_ECCENTRICITY: f32 = 0.320_889;
pub const NEAR_SYNC_SEGMENT_MAX_SECONDS: f32 = 300.0;
pub const NEAR_SYNC_TAYLOR_ORDER: u32 = 4;
pub const NEAR_SYNC_TRUST_RADIUS_METERS: f32 = PLANNING_TRAJECTORY_TUBE_RADIUS_METERS;
pub const NEAR_SYNC_TRANSVERSE_LIMIT_METERS: f32 = PLANNING_TRAJECTORY_TUBE_RADIUS_METERS;
pub const NEAR_SYNC_RELATIVE_REMAINDER_TARGET: f32 = 1.0e-3;

pub const PLANNING_GRAVITY_ERROR_LIMIT: f32 = 1.0e-3;
pub const PLANNING_GRADIENT_ERROR_LIMIT: f32 = 1.0e-2;
pub const PLANNING_PERICENTER_ERROR_LIMIT_METERS: f32 = 1.0;
pub const PLANNING_EQ106_MAX_SEGMENTS: u32 = 16;
pub const PLANNING_SOURCE_COUNTS: [u32; 5] = [1_024, 2_048, 4_096, 65_536, 262_144];
pub const PLANNING_SOURCE_REPEATS: u32 = 10;
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
    NearSynchronousPlanning,
    Custom,
}

impl ProbeOrbitPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::CurrentBenchmark => "Current orbit",
            Self::NearSynchronousPlanning => "Near-sync ellipse",
            Self::Custom => "Custom",
        }
    }

    pub fn conditions(self) -> ProbeInitialConditions {
        match self {
            Self::CurrentBenchmark | Self::Custom => ProbeInitialConditions::default(),
            Self::NearSynchronousPlanning => ProbeInitialConditions {
                position: NEAR_SYNC_POSITION,
                speed_factor: NEAR_SYNC_SPEED_FACTOR,
                orbit_normal: RYUGU_SPIN_AXIS,
                preset: self,
            },
        }
    }
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
    InteractiveStress,
}

impl PlanningWorkloadProfile {
    pub fn is_compute_benchmark(self) -> bool {
        // Both visible workloads use the same fairness contract. "Interactive"
        // describes the live progress UI, not adaptive benchmark scheduling.
        matches!(self, Self::First | Self::InteractiveStress)
    }

    pub fn dimensions(self) -> (u32, u32, u32) {
        match self {
            Self::First => (PLANNING_FIRST_CANDIDATE_COUNT, 4, 241),
            Self::InteractiveStress => (PLANNING_CANDIDATE_COUNT, 32, 512),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::First => "First 32x4x241",
            Self::InteractiveStress => "Interactive Stress 2048x32x512",
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
    pub const ALL: [Self; 11] = [
        Self::DensityFit,
        Self::InversionTime,
        Self::GravityRelativeError,
        Self::GradientRelativeError,
        Self::PericenterError,
        Self::MinimumAltitude,
        Self::ModelDiscrimination,
        Self::PlanningObjective,
        Self::SegmentCount,
        Self::SpeedupVsGpuFmm,
        Self::ColdStartAmortization,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::DensityFit => "Density fit",
            Self::InversionTime => "Inversion time",
            Self::GravityRelativeError => "Gravity error",
            Self::GradientRelativeError => "Gradient error",
            Self::PericenterError => "Pericenter error",
            Self::MinimumAltitude => "Minimum altitude",
            Self::ModelDiscrimination => "Reference separation",
            Self::PlanningObjective => "Planning objective",
            Self::SegmentCount => "Segments",
            Self::SpeedupVsGpuFmm => "Speedup / baselines",
            Self::ColdStartAmortization => "Cold amortization",
        }
    }

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
    pub candidate_index: u32,
    pub minimum_altitude_m: f32,
    pub reference_model_separation: f32,
    pub gradient_information: f32,
    pub objective: f32,
}

impl Default for PlanningCandidateScore {
    fn default() -> Self {
        Self {
            candidate_index: 0,
            minimum_altitude_m: f32::NAN,
            reference_model_separation: f32::NAN,
            gradient_information: f32::NAN,
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
    pub common_preparation_ms: f64,
    pub preprocessing_ms: f64,
    pub command_submission_ms: f64,
    pub reduction_ms: f64,
    pub verification_ms: f64,
    pub gpu_completion_map_ms: f64,
    pub warm_evaluation_ms: f64,
    pub total_ms: f64,
    pub relative_gravity_error: f32,
    pub gradient_relative_error: f32,
    pub raw_relative_gravity_error: f32,
    pub raw_gradient_relative_error: f32,
    pub rejected_sample_count: u64,
    pub rejection_counts: [u64; 6],
    pub self_fd_step_maxima: [f32; 5],
    pub first_rejection: Option<[u32; 5]>,
    pub pericenter_error_m: f32,
    pub minimum_altitude_m: f32,
    pub model_discrimination: f32,
    pub planning_objective: f32,
    pub segment_count: u32,
    pub valid_candidate_count: u32,
    /// Number of common f64 reference states used in the error denominator.
    pub verification_sample_count: u64,
    pub cold_amortization_candidates: u32,
    pub dispatch_count: u32,
    pub forward_kernel_evaluations: u64,
    pub density_combinations: u64,
    pub gpu_request_count: u32,
    pub minimum_tile_candidates: u32,
    pub maximum_tile_candidates: u32,
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
    pub last_request_candidate_count: u32,
    pub awaiting_gpu: bool,
    pub warm_repetition: bool,
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
    pub common_preparation_ms: f64,
    pub preprocessing_ms: f64,
    pub command_submission_ms: f64,
    pub reduction_ms: f64,
    pub verification_ms: f64,
    pub gpu_completion_map_ms: f64,
    pub warm_evaluation_ms: f64,
    pub dispatch_count: u32,
    pub forward_kernel_evaluations: u64,
    pub spectral_element_count: u32,
}

impl PlanningMethodMetrics {
    pub fn accuracy_eligible(self) -> bool {
        self.gpu_batch_verified
            && self.workload.is_complete()
            && self.relative_gravity_error.is_finite()
            && self.relative_gravity_error <= PLANNING_GRAVITY_ERROR_LIMIT
            && self.gradient_relative_error.is_finite()
            && self.gradient_relative_error <= PLANNING_GRADIENT_ERROR_LIMIT
            && self.pericenter_error_m.is_finite()
            && self.pericenter_error_m <= PLANNING_PERICENTER_ERROR_LIMIT_METERS
            && self.top_candidates[0].objective.is_finite()
            && self.valid_candidate_count == self.workload.candidate_count
            && self.total_ms.is_finite()
            && self.total_ms > 0.0
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
    pub times_ms: [f64; 3],
    pub eligible: [bool; 3],
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
    pub fn blocks_realtime_gpu(&self) -> bool {
        // Interactive Stress must remain a live application mode: the probe,
        // Eq.106 residual, and Jacobi histories continue advancing while its
        // planning batches run. Only the short First benchmark owns the GPU
        // and physics clock exclusively.
        self.run_requested && self.workload_profile == PlanningWorkloadProfile::First
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
        let methods = [("Eq.106", eq106), ("FFT-grid", mmfft), ("treecode", fmm)];
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
            if name == "Eq.106" && result.segment_count > PLANNING_EQ106_MAX_SEGMENTS {
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
    use super::PlanningWorkloadProfile;

    #[test]
    fn both_visible_profiles_use_fixed_batch_scheduling() {
        assert!(PlanningWorkloadProfile::First.is_compute_benchmark());
        assert!(PlanningWorkloadProfile::InteractiveStress.is_compute_benchmark());
        assert_eq!(
            PlanningWorkloadProfile::InteractiveStress.dimensions(),
            (2_048, 32, 512)
        );
    }
}
