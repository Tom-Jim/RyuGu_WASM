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
    CppFlups,
    CppExafmm,
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
            (
                self.certified_gravity_error_p99,
                self.certified_gravity_error_max,
                self.certified_gradient_error_p99,
                self.certified_gradient_error_max,
            )
        } else {
            (
                self.gravity_error_p99,
                self.gravity_error_max,
                self.gradient_error_p99,
                self.gradient_error_max,
            )
        };
        let failures = [
            !self.gpu_batch_verified || !self.workload.is_complete(),
            !within(gravity, limits.gravity),
            !within(gradient, limits.gradient),
            !within(gravity_p99, limits.gravity_p99) || !within(gravity_max, limits.gravity_max),
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
