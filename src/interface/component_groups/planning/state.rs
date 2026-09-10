#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Resource)]
pub struct PlanningComparisonState {
    pub selected_metric: ComparisonMetric,
    pub accuracy_profile: PlanningAccuracyProfile,
    pub workload_profile: PlanningWorkloadProfile,
    /// False until the user explicitly starts First or Stress. The profile
    /// remains useful as an internal default but must not paint a button as
    /// selected before the user requests a calculation.
    pub workload_selected: bool,
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
            workload_selected: false,
            results: std::array::from_fn(|_| None),
            run_requested: false,
            run_id: 0,
            computation_complete: false,
            source_curve_all_parameters: false,
            source_curve_run_id: 0,
            stopped_operation_work: 0.0,
            reference_duration_seconds: 0.0,
            status: "Select First or Stress to start a planning calculation.".into(),
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
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub fn operation_work(&self) -> (f64, f64) {
        let source_curve = self.workload_profile == PlanningWorkloadProfile::SourceCrossover;
        let (b, k, nt) = self.dimensions();
        let ns = self.requested_source_count;
        let one_batch = planning_batch_work(ns, b, k, nt);
        let total = if source_curve {
            PLANNING_SOURCE_COUNTS
                .iter()
                .map(|&sources| {
                    if self.source_curve_all_parameters {
                        PLANNING_TARGET_COUNTS
                            .iter()
                            .map(|&targets| {
                                PLANNING_DENSITY_MODEL_COUNTS
                                    .iter()
                                    .map(|&density| {
                                        planning_source_cell_work(sources, density, targets)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                    } else {
                        planning_source_cell_work(sources, k, nt)
                    }
                })
                .sum::<f64>()
        } else {
            one_batch
        };
        if self.computation_complete {
            return (total, total);
        }
        let finished = if source_curve {
            self.source_curve_samples
                .iter()
                .map(|sample| {
                    planning_repeat_work(
                        sample.source_count,
                        1,
                        sample.density_model_count,
                        sample.target_count,
                        sample.repeat,
                    )
                })
                .sum::<f64>()
        } else {
            0.0
        };
        let preparation = planning_preparation_work(ns, b, k, nt);
        let current = self.batch_job.as_ref().map_or(
            preparation * self.preparation_progress.clamp(0.0, 1.0),
            |job| {
                let budget = PlanningOperationBudget::for_method(job.method, ns, nt, b);
                let done = job.method_order[..job.method_order_index]
                    .iter()
                    .map(|&method| {
                        PlanningOperationBudget::for_method(method, ns, nt, b).total(
                            b,
                            k,
                            job.candidate_tile_size.min(b),
                        )
                    })
                    .sum::<f64>();
                // Reference generation is shared. Credit it during the first
                // method's raw pass only, after the reference results exist.
                let reference_fraction = if job.method_order_index > 0 || job.warm_repetition {
                    1.0
                } else {
                    (f64::from(job.density_model) * f64::from(b)
                        + f64::from(job.candidate_start)
                        + job.reference_inflight_fraction)
                        / (f64::from(k) * f64::from(b)).max(1.0)
                };
                preparation
                    + done
                    + budget.completed(job)
                    + reference_fraction
                        * planning_validation_work(
                            if source_curve && self.source_curve_repeat > 0 {
                                0
                            } else {
                                ns
                            },
                            b,
                            k,
                            nt,
                        )
            },
        );
        (
            (finished + current)
                .max(self.stopped_operation_work)
                .min(total),
            total,
        )
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub fn progress_fraction(&self) -> f64 {
        if self.computation_complete {
            return 1.0;
        }
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
        self.run_requested
            && (self.batch_job.is_some()
                || self.workload_profile == PlanningWorkloadProfile::SourceCrossover)
    }

    pub fn completed_workload(&self) -> Option<PlanningWorkloadIdentity> {
        let frequency_domain =
            self.results[ActiveGravityMethod::FrequencyDomain.performance_index()]?;
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
            && mmfft.backend == PlanningExecutionBackend::CppFlups
            && fmm.backend == PlanningExecutionBackend::CppExafmm)
            .then_some(frequency_domain.workload)
    }

    pub fn fair_verdict(&self) -> Option<String> {
        self.completed_workload()?;
        let methods = [
            (
                "Frequency-domain algorithm",
                self.results[ActiveGravityMethod::FrequencyDomain.performance_index()]?,
            ),
            (
                "FFT",
                self.results[ActiveGravityMethod::MmfftCompressed.performance_index()]?,
            ),
            (
                "FMM",
                self.results[ActiveGravityMethod::Fmm.performance_index()]?,
            ),
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
            backend: PlanningExecutionBackend::CppExafmm,
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
                let expected_work = PLANNING_SOURCE_COUNTS
                    .iter()
                    .map(|&sources| planning_source_cell_work(sources, dimensions.1, dimensions.2))
                    .sum::<f64>();
                assert_eq!(state.operation_work(), (0.0, expected_work));
                let mut visited = 0;
                loop {
                    assert_eq!(state.dimensions(), dimensions);
                    visited += 1;
                    if !state.advance_source_curve() {
                        break;
                    }
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
