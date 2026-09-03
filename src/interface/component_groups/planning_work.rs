// Deterministic operation estimates, not elapsed-time estimates or measured
// hardware FLOPs. Keep the same budget throughout a run: random method order,
// timestamp support and the displayed chart selection must not move the goal.
// Constants model scalar arithmetic (including expensive source interactions).
// Frequency-domain algorithm counts one density transform per reciprocal-space
// sample; FMM uses a conservative near-field bound.
#[derive(Clone, Copy)]
pub struct PlanningOperationBudget {
    pub basis: f64,
    pub density: f64,
    pub target: f64,
    pub checked_target: f64,
}

impl PlanningOperationBudget {
    pub fn for_method(
        method: ActiveGravityMethod,
        sources: u32,
        targets: u32,
        candidates: u32,
    ) -> Self {
        let ns = f64::from(sources.max(1));
        let nt = f64::from(targets.max(1));
        let states = nt * f64::from(candidates.max(1));
        match method {
            ActiveGravityMethod::FrequencyDomain => Self {
                basis: ns * 64.0 + 56.0 * 64.0,
                density: 56.0 * 64.0,
                target: nt * 64.0 * 24.0,
                checked_target: nt * 64.0 * 24.0,
            },
            ActiveGravityMethod::MmfftCompressed => {
                // GPU compensated-complex butterflies: two transforms and
                // one product per column, plus one shared kernel FFT per level.
                // These are scalar operation estimates, never measured FLOPs.
                let mut column_work = 0.0;
                let mut kernel_work = 0.0;
                for side in [128.0_f64, 32.0] {
                    let m = side.powi(3);
                    column_work += 40.0 * m * m.log2() + 32.0 * m;
                    kernel_work += 20.0 * m * m.log2() + 32.0 * m;
                }
                Self {
                    basis: ns * 2.0 * 8.0 * 32.0 + 56.0 * column_work + kernel_work,
                    density: 56.0 * (64.0_f64.powi(3) + 16.0_f64.powi(3)) * 20.0,
                    target: nt * 216.0 * 32.0,
                    checked_target: nt * 216.0 * 32.0,
                }
            }
            _ => {
                let moments = ns * 640.0 + 4681.0 * 56.0 * 8.0 * 10.0 * 16.0;
                let responses = states * (ns * 60.0 + 56.0 * 512.0 * 80.0);
                // GPU density mixing occurs inside each target evaluator.
                // Large stress jobs retain source moments but stream target
                // basis windows; include their rebuilds in subsequent passes.
                Self {
                    basis: moments + if states <= 8192.0 { responses } else { 0.0 },
                    density: 0.0,
                    target: nt * 56.0 * 16.0 * 2.0
                        + if states > 8192.0 { responses / f64::from(candidates.max(1)) } else { 0.0 },
                    checked_target: nt * 56.0 * 16.0 * 2.0
                        + if states > 8192.0 { responses / f64::from(candidates.max(1)) } else { 0.0 },
                }
            }
        }
    }

    pub fn total(self, candidates: u32, density: u32, warm: u32) -> f64 {
        let b = f64::from(candidates);
        let k = f64::from(density);
        // K=1 reuses the same density payload for both warm and checked.
        let checked_density = if density > 1 { k * self.density } else { 0.0 };
        self.basis
            + k * self.density
            + k * b * self.target
            + f64::from(warm) * self.target
            + checked_density
            + k * b * self.checked_target
    }

    pub fn completed(self, job: &PlanningBatchJob) -> f64 {
        let b = f64::from(job.candidate_count);
        let k = f64::from(job.density_model_count);
        let models = f64::from(job.density_model) + if job.candidate_start > 0 { 1.0 } else { 0.0 };
        let candidates = f64::from(job.density_model) * b + f64::from(job.candidate_start);
        if !job.warm_repetition {
            // Credit a stage only after its result has arrived and been reduced.
            return if candidates == 0.0 {
                self.basis * job.gpu_basis_progress.clamp(0.0, 1.0)
            } else {
                self.basis + models * self.density + candidates * self.target
            };
        }
        let raw = self.basis + k * self.density + k * b * self.target;
        if !job.certified_repetition {
            return raw;
        }
        raw + f64::from(job.candidate_tile_size.min(job.candidate_count)) * self.target
            + if job.density_model_count > 1 {
                models * self.density
            } else {
                0.0
            }
            + candidates * self.checked_target
    }
}

fn planning_preparation_work(sources: u32, b: u32, k: u32, nt: u32) -> f64 {
    f64::from(sources) * 64.0 + f64::from(b) * f64::from(nt) * 512.0 + f64::from(k) * 56.0 * 8.0
}

fn planning_validation_work(sources: u32, b: u32, k: u32, nt: u32) -> f64 {
    // The frequency-domain oracle needs the direct f64 field along the whole
    // trajectory before applying its independent Laplace integral. Those
    // cached fields also cover the sparse pointwise FFT/FMM checks, followed
    // by six field/Jacobian comparison rows per selected checkpoint.
    let full = b <= PLANNING_FIRST_CANDIDATE_COUNT && k <= 4;
    let models = if full {
        k
    } else {
        let stride = PLANNING_REFERENCE_MODEL_STRIDE.min((k / 4).max(1));
        k.div_ceil(stride).saturating_add(3).min(k)
    };
    let candidates = if full {
        b
    } else {
        b.div_ceil(PLANNING_REFERENCE_CANDIDATE_STRIDE.min((b / 8).max(1)))
            .saturating_add(3)
            .min(b)
    };
    let targets = if full {
        nt
    } else {
        nt.div_ceil(PLANNING_REFERENCE_STRIDE)
            .saturating_add(8)
            .min(nt)
    };
    f64::from(models)
        * f64::from(candidates)
        * (f64::from(nt) * f64::from(sources) * 50.0
            + f64::from(targets) * 6.0 * 100.0)
}

fn planning_batch_work(sources: u32, b: u32, k: u32, nt: u32) -> f64 {
    let warm = PLANNING_GPU_TILE_INITIAL_CANDIDATES.min(b);
    planning_preparation_work(sources, b, k, nt)
        + planning_validation_work(sources, b, k, nt)
        + [
            ActiveGravityMethod::FrequencyDomain,
            ActiveGravityMethod::MmfftCompressed,
            ActiveGravityMethod::Fmm,
        ]
        .into_iter()
        .map(|method| PlanningOperationBudget::for_method(method, sources, nt, b).total(b, k, warm))
        .sum::<f64>()
}

fn planning_repeat_work(sources: u32, b: u32, k: u32, nt: u32, repeat: u32) -> f64 {
    let cold = planning_batch_work(sources, b, k, nt);
    if repeat <= 1 {
        return cold;
    }
    // Reference fields are keyed by geometry/density/target hashes, not run_id.
    // Fresh method bases are rebuilt each repeat, but the shared f64 reference
    // solve is reused across the seven identical repetitions of a cell.
    cold - planning_validation_work(sources, b, k, nt) + planning_validation_work(0, b, k, nt)
}

fn planning_source_cell_work(sources: u32, k: u32, nt: u32) -> f64 {
    (1..=PLANNING_SOURCE_REPEATS)
        .map(|repeat| planning_repeat_work(sources, 1, k, nt, repeat))
        .sum()
}

#[cfg(test)]
mod planning_operation_tests {
    use super::*;

    #[test]
    fn source_growth_changes_work_budget_not_just_the_point_counter() {
        let small = planning_batch_work(32_000, 1, 4, 8);
        let large = planning_batch_work(8_192_000, 1, 4, 8);
        assert!(large > small * 2.0);
        assert!(
            planning_repeat_work(8_192_000, 1, 512, 8192, 2)
                < planning_repeat_work(8_192_000, 1, 512, 8192, 1)
        );
        assert!(planning_batch_work(32_000, 1, 512, 241) > small);
        let fft =
            PlanningOperationBudget::for_method(ActiveGravityMethod::MmfftCompressed, 32_000, 8, 1);
        let refined = PlanningOperationBudget::for_method(
            ActiveGravityMethod::MmfftCompressed,
            8_192_000,
            8,
            1,
        );
        assert_eq!(fft.density, refined.density); // cached RHS never revisits sources
        assert!(refined.basis > fft.basis);
    }

    #[test]
    fn timestamp_totals_fail_closed_after_one_unmeasured_request() {
        let mut totals = PlanningKernelTotals::default();
        totals.record(PlanningGpuTiming {
            kernel_ms: Some(3.0),
            evaluation_kernel_ms: Some(1.0),
            basis_kernel_ms: Some(2.0),
            ..Default::default()
        });
        totals.record(PlanningGpuTiming::default());
        totals.record(PlanningGpuTiming {
            kernel_ms: Some(3.0),
            evaluation_kernel_ms: Some(1.0),
            basis_kernel_ms: Some(0.0),
            ..Default::default()
        });
        assert!(totals.all_ms.is_none());
        assert!(totals.evaluation_ms.is_none());
        assert!(totals.basis_ms.is_none());
    }
}
