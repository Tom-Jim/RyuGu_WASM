fn matrix_norm_squared(matrix: DMat3) -> f64 {
    matrix.x_axis.length_squared() + matrix.y_axis.length_squared() + matrix.z_axis.length_squared()
}

fn adapt_candidate_tile(job: &mut PlanningBatchJob, packet: &PlanningGpuPacket) {
    // First is a fixed throughput benchmark. Browser FPS depends on method
    // order and must not alter later methods' request counts or batch widths.
    if job.profile.is_compute_benchmark() {
        return;
    }
    let request_ms = packet.timing.method_preprocess_ms
        + packet.timing.command_submission_ms
        + packet.timing.gpu_completion_map_ms
        + packet.timing.readback_decode_ms;
    let frame_rate = crate::browser_frame_rate();
    let recent_frame_ms = crate::browser_recent_frame_ms();
    let should_shrink = request_ms > PLANNING_MAX_REQUEST_MS
        || frame_rate.is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS)
        || recent_frame_ms.is_some_and(|milliseconds| milliseconds > PLANNING_MAX_RECENT_FRAME_MS);
    let can_grow = request_ms < PLANNING_TARGET_REQUEST_MS
        && frame_rate.is_none_or(|fps| fps >= 59.0)
        && recent_frame_ms.is_none_or(|milliseconds| milliseconds <= 17.2);
    let (minimum, maximum) = if job.method == ActiveGravityMethod::FrequencyDomain {
        (
            PLANNING_GPU_TILE_MIN_CANDIDATES,
            PLANNING_GPU_TILE_MAX_CANDIDATES,
        )
    } else {
        (
            PLANNING_GENERIC_TILE_MIN_CANDIDATES,
            PLANNING_GENERIC_TILE_MAX_CANDIDATES,
        )
    };
    job.candidate_tile_size = if should_shrink {
        (job.candidate_tile_size / 2).max(minimum)
    } else if can_grow {
        job.candidate_tile_size.saturating_mul(2).min(maximum)
    } else {
        job.candidate_tile_size
    };
}

fn advance_planning_tile(job: &mut PlanningBatchJob, completed_candidates: u32) {
    job.candidate_start += completed_candidates;
    if job.candidate_start < job.candidate_count {
        return;
    }
    job.candidate_start = 0;
    job.density_model += 1;
    if job.density_model < job.density_model_count {
        return;
    }
    job.density_model = job.density_model_count - 1;
    job.candidate_start = job
        .candidate_count
        .saturating_sub(job.candidate_tile_size.min(job.candidate_count));
    job.warm_repetition = true;
    job.certified_repetition = false;
}

/// Advances the complete certified pass without scheduling another warm-only
/// tail tile. Returns true only after every density model and candidate tile
/// has been covered.
fn advance_certified_tile(job: &mut PlanningBatchJob, completed_candidates: u32) -> bool {
    job.candidate_start += completed_candidates;
    if job.candidate_start < job.candidate_count {
        return false;
    }
    job.candidate_start = 0;
    job.density_model += 1;
    job.density_model >= job.density_model_count
}

fn top_candidate_scores(
    job: &PlanningBatchJob,
    accuracy_penalty: f32,
) -> [PlanningCandidateScore; 5] {
    let normalization =
        f64::from(job.density_model_count.max(1)) * f64::from(job.samples_per_candidate.max(1));
    let mut scores = (0..job.candidate_count as usize)
        .filter_map(|candidate_index| {
            if !job.candidate_valid[candidate_index] {
                return None;
            }
            let reference = job.candidate_reference_sum[candidate_index];
            let altitude = job.candidate_minimum_altitude_m[candidate_index];
            if (job.density_model_count > 1 && reference <= f64::MIN_POSITIVE)
                || !altitude.is_finite()
                || altitude <= 0.0
            {
                return None;
            }
            let separation = (job.candidate_discrimination_sum[candidate_index]
                / reference.max(f64::MIN_POSITIVE))
            .sqrt() as f32;
            let gradient_information =
                (job.candidate_gradient_sum[candidate_index] / normalization).sqrt() as f32;
            let objective = separation * gradient_information / accuracy_penalty;
            objective
                .is_finite()
                .then_some(PlanningCandidateScore { objective })
        })
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| right.objective.total_cmp(&left.objective));
    let mut top = [PlanningCandidateScore::default(); 5];
    for (destination, score) in top.iter_mut().zip(scores) {
        *destination = score;
    }
    top
}

fn error_distribution(values: &[f32]) -> (f32, f32, f32) {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    finite.sort_by(f32::total_cmp);
    let quantile = |numerator: usize, denominator: usize| {
        let index = ((finite.len() - 1) * numerator).div_ceil(denominator);
        finite[index.min(finite.len() - 1)]
    };
    (
        quantile(95, 100),
        quantile(99, 100),
        finite[finite.len() - 1],
    )
}

fn finish_planning_method(
    job: &PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    backend: PlanningExecutionBackend,
    planning: &mut PlanningComparisonState,
) {
    let (gravity_error_p95, gravity_error_p99, gravity_error_max) =
        error_distribution(&job.pointwise_gravity_errors);
    let (gradient_error_p95, gradient_error_p99, gradient_error_max) =
        error_distribution(&job.pointwise_gradient_errors);
    let (_, certified_gravity_error_p99, certified_gravity_error_max) =
        error_distribution(&job.certified_pointwise_gravity_errors);
    let (_, certified_gradient_error_p99, certified_gradient_error_max) =
        error_distribution(&job.certified_pointwise_gradient_errors);
    let gravity_error =
        (job.gravity_error_sum / job.gravity_reference_sum.max(f64::MIN_POSITIVE)).sqrt() as f32;
    let gradient_error =
        (job.gradient_error_sum / job.gradient_reference_sum.max(f64::MIN_POSITIVE)).sqrt() as f32;
    let model_discrimination = (job.discrimination_sum
        / job.discrimination_reference_sum.max(f64::MIN_POSITIVE))
    .sqrt() as f32;
    let minimum_altitude_m = job.minimum_altitude_m;
    let gradient_information =
        (job.gradient_information_sum / job.total_evaluations.max(1) as f64).sqrt() as f32;
    let planning_objective = model_discrimination * gradient_information
        / (1.0 + gravity_error.max(0.0) + gradient_error.max(0.0));
    let accuracy_penalty = 1.0 + gravity_error.max(0.0) + gradient_error.max(0.0);
    let top_candidates = top_candidate_scores(job, accuracy_penalty);
    let warm_per_candidate =
        job.warm_evaluation_ms / f64::from(job.last_request_candidate_count.max(1));
    let cold_amortization_candidates = ((job.density_payload_preparation_ms
        + job.gpu_preprocessing_ms)
        / warm_per_candidate.max(f64::MIN_POSITIVE))
    .ceil() as u32;
    let raw_gpu_ms = job.gpu_preprocessing_ms
        + job.command_submission_ms
        + job.gpu_completion_map_ms
        + job.readback_decode_ms;
    // CPU preparation and GPU kernel timestamps are separate ledgers. Do not
    // infer a basis cost by subtracting a noisy warm request from cold wall time.
    let geometry_basis_build_ms = job.common_geometry_basis_ms + job.method_geometry_basis_ms;
    let density_model_ms =
        job.density_payload_preparation_ms / f64::from(job.density_model_count.max(1));
    // Request wall cost per output, including amortized GPU setup and readback.
    let target_point_ms = raw_gpu_ms / job.total_evaluations.max(1) as f64;
    let total_ms = geometry_basis_build_ms
        + job.density_payload_preparation_ms
        + job.gpu_preprocessing_ms
        + job.command_submission_ms
        + job.gpu_completion_map_ms
        + job.readback_decode_ms
        + job.reduction_ms;
    // Cumulative checked total, not a warm pass plus a CPU-only approximation
    // of cold setup. This includes the actual Frequency-domain algorithm GPU basis once, just as
    // it includes the FFT/FMM GPU bases once for fixed-target batches. Warm calibration is not charged.
    let certified_estimated_total_ms = total_ms
        + job.certified_density_payload_preparation_ms
        + job.certified_full_pass_ms
        + job.certified_reduction_ms;
    let raw_gpu_request_count = job.raw_gpu_request_count;
    let certified_gpu_request_count = job.gpu_request_count.saturating_sub(raw_gpu_request_count);
    let certified_gravity_error = (job.certified_gravity_error_sum
        / job.certified_gravity_reference_sum.max(f64::MIN_POSITIVE))
    .sqrt() as f32;
    let certified_gradient_error = (job.certified_gradient_error_sum
        / job.certified_gradient_reference_sum.max(f64::MIN_POSITIVE))
    .sqrt() as f32;
    let verified = gravity_error.is_finite()
        && gradient_error.is_finite()
        && (job.method == ActiveGravityMethod::FrequencyDomain
            || job.pericenter_error_m.is_finite())
        && minimum_altitude_m.is_finite()
        && minimum_altitude_m > 0.0
        && model_discrimination.is_finite()
        && planning_objective.is_finite()
        && job.gravity_samples > 0
        && job.gradient_samples > 0
        // K=1 is a valid forward benchmark; separation is zero by definition.
        && (job.density_model_count == 1 || job.discrimination_samples > 0)
        && matches!(
            (job.method, backend),
            (
                ActiveGravityMethod::FrequencyDomain,
                PlanningExecutionBackend::GpuFrequencyDomain
            ) | (
                ActiveGravityMethod::MmfftCompressed,
                PlanningExecutionBackend::CppFlups
            ) | (ActiveGravityMethod::Fmm, PlanningExecutionBackend::CppExafmm)
        );
    planning.results[job.method.performance_index()] = Some(PlanningMethodMetrics {
        method: job.method,
        backend,
        gpu_batch_verified: verified,
        workload: batch.workload_identity(),
        certified_full_pass_ms: job.certified_full_pass_ms,
        certified_estimated_total_ms,
        raw_kernels: job.raw_kernels,
        checked_kernels: job.raw_kernels.plus(job.certified_kernels),
        external_validation_ms: job.verification_ms,
        total_ms,
        geometry_basis_build_ms,
        density_model_ms,
        target_point_ms,
        relative_gravity_error: gravity_error,
        gradient_relative_error: gradient_error,
        certified_relative_gravity_error: certified_gravity_error,
        certified_gradient_relative_error: certified_gradient_error,
        gravity_error_p99,
        gravity_error_max,
        gradient_error_p99,
        gradient_error_max,
        certified_gravity_error_p99,
        certified_gravity_error_max,
        certified_gradient_error_p99,
        certified_gradient_error_max,
        pericenter_error_m: job.pericenter_error_m,
        minimum_altitude_m,
        model_discrimination,
        planning_objective,
        segment_count: if job.method == ActiveGravityMethod::FrequencyDomain {
            job.trajectory_block_count
        } else {
            0
        },
        valid_candidate_count: job.candidate_valid.iter().filter(|valid| **valid).count() as u32,
        verification_sample_count: job.verification_sample_count,
        certified_verification_sample_count: job.certified_verification_sample_count,
        certified_rejected_sample_count: job.certified_rejected_sample_count,
        certified_valid_candidate_count: job
            .certified_candidate_valid
            .iter()
            .filter(|valid| **valid)
            .count() as u32,
        cold_amortization_candidates,
        top_candidates,
    });
    info!(
        target: "planning::benchmark",
        method = ?job.method,
        backend = ?backend,
        total_ms,
        geometry_basis_build_ms,
        density_model_ms,
        target_point_ms,
        method_geometry_basis_ms = job.method_geometry_basis_ms,
        density_payload_preparation_ms = job.density_payload_preparation_ms,
        certified_density_payload_preparation_ms = job.certified_density_payload_preparation_ms,
        gpu_preprocessing_ms = job.gpu_preprocessing_ms,
        command_submission_ms = job.command_submission_ms,
        gpu_completion_map_ms = job.gpu_completion_map_ms,
        readback_decode_ms = job.readback_decode_ms,
        reduction_ms = job.reduction_ms,
        certified_reduction_ms = job.certified_reduction_ms,
        raw_gpu_kernel_ms = ?job.raw_kernels.all_ms,
        raw_gpu_evaluation_ms = ?job.raw_kernels.evaluation_ms,
        raw_gpu_basis_ms = ?job.raw_kernels.basis_ms,
        checked_gpu_kernel_ms = ?job.raw_kernels.plus(job.certified_kernels).all_ms,
        verification_ms = job.verification_ms,
        warm_evaluation_ms = job.warm_evaluation_ms,
        certified_warm_evaluation_ms = job.certified_warm_evaluation_ms,
        certified_estimated_total_ms,
        certified_full_pass_ms = job.certified_full_pass_ms,
        certified_gravity_error,
        certified_gradient_error,
        certified_verification_samples = job.certified_verification_sample_count,
        certified_rejected_samples = job.certified_rejected_sample_count,
        gravity_error,
        gradient_error,
        gravity_error_p95,
        gravity_error_p99,
        gravity_error_max,
        gradient_error_p95,
        gradient_error_p99,
        gradient_error_max,
        valid_candidates = job.candidate_valid.iter().filter(|valid| **valid).count(),
        gpu_requests = raw_gpu_request_count,
        certified_probe_requests = certified_gpu_request_count,
        dispatch_count = job.dispatch_count,
        minimum_tile = job.minimum_tile_size_used.min(job.maximum_tile_size_used),
        maximum_tile = job.maximum_tile_size_used,
        "planning method complete"
    );
    if let Some(verdict) = planning.fair_verdict() {
        info!(target: "planning::benchmark", %verdict, "planning fairness verdict");
    }
}

fn advance_planning_method(job: &mut PlanningBatchJob) {
    job.method_order_index += 1;
    job.method = job.method_order[job.method_order_index];
    job.density_model = 0;
    job.candidate_start = 0;
    job.candidate_tile_size = if job.profile.is_compute_benchmark() {
        // First is deliberately fixed at eight candidates for every method.
        PLANNING_GPU_TILE_INITIAL_CANDIDATES
    } else {
        PLANNING_GENERIC_TILE_INITIAL_CANDIDATES
    };
    job.minimum_tile_size_used = u32::MAX;
    job.maximum_tile_size_used = 0;
    job.gpu_request_count = 0;
    job.raw_gpu_request_count = 0;
    job.last_request_candidate_count = 0;
    job.awaiting_gpu = false;
    job.awaiting_gpu_seconds = 0.0;
    job.awaiting_gpu_last_poll = None;
    job.warm_repetition = false;
    job.certified_repetition = false;
    job.gravity_error_sum = 0.0;
    job.gpu_basis_progress = 0.0;
    job.reference_inflight_fraction = 0.0;
    job.gpu_preparation_submission = 0;
    job.gravity_reference_sum = 0.0;
    job.gravity_samples = 0;
    job.gradient_error_sum = 0.0;
    job.gradient_reference_sum = 0.0;
    job.gradient_samples = 0;
    job.verification_sample_count = 0;
    job.raw_gravity_error_sum = 0.0;
    job.raw_gradient_error_sum = 0.0;
    job.pointwise_gravity_errors.clear();
    job.pointwise_gradient_errors.clear();
    job.certified_pointwise_gravity_errors.clear();
    job.certified_pointwise_gradient_errors.clear();
    job.certified_gravity_error_sum = 0.0;
    job.certified_gravity_reference_sum = 0.0;
    job.certified_gradient_error_sum = 0.0;
    job.certified_gradient_reference_sum = 0.0;
    job.certified_gravity_samples = 0;
    job.certified_gradient_samples = 0;
    job.certified_verification_sample_count = 0;
    job.certified_rejected_sample_count = 0;
    job.certified_candidate_valid.fill(true);
    job.rejected_sample_count = 0;
    job.pericenter_error_m = 0.0;
    job.minimum_altitude_m = f32::INFINITY;
    job.discrimination_sum = 0.0;
    job.discrimination_reference_sum = 0.0;
    job.discrimination_samples = 0;
    job.gradient_information_sum = 0.0;
    job.candidate_discrimination_sum.fill(0.0);
    job.candidate_reference_sum.fill(0.0);
    job.candidate_gradient_sum.fill(0.0);
    job.candidate_minimum_altitude_m.fill(f32::INFINITY);
    job.candidate_valid.fill(true);
    job.method_geometry_basis_ms = 0.0;
    job.density_payload_preparation_ms = 0.0;
    job.certified_density_payload_preparation_ms = 0.0;
    job.gpu_preprocessing_ms = 0.0;
    job.command_submission_ms = 0.0;
    job.reduction_ms = 0.0;
    job.certified_reduction_ms = 0.0;
    job.verification_ms = 0.0;
    job.gpu_completion_map_ms = 0.0;
    job.readback_decode_ms = 0.0;
    job.warm_evaluation_ms = 0.0;
    job.certified_warm_evaluation_ms = 0.0;
    job.certified_full_pass_ms = 0.0;
    job.raw_kernels = PlanningKernelTotals::default();
    job.certified_kernels = PlanningKernelTotals::default();
    job.dispatch_count = 0;
    job.forward_kernel_evaluations = 0;
    job.trajectory_block_count = 0;
}

fn planning_progress_text(job: &PlanningBatchJob) -> String {
    let completed = (u64::from(job.density_model) * u64::from(job.candidate_count)
        + u64::from(job.candidate_start))
        * u64::from(job.samples_per_candidate);
    let phase = if job.warm_repetition {
        "warm repeat"
    } else {
        "cold batch"
    };
    format!(
        "{} {}: {} / {} density combinations, {} model {}, tile {}, GPU requests {}, dispatches {}; random seed {}, mass rel. error {:.2e}.",
        job.profile.label(),
        job.method.planning_label(),
        completed.min(job.total_evaluations),
        job.total_evaluations,
        phase,
        job.density_model + 1,
        job.candidate_tile_size,
        job.gpu_request_count,
        job.dispatch_count,
        job.density_seed,
        job.maximum_density_mass_relative_error,
    )
}
