use bevy::math::{DMat3, DQuat, DVec3};
use std::collections::HashMap;

use crate::cpu::frequency_domain::{
    eq184_laplace_sigma, eq184_quadrature_node, eq184_trajectory_term,
};
use num_complex::Complex64;

// Independent of foreground/background frame rate. Long scheduling gaps are
// suspension, not evidence that a GPU request failed to make progress.
const PLANNING_GPU_WAIT_TIMEOUT_SECONDS: f64 = 300.0;

#[derive(Default)]
pub(crate) struct PlanningReferenceCache {
    identity: Option<(u64, u64, u64)>,
    fields: HashMap<(u64, u64, u32, [u32; 3]), (DVec3, DMat3)>,
    packet_id: Option<u64>,
    target_indices: Vec<u32>,
    target_cursor: usize,
    source_cursor: usize,
    partial_field: DVec3,
    partial_gradient: DMat3,
    frequency_domain_identity: Option<(u64, u64, u32)>,
    frequency_domain_quadrature: Vec<(DVec3, f64)>,
    frequency_domain_density_spectrum: Vec<Complex64>,
    frequency_domain_partial_density_spectrum: Vec<Complex64>,
    frequency_domain_source_cursor: usize,
    frequency_domain_observations: HashMap<(usize, usize), (DVec3, DMat3)>,
}

pub fn planning_batch_evaluator_system(
    batch: Res<PlanningCandidateBatch>,
    channel: Res<PlanningGpuReadbackChannel>,
    mut request: ResMut<PlanningGpuRequest>,
    mut payload: ResMut<PlanningMethodPayload>,
    mut gpu_result: ResMut<PlanningGpuResult>,
    mut planning: ResMut<PlanningComparisonState>,
    mut frequency_domain_workspace: Local<crate::gpu::frequency_domain::PlanningFrequencyDomainWorkspace>,
    mut mmfft_workspace: Local<crate::gpu::mmfft::PlanningMmfftWorkspace>,
    mut reference_cache: Local<PlanningReferenceCache>,
) {
    let Some(mut job) = planning.batch_job.take() else {
        return;
    };
    let render_failure = channel
        .error
        .try_lock()
        .ok()
        .and_then(|mut error| error.take());
    if let Some((failed_request_id, message)) = render_failure
        && failed_request_id == job.request_id
    {
        planning.status = format!(
            "{} stopped: {message}. The GPU lock was released; fix the reported pipeline error and click Quadrature again.",
            job.method.planning_label(),
        );
        planning.run_requested = false;
        planning.source_curve_active = false;
        planning.batch_job = None;
        *request = PlanningGpuRequest::default();
        *payload = PlanningMethodPayload::default();
        gpu_result.0 = None;
        channel
            .in_flight
            .store(false, std::sync::atomic::Ordering::Release);
        return;
    }
    if batch.batch_id == 0 || batch.batch_id != job.batch_id {
        planning.status = "Planning is waiting for the propagated candidate buffers.".into();
        planning.batch_job = Some(job);
        return;
    }
    if !batch.density_mass_is_conserved() {
        planning.status =
            "Planning stopped: randomized voxel densities failed asteroid-mass conservation."
                .into();
        planning.run_requested = false;
        *request = PlanningGpuRequest::default();
        *payload = PlanningMethodPayload::default();
        return;
    }
    if let Some(packet) = gpu_result.0.take() {
        let packet_belongs_to_job = packet.request.batch_id == job.batch_id
            && packet.request.method == Some(job.method)
            && packet.request.warm_repetition == job.warm_repetition;
        if packet_belongs_to_job && packet.request.request_id != job.request_id {
            // A packet from a cancelled/previous request must not strand the
            // current job in `awaiting_gpu=true`.  Drop it and retry the
            // current request on this frame.
            job.awaiting_gpu = false;
            job.awaiting_gpu_seconds = 0.0;
            job.awaiting_gpu_last_poll = None;
            planning.status = format!(
                "{} discarded stale GPU packet {}; retrying request {}.",
                job.method.planning_label(),
                packet.request.request_id,
                job.request_id
            );
        } else if !packet_belongs_to_job && job.awaiting_gpu {
            job.awaiting_gpu = false;
            job.awaiting_gpu_seconds = 0.0;
            job.awaiting_gpu_last_poll = None;
            planning.status = format!(
                "{} discarded mismatched GPU packet; retrying request {}.",
                job.method.planning_label(),
                job.request_id
            );
        } else if packet.request.request_id == job.request_id
            && packet_belongs_to_job
        {
            // Validation must not turn a GPU callback into millions of
            // synchronous CPU source interactions before the browser can paint.
            if packet.readback_valid
                && (!job.warm_repetition || job.certified_repetition)
            {
                let started = bevy::platform::time::Instant::now();
                let ready = prepare_planning_references(&batch, &packet, &mut reference_cache);
                job.verification_ms += started.elapsed().as_secs_f64() * 1.0e3;
                if !ready {
                    let fraction = (reference_cache.target_cursor as f64
                        + reference_cache.source_cursor as f64 / batch.basis_records.len().max(1) as f64)
                        / reference_cache.target_indices.len().max(1) as f64;
                    job.reference_inflight_fraction = f64::from(packet.request.candidate_count) * fraction.clamp(0.0, 1.0);
                    planning.status = format!("{} independent f64 verification: target {}/{}, source {}/{} (time-sliced)",
                        job.method.planning_label(), reference_cache.target_cursor + 1,
                        reference_cache.target_indices.len(), reference_cache.source_cursor, batch.basis_records.len());
                    job.awaiting_gpu_seconds = 0.0;
                    job.awaiting_gpu_last_poll = None;
                    gpu_result.0 = Some(packet);
                    planning.batch_job = Some(job);
                    return;
                }
            }
            if job.warm_repetition {
                let repetition_ms = packet.timing.method_preprocess_ms
                    + packet.timing.command_submission_ms
                    + packet.timing.gpu_completion_map_ms
                    + packet.timing.readback_decode_ms;
                if !job.certified_repetition {
                    job.warm_evaluation_ms = repetition_ms;
                    job.raw_gpu_request_count = job.gpu_request_count;
                    job.certified_repetition = true;
                    job.density_model = 0;
                    job.candidate_start = 0;
                    job.awaiting_gpu = false;
                    job.awaiting_gpu_seconds = 0.0;
                    job.awaiting_gpu_last_poll = None;
                    planning.status = format!(
                        "{} raw pass complete; starting the full independently certified BxKxH pass over the common f64 validation strata.",
                        job.method.planning_label()
                    );
                    planning.batch_job = Some(job);
                    return;
                }
                job.certified_warm_evaluation_ms = repetition_ms;
                job.certified_full_pass_ms += repetition_ms;
                job.certified_kernels.record(packet.timing);
                reduce_certified_packet(&mut job, &batch, &packet, &mut reference_cache);
                if !advance_certified_tile(&mut job, packet.request.candidate_count) {
                    job.awaiting_gpu = false;
                    job.awaiting_gpu_seconds = 0.0;
                    job.awaiting_gpu_last_poll = None;
                    planning.status = planning_progress_text(&job);
                    planning.batch_job = Some(job);
                    return;
                }
                finish_planning_method(&job, &batch, packet.backend, &mut planning);
                *request = PlanningGpuRequest::default();
                *payload = PlanningMethodPayload::default();
                if job.method_order_index + 1 == job.method_order.len() {
                    if planning.source_curve_active {
                        let frequency_domain = planning.results[2].expect("completed Frequency-domain algorithm curve result");
                        let mmfft = planning.results[3].expect("completed FFT curve result");
                        let fmm = planning.results[4].expect("completed FMM curve result");
                        let source_count = planning.requested_source_count;
                        let repeat = planning.source_curve_repeat + 1;
                        let outputs = [(frequency_domain, false), (frequency_domain, true), (mmfft, false),
                            (mmfft, true), (fmm, false), (fmm, true)];
                        let common_samples = frequency_domain.verification_sample_count == mmfft.verification_sample_count
                            && frequency_domain.verification_sample_count == fmm.verification_sample_count;
                        let failure_masks = |profile| outputs.map(|(result, certified)|
                            result.accuracy_failure_mask(profile, certified)
                                | if common_samples { 0 } else { 1 << 8 });
                        let strict_failures = failure_masks(PlanningAccuracyProfile::Strict);
                        let order_seed = planning.source_curve_order_seed
                            ^ (planning.source_curve_samples.len() as u64)
                                .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                        planning.source_curve_samples.push(PlanningSourceCurveSample {
                            source_count,
                            density_model_count: job.density_model_count,
                            target_count: job.samples_per_candidate,
                            repeat,
                            order_seed,
                            method_order: job.method_order.map(|method| method.performance_index()),
                            times_ms: [
                                frequency_domain.total_ms,
                                frequency_domain.certified_estimated_total_ms,
                                mmfft.total_ms,
                                mmfft.certified_estimated_total_ms,
                                fmm.total_ms,
                                fmm.certified_estimated_total_ms,
                            ],
                            kernel_times_ms: [frequency_domain.raw_kernels.all_ms, frequency_domain.checked_kernels.all_ms,
                                mmfft.raw_kernels.all_ms, mmfft.checked_kernels.all_ms,
                                fmm.raw_kernels.all_ms, fmm.checked_kernels.all_ms],
                            evaluation_kernel_times_ms: [frequency_domain.raw_kernels.evaluation_ms, frequency_domain.checked_kernels.evaluation_ms,
                                mmfft.raw_kernels.evaluation_ms, mmfft.checked_kernels.evaluation_ms,
                                fmm.raw_kernels.evaluation_ms, fmm.checked_kernels.evaluation_ms],
                            basis_kernel_times_ms: [frequency_domain.raw_kernels.basis_ms, mmfft.raw_kernels.basis_ms, fmm.raw_kernels.basis_ms],
                            geometry_basis_build_ms: [
                                frequency_domain.geometry_basis_build_ms,
                                mmfft.geometry_basis_build_ms,
                                fmm.geometry_basis_build_ms,
                            ],
                            density_model_ms: [
                                frequency_domain.density_model_ms,
                                mmfft.density_model_ms,
                                fmm.density_model_ms,
                            ],
                            target_point_ms: [
                                frequency_domain.target_point_ms,
                                mmfft.target_point_ms,
                                fmm.target_point_ms,
                            ],
                            eligible: strict_failures.map(|mask| mask == 0),
                            strict_failures,
                            screening_failures: failure_masks(PlanningAccuracyProfile::Screening),
                            gravity_errors: [frequency_domain.relative_gravity_error, frequency_domain.certified_relative_gravity_error,
                                mmfft.relative_gravity_error, mmfft.certified_relative_gravity_error,
                                fmm.relative_gravity_error, fmm.certified_relative_gravity_error],
                            gradient_errors: [frequency_domain.gradient_relative_error, frequency_domain.certified_gradient_relative_error,
                                mmfft.gradient_relative_error, mmfft.certified_gradient_relative_error,
                                fmm.gradient_relative_error, fmm.certified_gradient_relative_error],
                        });
                        if planning.advance_source_curve() {
                            planning.preparation_progress = 0.0;
                            planning.results = std::array::from_fn(|_| None);
                            planning.run_id = planning.run_id.wrapping_add(1);
                            planning.status = format!(
                                "Quadrature sweep queued: {} sources, {} density models, {} targets, repeat {}/{} (random method order).",
                                planning.requested_source_count,
                                planning.dimensions().1,
                                planning.dimensions().2,
                                planning.source_curve_repeat + 1,
                                PLANNING_SOURCE_REPEATS
                            );
                            return;
                        }
                        planning.computation_complete = true;
                        planning.source_curve_active = false;
                        planning.source_curve_visible = true;
                        planning.run_requested = false;
                        planning.status =
                            "Quadrature sweep complete: seven repeats per source/K/target cell; only fully accuracy-qualified cells enter timing comparisons.".into();
                        return;
                    }
                    planning.run_requested = false;
                    planning.computation_complete = true;
                    planning.status = format!(
                        "{} screening complete: all methods used identical nominal-density leapfrog-propagated candidates. Density rows share those trajectories; model-specific repropagation is not claimed.",
                        job.profile.label()
                    );
                    return;
                }
                advance_planning_method(&mut job);
            } else {
                reduce_planning_packet(&mut job, &batch, &packet, &mut reference_cache);
                adapt_candidate_tile(&mut job, &packet);
                advance_planning_tile(&mut job, packet.request.candidate_count);
            }
            job.reference_inflight_fraction = 0.0;
            job.awaiting_gpu = false;
            job.awaiting_gpu_seconds = 0.0;
            job.awaiting_gpu_last_poll = None;
        }
    }
    if job.awaiting_gpu {
        let preparation = channel.preparation.try_lock().ok()
            .and_then(|progress| progress.as_ref().filter(|p| p.request_id == job.request_id).cloned());
        if let Some(progress) = &preparation {
            if job.gpu_preparation_submission != progress.completed_submissions {
                job.gpu_preparation_submission = progress.completed_submissions;
                job.awaiting_gpu_seconds = 0.0; // A completed GPU stage is real progress.
                job.awaiting_gpu_last_poll = None;
            }
            if !job.warm_repetition { job.gpu_basis_progress = progress.basis_fraction; }
        }
        let now = bevy::platform::time::Instant::now();
        if let Some(last) = job.awaiting_gpu_last_poll.replace(now) {
            let elapsed = now.duration_since(last).as_secs_f64();
            if elapsed <= 2.0 { job.awaiting_gpu_seconds += elapsed; }
        }
        if job.awaiting_gpu_seconds >= PLANNING_GPU_WAIT_TIMEOUT_SECONDS {
            planning.status = format!(
                "{} stopped after {} active seconds without a matching GPU readback (request {}).",
                job.method.planning_label(),
                PLANNING_GPU_WAIT_TIMEOUT_SECONDS,
                job.request_id
            );
            planning.run_requested = false;
            job.awaiting_gpu = false;
            *request = PlanningGpuRequest::default();
            *payload = PlanningMethodPayload::default();
            planning.batch_job = None;
            return;
        }
        planning.status = preparation.map(|p| p.status).unwrap_or_else(|| planning_progress_text(&job));
        planning.batch_job = Some(job);
        return;
    }
    let payload_key = job.run_id
        ^ (job.method.performance_index() as u64).rotate_left(17)
        ^ u64::from(job.density_model).rotate_left(33);
    if payload.request_id != payload_key
        || payload.method != Some(job.method)
        || payload.density_model != job.density_model
    {
        let prepared = match job.method {
            ActiveGravityMethod::FrequencyDomain => {
                crate::gpu::frequency_domain::build_planning_frequency_domain_payload(
                    &batch,
                    job.density_model,
                    payload_key,
                    &mut frequency_domain_workspace,
                )
            }
            ActiveGravityMethod::MmfftCompressed => {
                crate::gpu::mmfft::build_planning_mmfft_payload(
                    &batch,
                    job.density_model,
                    payload_key,
                    &mut mmfft_workspace,
                )
            }
            ActiveGravityMethod::Fmm => crate::gpu::fmm::build_planning_fmm_gpu_payload(
                &batch,
                job.density_model,
                payload_key,
            ),
            _ => None,
        };
        let Some(prepared) = prepared else {
            planning.status = format!(
                "{} planning stopped: payload preparation failed for density model {}.",
                job.method.planning_label(),
                job.density_model
            );
            planning.run_requested = false;
            *request = PlanningGpuRequest::default();
            *payload = PlanningMethodPayload::default();
            return;
        };
        if !job.warm_repetition {
            job.method_geometry_basis_ms += prepared.geometry_basis_preparation_ms;
            job.density_payload_preparation_ms += prepared.density_payload_preparation_ms;
        } else if job.certified_repetition {
            job.certified_density_payload_preparation_ms +=
                prepared.density_payload_preparation_ms;
        }
        *payload = prepared;
    }
    job.gpu_preparation_submission = 0;
    job.request_id = job.request_id.wrapping_add(1).max(1);
    job.candidate_tile_size = job.candidate_tile_size.min((8192 / job.samples_per_candidate.max(1)).max(1));
    let request_candidate_count = job
        .candidate_tile_size
        .min(job.candidate_count - job.candidate_start)
        // Keep target basis windows bounded for every method, even Nt=8192.
        .min((8192 / job.samples_per_candidate.max(1)).max(1));
    job.last_request_candidate_count = request_candidate_count;
    job.minimum_tile_size_used = job.minimum_tile_size_used.min(request_candidate_count);
    job.maximum_tile_size_used = job.maximum_tile_size_used.max(request_candidate_count);
    job.gpu_request_count = job.gpu_request_count.saturating_add(1);
    *request = PlanningGpuRequest {
        request_id: job.request_id,
        batch_id: job.batch_id,
        method: Some(job.method),
        density_model: job.density_model,
        candidate_start: job.candidate_start,
        candidate_count: request_candidate_count,
        warm_repetition: job.warm_repetition,
        compute_benchmark: job.profile.is_compute_benchmark(),
    };
    job.awaiting_gpu = true;
    job.awaiting_gpu_seconds = 0.0;
    job.awaiting_gpu_last_poll = None;
    planning.status = planning_progress_text(&job);
    planning.batch_job = Some(job);
}

fn reduce_planning_packet(
    job: &mut PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    reference_cache: &mut PlanningReferenceCache,
) {
    let reduction_started = bevy::platform::time::Instant::now();
    let mut verification_ms = 0.0;
    job.raw_kernels.record(packet.timing);
    job.verification_sample_count = job
        .verification_sample_count
        .saturating_add(packet.state_indices.len() as u64);
    job.rejected_sample_count = job
        .rejected_sample_count
        .saturating_add(packet.rejected_sample_count);
    if !packet.readback_valid
        || packet.rows.len() != packet.state_indices.len() * 4
        || packet.raw_rows.len() != packet.state_indices.len() * 4
        || packet.candidate_metrics.len() != packet.request.candidate_count as usize
        || packet
            .rows
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
        || packet
            .raw_rows
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
        || packet
            .candidate_metrics
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        job.gravity_error_sum = f64::NAN;
        job.gradient_error_sum = f64::NAN;
        job.gpu_preprocessing_ms += packet.timing.method_preprocess_ms;
        job.command_submission_ms += packet.timing.command_submission_ms;
        job.gpu_completion_map_ms += packet.timing.gpu_completion_map_ms;
        job.readback_decode_ms += packet.timing.readback_decode_ms;
        job.dispatch_count = job.dispatch_count.saturating_add(packet.timing.dispatch_count);
        job.forward_kernel_evaluations = job
            .forward_kernel_evaluations
            .saturating_add(packet.timing.forward_kernel_evaluations);
        job.trajectory_block_count = job
            .trajectory_block_count
            .max(packet.timing.trajectory_block_count);
        return;
    }
    for (local_candidate, metric) in packet.candidate_metrics.iter().enumerate() {
        let candidate_index = packet.request.candidate_start as usize + local_candidate;
        if metric[0] < 0.0 {
            job.candidate_valid[candidate_index] = false;
            continue;
        }
        job.minimum_altitude_m = job.minimum_altitude_m.min(metric[2]);
        job.gradient_information_sum += f64::from(metric[3]);
        job.candidate_gradient_sum[candidate_index] += f64::from(metric[3]);
        job.candidate_minimum_altitude_m[candidate_index] =
            job.candidate_minimum_altitude_m[candidate_index].min(metric[2]);
        if packet.request.density_model > 0 {
            job.discrimination_sum += f64::from(metric[0]);
            job.discrimination_reference_sum += f64::from(metric[1]);
            job.discrimination_samples += 1;
            job.candidate_discrimination_sum[candidate_index] += f64::from(metric[0]);
            job.candidate_reference_sum[candidate_index] += f64::from(metric[1]);
        }
    }
    let global_start = packet.request.candidate_start as usize
        * batch.samples_per_candidate as usize;
    let mut accumulated_position_error =
        vec![DVec3::ZERO; packet.request.candidate_count as usize];
    let mut accumulated_velocity_error =
        vec![DVec3::ZERO; packet.request.candidate_count as usize];
    let mut previous_verified_time = vec![None; packet.request.candidate_count as usize];
    for (verification_index, local_target) in packet.state_indices.iter().copied().enumerate() {
        let local = local_target as usize;
        let state = batch.states[global_start + local];
        let sample = state.identity[1];
        let verify_pericenter = sample.abs_diff(batch.samples_per_candidate / 2) <= 1;
        let method_field = DVec3::new(
            f64::from(packet.rows[verification_index * 4][0]),
            f64::from(packet.rows[verification_index * 4][1]),
            f64::from(packet.rows[verification_index * 4][2]),
        );
        let method_gradient = DMat3::from_cols(
            DVec3::new(
                f64::from(packet.rows[verification_index * 4 + 1][0]),
                f64::from(packet.rows[verification_index * 4 + 1][1]),
                f64::from(packet.rows[verification_index * 4 + 1][2]),
            ),
            DVec3::new(
                f64::from(packet.rows[verification_index * 4 + 2][0]),
                f64::from(packet.rows[verification_index * 4 + 2][1]),
                f64::from(packet.rows[verification_index * 4 + 2][2]),
            ),
            DVec3::new(
                f64::from(packet.rows[verification_index * 4 + 3][0]),
                f64::from(packet.rows[verification_index * 4 + 3][1]),
                f64::from(packet.rows[verification_index * 4 + 3][2]),
            ),
        );
        let raw_field = DVec3::new(
            f64::from(packet.raw_rows[verification_index * 4][0]),
            f64::from(packet.raw_rows[verification_index * 4][1]),
            f64::from(packet.raw_rows[verification_index * 4][2]),
        );
        let raw_gradient = DMat3::from_cols(
            DVec3::new(
                f64::from(packet.raw_rows[verification_index * 4 + 1][0]),
                f64::from(packet.raw_rows[verification_index * 4 + 1][1]),
                f64::from(packet.raw_rows[verification_index * 4 + 1][2]),
            ),
            DVec3::new(
                f64::from(packet.raw_rows[verification_index * 4 + 2][0]),
                f64::from(packet.raw_rows[verification_index * 4 + 2][1]),
                f64::from(packet.raw_rows[verification_index * 4 + 2][2]),
            ),
            DVec3::new(
                f64::from(packet.raw_rows[verification_index * 4 + 3][0]),
                f64::from(packet.raw_rows[verification_index * 4 + 3][1]),
                f64::from(packet.raw_rows[verification_index * 4 + 3][2]),
            ),
        );
        if packet.request.method == Some(ActiveGravityMethod::FrequencyDomain) {
            let local_candidate = local / batch.samples_per_candidate as usize;
            let observation_index = local % batch.samples_per_candidate as usize;
            let verification_started = bevy::platform::time::Instant::now();
            let (aggregate_field, aggregate_gradient) = frequency_domain_reference_integral(
                batch,
                packet.request.candidate_start as usize + local_candidate,
                observation_index,
                reference_cache,
            );
            verification_ms += verification_started.elapsed().as_secs_f64() * 1.0e3;
            if !method_field.is_finite()
                || !method_gradient.is_finite()
                || !aggregate_field.is_finite()
                || !aggregate_gradient.is_finite()
            {
                job.gravity_error_sum = f64::NAN;
                continue;
            }
            job.gravity_error_sum += (method_field - aggregate_field).length_squared();
            job.gravity_reference_sum += aggregate_field.length_squared();
            job.gravity_samples += 1;
            job.gradient_error_sum += matrix_norm_squared(method_gradient - aggregate_gradient);
            job.gradient_reference_sum += matrix_norm_squared(aggregate_gradient);
            job.gradient_samples += 1;
            job.pointwise_gravity_errors.push(
                ((method_field - aggregate_field).length()
                    / aggregate_field.length().max(f64::MIN_POSITIVE)) as f32,
            );
            job.pointwise_gradient_errors.push(
                (matrix_norm_squared(method_gradient - aggregate_gradient).sqrt()
                    / matrix_norm_squared(aggregate_gradient).sqrt().max(f64::MIN_POSITIVE))
                    as f32,
            );
            continue;
        }
        let verification_started = bevy::platform::time::Instant::now();
        let (reference_field, reference_gradient) = direct_planning_reference_cached(
            state.body_position().as_dvec3(),
            batch,
            packet.request.density_model,
            reference_cache,
        );
        verification_ms += verification_started.elapsed().as_secs_f64() * 1.0e3;
        if !method_field.is_finite()
            || !method_gradient.is_finite()
            || !reference_field.is_finite()
            || !reference_gradient.is_finite()
        {
            job.gravity_error_sum = f64::NAN;
            continue;
        }
        job.gravity_error_sum += (method_field - reference_field).length_squared();
        job.gravity_reference_sum += reference_field.length_squared();
        job.gravity_samples += 1;
        job.gradient_error_sum += matrix_norm_squared(method_gradient - reference_gradient);
        job.gradient_reference_sum += matrix_norm_squared(reference_gradient);
        job.gradient_samples += 1;
        job.raw_gravity_error_sum += (raw_field - reference_field).length_squared();
        job.raw_gradient_error_sum += matrix_norm_squared(raw_gradient - reference_gradient);
        job.pointwise_gravity_errors.push(
            ((method_field - reference_field).length()
                / reference_field.length().max(f64::MIN_POSITIVE)) as f32,
        );
        job.pointwise_gradient_errors.push(
            (matrix_norm_squared(method_gradient - reference_gradient).sqrt()
                / matrix_norm_squared(reference_gradient)
                    .sqrt()
                    .max(f64::MIN_POSITIVE)) as f32,
        );
        let local_candidate = local / batch.samples_per_candidate as usize;
        let current_time = f64::from(state.position_time[3]);
        if let Some(previous_time) = previous_verified_time[local_candidate] {
            let delta_time = current_time - previous_time;
            let rotation = DQuat::from_xyzw(
                f64::from(state.body_rotation[0]),
                f64::from(state.body_rotation[1]),
                f64::from(state.body_rotation[2]),
                f64::from(state.body_rotation[3]),
            );
            let acceleration_error = rotation * (method_field - reference_field);
            accumulated_position_error[local_candidate] +=
                accumulated_velocity_error[local_candidate] * delta_time
                    + 0.5 * acceleration_error * delta_time * delta_time;
            accumulated_velocity_error[local_candidate] += acceleration_error * delta_time;
        }
        previous_verified_time[local_candidate] = Some(current_time);
        if verify_pericenter {
            let rotation = DQuat::from_xyzw(
                f64::from(state.body_rotation[0]),
                f64::from(state.body_rotation[1]),
                f64::from(state.body_rotation[2]),
                f64::from(state.body_rotation[3]),
            );
            let radial = (rotation * state.body_position().as_dvec3()).normalize_or_zero();
            job.pericenter_error_m = job
                .pericenter_error_m
                .max(accumulated_position_error[local_candidate].dot(radial).abs() as f32);
        }
    }
    job.gpu_preprocessing_ms += packet.timing.method_preprocess_ms;
    job.command_submission_ms += packet.timing.command_submission_ms;
    job.gpu_completion_map_ms += packet.timing.gpu_completion_map_ms;
    job.readback_decode_ms += packet.timing.readback_decode_ms;
    job.dispatch_count = job.dispatch_count.saturating_add(packet.timing.dispatch_count);
    job.forward_kernel_evaluations = job
        .forward_kernel_evaluations
        .saturating_add(packet.timing.forward_kernel_evaluations);
    job.trajectory_block_count = job
        .trajectory_block_count
        .max(packet.timing.trajectory_block_count);
    let total_reduction_ms = reduction_started.elapsed().as_secs_f64() * 1.0e3;
    job.verification_ms += verification_ms;
    job.reduction_ms += (total_reduction_ms - verification_ms).max(0.0);
}

fn reduce_certified_packet(
    job: &mut PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    reference_cache: &mut PlanningReferenceCache,
) {
    let reduction_started = bevy::platform::time::Instant::now();
    let mut verification_ms = 0.0;
    // Every frequency-domain row is an independent whole-trajectory
    // equation-(184) observation at its own Laplace frequency.
    let frequency_domain = packet.request.method == Some(ActiveGravityMethod::FrequencyDomain);
    job.certified_verification_sample_count = job
        .certified_verification_sample_count
        .saturating_add(packet.state_indices.len() as u64);
    job.certified_rejected_sample_count = job
        .certified_rejected_sample_count
        .saturating_add(packet.rejected_sample_count);
    for (local_candidate, metric) in packet.candidate_metrics.iter().enumerate() {
        if (metric[0] < 0.0 || metric.iter().any(|value| !value.is_finite()))
            && let Some(valid) = job.certified_candidate_valid.get_mut(
                packet.request.candidate_start as usize + local_candidate,
            )
        {
            *valid = false;
        }
    }
    if !packet.readback_valid || packet.rows.len() != packet.state_indices.len() * 4 {
        job.certified_gravity_error_sum = f64::NAN;
        job.certified_gradient_error_sum = f64::NAN;
        job.certified_reduction_ms += reduction_started.elapsed().as_secs_f64() * 1.0e3;
        return;
    }
    let global_start = packet.request.candidate_start as usize
        * batch.samples_per_candidate as usize;
    if frequency_domain {
        // Verify every selected Laplace-frequency observation against the same
        // aggregate spectral operator. Never compare with an instantaneous
        // direct field or integrate the transform as a physical acceleration.
        for (verification_index, local_target) in packet.state_indices.iter().copied().enumerate() {
            let local = local_target as usize;
            let local_candidate = local / batch.samples_per_candidate as usize;
            let observation_index = local % batch.samples_per_candidate as usize;
            let row = verification_index * 4;
            let method_field = DVec3::new(
                f64::from(packet.rows[row][0]),
                f64::from(packet.rows[row][1]),
                f64::from(packet.rows[row][2]),
            );
            let method_gradient = DMat3::from_cols(
                DVec3::new(f64::from(packet.rows[row + 1][0]), f64::from(packet.rows[row + 1][1]), f64::from(packet.rows[row + 1][2])),
                DVec3::new(f64::from(packet.rows[row + 2][0]), f64::from(packet.rows[row + 2][1]), f64::from(packet.rows[row + 2][2])),
                DVec3::new(f64::from(packet.rows[row + 3][0]), f64::from(packet.rows[row + 3][1]), f64::from(packet.rows[row + 3][2])),
            );
            let verification_started = bevy::platform::time::Instant::now();
            let (reference_field, reference_gradient) = frequency_domain_reference_integral(
                batch,
                packet.request.candidate_start as usize + local_candidate,
                observation_index,
                reference_cache,
            );
            verification_ms += verification_started.elapsed().as_secs_f64() * 1.0e3;
            if !method_field.is_finite() || !method_gradient.is_finite()
                || !reference_field.is_finite() || !reference_gradient.is_finite()
            {
                job.certified_gravity_error_sum = f64::NAN;
                job.certified_gradient_error_sum = f64::NAN;
                continue;
            }
            job.certified_gravity_error_sum += (method_field - reference_field).length_squared();
            job.certified_gravity_reference_sum += reference_field.length_squared();
            job.certified_gradient_error_sum += matrix_norm_squared(method_gradient - reference_gradient);
            job.certified_gradient_reference_sum += matrix_norm_squared(reference_gradient);
            job.certified_pointwise_gravity_errors.push((
                (method_field - reference_field).length() / reference_field.length().max(f64::MIN_POSITIVE)
            ) as f32);
            job.certified_pointwise_gradient_errors.push((
                matrix_norm_squared(method_gradient - reference_gradient).sqrt()
                    / matrix_norm_squared(reference_gradient).sqrt().max(f64::MIN_POSITIVE)
            ) as f32);
            job.certified_gravity_samples += 1;
            job.certified_gradient_samples += 1;
        }
        job.verification_ms += verification_ms;
        job.certified_reduction_ms +=
            (reduction_started.elapsed().as_secs_f64() * 1.0e3 - verification_ms).max(0.0);
        return;
    }
    for (verification_index, local_target) in packet.state_indices.iter().copied().enumerate() {
        let state = batch.states[global_start + local_target as usize];
        let row = verification_index * 4;
        let method_field = DVec3::new(
            f64::from(packet.rows[row][0]),
            f64::from(packet.rows[row][1]),
            f64::from(packet.rows[row][2]),
        );
        let method_gradient = DMat3::from_cols(
            DVec3::new(
                f64::from(packet.rows[row + 1][0]),
                f64::from(packet.rows[row + 1][1]),
                f64::from(packet.rows[row + 1][2]),
            ),
            DVec3::new(
                f64::from(packet.rows[row + 2][0]),
                f64::from(packet.rows[row + 2][1]),
                f64::from(packet.rows[row + 2][2]),
            ),
            DVec3::new(
                f64::from(packet.rows[row + 3][0]),
                f64::from(packet.rows[row + 3][1]),
                f64::from(packet.rows[row + 3][2]),
            ),
        );
        let verification_started = bevy::platform::time::Instant::now();
        let (reference_field, reference_gradient) = direct_planning_reference_cached(
            state.body_position().as_dvec3(),
            batch,
            packet.request.density_model,
            reference_cache,
        );
        verification_ms += verification_started.elapsed().as_secs_f64() * 1.0e3;
        if !method_field.is_finite()
            || !method_gradient.is_finite()
            || !reference_field.is_finite()
            || !reference_gradient.is_finite()
        {
            job.certified_gravity_error_sum = f64::NAN;
            job.certified_gradient_error_sum = f64::NAN;
            continue;
        }
        job.certified_gravity_error_sum += (method_field - reference_field).length_squared();
        job.certified_gravity_reference_sum += reference_field.length_squared();
        job.certified_gradient_error_sum +=
            matrix_norm_squared(method_gradient - reference_gradient);
        job.certified_gradient_reference_sum += matrix_norm_squared(reference_gradient);
        job.certified_pointwise_gravity_errors.push(((method_field - reference_field).length()
            / reference_field.length().max(f64::MIN_POSITIVE)) as f32);
        job.certified_pointwise_gradient_errors.push((matrix_norm_squared(method_gradient - reference_gradient).sqrt()
            / matrix_norm_squared(reference_gradient).sqrt().max(f64::MIN_POSITIVE)) as f32);
        job.certified_gravity_samples += 1;
        job.certified_gradient_samples += 1;
    }
    job.verification_ms += verification_ms;
    job.certified_reduction_ms += (reduction_started.elapsed().as_secs_f64() * 1.0e3 - verification_ms).max(0.0);

}

fn reference_key(target: DVec3, batch: &PlanningCandidateBatch, model: u32) -> (u64, u64, u32, [u32; 3]) {
    (batch.basis_hash, batch.density_model_hash, model,
     [(target.x as f32).to_bits(), (target.y as f32).to_bits(), (target.z as f32).to_bits()])
}

fn prepare_planning_references(
    batch: &PlanningCandidateBatch, packet: &PlanningGpuPacket, cache: &mut PlanningReferenceCache,
) -> bool {
    if packet.request.method == Some(ActiveGravityMethod::FrequencyDomain) {
        return prepare_frequency_domain_reference(batch, packet, cache);
    }
    let identity = (batch.basis_hash, batch.density_model_hash, batch.sample_hash);
    if cache.identity != Some(identity) {
        cache.fields.clear();
        cache.identity = Some(identity);
        cache.packet_id = None;
    }
    if cache.packet_id != Some(packet.request.request_id) {
        cache.packet_id = Some(packet.request.request_id);
        cache.target_indices = if packet.request.method
            == Some(ActiveGravityMethod::FrequencyDomain)
        {
            let count = packet.request.candidate_count as usize
                * batch.samples_per_candidate as usize;
            (0..count).map(|index| index as u32).collect()
        } else {
            packet.state_indices.clone()
        };
        cache.target_cursor = 0;
        cache.source_cursor = 0;
        cache.partial_field = DVec3::ZERO;
        cache.partial_gradient = DMat3::ZERO;
    }
    let started = bevy::platform::time::Instant::now();
    let global_start = packet.request.candidate_start as usize * batch.samples_per_candidate as usize;
    let row = packet.request.density_model as usize * 56;
    while cache.target_cursor < cache.target_indices.len() {
        let state_index = global_start + cache.target_indices[cache.target_cursor] as usize;
        let Some(state) = batch.states.get(state_index) else { return true; }; // reduction rejects malformed output
        let target = state.body_position().as_dvec3();
        let key = reference_key(target, batch, packet.request.density_model);
        if cache.fields.contains_key(&key) {
            cache.target_cursor += 1;
            continue;
        }
        let end = (cache.source_cursor + 512).min(batch.basis_records.len());
        let valid = crate::cpu::planning::accumulate_planning_reference_chunk(
            target, &batch.basis_records[cache.source_cursor..end], &batch.density_models[row..row+56],
            &mut cache.partial_field, &mut cache.partial_gradient).is_some();
        cache.source_cursor = end;
        if !valid || end == batch.basis_records.len() {
            let value = if valid {(cache.partial_field, cache.partial_gradient)} else {(DVec3::NAN, DMat3::NAN)};
            cache.fields.insert(key, value);
            cache.partial_field = DVec3::ZERO;
            cache.partial_gradient = DMat3::ZERO;
            cache.source_cursor = 0;
            cache.target_cursor += 1;
        }
        if started.elapsed().as_secs_f64() >= 0.003 { return false; }
    }
    true
}

fn prepare_frequency_domain_reference(
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    cache: &mut PlanningReferenceCache,
) -> bool {
    let identity = (
        batch.basis_hash,
        batch.density_model_hash,
        packet.request.density_model,
    );
    if cache.frequency_domain_identity != Some(identity) {
        let quadrature = (0..EQ184_QUADRATURE_COUNT)
            .map(|index| {
                let (wave_vector, volume_weight) = eq184_quadrature_node(
                    index,
                    f64::from(batch.frequency_domain_source_radius),
                )?;
                let coefficient = f64::from(crate::interface::components::G)
                    * 4.0
                    * std::f64::consts::PI
                    / std::f64::consts::TAU.powi(3)
                    * volume_weight
                    / wave_vector.length_squared().max(1.0e-18);
                Some((wave_vector, coefficient.clamp(-1.0e20, 1.0e20)))
            })
            .collect::<Option<Vec<_>>>();
        let Some(quadrature) = quadrature else {
            return false;
        };
        cache.frequency_domain_identity = Some(identity);
        cache.frequency_domain_quadrature = quadrature;
        cache.frequency_domain_density_spectrum = Vec::new();
        cache.frequency_domain_partial_density_spectrum =
            vec![Complex64::new(0.0, 0.0); crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT];
        cache.frequency_domain_source_cursor = 0;
        cache.frequency_domain_observations.clear();
    }

    let density_row = packet.request.density_model as usize * 56;
    let Some(densities) = batch.density_models.get(density_row..density_row + 56) else {
        return false;
    };
    let started = bevy::platform::time::Instant::now();
    while cache.frequency_domain_source_cursor < batch.basis_records.len() {
        let end = (cache.frequency_domain_source_cursor + 256).min(batch.basis_records.len());
        for source in &batch.basis_records[cache.frequency_domain_source_cursor..end] {
            let voxel_density = f64::from(*densities.get(source.voxel_index as usize).unwrap_or(&f32::NAN));
            let volume_density = f64::from(source.position_volume[3]) * voxel_density;
            let position = DVec3::new(
                f64::from(source.position_volume[0]),
                f64::from(source.position_volume[1]),
                f64::from(source.position_volume[2]),
            );
            if !position.is_finite() || !volume_density.is_finite() {
                return false;
            }
            for (spectrum, (wave_vector, _)) in cache
                .frequency_domain_partial_density_spectrum
                .iter_mut()
                .zip(&cache.frequency_domain_quadrature)
            {
                *spectrum += Complex64::from_polar(volume_density, -wave_vector.dot(position));
            }
        }
        cache.frequency_domain_source_cursor = end;
        if started.elapsed().as_secs_f64() >= 0.003 {
            return false;
        }
    }
    if cache.frequency_domain_density_spectrum.is_empty() {
        cache.frequency_domain_density_spectrum =
            std::mem::take(&mut cache.frequency_domain_partial_density_spectrum);
    }
    true
}

fn direct_planning_reference_cached(
    target: DVec3, batch: &PlanningCandidateBatch, density_model: u32,
    cache: &mut PlanningReferenceCache,
) -> (DVec3, DMat3) {
    // Preflight populated every requested reference in bounded frame slices.
    // Missing entries fail accuracy; never hide a synchronous full-source solve
    // here, and never validate one GPU algorithm against its own approximation.
    cache.fields.get(&reference_key(target, batch, density_model)).copied()
        .unwrap_or((DVec3::NAN, DMat3::NAN))
}

/// Independent f64 reference for one discrete frequency-domain observation.
/// This mirrors the shader's rho-hat(k) * T_gamma(s,k) reciprocal-space
/// operator, including its quadrature, phase convention, Laplace attenuation,
/// Newton multiplier, and Jacobian column layout.
fn frequency_domain_reference_integral(
    batch: &PlanningCandidateBatch,
    candidate_index: usize,
    observation_index: usize,
    cache: &mut PlanningReferenceCache,
) -> (DVec3, DMat3) {
    let key = (candidate_index, observation_index);
    if let Some(result) = cache.frequency_domain_observations.get(&key) {
        return *result;
    }
    let samples = batch.samples_per_candidate as usize;
    let start = candidate_index.saturating_mul(samples);
    if start + samples > batch.states.len() {
        return (DVec3::NAN, DMat3::NAN);
    }
    if cache.frequency_domain_density_spectrum.len()
        != crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT
        || cache.frequency_domain_quadrature.len()
            != crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT
    {
        return (DVec3::NAN, DMat3::NAN);
    }
    let laplace_frequency = eq184_laplace_sigma(observation_index, samples);
    let mut result_field = DVec3::ZERO;
    let mut result_gradient = DMat3::ZERO;
    for (index, (wave_vector, coefficient)) in cache
        .frequency_domain_quadrature
        .iter()
        .enumerate()
    {
        let trajectory = (0..samples).try_fold(Complex64::new(0.0, 0.0), |sum, sample_index| {
            let sample = batch.states[start + sample_index];
            let previous = if sample_index > 0 {
                batch.states[start + sample_index - 1]
            } else {
                sample
            };
            let next = if sample_index + 1 < samples {
                batch.states[start + sample_index + 1]
            } else {
                sample
            };
            Some(
                sum + eq184_trajectory_term(
                    *wave_vector,
                    sample.body_position().as_dvec3(),
                    f64::from(previous.position_time[3]),
                    f64::from(sample.position_time[3]),
                    f64::from(next.position_time[3]),
                    sample_index,
                    samples,
                    laplace_frequency,
                )?,
            )
        });
        let Some(trajectory) = trajectory else {
            return (DVec3::NAN, DMat3::NAN);
        };
        let product = cache.frequency_domain_density_spectrum[index] * trajectory;
        result_field += -*coefficient * product.im * *wave_vector;
        let hessian_scale = -*coefficient * product.re;
        let jacobian_x = hessian_scale * *wave_vector * wave_vector.x;
        let jacobian_y = hessian_scale * *wave_vector * wave_vector.y;
        let jacobian_z = hessian_scale * *wave_vector * wave_vector.z;
        result_gradient += DMat3::from_cols(jacobian_x, jacobian_y, jacobian_z);
    }
    let result = (result_field, result_gradient);
    if result_field.is_finite() && result_gradient.is_finite() {
        cache.frequency_domain_observations.insert(key, result);
    }
    result
}

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
        || recent_frame_ms.is_some_and(|milliseconds| {
            milliseconds > PLANNING_MAX_RECENT_FRAME_MS
        });
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
    let normalization = f64::from(job.density_model_count.max(1))
        * f64::from(job.samples_per_candidate.max(1));
    let mut scores = (0..job.candidate_count as usize)
        .filter_map(|candidate_index| {
            if !job.candidate_valid[candidate_index] {
                return None;
            }
            let reference = job.candidate_reference_sum[candidate_index];
            let altitude = job.candidate_minimum_altitude_m[candidate_index];
            if (job.density_model_count > 1 && reference <= f64::MIN_POSITIVE)
                || !altitude.is_finite() || altitude <= 0.0 {
                return None;
            }
            let separation =
                (job.candidate_discrimination_sum[candidate_index] / reference.max(f64::MIN_POSITIVE)).sqrt() as f32;
            let gradient_information =
                (job.candidate_gradient_sum[candidate_index] / normalization).sqrt() as f32;
            let objective = separation * gradient_information / accuracy_penalty;
            objective.is_finite().then_some(PlanningCandidateScore {
                objective,
            })
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
    (quantile(95, 100), quantile(99, 100), finite[finite.len() - 1])
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
    let gravity_error = (job.gravity_error_sum
        / job.gravity_reference_sum.max(f64::MIN_POSITIVE))
    .sqrt() as f32;
    let gradient_error = (job.gradient_error_sum
        / job.gradient_reference_sum.max(f64::MIN_POSITIVE))
    .sqrt() as f32;
    let model_discrimination = (job.discrimination_sum
        / job
            .discrimination_reference_sum
            .max(f64::MIN_POSITIVE))
    .sqrt() as f32;
    let minimum_altitude_m = job.minimum_altitude_m;
    let gradient_information = (job.gradient_information_sum
        / job.total_evaluations.max(1) as f64)
        .sqrt() as f32;
    let planning_objective = model_discrimination * gradient_information
        / (1.0 + gravity_error.max(0.0) + gradient_error.max(0.0));
    let accuracy_penalty = 1.0 + gravity_error.max(0.0) + gradient_error.max(0.0);
    let top_candidates = top_candidate_scores(job, accuracy_penalty);
    let warm_per_candidate =
        job.warm_evaluation_ms / f64::from(job.last_request_candidate_count.max(1));
    let cold_amortization_candidates =
        ((job.density_payload_preparation_ms + job.gpu_preprocessing_ms)
            / warm_per_candidate.max(f64::MIN_POSITIVE))
        .ceil() as u32;
    let raw_gpu_ms =
        job.gpu_preprocessing_ms + job.command_submission_ms + job.gpu_completion_map_ms + job.readback_decode_ms;
    // CPU preparation and GPU kernel timestamps are separate ledgers. Do not
    // infer a basis cost by subtracting a noisy warm request from cold wall time.
    let geometry_basis_build_ms = job.common_geometry_basis_ms + job.method_geometry_basis_ms;
    let density_model_ms = job.density_payload_preparation_ms
        / f64::from(job.density_model_count.max(1));
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
                PlanningExecutionBackend::GpuMmfft
            ) | (ActiveGravityMethod::Fmm, PlanningExecutionBackend::GpuFmm)
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
