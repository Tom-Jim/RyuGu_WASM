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
    mut frequency_domain_workspace: Local<
        crate::gpu::frequency_domain::PlanningFrequencyDomainWorkspace,
    >,
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
        } else if packet.request.request_id == job.request_id && packet_belongs_to_job {
            // Validation must not turn a GPU callback into millions of
            // synchronous CPU source interactions before the browser can paint.
            if packet.readback_valid && (!job.warm_repetition || job.certified_repetition) {
                let started = bevy::platform::time::Instant::now();
                let ready = prepare_planning_references(&batch, &packet, &mut reference_cache);
                job.verification_ms += started.elapsed().as_secs_f64() * 1.0e3;
                if !ready {
                    let fraction = (reference_cache.target_cursor as f64
                        + reference_cache.source_cursor as f64
                            / batch.basis_records.len().max(1) as f64)
                        / reference_cache.target_indices.len().max(1) as f64;
                    job.reference_inflight_fraction =
                        f64::from(packet.request.candidate_count) * fraction.clamp(0.0, 1.0);
                    planning.status = format!(
                        "{} independent f64 verification: target {}/{}, source {}/{} (time-sliced)",
                        job.method.planning_label(),
                        reference_cache.target_cursor + 1,
                        reference_cache.target_indices.len(),
                        reference_cache.source_cursor,
                        batch.basis_records.len()
                    );
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
                        let frequency_domain = planning.results
                            [ActiveGravityMethod::FrequencyDomain.performance_index()]
                        .expect("completed Frequency-domain algorithm curve result");
                        let mmfft = planning.results
                            [ActiveGravityMethod::MmfftCompressed.performance_index()]
                        .expect("completed FFT curve result");
                        let fmm = planning.results[ActiveGravityMethod::Fmm.performance_index()]
                            .expect("completed FMM curve result");
                        let source_count = planning.requested_source_count;
                        let repeat = planning.source_curve_repeat + 1;
                        let outputs = [
                            (frequency_domain, false),
                            (frequency_domain, true),
                            (mmfft, false),
                            (mmfft, true),
                            (fmm, false),
                            (fmm, true),
                        ];
                        let common_samples = frequency_domain.verification_sample_count
                            == mmfft.verification_sample_count
                            && frequency_domain.verification_sample_count
                                == fmm.verification_sample_count;
                        let failure_masks = |profile| {
                            outputs.map(|(result, certified)| {
                                result.accuracy_failure_mask(profile, certified)
                                    | if common_samples { 0 } else { 1 << 8 }
                            })
                        };
                        let strict_failures = failure_masks(PlanningAccuracyProfile::Strict);
                        let order_seed = planning.source_curve_order_seed
                            ^ (planning.source_curve_samples.len() as u64)
                                .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                        planning
                            .source_curve_samples
                            .push(PlanningSourceCurveSample {
                                source_count,
                                density_model_count: job.density_model_count,
                                target_count: job.samples_per_candidate,
                                repeat,
                                order_seed,
                                method_order: job
                                    .method_order
                                    .map(|method| method.performance_index()),
                                times_ms: [
                                    frequency_domain.total_ms,
                                    frequency_domain.certified_estimated_total_ms,
                                    mmfft.total_ms,
                                    mmfft.certified_estimated_total_ms,
                                    fmm.total_ms,
                                    fmm.certified_estimated_total_ms,
                                ],
                                kernel_times_ms: [
                                    frequency_domain.raw_kernels.all_ms,
                                    frequency_domain.checked_kernels.all_ms,
                                    mmfft.raw_kernels.all_ms,
                                    mmfft.checked_kernels.all_ms,
                                    fmm.raw_kernels.all_ms,
                                    fmm.checked_kernels.all_ms,
                                ],
                                evaluation_kernel_times_ms: [
                                    frequency_domain.raw_kernels.evaluation_ms,
                                    frequency_domain.checked_kernels.evaluation_ms,
                                    mmfft.raw_kernels.evaluation_ms,
                                    mmfft.checked_kernels.evaluation_ms,
                                    fmm.raw_kernels.evaluation_ms,
                                    fmm.checked_kernels.evaluation_ms,
                                ],
                                basis_kernel_times_ms: [
                                    frequency_domain.raw_kernels.basis_ms,
                                    mmfft.raw_kernels.basis_ms,
                                    fmm.raw_kernels.basis_ms,
                                ],
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
                                screening_failures: failure_masks(
                                    PlanningAccuracyProfile::Screening,
                                ),
                                gravity_errors: [
                                    frequency_domain.relative_gravity_error,
                                    frequency_domain.certified_relative_gravity_error,
                                    mmfft.relative_gravity_error,
                                    mmfft.certified_relative_gravity_error,
                                    fmm.relative_gravity_error,
                                    fmm.certified_relative_gravity_error,
                                ],
                                gradient_errors: [
                                    frequency_domain.gradient_relative_error,
                                    frequency_domain.certified_gradient_relative_error,
                                    mmfft.gradient_relative_error,
                                    mmfft.certified_gradient_relative_error,
                                    fmm.gradient_relative_error,
                                    fmm.certified_gradient_relative_error,
                                ],
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
        let preparation = channel.preparation.try_lock().ok().and_then(|progress| {
            progress
                .as_ref()
                .filter(|p| p.request_id == job.request_id)
                .cloned()
        });
        if let Some(progress) = &preparation {
            if job.gpu_preparation_submission != progress.completed_submissions {
                job.gpu_preparation_submission = progress.completed_submissions;
                job.awaiting_gpu_seconds = 0.0; // A completed GPU stage is real progress.
                job.awaiting_gpu_last_poll = None;
            }
            if !job.warm_repetition {
                job.gpu_basis_progress = progress.basis_fraction;
            }
        }
        let now = bevy::platform::time::Instant::now();
        if let Some(last) = job.awaiting_gpu_last_poll.replace(now) {
            let elapsed = now.duration_since(last).as_secs_f64();
            if elapsed <= 2.0 {
                job.awaiting_gpu_seconds += elapsed;
            }
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
        planning.status = preparation
            .map(|p| p.status)
            .unwrap_or_else(|| planning_progress_text(&job));
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
            ActiveGravityMethod::MmfftCompressed | ActiveGravityMethod::Fmm => Some(PlanningMethodPayload {
                request_id: payload_key, method: Some(job.method), density_model: job.density_model,
                ..Default::default()
            }),
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
            job.certified_density_payload_preparation_ms += prepared.density_payload_preparation_ms;
        }
        *payload = prepared;
    }
    job.gpu_preparation_submission = 0;
    job.request_id = job.request_id.wrapping_add(1).max(1);
    job.candidate_tile_size = job
        .candidate_tile_size
        .min((8192 / job.samples_per_candidate.max(1)).max(1));
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
