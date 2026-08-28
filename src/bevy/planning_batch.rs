use bevy::math::{DMat3, DQuat, DVec3};
use std::collections::HashMap;

const PLANNING_GPU_WAIT_TIMEOUT_FRAMES: u32 = 1_800;

#[derive(Default)]
pub(crate) struct PlanningReferenceCache {
    fields: HashMap<(u64, u64, u32, [u32; 3]), (DVec3, DMat3)>,
}

pub fn planning_batch_evaluator_system(
    batch: Res<PlanningCandidateBatch>,
    channel: Res<PlanningGpuReadbackChannel>,
    mut request: ResMut<PlanningGpuRequest>,
    mut payload: ResMut<PlanningMethodPayload>,
    mut gpu_result: ResMut<PlanningGpuResult>,
    mut planning: ResMut<PlanningComparisonState>,
    mut mmfft_workspace: Local<crate::gpu::mmfft::PlanningMmfftWorkspace>,
    mut fmm_workspace: Local<crate::gpu::fmm::PlanningFmmWorkspace>,
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
            job.awaiting_gpu_frames = 0;
            planning.status = format!(
                "{} discarded stale GPU packet {}; retrying request {}.",
                job.method.planning_label(),
                packet.request.request_id,
                job.request_id
            );
        } else if !packet_belongs_to_job && job.awaiting_gpu {
            job.awaiting_gpu = false;
            job.awaiting_gpu_frames = 0;
            planning.status = format!(
                "{} discarded mismatched GPU packet; retrying request {}.",
                job.method.planning_label(),
                job.request_id
            );
        } else if packet.request.request_id == job.request_id
            && packet_belongs_to_job
        {
            if job.warm_repetition {
                let repetition_ms = packet.timing.method_preprocess_ms
                    + packet.timing.command_submission_ms
                    + packet.timing.gpu_completion_map_ms;
                if !job.certified_repetition {
                    job.warm_evaluation_ms = repetition_ms;
                    job.raw_gpu_request_count = job.gpu_request_count;
                    job.certified_repetition = true;
                    job.density_model = 0;
                    job.candidate_start = 0;
                    job.awaiting_gpu = false;
                    job.awaiting_gpu_frames = 0;
                    planning.status = format!(
                        "{} raw pass complete; starting the full independently certified BxKxH pass over the common f64 validation strata.",
                        job.method.planning_label()
                    );
                    planning.batch_job = Some(job);
                    return;
                }
                job.certified_warm_evaluation_ms = repetition_ms;
                job.certified_full_pass_ms += repetition_ms;
                reduce_certified_packet(&mut job, &batch, &packet, &mut reference_cache);
                if !advance_certified_tile(&mut job, packet.request.candidate_count) {
                    job.awaiting_gpu = false;
                    job.awaiting_gpu_frames = 0;
                    planning.status = planning_progress_text(&job);
                    planning.batch_job = Some(job);
                    return;
                }
                finish_planning_method(&job, &batch, packet.backend, &mut planning);
                *request = PlanningGpuRequest::default();
                *payload = PlanningMethodPayload::default();
                if job.method_order_index + 1 == job.method_order.len() {
                    if planning.source_curve_active {
                        let eq106 = planning.results[2].expect("completed Eq.106 curve result");
                        let mmfft = planning.results[3].expect("completed FFT curve result");
                        let fmm = planning.results[4].expect("completed FMM curve result");
                        let source_count = planning.requested_source_count;
                        planning.source_curve_samples.push(PlanningSourceCurveSample {
                            source_count,
                            times_ms: [
                                eq106.total_ms,
                                eq106.certified_estimated_total_ms,
                                mmfft.total_ms,
                                mmfft.certified_estimated_total_ms,
                                fmm.total_ms,
                                fmm.certified_estimated_total_ms,
                            ],
                            eligible: [
                                eq106.accuracy_eligible(),
                                eq106.certified_accuracy_eligible(),
                                mmfft.accuracy_eligible(),
                                mmfft.certified_accuracy_eligible(),
                                fmm.accuracy_eligible(),
                                fmm.certified_accuracy_eligible(),
                            ],
                        });
                        planning.source_curve_repeat += 1;
                        if planning.source_curve_repeat >= PLANNING_SOURCE_REPEATS {
                            planning.source_curve_repeat = 0;
                            planning.source_curve_index += 1;
                        }
                        if planning.source_curve_index < PLANNING_SOURCE_COUNTS.len() {
                            planning.requested_source_count =
                                PLANNING_SOURCE_COUNTS[planning.source_curve_index];
                            planning.results = std::array::from_fn(|_| None);
                            planning.run_id = planning.run_id.wrapping_add(1);
                            planning.status = format!(
                                "Quadrature-source curve queued: {} distinct points, fixed 56 density unknowns, repeat {}/{}.",
                                planning.requested_source_count,
                                planning.source_curve_repeat + 1,
                                PLANNING_SOURCE_REPEATS
                            );
                            return;
                        }
                        planning.source_curve_active = false;
                        planning.source_curve_visible = true;
                        planning.run_requested = false;
                        planning.status =
                            "Quadrature-source crossover complete: directly measured medians and P10/P90 ready; density K remained 56.".into();
                        return;
                    }
                    planning.run_requested = false;
                    planning.status = format!(
                        "{} screening complete: all methods used identical nominal-density Volterra-propagated candidates inside the certified 15 m tube. Density rows share those trajectories; model-specific repropagation is not claimed.",
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
            job.awaiting_gpu = false;
            job.awaiting_gpu_frames = 0;
        }
    }
    if job.awaiting_gpu {
        job.awaiting_gpu_frames = job.awaiting_gpu_frames.saturating_add(1);
        if job.awaiting_gpu_frames >= PLANNING_GPU_WAIT_TIMEOUT_FRAMES {
            planning.status = format!(
                "{} stopped after {} frames without a matching GPU readback (request {}).",
                job.method.planning_label(),
                PLANNING_GPU_WAIT_TIMEOUT_FRAMES,
                job.request_id
            );
            planning.run_requested = false;
            job.awaiting_gpu = false;
            *request = PlanningGpuRequest::default();
            *payload = PlanningMethodPayload::default();
            planning.batch_job = None;
            return;
        }
        planning.status = planning_progress_text(&job);
        planning.batch_job = Some(job);
        return;
    }
    if rendering_needs_priority() {
        planning.status = format!(
            "{} planning yielded to rendering at {:.1} FPS / {:.1} ms recent frame.",
            job.method.planning_label(),
            crate::browser_frame_rate().unwrap_or(0.0),
            crate::browser_recent_frame_ms().unwrap_or(0.0),
        );
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
            ActiveGravityMethod::CurvedArcEq106 => {
                crate::gpu::eq106::build_planning_eq106_payload(
                    &batch,
                    job.density_model,
                    payload_key,
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
            ActiveGravityMethod::Fmm => crate::gpu::fmm::build_planning_fmm_payload(
                &batch,
                job.density_model,
                payload_key,
                &mut fmm_workspace,
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
            job.one_time_preparation_ms += prepared.one_time_preparation_ms;
            job.preprocessing_ms += prepared.preparation_ms;
        }
        *payload = prepared;
    }
    job.request_id = job.request_id.wrapping_add(1).max(1);
    let request_candidate_count = job
        .candidate_tile_size
        .min(job.candidate_count - job.candidate_start);
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
        eq106_certified: job.method == ActiveGravityMethod::CurvedArcEq106
            && job.certified_repetition,
        compute_benchmark: job.profile.is_compute_benchmark(),
    };
    job.awaiting_gpu = true;
    job.awaiting_gpu_frames = 0;
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
    if packet.request.candidate_start == 0 {
        job.first_tile_ms += packet.timing.method_preprocess_ms
            + packet.timing.command_submission_ms
            + packet.timing.gpu_completion_map_ms;
    }
    job.verification_sample_count = job
        .verification_sample_count
        .saturating_add(packet.state_indices.len() as u64);
    job.rejected_sample_count = job
        .rejected_sample_count
        .saturating_add(packet.rejected_sample_count);
    if job.first_rejection.is_none() {
        job.first_rejection = packet.first_rejection;
    }
    for (total, count) in job
        .rejection_counts
        .iter_mut()
        .zip(packet.rejection_counts)
    {
        *total = total.saturating_add(count);
    }
    for (maximum, value) in job
        .self_fd_step_maxima
        .iter_mut()
        .zip(packet.self_fd_step_maxima)
    {
        if value.is_finite() {
            *maximum = maximum.max(value);
        }
    }
    job.maximum_gradient_self_fd_relative_error = job
        .maximum_gradient_self_fd_relative_error
        .max(packet.timing.gradient_self_fd_relative_error);
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
        job.preprocessing_ms += packet.timing.method_preprocess_ms;
        job.command_submission_ms += packet.timing.command_submission_ms;
        job.gpu_completion_map_ms += packet.timing.gpu_completion_map_ms;
        job.dispatch_count = job.dispatch_count.saturating_add(packet.timing.dispatch_count);
        job.forward_kernel_evaluations = job
            .forward_kernel_evaluations
            .saturating_add(packet.timing.forward_kernel_evaluations);
        job.spectral_element_count = job
            .spectral_element_count
            .max(packet.timing.spectral_element_count);
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
    job.preprocessing_ms += packet.timing.method_preprocess_ms;
    job.command_submission_ms += packet.timing.command_submission_ms;
    job.gpu_completion_map_ms += packet.timing.gpu_completion_map_ms;
    job.dispatch_count = job.dispatch_count.saturating_add(packet.timing.dispatch_count);
    job.forward_kernel_evaluations = job
        .forward_kernel_evaluations
        .saturating_add(packet.timing.forward_kernel_evaluations);
    job.spectral_element_count = job
        .spectral_element_count
        .max(packet.timing.spectral_element_count);
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
    job.certified_verification_sample_count = job
        .certified_verification_sample_count
        .saturating_add(packet.state_indices.len() as u64);
    job.certified_rejected_sample_count = job
        .certified_rejected_sample_count
        .saturating_add(packet.rejected_sample_count);
    for (maximum, value) in job
        .self_fd_step_maxima
        .iter_mut()
        .zip(packet.self_fd_step_maxima)
    {
        if value.is_finite() {
            *maximum = maximum.max(value);
        }
    }
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
        return;
    }
    let global_start = packet.request.candidate_start as usize
        * batch.samples_per_candidate as usize;
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
        let (reference_field, reference_gradient) = direct_planning_reference_cached(
            state.body_position().as_dvec3(),
            batch,
            packet.request.density_model,
            reference_cache,
        );
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
        job.certified_gravity_samples += 1;
        job.certified_gradient_samples += 1;
    }
}

fn direct_planning_reference_cached(
    target: DVec3,
    batch: &PlanningCandidateBatch,
    density_model: u32,
    cache: &mut PlanningReferenceCache,
) -> (DVec3, DMat3) {
    let key = (
        batch.basis_hash,
        batch.density_model_hash,
        density_model,
        [
            (target.x as f32).to_bits(),
            (target.y as f32).to_bits(),
            (target.z as f32).to_bits(),
        ],
    );
    if let Some(reference) = cache.fields.get(&key) {
        return *reference;
    }
    let row_start = density_model as usize * 56;
    let densities = &batch.density_models[row_start..row_start + 56];
    let reference = crate::cpu::planning::evaluate_planning_reference_field(
        target,
        &batch.basis_records,
        densities,
    )
    .unwrap_or((DVec3::splat(f64::NAN), DMat3::NAN));
    cache.fields.insert(key, reference);
    reference
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
        + packet.timing.gpu_completion_map_ms;
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
    let (minimum, maximum) = if job.method == ActiveGravityMethod::CurvedArcEq106 {
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

fn rendering_needs_priority() -> bool {
    crate::browser_frame_rate().is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS)
        || crate::browser_recent_frame_ms()
            .is_some_and(|milliseconds| milliseconds > PLANNING_MAX_RECENT_FRAME_MS)
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
            if reference <= f64::MIN_POSITIVE || !altitude.is_finite() || altitude <= 0.0 {
                return None;
            }
            let separation =
                (job.candidate_discrimination_sum[candidate_index] / reference).sqrt() as f32;
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
    let total_ms = job.preprocessing_ms
        + job.command_submission_ms
        + job.reduction_ms
        + job.gpu_completion_map_ms
        // Charge the same CPU f64 reference comparison to every backend.
        // Previously this common validation work was measured but omitted
        // from total_ms, making the certified Eq.106 line incomparable.
        + job.verification_ms;
    let warm_per_candidate =
        job.warm_evaluation_ms / f64::from(job.last_request_candidate_count.max(1));
    let cold_amortization_candidates =
        (job.preprocessing_ms / warm_per_candidate.max(f64::MIN_POSITIVE)).ceil() as u32;
    let gpu_build_ms = (job.first_tile_ms
        - job.warm_evaluation_ms * f64::from(job.density_model_count))
    .max(0.0);
    let build_ms = job.preprocessing_ms + gpu_build_ms;
    let certified_estimated_total_ms =
        build_ms + job.certified_full_pass_ms + job.verification_ms;
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
        && job.pericenter_error_m.is_finite()
        && minimum_altitude_m.is_finite()
        && minimum_altitude_m > 0.0
        && model_discrimination.is_finite()
        && planning_objective.is_finite()
        && job.gravity_samples > 0
        && job.gradient_samples > 0
        && job.discrimination_samples > 0
        && matches!(
            (job.method, backend),
            (
                ActiveGravityMethod::CurvedArcEq106,
                PlanningExecutionBackend::GpuEq106
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
        total_ms,
        relative_gravity_error: gravity_error,
        gradient_relative_error: gradient_error,
        certified_relative_gravity_error: certified_gravity_error,
        certified_gradient_relative_error: certified_gradient_error,
        gravity_error_p99,
        gravity_error_max,
        gradient_error_p99,
        gradient_error_max,
        pericenter_error_m: job.pericenter_error_m,
        minimum_altitude_m,
        model_discrimination,
        planning_objective,
        segment_count: if job.method == ActiveGravityMethod::CurvedArcEq106 {
            job.spectral_element_count
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
        preprocessing_ms = job.preprocessing_ms,
        one_time_preparation_ms = job.one_time_preparation_ms,
        command_submission_ms = job.command_submission_ms,
        gpu_completion_map_ms = job.gpu_completion_map_ms,
        reduction_ms = job.reduction_ms,
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
        gradient_self_fd_error = job.maximum_gradient_self_fd_relative_error,
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
    job.awaiting_gpu_frames = 0;
    job.warm_repetition = false;
    job.certified_repetition = false;
    job.gravity_error_sum = 0.0;
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
    job.rejection_counts = [0; 6];
    job.self_fd_step_maxima = [0.0; 5];
    job.first_rejection = None;
    job.maximum_gradient_self_fd_relative_error = 0.0;
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
    job.preprocessing_ms = 0.0;
    job.one_time_preparation_ms = 0.0;
    job.command_submission_ms = 0.0;
    job.reduction_ms = 0.0;
    job.verification_ms = 0.0;
    job.gpu_completion_map_ms = 0.0;
    job.warm_evaluation_ms = 0.0;
    job.certified_warm_evaluation_ms = 0.0;
    job.certified_full_pass_ms = 0.0;
    job.first_tile_ms = 0.0;
    job.dispatch_count = 0;
    job.forward_kernel_evaluations = 0;
    job.spectral_element_count = 0;
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
