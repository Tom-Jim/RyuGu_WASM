pub fn planning_batch_evaluator_system(
    batch: Res<PlanningCandidateBatch>,
    mut request: ResMut<PlanningGpuRequest>,
    mut payload: ResMut<PlanningMethodPayload>,
    mut gpu_result: ResMut<PlanningGpuResult>,
    mut planning: ResMut<PlanningComparisonState>,
    mut mmfft_workspace: Local<crate::gpu::mmfft::PlanningMmfftWorkspace>,
    mut dispatch_cooldown: Local<u8>,
) {
    let Some(mut job) = planning.batch_job.take() else {
        return;
    };
    if batch.batch_id == 0 || batch.batch_id != job.batch_id {
        planning.status = "Planning is waiting for the frozen candidate buffers.".into();
        planning.batch_job = Some(job);
        return;
    }
    if let Some(packet) = gpu_result.0.take() {
        if packet.request.request_id == job.request_id
            && packet.request.batch_id == job.batch_id
            && packet.request.method == Some(job.method)
            && packet.request.warm_repetition == job.warm_repetition
        {
            if job.warm_repetition {
                job.warm_evaluation_ms = packet.timing.method_preprocess_ms
                    + packet.timing.encode_ms
                    + packet.timing.readback_ms;
                finish_planning_method(&job, &batch, packet.backend, &mut planning);
                *request = PlanningGpuRequest::default();
                *payload = PlanningMethodPayload::default();
                if job.method == ActiveGravityMethod::Fmm {
                    planning.run_requested = false;
                    planning.status = format!(
                        "{} GPU batch complete: all methods used the identical frozen 15 m tube and density matrix.",
                        job.profile.label()
                    );
                    return;
                }
                advance_planning_method(&mut job);
            } else {
                reduce_planning_packet(&mut job, &batch, &packet);
                advance_planning_tile(&mut job);
            }
            job.awaiting_gpu = false;
            *dispatch_cooldown = PLANNING_DISPATCH_COOLDOWN_FRAMES;
        }
    }
    if job.awaiting_gpu {
        planning.status = planning_progress_text(&job);
        planning.batch_job = Some(job);
        return;
    }
    if *dispatch_cooldown > 0 {
        *dispatch_cooldown -= 1;
        planning.status = planning_progress_text(&job);
        planning.batch_job = Some(job);
        return;
    }
    if crate::browser_frame_rate().is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS) {
        planning.status = format!(
            "{} stress yielded to rendering at {:.0} FPS.",
            job.method.as_str(),
            crate::browser_frame_rate().unwrap_or(0.0)
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
            ),
            _ => None,
        };
        let Some(prepared) = prepared else {
            planning.status = format!(
                "{} planning stopped: payload preparation failed for density model {}.",
                job.method.as_str(),
                job.density_model
            );
            planning.run_requested = false;
            *request = PlanningGpuRequest::default();
            *payload = PlanningMethodPayload::default();
            return;
        };
        if !job.warm_repetition {
            job.preprocessing_ms += prepared.preparation_ms;
        }
        *payload = prepared;
    }
    job.request_id = job.request_id.wrapping_add(1).max(1);
    *request = PlanningGpuRequest {
        request_id: job.request_id,
        batch_id: job.batch_id,
        method: Some(job.method),
        density_model: job.density_model,
        candidate_start: job.candidate_start,
        candidate_count: PLANNING_GPU_TILE_CANDIDATES
            .min(job.candidate_count - job.candidate_start),
        warm_repetition: job.warm_repetition,
    };
    job.awaiting_gpu = true;
    planning.status = planning_progress_text(&job);
    planning.batch_job = Some(job);
}

fn reduce_planning_packet(
    job: &mut PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
) {
    let reduction_started = bevy::platform::time::Instant::now();
    let mut verification_ms = 0.0;
    if !packet.readback_valid
        || packet.rows.len() != packet.state_indices.len() * 4
        || packet.candidate_metrics.len() != packet.request.candidate_count as usize
        || packet
            .rows
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
        || packet
            .candidate_metrics
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
        || packet.candidate_metrics.iter().any(|row| row[0] < 0.0)
    {
        job.gravity_error_sum = f64::NAN;
        job.gradient_error_sum = f64::NAN;
        job.preprocessing_ms += packet.timing.method_preprocess_ms;
        job.evaluation_ms += packet.timing.encode_ms;
        job.readback_ms += packet.timing.readback_ms;
        job.dispatch_count = job.dispatch_count.saturating_add(packet.timing.dispatch_count);
        job.forward_kernel_evaluations = job
            .forward_kernel_evaluations
            .saturating_add(packet.timing.forward_kernel_evaluations);
        job.spectral_element_count = job
            .spectral_element_count
            .max(packet.timing.spectral_element_count);
        return;
    }
    for metric in &packet.candidate_metrics {
        job.minimum_altitude_m = job.minimum_altitude_m.min(metric[2]);
        job.gradient_information_sum += f64::from(metric[3]);
        if packet.request.density_model > 0 {
            job.discrimination_sum += f64::from(metric[0]);
            job.discrimination_reference_sum += f64::from(metric[1]);
            job.discrimination_samples += 1;
        }
    }
    let global_start = packet.request.candidate_start as usize
        * batch.samples_per_candidate as usize;
    let mut accumulated_position_error =
        vec![Vec3::ZERO; packet.request.candidate_count as usize];
    let mut accumulated_velocity_error =
        vec![Vec3::ZERO; packet.request.candidate_count as usize];
    let mut previous_verified_time = vec![None; packet.request.candidate_count as usize];
    for (verification_index, local_target) in packet.state_indices.iter().copied().enumerate() {
        let local = local_target as usize;
        let state = batch.states[global_start + local];
        let sample = state.identity[1];
        let verify_pericenter = sample.abs_diff(batch.samples_per_candidate / 2) <= 1;
        let method_field = Vec3::new(
            packet.rows[verification_index * 4][0],
            packet.rows[verification_index * 4][1],
            packet.rows[verification_index * 4][2],
        );
        let method_gradient = Mat3::from_cols(
            Vec3::from_array(
                packet.rows[verification_index * 4 + 1][..3]
                    .try_into()
                    .unwrap(),
            ),
            Vec3::from_array(
                packet.rows[verification_index * 4 + 2][..3]
                    .try_into()
                    .unwrap(),
            ),
            Vec3::from_array(
                packet.rows[verification_index * 4 + 3][..3]
                    .try_into()
                    .unwrap(),
            ),
        );
        let verification_started = bevy::platform::time::Instant::now();
        let (reference_field, reference_gradient) = direct_planning_reference(
            state.body_position(),
            batch,
            packet.request.density_model,
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
        job.gravity_error_sum += f64::from((method_field - reference_field).length_squared());
        job.gravity_reference_sum += f64::from(reference_field.length_squared());
        job.gravity_samples += 1;
        job.gradient_error_sum +=
            f64::from(matrix_norm_squared(method_gradient - reference_gradient));
        job.gradient_reference_sum += f64::from(matrix_norm_squared(reference_gradient));
        job.gradient_samples += 1;
        let local_candidate = local / batch.samples_per_candidate as usize;
        let current_time = state.position_time[3];
        if let Some(previous_time) = previous_verified_time[local_candidate] {
            let delta_time = current_time - previous_time;
            let acceleration_error = method_field - reference_field;
            accumulated_position_error[local_candidate] +=
                accumulated_velocity_error[local_candidate] * delta_time
                    + 0.5 * acceleration_error * delta_time * delta_time;
            accumulated_velocity_error[local_candidate] += acceleration_error * delta_time;
        }
        previous_verified_time[local_candidate] = Some(current_time);
        if verify_pericenter {
            let radial = state.body_position().normalize_or_zero();
            job.pericenter_error_m = job
                .pericenter_error_m
                .max(accumulated_position_error[local_candidate].dot(radial).abs());
        }
    }
    job.preprocessing_ms += packet.timing.method_preprocess_ms;
    job.evaluation_ms += packet.timing.encode_ms;
    job.readback_ms += packet.timing.readback_ms;
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

fn direct_planning_reference(
    target: Vec3,
    batch: &PlanningCandidateBatch,
    density_model: u32,
) -> (Vec3, Mat3) {
    let row_start = density_model as usize * 56;
    let densities = &batch.density_models[row_start..row_start + 56];
    let mut acceleration = Vec3::ZERO;
    let mut gradient = Mat3::ZERO;
    for source in batch.basis_records.iter() {
        let position = Vec3::from_array(source.position_volume[..3].try_into().unwrap());
        let displacement = position - target;
        let radius2 = displacement.length_squared().max(1.0e-8);
        let inverse_radius = radius2.sqrt().recip();
        let inverse_radius3 = inverse_radius / radius2;
        let mass = source.position_volume[3] * densities[source.voxel_index as usize];
        acceleration += G * mass * displacement * inverse_radius3;
        let outer = Mat3::from_cols(
            displacement * displacement.x,
            displacement * displacement.y,
            displacement * displacement.z,
        );
        gradient += G
            * mass
            * (-Mat3::IDENTITY * inverse_radius3
                + outer * (3.0 * inverse_radius3 / radius2));
    }
    (acceleration, gradient)
}

fn matrix_norm_squared(matrix: Mat3) -> f32 {
    matrix.x_axis.length_squared() + matrix.y_axis.length_squared() + matrix.z_axis.length_squared()
}

fn advance_planning_tile(job: &mut PlanningBatchJob) {
    job.candidate_start += PLANNING_GPU_TILE_CANDIDATES
        .min(job.candidate_count - job.candidate_start);
    if job.candidate_start < job.candidate_count {
        return;
    }
    job.candidate_start = 0;
    job.density_model += 1;
    if job.density_model < job.density_model_count {
        return;
    }
    job.density_model = job.density_model_count - 1;
    job.candidate_start = job.candidate_count - PLANNING_GPU_TILE_CANDIDATES;
    job.warm_repetition = true;
}

fn finish_planning_method(
    job: &PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    backend: PlanningExecutionBackend,
    planning: &mut PlanningComparisonState,
) {
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
    let total_ms = job.preprocessing_ms + job.evaluation_ms + job.reduction_ms + job.readback_ms;
    let warm_per_candidate = job.warm_evaluation_ms / PLANNING_GPU_TILE_CANDIDATES as f64;
    let cold_amortization_candidates =
        (job.preprocessing_ms / warm_per_candidate.max(f64::MIN_POSITIVE)).ceil() as u32;
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
        common_preparation_ms: job.common_preparation_ms,
        preprocessing_ms: job.preprocessing_ms,
        evaluation_ms: job.evaluation_ms,
        reduction_ms: job.reduction_ms,
        verification_ms: job.verification_ms,
        readback_ms: job.readback_ms,
        warm_evaluation_ms: job.warm_evaluation_ms,
        total_ms,
        relative_gravity_error: gravity_error,
        gradient_relative_error: gradient_error,
        pericenter_error_m: job.pericenter_error_m,
        minimum_altitude_m,
        model_discrimination,
        planning_objective,
        segment_count: if job.method == ActiveGravityMethod::CurvedArcEq106 {
            job.spectral_element_count
        } else {
            0
        },
        cold_amortization_candidates,
        dispatch_count: job.dispatch_count,
        forward_kernel_evaluations: job.forward_kernel_evaluations,
        density_combinations: job.total_evaluations,
    });
}

fn advance_planning_method(job: &mut PlanningBatchJob) {
    job.method = match job.method {
        ActiveGravityMethod::CurvedArcEq106 => ActiveGravityMethod::MmfftCompressed,
        ActiveGravityMethod::MmfftCompressed => ActiveGravityMethod::Fmm,
        _ => ActiveGravityMethod::Fmm,
    };
    job.density_model = 0;
    job.candidate_start = 0;
    job.awaiting_gpu = false;
    job.warm_repetition = false;
    job.gravity_error_sum = 0.0;
    job.gravity_reference_sum = 0.0;
    job.gravity_samples = 0;
    job.gradient_error_sum = 0.0;
    job.gradient_reference_sum = 0.0;
    job.gradient_samples = 0;
    job.pericenter_error_m = 0.0;
    job.minimum_altitude_m = f32::INFINITY;
    job.discrimination_sum = 0.0;
    job.discrimination_reference_sum = 0.0;
    job.discrimination_samples = 0;
    job.gradient_information_sum = 0.0;
    job.preprocessing_ms = 0.0;
    job.evaluation_ms = 0.0;
    job.reduction_ms = 0.0;
    job.verification_ms = 0.0;
    job.readback_ms = 0.0;
    job.warm_evaluation_ms = 0.0;
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
        "{} {}: {} / {} density combinations, {} model {}, dispatches {}.",
        job.profile.label(),
        job.method.as_str(),
        completed.min(job.total_evaluations),
        job.total_evaluations,
        phase,
        job.density_model + 1,
        job.dispatch_count,
    )
}
