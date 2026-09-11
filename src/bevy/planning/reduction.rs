const PLANNING_REDUCTION_SLICE_SECS: f64 = 0.008;

fn begin_planning_reduction(
    cache: &mut PlanningReferenceCache,
    request_id: u64,
    candidate_count: usize,
    certified: bool,
) {
    let key = request_id ^ if certified { 1 << 63 } else { 0 };
    if cache.reduction_request_id != Some(key) {
        cache.reduction_request_id = Some(key);
        cache.reduction_index = 0;
        cache.reduction_header_done = false;
        cache.reduction_position_error = vec![DVec3::ZERO; candidate_count];
        cache.reduction_velocity_error = vec![DVec3::ZERO; candidate_count];
        cache.reduction_previous_time = vec![None; candidate_count];
    }
}

fn finish_planning_reduction(cache: &mut PlanningReferenceCache) {
    cache.reduction_request_id = None;
    cache.reduction_index = 0;
    cache.reduction_header_done = false;
    cache.reduction_position_error.clear();
    cache.reduction_velocity_error.clear();
    cache.reduction_previous_time.clear();
}

fn reduce_planning_packet(
    job: &mut PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    reference_cache: &mut PlanningReferenceCache,
) -> bool {
    let reduction_started = bevy::platform::time::Instant::now();
    let mut verification_ms = 0.0;
    begin_planning_reduction(
        reference_cache,
        packet.request.request_id,
        packet.request.candidate_count as usize,
        false,
    );
    if !reference_cache.reduction_header_done {
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
            job.dispatch_count = job
                .dispatch_count
                .saturating_add(packet.timing.dispatch_count);
            job.forward_kernel_evaluations = job
                .forward_kernel_evaluations
                .saturating_add(packet.timing.forward_kernel_evaluations);
            job.trajectory_block_count = job
                .trajectory_block_count
                .max(packet.timing.trajectory_block_count);
            finish_planning_reduction(reference_cache);
            return true;
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
        reference_cache.reduction_header_done = true;
    }
    let global_start =
        packet.request.candidate_start as usize * batch.samples_per_candidate as usize;
    let start_index = reference_cache.reduction_index;
    for (verification_index, local_target) in packet
        .state_indices
        .iter()
        .copied()
        .enumerate()
        .skip(start_index)
    {
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
                    / matrix_norm_squared(aggregate_gradient)
                        .sqrt()
                        .max(f64::MIN_POSITIVE)) as f32,
            );
            if reduction_started.elapsed().as_secs_f64() >= PLANNING_REDUCTION_SLICE_SECS {
                reference_cache.reduction_index = verification_index + 1;
                job.verification_ms += verification_ms;
                return false;
            }
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
        if let Some(previous_time) = reference_cache.reduction_previous_time[local_candidate] {
            let delta_time = current_time - previous_time;
            let rotation = DQuat::from_xyzw(
                f64::from(state.body_rotation[0]),
                f64::from(state.body_rotation[1]),
                f64::from(state.body_rotation[2]),
                f64::from(state.body_rotation[3]),
            );
            let acceleration_error = rotation * (method_field - reference_field);
            reference_cache.reduction_position_error[local_candidate] +=
                reference_cache.reduction_velocity_error[local_candidate] * delta_time
                    + 0.5 * acceleration_error * delta_time * delta_time;
            reference_cache.reduction_velocity_error[local_candidate] +=
                acceleration_error * delta_time;
        }
        reference_cache.reduction_previous_time[local_candidate] = Some(current_time);
        if verify_pericenter {
            let rotation = DQuat::from_xyzw(
                f64::from(state.body_rotation[0]),
                f64::from(state.body_rotation[1]),
                f64::from(state.body_rotation[2]),
                f64::from(state.body_rotation[3]),
            );
            let radial = (rotation * state.body_position().as_dvec3()).normalize_or_zero();
            job.pericenter_error_m = job.pericenter_error_m.max(
                reference_cache.reduction_position_error[local_candidate]
                    .dot(radial)
                    .abs() as f32,
            );
        }
        if reduction_started.elapsed().as_secs_f64() >= PLANNING_REDUCTION_SLICE_SECS {
            reference_cache.reduction_index = verification_index + 1;
            job.verification_ms += verification_ms;
            return false;
        }
    }
    job.gpu_preprocessing_ms += packet.timing.method_preprocess_ms;
    job.command_submission_ms += packet.timing.command_submission_ms;
    job.gpu_completion_map_ms += packet.timing.gpu_completion_map_ms;
    job.readback_decode_ms += packet.timing.readback_decode_ms;
    job.dispatch_count = job
        .dispatch_count
        .saturating_add(packet.timing.dispatch_count);
    job.forward_kernel_evaluations = job
        .forward_kernel_evaluations
        .saturating_add(packet.timing.forward_kernel_evaluations);
    job.trajectory_block_count = job
        .trajectory_block_count
        .max(packet.timing.trajectory_block_count);
    let total_reduction_ms = reduction_started.elapsed().as_secs_f64() * 1.0e3;
    job.verification_ms += verification_ms;
    job.reduction_ms += (total_reduction_ms - verification_ms).max(0.0);
    finish_planning_reduction(reference_cache);
    true
}

fn reduce_certified_packet(
    job: &mut PlanningBatchJob,
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    reference_cache: &mut PlanningReferenceCache,
) -> bool {
    let reduction_started = bevy::platform::time::Instant::now();
    let mut verification_ms = 0.0;
    begin_planning_reduction(
        reference_cache,
        packet.request.request_id,
        packet.request.candidate_count as usize,
        true,
    );
    let frequency_domain = packet.request.method == Some(ActiveGravityMethod::FrequencyDomain);
    if !reference_cache.reduction_header_done {
        job.certified_kernels.record(packet.timing);
        job.certified_verification_sample_count = job
            .certified_verification_sample_count
            .saturating_add(packet.state_indices.len() as u64);
        job.certified_rejected_sample_count = job
            .certified_rejected_sample_count
            .saturating_add(packet.rejected_sample_count);
        for (local_candidate, metric) in packet.candidate_metrics.iter().enumerate() {
            if (metric[0] < 0.0 || metric.iter().any(|value| !value.is_finite()))
                && let Some(valid) = job
                    .certified_candidate_valid
                    .get_mut(packet.request.candidate_start as usize + local_candidate)
            {
                *valid = false;
            }
        }
        if !packet.readback_valid || packet.rows.len() != packet.state_indices.len() * 4 {
            job.certified_gravity_error_sum = f64::NAN;
            job.certified_gradient_error_sum = f64::NAN;
            job.certified_reduction_ms += reduction_started.elapsed().as_secs_f64() * 1.0e3;
            finish_planning_reduction(reference_cache);
            return true;
        }
        reference_cache.reduction_header_done = true;
    }
    let global_start =
        packet.request.candidate_start as usize * batch.samples_per_candidate as usize;
    let start_index = reference_cache.reduction_index;
    if frequency_domain {
        for (verification_index, local_target) in packet
            .state_indices
            .iter()
            .copied()
            .enumerate()
            .skip(start_index)
        {
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
            let (reference_field, reference_gradient) = frequency_domain_reference_integral(
                batch,
                packet.request.candidate_start as usize + local_candidate,
                observation_index,
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
            job.certified_pointwise_gravity_errors.push(
                ((method_field - reference_field).length()
                    / reference_field.length().max(f64::MIN_POSITIVE)) as f32,
            );
            job.certified_pointwise_gradient_errors.push(
                (matrix_norm_squared(method_gradient - reference_gradient).sqrt()
                    / matrix_norm_squared(reference_gradient)
                        .sqrt()
                        .max(f64::MIN_POSITIVE)) as f32,
            );
            job.certified_gravity_samples += 1;
            job.certified_gradient_samples += 1;
            if reduction_started.elapsed().as_secs_f64() >= PLANNING_REDUCTION_SLICE_SECS {
                reference_cache.reduction_index = verification_index + 1;
                job.verification_ms += verification_ms;
                return false;
            }
        }
        job.verification_ms += verification_ms;
        job.certified_reduction_ms +=
            (reduction_started.elapsed().as_secs_f64() * 1.0e3 - verification_ms).max(0.0);
        finish_planning_reduction(reference_cache);
        return true;
    }
    for (verification_index, local_target) in packet
        .state_indices
        .iter()
        .copied()
        .enumerate()
        .skip(start_index)
    {
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
        job.certified_pointwise_gravity_errors.push(
            ((method_field - reference_field).length()
                / reference_field.length().max(f64::MIN_POSITIVE)) as f32,
        );
        job.certified_pointwise_gradient_errors.push(
            (matrix_norm_squared(method_gradient - reference_gradient).sqrt()
                / matrix_norm_squared(reference_gradient)
                    .sqrt()
                    .max(f64::MIN_POSITIVE)) as f32,
        );
        job.certified_gravity_samples += 1;
        job.certified_gradient_samples += 1;
        if reduction_started.elapsed().as_secs_f64() >= PLANNING_REDUCTION_SLICE_SECS {
            reference_cache.reduction_index = verification_index + 1;
            job.verification_ms += verification_ms;
            return false;
        }
    }
    job.verification_ms += verification_ms;
    job.certified_reduction_ms +=
        (reduction_started.elapsed().as_secs_f64() * 1.0e3 - verification_ms).max(0.0);
    finish_planning_reduction(reference_cache);
    true
}
