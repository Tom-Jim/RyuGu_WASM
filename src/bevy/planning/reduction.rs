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
        job.dispatch_count = job
            .dispatch_count
            .saturating_add(packet.timing.dispatch_count);
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
    let global_start =
        packet.request.candidate_start as usize * batch.samples_per_candidate as usize;
    let mut accumulated_position_error = vec![DVec3::ZERO; packet.request.candidate_count as usize];
    let mut accumulated_velocity_error = vec![DVec3::ZERO; packet.request.candidate_count as usize];
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
                    / matrix_norm_squared(aggregate_gradient)
                        .sqrt()
                        .max(f64::MIN_POSITIVE)) as f32,
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
            job.pericenter_error_m = job.pericenter_error_m.max(
                accumulated_position_error[local_candidate]
                    .dot(radial)
                    .abs() as f32,
            );
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
        return;
    }
    let global_start =
        packet.request.candidate_start as usize * batch.samples_per_candidate as usize;
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
    }
    job.verification_ms += verification_ms;
    job.certified_reduction_ms +=
        (reduction_started.elapsed().as_secs_f64() * 1.0e3 - verification_ms).max(0.0);
}
