#[derive(Default)]
pub(crate) struct DensityWorkerState {
    identity: Option<(u64, u64, ActiveGravityMethod, u64)>,
    density_sum: Vec<f64>,
    realization: usize,
    request_id: u64,
    pending_snapshot: Option<crate::cpp_backend::BackendDensitySnapshot>,
    convex_started: Option<Instant>,
    design_matrix_assembly_ms: f64,
}

pub fn convex_optimization_system(
    mut inversion: ResMut<TrajectoryInversionState>,
    mut performance: ResMut<FrequencyDomainPerformanceMetrics>,
    frequency_domain_sensitivity: Res<FrequencyDomainSensitivityMatrix>,
    density_channel: Res<crate::cpp_backend::BackendDensityChannel>,
    mut worker: Local<DensityWorkerState>,
) {
    let Some(mut job) = inversion.optimizer.take() else {
        if worker.identity.is_some() {
            density_channel.reset();
            *worker = DensityWorkerState::default();
        }
        return;
    };
    let identity = (
        job.capture_id,
        job.source_hash,
        job.method,
        inversion.capture_epoch,
    );
    let matrix_assembly_started = Instant::now();
    let mut design_matrix_assembly_ms = 0.0;
    if worker.identity != Some(identity)
        && job.method == ActiveGravityMethod::FrequencyDomain
    {
        if frequency_domain_sensitivity.capture_id != Some(job.capture_id)
            || frequency_domain_sensitivity.source_hash != job.source_hash
            || frequency_domain_sensitivity.basis_hash != job.basis_sources.hash
            || frequency_domain_sensitivity.configuration_hash
                != crate::gpu::frequency_domain::frequency_domain_sensitivity_configuration_hash()
            || frequency_domain_sensitivity.voxel_count != job.voxels.len()
        {
            inversion.displayed_density = None;
            inversion.error = Some("Frequency-domain algorithm sensitivity cache identity does not match the frozen trajectory.".into());
            return;
        }
        if frequency_domain_sensitivity.columns.len() < job.voxels.len() {
            inversion.optimizer = Some(job);
            return;
        }
        if frequency_domain_sensitivity.columns.len() != job.voxels.len()
            || frequency_domain_sensitivity.sample_count != job.observed_accelerations.len()
            || frequency_domain_sensitivity
                .columns
                .iter()
                .any(|column| column.len() != frequency_domain_sensitivity.sample_count)
        {
            inversion.displayed_density = None;
            inversion.error = Some(format!(
                "Frequency-domain algorithm sensitivity matrix is invalid: {} columns, {} samples; expected {} x {}.",
                frequency_domain_sensitivity.columns.len(),
                frequency_domain_sensitivity.sample_count,
                job.voxels.len(),
                job.observed_accelerations.len(),
            ));
            return;
        }
        job.sensitivities.clear();
        job.sensitivities.reserve(
            frequency_domain_sensitivity.sample_count * frequency_domain_sensitivity.voxel_count,
        );
        for sample in 0..frequency_domain_sensitivity.sample_count {
            for column in &frequency_domain_sensitivity.columns {
                job.sensitivities.push(column[sample]);
            }
        }
        job.data_error_scale = trajectory_data_error(&job).max(1.0e-24);
        job.initial_objective = objective(&job);
        design_matrix_assembly_ms = matrix_assembly_started.elapsed().as_secs_f64() * 1.0e3;
        if !job.timing.matrix_cache_hit {
            job.timing.matrix_build_ms = job.started_at.elapsed().as_secs_f64() * 1.0e3;
        }
        if !job.initial_objective.is_finite() {
            inversion.displayed_density = None;
            inversion.error = Some("The Frequency-domain algorithm sensitivity matrix is not finite.".into());
            return;
        }
    }
    if worker.identity != Some(identity) {
        density_channel.reset();
        *worker = DensityWorkerState {
            identity: Some(identity),
            density_sum: vec![0.0; job.voxels.len()],
            convex_started: Some(Instant::now()),
            design_matrix_assembly_ms,
            ..Default::default()
        };
    }

    if cfg!(not(target_arch = "wasm32")) {
        inversion.displayed_density = None;
        inversion.error = Some("Density solving requires the independent WASM backend".into());
        *worker = DensityWorkerState::default();
        return;
    }

    if let Some(pending) = worker.pending_snapshot {
        let packet = density_channel
            .data
            .lock()
            .expect("backend density result channel poisoned")
            .take();
        let Some(packet) = packet else {
            inversion.optimizer = Some(job);
            return;
        };
        if packet.snapshot != pending || packet.snapshot.epoch != inversion.capture_epoch {
            density_channel.reset();
            *worker = DensityWorkerState::default();
            inversion.optimizer = Some(job);
            return;
        }
        let densities = match packet.result {
            Ok(densities)
                if densities.len() == job.voxels.len()
                    && densities.iter().all(|density| density.is_finite()) =>
            {
                densities
            }
            Ok(_) => {
                inversion.displayed_density = None;
                inversion.error = Some("Density Worker returned an invalid solution".into());
                *worker = DensityWorkerState::default();
                return;
            }
            Err(message) => {
                inversion.displayed_density = None;
                inversion.error = Some(message);
                *worker = DensityWorkerState::default();
                return;
            }
        };
        for (sum, density) in worker.density_sum.iter_mut().zip(densities) {
            *sum += f64::from(density);
        }
        worker.realization += 1;
        worker.pending_snapshot = None;
    }

    if worker.realization < OBSERVATION_NOISE_REALIZATIONS {
        let seed = job.capture_id
            ^ job.source_hash.rotate_left(19)
            ^ (worker.realization as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let observations = noisy_observations(&job.observed_accelerations, seed);
        let data = density_request_json(&job, &observations);
        worker.request_id = worker.request_id.wrapping_add(1).max(1);
        let snapshot = crate::cpp_backend::BackendDensitySnapshot {
            request_id: job.capture_id.rotate_left(27) ^ worker.request_id,
            epoch: inversion.capture_epoch,
        };
        match crate::cpp_backend::request_density(&density_channel, snapshot, &data) {
            Ok(true) => worker.pending_snapshot = Some(snapshot),
            Ok(false) => {}
            Err(message) => {
                inversion.displayed_density = None;
                inversion.error = Some(message);
                *worker = DensityWorkerState::default();
                return;
            }
        }
        inversion.optimizer = Some(job);
        return;
    }

    let convex_solve_ms = worker
        .convex_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1.0e3);
    let verification_started = Instant::now();
    let densities = std::mem::take(&mut worker.density_sum)
        .into_iter()
        .map(|sum| (sum / OBSERVATION_NOISE_REALIZATIONS as f64) as f32)
        .collect::<Vec<_>>();
    job.best_densities.clone_from(&densities);
    job.current_densities.clone_from(&densities);
    for (voxel, density) in job.voxels.iter_mut().zip(&densities) {
        voxel.density = *density;
    }
    let completed_method = job.method;
    let verification_ms = verification_started.elapsed().as_secs_f64() * 1.0e3;
    job.timing.convex_solve_ms = convex_solve_ms;
    job.timing.verification_ms = verification_ms;
    let inversion_time_ms = job.started_at.elapsed().as_secs_f64() * 1.0e3;
    job.timing.total_ms = inversion_time_ms;
    performance.full_inversion_iteration_ms = Some(inversion_time_ms);
    let result = density_result_from_job(&job, &densities, inversion_time_ms);
    if completed_method == ActiveGravityMethod::FrequencyDomain {
        let timing = performance.inversion.get_or_insert_default();
        timing.source_preparation_ms = job.source_preparation_ms;
        timing.design_matrix_assembly_ms += worker.design_matrix_assembly_ms;
        timing.convex_solve_ms = convex_solve_ms;
        timing.verification_ms = verification_ms;
        timing.total_ms = inversion_time_ms;
    }
    let index = completed_method.performance_index();
    inversion.results[index] = Some(result.clone());
    let replaces_best = inversion.best_results[index].as_ref().is_none_or(|best| {
        result.model_fit > best.model_fit
            || (result.model_fit == best.model_fit
                && result.inversion_time_ms < best.inversion_time_ms)
    });
    if replaces_best {
        inversion.best_results[index] = Some(result.clone());
    }
    // The result shown in the central section is the convex QP solution.
    inversion.displayed_density = Some(result);
    density_channel.reset();
    *worker = DensityWorkerState::default();
}

fn noisy_observations(reference: &[Vec3], seed: u64) -> Vec<Vec3> {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rand_distr::StandardNormal;

    let mut rng = StdRng::seed_from_u64(seed);
    reference
        .iter()
        .map(|observation| {
            let sigma = (observation.length() * OBSERVATION_NOISE_FRACTION)
                .max(OBSERVATION_NOISE_FLOOR);
            let noise = Vec3::new(
                rng.sample::<f32, _>(StandardNormal),
                rng.sample::<f32, _>(StandardNormal),
                rng.sample::<f32, _>(StandardNormal),
            ) * sigma;
            *observation + noise
        })
        .collect()
}


fn density_request_json(job: &ConvexOptimizationJob, observations: &[Vec3]) -> String {
    serde_json::json!({
        "voxels": job.voxels.iter().map(|v| serde_json::json!({
            "volume": v.volume, "baseline_density": v.baseline_density, "center": v.center.to_array()
        })).collect::<Vec<_>>(),
        "observations": observations.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
        "observed_accelerations": job.observed_accelerations.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
        "sensitivities": job.sensitivities.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
        "data_error_scale": job.data_error_scale,
        "neighbours": job.neighbours, "voxel_size": job.voxel_size,
        "homogeneous": job.method == ActiveGravityMethod::HomogeneousWerner
    })
    .to_string()
}
