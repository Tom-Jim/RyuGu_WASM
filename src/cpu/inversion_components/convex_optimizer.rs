pub fn convex_optimization_system(
    mut inversion: ResMut<TrajectoryInversionState>,
    mut performance: ResMut<FrequencyDomainPerformanceMetrics>,
    frequency_domain_sensitivity: Res<FrequencyDomainSensitivityMatrix>,
) {
    let Some(mut job) = inversion.optimizer.take() else {
        return;
    };
    let matrix_assembly_started = Instant::now();
    let mut design_matrix_assembly_ms = 0.0;
    if job.method == ActiveGravityMethod::FrequencyDomain {
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
    let convex_started = Instant::now();
    let mut density_sum = vec![0.0_f64; job.voxels.len()];
    for realization in 0..OBSERVATION_NOISE_REALIZATIONS {
        let seed = job.capture_id
            ^ job.source_hash.rotate_left(19)
            ^ (realization as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let observations = noisy_observations(&job.observed_accelerations, seed);
        let densities = match solve_density_qp(&job, &observations) {
            Ok(densities) => densities,
            Err(error) => {
                inversion.displayed_density = None;
                inversion.error = Some(error);
                return;
            }
        };
        for (sum, density) in density_sum.iter_mut().zip(densities) {
            *sum += f64::from(density);
        }
    }
    let convex_solve_ms = convex_started.elapsed().as_secs_f64() * 1.0e3;
    let verification_started = Instant::now();
    let densities = density_sum
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
        timing.design_matrix_assembly_ms += design_matrix_assembly_ms;
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


fn solve_density_qp(job: &ConvexOptimizationJob, observations: &[Vec3]) -> Result<Vec<f32>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        let data = serde_json::json!({
            "voxels": job.voxels.iter().map(|v| serde_json::json!({
                "volume": v.volume, "baseline_density": v.baseline_density, "center": v.center.to_array()
            })).collect::<Vec<_>>(),
            "observations": observations.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
            "observed_accelerations": job.observed_accelerations.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
            "sensitivities": job.sensitivities.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
            "data_error_scale": job.data_error_scale,
            "neighbours": job.neighbours, "voxel_size": job.voxel_size,
            "homogeneous": job.method == ActiveGravityMethod::HomogeneousWerner
        });
        crate::cpp_backend::solve_density(&data.to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    { let _ = (job, observations); Err("Density solving requires the independent WASM backend".into()) }
}
