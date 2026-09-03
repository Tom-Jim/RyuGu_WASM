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

fn solve_density_qp(
    job: &ConvexOptimizationJob,
    observations: &[Vec3],
) -> Result<Vec<f32>, String> {
    use clarabel::{algebra::CscMatrix, solver::*};
    let n = job.voxels.len();
    if n == 0
        || observations.is_empty()
        || observations.len() != job.observed_accelerations.len()
        || job.sensitivities.len() != n * observations.len()
    {
        return Err("Clarabel QP dimensions do not match the frozen observations.".into());
    }
    let scale = job.data_error_scale.max(1.0e-24);
    let mut h = vec![vec![0.0_f64; n]; n];
    let mut g = vec![0.0_f64; n];
    for (observation, observed) in observations.iter().enumerate() {
        let base = observation * n;
        let sigma = (job.observed_accelerations[observation].length()
            * OBSERVATION_NOISE_FRACTION)
            .max(OBSERVATION_NOISE_FLOOR);
        let weight = 1.0
            / (observations.len().max(1) as f64
                * f64::from(sigma * sigma)
                * scale);
        for i in 0..n {
            let si = job.sensitivities[base + i];
            g[i] += si.dot(*observed) as f64 * weight;
            for (j, sj) in job.sensitivities[base..base + i + 1].iter().enumerate() {
                h[j][i] += si.dot(*sj) as f64 * weight;
            }
        }
    }
    let mean_density = (RYUGU_MASS as f64
        / job
            .voxels
            .iter()
            .map(|voxel| voxel.volume as f64)
            .sum::<f64>())
    .max(f64::MIN_POSITIVE);
    for i in 0..n {
        let baseline = job.voxels[i].baseline_density.max(f32::MIN_POSITIVE) as f64;
        let prior = PRIOR_WEIGHT / (n as f64 * baseline * baseline);
        h[i][i] += prior + 1.0e-12;
        g[i] += prior * baseline;
    }
    for &(left, right) in &job.neighbours {
        add_pair_penalty(
            &mut h,
            left,
            right,
            SMOOTHNESS_WEIGHT
                / (job.neighbours.len().max(1) as f64 * mean_density * mean_density),
        );
    }
    let radial_pairs = radial_density_pairs(job);
    for &(left, right) in &radial_pairs {
        add_pair_penalty(
            &mut h,
            left,
            right,
            RADIAL_SYMMETRY_WEIGHT
                / (radial_pairs.len().max(1) as f64 * mean_density * mean_density),
        );
    }
    let mut p_rows = Vec::new();
    let mut p_cols = Vec::new();
    let mut p_vals = Vec::new();
    for (row, h_row) in h.iter().enumerate() {
        for (col, value) in h_row.iter().enumerate().skip(row) {
            if value.abs() > 0.0 {
                p_rows.push(row);
                p_cols.push(col);
                p_vals.push(2.0 * value);
            }
        }
    }
    let p = CscMatrix::new_from_triplets(n, n, p_rows, p_cols, p_vals);
    let homogeneous_equalities = if job.method == ActiveGravityMethod::HomogeneousWerner {
        n.saturating_sub(1)
    } else {
        0
    };
    let rows = 2 * n + 1 + homogeneous_equalities;
    let nonzeros = 3 * n + 2 * homogeneous_equalities;
    let mut a_rows = Vec::with_capacity(nonzeros);
    let mut a_cols = Vec::with_capacity(nonzeros);
    let mut a_vals = Vec::with_capacity(nonzeros);
    let mut b = vec![0.0_f64; rows];
    for col in 0..n {
        a_rows.extend([col, n + col, 2 * n]);
        a_cols.extend([col, col, col]);
        a_vals.extend([1.0, -1.0, job.voxels[col].volume as f64]);
        b[col] = 8.0 * job.voxels[col].baseline_density as f64;
        b[n + col] = -0.02 * job.voxels[col].baseline_density as f64;
    }
    b[2 * n] = RYUGU_MASS as f64;
    for index in 1..=homogeneous_equalities {
        let row = 2 * n + index;
        a_rows.extend([row, row]);
        a_cols.extend([0, index]);
        a_vals.extend([-1.0, 1.0]);
    }
    let a = CscMatrix::new_from_triplets(rows, n, a_rows, a_cols, a_vals);
    let cones = [
        NonnegativeConeT(2 * n),
        ZeroConeT(1 + homogeneous_equalities),
    ];
    let settings = DefaultSettings {
        verbose: false,
        max_iter: 500,
        ..Default::default()
    };
    let q = g.iter().map(|value| -2.0 * value).collect::<Vec<_>>();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
        .map_err(|error| format!("Clarabel rejected the 56x56 QP: {error}"))?;
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(format!(
            "Clarabel terminated with status {:?}.",
            solver.solution.status
        ));
    }
    let densities = solver
        .solution
        .x
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    if densities.iter().any(|density| !density.is_finite()) {
        return Err("Clarabel returned a non-finite density solution.".into());
    }
    info!(
        "[inversion] Clarabel {:?}: {} variables, {} observations",
        solver.solution.status,
        densities.len(),
        observations.len(),
    );
    Ok(densities)
}

fn add_pair_penalty(h: &mut [Vec<f64>], left: usize, right: usize, weight: f64) {
    h[left][left] += weight;
    h[right][right] += weight;
    let (row, col) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    h[row][col] -= weight;
}

fn radial_density_pairs(job: &ConvexOptimizationJob) -> Vec<(usize, usize)> {
    job.voxels
        .iter()
        .enumerate()
        .flat_map(|(left, a)| {
            job.voxels
                .iter()
                .enumerate()
                .skip(left + 1)
                .filter_map(move |(right, b)| {
                    ((a.center.length() - b.center.length()).abs() <= job.voxel_size * 0.45)
                        .then_some((left, right))
                })
        })
        .collect()
}
