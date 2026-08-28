pub fn convex_optimization_system(
    mut inversion: ResMut<TrajectoryInversionState>,
    mut performance: ResMut<Eq106PerformanceMetrics>,
    eq106_sensitivity: Res<Eq106SensitivityMatrix>,
) {
    let Some(mut job) = inversion.optimizer.take() else {
        return;
    };
    let matrix_assembly_started = Instant::now();
    let mut design_matrix_assembly_ms = 0.0;
    if job.method == ActiveGravityMethod::CurvedArcEq106 {
        if eq106_sensitivity.capture_id != Some(job.capture_id)
            || eq106_sensitivity.source_hash != job.source_hash
            || eq106_sensitivity.basis_hash != job.basis_sources.hash
            || eq106_sensitivity.configuration_hash
                != crate::gpu::eq106::eq106_sensitivity_configuration_hash()
            || eq106_sensitivity.voxel_count != job.voxels.len()
        {
            inversion.displayed_density = None;
            inversion.error = Some("Eq.106 sensitivity cache identity does not match the frozen trajectory.".into());
            return;
        }
        if eq106_sensitivity.columns.len() < job.voxels.len() {
            inversion.optimizer = Some(job);
            return;
        }
        if eq106_sensitivity.columns.len() != job.voxels.len()
            || eq106_sensitivity.sample_count != job.observed_accelerations.len()
            || eq106_sensitivity
                .columns
                .iter()
                .any(|column| column.len() != eq106_sensitivity.sample_count)
        {
            inversion.displayed_density = None;
            inversion.error = Some(format!(
                "Eq.106 sensitivity matrix is invalid: {} columns, {} samples; expected {} x {}.",
                eq106_sensitivity.columns.len(),
                eq106_sensitivity.sample_count,
                job.voxels.len(),
                job.observed_accelerations.len(),
            ));
            return;
        }
        job.sensitivities.clear();
        job.sensitivities.reserve(
            eq106_sensitivity.sample_count * eq106_sensitivity.voxel_count,
        );
        for sample in 0..eq106_sensitivity.sample_count {
            for column in &eq106_sensitivity.columns {
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
            inversion.error = Some("The Eq.106 sensitivity matrix is not finite.".into());
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
    let best_objective = objective_for_densities(&job, &densities);
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
    let result = density_result_from_job(&job, &densities, best_objective, inversion_time_ms);
    if completed_method == ActiveGravityMethod::CurvedArcEq106 {
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

fn objective_for_densities(job: &ConvexOptimizationJob, densities: &[f32]) -> f64 {
    objective_for_density_slice(job, densities)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inversion_problem_id_depends_only_on_frozen_data_and_source() {
        let first = inversion_problem_id(0x1234, 0x5678);
        assert_eq!(first, inversion_problem_id(0x1234, 0x5678));
        assert_ne!(first, inversion_problem_id(0x1235, 0x5678));
        assert_ne!(first, inversion_problem_id(0x1234, 0x5679));
    }

    #[test]
    fn method_independent_reference_matrix_is_shared_before_backend_replacement() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes };
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::new(1000.0, 1200.0, 100.0),
                velocity: Vec3::Y,
                simulation_time_seconds: 0.0,
                baseline_acceleration: Vec3::new(-1.0e-5, -2.0e-5, 0.0),
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: Vec3::new(1000.0, 1201.0, 100.0),
                velocity: Vec3::Y,
                simulation_time_seconds: 1.0,
                baseline_acceleration: Vec3::new(-1.0e-5, -2.0e-5, 0.0),
                body_rotation: Quat::IDENTITY,
            },
        ];
        let sensitivities_for = |method| {
            let (voxels, _) = build_density_voxels(&source, method).unwrap();
            build_observations_and_sensitivities(&knots, &voxels, &source)
                .unwrap()
                .1
        };
        let direct = sensitivities_for(ActiveGravityMethod::RadialAnalytic);
        assert_eq!(
            direct,
            sensitivities_for(ActiveGravityMethod::CurvedArcEq106)
        );
        assert_eq!(direct, sensitivities_for(ActiveGravityMethod::Fmm));
        assert_eq!(
            direct,
            sensitivities_for(ActiveGravityMethod::MmfftCompressed)
        );
    }

    fn source_record(direction: Vec3, density: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for value in [
            direction.x,
            direction.y,
            direction.z,
            1.0, // solid angle
            0.0, // inner radius
            100.0,
            density,
            0.0, // record padding
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn voxel_prior_does_not_copy_forward_density() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes };

        let (voxels, _) = build_density_voxels(&source, ActiveGravityMethod::HomogeneousWerner)
            .expect("valid voxel source");

        assert_eq!(voxels.len(), 2);
        assert_eq!(voxels[0].density, voxels[1].density);
        assert_eq!(voxels[0].baseline_density, voxels[1].baseline_density);
    }

    #[test]
    fn werner_reference_field_is_uniform_and_scores_one_hundred_percent() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes };

        let (voxels, _) = build_density_voxels(&source, ActiveGravityMethod::HomogeneousWerner)
            .expect("valid voxel source");

        assert!(voxels.iter().all(|voxel| {
            voxel.reference_density == voxel.baseline_density
                && voxel.density == voxel.reference_density
        }));
        assert_eq!(density_model_deviation(&voxels), 0.0);
    }

    #[test]
    fn logarithmic_reference_is_separate_from_uniform_optimizer_prior() {
        let mut bytes = source_record(Vec3::X, 100.0);
        let mut outer = source_record(Vec3::NEG_X, 10_000.0);
        outer[20..24].copy_from_slice(&200.0_f32.to_le_bytes());
        bytes.extend(outer);
        let source = RadialGravitySource { bytes };

        let (voxels, _) =
            build_density_voxels(&source, ActiveGravityMethod::RadialAnalytic).unwrap();

        let mut ordered = voxels.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.center.length().total_cmp(&b.center.length()));
        assert_eq!(ordered[0].density, ordered[1].density);
        assert_eq!(ordered[0].density, ordered[0].baseline_density);
        assert!(ordered[0].reference_density < ordered[1].reference_density);
        assert!(density_model_deviation(&voxels) > 0.0);
    }

    #[test]
    fn model_deviation_is_volume_weighted_relative_rmse() {
        let voxels = [
            InvertedDensityVoxel {
                center: Vec3::ZERO,
                volume: 1.0,
                density: 2.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [0, 0, 0],
            },
            InvertedDensityVoxel {
                center: Vec3::X,
                volume: 3.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [1, 0, 0],
            },
        ];
        assert!((density_model_deviation(&voxels) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn spatial_sensitivity_distinguishes_voxel_locations() {
        let start = TrajectoryInversionKnot {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            simulation_time_seconds: 0.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let end = TrajectoryInversionKnot {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            simulation_time_seconds: 1.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let voxels = [
            InvertedDensityVoxel {
                center: Vec3::new(1.0, 0.0, 0.0),
                volume: 1.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [0, 0, 0],
            },
            InvertedDensityVoxel {
                center: Vec3::new(-1.0, 0.0, 0.0),
                volume: 1.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [1, 0, 0],
            },
        ];
        let source = RadialGravitySource {
            bytes: source_record(Vec3::X, 100.0),
        };
        let (_, sensitivities) =
            build_observations_and_sensitivities(&[start, end], &voxels, &source).unwrap();
        assert_ne!(sensitivities[0], sensitivities[1]);
        assert!(sensitivities[0].length() > sensitivities[1].length());
    }

    #[test]
    fn quintic_track_is_densely_sampled_with_consistent_velocity() {
        let acceleration = Vec3::new(2.0, -1.0, 0.5);
        let initial_velocity = Vec3::new(1.0, 2.0, 3.0);
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::ZERO,
                velocity: initial_velocity,
                simulation_time_seconds: 0.0,
                baseline_acceleration: Vec3::splat(99.0),
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: initial_velocity + 0.5 * acceleration,
                velocity: initial_velocity + acceleration,
                simulation_time_seconds: 1.0,
                baseline_acceleration: Vec3::splat(-99.0),
                body_rotation: Quat::IDENTITY,
            },
        ];
        let voxels = [InvertedDensityVoxel {
            center: Vec3::ZERO,
            volume: 1.0,
            density: 1.0,
            baseline_density: 1.0,
            reference_density: 2.0,
            grid: [0, 0, 0],
        }];

        let (observations, sensitivities) = build_observations_and_sensitivities(
            &knots,
            &voxels,
            &RadialGravitySource {
                bytes: source_record(Vec3::X, 100.0),
            },
        )
        .unwrap();
        let samples = sample_frozen_trajectory(&knots).unwrap();

        assert_eq!(observations.len(), TRAJECTORY_SAMPLES_PER_SEGMENT + 1);
        assert_eq!(sensitivities.len(), observations.len());
        assert!(observations.iter().all(|sample| sample.is_finite()));
        assert!((samples[0].velocity - initial_velocity).length() < 1.0e-5);
        assert!((samples[TRAJECTORY_SAMPLES_PER_SEGMENT].velocity - (initial_velocity + acceleration)).length() < 1.0e-5);
    }

    #[test]
    fn unedited_capture_uses_snapshot_aligned_acceleration() {
        let captured = Vec3::new(-2.0e-5, 3.0e-5, 1.0e-5);
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
                simulation_time_seconds: 0.0,
                baseline_acceleration: captured,
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: captured * 0.5,
                velocity: captured,
                simulation_time_seconds: 1.0,
                baseline_acceleration: captured,
                body_rotation: Quat::IDENTITY,
            },
        ];
        let voxels = [InvertedDensityVoxel {
            center: Vec3::ZERO,
            volume: 1.0,
            density: 1.0,
            baseline_density: 1.0,
            reference_density: 1.0,
            grid: [0, 0, 0],
        }];

        let (observations, _) = build_observations_and_sensitivities(
            &knots,
            &voxels,
            &RadialGravitySource {
                bytes: source_record(Vec3::X, 100.0),
            },
        )
        .unwrap();

        assert!(
            observations.iter().all(|observation| observation.is_finite())
        );
    }
}
