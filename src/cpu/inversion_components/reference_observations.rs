fn training_and_holdout_reference(
    knots: &[TrajectoryInversionKnot],
    training_samples: &[TrajectoryInversionKnot],
    voxels: &[InvertedDensityVoxel],
    radial_source: &RadialGravitySource,
) -> Option<(Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>)> {
    if training_samples.is_empty() {
        return None;
    }
    let holdout_samples = holdout_frozen_trajectory(knots)?;
    let tree = high_resolution_reference_tree(radial_source)?;
    let basis_trees = high_resolution_reference_basis_trees(voxels, radial_source)?;
    Some((
        reference_observations(training_samples, &tree)?,
        evaluate_reference_basis(&basis_trees, training_samples),
        reference_observations(&holdout_samples, &tree)?,
        evaluate_reference_basis(&basis_trees, &holdout_samples),
    ))
}

fn reference_observations(samples: &[TrajectoryInversionKnot], tree: &FmmNode) -> Option<Vec<Vec3>> {
    samples
        .iter()
        .map(|sample| evaluate_reference_tree(tree, sample))
        .collect()
}

fn trajectory_data_error(job: &ConvexOptimizationJob) -> f64 {
    density_data_error(job, &job.current_densities)
}

fn density_data_error(job: &ConvexOptimizationJob, densities: &[f32]) -> f64 {
    normalized_field_error(
        &job.observed_accelerations,
        &job.sensitivities,
        job.voxels.len(),
        densities,
    )
}

fn holdout_data_error(job: &ConvexOptimizationJob, densities: &[f32]) -> f64 {
    normalized_field_error(
        &job.holdout_observations,
        &job.holdout_sensitivities,
        job.voxels.len(),
        densities,
    )
}

fn normalized_field_error(
    observations: &[Vec3],
    sensitivities: &[Vec3],
    voxel_count: usize,
    densities: &[f32],
) -> f64 {
    if observations.is_empty() || sensitivities.len() != observations.len() * voxel_count {
        return f64::NAN;
    }
    let mut error = 0.0_f64;
    for (sample, observed) in observations.iter().enumerate() {
        let prediction = (0..voxel_count).fold(Vec3::ZERO, |sum, voxel| {
            sum + sensitivities[sample * voxel_count + voxel] * densities[voxel]
        });
        let sigma = (observed.length() * OBSERVATION_NOISE_FRACTION)
            .max(OBSERVATION_NOISE_FLOOR);
        error += (prediction - *observed).length_squared() as f64 / f64::from(sigma * sigma);
    }
    error / observations.len() as f64
}

/// Discrete equation-(184) observation operator used by the frequency-domain
/// inverse. Every row is the transform of the complete known trajectory at a
/// distinct positive Laplace frequency; no aggregate row is duplicated or
/// interpreted as an instantaneous field sample.
pub(crate) fn frequency_domain_training_and_holdout_reference(
    knots: &[TrajectoryInversionKnot],
    training_samples: &[TrajectoryInversionKnot],
    basis: &VoxelBasisSources,
    voxels: &[InvertedDensityVoxel],
    spectral_radius: f64,
) -> Option<(Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>)> {
    if basis.columns.len() != voxels.len() || basis.columns.is_empty() {
        return None;
    }
    let holdout = holdout_frozen_trajectory(knots)?;
    let (training, training_basis) = frequency_domain_observation_rows(
        training_samples, &basis.columns, voxels, spectral_radius,
    )?;
    let (holdout_observations, holdout_basis) = frequency_domain_observation_rows(
        &holdout, &basis.columns, voxels, spectral_radius,
    )?;
    Some((training, training_basis, holdout_observations, holdout_basis))
}

fn frequency_domain_observation_rows(
    samples: &[TrajectoryInversionKnot],
    columns: &[Vec<VoxelBasisSource>],
    voxels: &[InvertedDensityVoxel],
    spectral_radius: f64,
) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    if samples.is_empty() || columns.len() != voxels.len() {
        return None;
    }
    let quadrature = (0..EQ184_QUADRATURE_COUNT)
        .map(|index| {
            let (wave_vector, weight) = eq184_quadrature_node(index, spectral_radius)?;
            let coefficient = G as f64 * 4.0 * std::f64::consts::PI
                / std::f64::consts::TAU.powi(3)
                * weight
                / wave_vector.length_squared().max(1.0e-18);
            Some((wave_vector, coefficient))
        })
        .collect::<Option<Vec<_>>>()?;

    // T_gamma depends on trajectory and Laplace frequency, never on density.
    // Compute it once per observation/k node instead of once per voxel column.
    let trajectory_spectra = (0..samples.len())
        .map(|observation_index| {
            let laplace_frequency =
                frequency_domain_laplace_frequency(observation_index, samples.len());
            quadrature
                .iter()
                .map(|(wave_vector, _)| {
                    samples
                        .iter()
                        .enumerate()
                        .try_fold(Complex64::new(0.0, 0.0), |sum, (sample_index, sample)| {
                            let previous = samples
                                .get(sample_index.wrapping_sub(1))
                                .unwrap_or(sample);
                            let next = samples.get(sample_index + 1).unwrap_or(sample);
                            let body_position = sample
                                .body_rotation
                                .inverse()
                                .mul_vec3(sample.position)
                                .as_dvec3();
                            Some(
                                sum + eq184_trajectory_term(
                                    *wave_vector,
                                    body_position,
                                    previous.simulation_time_seconds,
                                    sample.simulation_time_seconds,
                                    next.simulation_time_seconds,
                                    sample_index,
                                    samples.len(),
                                    laplace_frequency,
                                )?,
                            )
                        })
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;

    // rho_hat for a unit-density voxel is independent of Laplace frequency.
    let density_spectra = columns
        .iter()
        .map(|column| {
            quadrature
                .iter()
                .map(|(wave_vector, _)| {
                    column.iter().fold(Complex64::new(0.0, 0.0), |sum, source| {
                        sum + Complex64::from_polar(
                            source.volume,
                            -wave_vector.dot(source.position),
                        )
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut observations = Vec::with_capacity(samples.len());
    let mut matrix = Vec::with_capacity(samples.len() * columns.len());
    for trajectory_spectrum in &trajectory_spectra {
        let row = density_spectra
            .iter()
            .map(|density_spectrum| {
                quadrature
                    .iter()
                    .enumerate()
                    .fold(DVec3::ZERO, |sum, (index, (wave_vector, coefficient))| {
                        let product = density_spectrum[index] * trajectory_spectrum[index];
                        sum - *coefficient * product.im * *wave_vector
                    })
                    .as_vec3()
            })
            .collect::<Vec<_>>();
        observations.push(
            row.iter().zip(voxels).fold(Vec3::ZERO, |sum, (field, voxel)| {
                sum + *field * voxel.reference_density
            }),
        );
        matrix.extend(row);
    }
    Some((observations, matrix))
}

fn frequency_domain_laplace_frequency(index: usize, count: usize) -> f64 {
    eq184_laplace_sigma(index, count)
}
