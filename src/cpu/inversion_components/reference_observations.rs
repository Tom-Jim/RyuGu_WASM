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
    let mut observations = Vec::with_capacity(samples.len());
    let mut matrix = Vec::with_capacity(samples.len() * columns.len());
    for observation_index in 0..samples.len() {
        let laplace_frequency = frequency_domain_laplace_frequency(
            observation_index, samples.len(),
        );
        let row = columns
            .iter()
            .map(|column| {
                frequency_domain_transform_for_column(
                    samples, column, spectral_radius, laplace_frequency,
                )
            })
            .collect::<Option<Vec<_>>>()?;
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

fn frequency_domain_transform_for_column(
    samples: &[TrajectoryInversionKnot],
    column: &[VoxelBasisSource],
    spectral_radius: f64,
    laplace_frequency: f64,
) -> Option<Vec3> {
    if samples.is_empty()
        || column.is_empty()
        || samples.windows(2).any(|pair| {
            pair[1].simulation_time_seconds < pair[0].simulation_time_seconds
        })
    {
        return None;
    }
    let mut total = Vec3::ZERO;
    for index in 0..EQ184_QUADRATURE_COUNT {
        let (k, weight) = eq184_quadrature_node(index, spectral_radius)?;
        let k2 = k.length_squared().max(1.0e-18);
        let mut characteristic = Complex64::new(0.0, 0.0);
        for (sample_index, sample) in samples.iter().enumerate() {
            let previous = samples.get(sample_index.wrapping_sub(1)).unwrap_or(sample);
            let next = samples.get(sample_index + 1).unwrap_or(sample);
            let body_position = sample
                .body_rotation
                .inverse()
                .mul_vec3(sample.position)
                .as_dvec3();
            characteristic += eq184_trajectory_term(
                k,
                body_position,
                previous.simulation_time_seconds,
                sample.simulation_time_seconds,
                next.simulation_time_seconds,
                sample_index,
                samples.len(),
                laplace_frequency,
            )?;
        }
        let mut density = Complex64::new(0.0, 0.0);
        for source in column {
            let phase = -k.dot(source.position);
            density += Complex64::from_polar(source.volume, phase);
        }
        let coefficient = G as f64 * 4.0 * std::f64::consts::PI
            / std::f64::consts::TAU.powi(3)
            * weight
            / k2;
        let contribution = (-coefficient * (density * characteristic).im * k).as_vec3();
        total += contribution;
    }
    Some(total)
}
