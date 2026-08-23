#[cfg(test)]
fn build_observations_and_sensitivities(
    knots: &[TrajectoryInversionKnot],
    voxels: &[InvertedDensityVoxel],
    radial_source: &RadialGravitySource,
) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    if knots.len() < 2 {
        return None;
    }
    let sample_count = (knots.len() - 1) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1;
    let samples = sample_frozen_trajectory(knots)?;
    let result = reference_observations_and_sensitivities(&samples, voxels, radial_source)?;
    (result.0.len() == sample_count).then_some(result)
}

#[cfg(test)]
fn reference_observations_and_sensitivities(
    samples: &[TrajectoryInversionKnot],
    voxels: &[InvertedDensityVoxel],
    radial_source: &RadialGravitySource,
) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    if samples.is_empty() {
        return None;
    }
    let tree = high_resolution_reference_tree(radial_source)?;
    let basis_trees = high_resolution_reference_basis_trees(voxels, radial_source)?;
    let observations = reference_observations(samples, &tree)?;
    let sensitivities = evaluate_reference_basis(&basis_trees, samples);
    Some((observations, sensitivities))
}

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
