pub fn start_density_inversion_system(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<Button>,
            With<TrajectoryInversionButton>,
        ),
    >,
    active_method: Res<ActiveGravityMethod>,
    radial_source: Option<Res<RadialGravitySource>>,
    aggregated_source: Option<Res<AggregatedGravitySource>>,
    mut sensitivity_caches: ResMut<DensitySensitivityCaches>,
    mut eq106_sensitivity: ResMut<Eq106SensitivityMatrix>,
    mut eq106_performance: ResMut<Eq106PerformanceMetrics>,
    mut show_section: ResMut<ShowSection>,
    mut inversion: ResMut<TrajectoryInversionState>,
    mut planning: ResMut<PlanningComparisonState>,
) {
    let pressed = interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed);
    if pressed {
        planning.run_requested = true;
        planning.run_id = planning.run_id.wrapping_add(1);
        planning.status = format!(
            "{} planning run queued for the frozen capture.",
            planning.workload_profile.label()
        );
    }
    if !validate_inversion_request(pressed, *active_method, &mut inversion) {
        return;
    }
    let common_started = Instant::now();
    let Some(source) = radial_source else {
        inversion.error = Some("The asteroid volume source is not ready.".into());
        return;
    };
    let Some(aggregated) = aggregated_source else {
        inversion.error = Some("The aggregated gravity source is not ready.".into());
        return;
    };
    let capture_id = inversion.capture_id.expect("validated inversion capture");
    let method = *active_method;
    let source_hash = inversion.capture_source_hash;
    let Some((voxels, voxel_size)) = build_density_voxels(&source, method) else {
        inversion.error = Some("The asteroid volume could not be voxelized.".into());
        return;
    };
    if voxels.len() != EXPECTED_VOXEL_COUNT {
        inversion.error = Some(format!(
            "The convex inverse requires 56 voxels, but voxelization produced {}.",
            voxels.len()
        ));
        return;
    }
    let Some(basis_sources) = build_voxel_basis_sources(&voxels, &aggregated) else {
        inversion.error = Some("The shared 1024-source voxel basis is not ready.".into());
        return;
    };
    let Some(samples) = sample_frozen_trajectory(&inversion.knots) else {
        inversion.error = Some("The frozen trajectory cannot be sampled.".into());
        return;
    };
    let training_count =
        (inversion.knots.len() - 1) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1;
    let holdout_count = (inversion.knots.len() - 1) * HOLDOUT_SAMPLES_PER_SEGMENT;
    prepare_reference_cache(
        capture_id,
        source_hash,
        training_count,
        holdout_count,
        &samples,
        &voxels,
        &source,
        &mut inversion,
    );
    if inversion.error.is_some() {
        return;
    }
    let truth_prepare_ms = common_started.elapsed().as_secs_f64() * 1.0e3;
    let method_started = Instant::now();
    let mut timing = InversionTimingBreakdown {
        truth_prepare_ms,
        ..default()
    };
    let mut sensitivities = inversion.reference_training_sensitivities.clone();
    match method {
        ActiveGravityMethod::CurvedArcEq106 => {
            timing.matrix_cache_hit = prepare_eq106_cache(
                capture_id,
                source_hash,
                basis_sources.hash,
                voxels.len(),
                training_count,
                &mut eq106_sensitivity,
                &mut eq106_performance,
                truth_prepare_ms,
            );
        }
        ActiveGravityMethod::MmfftCompressed | ActiveGravityMethod::Fmm => {
            let cache = &mut sensitivity_caches.0[method.performance_index()];
            let cache_hit = cache.capture_id == Some(capture_id)
                && cache.source_hash == source_hash
                && cache.basis_hash == basis_sources.hash
                && cache.sample_count == samples.len()
                && cache.values.len() == samples.len() * voxels.len();
            timing.matrix_cache_hit = cache_hit;
            if cache_hit {
                sensitivities.clone_from(&cache.values);
            } else {
                let matrix_started = Instant::now();
                sensitivities = if method == ActiveGravityMethod::MmfftCompressed {
                    crate::gpu::mmfft::voxel_basis_sensitivities(&basis_sources, &samples)
                } else {
                    fmm_voxel_basis_sensitivities(&basis_sources, &samples)
                };
                timing.matrix_build_ms = matrix_started.elapsed().as_secs_f64() * 1.0e3;
                *cache = DensitySensitivityCache {
                    capture_id: Some(capture_id),
                    source_hash,
                    basis_hash: basis_sources.hash,
                    sample_count: samples.len(),
                    values: sensitivities.clone(),
                };
            }
        }
        _ => unreachable!("forward-only methods were rejected"),
    }
    let observed_accelerations = inversion.reference_training_observations.clone();
    let holdout_observations = inversion.reference_holdout_observations.clone();
    let holdout_sensitivities = inversion.reference_holdout_sensitivities.clone();
    let current_densities = voxels.iter().map(|voxel| voxel.density).collect::<Vec<_>>();
    let mut job = ConvexOptimizationJob {
        method,
        capture_id,
        source_hash,
        capture_epoch: inversion.capture_epoch,
        problem_id: inversion_problem_id(capture_id, source_hash),
        neighbours: build_neighbours(&voxels),
        voxels,
        basis_sources,
        frozen_samples: samples,
        sensitivities,
        observed_accelerations,
        holdout_observations,
        holdout_sensitivities,
        current_densities: current_densities.clone(),
        best_densities: current_densities,
        initial_objective: f64::INFINITY,
        data_error_scale: 1.0,
        iterations: QP_SOLVE_COUNT,
        voxel_size,
        started_at: method_started,
        source_preparation_ms: truth_prepare_ms,
        timing,
    };
    job.data_error_scale = trajectory_data_error(&job).max(1.0e-24);
    job.initial_objective = objective(&job);
    if !job.initial_objective.is_finite() {
        inversion.error = Some("The voxel sensitivity matrix is not finite.".into());
        return;
    }
    inversion.inverted = true;
    show_section.0 = false;
    inversion.displayed_density = Some(density_result_from_job(
        &job,
        &job.best_densities,
        job.initial_objective,
        0.0,
    ));
    inversion.selected = None;
    inversion.edit_buffer.clear();
    inversion.error = None;
    inversion.optimizer = Some(job);
}

fn validate_inversion_request(
    pressed: bool,
    method: ActiveGravityMethod,
    inversion: &mut TrajectoryInversionState,
) -> bool {
    if !pressed || inversion.optimizer.is_some() {
        return false;
    }
    if matches!(method, ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner)
    {
        inversion.error = Some(
            "Radial generates observations and Werner is forward-only; neither is inverted."
                .into(),
        );
        return false;
    }
    if !inversion.ready || inversion.knots.len() != TRAJECTORY_INVERSION_SAMPLE_COUNT {
        return false;
    }
    let Some(capture_id) = inversion.capture_id else {
        inversion.error = Some("The frozen trajectory capture has no identity.".into());
        return false;
    };
    inversion.optimizer = None;
    let source_hash = inversion.capture_source_hash;
    let source_changed = inversion
        .best_results
        .iter()
        .flatten()
        .any(|result| result.source_hash != source_hash);
    if inversion.batch_capture_id != Some(capture_id) || source_changed {
        inversion.results = std::array::from_fn(|_| None);
        inversion.best_results = std::array::from_fn(|_| None);
        inversion.batch_capture_id = Some(capture_id);
    }
    inversion.results[method.performance_index()] = None;
    inversion.displayed_density = None;
    inversion.error = None;
    true
}

fn prepare_reference_cache(
    capture_id: u64,
    source_hash: u64,
    training_count: usize,
    holdout_count: usize,
    training_samples: &[TrajectoryInversionKnot],
    voxels: &[InvertedDensityVoxel],
    source: &RadialGravitySource,
    inversion: &mut TrajectoryInversionState,
) {
    let matches = inversion.reference_cache_capture_id == Some(capture_id)
        && inversion.reference_cache_source_hash == source_hash
        && inversion.reference_training_observations.len() == training_count
        && inversion.reference_training_sensitivities.len() == training_count * voxels.len()
        && inversion.reference_holdout_observations.len() == holdout_count
        && inversion.reference_holdout_sensitivities.len() == holdout_count * voxels.len();
    if matches {
        return;
    }
    let Some((training, training_basis, holdout, holdout_basis)) =
        training_and_holdout_reference(&inversion.knots, training_samples, voxels, source)
    else {
        inversion.error = Some("The frozen trajectory has no valid reference observations.".into());
        return;
    };
    inversion.reference_cache_capture_id = Some(capture_id);
    inversion.reference_cache_source_hash = source_hash;
    inversion.reference_training_observations = training;
    inversion.reference_training_sensitivities = training_basis;
    inversion.reference_holdout_observations = holdout;
    inversion.reference_holdout_sensitivities = holdout_basis;
}

fn prepare_eq106_cache(
    capture_id: u64,
    source_hash: u64,
    basis_hash: u64,
    voxel_count: usize,
    sample_count: usize,
    sensitivity: &mut Eq106SensitivityMatrix,
    performance: &mut Eq106PerformanceMetrics,
    truth_prepare_ms: f64,
) -> bool {
    let cache_hit = sensitivity.capture_id == Some(capture_id)
        && sensitivity.source_hash == source_hash
        && sensitivity.basis_hash == basis_hash
        && sensitivity.configuration_hash
            == crate::gpu::eq106::eq106_sensitivity_configuration_hash()
        && sensitivity.voxel_count == voxel_count
        && sensitivity.sample_count == sample_count
        && sensitivity.columns.len() == voxel_count;
    if !cache_hit {
        sensitivity.capture_id = Some(capture_id);
        sensitivity.source_hash = source_hash;
        sensitivity.basis_hash = basis_hash;
        sensitivity.configuration_hash = crate::gpu::eq106::eq106_sensitivity_configuration_hash();
        sensitivity.voxel_count = voxel_count;
        sensitivity.sample_count = 0;
        sensitivity.columns.clear();
    }
    performance.inversion = Some(Eq106InversionTiming {
        source_preparation_ms: truth_prepare_ms,
        matrix_cache_hit: cache_hit,
        ..default()
    });
    cache_hit
}

fn inversion_problem_id(capture_id: u64, source_hash: u64) -> u64 {
    0x9e37_79b9_7f4a_7c15_u64 ^ capture_id.rotate_left(17) ^ source_hash.rotate_right(11)
}
