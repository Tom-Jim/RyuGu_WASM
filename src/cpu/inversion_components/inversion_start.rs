/// Inputs of a requested inversion, kept while the numerical Worker evaluates
/// the high-resolution reference observations and the method's voxel
/// sensitivity matrix. Both are `source_sets` requests on the sensitivity
/// channel, issued one after the other.
pub(crate) struct InversionStartJob {
    method: ActiveGravityMethod,
    capture_id: u64,
    source_hash: u64,
    capture_epoch: u64,
    voxels: Vec<InvertedDensityVoxel>,
    voxel_size: f32,
    basis_sources: VoxelBasisSources,
    samples: Vec<TrajectoryInversionKnot>,
    holdout_samples: Vec<TrajectoryInversionKnot>,
    /// Reference source sets still to be evaluated; `None` once the reference
    /// cache in `TrajectoryInversionState` is valid for this capture.
    reference_sets: Option<Vec<Vec<(DVec3, f64)>>>,
    /// Method sensitivity matrix once it is known (cache hit or delivered).
    sensitivities: Option<Vec<Vec3>>,
    pending: Option<PendingInversionRequest>,
    common_started: Instant,
    watchdog_reset: bool,
    truth_prepare_ms: Option<f64>,
    method_started: Instant,
    matrix_started: Option<Instant>,
    timing: InversionTimingBreakdown,
}

enum PendingInversionRequest {
    Reference(crate::cpp_backend::BackendSensitivitySnapshot),
    Sensitivity(crate::cpp_backend::BackendSensitivitySnapshot),
}

impl PendingInversionRequest {
    fn snapshot(&self) -> crate::cpp_backend::BackendSensitivitySnapshot {
        match self {
            Self::Reference(snapshot) | Self::Sensitivity(snapshot) => *snapshot,
        }
    }
}

#[derive(Default)]
pub(crate) struct InversionStartWorker {
    job: Option<InversionStartJob>,
    next_request_id: u64,
}

pub fn start_density_inversion_system(
    active_method: Res<ActiveGravityMethod>,
    radial_source: Option<Res<DensityQuadratureSource>>,
    aggregated_source: Option<Res<AggregatedGravitySource>>,
    channel: Res<crate::cpp_backend::BackendSensitivityChannel>,
    mut sensitivity_caches: ResMut<DensitySensitivityCaches>,
    mut frequency_domain_sensitivity: ResMut<FrequencyDomainSensitivityMatrix>,
    mut frequency_domain_performance: ResMut<FrequencyDomainPerformanceMetrics>,
    mut show_section: ResMut<ShowSection>,
    mut inversion: ResMut<TrajectoryInversionState>,
    mut worker: Local<InversionStartWorker>,
) {
    let InversionStartWorker {
        job: job_slot,
        next_request_id,
    } = &mut *worker;
    // Invert is independent of the First/Stress metric. Switching Speedup
    // used to abandon a running FMM/FFT invert and leave the 3D view blank.
    // A repeated Invert click while a job is in flight is ignored instead of
    // restarting from scratch. Capture/method changes still cancel it.
    let abandoned = job_slot.as_ref().is_some_and(|job| {
        !inversion.ready
            || inversion.capture_id != Some(job.capture_id)
            || inversion.capture_epoch != job.capture_epoch
            || *active_method != job.method
    });
    if abandoned {
        *job_slot = None;
        channel.reset();
        inversion.preparing = false;
    }
    if job_slot.is_some() && inversion.start_requested {
        inversion.start_requested = false;
    }
    if job_slot.is_none() {
        if !validate_inversion_request(inversion.start_requested, *active_method, &mut inversion)
        {
            inversion.preparing = false;
            return;
        }
        inversion.start_requested = false;
        match begin_inversion_start(
            *active_method,
            radial_source.as_deref(),
            aggregated_source.as_deref(),
            &mut inversion,
        ) {
            Ok(job) => {
                *job_slot = Some(job);
                inversion.preparing = true;
            }
            Err(message) => {
                inversion.error = Some(message);
                inversion.preparing = false;
                return;
            }
        }
    }
    inversion.preparing = true;
    let job = job_slot.as_mut().expect("inversion start job");

    if let Some(pending) = job.pending.take() {
        match channel.take() {
            None if !channel.is_idle() => {
                let waited = job.common_started.elapsed().as_secs_f64();
                if waited > 20.0 {
                    inversion.error = Some(
                        "Inversion observations did not return from the numerical Worker.".into(),
                    );
                    inversion.preparing = false;
                    *job_slot = None;
                    channel.reset();
                    return;
                }
                if waited > 8.0 && !job.watchdog_reset {
                    channel.reset();
                    job.watchdog_reset = true;
                } else {
                    job.pending = Some(pending);
                    return;
                }
            }
            // An experiment reset discarded the request; it is re-issued below.
            None => {}
            Some(packet)
                if packet.snapshot == pending.snapshot()
                    && packet.snapshot.epoch == job.capture_epoch =>
            {
                let outcome = match pending {
                    PendingInversionRequest::Reference(_) => {
                        apply_reference_answer(job, packet.result, &mut inversion)
                    }
                    PendingInversionRequest::Sensitivity(_) => {
                        apply_sensitivity_answer(job, packet.result, &mut sensitivity_caches)
                    }
                };
                if let Err(message) = outcome {
                    inversion.error = Some(message);
                    inversion.preparing = false;
                    *job_slot = None;
                    return;
                }
            }
            // A mismatching answer belongs to a superseded request.
            Some(_) => {}
        }
    }

    let waited = job.common_started.elapsed().as_secs_f64();
    if waited > 20.0 {
        inversion.error = Some(
            "Inversion observations did not return from the numerical Worker.".into(),
        );
        inversion.preparing = false;
        *job_slot = None;
        channel.reset();
        return;
    }
    if waited > 8.0 && !job.watchdog_reset {
        // A mismatched u64 id or a leftover in_flight flag leaves this
        // channel wedged: begin() returns false forever and Invert stays
        // on Preparing. One reset re-issues the same source_sets job.
        channel.reset();
        job.pending = None;
        job.watchdog_reset = true;
    }

    if let Some(sets) = job.reference_sets.as_ref() {
        let mut targets = knot_body_targets(&job.samples);
        targets.extend(knot_body_targets(&job.holdout_samples));
        let snapshot = next_inversion_snapshot(next_request_id, job);
        match crate::cpp_backend::request_source_sets(&channel, snapshot, "direct", sets, &targets)
        {
            Ok(true) => job.pending = Some(PendingInversionRequest::Reference(snapshot)),
            Ok(false) => {}
            Err(message) => {
                inversion.error = Some(message);
                inversion.preparing = false;
                *job_slot = None;
            }
        }
        return;
    }
    if job.truth_prepare_ms.is_none() {
        job.truth_prepare_ms = Some(job.common_started.elapsed().as_secs_f64() * 1.0e3);
        job.method_started = Instant::now();
    }
    let truth_prepare_ms = job.truth_prepare_ms.expect("truth preparation timed");

    match job.method {
        ActiveGravityMethod::FrequencyDomain => {
            if job.sensitivities.is_none() {
                job.timing.matrix_cache_hit = prepare_frequency_domain_cache(
                    job.capture_id,
                    job.source_hash,
                    job.basis_sources.hash,
                    job.voxels.len(),
                    job.samples.len(),
                    &mut frequency_domain_sensitivity,
                    &mut frequency_domain_performance,
                    truth_prepare_ms,
                );
                job.sensitivities = Some(inversion.reference_training_sensitivities.clone());
            }
        }
        ActiveGravityMethod::MmfftCompressed | ActiveGravityMethod::Fmm => {
            if job.sensitivities.is_none() {
                let cache = &sensitivity_caches.0[job.method.performance_index()];
                let cache_hit = cache.capture_id == Some(job.capture_id)
                    && cache.source_hash == job.source_hash
                    && cache.basis_hash == job.basis_sources.hash
                    && cache.sample_count == job.samples.len()
                    && cache.values.len() == job.samples.len() * job.voxels.len();
                if cache_hit {
                    job.timing.matrix_cache_hit = true;
                    job.sensitivities = Some(cache.values.clone());
                } else {
                    job.matrix_started.get_or_insert_with(Instant::now);
                    let key = if job.method == ActiveGravityMethod::MmfftCompressed {
                        "fft"
                    } else {
                        "fmm"
                    };
                    let columns: Vec<Vec<(DVec3, f64)>> = job
                        .basis_sources
                        .columns
                        .iter()
                        .map(|column| column.iter().map(|s| (s.position, s.volume)).collect())
                        .collect();
                    let targets = knot_body_targets(&job.samples);
                    let snapshot = next_inversion_snapshot(next_request_id, job);
                    match crate::cpp_backend::request_source_sets(
                        &channel, snapshot, key, &columns, &targets,
                    ) {
                        Ok(true) => {
                            job.pending = Some(PendingInversionRequest::Sensitivity(snapshot));
                        }
                        Ok(false) => {}
                        Err(message) => {
                            inversion.error = Some(message);
                            inversion.preparing = false;
                            *job_slot = None;
                        }
                    }
                    return;
                }
            }
        }
        _ => unreachable!("forward-only methods were rejected"),
    }

    let job = job_slot.take().expect("inversion start job");
    inversion.preparing = false;
    let sensitivities = job.sensitivities.expect("sensitivity matrix resolved");
    let current_densities = job
        .voxels
        .iter()
        .map(|voxel| voxel.density)
        .collect::<Vec<_>>();
    let mut optimizer = ConvexOptimizationJob {
        method: job.method,
        capture_id: job.capture_id,
        source_hash: job.source_hash,
        neighbours: build_neighbours(&job.voxels),
        voxels: job.voxels,
        basis_sources: job.basis_sources,
        frozen_samples: job.samples,
        sensitivities,
        observed_accelerations: inversion.reference_training_observations.clone(),
        holdout_observations: inversion.reference_holdout_observations.clone(),
        holdout_sensitivities: inversion.reference_holdout_sensitivities.clone(),
        current_densities: current_densities.clone(),
        best_densities: current_densities,
        initial_objective: f64::INFINITY,
        data_error_scale: 1.0,
        voxel_size: job.voxel_size,
        started_at: job.method_started,
        source_preparation_ms: truth_prepare_ms,
        timing: job.timing,
    };
    optimizer.data_error_scale = trajectory_data_error(&optimizer).max(1.0e-24);
    optimizer.initial_objective = objective(&optimizer);
    if !optimizer.initial_objective.is_finite() {
        inversion.error = Some("The voxel sensitivity matrix is not finite.".into());
        return;
    }
    inversion.inverted = true;
    // Recovered voxels are the overlay, not forward D. Do not auto-enable Section.
    show_section.0 = false;
    inversion.displayed_density = Some(density_result_from_job(
        &optimizer,
        &optimizer.best_densities,
        0.0,
    ));
    inversion.error = None;
    inversion.optimizer = Some(optimizer);
}

fn next_inversion_snapshot(
    next_request_id: &mut u64,
    job: &InversionStartJob,
) -> crate::cpp_backend::BackendSensitivitySnapshot {
    *next_request_id = next_request_id.wrapping_add(1).max(1);
    crate::cpp_backend::BackendSensitivitySnapshot {
        request_id: *next_request_id,
        epoch: job.capture_epoch,
    }
}

/// Assembles everything the inversion needs before any Worker round trip.
fn begin_inversion_start(
    method: ActiveGravityMethod,
    radial_source: Option<&DensityQuadratureSource>,
    aggregated_source: Option<&AggregatedGravitySource>,
    inversion: &mut TrajectoryInversionState,
) -> Result<InversionStartJob, String> {
    let common_started = Instant::now();
    let source = radial_source.ok_or("The asteroid volume source is not ready.")?;
    let aggregated = aggregated_source.ok_or("The aggregated gravity source is not ready.")?;
    let capture_id = inversion.capture_id.expect("validated inversion capture");
    let source_hash = inversion.capture_source_hash;
    let (voxels, voxel_size) = build_density_voxels(source, method)
        .ok_or("The asteroid volume could not be voxelized.")?;
    if voxels.len() != EXPECTED_VOXEL_COUNT {
        return Err(format!(
            "The convex inverse requires 56 voxels, but voxelization produced {}.",
            voxels.len()
        ));
    }
    let live_bytes = crate::cpu::density::reduce_live_quadrature_bytes(&source.bytes);
    let live_aggregated = AggregatedGravitySource {
        sources: crate::cpu::frequency_domain::point_sources_from_quadrature_bytes(&live_bytes),
        constant_sources: Vec::new(),
        total_mass: aggregated.total_mass,
        constant_total_mass: aggregated.constant_total_mass,
        radius: aggregated.radius,
        source_hash: aggregated.source_hash,
        constant_hash: aggregated.constant_hash,
    };
    // FMM/FFT invert must use the same 32-angular live mesh as Verlet. FD
    // invert is the Eq.(184) operator on the full quadrature and stays that way.
    let basis_geometry = if method == ActiveGravityMethod::FrequencyDomain {
        aggregated
    } else {
        &live_aggregated
    };
    let basis_sources = build_voxel_basis_sources(&voxels, basis_geometry, voxel_size)
        .ok_or("The shared mass-preserving voxel basis is not ready.")?;
    let samples = sample_frozen_trajectory(&inversion.knots)
        .ok_or("The frozen trajectory cannot be sampled.")?;
    if !cfg!(target_arch = "wasm32") && method != ActiveGravityMethod::FrequencyDomain {
        return Err("Density inversion requires the browser numerical Worker.".into());
    }
    let training_count = (inversion.knots.len() - 1) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1;
    let holdout_count = (inversion.knots.len() - 1) * HOLDOUT_SAMPLES_PER_SEGMENT;
    let mut reference_sets = None;
    // Only the pointwise reference evaluates holdout samples in the Worker;
    // the frequency-domain operator derives its own holdout rows.
    let mut holdout_samples = Vec::new();
    if method == ActiveGravityMethod::FrequencyDomain {
        // Equation (184) is an integral observation operator.  For this method
        // the RHS and unit-density columns must be generated by the same
        // discrete Fourier--Laplace operator; do not mix them with the
        // pointwise reference cache.
        let (training, training_basis, holdout, holdout_basis) =
            frequency_domain_training_and_holdout_reference(
                &inversion.knots,
                &samples,
                &basis_sources,
                &voxels,
                aggregated.radius,
            )
            .ok_or("The frequency-domain observation operator could not be assembled.")?;
        inversion.reference_cache_capture_id = None;
        inversion.reference_training_observations = training;
        inversion.reference_training_sensitivities = training_basis;
        inversion.reference_holdout_observations = holdout;
        inversion.reference_holdout_sensitivities = holdout_basis;
    } else if !reference_cache_matches(
        inversion,
        capture_id,
        source_hash,
        training_count,
        holdout_count,
        voxels.len(),
    ) {
        let message = "The frozen trajectory has no valid reference observations.";
        holdout_samples = holdout_frozen_trajectory(&inversion.knots).ok_or(message)?;
        reference_sets = Some(reference_source_sets(&voxels, source).ok_or(message)?);
    }
    Ok(InversionStartJob {
        method,
        capture_id,
        source_hash,
        capture_epoch: inversion.capture_epoch,
        voxels,
        voxel_size,
        basis_sources,
        samples,
        holdout_samples,
        reference_sets,
        sensitivities: None,
        pending: None,
        common_started,
        watchdog_reset: false,
        truth_prepare_ms: None,
        method_started: common_started,
        matrix_started: None,
        timing: InversionTimingBreakdown::default(),
    })
}

fn apply_reference_answer(
    job: &mut InversionStartJob,
    result: Result<Vec<f64>, String>,
    inversion: &mut TrajectoryInversionState,
) -> Result<(), String> {
    let values = result?;
    let expected = (1 + job.voxels.len()) * (job.samples.len() + job.holdout_samples.len()) * 4;
    let (training, training_basis, holdout, holdout_basis) = decode_training_and_holdout_reference(
        &values,
        &job.samples,
        &job.holdout_samples,
        job.voxels.len(),
    )
    .ok_or_else(|| {
        format!(
            "The frozen trajectory has no valid reference observations (got {} values, expected {expected}).",
            values.len()
        )
    })?;
    inversion.reference_cache_capture_id = Some(job.capture_id);
    inversion.reference_cache_source_hash = job.source_hash;
    inversion.reference_training_observations = training;
    inversion.reference_training_sensitivities = training_basis;
    inversion.reference_holdout_observations = holdout;
    inversion.reference_holdout_sensitivities = holdout_basis;
    job.reference_sets = None;
    Ok(())
}

fn apply_sensitivity_answer(
    job: &mut InversionStartJob,
    result: Result<Vec<f64>, String>,
    sensitivity_caches: &mut DensitySensitivityCaches,
) -> Result<(), String> {
    let values = result.and_then(|values| {
        decode_voxel_basis_sensitivities(&values, &job.basis_sources, &job.samples)
    })?;
    job.timing.matrix_build_ms = job
        .matrix_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1.0e3);
    sensitivity_caches.0[job.method.performance_index()] = DensitySensitivityCache {
        capture_id: Some(job.capture_id),
        source_hash: job.source_hash,
        basis_hash: job.basis_sources.hash,
        sample_count: job.samples.len(),
        values: values.clone(),
    };
    job.sensitivities = Some(values);
    Ok(())
}

fn validate_inversion_request(
    pressed: bool,
    method: ActiveGravityMethod,
    inversion: &mut TrajectoryInversionState,
) -> bool {
    if !pressed || inversion.optimizer.is_some() || inversion.preparing {
        return false;
    }
    if matches!(
        method,
        ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner
    ) {
        inversion.error = Some(
            "Radial generates observations and Werner is forward-only; neither is inverted.".into(),
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
    if inversion.batch_capture_id != Some(capture_id) {
        inversion.results = std::array::from_fn(|_| None);
        inversion.batch_capture_id = Some(capture_id);
    }
    let source_changed = inversion
        .best_results
        .iter()
        .flatten()
        .any(|result| result.source_hash != source_hash);
    if source_changed {
        inversion.best_results = std::array::from_fn(|_| None);
    }
    inversion.results[method.performance_index()] = None;
    inversion.displayed_density = None;
    inversion.error = None;
    true
}

/// Method-independent truth observations are cached for one immutable
/// trajectory/source identity and reused across inverse methods.
fn reference_cache_matches(
    inversion: &TrajectoryInversionState,
    capture_id: u64,
    source_hash: u64,
    training_count: usize,
    holdout_count: usize,
    voxel_count: usize,
) -> bool {
    inversion.reference_cache_capture_id == Some(capture_id)
        && inversion.reference_cache_source_hash == source_hash
        && inversion.reference_training_observations.len() == training_count
        && inversion.reference_training_sensitivities.len() == training_count * voxel_count
        && inversion.reference_holdout_observations.len() == holdout_count
        && inversion.reference_holdout_sensitivities.len() == holdout_count * voxel_count
}

fn prepare_frequency_domain_cache(
    capture_id: u64,
    source_hash: u64,
    basis_hash: u64,
    voxel_count: usize,
    sample_count: usize,
    sensitivity: &mut FrequencyDomainSensitivityMatrix,
    performance: &mut FrequencyDomainPerformanceMetrics,
    truth_prepare_ms: f64,
) -> bool {
    let cache_hit = sensitivity.capture_id == Some(capture_id)
        && sensitivity.source_hash == source_hash
        && sensitivity.basis_hash == basis_hash
        && sensitivity.configuration_hash
            == crate::gpu::frequency_domain::frequency_domain_sensitivity_configuration_hash()
        && sensitivity.voxel_count == voxel_count
        && sensitivity.sample_count == sample_count
        && sensitivity.columns.len() == voxel_count;
    if !cache_hit {
        sensitivity.capture_id = Some(capture_id);
        sensitivity.source_hash = source_hash;
        sensitivity.basis_hash = basis_hash;
        sensitivity.configuration_hash =
            crate::gpu::frequency_domain::frequency_domain_sensitivity_configuration_hash();
        sensitivity.voxel_count = voxel_count;
        sensitivity.sample_count = 0;
        sensitivity.columns.clear();
    }
    performance.inversion = Some(FrequencyDomainInversionTiming {
        source_preparation_ms: truth_prepare_ms,
        ..default()
    });
    cache_hit
}
