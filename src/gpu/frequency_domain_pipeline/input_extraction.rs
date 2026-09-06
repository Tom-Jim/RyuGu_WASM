fn extract_frequency_domain_input(
    mut extracted: ResMut<ExtractedFrequencyDomainInput>,
    volume_source: Extract<Option<Res<DensityQuadratureSource>>>,
    active: Extract<Res<ActiveGravityMethod>>,
    planning: Extract<Res<PlanningComparisonState>>,
    performance: Extract<Res<PerformanceComparisonState>>,
    clock: Extract<Res<SimulationClock>>,
    inversion: Extract<Res<TrajectoryInversionState>>,
    sensitivity: Extract<Res<FrequencyDomainSensitivityMatrix>>,
    density_mode: Extract<Res<DensityMode>>,
) {
    extracted.enabled =
        **active == ActiveGravityMethod::FrequencyDomain && !planning.blocks_realtime_gpu();
    extracted.snapshot = None;
    extracted.target_bytes.clear();
    extracted.observation_count = 0;
    extracted.batch_elements.clear();
    extracted.batch_capture_id = None;
    extracted.runtime_revision = 0;
    extracted.sensitivity_sources.clear();
    extracted.sensitivity_source_counts.clear();
    extracted.sensitivity_source_hash = 0;
    extracted.sensitivity_basis_hash = 0;
    extracted.source_layout = 1;
    if !extracted.enabled {
        return;
    }
    let Some(volume_source) = volume_source.as_ref() else {
        return;
    };
    let (volume_bytes, volume_hash) = match **density_mode {
        DensityMode::Variable => (&volume_source.bytes, volume_source.source_hash),
        DensityMode::Constant => (&volume_source.constant_bytes, volume_source.constant_hash),
    };
    extracted.source_count = (volume_bytes.len() / 32) as u32;
    extracted.radius = volume_source.radius;

    let pending_sensitivity = inversion.optimizer.as_ref().and_then(|job| {
        (job.method == ActiveGravityMethod::FrequencyDomain
            && sensitivity.capture_id == Some(job.capture_id)
            && sensitivity.source_hash == job.source_hash
            && sensitivity.basis_hash == job.basis_sources.hash
            && sensitivity.configuration_hash == frequency_domain_sensitivity_configuration_hash()
            && sensitivity.columns.is_empty())
        .then_some((job.capture_id, job))
    });
    if let Some((capture_id, job)) = pending_sensitivity {
        let samples = &job.frozen_samples;
        if upload_known_trajectory(&mut extracted, samples, inversion.capture_epoch) {
            extracted.batch_capture_id = Some(capture_id);
            extracted.sensitivity_source_hash = job.source_hash;
            extracted.sensitivity_basis_hash = job.basis_sources.hash;
            extracted.source_layout = 0;
            extracted
                .sensitivity_sources
                .reserve(job.basis_sources.columns.len());
            extracted
                .sensitivity_source_counts
                .reserve(job.basis_sources.columns.len());
            for column in &job.basis_sources.columns {
                let mut bytes = Vec::with_capacity(column.len() * 16);
                for source in column {
                    let record = [
                        source.position.x as f32,
                        source.position.y as f32,
                        source.position.z as f32,
                        source.volume as f32,
                    ];
                    bytes.extend_from_slice(bytemuck::bytes_of(&record));
                }
                extracted.sensitivity_sources.push(bytes);
                extracted
                    .sensitivity_source_counts
                    .push(column.len() as u32);
            }
            if let Some(first) = extracted.sensitivity_sources.first() {
                extracted.sources = Some(first.clone());
                extracted.source_count = extracted.sensitivity_source_counts[0];
                extracted.source_hash = capture_id
                    ^ job.source_hash.rotate_left(29)
                    ^ job.basis_sources.hash.rotate_right(7);
            }
        }
    } else if inversion.ready
        && let Some(capture_id) = inversion.capture_id
    {
        let sample_count = inversion.knots.len();
        let revision = if performance.active && performance.measuring {
            clock.request_id
        } else {
            0
        };
        if upload_known_trajectory(
            &mut extracted,
            &inversion.knots[..sample_count],
            inversion.capture_epoch,
        ) {
            extracted.batch_capture_id = Some(capture_id);
            extracted.runtime_revision = revision;
        }
    }

    let source_hash = volume_hash;
    if extracted.sensitivity_sources.is_empty()
        && (extracted.sources.is_none() || extracted.source_hash != source_hash)
    {
        extracted.sources = Some(volume_bytes.clone());
    }
    if extracted.sensitivity_sources.is_empty() {
        extracted.source_hash = source_hash;
    }
}

fn upload_known_trajectory(
    extracted: &mut ExtractedFrequencyDomainInput,
    samples: &[TrajectoryInversionKnot],
    epoch: u64,
) -> bool {
    if samples.len() < 2
        || samples.iter().any(|sample| {
            !sample.position.is_finite()
                || !sample.body_rotation.is_finite()
                || !sample.simulation_time_seconds.is_finite()
                || !(sample.simulation_time_seconds as f32).is_finite()
        })
        || samples[0].simulation_time_seconds < 0.0
        || samples
            .windows(2)
            .any(|pair| pair[1].simulation_time_seconds < pair[0].simulation_time_seconds)
    {
        return false;
    }
    let mut positions = Vec::with_capacity(samples.len());
    let mut times = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        let body_position = sample.body_rotation.inverse() * sample.position;
        positions.push(body_position);
        times.push(sample.simulation_time_seconds as f32);
        let record = [
            body_position.x,
            body_position.y,
            body_position.z,
            sample.simulation_time_seconds as f32,
        ];
        extracted
            .target_bytes
            .extend_from_slice(bytemuck::bytes_of(&record));
        if index == 0 {
            extracted.snapshot = Some(GravityRequestSnapshot {
                request_id: 0,
                epoch,
                simulation_time_seconds: sample.simulation_time_seconds,
            });
        }
    }
    extracted.observation_count = samples.len() as u32;
    extracted.batch_elements = build_trajectory_batch_elements(&positions, &times);
    !extracted.batch_elements.is_empty()
}

fn initialize_frequency_domain_pipeline(world: &mut World) {
    let enabled = world.resource::<ExtractedFrequencyDomainInput>().enabled
        || world
            .get_resource::<crate::gpu::planning::ExtractedPlanningInput>()
            .is_some_and(|planning| {
                planning.request.method == Some(ActiveGravityMethod::FrequencyDomain)
            });
    if enabled && !world.contains_resource::<FrequencyDomainComputePipeline>() {
        // Render schedules do not automatically flush ordinary Commands at
        // every set boundary; initialize directly in this exclusive system.
        world.init_resource::<FrequencyDomainComputePipeline>();
    }
}

fn report_frequency_domain_pipeline_errors(
    cache: &PipelineCache,
    pipelines: &FrequencyDomainComputePipeline,
    channel: &FrequencyDomainGpuReadbackChannel,
) {
    for (name, id) in [
        ("density spectrum", pipelines.density_spectrum_id),
        ("reciprocal-space samples", pipelines.assemble_id),
        ("trajectory field", pipelines.evaluate_id),
    ] {
        match cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            ) => {}
            CachedPipelineState::Err(error) => {
                error!(
                    target: "wgsl::frequency_domain",
                    pipeline = name,
                    error = ?error,
                    "Frequency-domain algorithm compute pipeline compilation failed"
                );
                if let Ok(mut slot) = channel.pipeline_error.try_lock()
                    && slot.is_none()
                {
                    *slot = Some(format!(
                        "Frequency-domain algorithm {name} GPU pipeline failed: {error}"
                    ));
                }
            }
            _ => {}
        }
    }
}
