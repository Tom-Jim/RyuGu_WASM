use std::sync::atomic::AtomicU32;

#[derive(Resource)]
struct PlanningFrequencyDomainDispatchState {
    last_request_id: u64,
    active_request_id: u64,
    next_stage: usize,
    active_started: Option<Instant>,
    active_method_preprocess_ms: f64,
    active_command_submission_ms: f64,
    active_elements: Vec<FrequencyDomainBatchElement>,
    active_invalid_candidates: Vec<u32>,
    active_maximum_elements_per_candidate: u32,
    active_build_spectrum: bool,
    active_build_basis_spectrum: bool,
    // A staged request owns the shared readback lock until either its map
    // callback runs or, when no map was scheduled yet, the submitted stages
    // have drained. Releasing it unconditionally on cancellation lets a new
    // request reuse a staging buffer still referenced by the old callback.
    active_map_scheduled: bool,
    active_verification_targets: Vec<u32>,
    active_uniform: Option<Buffer>,
    active_bind_groups: Vec<BindGroup>,
    active_timestamps: Option<crate::gpu::planning_timestamps::PlanningTimestampQueries>,
    batch_id: u64,
    payload_request_id: u64,
    output_size: u64,
    output_storage_size: u64,
    staging_size: u64,
    baseline_size: u64,
    metric_size: u64,
    layout: Option<BindGroupLayout>,
    sources: Option<Buffer>,
    quadrature: Option<Buffer>,
    spectrum: Option<Buffer>,
    density_spectra: Option<Buffer>,
    source_groups: Option<Buffer>,
    output: Option<Buffer>,
    baseline: Option<Buffer>,
    metrics: Option<Buffer>,
    staging: Option<Buffer>,
    canonical_elements: Vec<FrequencyDomainBatchElement>,
    spectrum_ready: bool,
    basis_spectrum_ready: bool,
    last_block_key: Option<(u64, u8)>,
    /// Number of Frequency-domain algorithm stages allowed in the next submission. The map
    /// callback updates this from the previous request's GPU timestamps.
    adaptive_stage_budget: Arc<AtomicU32>,
}

impl Default for PlanningFrequencyDomainDispatchState {
    fn default() -> Self {
        Self {
            last_request_id: 0,
            active_request_id: 0,
            next_stage: 0,
            active_started: None,
            active_method_preprocess_ms: 0.0,
            active_command_submission_ms: 0.0,
            active_elements: Vec::new(),
            active_invalid_candidates: Vec::new(),
            active_maximum_elements_per_candidate: 0,
            active_build_spectrum: false,
            active_build_basis_spectrum: false,
            active_map_scheduled: false,
            active_verification_targets: Vec::new(),
            active_uniform: None,
            active_bind_groups: Vec::new(),
            active_timestamps: None,
            batch_id: 0,
            payload_request_id: 0,
            output_size: 0,
            output_storage_size: 0,
            staging_size: 0,
            baseline_size: 0,
            metric_size: 0,
            layout: None,
            sources: None,
            quadrature: None,
            spectrum: None,
            density_spectra: None,
            source_groups: None,
            output: None,
            baseline: None,
            metrics: None,
            staging: None,
            canonical_elements: Vec::new(),
            spectrum_ready: false,
            basis_spectrum_ready: false,
            last_block_key: None,
            adaptive_stage_budget: Arc::new(AtomicU32::new(1)),
        }
    }
}

impl PlanningFrequencyDomainDispatchState {
    fn clear_active(&mut self) {
        self.active_request_id = 0;
        self.next_stage = 0;
        self.active_started = None;
        self.active_method_preprocess_ms = 0.0;
        self.active_command_submission_ms = 0.0;
        self.active_elements.clear();
        self.active_invalid_candidates.clear();
        self.active_maximum_elements_per_candidate = 0;
        self.active_build_spectrum = false;
        self.active_build_basis_spectrum = false;
        self.active_map_scheduled = false;
        self.active_verification_targets.clear();
        self.active_uniform = None;
        self.active_bind_groups.clear();
        self.active_timestamps = None;
    }
}

fn dispatch_planning_frequency_domain(
    planning: Res<crate::gpu::planning::ExtractedPlanningInput>,
    shared: Res<crate::gpu::planning::PlanningSharedGpuBuffers>,
    pipelines: Option<Res<FrequencyDomainComputePipeline>>,
    reduction: Res<crate::gpu::planning_reduction::PlanningReductionPipeline>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    channel: Res<PlanningGpuReadbackChannel>,
    mut state: ResMut<PlanningFrequencyDomainDispatchState>,
    timestamp_pool: Res<crate::gpu::planning_timestamps::PlanningTimestampPool>,
) {
    let request = &planning.request;
    let request_matches = request.method == Some(ActiveGravityMethod::FrequencyDomain)
        && planning.payload.method == request.method
        && planning.payload.density_model == request.density_model;
    if state.active_request_id != 0
        && (!request_matches || state.active_request_id != request.request_id)
    {
        let map_scheduled = state.active_map_scheduled;
        let timestamps = state.active_timestamps.take();
        state.clear_active();
        if !map_scheduled {
            let in_flight = Arc::clone(&channel.in_flight);
            render_queue.on_submitted_work_done(move || {
                // Submitted stages may still write their query indices. Keep
                // the shared lease until those writes have drained.
                drop(timestamps);
                in_flight.store(false, Ordering::Release);
            });
        }
        return;
    }
    if !request_matches || state.last_request_id == request.request_id {
        return;
    }
    let starting_request = state.active_request_id != request.request_id;
    let Some(batch) = planning
        .batch
        .as_ref()
        .filter(|batch| batch.batch_id == request.batch_id)
    else {
        return;
    };
    let Some(shared) = shared.0.as_ref().filter(|shared| shared.matches(batch)) else {
        report_frequency_domain_block(
            &mut state,
            request.request_id,
            1,
            "shared planning buffers are not fully uploaded",
        );
        return;
    };
    let Some(pipelines) = pipelines else {
        report_frequency_domain_block(
            &mut state,
            request.request_id,
            2,
            "Frequency-domain algorithm planning pipeline resource is unavailable",
        );
        return;
    };
    if let Some(message) = first_planning_pipeline_error(
        &cache,
        &[
            (
                "voxel line samples",
                pipelines.planning_voxel_density_spectrum_id,
            ),
            ("voxel basis spectrum", pipelines.planning_voxel_spectrum_id),
            (
                "spectrum combination",
                pipelines.planning_combine_spectrum_id,
            ),
            (
                "frequency-domain evaluation",
                pipelines.planning_evaluate_id,
            ),
            ("planning reduction", reduction.0),
        ],
    ) {
        fail_planning_frequency_domain(
            &mut state,
            &channel,
            &render_queue,
            request.request_id,
            message,
        );
        return;
    }
    let (
        Some(voxel_density_spectrum_pipeline),
        Some(voxel_spectrum_pipeline),
        Some(combine_spectrum_pipeline),
        Some(evaluate_pipeline),
    ) = (
        cache.get_compute_pipeline(pipelines.planning_voxel_density_spectrum_id),
        cache.get_compute_pipeline(pipelines.planning_voxel_spectrum_id),
        cache.get_compute_pipeline(pipelines.planning_combine_spectrum_id),
        cache.get_compute_pipeline(pipelines.planning_evaluate_id),
    )
    else {
        report_frequency_domain_block(
            &mut state,
            request.request_id,
            3,
            "Frequency-domain algorithm planning shader pipelines are not cached",
        );
        return;
    };
    let Some(reduction_pipeline) = cache.get_compute_pipeline(reduction.0) else {
        report_frequency_domain_block(
            &mut state,
            request.request_id,
            4,
            "planning reduction pipeline is not cached",
        );
        return;
    };
    if planning.source_radius <= 0.0
        || planning.payload.primary.is_empty()
        || planning.payload.secondary.len() != 113 * 16
        || planning.payload.secondary_count != 56
    {
        report_frequency_domain_block(
            &mut state,
            request.request_id,
            5,
            "Frequency-domain algorithm payload prerequisites are incomplete",
        );
        return;
    }
    if starting_request {
        let std::task::Poll::Ready(timestamps) =
            timestamp_pool.acquire(&render_device, &render_queue, 4)
        else {
            return;
        };
        if channel
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            report_frequency_domain_block(
                &mut state,
                request.request_id,
                6,
                "shared planning readback channel is still in flight",
            );
            return;
        }
        state.active_request_id = request.request_id;
        state.active_timestamps = timestamps;
        state.next_stage = 0;
        state.active_started = Some(Instant::now());
        state.active_method_preprocess_ms = 0.0;
        state.active_command_submission_ms = 0.0;
    }
    let method_preprocess_started = Instant::now();
    let samples_per_candidate = batch.samples_per_candidate as usize;
    let global_state_start = request.candidate_start as usize * samples_per_candidate;
    let target_count = request.candidate_count as usize * samples_per_candidate;
    let batch_changed = state.batch_id != batch.batch_id;
    if batch_changed {
        let positions = batch
            .reference_states
            .iter()
            .map(|state| state.body_position())
            .collect::<Vec<_>>();
        let times = batch
            .reference_states
            .iter()
            .map(|state| state.position_time[3])
            .collect::<Vec<_>>();
        state.canonical_elements = build_known_trajectory_elements(
            &positions,
            &times,
        );
        let mut expected_offset = 0_u32;
        let complete = state.canonical_elements.iter().all(|element| {
            let contiguous = element.target_offset == expected_offset;
            expected_offset = expected_offset.saturating_add(element.target_count);
            contiguous
        }) && expected_offset == batch.samples_per_candidate;
        if !complete || state.canonical_elements.is_empty() {
            error!(
                target: "planning::frequency_domain",
                expected = batch.samples_per_candidate,
                covered = expected_offset,
                "frequency-domain trajectory samples did not cover the frozen reference trajectory"
            );
            state.clear_active();
            channel.in_flight.store(false, Ordering::Release);
            return;
        }
        state.spectrum = None;
        state.density_spectra = None;
        state.sources = None;
        state.source_groups = None;
        state.spectrum_ready = false;
        state.basis_spectrum_ready = false;
    }
    let (elements, maximum_elements_per_candidate, invalid_candidates) = if !starting_request {
        (
            state.active_elements.clone(),
            state.active_maximum_elements_per_candidate,
            state.active_invalid_candidates.clone(),
        )
    } else {
        let mut elements =
            Vec::with_capacity(request.candidate_count as usize * state.canonical_elements.len());
        let mut invalid_candidates = Vec::new();
        for local_candidate in 0..request.candidate_count as usize {
            let start = global_state_start + local_candidate * samples_per_candidate;
            let end = start + samples_per_candidate;
            let local_offset = local_candidate as u32 * batch.samples_per_candidate;
            let candidate_states = &batch.states[start..end];
            let covered = candidate_states
                .iter()
                .all(|candidate| candidate.body_position().is_finite());
            if !covered {
                invalid_candidates.push(local_candidate as u32);
                continue;
            }
            for canonical in &state.canonical_elements {
                let mut element = *canonical;
                element.target_offset += local_offset;
                elements.push(element);
            }
        }
        (
            elements,
            state.canonical_elements.len() as u32,
            invalid_candidates,
        )
    };
    if starting_request {
        state.active_elements.clone_from(&elements);
        state
            .active_invalid_candidates
            .clone_from(&invalid_candidates);
        state.active_maximum_elements_per_candidate = maximum_elements_per_candidate;
    }
    let common_changed = batch_changed || state.quadrature.is_none();
    if common_changed {
        state.batch_id = batch.batch_id;
        state.quadrature = Some(
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("planning_frequency_domain_quadrature"),
                contents: &reciprocal_space_quadrature_bytes(planning.source_radius),
                usage: BufferUsages::STORAGE,
            }),
        );
    }
    if batch_changed || state.sources.is_none() {
        state.sources = Some(
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("planning_frequency_domain_volume_sources"),
                contents: &planning.payload.primary,
                usage: BufferUsages::STORAGE,
            }),
        );
        state.basis_spectrum_ready = false;
        state.spectrum_ready = false;
    }
    let canonical_count = state.canonical_elements.len().max(1) as u64;
    if state.payload_request_id != planning.payload.request_id || state.source_groups.is_none() {
        state.payload_request_id = planning.payload.request_id;
        state.spectrum_ready = false;
        let mut metadata = vec![0_u8; 544 * 16];
        metadata[..planning.payload.secondary.len()].copy_from_slice(&planning.payload.secondary);
        let low_basis_offset_vec4 =
            (QUADRATURE_COUNT as u64 * canonical_count * 56) as f32;
        metadata[112 * 16..112 * 16 + 4].copy_from_slice(&low_basis_offset_vec4.to_le_bytes());
        state.source_groups = Some(
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("planning_frequency_domain_source_groups_and_densities"),
                contents: &metadata,
                usage: BufferUsages::UNIFORM,
            }),
        );
    }
    if state.spectrum.is_none() {
        state.spectrum = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_frequency_domain_spectrum"),
            size: FREQUENCY_COUNT as u64 * 32 * canonical_count,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    if state.density_spectra.is_none() {
        let line_sample_bytes =
            QUADRATURE_COUNT as u64 * 16 * canonical_count * 56;
        let basis_bank_bytes =
            FREQUENCY_COUNT as u64 * 16 * canonical_count * 28;
        state.density_spectra = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_frequency_domain_voxel_lines_and_low_basis"),
            size: line_sample_bytes + basis_bank_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    let basis_bank_size = FREQUENCY_COUNT as u64 * 16 * canonical_count * 28;
    let output_size = target_count as u64 * OUTPUT_BYTES;
    let verification_targets = if starting_request {
        // Use exactly the same verification population as MMFFT/FMM.  An
        // Frequency-domain algorithm rejection is a measured failure, not a reason to shrink the
        // denominator after seeing the result.
        let targets = crate::gpu::planning_reduction::planning_verification_targets(request, batch);
        state.active_verification_targets.clone_from(&targets);
        targets
    } else {
        state.active_verification_targets.clone()
    };
    let metric_size = u64::from(request.candidate_count) * 16;
    // Six contiguous evaluator rows plus the two non-contiguous FD scan rows.
    let data_size = metric_size + verification_targets.len() as u64 * 8 * 16;
    // At most four method passes, two u64 timestamps per pass.
    let staging_size = data_size + 64;
    let output_storage_size = 90_112_u64 * 16 + basis_bank_size;
    if state.output_storage_size != output_storage_size || state.output.is_none() {
        state.output_storage_size = output_storage_size;
        state.output = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_frequency_domain_output_and_high_basis"),
            size: output_storage_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        state.basis_spectrum_ready = false;
        state.spectrum_ready = false;
    }
    state.output_size = output_size;
    let baseline_size = batch.state_count() as u64 * 16;
    if state.baseline_size != baseline_size || state.baseline.is_none() {
        state.baseline_size = baseline_size;
        state.baseline = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_frequency_domain_baseline"),
            size: baseline_size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    if state.metric_size != metric_size || state.metrics.is_none() {
        state.metric_size = metric_size;
        state.metrics = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_frequency_domain_metrics"),
            size: metric_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    if state.staging_size != staging_size || state.staging.is_none() {
        state.staging_size = staging_size;
        state.staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_frequency_domain_staging"),
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    let sources = state
        .sources
        .as_ref()
        .expect("planning Frequency-domain algorithm sources")
        .clone();
    let quadrature = state
        .quadrature
        .as_ref()
        .expect("planning Frequency-domain algorithm quadrature")
        .clone();
    let spectrum = state
        .spectrum
        .as_ref()
        .expect("planning Frequency-domain algorithm spectrum")
        .clone();
    let density_spectra = state
        .density_spectra
        .as_ref()
        .expect("planning Frequency-domain algorithm samples")
        .clone();
    let source_groups = state
        .source_groups
        .as_ref()
        .expect("planning Frequency-domain algorithm source groups")
        .clone();
    let output = state
        .output
        .as_ref()
        .expect("planning Frequency-domain algorithm output")
        .clone();
    let baseline = state
        .baseline
        .as_ref()
        .expect("planning Frequency-domain algorithm baseline")
        .clone();
    let metrics = state
        .metrics
        .as_ref()
        .expect("planning Frequency-domain algorithm metrics")
        .clone();
    let staging = state
        .staging
        .as_ref()
        .expect("planning Frequency-domain algorithm staging")
        .clone();
    if state.layout.is_none() {
        state.layout = Some(render_device.create_bind_group_layout(
            "planning_frequency_domain_bgl",
            &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
                storage_rw_entry(4),
                storage_rw_entry(6),
                uniform_entry(7),
                storage_ro_entry(9),
            ],
        ));
    }
    let layout = state
        .layout
        .as_ref()
        .expect("planning Frequency-domain algorithm layout")
        .clone();
    state.active_method_preprocess_ms += method_preprocess_started.elapsed().as_secs_f64() * 1.0e3;
    if starting_request {
        state.active_build_spectrum = !state.spectrum_ready;
        state.active_build_basis_spectrum = !state.basis_spectrum_ready;
        let uniform_size = 48_u64;
        let mut uniform_data = vec![0_u8; uniform_size as usize * 256];
        if elements.len() > 256 {
            error!(
                target: "planning::frequency_domain",
                evaluator_elements = elements.len(),
                "canonical Frequency-domain algorithm evaluator exceeded the 256-element uniform capacity"
            );
            state.clear_active();
            channel.in_flight.store(false, Ordering::Release);
            return;
        }
        for (element_index, element) in elements.iter().enumerate() {
            let mut bytes = uniform_bytes(
                element.trajectory_origin,
                planning.payload.item_count,
                element.spectrum_index,
                3,
                element.target_count,
                element.target_offset,
            );
            bytes[28..32].copy_from_slice(&(global_state_start as u32).to_le_bytes());
            let offset = element_index * uniform_size as usize;
            uniform_data[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_frequency_domain_uniforms"),
            contents: &uniform_data,
            usage: BufferUsages::STORAGE,
        });
        let bind_group = render_device.create_bind_group(
            "planning_frequency_domain_batch_bg",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sources.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: quadrature.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: spectrum.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: density_spectra.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: source_groups.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: shared.positions.as_entire_binding(),
                },
            ],
        );
        state.active_bind_groups = vec![bind_group];
        state.active_uniform = Some(uniform);
    }
    let build_spectrum = state.active_build_spectrum;
    let build_basis_spectrum = state.active_build_basis_spectrum;
    // Geometry spectra are built once, combined with the selected voxel
    // densities, and evaluated at every known trajectory sample.
    let total_stages = if build_basis_spectrum {
        4
    } else if build_spectrum {
        2
    } else {
        1
    };
    if starting_request && let Some(queries) = &mut state.active_timestamps {
        queries.set_pass_count(total_stages as u32);
    }
    let stage_budget = planning_frequency_domain_stage_budget(
        request.compute_benchmark,
        total_stages,
        state.adaptive_stage_budget.load(Ordering::Acquire) as usize,
    );
    if stage_budget == 0 {
        return;
    }
    let stage_end = (state.next_stage + stage_budget).min(total_stages);
    let final_submission = stage_end == total_stages;
    let _uniform = state.active_uniform.as_ref();
    let bind_groups = state.active_bind_groups.clone();
    let encode_started = Instant::now();
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("planning_frequency_domain_paced_encoder"),
    });
    if state.next_stage == 0 {
        // Preserve the high voxel-basis bank stored after the fixed evaluator
        // prefix while clearing only the target rows for this request.
        encoder.clear_buffer(&output, 0, Some(output_size));
    }
    let canonical_trajectory_block_count = state.canonical_elements.len() as u32;
    let evaluator_element_count = elements.len() as u32;
    for stage in state.next_stage..stage_end {
        let physical_stage = if build_basis_spectrum {
            [0, 1, 3, 4][stage]
        } else if build_spectrum {
            [3, 4][stage]
        } else {
            4
        };
        let (label, pipeline, width, height, depth) = match physical_stage {
            0 => (
                "planning_frequency_domain_voxel_density_spectrum",
                voxel_density_spectrum_pipeline,
                QUADRATURE_COUNT,
                canonical_trajectory_block_count,
                56,
            ),
            1 => (
                "planning_frequency_domain_voxel_basis_spectrum",
                voxel_spectrum_pipeline,
                QUADRATURE_COUNT.div_ceil(64),
                canonical_trajectory_block_count,
                56,
            ),
            3 => (
                "planning_frequency_domain_combine_voxel_spectrum",
                combine_spectrum_pipeline,
                QUADRATURE_COUNT.div_ceil(64),
                canonical_trajectory_block_count,
                1,
            ),
            _ => {
                let (width, height) = (1, batch.samples_per_candidate);
                (
                    "planning_frequency_domain_evaluate",
                    evaluate_pipeline,
                    width,
                    height,
                    evaluator_element_count,
                )
            }
        };
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: state
                .active_timestamps
                .as_ref()
                .map(|queries| queries.writes(stage as u32)),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_groups[0], &[]);
        pass.dispatch_workgroups(width, height, depth);
    }
    let reduction_resources = final_submission.then(|| {
        let resources = crate::gpu::planning_reduction::encode_planning_reduction(
            &render_device,
            &mut encoder,
            reduction_pipeline,
            request,
            batch,
            OUTPUT_ROWS_PER_BLOCK as u32,
            &output,
            &shared.positions,
            &baseline,
            &metrics,
        );
        encoder.copy_buffer_to_buffer(&metrics, 0, &staging, 0, metric_size);
        for (compact_index, target_index) in verification_targets.iter().copied().enumerate() {
            let compact_offset = metric_size + compact_index as u64 * 8 * 16;
            encoder.copy_buffer_to_buffer(
                &output,
                u64::from(target_index) * OUTPUT_BYTES,
                &staging,
                compact_offset,
                6 * 16,
            );
            encoder.copy_buffer_to_buffer(
                &output,
                u64::from(target_index) * OUTPUT_BYTES + 9 * 16,
                &staging,
                compact_offset + 6 * 16,
                2 * 16,
            );
        }
        resources
    });
    if final_submission && let Some(queries) = &state.active_timestamps {
        queries.resolve_into(&mut encoder, &staging, data_size);
    }
    render_queue.submit([encoder.finish()]);
    state.active_command_submission_ms += encode_started.elapsed().as_secs_f64() * 1.0e3;
    state.next_stage = stage_end;
    if !final_submission {
        return;
    }
    if build_spectrum {
        state.spectrum_ready = true;
    }
    if build_basis_spectrum {
        state.basis_spectrum_ready = true;
    }
    drop(reduction_resources);
    state.last_request_id = request.request_id;
    let method_preprocess_ms = state.active_method_preprocess_ms;
    let command_submission_ms = state.active_command_submission_ms;
    let request_wall_started = state.active_started.unwrap_or_else(Instant::now);
    let packet_request = request.clone();
    let shared_data = Arc::clone(&channel.data);
    let in_flight = Arc::clone(&channel.in_flight);
    let mapped = staging.clone();
    let canonical_element_count = state.canonical_elements.len() as u32;
    let evaluator_element_count = elements.len() as u32;
    let reported_trajectory_block_count = maximum_elements_per_candidate;
    let source_count = planning.payload.item_count;
    let state_indices = verification_targets;
    let request_candidate_count = request.candidate_count;
    let samples_per_candidate = batch.samples_per_candidate;
    let invalid_candidates_for_readback = invalid_candidates;
    // Exact project-owned WebGPU buffer bytes for this request.  Driver-private
    // allocations and the target buffer shared by all planning methods are not
    // observable here and are intentionally excluded.
    let estimated_gpu_buffer_bytes = planning.payload.primary.len() as u64
        + QUADRATURE_COUNT as u64 * 16
        + 2 * 544 * 16
        + FREQUENCY_COUNT as u64 * 32 * canonical_count
        + QUADRATURE_COUNT as u64 * 16 * canonical_count * 56
        + basis_bank_size
        + output_storage_size
        + baseline_size
        + metric_size
        + staging_size
        + 48 * 256;
    info!(
        target: "planning::frequency_domain",
        request_id = request.request_id,
        candidate_count = request.candidate_count,
        target_count,
        trajectory_blocks = canonical_element_count,
        evaluator_elements = evaluator_element_count,
        spectrum_cache_hit = !build_spectrum,
        voxel_basis_cache_hit = !build_basis_spectrum,
        source_parallel_lanes = 128,
        refined_source_count = batch.source_count,
        payload_source_count = source_count,
        payload_source_ratio = f64::from(source_count) / f64::from(batch.source_count.max(1)),
        voxel_basis_count = 56,
        estimated_gpu_buffer_bytes,
        maximum_elements_per_candidate = reported_trajectory_block_count,
        "Frequency-domain algorithm: reciprocal-space density basis shared across known trajectory samples"
    );
    let timestamps = state.active_timestamps.take();
    let adaptive_stage_budget = Arc::clone(&state.adaptive_stage_budget);
    state.clear_active();
    state.active_map_scheduled = true;
    mapped
        .slice(..staging_size)
        .map_async(MapMode::Read, move |result| {
            let request_wall_ms = request_wall_started.elapsed().as_secs_f64() * 1.0e3;
            let gpu_completion_map_ms =
                (request_wall_ms - method_preprocess_ms - command_submission_ms).max(0.0);
            let decode_started = bevy::platform::time::Instant::now();
            let mut pass_ms = None;
            let rows = if result.is_ok() {
                let view = staging.slice(..staging_size).get_mapped_range();
                pass_ms = timestamps
                    .as_ref()
                    .and_then(|queries| queries.decode(&view[data_size as usize..]));
                if let Some(values) = pass_ms.as_ref() {
                    let maximum_ms = values.iter().copied().fold(0.0_f64, f64::max);
                    let next_budget = if maximum_ms > PLANNING_GPU_MAX_SUBMISSION_MS
                        || maximum_ms > PLANNING_GPU_TARGET_SUBMISSION_MS
                    {
                        PLANNING_GPU_MIN_STAGE_BUDGET
                    } else if maximum_ms < PLANNING_GPU_TARGET_SUBMISSION_MS * 0.45 {
                        PLANNING_GPU_MAX_STAGE_BUDGET
                    } else {
                        PLANNING_GPU_MIN_STAGE_BUDGET
                    };
                    adaptive_stage_budget.store(next_budget as u32, Ordering::Release);
                }
                let full_rows = bytes_to_f32x4(&view[..data_size as usize]);
                let mut candidate_metrics = full_rows[..request_candidate_count as usize].to_vec();
                let mut rejected_candidates = vec![false; request_candidate_count as usize];
                for (candidate, metric) in candidate_metrics.iter_mut().enumerate() {
                    if metric[0] < 0.0 || metric.iter().copied().any(|value| !value.is_finite()) {
                        *metric = [-1.0, 0.0, 0.0, 0.0];
                        rejected_candidates[candidate] = true;
                    }
                }
                for candidate in &invalid_candidates_for_readback {
                    if let Some(metric) = candidate_metrics.get_mut(*candidate as usize) {
                        *metric = [-1.0, 0.0, 0.0, 0.0];
                        rejected_candidates[*candidate as usize] = true;
                    }
                }
                let compact_rows = &full_rows[request_candidate_count as usize..];
                let mut valid_targets = Vec::with_capacity(state_indices.len());
                for (target_index, target_rows) in state_indices.iter().copied().zip(
                    compact_rows
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .take(state_indices.len()),
                ) {
                    let finite = [
                        target_rows[0],
                        target_rows[2],
                        target_rows[3],
                        target_rows[4],
                    ]
                    .into_iter()
                    .flatten()
                    .all(f32::is_finite);
                    let local_candidate = target_index / samples_per_candidate;
                    let valid = finite
                        && !invalid_candidates_for_readback.contains(&local_candidate);
                    valid_targets.push(valid);
                    if !valid {
                        let candidate = local_candidate;
                        if let Some(rejected) = rejected_candidates.get_mut(candidate as usize) {
                            *rejected = true;
                        }
                        if let Some(metric) = candidate_metrics.get_mut(candidate as usize) {
                            *metric = [-1.0, 0.0, 0.0, 0.0];
                        }
                    }
                }
                let mut common_indices = Vec::with_capacity(state_indices.len());
                let mut rows = Vec::with_capacity(state_indices.len() * 4);
                let mut raw_rows = Vec::with_capacity(state_indices.len() * 4);
                for ((target_index, target_rows), target_valid) in state_indices
                    .iter()
                    .copied()
                    .zip(
                        compact_rows
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .take(state_indices.len()),
                    )
                    .zip(valid_targets)
                {
                    let candidate = target_index / samples_per_candidate;
                    common_indices.push(target_index);
                    raw_rows.extend([
                        target_rows[0],
                        target_rows[2],
                        target_rows[3],
                        target_rows[4],
                    ]);
                    if target_valid && !rejected_candidates[candidate as usize] {
                        rows.extend([
                            target_rows[0],
                            target_rows[2],
                            target_rows[3],
                            target_rows[4],
                        ]);
                    } else {
                        // Keep the point and charge the rejected candidate a full
                        // reference-relative error instead of survivor-biasing
                        // Frequency-domain algorithm by silently omitting it.
                        rows.extend([[0.0; 4]; 4]);
                    }
                }
                let rejected_sample_count = state_indices
                    .iter()
                    .filter(|target_index| {
                        rejected_candidates[(**target_index / samples_per_candidate) as usize]
                    })
                    .count() as u64;
                drop(view);
                staging.unmap();
                (
                    rows,
                    raw_rows,
                    candidate_metrics,
                    common_indices,
                    rejected_sample_count,
                )
            } else {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    0,
                )
            };
            if let Ok(mut guard) = shared_data.lock() {
                *guard = Some(PlanningGpuPacket {
                    request: packet_request,
                    state_indices: rows.3,
                    rows: rows.0,
                    raw_rows: rows.1,
                    candidate_metrics: rows.2,
                    rejected_sample_count: rows.4,
                    readback_valid: result.is_ok(),
                    timing: PlanningGpuTiming {
                        method_preprocess_ms,
                        command_submission_ms,
                        gpu_completion_map_ms,
                        readback_decode_ms: decode_started.elapsed().as_secs_f64() * 1.0e3,
                        kernel_ms: pass_ms.as_ref().map(|values| values.iter().sum()),
                        evaluation_kernel_ms: pass_ms
                            .as_ref()
                            .and_then(|values| values.last().copied()),
                        basis_kernel_ms: pass_ms.as_ref().map(|values| {
                            if build_basis_spectrum {
                                values[..2].iter().sum()
                            } else {
                                0.0
                            }
                        }),
                        // First geometry: voxel density spectrum
                        // + combine + evaluate + reduction. New density: combine
                        // + evaluate + reduction.
                        dispatch_count: if build_basis_spectrum {
                            5
                        } else if build_spectrum {
                            3
                        } else {
                            2
                        },
                        forward_kernel_evaluations: u64::from(build_basis_spectrum)
                            * u64::from(source_count)
                            * u64::from(QUADRATURE_COUNT)
                            * u64::from(canonical_element_count)
                            + (u64::from(build_basis_spectrum) * u64::from(QUADRATURE_COUNT)
                                + u64::from(build_spectrum))
                                * 56
                                * u64::from(QUADRATURE_COUNT)
                                * u64::from(canonical_element_count)
                            + target_count as u64
                                * u64::from(QUADRATURE_COUNT),
                        trajectory_block_count: reported_trajectory_block_count,
                    },
                    backend: PlanningExecutionBackend::GpuFrequencyDomain,
                });
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn report_frequency_domain_block(
    state: &mut PlanningFrequencyDomainDispatchState,
    request_id: u64,
    code: u8,
    reason: &'static str,
) {
    if request_id == 0 || state.last_block_key == Some((request_id, code)) {
        return;
    }
    state.last_block_key = Some((request_id, code));
    warn!(target: "planning::frequency_domain", request_id, code, reason);
}

fn first_planning_pipeline_error(
    cache: &PipelineCache,
    pipelines: &[(&str, CachedComputePipelineId)],
) -> Option<String> {
    pipelines.iter().find_map(|(name, pipeline_id)| {
        let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(*pipeline_id) else {
            return None;
        };
        if matches!(
            error,
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable
        ) {
            return None;
        }
        Some(format!(
            "Frequency-domain algorithm {name} GPU pipeline failed to compile: {error}"
        ))
    })
}

fn fail_planning_frequency_domain(
    state: &mut PlanningFrequencyDomainDispatchState,
    channel: &PlanningGpuReadbackChannel,
    queue: &RenderQueue,
    request_id: u64,
    message: String,
) {
    error!(target: "planning::frequency_domain", request_id, error = %message);
    let timestamps = state.active_timestamps.take();
    let has_submitted_stages = state.next_stage > 0;
    state.clear_active();
    state.last_request_id = request_id;
    if has_submitted_stages {
        let in_flight = Arc::clone(&channel.in_flight);
        queue.on_submitted_work_done(move || {
            drop(timestamps);
            in_flight.store(false, Ordering::Release);
        });
    } else {
        drop(timestamps);
        channel.in_flight.store(false, Ordering::Release);
    }
    if let Ok(mut slot) = channel.error.try_lock()
        && slot.is_none()
    {
        *slot = Some((request_id, message));
    }
}

fn planning_frequency_domain_stage_budget(
    compute_benchmark: bool,
    total_stages: usize,
    adaptive_budget: usize,
) -> usize {
    if compute_benchmark {
        return adaptive_budget
            .clamp(PLANNING_GPU_MIN_STAGE_BUDGET, PLANNING_GPU_MAX_STAGE_BUDGET)
            .min(total_stages.max(1));
    }
    let average_fps = crate::browser_frame_rate();
    let recent_frame_ms = crate::browser_recent_frame_ms();
    let frame_over_budget = average_fps.is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS)
        || recent_frame_ms.is_some_and(|milliseconds| milliseconds > PLANNING_MAX_RECENT_FRAME_MS);
    if frame_over_budget {
        return 0;
    }
    // Reserved for non-benchmark diagnostic callers. Both user-selectable
    // planning profiles use the full-stage fairness path above.
    2.min(total_stages.max(1))
}
