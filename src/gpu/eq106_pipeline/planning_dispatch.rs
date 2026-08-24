#[derive(Resource, Default)]
struct PlanningEq106DispatchState {
    last_request_id: u64,
    active_request_id: u64,
    next_stage: usize,
    active_started: Option<Instant>,
    active_method_preprocess_ms: f64,
    active_command_submission_ms: f64,
    active_elements: Vec<Eq106BatchElement>,
    active_invalid_candidates: Vec<u32>,
    active_maximum_elements_per_candidate: u32,
    active_build_spectrum: bool,
    active_verification_targets: Vec<u32>,
    active_uniform: Option<Buffer>,
    active_bind_groups: Vec<BindGroup>,
    batch_id: u64,
    payload_request_id: u64,
    output_size: u64,
    staging_size: u64,
    baseline_size: u64,
    metric_size: u64,
    layout: Option<BindGroupLayout>,
    sources: Option<Buffer>,
    quadrature: Option<Buffer>,
    operator: Option<Buffer>,
    psi: Option<Buffer>,
    dummy_modes: Option<Buffer>,
    spectrum: Option<Buffer>,
    line_samples: Option<Buffer>,
    output: Option<Buffer>,
    baseline: Option<Buffer>,
    metrics: Option<Buffer>,
    staging: Option<Buffer>,
    canonical_elements: Vec<Eq106BatchElement>,
    spectrum_ready: bool,
}

impl PlanningEq106DispatchState {
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
        self.active_verification_targets.clear();
        self.active_uniform = None;
        self.active_bind_groups.clear();
    }
}

fn dispatch_planning_eq106(
    planning: Res<crate::gpu::planning::ExtractedPlanningInput>,
    shared: Res<crate::gpu::planning::PlanningSharedGpuBuffers>,
    pipelines: Option<Res<Eq106ComputePipeline>>,
    reduction: Res<crate::gpu::planning_reduction::PlanningReductionPipeline>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    channel: Res<PlanningGpuReadbackChannel>,
    mut state: ResMut<PlanningEq106DispatchState>,
) {
    let request = &planning.request;
    let request_matches = request.method == Some(ActiveGravityMethod::CurvedArcEq106)
        && planning.payload.method == request.method
        && planning.payload.density_model == request.density_model;
    if state.active_request_id != 0
        && (!request_matches || state.active_request_id != request.request_id)
    {
        state.clear_active();
        let in_flight = Arc::clone(&channel.in_flight);
        render_queue.on_submitted_work_done(move || {
            in_flight.store(false, Ordering::Release);
        });
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
    let Some(shared) = shared
        .0
        .as_ref()
        .filter(|shared| shared.matches(batch))
    else {
        return;
    };
    let Some(pipelines) = pipelines else { return };
    let (Some(line_samples_pipeline), Some(assemble_pipeline), Some(evaluate_pipeline)) = (
        cache.get_compute_pipeline(pipelines.line_samples_id),
        cache.get_compute_pipeline(pipelines.assemble_id),
        cache.get_compute_pipeline(pipelines.evaluate_id),
    ) else {
        return;
    };
    let Some(reduction_pipeline) = cache.get_compute_pipeline(reduction.0) else {
        return;
    };
    if planning.eq106_operator.is_empty()
        || planning.eq106_psi.is_empty()
        || planning.source_radius <= 0.0
        || planning.payload.primary.is_empty()
    {
        return;
    }
    if starting_request {
        if channel
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        state.active_request_id = request.request_id;
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
        let velocities = batch
            .reference_states
            .iter()
            .map(|state| state.body_velocity())
            .collect::<Vec<_>>();
        let times = batch
            .reference_states
            .iter()
            .map(|state| state.position_time[3])
            .collect::<Vec<_>>();
        state.canonical_elements = build_canonical_tube_elements(
            &positions,
            &velocities,
            &times,
            planning.source_radius,
            4.0 * planning.source_radius,
            PLANNING_TRAJECTORY_TUBE_RADIUS_METERS,
        );
        let mut expected_offset = 0_u32;
        let complete = state.canonical_elements.iter().all(|element| {
            let contiguous = element.target_offset == expected_offset;
            expected_offset = expected_offset.saturating_add(element.target_count);
            contiguous
        }) && expected_offset == batch.samples_per_candidate;
        if !complete || state.canonical_elements.is_empty() {
            error!(
                target: "planning::eq106",
                expected = batch.samples_per_candidate,
                covered = expected_offset,
                "canonical Eq.106 tube segmentation did not cover the frozen centre arc"
            );
            state.clear_active();
            channel.in_flight.store(false, Ordering::Release);
            return;
        }
        state.spectrum = None;
        state.line_samples = None;
        state.spectrum_ready = false;
    }
    let (elements, maximum_elements_per_candidate, invalid_candidates) = if !starting_request {
        (
            state.active_elements.clone(),
            state.active_maximum_elements_per_candidate,
            state.active_invalid_candidates.clone(),
        )
    } else {
        let mut elements = Vec::with_capacity(
            request.candidate_count as usize * state.canonical_elements.len(),
        );
        let mut invalid_candidates = Vec::new();
        for local_candidate in 0..request.candidate_count as usize {
            let start = global_state_start + local_candidate * samples_per_candidate;
            let end = start + samples_per_candidate;
            let local_offset = local_candidate as u32 * batch.samples_per_candidate;
            let candidate_states = &batch.states[start..end];
            let covered = state.canonical_elements.iter().all(|element| {
                let range_start = element.target_offset as usize;
                let range_end = range_start + element.target_count as usize;
                candidate_states[range_start..range_end].iter().all(|candidate| {
                    canonical_element_accepts(
                        element,
                        candidate.body_position(),
                        planning.source_radius,
                    )
                })
            });
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
    let common_changed = batch_changed
        || state.quadrature.is_none()
        || state.operator.is_none()
        || state.psi.is_none()
        || state.dummy_modes.is_none();
    if common_changed {
        state.batch_id = batch.batch_id;
        state.quadrature = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_quadrature"),
            contents: &half_line_quadrature_bytes(0.5 * planning.source_radius.max(1.0)),
            usage: BufferUsages::STORAGE,
        }));
        state.operator = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_operator"),
            contents: &planning.eq106_operator,
            usage: BufferUsages::UNIFORM,
        }));
        state.psi = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_psi"),
            contents: &planning.eq106_psi,
            usage: BufferUsages::STORAGE,
        }));
        state.dummy_modes = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_unused_modes"),
            contents: &vec![0_u8; 544 * 16],
            usage: BufferUsages::UNIFORM,
        }));
    }
    if state.payload_request_id != planning.payload.request_id || state.sources.is_none() {
        state.payload_request_id = planning.payload.request_id;
        state.spectrum_ready = false;
        state.sources = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_sources"),
            contents: &planning.payload.primary,
            usage: BufferUsages::STORAGE,
        }));
    }
    let coefficient_count = taylor_coefficient_count(TAYLOR_MAX_ORDER) as u64;
    let canonical_count = state.canonical_elements.len().max(1) as u64;
    if state.spectrum.is_none() {
        state.spectrum = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_spectrum"),
            size: coefficient_count * FREQUENCY_COUNT as u64 * 32 * canonical_count,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    if state.line_samples.is_none() {
        state.line_samples = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_line_samples"),
            size: coefficient_count * QUADRATURE_COUNT as u64 * 16 * canonical_count,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    let output_size = target_count as u64 * OUTPUT_BYTES;
    let verification_targets = if starting_request {
        // Use exactly the same verification population as MMFFT/FMM.  An
        // Eq.106 rejection is a measured failure, not a reason to shrink the
        // denominator after seeing the result.
        let targets =
            crate::gpu::planning_reduction::planning_verification_targets(request, batch);
        state.active_verification_targets.clone_from(&targets);
        targets
    } else {
        state.active_verification_targets.clone()
    };
    let metric_size = u64::from(request.candidate_count) * 16;
    // Six contiguous evaluator rows plus the two non-contiguous FD scan rows.
    let staging_size = metric_size + verification_targets.len() as u64 * 8 * 16;
    if state.output_size != output_size || state.output.is_none() {
        state.output_size = output_size;
        state.output = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    let baseline_size = batch.state_count() as u64 * 16;
    if state.baseline_size != baseline_size || state.baseline.is_none() {
        state.baseline_size = baseline_size;
        state.baseline = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_baseline"),
            size: baseline_size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    if state.metric_size != metric_size || state.metrics.is_none() {
        state.metric_size = metric_size;
        state.metrics = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_metrics"),
            size: metric_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    if state.staging_size != staging_size || state.staging.is_none() {
        state.staging_size = staging_size;
        state.staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_staging"),
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    let sources = state.sources.as_ref().expect("planning Eq.106 sources").clone();
    let quadrature = state.quadrature.as_ref().expect("planning Eq.106 quadrature").clone();
    let operator = state.operator.as_ref().expect("planning Eq.106 operator").clone();
    let psi = state.psi.as_ref().expect("planning Eq.106 psi").clone();
    let dummy_modes = state.dummy_modes.as_ref().expect("planning Eq.106 modes").clone();
    let spectrum = state.spectrum.as_ref().expect("planning Eq.106 spectrum").clone();
    let line_samples = state.line_samples.as_ref().expect("planning Eq.106 samples").clone();
    let output = state.output.as_ref().expect("planning Eq.106 output").clone();
    let baseline = state.baseline.as_ref().expect("planning Eq.106 baseline").clone();
    let metrics = state.metrics.as_ref().expect("planning Eq.106 metrics").clone();
    let staging = state.staging.as_ref().expect("planning Eq.106 staging").clone();
    if state.layout.is_none() {
        state.layout = Some(render_device.create_bind_group_layout(
            "planning_eq106_bgl",
            &[
                storage_ro_entry(0), storage_ro_entry(1), storage_ro_entry(2), storage_rw_entry(3),
                storage_rw_entry(4), uniform_entry(5), storage_rw_entry(6), uniform_entry(7),
                storage_ro_entry(8), storage_ro_entry(9),
            ],
        ));
    }
    let layout = state.layout.as_ref().expect("planning Eq.106 layout").clone();
    state.active_method_preprocess_ms +=
        method_preprocess_started.elapsed().as_secs_f64() * 1.0e3;
    if starting_request {
        state.active_build_spectrum = !state.spectrum_ready;
        let uniform_size = 96_u64;
        let mut uniform_data = vec![0_u8; uniform_size as usize * 256];
        if elements.len() > 256 {
            error!(
                target: "planning::eq106",
                evaluator_elements = elements.len(),
                "canonical Eq.106 evaluator exceeded the 256-element uniform capacity"
            );
            state.clear_active();
            channel.in_flight.store(false, Ordering::Release);
            return;
        }
        for (element_index, element) in elements.iter().enumerate() {
            let mut bytes = uniform_bytes(
                element.line_origin,
                element.line_origin,
                element.line_direction,
                planning.payload.item_count,
                planning.source_radius,
                element.line_limit,
                element.taylor_order,
                0,
                element.spectrum_index,
                request.eq106_certified,
                false,
                element.target_count,
                element.target_offset,
            );
            bytes[92..96].copy_from_slice(&(global_state_start as u32).to_le_bytes());
            let offset = element_index * uniform_size as usize;
            uniform_data[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_uniforms"),
            contents: &uniform_data,
            usage: BufferUsages::STORAGE,
        });
        let bind_group = render_device.create_bind_group(
            "planning_eq106_batch_bg",
            &layout,
            &[
                BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: sources.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: quadrature.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: spectrum.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: operator.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: line_samples.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: dummy_modes.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: psi.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: shared.positions.as_entire_binding() },
            ],
        );
        state.active_bind_groups = vec![bind_group];
        state.active_uniform = Some(uniform);
    }
    let build_spectrum = state.active_build_spectrum;
    // The planning benchmark deliberately evaluates the coherent spectrum assembled from the
    // sampled line.  Mixing in the separate analytic zero-order correction makes coefficient 0
    // follow a different discretisation from coefficients 1..A, which is especially harmful for
    // the spatial derivative used by the gradient benchmark.
    let total_stages = if build_spectrum { 3 } else { 1 };
    let stage_budget = planning_eq106_stage_budget(request.compute_benchmark, total_stages);
    if stage_budget == 0 {
        return;
    }
    let stage_end = (state.next_stage + stage_budget).min(total_stages);
    let final_submission = stage_end == total_stages;
    let _uniform = state.active_uniform.as_ref();
    let bind_groups = state.active_bind_groups.clone();
    let encode_started = Instant::now();
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("planning_eq106_paced_encoder"),
    });
    if state.next_stage == 0 {
        encoder.clear_buffer(&output, 0, None);
    }
    let canonical_segment_count = state.canonical_elements.len() as u32;
    let evaluator_element_count = elements.len() as u32;
    let max_targets = elements.iter().map(|element| element.target_count).max().unwrap_or(1);
    for stage in state.next_stage..stage_end {
        let physical_stage = if build_spectrum {
            if stage < 2 { stage } else { 3 }
        } else {
            3
        };
        let (label, pipeline, width, height, depth) = match physical_stage {
            0 => (
                "planning_eq106_line",
                line_samples_pipeline,
                QUADRATURE_COUNT.div_ceil(64),
                canonical_segment_count,
                1,
            ),
            1 => (
                "planning_eq106_spectrum",
                assemble_pipeline,
                (taylor_coefficient_count(TAYLOR_MAX_ORDER) * FREQUENCY_COUNT).div_ceil(64),
                canonical_segment_count,
                1,
            ),
            _ => {
                let (width, height) = target_dispatch_grid(max_targets);
                (
                    "planning_eq106_evaluate",
                    evaluate_pipeline,
                    width,
                    height,
                    evaluator_element_count,
                )
            }
        };
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
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
    render_queue.submit([encoder.finish()]);
    state.active_command_submission_ms += encode_started.elapsed().as_secs_f64() * 1.0e3;
    state.next_stage = stage_end;
    if !final_submission {
        return;
    }
    if build_spectrum {
        state.spectrum_ready = true;
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
    let reported_segment_count = maximum_elements_per_candidate;
    let source_count = planning.payload.item_count;
    let state_indices = verification_targets;
    let request_candidate_count = request.candidate_count;
    let samples_per_candidate = batch.samples_per_candidate;
    let invalid_candidates_for_readback = invalid_candidates;
    info!(
        target: "planning::eq106",
        request_id = request.request_id,
        candidate_count = request.candidate_count,
        target_count,
        canonical_spectral_elements = canonical_element_count,
        evaluator_elements = evaluator_element_count,
        spectrum_cache_hit = !build_spectrum,
        analytic_zero_correction = false,
        maximum_elements_per_candidate = reported_segment_count,
        "Eq.106 canonical centre-arc spectrum is shared across candidates and candidate tiles"
    );
    state.clear_active();
    mapped.slice(..staging_size).map_async(MapMode::Read, move |result| {
        let request_wall_ms = request_wall_started.elapsed().as_secs_f64() * 1.0e3;
        let gpu_completion_map_ms =
            (request_wall_ms - method_preprocess_ms - command_submission_ms).max(0.0);
        let rows = if result.is_ok() {
            let view = staging.slice(..staging_size).get_mapped_range();
            let full_rows = bytes_to_f32x4(&view);
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
            let mut rejection_counts = [0_u64; 6];
            let mut self_fd_step_maxima = [0.0_f32; 5];
            let mut first_rejection = None;
            for (target_index, target_rows) in state_indices.iter().copied().zip(
                compact_rows
                .chunks_exact(8)
                .take(state_indices.len()),
            ) {
                let certificate = target_rows[1];
                let finite = [target_rows[0], target_rows[2], target_rows[3], target_rows[4]]
                    .into_iter()
                    .flatten()
                    .all(f32::is_finite);
                let fd_errors = [
                    target_rows[6][0],
                    target_rows[6][1],
                    target_rows[6][2],
                    target_rows[6][3],
                    target_rows[7][0],
                ];
                for (maximum, error) in self_fd_step_maxima.iter_mut().zip(fd_errors) {
                    if error.is_finite() {
                        *maximum = maximum.max(error);
                    }
                }
                let local_candidate = target_index / samples_per_candidate;
                let outside_tube = invalid_candidates_for_readback.contains(&local_candidate);
                let rejection = if outside_tube {
                    Some(3)
                } else if !finite
                    || certificate.iter().copied().any(|value| !value.is_finite())
                {
                    Some(5)
                } else if certificate[0] > GRAVITY_BENCHMARK_RELATIVE_TOLERANCE {
                    Some(0)
                } else if certificate[1] > GRAVITY_BENCHMARK_RELATIVE_TOLERANCE {
                    Some(1)
                } else if certificate[2] > GRAVITY_BENCHMARK_RELATIVE_TOLERANCE {
                    Some(2)
                } else if certificate[3] > 0.30 {
                    Some(3)
                } else {
                    None
                };
                // The f32 self-FD scan is an internal consistency warning,
                // not an accuracy oracle. Cancellation at small step sizes
                // must not zero otherwise finite fields; f64 truth below
                // remains the common qualification criterion.
                if fd_errors.iter().copied().any(|value| !value.is_finite())
                    || fd_errors[2] > PLANNING_GRADIENT_ERROR_LIMIT
                {
                    rejection_counts[4] += 1;
                }
                if let Some(reason) = rejection {
                    rejection_counts[reason] += 1;
                    first_rejection.get_or_insert([
                        packet_request.density_model,
                        packet_request.candidate_start + target_index / samples_per_candidate,
                        target_index % samples_per_candidate,
                        target_rows[5][3].max(0.0) as u32,
                        reason as u32,
                    ]);
                }
                let valid = rejection.is_none();
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
            let maximum_gradient_self_fd_relative_error = compact_rows
                .chunks_exact(8)
                .take(state_indices.len())
                .map(|target_rows| target_rows[4][3])
                .filter(|error| error.is_finite())
                .fold(0.0_f32, f32::max);
            let mut common_indices = Vec::with_capacity(state_indices.len());
            let mut rows = Vec::with_capacity(state_indices.len() * 4);
            let mut raw_rows = Vec::with_capacity(state_indices.len() * 4);
            for ((target_index, target_rows), target_valid) in state_indices
                .iter()
                .copied()
                .zip(compact_rows.chunks_exact(8).take(state_indices.len()))
                .zip(valid_targets)
            {
                let candidate = target_index / samples_per_candidate;
                common_indices.push(target_index);
                raw_rows.extend([target_rows[0], target_rows[2], target_rows[3], target_rows[4]]);
                if target_valid && !rejected_candidates[candidate as usize] {
                    rows.extend([target_rows[0], target_rows[2], target_rows[3], target_rows[4]]);
                } else {
                    // Keep the point and charge the rejected candidate a full
                    // reference-relative error instead of survivor-biasing
                    // Eq.106 by silently omitting it.
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
                maximum_gradient_self_fd_relative_error,
                rejection_counts,
                self_fd_step_maxima,
                first_rejection,
                rejected_sample_count,
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                f32::NAN,
                [0; 6],
                [f32::NAN; 5],
                None,
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
                rejection_counts: rows.5,
                self_fd_step_maxima: rows.6,
                first_rejection: rows.7,
                rejected_sample_count: rows.8,
                readback_valid: result.is_ok(),
                timing: PlanningGpuTiming {
                    method_preprocess_ms,
                    command_submission_ms,
                    gpu_completion_map_ms,
                    // A cache miss has three Eq.106 passes plus reduction; a
                    // cache hit executes only evaluate plus reduction.
                    dispatch_count: if build_spectrum { 4 } else { 2 },
                    forward_kernel_evaluations: u64::from(build_spectrum)
                        * u64::from(source_count)
                        * u64::from(QUADRATURE_COUNT)
                        * u64::from(canonical_element_count)
                        + target_count as u64
                            * u64::from(FREQUENCY_COUNT)
                            * u64::from(taylor_coefficient_count(TAYLOR_MAX_ORDER)),
                    spectral_element_count: reported_segment_count,
                    gradient_self_fd_relative_error: rows.4,
                },
                backend: PlanningExecutionBackend::GpuEq106,
            });
        }
        in_flight.store(false, Ordering::Release);
    });
}

fn canonical_element_accepts(
    element: &Eq106BatchElement,
    position: Vec3,
    source_radius: f32,
) -> bool {
    if !position.is_finite() {
        return false;
    }
    let relative = position - element.line_origin;
    let h = relative.dot(element.line_direction);
    if !h.is_finite() || h < -1.0e-3 || h > element.line_limit {
        return false;
    }
    let line_point = element.line_origin + h.max(0.0) * element.line_direction;
    let distance_lower_bound = line_point.length() - source_radius;
    if !distance_lower_bound.is_finite() || distance_lower_bound <= 0.0 {
        return false;
    }
    let transverse = position.distance(line_point);
    let epsilon = transverse / distance_lower_bound;
    select_batch_taylor_order(epsilon).is_some_and(|order| order <= element.taylor_order)
}

fn planning_eq106_stage_budget(compute_benchmark: bool, total_stages: usize) -> usize {
    if compute_benchmark {
        return total_stages.max(1);
    }
    let average_fps = crate::browser_frame_rate();
    let recent_frame_ms = crate::browser_recent_frame_ms();
    let frame_over_budget = average_fps.is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS)
        || recent_frame_ms
            .is_some_and(|milliseconds| milliseconds > PLANNING_MAX_RECENT_FRAME_MS);
    if frame_over_budget {
        return 0;
    }
    // Reserved for non-benchmark diagnostic callers. Both user-selectable
    // planning profiles use the full-stage fairness path above.
    2.min(total_stages.max(1))
}

#[cfg(test)]
mod planning_stage_budget_tests {
    use super::planning_eq106_stage_budget;

    #[test]
    fn fixed_benchmarks_encode_every_stage_but_interactive_runs_remain_paced() {
        assert_eq!(planning_eq106_stage_budget(true, 4), 4);
        assert!(planning_eq106_stage_budget(false, 4) <= 2);
    }
}
