#[derive(Resource, Default)]
struct PlanningEq106DispatchState {
    last_request_id: u64,
    batch_id: u64,
    payload_request_id: u64,
    output_size: u64,
    staging_size: u64,
    baseline_size: u64,
    metric_size: u64,
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
    element_tiles: std::collections::HashMap<(u32, u32), (Vec<Eq106BatchElement>, u32)>,
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
    if request.method != Some(ActiveGravityMethod::CurvedArcEq106)
        || planning.payload.method != request.method
        || planning.payload.density_model != request.density_model
        || state.last_request_id == request.request_id
    {
        return;
    }
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
    let (Some(line_samples_pipeline), Some(assemble_pipeline), Some(analytic_pipeline), Some(evaluate_pipeline)) = (
        cache.get_compute_pipeline(pipelines.line_samples_id),
        cache.get_compute_pipeline(pipelines.assemble_id),
        cache.get_compute_pipeline(pipelines.analytic_id),
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
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let method_preprocess_started = Instant::now();
    let samples_per_candidate = batch.samples_per_candidate as usize;
    let global_state_start = request.candidate_start as usize * samples_per_candidate;
    let target_count = request.candidate_count as usize * samples_per_candidate;
    if state.batch_id != batch.batch_id {
        state.element_tiles.clear();
    }
    let tile_key = (request.candidate_start, request.candidate_count);
    let (elements, maximum_elements_per_candidate) = if let Some(cached) =
        state.element_tiles.get(&tile_key)
    {
        cached.clone()
    } else {
        let mut elements = Vec::new();
        let mut maximum_elements_per_candidate = 0_u32;
        for local_candidate in 0..request.candidate_count as usize {
            let start = global_state_start + local_candidate * samples_per_candidate;
            let end = start + samples_per_candidate;
            let positions = batch.states[start..end]
                .iter()
                .map(|state| state.body_position())
                .collect::<Vec<_>>();
            let velocities = batch.states[start..end]
                .iter()
                .map(|state| state.body_velocity())
                .collect::<Vec<_>>();
            let local_offset = local_candidate as u32 * batch.samples_per_candidate;
            let mut candidate_elements = build_trajectory_batch_elements(
                &positions,
                &velocities,
                planning.source_radius,
                4.0 * planning.source_radius,
            );
            maximum_elements_per_candidate =
                maximum_elements_per_candidate.max(candidate_elements.len() as u32);
            for element in &mut candidate_elements {
                element.target_offset += local_offset;
            }
            elements.extend(candidate_elements);
        }
        state.element_tiles.insert(
            tile_key,
            (elements.clone(), maximum_elements_per_candidate),
        );
        (elements, maximum_elements_per_candidate)
    };
    if elements.is_empty() {
        channel.in_flight.store(false, Ordering::Release);
        return;
    }
    let common_changed = state.batch_id != batch.batch_id
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
            usage: BufferUsages::STORAGE,
        }));
        state.psi = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_psi"),
            contents: &planning.eq106_psi,
            usage: BufferUsages::STORAGE,
        }));
        state.dummy_modes = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_unused_modes"),
            contents: &[0; 16],
            usage: BufferUsages::STORAGE,
        }));
    }
    if state.payload_request_id != planning.payload.request_id || state.sources.is_none() {
        state.payload_request_id = planning.payload.request_id;
        state.sources = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_sources"),
            contents: &planning.payload.primary,
            usage: BufferUsages::STORAGE,
        }));
    }
    let coefficient_count = taylor_coefficient_count(TAYLOR_MAX_ORDER) as u64;
    if state.spectrum.is_none() {
        state.spectrum = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_spectrum"),
            size: coefficient_count * FREQUENCY_COUNT as u64 * 32,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    if state.line_samples.is_none() {
        state.line_samples = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_line_samples"),
            size: coefficient_count * QUADRATURE_COUNT as u64 * 16,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    let output_size = target_count as u64 * OUTPUT_BYTES;
    let verification_targets =
        crate::gpu::planning_reduction::planning_verification_targets(request, batch);
    let metric_size = u64::from(request.candidate_count) * 16;
    let staging_size = metric_size + verification_targets.len() as u64 * 5 * 16;
    if state.output_size != output_size || state.output.is_none() {
        state.output_size = output_size;
        state.output = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("planning_eq106_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
    let layout = render_device.create_bind_group_layout(
        "planning_eq106_bgl",
        &[
            uniform_entry(0), storage_ro_entry(1), storage_ro_entry(2), storage_rw_entry(3),
            storage_rw_entry(4), storage_ro_entry(5), storage_rw_entry(6), storage_ro_entry(7),
            storage_ro_entry(8), storage_ro_entry(9),
        ],
    );
    let method_preprocess_ms = method_preprocess_started.elapsed().as_secs_f64() * 1.0e3;
    let encode_started = Instant::now();
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("planning_eq106_encoder"),
    });
    let mut uniforms = Vec::with_capacity(elements.len());
    let mut bind_groups = Vec::with_capacity(elements.len());
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
            element_index as u32 + 1,
            false,
            false,
            element.target_count,
            element.target_offset,
        );
        bytes[92..96].copy_from_slice(&(global_state_start as u32).to_le_bytes());
        let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_eq106_uniform"),
            contents: &bytes,
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = render_device.create_bind_group(
            "planning_eq106_bg",
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
        for (label, pipeline, groups) in [
            ("planning_eq106_line", line_samples_pipeline, QUADRATURE_COUNT.div_ceil(64)),
            ("planning_eq106_spectrum", assemble_pipeline, (taylor_coefficient_count(element.taylor_order) * FREQUENCY_COUNT).div_ceil(64)),
            ("planning_eq106_analytic", analytic_pipeline, FREQUENCY_COUNT.div_ceil(64)),
        ] {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("planning_eq106_evaluate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(evaluate_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (width, height) = target_dispatch_grid(element.target_count);
            pass.dispatch_workgroups(width, height, 1);
        }
        uniforms.push(uniform);
        bind_groups.push(bind_group);
    }
    let (_reduction_uniform, _reduction_bind_group) =
        crate::gpu::planning_reduction::encode_planning_reduction(
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
        encoder.copy_buffer_to_buffer(
            &output,
            u64::from(target_index) * OUTPUT_BYTES,
            &staging,
            metric_size + compact_index as u64 * 5 * 16,
            5 * 16,
        );
    }
    render_queue.submit([encoder.finish()]);
    state.last_request_id = request.request_id;
    let encode_ms = encode_started.elapsed().as_secs_f64() * 1.0e3;
    let packet_request = request.clone();
    let shared_data = Arc::clone(&channel.data);
    let in_flight = Arc::clone(&channel.in_flight);
    let mapped = staging.clone();
    let element_count = elements.len() as u32;
    let reported_segment_count = maximum_elements_per_candidate;
    let source_count = planning.payload.item_count;
    let state_indices = verification_targets;
    let request_candidate_count = request.candidate_count;
    let readback_started = Instant::now();
    mapped.slice(..staging_size).map_async(MapMode::Read, move |result| {
        let readback_ms = readback_started.elapsed().as_secs_f64() * 1.0e3;
        let rows = if result.is_ok() {
            let view = staging.slice(..staging_size).get_mapped_range();
            let full_rows = bytes_to_f32x4(&view);
            let candidate_metrics = full_rows[..request_candidate_count as usize].to_vec();
            let mut rows = Vec::with_capacity(state_indices.len() * 4);
            for target_rows in full_rows[request_candidate_count as usize..]
                .chunks_exact(5)
                .take(state_indices.len())
            {
                let certificate = target_rows[1];
                let valid = certificate[0] <= GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
                    && certificate[1] <= GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
                    && certificate[2] <= GRAVITY_BENCHMARK_RELATIVE_TOLERANCE
                    && certificate[3] <= 0.30;
                if valid {
                    rows.extend([target_rows[0], target_rows[2], target_rows[3], target_rows[4]]);
                } else {
                    rows.extend([[f32::NAN; 4]; 4]);
                }
            }
            drop(view);
            staging.unmap();
            (rows, candidate_metrics)
        } else {
            (Vec::new(), Vec::new())
        };
        if let Ok(mut guard) = shared_data.lock() {
            *guard = Some(PlanningGpuPacket {
                request: packet_request,
                state_indices,
                rows: rows.0,
                candidate_metrics: rows.1,
                readback_valid: result.is_ok(),
                timing: PlanningGpuTiming {
                    method_preprocess_ms,
                    encode_ms,
                    readback_ms,
                    dispatch_count: element_count.saturating_mul(4).saturating_add(1),
                    forward_kernel_evaluations: u64::from(source_count)
                        * u64::from(QUADRATURE_COUNT)
                        * u64::from(element_count)
                        + target_count as u64
                            * u64::from(FREQUENCY_COUNT)
                            * u64::from(taylor_coefficient_count(TAYLOR_MAX_ORDER)),
                    spectral_element_count: reported_segment_count,
                },
                backend: PlanningExecutionBackend::GpuEq106,
            });
        }
        in_flight.store(false, Ordering::Release);
    });
}
