fn extract_eq106_input(
    mut extracted: ResMut<ExtractedEq106Input>,
    source: Extract<Option<Res<AggregatedGravitySource>>>,
    operator_tensor: Extract<Option<Res<Eq106OperatorTensorResource>>>,
    active: Extract<Res<ActiveGravityMethod>>,
    clock: Extract<Res<SimulationClock>>,
    planner: Extract<Res<CurvedArcPlannerState>>,
    benchmark: Extract<Res<GravityBenchmarkTrajectory>>,
    inversion: Extract<Res<TrajectoryInversionState>>,
    batch_result: Extract<Res<Eq106TrajectoryBatchResult>>,
    sensitivity: Extract<Res<Eq106SensitivityMatrix>>,
    cassini: Extract<Query<(&Transform, &Velocity), With<CassiniMarker>>>,
    ryugu: Extract<Query<&Transform, With<RyuguMarker>>>,
) {
    // Selection alone is insufficient: an uncertified segment must never be
    // submitted after adaptive splitting leaves the Taylor convergence disk.
    // Bootstrap the first Eq.106 sample before the adaptive planner has a
    // trajectory window. The planner itself is fed by completed GPU snapshots;
    // requiring `kernel_ready` here creates a circular wait after switching
    // methods because the switch clears both the orbit history and planner.
    extracted.enabled = **active == ActiveGravityMethod::CurvedArcEq106
        && planner.mode != crate::cpu::curved_arc::CurvedArcMode::Error;
    if !extracted.enabled {
        return;
    }
    let (Some(source), Ok((probe, velocity)), Ok(ryugu)) =
        (source.as_ref(), cassini.single(), ryugu.single())
    else {
        return;
    };
    let relative_world_position = probe.translation - ryugu.translation;
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    extracted.probe = ryugu.rotation.inverse() * relative_world_position;
    // Eq.106's reference line lives in the body-fixed density coordinates, so
    // its tangent must be dq_body/dt, not the inertial velocity merely rotated
    // into body axes. Omitting omega x r tilts the spectral line by a large
    // fraction of the orbital velocity and causes secular trajectory error.
    extracted.velocity = ryugu.rotation.inverse()
        * (velocity.0 - angular_velocity_world.cross(relative_world_position));
    extracted.snapshot = Some(GravityRequestSnapshot {
        request_id: clock.request_id,
        epoch: clock.epoch,
        simulation_time_seconds: clock.elapsed_seconds,
        body_position: extracted.probe,
        ryugu_transform: *ryugu,
        probe_position: probe.translation,
        probe_velocity: velocity.0,
    });
    extracted.target_bytes.clear();
    extracted.target_snapshots.clear();
    extracted.batch_elements.clear();
    extracted.batch_capture_id = None;
    extracted.sensitivity_sources.clear();
    extracted.sensitivity_source_counts.clear();
    extracted.sensitivity_source_hash = 0;
    extracted.sensitivity_basis_hash = 0;
    extracted.source_count = source.sources.len() as u32;
    extracted.density_mode_count = source.fourier_modes.len() as u32;
    extracted.radius = source.radius as f32;
    extracted.certified_line_limit = planner
        .active_segment
        .as_ref()
        .filter(|segment| segment.taylor_order.is_some() && segment.epsilon_max < 1.0)
        .map(|segment| {
            let curvature_limit = if segment.maximum_curvature > f64::MIN_POSITIVE {
                (8.0 * 0.20 * segment.distance_lower_bound / segment.maximum_curvature).sqrt()
            } else {
                f64::INFINITY
            };
            curvature_limit
                .min(4.0 * source.radius)
                .max(0.35 * source.radius) as f32
        })
        .unwrap_or(4.0 * extracted.radius);
    extracted.taylor_order = planner.taylor_order.clamp(1, TAYLOR_MAX_ORDER);

    let pending_benchmark_id = benchmark
        .complete
        .then_some(benchmark.capture_id)
        .flatten()
        .filter(|capture_id| batch_result.capture_id != Some(*capture_id));
    let pending_inversion_id = inversion
        .ready
        .then_some(inversion.capture_id)
        .flatten()
        .filter(|capture_id| batch_result.capture_id != Some(*capture_id));
    let pending_sensitivity = inversion.optimizer.as_ref().and_then(|job| {
        (job.method == ActiveGravityMethod::CurvedArcEq106
            && sensitivity.capture_id == Some(job.capture_id)
            && sensitivity.source_hash == job.source_hash
            && sensitivity.basis_hash == job.basis_sources.hash
            && sensitivity.configuration_hash == eq106_sensitivity_configuration_hash()
            && sensitivity.columns.is_empty())
            .then_some((job.capture_id, job))
    });
    if let Some((capture_id, job)) = pending_sensitivity {
        let samples = &job.frozen_samples;
        if samples.len() >= 2 {
            extracted.batch_capture_id = Some(capture_id);
            extracted.sensitivity_source_hash = job.source_hash;
            extracted.sensitivity_basis_hash = job.basis_sources.hash;
            let mut positions = Vec::with_capacity(samples.len());
            let mut velocities = Vec::with_capacity(samples.len());
            let mut times = Vec::with_capacity(samples.len());
            for (index, sample) in samples.iter().enumerate() {
                let rotation = sample.body_rotation;
                let body_position = rotation.inverse() * sample.position;
                let body_velocity = rotation.inverse()
                    * (sample.velocity - angular_velocity_world.cross(sample.position));
                positions.push(body_position);
                velocities.push(body_velocity);
                times.push(sample.simulation_time_seconds as f32);
                for value in [body_position.x, body_position.y, body_position.z, 0.0] {
                    extracted.target_bytes.extend_from_slice(&value.to_le_bytes());
                }
                extracted.target_snapshots.push(GravityRequestSnapshot {
                    request_id: index as u64,
                    epoch: inversion.capture_epoch,
                    simulation_time_seconds: sample.simulation_time_seconds,
                    body_position,
                    ryugu_transform: Transform::from_rotation(rotation),
                    probe_position: sample.position,
                    probe_velocity: sample.velocity,
                });
            }
            extracted.probe = positions[0];
            extracted.velocity = velocities[0];
            extracted.snapshot = extracted.target_snapshots.first().cloned();
            extracted.batch_elements = build_trajectory_batch_elements(
                &positions,
                &velocities,
                &times,
                extracted.radius,
                extracted.certified_line_limit,
            );
            extracted
                .sensitivity_sources
                .reserve(job.basis_sources.columns.len());
            extracted
                .sensitivity_source_counts
                .reserve(job.basis_sources.columns.len());
            for column in &job.basis_sources.columns {
                let mut bytes = Vec::with_capacity(column.len() * 16);
                for source in column {
                    for value in [
                        source.position.x as f32,
                        source.position.y as f32,
                        source.position.z as f32,
                        source.volume as f32,
                    ] {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                extracted.sensitivity_sources.push(bytes);
                extracted.sensitivity_source_counts.push(column.len() as u32);
            }
            if let Some(first) = extracted.sensitivity_sources.first() {
                extracted.sources = Some(first.clone());
                extracted.source_count = extracted.sensitivity_source_counts[0];
                extracted.source_hash = capture_id
                    ^ job.source_hash.rotate_left(29)
                    ^ job.basis_sources.hash.rotate_right(7);
            }
        }
    } else if let Some(capture_id) = pending_benchmark_id
        && benchmark.samples.len() >= 2
    {
        extracted.batch_capture_id = Some(capture_id);
        let mut batch_positions = Vec::with_capacity(benchmark.samples.len());
        let mut batch_velocities = Vec::with_capacity(benchmark.samples.len());
        let mut batch_times = Vec::with_capacity(benchmark.samples.len());
        for (index, sample) in benchmark.samples.iter().enumerate() {
            let body_position = sample.body_rotation.inverse() * sample.position;
            let body_velocity = sample.body_rotation.inverse()
                * (sample.velocity - angular_velocity_world.cross(sample.position));
            batch_positions.push(body_position);
            batch_velocities.push(body_velocity);
            batch_times.push(sample.simulation_time_seconds as f32);
            for value in [body_position.x, body_position.y, body_position.z, 0.0] {
                extracted
                    .target_bytes
                    .extend_from_slice(&value.to_le_bytes());
            }
            extracted.target_snapshots.push(GravityRequestSnapshot {
                request_id: index as u64,
                epoch: benchmark.epoch,
                simulation_time_seconds: sample.simulation_time_seconds,
                body_position,
                ryugu_transform: Transform::from_rotation(sample.body_rotation),
                probe_position: sample.position,
                probe_velocity: sample.velocity,
            });
        }
        extracted.probe = batch_positions[0];
        extracted.velocity = batch_velocities[0];
        extracted.snapshot = extracted.target_snapshots.first().cloned();
        extracted.batch_elements = build_trajectory_batch_elements(
            &batch_positions,
            &batch_velocities,
            &batch_times,
            extracted.radius,
            extracted.certified_line_limit,
        );
    } else if let Some(capture_id) = pending_inversion_id
        && inversion.raw_samples.len() >= 2
    {
        extracted.batch_capture_id = Some(capture_id);
        let mut batch_positions = Vec::with_capacity(inversion.raw_samples.len());
        let mut batch_velocities = Vec::with_capacity(inversion.raw_samples.len());
        let mut batch_times = Vec::with_capacity(inversion.raw_samples.len());
        for (index, sample) in inversion.raw_samples.iter().enumerate() {
            let rotation = sample.knot.body_rotation;
            let body_position = rotation.inverse() * sample.knot.position;
            let body_velocity = rotation.inverse()
                * (sample.knot.velocity - angular_velocity_world.cross(sample.knot.position));
            batch_positions.push(body_position);
            batch_velocities.push(body_velocity);
            batch_times.push(sample.knot.simulation_time_seconds as f32);
            for value in [body_position.x, body_position.y, body_position.z, 0.0] {
                extracted
                    .target_bytes
                    .extend_from_slice(&value.to_le_bytes());
            }
            extracted.target_snapshots.push(GravityRequestSnapshot {
                request_id: index as u64,
                epoch: inversion.capture_epoch,
                simulation_time_seconds: sample.knot.simulation_time_seconds,
                body_position,
                ryugu_transform: Transform::from_rotation(rotation),
                probe_position: sample.knot.position,
                probe_velocity: sample.knot.velocity,
            });
        }
        let first = inversion.raw_samples[0];
        let first_rotation = first.knot.body_rotation;
        extracted.probe = first_rotation.inverse() * first.knot.position;
        extracted.velocity = first_rotation.inverse()
            * (first.knot.velocity - angular_velocity_world.cross(first.knot.position));
        extracted.snapshot = extracted.target_snapshots.first().cloned();
        extracted.batch_elements = build_trajectory_batch_elements(
            &batch_positions,
            &batch_velocities,
            &batch_times,
            extracted.radius,
            extracted.certified_line_limit,
        );
    } else if let Some(snapshot) = extracted.snapshot.clone() {
        for value in [
            snapshot.body_position.x,
            snapshot.body_position.y,
            snapshot.body_position.z,
            0.0,
        ] {
            extracted
                .target_bytes
                .extend_from_slice(&value.to_le_bytes());
        }
        extracted.target_snapshots.push(snapshot);
    }
    // The CPU owns the authoritative orbit integration. Production Eq.106
    // queries only the current anchor; every CPU substep evaluates the same
    // cached local spectrum/Jacobian rather than consuming a GPU-predicted
    // future trajectory as if it were measured state.
    let source_hash = source.source_hash;
    if extracted.sensitivity_sources.is_empty()
        && (extracted.sources.is_none() || extracted.source_hash != source_hash)
    {
        let mut bytes = Vec::with_capacity(source.sources.len() * 16);
        for item in &source.sources {
            for value in [
                item.position.x as f32,
                item.position.y as f32,
                item.position.z as f32,
                item.mass as f32,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        extracted.sources = Some(bytes);
    }
    if extracted.fourier_modes.is_none() {
        let mut mode_bytes = Vec::with_capacity(source.fourier_modes.len() * 16);
        for record in &source.fourier_modes {
            for value in record {
                mode_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        extracted.fourier_modes = Some(mode_bytes);
    }
    if extracted.sensitivity_sources.is_empty() {
        extracted.source_hash = source_hash;
    }
    if extracted.operator_tensor.is_none() {
        extracted.operator_tensor = operator_tensor
            .as_ref()
            .map(|resource| resource.tensor.as_le_bytes());
    }
    if extracted.psi_operator.is_none() {
        extracted.psi_operator = operator_tensor
            .as_ref()
            .map(|resource| resource.psi.as_le_bytes());
    }
}

fn initialize_eq106_pipeline(world: &mut World) {
    let enabled = world.resource::<ExtractedEq106Input>().enabled
        || world
            .get_resource::<crate::gpu::planning::ExtractedPlanningInput>()
            .is_some_and(|planning| {
                planning.request.method == Some(ActiveGravityMethod::CurvedArcEq106)
            });
    if enabled && !world.contains_resource::<Eq106ComputePipeline>() {
        // Render schedules do not automatically flush ordinary Commands at
        // every set boundary; initialize directly in this exclusive system.
        world.init_resource::<Eq106ComputePipeline>();
    }
}

fn dispatch_eq106_single_target(
    inner: &mut Eq106GpuBuffersInner,
    line_samples: &wgpu29::ComputePipeline,
    assemble: &wgpu29::ComputePipeline,
    analytic: &wgpu29::ComputePipeline,
    evaluate: &wgpu29::ComputePipeline,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    extracted: &ExtractedEq106Input,
    channel: &Eq106GpuReadbackChannel,
    snapshot: &GravityRequestSnapshot,
) {
    let relative = extracted.probe - inner.line_origin;
    let h = relative.dot(inner.line_direction);
    let transverse = (relative - h * inner.line_direction).length();
    // Eq. (106) is a straight-reference operator; the curved trajectory must
    // be covered by local spectral elements. The frequency-grid Nyquist range
    // is not a convergence radius, so do not reuse one line for kilometres.
    // At 8x the probe advances roughly one old 0.15R segment per presented
    // frame. Size the cached line for two accelerated batches so spectrum
    // assembly is not repeated every frame, while keeping the operator local.
    let mut longitudinal_limit = (0.35 * extracted.radius)
        // Keep at least ~18 authoritative 1x frames in one element. This
        // gives the previous sample time to enter the zero-correction overlap
        // before the next reference line is installed, avoiding a visible
        // potential step without extending the spectral work per frame.
        .max(160.0)
        // The curved-arc planner supplies the docs/mathtidy.md curvature
        // bound.  Unlike the throughput heuristic above, this is a hard cap:
        // spectral correction is tapered to zero before leaving this disk.
        .min(extracted.certified_line_limit)
        .max(1.0);
    if channel.rebuild_requested.swap(false, Ordering::AcqRel) {
        inner.line_scale = (0.5 * inner.line_scale).max(0.125);
        inner.line_origin = extracted.probe;
        inner.line_direction = extracted.velocity.normalize_or_zero();
        inner.segment_id = inner.segment_id.wrapping_add(1).max(1);
        inner.spectrum_ready = false;
        inner.last_submitted = None;
    }
    longitudinal_limit = (longitudinal_limit * inner.line_scale).max(1.0);
    let transverse_limit = (0.10 * extracted.radius).max(20.0);
    let line_expired = inner.source_hash != extracted.source_hash
        || h < 0.0
        || h > 0.85 * longitudinal_limit
        || transverse > transverse_limit;
    if line_expired {
        inner.line_origin = extracted.probe;
        inner.line_direction = extracted.velocity.normalize_or_zero();
        inner.segment_id = inner.segment_id.wrapping_add(1).max(1);
        inner.source_hash = extracted.source_hash;
        // A fresh reference element gets a fresh curvature budget. A previous
        // certificate rejection only applies to the element it rejected;
        // carrying its halved scale into every later element causes needless
        // rebuilds and collapses the render rate at high simulation speed.
        inner.line_scale = 1.0;
        inner.spectrum_ready = false;
        inner.last_submitted = None;
    }
    if inner.line_direction == Vec3::ZERO {
        return;
    }
    let key = (snapshot.epoch, snapshot.request_id);
    if inner.last_submitted == Some(key) {
        return;
    }
    if channel
        .in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    inner.last_submitted = Some(key);
    render_queue.write_buffer(&inner.targets, 0, &extracted.target_bytes);
    let evaluate_dual_certificate = extracted.sensitivity_sources.is_empty()
        && cfg!(feature = "eq106-dual-certificate")
        && inner
            .dual_certificate_frame
            .is_multiple_of(DUAL_CERTIFICATE_CADENCE);
    inner.dual_certificate_frame = inner.dual_certificate_frame.wrapping_add(1);
    let uniform = uniform_bytes(
        extracted.probe,
        inner.line_origin,
        inner.line_direction,
        extracted.source_count,
        extracted.radius,
        longitudinal_limit,
        extracted.taylor_order,
        extracted.density_mode_count,
        1,
        evaluate_dual_certificate,
        false,
        inner.target_count,
        0,
    );
    render_queue.write_buffer(&inner.uniform, 0, &uniform);
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("eq106_complex_encoder"),
    });
    let query_set = inner.timing_query_set.as_ref();
    let mut timing_layout = Eq106TimingLayout::default();
    let mut next_query = 0_u32;
    let build_spectrum = !inner.spectrum_ready;
    let (build_begin, build_end) = if build_spectrum && query_set.is_some() {
        timing_layout.build_pairs.push((next_query, next_query + 1));
        next_query += 2;
        (Some(next_query - 2), Some(next_query - 1))
    } else {
        (None, None)
    };
    if build_spectrum {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_line_samples_pass"),
            timestamp_writes: timestamp_writes(query_set, build_begin, None),
        });
        pass.set_pipeline(line_samples);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(QUADRATURE_COUNT.div_ceil(64), 1, 1);
        drop(pass);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_assemble_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(assemble);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(
            (taylor_coefficient_count(extracted.taylor_order) * FREQUENCY_COUNT).div_ceil(64),
            1,
            1,
        );
        drop(pass);
        // Replace the zeroth (on-line) coefficient with Eqs. (47),(68)-(70)
        // evaluated from the certified complex Psi/Psi_x operator. Higher
        // coefficients remain the exact local Newton Taylor continuation of
        // Eq. (118), and vanish at the reference line itself.
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_analytic_spectrum_pass"),
            timestamp_writes: timestamp_writes(query_set, None, build_end),
        });
        pass.set_pipeline(analytic);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        pass.dispatch_workgroups(FREQUENCY_COUNT.div_ceil(64), 1, 1);
        drop(pass);
        inner.spectrum_ready = true;
    }
    let (evaluation_begin, evaluation_end) = if query_set.is_some() {
        timing_layout
            .evaluation_pairs
            .push((next_query, next_query + 1));
        next_query += 2;
        (Some(next_query - 2), Some(next_query - 1))
    } else {
        (None, None)
    };
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("eq106_evaluate_pass"),
            timestamp_writes: timestamp_writes(query_set, evaluation_begin, evaluation_end),
        });
        pass.set_pipeline(evaluate);
        pass.set_bind_group(0, &inner.bind_group, &[]);
        let (width, height) = target_dispatch_grid(inner.target_count);
        pass.dispatch_workgroups(width, height, 1);
    }
    encoder.copy_buffer_to_buffer(&inner.output, 0, &inner.staging, 0, inner.output_size);
    if let (Some(query_set), Some(resolve)) = (query_set, inner.timing_resolve.as_ref()) {
        let readback_begin = timing_layout.evaluation_pairs[0].1;
        let readback_end = next_query;
        next_query += 1;
        {
            let _pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq106_readback_timestamp"),
                timestamp_writes: timestamp_writes(Some(query_set), Some(readback_end), None),
            });
        }
        timing_layout.readback_pair = Some((readback_begin, readback_end));
        timing_layout.query_count = next_query;
        encoder.resolve_query_set(query_set, 0..next_query, resolve, 0);
        encoder.copy_buffer_to_buffer(
            resolve,
            0,
            &inner.staging,
            inner.output_size,
            next_query as u64 * TIMESTAMP_BYTES,
        );
    }
    render_queue.submit([encoder.finish()]);

    let shared = Arc::clone(&channel.data);
    let in_flight = Arc::clone(&channel.in_flight);
    let submitted_at = Arc::clone(&channel.submitted_at);
    let error_slot = Arc::clone(&channel.pipeline_error);
    let rebuild_requested = Arc::clone(&channel.rebuild_requested);
    let staging = inner.staging.clone();
    let map_staging = staging.clone();
    let snapshots = extracted.target_snapshots.clone();
    let batch_capture_id = extracted.batch_capture_id;
    let output_size = inner.output_size as usize;
    let target_count = inner.target_count;
    let timestamp_period_ns = render_queue.get_timestamp_period();
    let readback_started = Instant::now();
    if let Ok(mut submitted) = channel.submitted_at.lock() {
        *submitted = Some(readback_started);
    }
    map_staging
        .slice(..)
        .map_async(MapMode::Read, move |result| {
            match result {
                Ok(()) => {
                    let view = staging.slice(..).get_mapped_range();
                    let values = bytes_to_f32x4(&view[..output_size]);
                    let cpu_readback_wait_ms = readback_started.elapsed().as_secs_f64() * 1.0e3;
                    let timings = if timing_layout.query_count > 0 {
                        let end = output_size
                            + timing_layout.query_count as usize * TIMESTAMP_BYTES as usize;
                        decode_gpu_timings(
                            &view[output_size..end],
                            timestamp_period_ns,
                            &timing_layout,
                            cpu_readback_wait_ms,
                            target_count,
                            1,
                        )
                    } else {
                        Eq106TimingSample {
                            cpu_readback_wait_ms,
                            target_count,
                            spectral_element_count: 1,
                            ..default()
                        }
                    };
                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some(Eq106ReadbackPacket {
                            partial_sums: values,
                            snapshots,
                            batch_capture_id,
                            sensitivity_column_count: 0,
                            sensitivity_source_hash: 0,
                            sensitivity_basis_hash: 0,
                            sensitivity_configuration_hash: 0,
                            timings,
                        });
                    }
                    drop(view);
                    staging.unmap();
                }
                Err(error) => {
                    rebuild_requested.store(true, Ordering::Release);
                    if let Ok(mut slot) = error_slot.lock()
                        && slot.is_none()
                    {
                        *slot = Some(format!(
                            "Equation (106) GPU readback failed: {error:?}"
                        ));
                    }
                }
            }
            if let Ok(mut submitted) = submitted_at.lock() {
                submitted.take();
            }
            in_flight.store(false, Ordering::Release);
        });
}

fn report_eq106_pipeline_errors(
    cache: &PipelineCache,
    pipelines: &Eq106ComputePipeline,
    channel: &Eq106GpuReadbackChannel,
) {
    for (name, id) in [
        ("line samples", pipelines.line_samples_id),
        ("sampled spectrum", pipelines.assemble_id),
        ("analytic Eq.70 spectrum", pipelines.analytic_id),
        ("inverse spectrum", pipelines.evaluate_id),
    ] {
        match cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            ) => {}
            CachedPipelineState::Err(error) => {
                error!(
                    target: "wgsl::eq106",
                    pipeline = name,
                    error = ?error,
                    "Eq.106 compute pipeline compilation failed"
                );
                if let Ok(mut slot) = channel.pipeline_error.try_lock()
                    && slot.is_none()
                {
                    *slot = Some(format!(
                        "Equation (106) {name} GPU pipeline failed: {error}"
                    ));
                }
            }
            _ => {}
        }
    }
}
