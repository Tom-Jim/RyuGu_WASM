fn dispatch_eq106(
    mut buffers: ResMut<Eq106GpuBuffers>,
    pipelines: Option<Res<Eq106ComputePipeline>>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedEq106Input>,
    channel: Res<Eq106GpuReadbackChannel>,
) {
    let Some(pipelines) = pipelines else { return };
    report_eq106_pipeline_errors(&cache, &pipelines, &channel);
    let (Some(line_samples), Some(assemble), Some(analytic), Some(evaluate)) = (
        cache.get_compute_pipeline(pipelines.line_samples_id),
        cache.get_compute_pipeline(pipelines.assemble_id),
        cache.get_compute_pipeline(pipelines.analytic_id),
        cache.get_compute_pipeline(pipelines.evaluate_id),
    ) else {
        return;
    };
    if !extracted.enabled || extracted.source_count == 0 {
        return;
    }
    let buffer_taylor_order = extracted
        .batch_elements
        .iter()
        .map(|element| element.taylor_order)
        .max()
        .unwrap_or(extracted.taylor_order);
    let element_capacity = extracted.batch_elements.len().max(1) as u32;
    if buffers.0.as_ref().is_some_and(|inner| {
        inner.source_count != extracted.source_count
            || inner.density_mode_count != extracted.density_mode_count
            || inner.taylor_order != buffer_taylor_order
            || inner.target_count != extracted.target_snapshots.len() as u32
            || inner.element_capacity != element_capacity
    }) {
        // Rebuild only when buffer capacities or binding shapes change.
        buffers.0 = None;
    }
    if buffers.0.is_none() {
        let (Some(source_bytes), Some(operator_bytes), Some(psi_bytes), Some(mode_bytes)) = (
            extracted.sources.as_ref(),
            extracted.operator_tensor.as_ref(),
            extracted.psi_operator.as_ref(),
            extracted.fourier_modes.as_ref(),
        ) else {
            return;
        };
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_uniform"),
            size: 96 * 256,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sources = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_sources"),
            contents: source_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let quadrature_bytes = half_line_quadrature_bytes(0.5 * extracted.radius.max(1.0));
        let quadrature = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_quadrature_lut"),
            contents: &quadrature_bytes,
            usage: BufferUsages::STORAGE,
        });
        let operator_tensor = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_toroidal_operator_tensor"),
            contents: operator_bytes,
            usage: BufferUsages::UNIFORM,
        });
        let psi_operator = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_complex_psi_operator"),
            contents: psi_bytes,
            usage: BufferUsages::STORAGE,
        });
        let mut density_mode_bytes = vec![0_u8; 544 * 16];
        density_mode_bytes[..mode_bytes.len()].copy_from_slice(mode_bytes);
        let density_modes = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_density_fourier_modes"),
            contents: &density_mode_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let targets = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_targets"),
            contents: &extracted.target_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let spectrum = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_spectrum"),
            size: 45 * 129 * 32 * u64::from(element_capacity),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let line_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_line_samples"),
            size: 45 * 64 * 16 * u64::from(element_capacity),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let target_count = extracted.target_snapshots.len().max(1) as u32;
        let output_size = OUTPUT_BYTES * target_count as u64;
        // Sensitivity construction submits 56 short-lived source variants.
        // Reusing the compute buffers avoids deferred WebGPU allocations, and
        // timing queries are intentionally reserved for runtime/benchmark work.
        // Metal/WebGPU can advertise TIMESTAMP_QUERY while Dawn still cannot
        // allocate the native sample buffer. A failed query-set allocation
        // invalidates every following command buffer, so browser builds keep
        // timestamp instrumentation disabled and use CPU readback timing.
        // Timestamp query allocation is disabled completely. Do not leave a
        // lazy create_query_set closure here: Dawn/Metal may reject the native
        // sample buffer even when the feature bit is advertised.
        let timing_query_set: Option<QuerySet> = None;
        let timing_resolve: Option<Buffer> = None;
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("eq106_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "eq106_complex_bgl_runtime",
            &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
                storage_rw_entry(4),
                uniform_entry(5),
                storage_rw_entry(6),
                uniform_entry(7),
                storage_ro_entry(8),
                storage_ro_entry(9),
            ],
        );
        let bind_group = render_device.create_bind_group(
            "eq106_complex_bg",
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
                    binding: 5,
                    resource: operator_tensor.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: line_samples.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: density_modes.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: psi_operator.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: targets.as_entire_binding(),
                },
            ],
        );
        buffers.0 = Some(Eq106GpuBuffersInner {
            uniform,
            targets,
            output,
            staging,
            output_size,
            bind_group,
            layout,
            sources,
            quadrature,
            spectrum,
            operator_tensor,
            line_samples,
            density_modes,
            psi_operator,
            timing_query_set,
            timing_resolve,
            source_count: extracted.source_count,
            density_mode_count: extracted.density_mode_count,
            element_capacity,
            line_origin: extracted.probe,
            line_direction: extracted.velocity.normalize_or_zero(),
            segment_id: 1,
            source_hash: extracted.source_hash,
            spectrum_ready: false,
            line_scale: 1.0,
            taylor_order: buffer_taylor_order,
            target_count,
            dual_certificate_frame: 0,
            last_submitted: None,
        });
    }

    let inner = buffers.0.as_mut().expect("Eq106 GPU buffers initialized");
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        return;
    };
    if !extracted.sensitivity_sources.is_empty() {
        dispatch_eq106_sensitivity_matrix(
            inner,
            line_samples,
            assemble,
            analytic,
            evaluate,
            &render_device,
            &render_queue,
            &extracted,
            &channel,
            snapshot,
        );
        return;
    }
    if inner.source_hash != extracted.source_hash {
        let Some(source_bytes) = extracted.sources.as_ref() else {
            return;
        };
        render_queue.write_buffer(&inner.sources, 0, source_bytes);
        if let Some(mode_bytes) = extracted.fourier_modes.as_ref()
        {
            render_queue.write_buffer(&inner.density_modes, 0, mode_bytes);
        }
        inner.source_hash = extracted.source_hash;
        inner.spectrum_ready = false;
        inner.last_submitted = None;
    }
    if let Some(capture_id) = extracted.batch_capture_id
        && !extracted.batch_elements.is_empty()
    {
        let key = (snapshot.epoch, capture_id);
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
        let evaluate_dual_certificate = cfg!(feature = "eq106-dual-certificate")
            && inner
                .dual_certificate_frame
                .is_multiple_of(DUAL_CERTIFICATE_CADENCE);
        inner.dual_certificate_frame = inner.dual_certificate_frame.wrapping_add(1);
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("eq106_trajectory_batch_encoder"),
        });
        let query_set = inner.timing_query_set.as_ref();
        let mut timing_layout = Eq106TimingLayout::default();
        let mut next_query = 0_u32;
        // The WGSL uniform block has a fixed 256-element array. Keep the
        // binding range large enough for the declared block even when a
        // request contains fewer active trajectory elements.
        let mut parameter_bytes = vec![0_u8; 96 * 256];
        for (element_index, element) in extracted.batch_elements.iter().enumerate() {
            let bytes = uniform_bytes(
                element.line_origin,
                element.line_origin,
                element.line_direction,
                extracted.source_count,
                extracted.radius,
                element.line_limit,
                element.taylor_order,
                extracted.density_mode_count,
                element_index as u32 + 1,
                evaluate_dual_certificate && element.target_offset == 0,
                false,
                element.target_count,
                element.target_offset,
            );
            let offset = element_index * 96;
            parameter_bytes[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        let parameter_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eq106_trajectory_segment_params"),
            contents: &parameter_bytes,
            usage: BufferUsages::STORAGE,
        });
        let bind_group = render_device.create_bind_group(
            "eq106_trajectory_batch_bg",
            &inner.layout,
            &[
                BindGroupEntry { binding: 0, resource: parameter_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: inner.sources.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: inner.quadrature.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: inner.spectrum.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: inner.output.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: inner.operator_tensor.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: inner.line_samples.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: inner.density_modes.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: inner.psi_operator.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: inner.targets.as_entire_binding() },
            ],
        );
        let segment_count = extracted.batch_elements.len() as u32;
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq106_trajectory_line_samples_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(line_samples);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(QUADRATURE_COUNT.div_ceil(64), segment_count, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq106_trajectory_spectrum_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(assemble);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((45 * FREQUENCY_COUNT).div_ceil(64), segment_count, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq106_trajectory_analytic_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(analytic);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(FREQUENCY_COUNT.div_ceil(64), segment_count, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq106_trajectory_evaluate_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(evaluate);
            pass.set_bind_group(0, &bind_group, &[]);
            let max_targets = extracted
                .batch_elements
                .iter()
                .map(|element| element.target_count)
                .max()
                .unwrap_or(1);
            let (width, height) = target_dispatch_grid(max_targets);
            pass.dispatch_workgroups(width, height, segment_count);
        }
        encoder.copy_buffer_to_buffer(&inner.output, 0, &inner.staging, 0, inner.output_size);
        if let (Some(query_set), Some(resolve)) = (query_set, inner.timing_resolve.as_ref()) {
            let readback_begin = timing_layout
                .evaluation_pairs
                .last()
                .map(|pair| pair.1)
                .unwrap_or(0);
            let readback_end = next_query;
            next_query += 1;
            {
                let _pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("eq106_trajectory_readback_timestamp"),
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
        inner.segment_id = inner
            .segment_id
            .wrapping_add(extracted.batch_elements.len() as u32)
            .max(1);
        inner.spectrum_ready = false;

        let shared = Arc::clone(&channel.data);
        let in_flight = Arc::clone(&channel.in_flight);
        let staging = inner.staging.clone();
        let map_staging = staging.clone();
        let snapshots = extracted.target_snapshots.clone();
        let output_size = inner.output_size as usize;
        let target_count = inner.target_count;
        let element_count = extracted.batch_elements.len() as u32;
        let timestamp_period_ns = render_queue.get_timestamp_period();
        let readback_started = Instant::now();
        map_staging
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                if result.is_ok() {
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
                            element_count,
                        )
                    } else {
                        Eq106TimingSample {
                            cpu_readback_wait_ms,
                            target_count,
                            spectral_element_count: element_count,
                            dispatch_count: 1,
                            spectrum_rebuild_count: element_count,
                            ..default()
                        }
                    };
                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some(Eq106ReadbackPacket {
                            partial_sums: values,
                            snapshots,
                            batch_capture_id: Some(capture_id),
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
                in_flight.store(false, Ordering::Release);
            });
        return;
    }
    dispatch_eq106_single_target(
        inner,
        line_samples,
        assemble,
        analytic,
        evaluate,
        &render_device,
        &render_queue,
        &extracted,
        &channel,
        snapshot,
    );
}
