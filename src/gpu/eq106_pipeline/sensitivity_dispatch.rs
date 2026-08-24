fn dispatch_eq106_sensitivity_matrix(
    inner: &mut Eq106GpuBuffersInner,
    line_samples_pipeline: &wgpu29::ComputePipeline,
    assemble_pipeline: &wgpu29::ComputePipeline,
    analytic_pipeline: &wgpu29::ComputePipeline,
    evaluate_pipeline: &wgpu29::ComputePipeline,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    extracted: &ExtractedEq106Input,
    channel: &Eq106GpuReadbackChannel,
    snapshot: &GravityRequestSnapshot,
) {
    let Some(capture_id) = extracted.batch_capture_id else {
        return;
    };
    let column_count = extracted.sensitivity_sources.len();
    let target_count = extracted.target_snapshots.len();
    if column_count == 0
        || extracted.sensitivity_source_counts.len() != column_count
        || target_count == 0
        || extracted.batch_elements.is_empty()
    {
        return;
    }
    let key = (
        snapshot.epoch,
        capture_id ^ (column_count as u64).rotate_left(37) ^ 0x1061_06ba_7c4d_5e6f,
    );
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

    let compact_column_size = target_count as u64 * 16;
    let matrix_size = compact_column_size * column_count as u64;
    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("eq106_sensitivity_matrix_staging"),
        size: matrix_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let source_buffers = extracted
        .sensitivity_sources
        .iter()
        .map(|bytes| {
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("eq106_sensitivity_unit_source"),
                contents: bytes,
                usage: BufferUsages::STORAGE,
            })
        })
        .collect::<Vec<_>>();

    let resource_count = column_count * extracted.batch_elements.len();
    let mut uniforms = Vec::with_capacity(resource_count);
    let mut bind_groups = Vec::with_capacity(resource_count);
    for (column, source) in source_buffers.iter().enumerate() {
        for element in &extracted.batch_elements {
                let bytes = uniform_bytes(
                element.line_origin,
                element.line_origin,
                element.line_direction,
                extracted.sensitivity_source_counts[column],
                extracted.radius,
                element.line_limit,
                element.taylor_order,
                extracted.density_mode_count,
                    1,
                false,
                true,
                element.target_count,
                element.target_offset,
            );
            let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("eq106_sensitivity_element_uniform"),
                contents: &{
                    let mut data = vec![0_u8; 96 * 256];
                    data[..bytes.len()].copy_from_slice(&bytes);
                    data
                },
                usage: BufferUsages::STORAGE,
            });
            let bind_group = render_device.create_bind_group(
                "eq106_sensitivity_element_bg",
                &inner.layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: source.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: inner.quadrature.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: inner.spectrum.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: inner.output.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: inner.operator_tensor.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: inner.line_samples.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: inner.density_modes.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: inner.psi_operator.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 9,
                        resource: inner.targets.as_entire_binding(),
                    },
                ],
            );
            uniforms.push(uniform);
            bind_groups.push(bind_group);
        }
    }

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("eq106_sensitivity_matrix_encoder"),
    });
    let elements_per_column = extracted.batch_elements.len();
    let mut spectrum_encoding_ms = 0.0;
    let mut evaluation_encoding_ms = 0.0;
    for column in 0..column_count {
        for (element_index, element) in extracted.batch_elements.iter().enumerate() {
            let bind_group = &bind_groups[column * elements_per_column + element_index];
            let spectrum_encoding_started = Instant::now();
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("eq106_sensitivity_line_samples_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(line_samples_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(QUADRATURE_COUNT.div_ceil(64), 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("eq106_sensitivity_assemble_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(assemble_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(
                    (taylor_coefficient_count(element.taylor_order) * FREQUENCY_COUNT).div_ceil(64),
                    1,
                    1,
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("eq106_sensitivity_analytic_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(analytic_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(FREQUENCY_COUNT.div_ceil(64), 1, 1);
            }
            spectrum_encoding_ms += spectrum_encoding_started.elapsed().as_secs_f64() * 1.0e3;
            let evaluation_encoding_started = Instant::now();
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("eq106_sensitivity_evaluate_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(evaluate_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                let (width, height) = target_dispatch_grid(element.target_count);
                pass.dispatch_workgroups(width, height, 1);
            }
            evaluation_encoding_ms += evaluation_encoding_started.elapsed().as_secs_f64() * 1.0e3;
        }
        encoder.copy_buffer_to_buffer(
            &inner.output,
            0,
            &staging,
            column as u64 * compact_column_size,
            compact_column_size,
        );
    }
    render_queue.submit([encoder.finish()]);
    inner.segment_id = inner.segment_id.wrapping_add(resource_count as u32).max(1);
    inner.spectrum_ready = false;

    let shared = Arc::clone(&channel.data);
    let in_flight = Arc::clone(&channel.in_flight);
    let mapped_staging = staging.clone();
    let snapshots = extracted.target_snapshots.clone();
    let element_count = extracted.batch_elements.len() as u32;
    let sensitivity_source_hash = extracted.sensitivity_source_hash;
    let sensitivity_basis_hash = extracted.sensitivity_basis_hash;
    let sensitivity_configuration_hash = eq106_sensitivity_configuration_hash();
    let readback_started = Instant::now();
    mapped_staging
        .slice(..)
        .map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let view = staging.slice(..).get_mapped_range();
                let values = bytes_to_f32x4(&view);
                let cpu_readback_wait_ms = readback_started.elapsed().as_secs_f64() * 1.0e3;
                if let Ok(mut guard) = shared.lock() {
                    *guard = Some(Eq106ReadbackPacket {
                        partial_sums: values,
                        snapshots,
                        batch_capture_id: Some(capture_id),
                        sensitivity_column_count: column_count as u32,
                        sensitivity_source_hash,
                        sensitivity_basis_hash,
                        sensitivity_configuration_hash,
                        timings: Eq106TimingSample {
                            spectrum_build_ms: Some(spectrum_encoding_ms),
                            target_evaluation_ms: Some(evaluation_encoding_ms),
                            cpu_readback_wait_ms,
                            target_count: target_count as u32,
                            spectral_element_count: element_count,
                            dispatch_count: 1,
                            spectrum_rebuild_count: column_count as u32 * element_count,
                            ..default()
                        },
                    });
                }
                drop(view);
                staging.unmap();
            }
            in_flight.store(false, Ordering::Release);
        });
}
