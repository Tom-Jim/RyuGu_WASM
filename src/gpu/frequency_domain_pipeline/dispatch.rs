fn dispatch_frequency_domain(
    mut buffers: ResMut<FrequencyDomainGpuBuffers>,
    pipelines: Option<Res<FrequencyDomainComputePipeline>>,
    cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Res<ExtractedFrequencyDomainInput>,
    channel: Res<FrequencyDomainGpuReadbackChannel>,
) {
    let Some(pipelines) = pipelines else { return };
    if !extracted.enabled {
        return;
    }
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        // Source construction and fixed-trajectory capture may become visible
        // in adjacent frames. Wait rather than creating empty GPU buffers.
        return;
    };
    if extracted.source_count == 0 {
        if let Ok(mut slot) = channel.pipeline_error.try_lock()
            && slot.is_none()
        {
            *slot = Some(
                "Frequency-domain algorithm is enabled but no aggregated source has been uploaded."
                    .into(),
            );
        }
        return;
    }
    report_frequency_domain_pipeline_errors(&cache, &pipelines, &channel);
    let (Some(density_spectra), Some(assemble), Some(evaluate)) = (
        cache.get_compute_pipeline(pipelines.density_spectrum_id),
        cache.get_compute_pipeline(pipelines.assemble_id),
        cache.get_compute_pipeline(pipelines.evaluate_id),
    ) else {
        return;
    };
    let element_capacity = extracted.batch_elements.len().max(1) as u32;
    // The shader parameter storage is intentionally fixed at 256 records.
    // Reject oversized batches before constructing a buffer whose indexing
    // would otherwise read beyond the uploaded parameter data.
    if extracted.batch_elements.len() > 256 {
        error!(
            target: "frequency_domain",
            trajectory_blocks = extracted.batch_elements.len(),
            "Frequency-domain algorithm batch exceeds the 256-record parameter capacity"
        );
        return;
    }
    if buffers.0.as_ref().is_some_and(|inner| {
        inner.source_count != extracted.source_count
            || inner.source_radius.to_bits() != extracted.radius.to_bits()
            || inner.target_count != extracted.observation_count
            || inner.element_capacity != element_capacity
    }) {
        // Rebuild only when buffer capacities or binding shapes change.
        buffers.0 = None;
    }
    if buffers.0.is_none() {
        let Some(source_bytes) = extracted.sources.as_ref() else {
            return;
        };
        let sources = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("frequency_domain_sources"),
            contents: source_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let quadrature_bytes = reciprocal_space_quadrature_bytes(extracted.radius);
        let quadrature = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("frequency_domain_quadrature_lut"),
            contents: &quadrature_bytes,
            usage: BufferUsages::STORAGE,
        });
        let density_mode_bytes = vec![0_u8; 544 * 16];
        let density_modes = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("frequency_domain_density_fourier_modes"),
            contents: &density_mode_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let targets = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("frequency_domain_targets"),
            contents: &extracted.target_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let spectrum = render_device.create_buffer(&BufferDescriptor {
            label: Some("frequency_domain_spectrum"),
            size: u64::from(QUADRATURE_COUNT) * 32 * u64::from(element_capacity),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let density_spectra = render_device.create_buffer(&BufferDescriptor {
            label: Some("frequency_domain_density_spectra"),
            size: u64::from(QUADRATURE_COUNT) * 16 * u64::from(element_capacity),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let target_count = extracted.observation_count.max(1);
        let output_size = OUTPUT_BYTES * target_count as u64;
        let output = render_device.create_buffer(&BufferDescriptor {
            label: Some("frequency_domain_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("frequency_domain_staging"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = render_device.create_bind_group_layout(
            "frequency_domain_complex_bgl_runtime",
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
        );
        buffers.0 = Some(FrequencyDomainGpuBuffersInner {
            targets,
            output,
            staging,
            output_size,
            layout,
            sources,
            quadrature,
            spectrum,
            density_spectra,
            density_modes,
            source_count: extracted.source_count,
            source_radius: extracted.radius,
            element_capacity,
            source_hash: extracted.source_hash,
            target_count,
            last_submitted: None,
        });
    }

    let inner = buffers.0.as_mut().expect("FrequencyDomain GPU buffers initialized");
    if !extracted.sensitivity_sources.is_empty() {
        dispatch_frequency_domain_sensitivity_matrix(
            inner,
            density_spectra,
            assemble,
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
        inner.source_hash = extracted.source_hash;
        inner.last_submitted = None;
    }
    if let Some(capture_id) = extracted.batch_capture_id
        && !extracted.batch_elements.is_empty()
    {
        if channel.rebuild_requested.swap(false, Ordering::AcqRel) {
            inner.last_submitted = None;
        }
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
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("frequency_domain_trajectory_batch_encoder"),
        });
        // The WGSL uniform block has a fixed 256-element array. Keep the
        // binding range large enough for the declared block even when a
        // request contains fewer active trajectory elements.
        let mut parameter_bytes = vec![0_u8; 48 * 256];
        for (element_index, element) in extracted.batch_elements.iter().enumerate() {
            let bytes = uniform_bytes(
                element.trajectory_origin,
                extracted.source_count,
                element_index as u32 + 1,
                0,
                element.target_count,
                element.target_offset,
            );
            let offset = element_index * 48;
            parameter_bytes[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        let parameter_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("frequency_domain_trajectory_segment_params"),
            contents: &parameter_bytes,
            usage: BufferUsages::STORAGE,
        });
        let bind_group = render_device.create_bind_group(
            "frequency_domain_trajectory_batch_bg",
            &inner.layout,
            &[
                BindGroupEntry { binding: 0, resource: parameter_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: inner.sources.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: inner.quadrature.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: inner.spectrum.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: inner.output.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: inner.density_spectra.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: inner.density_modes.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: inner.targets.as_entire_binding() },
            ],
        );
        let trajectory_block_count = extracted.batch_elements.len() as u32;
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("frequency_domain_trajectory_density_spectra_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(density_spectra);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(QUADRATURE_COUNT, trajectory_block_count, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("frequency_domain_trajectory_spectrum_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(assemble);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(QUADRATURE_COUNT.div_ceil(64), trajectory_block_count, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("frequency_domain_trajectory_evaluate_batch"), timestamp_writes: None,
            });
            pass.set_pipeline(evaluate);
            pass.set_bind_group(0, &bind_group, &[]);
            // Each y workgroup evaluates one Laplace-frequency observation
            // for each trajectory block. Every observation integrates the
            // complete T_gamma over all uploaded samples; there is no spatial
            // target specialization.
            pass.dispatch_workgroups(1, inner.target_count, trajectory_block_count);
        }
        encoder.copy_buffer_to_buffer(&inner.output, 0, &inner.staging, 0, inner.output_size);
        render_queue.submit([encoder.finish()]);

        let shared = Arc::clone(&channel.data);
        let in_flight = Arc::clone(&channel.in_flight);
        let submitted_at = Arc::clone(&channel.submitted_at);
        let error_slot = Arc::clone(&channel.pipeline_error);
        let rebuild_requested = Arc::clone(&channel.rebuild_requested);
        let staging = inner.staging.clone();
        let map_staging = staging.clone();
        let observation_count = extracted.observation_count;
        let output_size = inner.output_size as usize;
        let target_count = inner.target_count;
        let element_count = extracted.batch_elements.len() as u32;
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
                        let cpu_readback_wait_ms =
                            readback_started.elapsed().as_secs_f64() * 1.0e3;
                        let timings = FrequencyDomainTimingSample {
                            cpu_readback_wait_ms,
                            target_count,
                            dispatch_count: 1,
                            spectrum_rebuild_count: element_count,
                            ..default()
                        };
                        if let Ok(mut guard) = shared.lock() {
                            *guard = Some(FrequencyDomainReadbackPacket {
                                partial_sums: values,
                                observation_count,
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
                    Err(error) => {
                        rebuild_requested.store(true, Ordering::Release);
                        if let Ok(mut slot) = error_slot.lock()
                            && slot.is_none()
                        {
                            *slot = Some(format!(
                                "Frequency-domain algorithm GPU batch readback failed: {error:?}"
                            ));
                        }
                    }
                }
                if let Ok(mut submitted) = submitted_at.lock() {
                    submitted.take();
                }
                in_flight.store(false, Ordering::Release);
            });
        return;
    }
    if let Ok(mut slot) = channel.pipeline_error.try_lock()
        && slot.is_none()
    {
        *slot = Some(
            "Frequency-domain algorithm requires a complete fixed equation-(185) trajectory."
                .into(),
        );
    }
}
