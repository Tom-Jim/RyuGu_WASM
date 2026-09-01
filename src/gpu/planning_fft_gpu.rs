// Included by planning.rs: render-world, GPU-resident 56-column FFT cache.
// A single padded workspace is reused between bounded batches of columns. Each transform covers
// thousands of independent FFT lines in parallel, without allocating 56 full
// padded workspaces (~1.8 GiB) or queueing an uninterruptible all-column batch.
const FFT_GPU_SOURCE_CHUNK: usize = 16_384;
const FFT_GPU_CELLS: u64 = 64 * 64 * 64 + 16 * 16 * 16;
const FFT_GPU_BANK_BYTES: u64 = 56 * FFT_GPU_CELLS * 8;
const FFT_GPU_WORK_BYTES: u64 = 128 * 128 * 128 * 16;
const FFT_GPU_ENTRIES: [&str; 7] = [
    "deposit",
    "seed_kernel",
    "transform",
    "load_column",
    "convolve",
    "store_column",
    "combine",
];

#[derive(Resource)]
struct PlanningFftPipelines([CachedComputePipelineId; 7]);
fn fft_gpu_layout() -> [BindGroupLayoutEntry; 8] {
    [
        uniform_entry(0),
        storage_ro_entry(1),
        storage_rw_entry(2),
        storage_rw_entry(3),
        storage_rw_entry(4),
        storage_ro_entry(5),
        storage_rw_entry(6),
        storage_ro_entry(7),
    ]
}
impl FromWorld for PlanningFftPipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::PlanningFftBasis,
        );
        Self(FFT_GPU_ENTRIES.map(|entry| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("planning_fft_{entry}").into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "planning_fft_basis_bgl",
                    &fft_gpu_layout(),
                )],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
            })
        }))
    }
}
#[derive(Clone, Copy)]
enum FftGpuStage {
    Deposit { level: usize, source: usize },
    Kernel { level: usize },
    Column { level: usize, column: u32 },
    Ready,
}
#[derive(Clone, Copy)]
struct PlanningPreparationCost {
    cpu_ms: f64,
    submission_ms: f64,
    completion_ms: f64,
    decode_ms: f64,
    all_ms: Option<f64>,
    basis_ms: Option<f64>,
    dispatches: u32,
}
impl Default for PlanningPreparationCost {
    fn default() -> Self {
        Self {
            cpu_ms: 0.0,
            submission_ms: 0.0,
            completion_ms: 0.0,
            decode_ms: 0.0,
            all_ms: Some(0.0),
            basis_ms: Some(0.0),
            dispatches: 0,
        }
    }
}
impl PlanningPreparationCost {
    fn add(&mut self, other: Self) {
        self.cpu_ms += other.cpu_ms;
        self.submission_ms += other.submission_ms;
        self.completion_ms += other.completion_ms;
        self.decode_ms += other.decode_ms;
        self.all_ms = self.all_ms.zip(other.all_ms).map(|(a, b)| a + b);
        self.basis_ms = self.basis_ms.zip(other.basis_ms).map(|(a, b)| a + b);
        self.dispatches += other.dispatches;
    }
}
struct PlanningPreparationCompletion {
    cost: PlanningPreparationCost,
    valid: bool,
}
struct FftGpuCache {
    batch_id: u64,
    stage: FftGpuStage,
    combined_model: Option<u32>,
    bank: Buffer,
    work: Buffer,
    kernel: Buffer,
    sources: Buffer,
    combined: Buffer,
    roots: Buffer,
    fence: Buffer,
    pending: Option<Arc<std::sync::Mutex<Option<PlanningPreparationCompletion>>>>,
    uncharged: PlanningPreparationCost,
    completed_work: f64,
    pending_work: f64,
    completed_submissions: u32,
    column_batch: u32,
    pending_columns: u32,
}
#[derive(Resource, Default)]
struct PlanningFftGpu(Option<FftGpuCache>);

impl Drop for FftGpuCache {
    fn drop(&mut self) {
        // Cancellation/method changes happen after submission. WebGPU retains
        // already submitted work; destroy releases these large banks as soon
        // as that work drains instead of waiting for browser wrapper GC.
        for buffer in [
            &self.bank,
            &self.work,
            &self.kernel,
            &self.sources,
            &self.combined,
            &self.roots,
        ] {
            buffer.destroy();
        }
        // A map callback may still own the tiny fence; it must finish normally.
    }
}

impl PlanningFftGpu {
    // None = asynchronous work remains. No CPU transform, GPU poll(wait),
    // full-grid readback or CPU density combination is allowed in this path.
    fn prepare(
        &mut self,
        input: &ExtractedPlanningInput,
        shared: &PlanningSharedGpuBuffersInner,
        pipelines: &PlanningFftPipelines,
        cache: &PipelineCache,
        device: &RenderDevice,
        queue: &RenderQueue,
        channel: &PlanningGpuReadbackChannel,
        timestamp_pool: &crate::gpu::planning_timestamps::PlanningTimestampPool,
    ) -> Option<Buffer> {
        let batch = input.batch.as_ref()?;
        let request = &input.request;
        for &pipeline in &pipelines.0 {
            if cache.get_compute_pipeline(pipeline).is_none() {
                report_planning_pipeline_failure(
                    cache,
                    pipeline,
                    request.request_id,
                    "GPU FFT basis",
                    channel,
                );
                return None;
            }
        }
        // Yield before allocating a workspace or advancing the job while the
        // shared bank is busy / its allocation error scopes are unresolved.
        let std::task::Poll::Ready(mut queries) = timestamp_pool.acquire(
            device,
            queue,
            crate::gpu::planning_timestamps::PLANNING_TIMESTAMP_MAX_PASSES,
        ) else {
            return None;
        };
        let cpu_started = bevy::platform::time::Instant::now();
        if self
            .0
            .as_ref()
            .is_none_or(|state| state.batch_id != batch.batch_id)
        {
            // Release the previous bank before allocating its replacement,
            // rather than briefly retaining two complete GPU workspaces.
            self.0 = None;
            // Fail explicitly on insufficient device limits; never fall back to
            // synchronously computing these large bases on the event-loop CPU.
            let limits = device.limits();
            if limits.max_storage_buffer_binding_size < FFT_GPU_BANK_BYTES
                || limits.max_buffer_size < FFT_GPU_BANK_BYTES
                || limits.max_storage_buffers_per_shader_stage < 7
            {
                if let Ok(mut error) = channel.error.try_lock() {
                    *error = Some((request.request_id, "GPU FFT requires a 114 MiB basis binding and 7 storage bindings; CPU FFT fallback is disabled".into()));
                }
                return None;
            }
            let buffer = |label, size| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let mut roots = Vec::with_capacity(128 * 16);
            for i in 0..128 {
                let angle = -std::f64::consts::TAU * f64::from(i) / 128.0;
                for value in [angle.cos(), angle.sin()] {
                    let hi = value as f32;
                    roots.extend_from_slice(&hi.to_le_bytes());
                    roots.extend_from_slice(&((value - f64::from(hi)) as f32).to_le_bytes());
                }
            }
            self.0 = Some(FftGpuCache {
                batch_id: batch.batch_id,
                stage: FftGpuStage::Deposit {
                    level: 0,
                    source: 0,
                },
                combined_model: None,
                bank: buffer("planning_fft_56_basis_hi_lo", FFT_GPU_BANK_BYTES),
                work: buffer("planning_fft_complex_hi_lo", FFT_GPU_WORK_BYTES),
                kernel: buffer("planning_fft_kernel_hi_lo", FFT_GPU_WORK_BYTES),
                sources: buffer(
                    "planning_fft_source_chunk",
                    FFT_GPU_SOURCE_CHUNK as u64 * 20,
                ),
                combined: buffer("planning_fft_combined_potential", FFT_GPU_CELLS * 4),
                roots: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("planning_fft_roots_hi_lo"),
                    contents: &roots,
                    usage: BufferUsages::STORAGE,
                }),
                fence: device.create_buffer(&BufferDescriptor {
                    label: Some("planning_fft_reusable_fence"),
                    size: u64::from(crate::gpu::planning_timestamps::PLANNING_TIMESTAMP_MAX_PASSES)
                        * 16,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                pending: None,
                uncharged: PlanningPreparationCost::default(),
                completed_work: 0.0,
                pending_work: 0.0,
                completed_submissions: 0,
                column_batch: 1,
                pending_columns: 0,
            });
        }
        let state = self.0.as_mut()?;
        if let Some(pending) = &state.pending {
            let completion = pending.try_lock().ok()?.take()?;
            if !completion.valid {
                if let Ok(mut error) = channel.error.try_lock() {
                    *error = Some((
                        request.request_id,
                        "GPU FFT preparation readback failed".into(),
                    ));
                }
                return None;
            }
            if state.pending_columns > 0
                && let Some(ms) = completion.cost.all_ms.filter(|ms| *ms > 0.0)
            {
                let per_column = ms / f64::from(state.pending_columns);
                // Timestamp the whole submission and change the next
                // batch conservatively. Slow work shrinks immediately;
                // growth requires a comfortable margin below the target.
                state.column_batch = if ms > PLANNING_GPU_MAX_SUBMISSION_MS {
                    PLANNING_GPU_MIN_BATCH
                } else if per_column > PLANNING_GPU_TARGET_SUBMISSION_MS {
                    state
                        .column_batch
                        .saturating_sub(1)
                        .max(PLANNING_GPU_MIN_BATCH)
                } else if ms < PLANNING_GPU_TARGET_SUBMISSION_MS * 0.45 {
                    state
                        .column_batch
                        .saturating_add(1)
                        .min(PLANNING_GPU_MAX_BATCH)
                } else {
                    state.column_batch
                };
            }
            state.uncharged.add(completion.cost);
            state.pending = None;
            state.completed_work += state.pending_work;
            state.completed_submissions += 1;
            let budget = PlanningOperationBudget::for_method(
                ActiveGravityMethod::MmfftCompressed,
                batch.source_count,
                batch.samples_per_candidate,
                batch.candidate_count,
            );
            if let Ok(mut progress) = channel.preparation.try_lock() {
                *progress = Some(PlanningGpuPreparation {
                    request_id: request.request_id,
                    completed_submissions: state.completed_submissions,
                    basis_fraction: (state.completed_work / budget.basis.max(1.0)).clamp(0.0, 1.0),
                    status: match state.stage {
                        FftGpuStage::Deposit { level, source } => format!(
                            "GPU FFT: level {}/2 source deposition {}/{}; 56 bases in VRAM",
                            level + 1,
                            source,
                            batch.basis_records.len()
                        ),
                        FftGpuStage::Kernel { level } => {
                            format!("GPU FFT: level {}/2 Newton-kernel FFT", level + 1)
                        }
                        FftGpuStage::Column { level, column } => format!(
                            "GPU FFT: level {}/2, basis {}/56 FFT convolution",
                            level + 1,
                            column + 1
                        ),
                        FftGpuStage::Ready => {
                            "GPU FFT: 56 bases ready; GPU density combination / target evaluation"
                                .into()
                        }
                    },
                });
                if let Some(current) = progress.as_mut() {
                    current.status = format!(
                        "Ns={}, Kρ={}, Nt={} · {}",
                        batch.source_count,
                        batch.density_model_count,
                        batch.samples_per_candidate,
                        current.status
                    );
                }
            }
        }
        if matches!(state.stage, FftGpuStage::Ready)
            && state.combined_model == Some(request.density_model)
        {
            return Some(state.combined.clone());
        }
        // (entry, level, axis, inverse, column, kernel buffer, source count)
        let mut operations: Vec<(usize, usize, u32, u32, u32, u32, u32)> = Vec::new();
        let basis_stage = !matches!(state.stage, FftGpuStage::Ready);
        state.pending_work = 0.0;
        state.pending_columns = 0;
        match state.stage {
            FftGpuStage::Deposit { level, source } => {
                let end = (source + FFT_GPU_SOURCE_CHUNK).min(batch.basis_records.len());
                let mut bytes = Vec::with_capacity((end - source) * 20);
                for record in &batch.basis_records[source..end] {
                    if record.voxel_index >= 56
                        || record.position_volume.iter().any(|v| !v.is_finite())
                        || record.position_volume[3] < 0.0
                    {
                        if let Ok(mut error) = channel.error.try_lock() {
                            *error = Some((
                                request.request_id,
                                "GPU FFT source upload contains an invalid source".into(),
                            ));
                        }
                        return None;
                    }
                    for value in record.position_volume {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    bytes.extend_from_slice(&record.voxel_index.to_le_bytes());
                }
                queue.write_buffer(&state.sources, 0, &bytes);
                operations.push((0, level, 0, 0, 0, 0, (end - source) as u32));
                state.pending_work = (end - source) as f64 * 8.0 * 32.0;
                state.stage = if end == batch.basis_records.len() {
                    FftGpuStage::Kernel { level }
                } else {
                    FftGpuStage::Deposit { level, source: end }
                };
            }
            FftGpuStage::Kernel { level } => {
                operations.push((1, level, 0, 0, 0, 1, 0));
                for axis in 0..3 {
                    operations.push((2, level, axis, 0, 0, 1, 0));
                }
                let side = if level == 0 { 128.0_f64 } else { 32.0_f64 };
                state.pending_work =
                    20.0 * side.powi(3) * side.powi(3).log2() + 32.0 * side.powi(3);
                state.stage = FftGpuStage::Column { level, column: 0 };
            }
            FftGpuStage::Column { level, column } => {
                // The coarse 32-point FFTs are tiny. Batch several columns in
                // one submission rather than spending one browser frame per
                // column. The large level adapts to pass timestamps (~8 ms).
                let count = (if level == 0 {
                    state.column_batch
                } else {
                    PLANNING_GPU_MAX_BATCH
                })
                .clamp(PLANNING_GPU_MIN_BATCH, PLANNING_GPU_MAX_BATCH)
                .min(56 - column);
                state.pending_columns = count;
                for current in column..column + count {
                    operations.push((3, level, 0, 0, current, 0, 0));
                    for axis in 0..3 {
                        operations.push((2, level, axis, 0, current, 0, 0));
                    }
                    operations.push((4, level, 0, 0, current, 0, 0));
                    for axis in 0..3 {
                        operations.push((2, level, axis, 1, current, 0, 0));
                    }
                    operations.push((5, level, 0, 0, current, 0, 0));
                }
                let side = if level == 0 { 128.0_f64 } else { 32.0_f64 };
                state.pending_work = f64::from(count)
                    * (40.0 * side.powi(3) * side.powi(3).log2() + 32.0 * side.powi(3));
                state.stage = if column + count < 56 {
                    FftGpuStage::Column {
                        level,
                        column: column + count,
                    }
                } else if level == 0 {
                    FftGpuStage::Deposit {
                        level: 1,
                        source: 0,
                    }
                } else {
                    FftGpuStage::Ready
                };
            }
            FftGpuStage::Ready => {
                for level in 0..2 {
                    operations.push((6, level, 0, 0, 0, 0, 0));
                }
                state.combined_model = Some(request.density_model);
            }
        }
        if let Some(queries) = &mut queries {
            queries.set_pass_count(operations.len() as u32);
        }
        // Even without optional timestamp support, map a tiny fence buffer.
        // At most one preparation submission is outstanding, and cancellation
        // stops the next submission. We never block the browser waiting for it.
        let staging = state.fence.clone();
        let layout = device.create_bind_group_layout("planning_fft_basis_bgl", &fft_gpu_layout());
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1.0e3;
        let submit_started = bevy::platform::time::Instant::now();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("planning_fft_stage"),
        });
        for (pass_index, &(entry, level, axis, inverse, column, buffer_kind, source_count)) in
            operations.iter().enumerate()
        {
            let n = input.payload.grid_sizes[level];
            let side = 2 * n;
            let cells = n * n * n;
            let offset = if level == 0 { 0 } else { 64 * 64 * 64 };
            let half = input.payload.half_extents[level];
            let spacing = 2.0 * f64::from(half) / f64::from(n);
            let mut bytes = Vec::with_capacity(64);
            for value in [
                n,
                side,
                axis,
                inverse,
                column,
                offset * 56,
                offset,
                source_count,
                request.density_model,
                buffer_kind,
                0,
                0,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in [
                spacing as f32,
                (spacing - f64::from(spacing as f32)) as f32,
                half,
                G,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("planning_fft_stage_params"),
                contents: &bytes,
                usage: BufferUsages::UNIFORM,
            });
            let bound = [
                &uniform,
                &state.sources,
                &state.bank,
                &state.work,
                &state.kernel,
                &shared.densities,
                &state.combined,
                &state.roots,
            ];
            let entries: Vec<_> = bound
                .iter()
                .enumerate()
                .map(|(binding, buffer)| BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let group = device.create_bind_group("planning_fft_stage_bg", &layout, &entries);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(FFT_GPU_ENTRIES[entry]),
                timestamp_writes: queries.as_ref().map(|q| q.writes(pass_index as u32)),
            });
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipelines.0[entry])
                    .expect("checked FFT pipeline"),
            );
            pass.set_bind_group(0, &group, &[]);
            let groups = match entry {
                0 => source_count.div_ceil(128),
                2 => side * side,
                5 | 6 => cells.div_ceil(128),
                _ => (side * side * side).div_ceil(128),
            };
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if let Some(q) = &queries {
            q.resolve_into(&mut encoder, &staging, 0);
        } else {
            encoder.clear_buffer(&staging, 0, None);
        }
        queue.submit([encoder.finish()]);
        let submission_ms = submit_started.elapsed().as_secs_f64() * 1.0e3;
        let completion_started = bevy::platform::time::Instant::now();
        let pending = Arc::new(std::sync::Mutex::new(None));
        state.pending = Some(Arc::clone(&pending));
        let mapped = staging.clone();
        staging.slice(..).map_async(MapMode::Read, move |result| {
            let completion_ms = completion_started.elapsed().as_secs_f64() * 1.0e3;
            let decode_started = bevy::platform::time::Instant::now();
            let all_ms = if result.is_ok() {
                let view = mapped.slice(..).get_mapped_range();
                let times = queries
                    .as_ref()
                    .and_then(|q| q.decode(&view))
                    .map(|values| values.into_iter().sum());
                drop(view);
                mapped.unmap();
                times
            } else {
                None
            };
            if let Ok(mut slot) = pending.lock() {
                *slot = Some(PlanningPreparationCompletion {
                    valid: result.is_ok(),
                    cost: PlanningPreparationCost {
                        cpu_ms,
                        submission_ms,
                        completion_ms,
                        decode_ms: decode_started.elapsed().as_secs_f64() * 1.0e3,
                        all_ms,
                        basis_ms: if basis_stage { all_ms } else { Some(0.0) },
                        dispatches: operations.len() as u32,
                    },
                });
            }
        });
        None
    }
    fn take_cost(&mut self) -> PlanningPreparationCost {
        self.0
            .as_mut()
            .map(|s| std::mem::take(&mut s.uncharged))
            .unwrap_or_default()
    }
}
