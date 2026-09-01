// GPU fixed-depth octree, compensated unit-density moments and 56 response
// columns. The CPU uploads bounded source chunks; no CPU sorting, P2M/M2M,
// target traversal, P2P or RHS combination occurs in the benchmark path.
const FMM_GPU_NODES: u64 = 37_449;
const FMM_GPU_CAPACITY: u32 = 8192;
const FMM_GPU_SOURCE_CHUNK_INITIAL: usize = 1_024;
const FMM_GPU_ENTRIES: [&str; 3] = ["p2m", "m2m", "response_basis"];
fn fmm_gpu_layout() -> [BindGroupLayoutEntry; 9] {
    [
        uniform_entry(0),
        storage_ro_entry(1),
        storage_rw_entry(2),
        storage_rw_entry(3),
        storage_rw_entry(4),
        storage_rw_entry(5),
        storage_rw_entry(6),
        storage_ro_entry(7),
        storage_rw_entry(8),
    ]
}
#[derive(Resource)]
struct PlanningFmmPipelines([CachedComputePipelineId; 3]);
impl FromWorld for PlanningFmmPipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let shader = crate::wgsl::load(
            world.resource::<AssetServer>(),
            crate::wgsl::EmbeddedShader::PlanningFmmBasis,
        );
        Self(FMM_GPU_ENTRIES.map(|entry| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("planning_fmm_{entry}").into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "planning_fmm_basis_bgl",
                    &fmm_gpu_layout(),
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
enum FmmGpuStage {
    Sources(usize),
    Merge(u32),
    Targets(u32),
    Ready,
}
struct FmmGpuCache {
    batch_id: u64,
    stage: FmmGpuStage,
    source_upload: Buffer,
    particles: Buffer,
    links: Buffer,
    heads: Buffer,
    moments_a: Buffer,
    moments_b: Buffer,
    responses: Buffer,
    fence: Buffer,
    response_start: u32,
    response_count: u32,
    response_tile: u32,
    source_chunk: usize,
    pending_sources: u32,
    pending_targets: u32,
    pending: Option<Arc<std::sync::Mutex<Option<PlanningPreparationCompletion>>>>,
    uncharged: PlanningPreparationCost,
    completed_work: f64,
    pending_work: f64,
    completed_submissions: u32,
}
#[derive(Resource, Default)]
struct PlanningFmmGpu(Option<FmmGpuCache>);

impl Drop for FmmGpuCache {
    fn drop(&mut self) {
        // No later request may submit these banks after a cache replacement.
        // Already submitted work retains them until it completes.
        for buffer in [
            &self.source_upload,
            &self.particles,
            &self.links,
            &self.heads,
            &self.moments_a,
            &self.moments_b,
            &self.responses,
        ] {
            buffer.destroy();
        }
        // Leave the tiny readback fence alive for any outstanding map callback.
    }
}
impl PlanningFmmGpu {
    fn prepare(
        &mut self,
        input: &ExtractedPlanningInput,
        shared: &PlanningSharedGpuBuffersInner,
        pipelines: &PlanningFmmPipelines,
        cache: &PipelineCache,
        device: &RenderDevice,
        queue: &RenderQueue,
        channel: &PlanningGpuReadbackChannel,
        timestamp_pool: &crate::gpu::planning_timestamps::PlanningTimestampPool,
    ) -> Option<(Buffer, u32)> {
        let batch = input.batch.as_ref()?;
        let request = &input.request;
        for &pipeline in &pipelines.0 {
            if cache.get_compute_pipeline(pipeline).is_none() {
                report_planning_pipeline_failure(
                    cache,
                    pipeline,
                    request.request_id,
                    "GPU FMM basis",
                    channel,
                );
                return None;
            }
        }
        let std::task::Poll::Ready(queries) = timestamp_pool.acquire(device, queue, 1) else {
            return None;
        };
        let started = bevy::platform::time::Instant::now();
        let response_start = if batch.state_count() <= FMM_GPU_CAPACITY as usize {
            0
        } else {
            request.candidate_start * batch.samples_per_candidate
        };
        let response_count = if batch.state_count() <= FMM_GPU_CAPACITY as usize {
            batch.state_count() as u32
        } else {
            request.candidate_count * batch.samples_per_candidate
        };
        if self.0.as_ref().is_none_or(|s| s.batch_id != batch.batch_id) {
            self.0 = None;
            let limits = device.limits();
            let largest = (batch.basis_records.len() as u64 * 16)
                .max(FMM_GPU_NODES * 56 * 12 * 4)
                .max(u64::from(response_count) * 56 * 96);
            if largest > u64::from(limits.max_storage_buffer_binding_size)
                || largest > limits.max_buffer_size
                || limits.max_storage_buffers_per_shader_stage < 8
                || response_count > FMM_GPU_CAPACITY
            {
                if let Ok(mut error) = channel.error.try_lock() {
                    *error = Some((
                        request.request_id,
                        "GPU FMM exceeds storage limits; synchronous CPU fallback is disabled"
                            .into(),
                    ));
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
            self.0 = Some(FmmGpuCache {
                batch_id: batch.batch_id,
                stage: FmmGpuStage::Sources(0),
                source_upload: buffer("planning_fmm_source_chunk", 16_384_u64 * 20),
                particles: buffer(
                    "planning_fmm_particles",
                    batch.basis_records.len() as u64 * 16,
                ),
                links: buffer(
                    "planning_fmm_source_links",
                    batch.basis_records.len() as u64 * 8,
                ),
                heads: buffer("planning_fmm_leaf_heads", 32768 * 4),
                moments_a: buffer("planning_fmm_unit_mass_dipole", FMM_GPU_NODES * 56 * 8 * 4),
                moments_b: buffer(
                    "planning_fmm_unit_second_moments",
                    FMM_GPU_NODES * 56 * 12 * 4,
                ),
                responses: buffer(
                    "planning_fmm_56_target_bases",
                    u64::from((batch.state_count() as u32).min(FMM_GPU_CAPACITY)) * 56 * 96,
                ),
                fence: device.create_buffer(&BufferDescriptor {
                    label: Some("planning_fmm_reusable_fence"),
                    size: 16,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                response_start,
                response_count,
                response_tile: PLANNING_GPU_MIN_BATCH,
                source_chunk: FMM_GPU_SOURCE_CHUNK_INITIAL,
                pending_sources: 0,
                pending_targets: 0,
                pending: None,
                uncharged: PlanningPreparationCost::default(),
                completed_work: 0.0,
                pending_work: 0.0,
                completed_submissions: 0,
            });
        }
        let state = self.0.as_mut()?;
        if let Some(pending) = &state.pending {
            let completion = pending.try_lock().ok()?.take()?;
            if !completion.valid {
                if let Ok(mut error) = channel.error.try_lock() {
                    *error = Some((
                        request.request_id,
                        "GPU FMM preparation readback failed".into(),
                    ));
                }
                return None;
            }
            if state.pending_targets > 0 {
                // Keep a bounded queue while giving fast devices enough work.
                // Use GPU timestamps when available, not browser frame delay.
                if let Some(ms) = completion.cost.all_ms.filter(|ms| *ms > 0.0) {
                    if ms > PLANNING_GPU_MAX_SUBMISSION_MS {
                        state.response_tile = PLANNING_GPU_MIN_BATCH;
                    } else if ms > PLANNING_GPU_TARGET_SUBMISSION_MS {
                        state.response_tile = state
                            .response_tile
                            .saturating_sub(1)
                            .max(PLANNING_GPU_MIN_BATCH);
                    } else if ms < PLANNING_GPU_TARGET_SUBMISSION_MS * 0.45 {
                        state.response_tile = state
                            .response_tile
                            .saturating_add(1)
                            .min(PLANNING_GPU_MAX_BATCH);
                    }
                }
            }
            if state.pending_sources > 0 {
                if let Some(ms) = completion.cost.all_ms.filter(|ms| *ms > 0.0) {
                    state.source_chunk = if ms > PLANNING_GPU_MAX_SUBMISSION_MS {
                        (state.source_chunk / 2).max(1)
                    } else if ms < PLANNING_GPU_TARGET_SUBMISSION_MS * 0.45 {
                        state.source_chunk.saturating_mul(2).min(16_384)
                    } else {
                        state.source_chunk
                    };
                }
            }
            state.uncharged.add(completion.cost);
            state.pending = None;
            state.completed_work += state.pending_work;
            state.completed_submissions += 1;
            let budget = PlanningOperationBudget::for_method(
                ActiveGravityMethod::Fmm,
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
                        FmmGpuStage::Sources(source) => format!(
                            "GPU FMM: source P2M/binning {}/{}",
                            source,
                            batch.basis_records.len()
                        ),
                        FmmGpuStage::Merge(level) => format!("GPU FMM: 56-basis M2M level {level}"),
                        FmmGpuStage::Targets(target) => format!(
                            "GPU FMM: 56-basis M2L/P2P targets {}/{}",
                            target - state.response_start,
                            state.response_count
                        ),
                        FmmGpuStage::Ready => {
                            "GPU FMM: response bases ready; GPU density combination".into()
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
        // Small/fixed-target sweeps retain all responses across every RHS.
        // Larger stress jobs stream a bounded target window while retaining
        // all 56 source moment banks. Rebuilt target windows are measured.
        if matches!(state.stage, FmmGpuStage::Ready) {
            if state.response_start == response_start && state.response_count == response_count {
                return Some((state.responses.clone(), state.response_start));
            }
            state.response_start = response_start;
            state.response_count = response_count;
            state.stage = FmmGpuStage::Targets(response_start);
        }
        state.pending_targets = 0;
        state.pending_sources = 0;
        let (entry, source_start, source_count, level, target_start, target_count, groups) =
            match state.stage {
                FmmGpuStage::Sources(source) => {
                    let end = (source + state.source_chunk).min(batch.basis_records.len());
                    let mut bytes = Vec::with_capacity((end - source) * 20);
                    for record in &batch.basis_records[source..end] {
                        if record.voxel_index >= 56
                            || record.position_volume.iter().any(|v| !v.is_finite())
                            || record.position_volume[3] < 0.0
                        {
                            if let Ok(mut error) = channel.error.try_lock() {
                                *error =
                                    Some((request.request_id, "Invalid GPU FMM source".into()));
                            }
                            return None;
                        }
                        for value in record.position_volume {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                        bytes.extend_from_slice(&record.voxel_index.to_le_bytes());
                    }
                    queue.write_buffer(&state.source_upload, 0, &bytes);
                    state.pending_sources = (end - source) as u32;
                    state.pending_work = (end - source) as f64 * 640.0;
                    state.stage = if end == batch.basis_records.len() {
                        FmmGpuStage::Merge(4)
                    } else {
                        FmmGpuStage::Sources(end)
                    };
                    (
                        0,
                        source as u32,
                        (end - source) as u32,
                        0,
                        0,
                        0,
                        [(end - source).div_ceil(128) as u32, 1],
                    )
                }
                FmmGpuStage::Merge(level) => {
                    let nodes = 1u32 << (3 * level);
                    state.pending_work = f64::from(nodes) * 56.0 * 8.0 * 10.0 * 16.0;
                    state.stage = if level == 0 {
                        FmmGpuStage::Targets(state.response_start)
                    } else {
                        FmmGpuStage::Merge(level - 1)
                    };
                    (1, 0, 0, level, 0, 0, [(nodes * 56).div_ceil(128), 1])
                }
                FmmGpuStage::Targets(start) => {
                    let count = (state.response_start + state.response_count - start)
                        .min(state.response_tile);
                    state.pending_targets = count;
                    state.pending_work = f64::from(count)
                        * (batch.basis_records.len() as f64 * 60.0 + 56.0 * 512.0 * 80.0);
                    state.stage = if start + count == state.response_start + state.response_count {
                        FmmGpuStage::Ready
                    } else {
                        FmmGpuStage::Targets(start + count)
                    };
                    (2, 0, 0, 0, start, count, [count, 56])
                }
                FmmGpuStage::Ready => unreachable!(),
            };
        let mut bytes = Vec::with_capacity(48);
        for value in [
            source_start,
            source_count,
            level,
            target_start,
            target_count,
            state.response_start,
            0,
            0,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [batch.eq106_source_radius, G, 0.05, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("planning_fmm_basis_params"),
            contents: &bytes,
            usage: BufferUsages::UNIFORM,
        });
        let bound = [
            &uniform,
            &state.source_upload,
            &state.particles,
            &state.links,
            &state.heads,
            &state.moments_a,
            &state.moments_b,
            &shared.positions,
            &state.responses,
        ];
        let entries: Vec<_> = bound
            .iter()
            .enumerate()
            .map(|(binding, b)| BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let layout = device.create_bind_group_layout("planning_fmm_basis_bgl", &fmm_gpu_layout());
        let group = device.create_bind_group("planning_fmm_basis_bg", &layout, &entries);
        let staging = state.fence.clone();
        let cpu_ms = started.elapsed().as_secs_f64() * 1e3;
        let submit_started = bevy::platform::time::Instant::now();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("planning_fmm_basis_stage"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(FMM_GPU_ENTRIES[entry]),
                timestamp_writes: queries.as_ref().map(|q| q.writes(0)),
            });
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipelines.0[entry])
                    .expect("checked FMM pipeline"),
            );
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        if let Some(q) = &queries {
            q.resolve_into(&mut encoder, &staging, 0);
        } else {
            encoder.clear_buffer(&staging, 0, None);
        }
        queue.submit([encoder.finish()]);
        let submission_ms = submit_started.elapsed().as_secs_f64() * 1e3;
        let complete_started = bevy::platform::time::Instant::now();
        let pending = Arc::new(std::sync::Mutex::new(None));
        state.pending = Some(Arc::clone(&pending));
        let mapped = staging.clone();
        staging.slice(..).map_async(MapMode::Read, move |result| {
            let completion_ms = complete_started.elapsed().as_secs_f64() * 1e3;
            let decode_started = bevy::platform::time::Instant::now();
            let all_ms = if result.is_ok() {
                let view = mapped.slice(..).get_mapped_range();
                let ms = queries
                    .as_ref()
                    .and_then(|q| q.decode(&view))
                    .and_then(|values| values.first().copied());
                drop(view);
                mapped.unmap();
                ms
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
                        decode_ms: decode_started.elapsed().as_secs_f64() * 1e3,
                        all_ms,
                        basis_ms: all_ms,
                        dispatches: 1,
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
