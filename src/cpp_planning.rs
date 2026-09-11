//! C++ WASM batch adapter using the existing planning result contract.
use crate::interface::components::*;
use bevy::math::DVec3;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Stop-loss for the C++ FMM/FFT planning slices.
///
/// Those two methods rebuild a target-dependent tree/grid on every request.
/// The outstanding Worker request is the rate limit; the configurable floor
/// (currently zero) only remains so a non-zero pacing policy can be restored
/// without changing the dispatch shape. Adaptive tile size
/// (`adapt_candidate_tile`) and the numerical contract are untouched.
///
/// `None` means no slice has been dispatched yet. The web `Instant` is
/// `performance.now()` and starts near zero, so "one interval ago" cannot be
/// expressed by subtracting from `Instant::now()` without risking an overflow
/// panic at startup.
#[derive(Default)]
pub(crate) struct PlanningDispatchPacer {
    last_dispatch: Option<Instant>,
}

fn planning_dispatch_interval(method: ActiveGravityMethod) -> Duration {
    match method {
        ActiveGravityMethod::Fmm => Duration::from_millis(PLANNING_FMM_DISPATCH_MIN_INTERVAL_MS),
        ActiveGravityMethod::MmfftCompressed => {
            Duration::from_millis(PLANNING_FFT_DISPATCH_MIN_INTERVAL_MS)
        }
        _ => Duration::ZERO,
    }
}

fn should_dispatch_planning(
    pacer: &PlanningDispatchPacer,
    method: ActiveGravityMethod,
    now: Instant,
) -> bool {
    pacer.last_dispatch.is_none_or(|last_dispatch| {
        now.saturating_duration_since(last_dispatch) >= planning_dispatch_interval(method)
    })
}

struct PendingPlanningEvaluation {
    snapshot: crate::cpp_backend::BackendEvaluateSourcesSnapshot,
    request: PlanningGpuRequest,
    method: ActiveGravityMethod,
    started: Instant,
    preprocess_ms: f64,
    target_count: usize,
}

#[derive(Default)]
pub(crate) struct PlanningCache {
    identity: Option<(u64, ActiveGravityMethod)>,
    last_request: u64,
    baseline: Vec<Option<[f32; 4]>>,
    pending: Option<PendingPlanningEvaluation>,
}

pub(crate) fn dispatch(
    batch: Res<PlanningCandidateBatch>,
    request: Res<PlanningGpuRequest>,
    mut result: ResMut<PlanningGpuResult>,
    channel: Res<PlanningGpuReadbackChannel>,
    backend_channel: Res<crate::cpp_backend::BackendEvaluateSourcesChannel>,
    mut cache: Local<PlanningCache>,
    mut pacer: Local<PlanningDispatchPacer>,
) {
    let Some(method) = request.method else {
        if backend_channel.in_flight.load(Ordering::Acquire) {
            backend_channel.reset();
        }
        cache.pending = None;
        return;
    };
    if !matches!(
        method,
        ActiveGravityMethod::Fmm | ActiveGravityMethod::MmfftCompressed
    ) || batch.batch_id != request.batch_id
        || request.request_id == 0
    {
        return;
    }
    let identity = Some((batch.batch_id, method));
    if cache.identity != identity {
        backend_channel.reset();
        cache.identity = identity;
        cache.last_request = 0;
        cache.baseline = vec![None; batch.states.len()];
        cache.pending = None;
    }

    let completed = backend_channel.take();
    if let Some(packet) = completed {
        let Some(pending) = cache.pending.take() else {
            cache.last_request = 0;
            return;
        };
        if pending.snapshot != packet.snapshot
            || pending.snapshot.epoch != batch.capture_epoch
            || pending.request.request_id != request.request_id
            || pending.request.batch_id != batch.batch_id
        {
            cache.last_request = 0;
            return;
        }
        let evaluated = packet
            .result
            .and_then(|values| finish_evaluation(&batch, pending, values, &mut cache));
        match evaluated {
            Ok(packet) => result.0 = Some(packet),
            Err(message) => {
                if let Ok(mut error) = channel.error.lock() {
                    *error = Some((request.request_id, message));
                }
            }
        }
        return;
    }
    if cache.pending.is_some() && backend_channel.is_idle() {
        // Cancelling an experiment clears the channel, so a free channel with
        // nothing delivered means this request was discarded. Re-issue it
        // instead of waiting for an answer that will never arrive.
        cache.pending = None;
        cache.last_request = 0;
    }
    if cache.last_request == request.request_id || backend_channel.in_flight.load(Ordering::Acquire)
    {
        return;
    }
    let now = Instant::now();
    if !should_dispatch_planning(&pacer, method, now) {
        return;
    }

    let prepared =
        prepare_evaluation(&batch, &request, method).and_then(|(pending, sources, targets)| {
            let key = if method == ActiveGravityMethod::Fmm {
                "fmm"
            } else {
                "fft"
            };
            crate::cpp_backend::request_evaluate_sources(
                &backend_channel,
                pending.snapshot,
                key,
                &sources,
                &targets,
            )
            .map(|submitted| (submitted, pending))
        });
    match prepared {
        Ok((true, pending)) => {
            pacer.last_dispatch = Some(now);
            cache.last_request = request.request_id;
            cache.pending = Some(pending);
        }
        Ok((false, _)) => {}
        Err(message) => {
            if let Ok(mut error) = channel.error.lock() {
                *error = Some((request.request_id, message));
            }
        }
    }
}

fn prepare_evaluation(
    batch: &PlanningCandidateBatch,
    request: &PlanningGpuRequest,
    method: ActiveGravityMethod,
) -> Result<(PendingPlanningEvaluation, Vec<(DVec3, f64)>, Vec<DVec3>), String> {
    let started = Instant::now();
    let densities = batch
        .density_models
        .get(request.density_model as usize * 56..(request.density_model as usize + 1) * 56)
        .ok_or("Invalid density row")?;
    let mut sources = Vec::with_capacity(batch.basis_records.len());
    for record in batch.basis_records.iter() {
        let density = *densities
            .get(record.voxel_index as usize)
            .ok_or("Invalid density column")?;
        sources.push((
            DVec3::new(
                record.position_volume[0] as f64,
                record.position_volume[1] as f64,
                record.position_volume[2] as f64,
            ),
            record.position_volume[3] as f64 * density as f64,
        ));
    }
    let start = request.candidate_start as usize * batch.samples_per_candidate as usize;
    let count = request.candidate_count as usize * batch.samples_per_candidate as usize;
    let states = batch
        .states
        .get(start..start + count)
        .ok_or("Invalid candidate range")?;
    let h = if method == ActiveGravityMethod::MmfftCompressed {
        64.0
    } else {
        0.5
    };
    let mut targets = Vec::with_capacity(count * 7);
    for state in states {
        let position = state.body_position().as_dvec3();
        targets.push(position);
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            targets.extend([position + axis * h, position - axis * h]);
        }
    }
    let snapshot = crate::cpp_backend::BackendEvaluateSourcesSnapshot {
        request_id: batch.batch_id.rotate_left(29) ^ request.request_id,
        epoch: batch.capture_epoch,
    };
    Ok((
        PendingPlanningEvaluation {
            snapshot,
            request: request.clone(),
            method,
            started,
            preprocess_ms: started.elapsed().as_secs_f64() * 1e3,
            target_count: targets.len(),
        },
        sources,
        targets,
    ))
}

fn finish_evaluation(
    batch: &PlanningCandidateBatch,
    pending: PendingPlanningEvaluation,
    flat_values: Vec<f64>,
    cache: &mut PlanningCache,
) -> Result<PlanningGpuPacket, String> {
    if flat_values.len() != pending.target_count * 4
        || !flat_values.iter().all(|value| value.is_finite())
    {
        return Err("Invalid C++ planning Worker response".into());
    }
    let request = &pending.request;
    let method = pending.method;
    let values = flat_values.as_chunks::<4>().0;
    let start = request.candidate_start as usize * batch.samples_per_candidate as usize;
    let count = request.candidate_count as usize * batch.samples_per_candidate as usize;
    let states = batch
        .states
        .get(start..start + count)
        .ok_or("Invalid candidate range")?;
    let h = if method == ActiveGravityMethod::MmfftCompressed {
        64.0
    } else {
        0.5
    };
    let mut rows = Vec::with_capacity(count * 4);
    for values in values.as_chunks::<7>().0 {
        rows.push(values[0].map(|value| value as f32));
        for axis in 0..3 {
            let mut column = [0.0; 4];
            for component in 0..3 {
                column[component] = ((values[1 + axis * 2][component]
                    - values[2 + axis * 2][component])
                    / (2.0 * h)) as f32;
            }
            rows.push(column);
        }
    }
    let mut metrics = vec![[0.0, 0.0, f32::INFINITY, 0.0]; request.candidate_count as usize];
    for (i, state) in states.iter().enumerate() {
        let field = rows[4 * i];
        if request.density_model == 0 {
            cache.baseline[start + i] = Some(field);
        }
        let baseline =
            cache.baseline[start + i].ok_or("Nominal density baseline has not been evaluated")?;
        let metric = &mut metrics[i / batch.samples_per_candidate as usize];
        for component in 0..3 {
            metric[0] += (field[component] - baseline[component]).powi(2);
            metric[1] += baseline[component].powi(2);
            for axis in 0..3 {
                metric[3] += rows[4 * i + 1 + axis][component].powi(2);
            }
        }
        metric[2] = metric[2].min(state.body_position().length() - batch.body_radius);
    }
    let indices = crate::gpu::planning_reduction::planning_verification_targets(request, batch);
    let compact: Vec<_> = indices
        .iter()
        .flat_map(|index| {
            rows[*index as usize * 4..*index as usize * 4 + 4]
                .iter()
                .copied()
        })
        .collect();
    if !rows
        .iter()
        .flatten()
        .chain(metrics.iter().flatten())
        .all(|value| value.is_finite())
    {
        return Err("C++ planning returned non-finite values".into());
    }
    Ok(PlanningGpuPacket {
        request: pending.request,
        state_indices: indices,
        rows: compact.clone(),
        raw_rows: compact,
        rejected_sample_count: 0,
        candidate_metrics: metrics,
        readback_valid: true,
        timing: PlanningGpuTiming {
            method_preprocess_ms: pending.preprocess_ms,
            command_submission_ms: (pending.started.elapsed().as_secs_f64() * 1e3
                - pending.preprocess_ms)
                .max(0.0),
            // Worker wall time must never be presented as GPU timestamps.
            dispatch_count: 1,
            forward_kernel_evaluations: pending.target_count as u64,
            ..Default::default()
        },
        backend: if method == ActiveGravityMethod::Fmm {
            PlanningExecutionBackend::CppExafmm
        } else {
            PlanningExecutionBackend::CppFlups
        },
    })
}

#[cfg(test)]
mod planning_dispatch_pacer_tests {
    use super::*;

    #[test]
    fn outstanding_request_is_the_only_dispatch_gate() {
        let start = Instant::now();
        let pacer = PlanningDispatchPacer::default();
        // Nothing dispatched yet: the first slice may go out immediately.
        assert!(should_dispatch_planning(
            &pacer,
            ActiveGravityMethod::Fmm,
            start
        ));

        let pacer = PlanningDispatchPacer {
            last_dispatch: Some(start),
        };
        // Zero-floor policy: once the Worker answered, the next slice may go
        // out on the same frame. Radial never waited.
        assert!(should_dispatch_planning(
            &pacer,
            ActiveGravityMethod::Fmm,
            start
        ));
        assert!(should_dispatch_planning(
            &pacer,
            ActiveGravityMethod::MmfftCompressed,
            start
        ));
        assert!(should_dispatch_planning(
            &pacer,
            ActiveGravityMethod::RadialAnalytic,
            start
        ));
        let _ = (
            PLANNING_FMM_DISPATCH_MIN_INTERVAL_MS,
            PLANNING_FFT_DISPATCH_MIN_INTERVAL_MS,
        );
    }
}
