//! C++ WASM batch adapter using the existing planning result contract.
use crate::interface::components::*;
use bevy::math::DVec3;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::collections::VecDeque;
use std::time::Duration;

const PLANNING_DISPATCH_TIMING_WINDOW: usize = 9;
const PLANNING_DISPATCH_FRAME_MARGIN: Duration = Duration::from_nanos(33_333_334);

#[derive(Default)]
struct MethodDispatchTiming {
    last_dispatch: Option<Instant>,
    request_durations: VecDeque<Duration>,
}

impl MethodDispatchTiming {
    fn measured_interval(&self) -> Duration {
        if self.request_durations.is_empty() {
            return Duration::ZERO;
        }
        let mut samples = self.request_durations.iter().copied().collect::<Vec<_>>();
        samples.sort_unstable();
        samples[samples.len() / 2] + PLANNING_DISPATCH_FRAME_MARGIN
    }

    fn record(&mut self, started: Instant, duration: Duration) {
        self.last_dispatch = Some(started);
        if self.request_durations.len() >= PLANNING_DISPATCH_TIMING_WINDOW {
            self.request_durations.pop_front();
        }
        self.request_durations.push_back(duration);
    }
}

#[derive(Default)]
pub(crate) struct PlanningDispatchPacer {
    fmm: MethodDispatchTiming,
    mmfft: MethodDispatchTiming,
}

impl PlanningDispatchPacer {
    fn timing(&self, method: ActiveGravityMethod) -> Option<&MethodDispatchTiming> {
        match method {
            ActiveGravityMethod::Fmm => Some(&self.fmm),
            ActiveGravityMethod::MmfftCompressed => Some(&self.mmfft),
            _ => None,
        }
    }

    fn timing_mut(&mut self, method: ActiveGravityMethod) -> Option<&mut MethodDispatchTiming> {
        match method {
            ActiveGravityMethod::Fmm => Some(&mut self.fmm),
            ActiveGravityMethod::MmfftCompressed => Some(&mut self.mmfft),
            _ => None,
        }
    }

    fn should_dispatch(&self, method: ActiveGravityMethod, now: Instant) -> bool {
        let Some(timing) = self.timing(method) else {
            return true;
        };
        timing.last_dispatch.is_none_or(|last_dispatch| {
            now.saturating_duration_since(last_dispatch) >= timing.measured_interval()
        })
    }

    fn record(&mut self, method: ActiveGravityMethod, started: Instant, duration: Duration) {
        if let Some(timing) = self.timing_mut(method) {
            timing.record(started, duration);
        }
    }
}

#[derive(Default)]
pub(crate) struct PlanningCache {
    identity: Option<(u64, ActiveGravityMethod)>,
    last_request: u64,
    baseline: Vec<Option<[f32; 4]>>,
}

pub(crate) fn dispatch(
    batch: Res<PlanningCandidateBatch>,
    request: Res<PlanningGpuRequest>,
    mut result: ResMut<PlanningGpuResult>,
    channel: Res<PlanningGpuReadbackChannel>,
    mut cache: Local<PlanningCache>,
    mut pacer: Local<PlanningDispatchPacer>,
) {
    let Some(method) = request.method else { return };
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
        cache.identity = identity;
        cache.last_request = 0;
        cache.baseline = vec![None; batch.states.len()];
    }
    if cache.last_request == request.request_id {
        return;
    }
    let started = Instant::now();
    if !pacer.should_dispatch(method, started) {
        return;
    }
    cache.last_request = request.request_id;
    let evaluated = evaluate(&batch, &request, method, &mut cache);
    pacer.record(method, started, started.elapsed());
    match evaluated {
        Ok(packet) => result.0 = Some(packet),
        Err(message) => {
            if let Ok(mut error) = channel.error.lock() {
                *error = Some((request.request_id, message));
            }
        }
    }
}

#[cfg(test)]
mod planning_dispatch_pacer_tests {
    use super::*;

    #[test]
    fn measured_request_time_adds_two_render_frames_of_headroom() {
        let mut pacer = PlanningDispatchPacer::default();
        let started = Instant::now();
        let request_time = Duration::from_millis(80);

        assert!(pacer.should_dispatch(ActiveGravityMethod::Fmm, started));
        pacer.record(ActiveGravityMethod::Fmm, started, request_time);
        assert!(!pacer.should_dispatch(
            ActiveGravityMethod::Fmm,
            started + request_time + Duration::from_millis(32)
        ));
        assert!(pacer.should_dispatch(
            ActiveGravityMethod::Fmm,
            started + request_time + Duration::from_millis(34)
        ));
    }

    #[test]
    fn non_cpp_planning_methods_are_not_throttled() {
        let pacer = PlanningDispatchPacer::default();
        let now = Instant::now();
        assert!(pacer.should_dispatch(ActiveGravityMethod::FrequencyDomain, now));
        assert!(pacer.should_dispatch(ActiveGravityMethod::RadialAnalytic, now));
        assert!(pacer.should_dispatch(ActiveGravityMethod::HomogeneousWerner, now));
    }
}

fn evaluate(
    batch: &PlanningCandidateBatch,
    request: &PlanningGpuRequest,
    method: ActiveGravityMethod,
    cache: &mut PlanningCache,
) -> Result<PlanningGpuPacket, String> {
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
        let p = state.body_position().as_dvec3();
        targets.push(p);
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            targets.extend([p + axis * h, p - axis * h]);
        }
    }
    let preprocess = started.elapsed().as_secs_f64() * 1e3;
    let key = if method == ActiveGravityMethod::Fmm {
        "fmm"
    } else {
        "fft"
    };
    let values = crate::cpp_backend::evaluate_sources(key, &sources, &targets)?;
    let mut rows = Vec::with_capacity(count * 4);
    for values in values.as_chunks::<7>().0 {
        rows.push(values[0].map(|v| v as f32));
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
        .flat_map(|i| rows[*i as usize * 4..*i as usize * 4 + 4].iter().copied())
        .collect();
    if !rows
        .iter()
        .flatten()
        .chain(metrics.iter().flatten())
        .all(|v| v.is_finite())
    {
        return Err("C++ planning returned non-finite values".into());
    }
    Ok(PlanningGpuPacket {
        request: request.clone(),
        state_indices: indices,
        rows: compact.clone(),
        raw_rows: compact,
        rejected_sample_count: 0,
        candidate_metrics: metrics,
        readback_valid: true,
        timing: PlanningGpuTiming {
            method_preprocess_ms: preprocess,
            command_submission_ms: (started.elapsed().as_secs_f64() * 1e3 - preprocess).max(0.0),
            // C++ wall-clock work must never be presented as GPU timestamps.
            dispatch_count: 1,
            forward_kernel_evaluations: targets.len() as u64,
            ..Default::default()
        },
        backend: if method == ActiveGravityMethod::Fmm {
            PlanningExecutionBackend::CppExafmm
        } else {
            PlanningExecutionBackend::CppFlups
        },
    })
}
