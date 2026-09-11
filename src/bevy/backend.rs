//! Browser-facing simulation backend.
//!
//! The web UI owns controls, text, SVG, and dialogs. This module only consumes
//! typed requests and advances Bevy resources used by physics and GPU work.

use crate::cpp_backend::{WernerAcceleration, WernerPotential};
use crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT;
use crate::interface::components::*;
use bevy::prelude::*;

include!("planning_batch.rs");
include!("planning_backend.rs");
include!("probe_backend.rs");

pub fn method_selection_system(
    mut active: ResMut<ActiveGravityMethod>,
    mut performance: ResMut<PerformanceComparisonState>,
    probe: Res<ProbeInitialConditions>,
    mut gravity_blend: ResMut<GravityBlendFactor>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut radial_potential: ResMut<GravityPotential>,
    mut werner_potential: Option<ResMut<WernerPotential>>,
    mut clock: ResMut<SimulationClock>,
    mut jacobi: ResMut<JacobiHistory>,
    mut inversion: ResMut<TrajectoryInversionState>,
    mut frequency_domain_result: ResMut<FrequencyDomainTrajectoryBatchResult>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        With<CassiniMarker>,
    >,
    mut ryugu_query: Query<&mut Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
) {
    let Some(next) = performance.pending_method.take() else {
        return;
    };
    if *active == next {
        return;
    }
    // Each inverse method must capture its own wall-clock live arc. Never
    // reuse FMM/FFT/frequency-domain observation knots across method switches.
    *active = next;
    runtime_error.clear();
    gravity_blend.0 = 0.0;
    radial_potential.0 = None;
    if let Some(potential) = werner_potential.as_deref_mut() {
        potential.0 = None;
    }
    clock.reset_state();
    jacobi.reset();
    if let Ok((mut transform, mut velocity, mut history)) = cassini_query.single_mut() {
        transform.translation = probe.position;
        velocity.0 = probe.velocity();
        history.0.clear();
        history.0.push_back(probe.position);
    }
    if let Some(mut transform) = ryugu_query.iter_mut().next() {
        transform.rotation = Quat::IDENTITY;
        transform.translation = Vec3::ZERO;
    }
    inversion.preserve_truth_track = false;
    inversion.optimizer = None;
    inversion.reset_live_capture();
    inversion.truth_knots.clear();
    inversion.truth_capture_id = None;
    inversion.capture_id = None;
    inversion.start_requested = false;
    frequency_domain_result.capture_id = None;
    frequency_domain_result.observations.clear();
}

pub fn clear_gpu_histories_on_method_change(
    active: Res<ActiveGravityMethod>,
    channels: crate::cpp_backend::BackendChannels,
    mut werner: Option<ResMut<WernerGravityHistory>>,
    mut mmfft: Option<ResMut<MmfftCompressedHistory>>,
    mut fmm: Option<ResMut<FmmGravityHistory>>,
    mut equation106: Option<ResMut<crate::gpu::equation106::Equation106History>>,
) {
    if !active.is_changed() {
        return;
    }
    channels.reset_all();
    // Radial history belongs only to its pointwise evaluator. Epoch checks
    // prevent any old sample from participating in a new experiment.
    if let Some(value) = werner.as_deref_mut() {
        value.0.clear();
    }
    if let Some(value) = mmfft.as_deref_mut() {
        value.0.clear();
    }
    if let Some(value) = fmm.as_deref_mut() {
        value.0.clear();
    }
    if let Some(value) = equation106.as_deref_mut() {
        value.0.clear();
    }
}

pub fn reset_inversion_on_method_change(
    active: Res<ActiveGravityMethod>,
    performance: Res<PerformanceComparisonState>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if !active.is_changed() || performance.active {
        return;
    }
    inversion.capture_id = None;
    inversion.capture_source_hash = 0;
    inversion.reset_live_capture();
    inversion.optimizer = None;
    inversion.preserve_best_results_on_next_epoch = true;
    // Recapture on the new method; do not auto-start Invert from a queued click.
    inversion.start_requested = false;
    inversion.error = None;
}

pub fn update_gpu_memory_estimate_system(
    quadrature: Option<Res<DensityQuadratureSource>>,
    frequency_domain_performance: Res<FrequencyDomainPerformanceMetrics>,
    mut estimate: ResMut<GpuMemoryEstimate>,
) {
    let mut bytes = [0_u64; 5];
    if let Some(source) = quadrature.as_ref() {
        let timing = frequency_domain_performance.latest.unwrap_or_default();
        let target_count = u64::from(timing.target_count.max(1));
        let quadrature_count = EQ184_QUADRATURE_COUNT as u64;
        bytes[ActiveGravityMethod::FrequencyDomain.performance_index()] = source.bytes.len()
            as u64
            + quadrature_count * 16
            + 96 * 256
            + quadrature_count * 16
            + quadrature_count * 32
            + target_count * 16
            + 2 * target_count * 11 * 16
            // Independent Eq.106 propagation source, nodes, spectrum and readback.
            + source.bytes.len() as u64
            + quadrature_count * 24
            + 64;
    }
    estimate.bytes = bytes;
}

pub fn performance_comparison_system(
    time: Res<Time>,
    clock: Res<SimulationClock>,
    active_method: Res<ActiveGravityMethod>,
    jacobi: Res<JacobiHistory>,
    frequency_domain: Res<FrequencyDomainTrajectoryBatchResult>,
    mut state: ResMut<PerformanceComparisonState>,
) {
    if !state.active || !state.measuring || clock.elapsed_seconds <= 0.0 {
        return;
    }
    let phase = state.phase;
    if *active_method != method_for_phase(phase) {
        return;
    }
    let fps = (1.0 / time.delta_secs_f64().max(f64::EPSILON)).clamp(0.0, 240.0);
    if let Some(history) = state.fps_history.get_mut(phase) {
        if history.len() == PERFORMANCE_HISTORY_CAPACITY {
            history.pop_front();
        }
        history.push_back(fps as f32);
    }
    let diagnostic = if *active_method == ActiveGravityMethod::FrequencyDomain {
        let revision = frequency_domain.revision;
        if frequency_domain.capture_id.is_some()
            && !frequency_domain.observations.is_empty()
            && state.diagnostic_last_ids[phase] != Some(revision)
        {
            let mean_square = frequency_domain
                .observations
                .iter()
                .map(|sample| f64::from(sample.transformed_field.length_squared()))
                .sum::<f64>()
                / frequency_domain.observations.len() as f64;
            Some((
                revision,
                PerformanceDiagnosticSample {
                    simulation_time_seconds: clock.elapsed_seconds,
                    value: mean_square.sqrt(),
                },
            ))
        } else {
            None
        }
    } else if jacobi.last_sample_method == Some(*active_method) {
        jacobi.last_request_id.and_then(|request_id| {
            (state.diagnostic_last_ids[phase] != Some(request_id)).then(|| {
                jacobi.samples.back().map(|sample| {
                    (
                        request_id,
                        PerformanceDiagnosticSample {
                            simulation_time_seconds: sample.simulation_time_seconds,
                            value: sample.jacobi_constant,
                        },
                    )
                })
            })?
        })
    } else {
        None
    };
    if let Some((identity, sample)) = diagnostic {
        let index = active_method.performance_index();
        if let Some(history) = state.diagnostic_history.get_mut(index) {
            if history.len() == PERFORMANCE_HISTORY_CAPACITY {
                history.pop_front();
            }
            history.push_back(sample);
        }
        state.diagnostic_last_ids[phase] = Some(identity);
    }
    state.phase_frames = state.phase_frames.saturating_add(1);
    state.phase_elapsed_seconds += time.delta_secs_f64();
    if clock.elapsed_seconds < PERFORMANCE_PHASE_SIMULATION_SECONDS {
        return;
    }
    state.frames_per_second[phase] =
        state.phase_frames as f64 / state.phase_elapsed_seconds.max(f64::EPSILON);
    state.completed_methods[phase] = true;
    if let Some((next_phase, next_method)) = state.next_uncompleted_enabled_method(phase) {
        state.phase = next_phase;
        state.phase_frames = 0;
        state.phase_elapsed_seconds = 0.0;
        state.pending_method = Some(next_method);
    } else {
        state.measuring = false;
        state.pending_method = None;
    }
}

fn method_for_phase(phase: usize) -> ActiveGravityMethod {
    match phase {
        0 => ActiveGravityMethod::Fmm,
        1 => ActiveGravityMethod::MmfftCompressed,
        2 => ActiveGravityMethod::HomogeneousWerner,
        3 => ActiveGravityMethod::RadialAnalytic,
        _ => ActiveGravityMethod::FrequencyDomain,
    }
}
