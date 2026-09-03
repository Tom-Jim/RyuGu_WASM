//! Browser-facing simulation backend.
//!
//! The web UI owns controls, text, SVG, and dialogs. This module only consumes
//! typed requests and advances Bevy resources used by physics and GPU work.

use crate::cpu::frequency_domain::{AggregatedGravitySource, EQ184_QUADRATURE_COUNT};
use crate::gpu::werner::{WernerAcceleration, WernerPotential};
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
    let preserve_radial_capture = inversion.ready || !inversion.truth_knots.is_empty();
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
    let queued_inversion = inversion.start_requested;
    inversion.preserve_truth_track = preserve_radial_capture;
    inversion.optimizer = None;
    inversion.ready = false;
    inversion.start_requested = queued_inversion;
    frequency_domain_result.capture_id = None;
    frequency_domain_result.observations.clear();
}

pub fn clear_gpu_histories_on_method_change(
    active: Res<ActiveGravityMethod>,
    mut werner: Option<ResMut<WernerGravityHistory>>,
    mut mmfft: Option<ResMut<MmfftCompressedHistory>>,
    mut fmm: Option<ResMut<FmmGravityHistory>>,
) {
    if !active.is_changed() {
        return;
    }
    // Keep the Radial history: it is the authoritative observation track
    // shared by the three inverse-capable methods. Probe changes clear it in
    // `apply_probe_input_system`, which starts a genuinely new experiment.
    if let Some(value) = werner.as_deref_mut() {
        value.0.clear();
    }
    if let Some(value) = mmfft.as_deref_mut() {
        value.0.clear();
    }
    if let Some(value) = fmm.as_deref_mut() {
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
    let queued_inversion = inversion.start_requested;
    inversion.capture_id = None;
    inversion.capture_source_hash = 0;
    inversion.ready = false;
    inversion.knots.clear();
    inversion.optimizer = None;
    inversion.start_requested = queued_inversion;
    inversion.error = None;
}

pub fn update_gpu_memory_estimate_system(
    aggregated: Option<Res<AggregatedGravitySource>>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    frequency_domain_performance: Res<FrequencyDomainPerformanceMetrics>,
    mmfft: Option<Res<MmfftCompressedSource>>,
    fmm: Option<Res<FmmSource>>,
    mut estimate: ResMut<GpuMemoryEstimate>,
) {
    let mut bytes = [0_u64; 5];
    if let Some(source) = aggregated.as_ref() {
        let count = source.sources.len() as u32;
        bytes[0] = count as u64 * 16 + 32 + 2 * reduction_buffer_bytes(count);
    }
    if let Some(topology) = topology {
        let face_count = (topology.triangles.len() / 3) as u64;
        let edge_count = face_count * 3 / 2;
        let item_count = edge_count.max(face_count) as u32;
        bytes[1] = edge_count * 80 + face_count * 64 + 32 + 2 * reduction_buffer_bytes(item_count);
    }
    if let Some(source) = aggregated.as_ref() {
        let timing = frequency_domain_performance.latest.unwrap_or_default();
        let target_count = u64::from(timing.target_count.max(1));
        let quadrature_count = EQ184_QUADRATURE_COUNT as u64;
        bytes[2] = source.sources.len() as u64 * 16
            + quadrature_count * 16
            + 96 * 256
            + quadrature_count * 16
            + quadrature_count * 32
            + target_count * 16
            + 2 * target_count * 11 * 16;
    }
    if let Some(source) = mmfft {
        bytes[3] = source.bytes.len() as u64 + 64 + 32;
    }
    if let Some(source) = fmm {
        bytes[4] = source.bytes.len() as u64
            + source.particle_bytes.len() as u64
            + 32
            + 2 * reduction_buffer_bytes(source.node_count);
    }
    estimate.bytes = bytes;
}

fn reduction_buffer_bytes(item_count: u32) -> u64 {
    item_count.div_ceil(64) as u64 * 16
}

pub fn performance_comparison_system(
    time: Res<Time>,
    clock: Res<SimulationClock>,
    active_method: Res<ActiveGravityMethod>,
    jacobi: Res<JacobiHistory>,
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
    let request_id = jacobi.last_request_id;
    if jacobi.last_sample_method == Some(*active_method)
        && request_id.is_some()
        && state.jacobi_last_request_ids[phase] != request_id
        && let Some(sample) = jacobi.samples.back()
    {
        let index = active_method.performance_index();
        if let Some(history) = state.jacobi_history.get_mut(index) {
            if history.len() == PERFORMANCE_HISTORY_CAPACITY {
                history.pop_front();
            }
            history.push_back(*sample);
        }
        state.jacobi_last_request_ids[phase] = request_id;
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
        0 => ActiveGravityMethod::RadialAnalytic,
        1 => ActiveGravityMethod::HomogeneousWerner,
        2 => ActiveGravityMethod::FrequencyDomain,
        3 => ActiveGravityMethod::MmfftCompressed,
        _ => ActiveGravityMethod::Fmm,
    }
}
