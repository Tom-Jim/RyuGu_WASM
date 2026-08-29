use crate::cpu::curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory};
use crate::cpu::volterra::VolterraPropagationStatus;
use crate::interface::components::*;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
export function take_ryugu_ui_action() {
    return window.ryuguUi?.takeAction?.() ?? "";
}

export function update_ryugu_ui(snapshot) {
    window.ryuguUi?.render?.(JSON.parse(snapshot));
}
"#)]
extern "C" {
    fn take_ryugu_ui_action() -> String;
    fn update_ryugu_ui(snapshot: &str);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn browser_ui_action_system(
    mut camera: ResMut<CameraMode>,
    mut normals: ResMut<ShowNormals>,
    mut section: ResMut<ShowSection>,
    mut acceleration: ResMut<SimulationAcceleration>,
    active_method: Res<ActiveGravityMethod>,
    mut performance: ResMut<PerformanceComparisonState>,
    mut rotation: ResMut<DisplayRotation>,
    mut planning: ResMut<PlanningComparisonState>,
    mut inversion: ResMut<TrajectoryInversionState>,
    mut request: ResMut<PlanningGpuRequest>,
    mut payload: ResMut<PlanningMethodPayload>,
    mut result: ResMut<PlanningGpuResult>,
    channel: Res<PlanningGpuReadbackChannel>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut probe: ResMut<ProbeInitialConditions>,
) {
    for _ in 0..32 {
        let action = take_ryugu_ui_action();
        if action.is_empty() {
            break;
        }
        let Ok(action) = serde_json::from_str::<Value>(&action) else {
            continue;
        };
        let Some(kind) = action.get("type").and_then(Value::as_str) else {
            continue;
        };
        let value = action.get("value");
        match kind {
            "method" => {
                let next = match value.and_then(Value::as_str) {
                    Some("radial") => Some(ActiveGravityMethod::RadialAnalytic),
                    Some("werner") => Some(ActiveGravityMethod::HomogeneousWerner),
                    Some("eq106") => Some(ActiveGravityMethod::CurvedArcEq106),
                    Some("fft") => Some(ActiveGravityMethod::MmfftCompressed),
                    Some("fmm") => Some(ActiveGravityMethod::Fmm),
                    _ => None,
                };
                if next.is_some() && next != Some(*active_method) {
                    performance.pending_method = next;
                }
            }
            "camera" => {
                *camera = if value.and_then(Value::as_str) == Some("follow") {
                    CameraMode::FollowCassini
                } else {
                    CameraMode::Overview
                };
            }
            "normals" => normals.0 = value.and_then(Value::as_bool).unwrap_or(!normals.0),
            "section" => section.0 = value.and_then(Value::as_bool).unwrap_or(!section.0),
            "acceleration" => {
                if let Some(value) = value.and_then(Value::as_u64) {
                    acceleration.0 = (value as u32)
                        .clamp(MIN_SIMULATION_ACCELERATION, MAX_SIMULATION_ACCELERATION);
                }
            }
            "rotate" => crate::set_display_rotation(rotation.advance()),
            "performance-open" => {
                if !performance.active {
                    performance.return_simulation_acceleration = acceleration.0;
                    acceleration.0 = MIN_SIMULATION_ACCELERATION;
                    performance.start(*active_method);
                }
            }
            "performance-close" => {
                if performance.active {
                    performance.stop();
                    acceleration.0 = performance.return_simulation_acceleration;
                }
            }
            "performance-repeat" => {
                if performance.active && !performance.measuring {
                    performance.restart();
                }
            }
            "performance-method" => {
                if !performance.measuring
                    && let Some(index) = value.and_then(Value::as_u64).map(|value| value as usize)
                    && let Some(enabled) = performance.enabled_methods.get_mut(index)
                {
                    *enabled = !*enabled;
                }
            }
            "inversion-start" => {
                if matches!(
                    *active_method,
                    ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner
                ) {
                    continue;
                }
                planning.selected_metric = ComparisonMetric::DensityFit;
                cancel_planning(
                    &mut planning,
                    &mut request,
                    &mut payload,
                    &mut result,
                    &channel,
                    "Trajectory inversion selected; planning work was cancelled.",
                );
                inversion.start_requested = true;
            }
            "planning-metric" => {
                if let Some(metric) = value.and_then(Value::as_str).and_then(metric_from_key) {
                    planning.selected_metric = metric;
                    planning.source_curve_active = false;
                    planning.source_curve_visible = false;
                    if metric.is_inversion() {
                        cancel_planning(
                            &mut planning,
                            &mut request,
                            &mut payload,
                            &mut result,
                            &channel,
                            "Density inversion selected; planning work was cancelled. Use Invert trajectory.",
                        );
                    } else if planning.completed_workload().is_none()
                        && planning.batch_job.is_none()
                    {
                        queue_planning_run(
                            &mut planning,
                            &mut request,
                            &mut payload,
                            &mut result,
                            &channel,
                        );
                    }
                }
            }
            "planning-workload" => {
                let profile = match value.and_then(Value::as_str) {
                    Some("first") => Some(PlanningWorkloadProfile::First),
                    Some("stress") => Some(PlanningWorkloadProfile::InteractiveStress),
                    _ => None,
                };
                if let Some(profile) = profile {
                    planning.workload_profile = profile;
                    if planning.selected_metric.is_inversion() {
                        planning.selected_metric = ComparisonMetric::SpeedupVsGpuFmm;
                    }
                    queue_planning_run(
                        &mut planning,
                        &mut request,
                        &mut payload,
                        &mut result,
                        &channel,
                    );
                }
            }
            "trajectory-knot" => {
                let Some(index) = action.get("index").and_then(Value::as_u64) else {
                    continue;
                };
                let Some(field) = action.get("field").and_then(Value::as_str) else {
                    continue;
                };
                let Some(values) = value.and_then(Value::as_array) else {
                    continue;
                };
                let Some(vector) = values
                    .iter()
                    .map(Value::as_f64)
                    .collect::<Option<Vec<_>>>()
                    .filter(|values| {
                        values.len() == 3 && values.iter().all(|value| value.is_finite())
                    })
                    .map(|values| Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32))
                else {
                    continue;
                };
                let Some(knot) = inversion.knots.get_mut(index as usize) else {
                    continue;
                };
                match field {
                    "position" => knot.position = vector,
                    "velocity" => knot.velocity = vector,
                    _ => continue,
                }
                // The displayed path uses the frozen authoritative knot set.
                // Update it atomically with the editable copy so a visible
                // control always changes the matching visible point and line.
                inversion.truth_knots = inversion.knots.clone();
                inversion.truth_capture_id = Some(
                    crate::bevy_app::render::hash_trajectory_capture(&inversion.truth_knots),
                );
                inversion.capture_id = inversion.truth_capture_id;
                inversion.inverted = false;
                inversion.optimizer = None;
                inversion.batch_capture_id = None;
                inversion.results = std::array::from_fn(|_| None);
                inversion.best_results = std::array::from_fn(|_| None);
                inversion.displayed_density = None;
                inversion.error = None;
                cancel_planning(
                    &mut planning,
                    &mut request,
                    &mut payload,
                    &mut result,
                    &channel,
                    "Trajectory point updated; planning work was cancelled and results invalidated.",
                );
            }
            "probe" => {
                let Some(parameter) = action.get("parameter").and_then(Value::as_str) else {
                    continue;
                };
                let Some(value) = value.and_then(Value::as_f64).map(|value| value as f32) else {
                    continue;
                };
                match parameter {
                    "x" => probe.position.x = quantize_probe_position(value),
                    "y" => probe.position.y = quantize_probe_position(value),
                    "z" => probe.position.z = quantize_probe_position(value),
                    "speed" => probe.speed_factor = quantize_probe_speed(value),
                    _ => continue,
                }
                probe.preset = ProbeOrbitPreset::Custom;
            }
            "quadrature-start" => {
                if channel.in_flight.load(Ordering::Acquire) {
                    // A cancelled map/submit can still be completing on the
                    // render thread. Do not silently discard a new click;
                    // expose the short drain window and let the next click
                    // start from a clean channel.
                    planning.status =
                        "Quadrature is finishing the previous GPU request; click Run again in a moment."
                            .into();
                    continue;
                }
                if let Ok(mut error) = channel.error.try_lock() {
                    error.take();
                }
                if let Ok(mut data) = channel.data.try_lock() {
                    data.take();
                }
                cancel_planning(
                    &mut planning,
                    &mut request,
                    &mut payload,
                    &mut result,
                    &channel,
                    "Starting a fresh quadrature workload.",
                );
                planning.workload_profile = PlanningWorkloadProfile::SourceCrossover;
                planning.selected_metric = ComparisonMetric::SpeedupVsGpuFmm;
                planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
                planning.source_curve_active = true;
                planning.source_curve_visible = true;
                planning.source_curve_index = 0;
                planning.source_curve_repeat = 0;
                planning.source_curve_samples.clear();
                planning.results = std::array::from_fn(|_| None);
                planning.batch_job = None;
                planning.preparation_progress = 0.0;
                planning.run_requested = true;
                planning.run_id = planning.run_id.wrapping_add(1);
                *request = PlanningGpuRequest::default();
                *payload = PlanningMethodPayload::default();
                result.0 = None;
                let (_, density_models, samples_per_candidate) =
                    PlanningWorkloadProfile::SourceCrossover.dimensions();
                planning.status = format!(
                    "Quadrature-source crossover queued: {} distinct positions, {} density models, {} samples/candidate, repeat 1/{}, Eq.106 first.",
                    PLANNING_SOURCE_COUNTS[0],
                    density_models,
                    samples_per_candidate,
                    PLANNING_SOURCE_REPEATS,
                );
            }
            "quadrature-cancel" => {
                cancel_planning(
                    &mut planning,
                    &mut request,
                    &mut payload,
                    &mut result,
                    &channel,
                    "Quadrature benchmark cancelled; no further GPU work will be submitted.",
                );
            }
            "runtime-reset" => {
                runtime_error.clear();
                *probe = ProbeInitialConditions::default();
            }
            _ => {}
        }
    }
}

fn quantize_probe_position(value: f32) -> f32 {
    (-2000.0 + ((value + 2000.0) / 40.0).round() * 40.0).clamp(-2000.0, 2000.0)
}

fn quantize_probe_speed(value: f32) -> f32 {
    ((value / 0.02).round() * 0.02).clamp(0.0, 2.0)
}

fn queue_planning_run(
    planning: &mut PlanningComparisonState,
    request: &mut PlanningGpuRequest,
    payload: &mut PlanningMethodPayload,
    result: &mut PlanningGpuResult,
    channel: &PlanningGpuReadbackChannel,
) {
    cancel_planning(
        planning,
        request,
        payload,
        result,
        channel,
        "Replacing the previous planning workload.",
    );
    planning.source_curve_active = false;
    planning.source_curve_visible = false;
    planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
    planning.results = std::array::from_fn(|_| None);
    planning.batch_job = None;
    planning.preparation_progress = 0.0;
    planning.reference_duration_seconds = 0.0;
    planning.run_requested = true;
    planning.run_id = planning.run_id.wrapping_add(1);
    *request = PlanningGpuRequest::default();
    *payload = PlanningMethodPayload::default();
    result.0 = None;
    planning.status = format!(
        "{} selected: full Eq.106, packed FFT, and FMM evaluation queued.",
        planning.workload_profile.label(),
    );
}

fn cancel_planning(
    planning: &mut PlanningComparisonState,
    request: &mut PlanningGpuRequest,
    payload: &mut PlanningMethodPayload,
    result: &mut PlanningGpuResult,
    channel: &PlanningGpuReadbackChannel,
    status: &str,
) {
    planning.run_requested = false;
    planning.batch_job = None;
    planning.preparation_progress = 0.0;
    planning.source_curve_active = false;
    planning.source_curve_visible = false;
    // Bump the generation so any builder/readback from the old run is stale
    // even if the browser delivers it during the GPU map drain window.
    planning.run_id = planning.run_id.wrapping_add(1);
    *request = PlanningGpuRequest::default();
    *payload = PlanningMethodPayload::default();
    result.0 = None;
    if let Ok(mut data) = channel.data.try_lock() {
        data.take();
    }
    if let Ok(mut error) = channel.error.try_lock() {
        error.take();
    }
    planning.status = status.into();
}

fn metric_from_key(key: &str) -> Option<ComparisonMetric> {
    Some(match key {
        "density" => ComparisonMetric::DensityFit,
        "inversion-time" => ComparisonMetric::InversionTime,
        "gravity-error" => ComparisonMetric::GravityRelativeError,
        "gradient-error" => ComparisonMetric::GradientRelativeError,
        "pericenter" => ComparisonMetric::PericenterError,
        "altitude" => ComparisonMetric::MinimumAltitude,
        "separation" => ComparisonMetric::ModelDiscrimination,
        "objective" => ComparisonMetric::PlanningObjective,
        "segments" => ComparisonMetric::SegmentCount,
        "speedup" => ComparisonMetric::SpeedupVsGpuFmm,
        "cold" => ComparisonMetric::ColdStartAmortization,
        _ => return None,
    })
}

#[derive(SystemParam)]
pub(crate) struct BrowserUiSnapshot<'w> {
    camera: Res<'w, CameraMode>,
    normals: Res<'w, ShowNormals>,
    section: Res<'w, ShowSection>,
    acceleration: Res<'w, SimulationAcceleration>,
    active_method: Res<'w, ActiveGravityMethod>,
    rotation: Res<'w, DisplayRotation>,
    planning: Res<'w, PlanningComparisonState>,
    performance: Res<'w, PerformanceComparisonState>,
    inversion: Res<'w, TrajectoryInversionState>,
    runtime_error: Res<'w, GravityRuntimeError>,
    probe: Res<'w, ProbeInitialConditions>,
    memory: Res<'w, GpuMemoryEstimate>,
    jacobi: Res<'w, JacobiHistory>,
    residual: Res<'w, CurvedArcResidualHistory>,
    planner: Res<'w, CurvedArcPlannerState>,
    propagation: Res<'w, VolterraPropagationStatus>,
}

pub(crate) fn browser_ui_publish_system(
    state: BrowserUiSnapshot,
    mut publish_state: Local<(u8, usize)>,
) {
    publish_state.0 = publish_state.0.wrapping_add(1);
    let curve_len = state.planning.source_curve_samples.len();
    let curve_changed = curve_len != publish_state.1;
    if !curve_changed && !publish_state.0.is_multiple_of(6) {
        return;
    }
    publish_state.1 = curve_len;
    let curve = state
        .planning
        .source_curve_samples
        .iter()
        .map(|sample| {
            json!({
                "sources": sample.source_count,
                "times": sample.times_ms,
                "geometry": sample.geometry_basis_build_ms,
                "density": sample.density_model_ms,
                "target": sample.target_point_ms,
                "eligible": sample.eligible,
            })
        })
        .collect::<Vec<_>>();
    let jacobi = state
        .jacobi
        .samples
        .iter()
        .map(|sample| [sample.simulation_time_seconds, sample.jacobi_constant])
        .collect::<Vec<_>>();
    let residual = state
        .residual
        .samples
        .iter()
        .filter(|sample| {
            sample.simulation_time_seconds.is_finite() && sample.epsilon_max.is_finite()
        })
        .map(|sample| {
            json!({
                "time": sample.simulation_time_seconds,
                "epsilon": sample.epsilon_max.abs(),
                "order": sample.taylor_order,
            })
        })
        .collect::<Vec<_>>();
    let performance_fps = state
        .performance
        .fps_history
        .iter()
        .map(|history| history.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let performance_jacobi = state
        .performance
        .jacobi_history
        .iter()
        .map(|history| {
            history
                .iter()
                .map(|sample| [sample.simulation_time_seconds, sample.jacobi_constant])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let inversion_results = state
        .inversion
        .results
        .iter()
        .map(|result| {
            result.as_ref().map(|result| {
                json!({
                    "method": method_key(result.method),
                    "density": result.density,
                    "densityScale": result.density_scale,
                    "fit": result.model_fit,
                    "timeMs": result.inversion_time_ms,
                    "trainingRmse": result.training_rmse,
                    "holdoutRmse": result.holdout_rmse,
                })
            })
        })
        .collect::<Vec<_>>();
    let displayed_density = state.inversion.displayed_density.as_ref().map(|result| {
        json!({
            "method": method_key(result.method),
            "density": result.density,
            "densityScale": result.density_scale,
            "fit": result.model_fit,
            "modelDeviation": result.model_deviation,
            "trainingRmse": result.training_rmse,
            "holdoutRmse": result.holdout_rmse,
            "timeMs": result.inversion_time_ms,
        })
    });
    let trajectory = state
        .inversion
        .knots
        .iter()
        .map(|knot| {
            json!({
                "position": [knot.position.x, knot.position.y, knot.position.z],
                "velocity": [knot.velocity.x, knot.velocity.y, knot.velocity.z],
            })
        })
        .collect::<Vec<_>>();
    let planning_results = state
        .planning
        .results
        .iter()
        .map(|result| {
            result.map(|result| {
                json!({
                    "method": method_key(result.method),
                    "totalMs": result.total_ms,
                    "geometryMs": result.geometry_basis_build_ms,
                    "densityModelMs": result.density_model_ms,
                    "targetPointMs": result.target_point_ms,
                    "gravityError": result.relative_gravity_error,
                    "gradientError": result.gradient_relative_error,
                    "pericenterError": result.pericenter_error_m,
                    "minimumAltitude": result.minimum_altitude_m,
                    "separation": result.model_discrimination,
                    "objective": result.planning_objective,
                    "segments": result.segment_count,
                    "coldCandidates": result.cold_amortization_candidates,
                })
            })
        })
        .collect::<Vec<_>>();
    let solve = state.propagation.latest;
    let progress = 100.0 * state.planning.progress_fraction();
    let accuracy = state.planning.batch_job.as_ref().map_or(0.0, |job| {
        let checks = [
            job.gravity_samples > 0,
            job.gradient_samples > 0,
            job.certified_gravity_samples > 0,
            job.certified_gradient_samples > 0,
        ];
        100.0 * checks.into_iter().filter(|passed| *passed).count() as f64 / checks.len() as f64
    });
    let snapshot = json!({
        "fps": crate::browser_frame_rate().unwrap_or(0.0),
        "method": method_key(*state.active_method),
        "methodLabel": state.active_method.as_str(),
        "camera": if *state.camera == CameraMode::FollowCassini { "follow" } else { "overview" },
        "normals": state.normals.0,
        "section": state.section.0,
        "acceleration": state.acceleration.0,
        "rotation": state.rotation.0,
        "probe": {
            "x": state.probe.position.x,
            "y": state.probe.position.y,
            "z": state.probe.position.z,
            "speed": state.probe.speed_factor,
        },
        "memoryBytes": state.memory.bytes,
        "activeVramBytes": state.memory.bytes[state.active_method.performance_index()],
        "planning": {
            "running": state.planning.run_requested,
            "runId": state.planning.run_id,
            "visible": state.planning.source_curve_visible,
            "status": state.planning.status,
            "sourceCount": state.planning.requested_source_count,
            "repeat": state.planning.source_curve_repeat + 1,
            "metric": metric_key(state.planning.selected_metric),
            "workload": workload_key(state.planning.workload_profile),
            "results": planning_results,
            "curve": curve,
            "progress": progress,
            "accuracy": accuracy,
        },
        "performance": {
            "active": state.performance.active,
            "measuring": state.performance.measuring,
            "phase": state.performance.phase,
            "enabled": state.performance.enabled_methods,
            "fps": state.performance.frames_per_second,
            "fpsHistory": performance_fps,
            "jacobiHistory": performance_jacobi,
        },
        "inversion": {
            "ready": state.inversion.ready,
            "running": state.inversion.optimizer.is_some(),
            "inverted": state.inversion.inverted,
            "error": state.inversion.error,
            "results": inversion_results,
            "displayed": displayed_density,
            "trajectory": trajectory,
        },
        "eq106Residual": {
            "visible": *state.active_method == ActiveGravityMethod::CurvedArcEq106,
            "mode": state.planner.mode.as_str(),
            "order": state.planner.taylor_order,
            "segments": state.planner.segments.len(),
            "remainder": state.planner.active_segment.as_ref().map(|segment| segment.remainder_bound),
            "accepted": state.propagation.accepted_segments,
            "rejected": state.propagation.rejected_segments,
            "picardIterations": solve.map(|solve| solve.picard_iterations),
            "endpointIterations": solve.map(|solve| solve.endpoint_iterations),
            "relativeResidual": solve.map(|solve| solve.relative_residual),
            "maximumTransverse": solve.map(|solve| solve.maximum_transverse_distance),
            "samples": residual,
        },
        "jacobi": jacobi,
        "runtimeError": state.runtime_error.message,
    });
    if let Ok(snapshot) = serde_json::to_string(&snapshot) {
        update_ryugu_ui(&snapshot);
    }
}

fn metric_key(metric: ComparisonMetric) -> &'static str {
    match metric {
        ComparisonMetric::DensityFit => "density",
        ComparisonMetric::InversionTime => "inversion-time",
        ComparisonMetric::GravityRelativeError => "gravity-error",
        ComparisonMetric::GradientRelativeError => "gradient-error",
        ComparisonMetric::PericenterError => "pericenter",
        ComparisonMetric::MinimumAltitude => "altitude",
        ComparisonMetric::ModelDiscrimination => "separation",
        ComparisonMetric::PlanningObjective => "objective",
        ComparisonMetric::SegmentCount => "segments",
        ComparisonMetric::SpeedupVsGpuFmm => "speedup",
        ComparisonMetric::ColdStartAmortization => "cold",
    }
}

fn workload_key(workload: PlanningWorkloadProfile) -> &'static str {
    match workload {
        PlanningWorkloadProfile::First => "first",
        PlanningWorkloadProfile::InteractiveStress => "stress",
        PlanningWorkloadProfile::SourceCrossover => "quadrature",
    }
}

fn method_key(method: ActiveGravityMethod) -> &'static str {
    match method {
        ActiveGravityMethod::RadialAnalytic => "radial",
        ActiveGravityMethod::HomogeneousWerner => "werner",
        ActiveGravityMethod::CurvedArcEq106 => "eq106",
        ActiveGravityMethod::MmfftCompressed => "fft",
        ActiveGravityMethod::Fmm => "fmm",
    }
}
