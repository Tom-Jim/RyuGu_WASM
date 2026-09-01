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

export function ryugu_page_visible() {
    return document.visibilityState === "visible";
}
"#)]
extern "C" {
    fn take_ryugu_ui_action() -> String;
    fn update_ryugu_ui(snapshot: &str);
    fn ryugu_page_visible() -> bool;
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
            "planning-accuracy" => {
                if let Some(profile) = match value.and_then(Value::as_str) {
                    Some("strict") => Some(PlanningAccuracyProfile::Strict),
                    Some("screening") => Some(PlanningAccuracyProfile::Screening),
                    _ => None,
                } {
                    // Reporting-only switch; keep measurements and the current
                    // GPU job intact, and reclassify every stored repetition.
                    planning.accuracy_profile = profile;
                }
            }
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
            "quadrature-open" => {
                // Opening the parameter picker does not launch a hidden sweep.
                planning.source_curve_visible = true;
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
                planning.computation_complete = false;
                planning.stopped_operation_work = 0.0;
                planning.workload_profile = PlanningWorkloadProfile::SourceCrossover;
                planning.selected_metric = ComparisonMetric::SpeedupVsGpuFmm;
                planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
                planning.source_curve_active = true;
                planning.source_curve_visible = true;
                planning.source_curve_index = 0;
                planning.source_curve_repeat = 0;
                planning.source_curve_all_parameters = action.get("scope")
                    .and_then(Value::as_str) == Some("all");
                planning.source_curve_density_index = if planning.source_curve_all_parameters { 0 } else {
                    action.get("densityModels").and_then(Value::as_u64)
                        .and_then(|k| PLANNING_DENSITY_MODEL_COUNTS.iter().position(|&value| u64::from(value) == k))
                        .unwrap_or(0)
                };
                planning.source_curve_target_index = if planning.source_curve_all_parameters { 0 } else {
                    action.get("targets").and_then(Value::as_u64)
                        .and_then(|nt| PLANNING_TARGET_COUNTS.iter().position(|&value| u64::from(value) == nt))
                        .unwrap_or(0)
                };
                let mut order_seed_bytes = [0u8; 8];
                if getrandom::fill(&mut order_seed_bytes).is_err() {
                    planning.source_curve_active = false;
                    planning.source_curve_visible = false;
                    planning.status = "Quadrature stopped: could not seed the randomized method order.".into();
                    continue;
                }
                planning.source_curve_order_seed = u64::from_le_bytes(order_seed_bytes);
                planning.source_curve_samples.clear();
                planning.results = std::array::from_fn(|_| None);
                planning.batch_job = None;
                planning.preparation_progress = 0.0;
                planning.run_requested = true;
                planning.run_id = planning.run_id.wrapping_add(1);
                planning.source_curve_run_id = planning.run_id;
                planning.computation_complete = false;
                planning.stopped_operation_work = 0.0;
                *request = PlanningGpuRequest::default();
                *payload = PlanningMethodPayload::default();
                result.0 = None;
                let (_, density_models, samples_per_candidate) =
                    planning.dimensions();
                planning.status = format!(
                    "Quadrature sweep queued: {} sources, {} density models, {} targets, repeat 1/{}; random method order; scope: {}.",
                    PLANNING_SOURCE_COUNTS[0],
                    density_models,
                    samples_per_candidate,
                    PLANNING_SOURCE_REPEATS,
                    if planning.source_curve_all_parameters { "all K x target combinations" } else { "selected K x target combination" },
                );
            }
            "quadrature-cancel" => {
                if planning.workload_profile != PlanningWorkloadProfile::SourceCrossover {
                    planning.source_curve_visible = false;
                    continue;
                }
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
    planning.computation_complete = false;
    planning.stopped_operation_work = 0.0;
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
    planning.stopped_operation_work = planning.operation_work().0;
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
    mut publish_state: Local<(u8, usize, u64, String, Option<bevy::platform::time::Instant>)>,
) {
    publish_state.0 = publish_state.0.wrapping_add(1);
    let curve_len = state.planning.source_curve_samples.len();
    let curve_changed = curve_len != publish_state.1
        || publish_state.2 != state.planning.source_curve_run_id
        || publish_state.3 != state.planning.accuracy_profile.key();
    let now = bevy::platform::time::Instant::now();
    // Export each new repetition even while hidden. Otherwise the nine
    // screenshot milestones are lost until someone returns to the tab.
    // Progress-only hidden snapshots are limited to once per second.
    if !curve_changed {
        if !ryugu_page_visible() {
            if publish_state.4.is_some_and(|last| now.duration_since(last).as_secs_f64() < 1.0) {
                return;
            }
        } else if !publish_state.0.is_multiple_of(6) {
            return;
        }
    }
    publish_state.4 = Some(now);
    publish_state.1 = curve_len;
    publish_state.2 = state.planning.source_curve_run_id;
    publish_state.3 = state.planning.accuracy_profile.key().into();
    // The browser retains immutable curve rows. Only send them when numerical
    // results or their accuracy profile change, not on every progress tick.
    let curve = curve_changed.then(|| state
        .planning
        .source_curve_samples
        .iter()
        .map(|sample| {
            let failures = match state.planning.accuracy_profile {
                PlanningAccuracyProfile::Strict => sample.strict_failures,
                PlanningAccuracyProfile::Screening => sample.screening_failures,
            };
            json!({
                "sources": sample.source_count,
                "densityModels": sample.density_model_count,
                "targets": sample.target_count,
                "repeat": sample.repeat,
                "orderSeed": sample.order_seed.to_string(),
                "methodOrder": sample.method_order,
                // Archival exports retain both gates and unfiltered values;
                // a later display-profile change cannot rewrite a screenshot.
                "rawTimes": sample.times_ms,
                "rawKernelTimes": sample.kernel_times_ms,
                "rawEvaluationKernelTimes": sample.evaluation_kernel_times_ms,
                "strictFailures": sample.strict_failures,
                "screeningFailures": sample.screening_failures,
                "screeningFailureReasons": sample.screening_failures.map(planning_accuracy_failure_labels),
                // Fail closed even for clients that forget the eligibility gate.
                "times": std::array::from_fn::<_, 6, _>(|index| {
                    (failures[index] == 0 && sample.times_ms[index].is_finite()
                        && sample.times_ms[index] > 0.0).then_some(sample.times_ms[index])
                }),
                "kernelTimes": std::array::from_fn::<_, 6, _>(|index| {
                    sample.kernel_times_ms[index].filter(|time| failures[index] == 0 && time.is_finite() && *time >= 0.0)
                }),
                "evaluationKernelTimes": std::array::from_fn::<_, 6, _>(|index| {
                    sample.evaluation_kernel_times_ms[index].filter(|time| failures[index] == 0 && time.is_finite() && *time >= 0.0)
                }),
                "basisKernelTimes": sample.basis_kernel_times_ms,
                "gravityErrors": sample.gravity_errors,
                "gradientErrors": sample.gradient_errors,
                "geometry": sample.geometry_basis_build_ms,
                "density": sample.density_model_ms,
                "target": sample.target_point_ms,
                "eligible": failures.map(|mask| mask == 0),
                "strictEligible": sample.eligible,
                "failureReasons": failures.map(planning_accuracy_failure_labels),
                "strictFailureReasons": sample.strict_failures.map(planning_accuracy_failure_labels),
                "accuracyProfile": state.planning.accuracy_profile.key(),
            })
        })
        .collect::<Vec<_>>());
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
                let failures = result.accuracy_failure_mask(state.planning.accuracy_profile, false);
                json!({
                    "method": method_key(result.method),
                    "implementation": result.method.planning_label(),
                    "totalMs": (failures == 0).then_some(result.total_ms),
                    "checkedTotalMs": (result.accuracy_failure_mask(state.planning.accuracy_profile, true) == 0)
                        .then_some(result.certified_estimated_total_ms),
                    "kernelMs": result.raw_kernels.all_ms,
                    "checkedKernelMs": result.checked_kernels.all_ms,
                    "evaluationKernelMs": result.raw_kernels.evaluation_ms,
                    "basisKernelMs": result.raw_kernels.basis_ms,
                    "externalValidationMs": result.external_validation_ms,
                    "eligible": failures == 0,
                    "strictEligible": result.accuracy_eligible(),
                    "failureReasons": planning_accuracy_failure_labels(failures),
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
            "runId": if state.planning.workload_profile == PlanningWorkloadProfile::SourceCrossover {
                state.planning.source_curve_run_id
            } else { state.planning.run_id },
            "completed": state.planning.computation_complete,
            "scope": if state.planning.source_curve_all_parameters { "all" } else { "selected" },
            "workCompleted": state.planning.operation_work().0,
            "workTotal": state.planning.operation_work().1,
            "progressUnit": "estimated arithmetic operation units (source/basis/FFT/RHS/target/reference work); not measured FLOPs or an ETA",
            "implementation": "Eq.106 Taylor/Chebyshev; GPU FFT (compensated f32), 56 GPU density bases + quintic evaluation; GPU order-2 FMM (P2M/M2M/M2L/P2P and 56-basis density mix). Independent f64 validation uses bounded CPU slices.",
            "timingDefinition": "Raw total = shared CPU preparation + method CPU preparation + GPU preparation/evaluation submission wall times + result processing. Cooperative gaps between submissions are excluded. Checked total = raw total + the additional full checked pass; fixed-target bases charged once, streamed FMM target windows charged whenever rebuilt. Warm calibration and shared f64 references are excluded. GPU views use only pass timestamps; no CPU or readback substitution.",
            "visible": state.planning.source_curve_visible,
            "status": state.planning.status,
            "sourceCount": state.planning.requested_source_count,
            "repeat": state.planning.source_curve_repeat + 1,
            "requiredRepeats": PLANNING_SOURCE_REPEATS,
            "accuracyProfile": state.planning.accuracy_profile.key(),
            "accuracyLimits": {
                "gravity": state.planning.accuracy_profile.limits().gravity,
                "gradient": state.planning.accuracy_profile.limits().gradient,
                "gravityP99": state.planning.accuracy_profile.limits().gravity_p99,
                "gradientP99": state.planning.accuracy_profile.limits().gradient_p99,
                "gravityMax": state.planning.accuracy_profile.limits().gravity_max,
                "gradientMax": state.planning.accuracy_profile.limits().gradient_max,
                "pericenterM": state.planning.accuracy_profile.limits().pericenter_m,
            },
            "densityModels": state.planning.dimensions().1,
            "targets": state.planning.dimensions().2,
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
