use crate::bevy_app::surface_field::{
    SurfaceFieldComputeState, cancel_surface_field, queue_surface_comparison, queue_surface_field,
};
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

#[derive(SystemParam)]
pub(crate) struct BrowserSurfaceControls<'w> {
    state: ResMut<'w, SurfaceFieldState>,
    compute: ResMut<'w, SurfaceFieldComputeState>,
}

#[derive(SystemParam)]
pub(crate) struct BrowserUiActions<'w> {
    camera: ResMut<'w, CameraMode>,
    normals: ResMut<'w, ShowNormals>,
    section: ResMut<'w, ShowSection>,
    acceleration: ResMut<'w, SimulationAcceleration>,
    active_method: Res<'w, ActiveGravityMethod>,
    performance: ResMut<'w, PerformanceComparisonState>,
    rotation: ResMut<'w, DisplayRotation>,
    planning: ResMut<'w, PlanningComparisonState>,
    inversion: ResMut<'w, TrajectoryInversionState>,
    sensitivity: ResMut<'w, DensitySensitivityCaches>,
    frequency_domain_sensitivity: ResMut<'w, FrequencyDomainSensitivityMatrix>,
    request: ResMut<'w, PlanningGpuRequest>,
    payload: ResMut<'w, PlanningMethodPayload>,
    result: ResMut<'w, PlanningGpuResult>,
    channel: Res<'w, PlanningGpuReadbackChannel>,
    runtime_error: ResMut<'w, GravityRuntimeError>,
    probe: ResMut<'w, ProbeInitialConditions>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn browser_ui_action_system(
    actions: BrowserUiActions,
    mut surface_controls: BrowserSurfaceControls,
) {
    let BrowserUiActions {
        mut camera,
        mut normals,
        mut section,
        mut acceleration,
        active_method,
        mut performance,
        mut rotation,
        mut planning,
        mut inversion,
        mut sensitivity,
        mut frequency_domain_sensitivity,
        mut request,
        mut payload,
        mut result,
        channel,
        mut runtime_error,
        mut probe,
    } = actions;

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
                    Some("frequency_domain") => Some(ActiveGravityMethod::FrequencyDomain),
                    Some("fft") => Some(ActiveGravityMethod::MmfftCompressed),
                    Some("fmm") => Some(ActiveGravityMethod::Fmm),
                    _ => None,
                };
                if let Some(next) = next
                    && next != *active_method
                {
                    performance.pending_method = Some(next);
                    // Selecting a gravity method changes the live trajectory.
                    // Surface products are an explicit, potentially expensive
                    // analysis job and are refreshed only by Calculate field.
                    cancel_surface_field(
                        &mut surface_controls.state,
                        &mut surface_controls.compute,
                        "Method changed; press Calculate field to refresh the surface product.",
                    );
                }
            }
            "surface-field-metric" => {
                if let Some(metric) = value
                    .and_then(Value::as_str)
                    .and_then(surface_metric_from_key)
                {
                    surface_controls.state.metric = metric;
                    // Relative error is a comparison product, not a scalar
                    // field. Selecting it must make the requested product
                    // available instead of leaving the overlay hidden until
                    // the user discovers the separate Compare button.
                    if metric == SurfaceFieldMetric::Error
                        && surface_controls.state.comparison.is_none()
                    {
                        if surface_controls.state.computing {
                            cancel_surface_field(
                                &mut surface_controls.state,
                                &mut surface_controls.compute,
                                "Switching to the relative-error comparison...",
                            );
                        }
                        queue_surface_comparison(
                            &mut surface_controls.state,
                            &mut surface_controls.compute,
                        );
                    } else if metric == SurfaceFieldMetric::Error {
                        if let Some(comparison) = surface_controls.state.comparison.as_ref() {
                            surface_controls.state.status = format!(
                                "Error map ready: {} compared with {}. Positive is overestimation; negative is underestimation.",
                                comparison.comparison.method.as_str(),
                                comparison.baseline.method.as_str()
                            );
                        }
                    } else if !surface_controls.state.computing {
                        // A comparison may have completed while Relative error
                        // was selected.  Switching back to a scalar metric must
                        // replace the comparison-only status text so the panel
                        // describes the product currently being displayed.
                        surface_controls.state.status = surface_controls
                            .state
                            .latest
                            .as_ref()
                            .map(|dataset| {
                                format!(
                                    "{} surface product ready: gravity, gradient, effective slope.",
                                    dataset.method.as_str()
                                )
                            })
                            .unwrap_or_else(|| "Surface field is ready to compute.".into());
                    }
                    surface_controls.state.revision =
                        surface_controls.state.revision.wrapping_add(1);
                }
            }
            "surface-field-select-patch" => {
                let Some(index) = value.and_then(Value::as_u64).map(|value| value as usize) else {
                    continue;
                };
                let sample_count = surface_controls
                    .state
                    .latest
                    .as_ref()
                    .map_or(0, |dataset| dataset.samples.len());
                if index < sample_count {
                    surface_controls.state.selected_patch = Some(index);
                    surface_controls.state.status =
                        format!("Inspecting surface patch {}/{}.", index + 1, sample_count);
                    surface_controls.state.revision =
                        surface_controls.state.revision.wrapping_add(1);
                }
            }
            "surface-field-compute" => {
                let method = performance.pending_method.unwrap_or(*active_method);
                queue_surface_field(
                    &mut surface_controls.state,
                    &mut surface_controls.compute,
                    method,
                );
            }
            "surface-field-baseline" => {
                if let Some(method) = value.and_then(Value::as_str).and_then(method_from_key) {
                    if surface_controls.state.computing {
                        cancel_surface_field(
                            &mut surface_controls.state,
                            &mut surface_controls.compute,
                            "Algorithm pair changed; choose Relative error or Compare again.",
                        );
                    }
                    surface_controls.state.baseline_method = method;
                    surface_controls.state.comparison = None;
                    surface_controls.state.latest = None;
                    surface_controls.state.selected_patch = None;
                    surface_controls.state.status =
                        "Algorithm pair changed; choose Relative error or Compare again.".into();
                    surface_controls.state.revision =
                        surface_controls.state.revision.wrapping_add(1);
                }
            }
            "surface-field-comparison" => {
                if let Some(method) = value.and_then(Value::as_str).and_then(method_from_key) {
                    if surface_controls.state.computing {
                        cancel_surface_field(
                            &mut surface_controls.state,
                            &mut surface_controls.compute,
                            "Algorithm pair changed; choose Relative error or Compare again.",
                        );
                    }
                    surface_controls.state.comparison_method = method;
                    surface_controls.state.comparison = None;
                    surface_controls.state.latest = None;
                    surface_controls.state.selected_patch = None;
                    surface_controls.state.status =
                        "Algorithm pair changed; choose Relative error or Compare again.".into();
                    surface_controls.state.revision =
                        surface_controls.state.revision.wrapping_add(1);
                }
            }
            "surface-field-compare" => {
                queue_surface_comparison(
                    &mut surface_controls.state,
                    &mut surface_controls.compute,
                );
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
                // Quadrature owns the GPU exclusively. First/Stress share the
                // Worker with invert and must keep running when Invert is pressed.
                if planning.workload_profile == PlanningWorkloadProfile::SourceCrossover {
                    cancel_planning(
                        &mut planning,
                        &mut request,
                        &mut payload,
                        &mut result,
                        &channel,
                        "Trajectory inversion selected; quadrature was cancelled.",
                    );
                }
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
                        planning.status =
                            "Metric selected. Press First or Stress to start the corresponding workload.".into();
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
                    planning.workload_selected = true;
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
                        values.len() == 3
                            && values
                                .iter()
                                .all(|value| value.is_finite() && (*value as f32).is_finite())
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
                inversion.reference_cache_capture_id = None;
                inversion.reference_training_observations.clear();
                inversion.reference_training_sensitivities.clear();
                inversion.reference_holdout_observations.clear();
                inversion.reference_holdout_sensitivities.clear();
                *sensitivity = DensitySensitivityCaches::default();
                *frequency_domain_sensitivity = FrequencyDomainSensitivityMatrix::default();
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
                planning.workload_selected = false;
                planning.selected_metric = ComparisonMetric::SpeedupVsGpuFmm;
                planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
                planning.source_curve_active = true;
                planning.source_curve_visible = true;
                planning.source_curve_index = 0;
                planning.source_curve_repeat = 0;
                planning.source_curve_all_parameters =
                    action.get("scope").and_then(Value::as_str) == Some("all");
                planning.source_curve_density_index = if planning.source_curve_all_parameters {
                    0
                } else {
                    action
                        .get("densityModels")
                        .and_then(Value::as_u64)
                        .and_then(|k| {
                            PLANNING_DENSITY_MODEL_COUNTS
                                .iter()
                                .position(|&value| u64::from(value) == k)
                        })
                        .unwrap_or(0)
                };
                planning.source_curve_target_index = if planning.source_curve_all_parameters {
                    0
                } else {
                    action
                        .get("targets")
                        .and_then(Value::as_u64)
                        .and_then(|nt| {
                            PLANNING_TARGET_COUNTS
                                .iter()
                                .position(|&value| u64::from(value) == nt)
                        })
                        .unwrap_or(0)
                };
                let mut order_seed_bytes = [0u8; 8];
                if getrandom::fill(&mut order_seed_bytes).is_err() {
                    planning.source_curve_active = false;
                    planning.source_curve_visible = false;
                    planning.status =
                        "Quadrature stopped: could not seed the randomized method order.".into();
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
                let (_, density_models, samples_per_candidate) = planning.dimensions();
                planning.status = format!(
                    "Quadrature sweep queued: {} sources, {} density models, {} targets, repeat 1/{}; random method order; scope: {}.",
                    PLANNING_SOURCE_COUNTS[0],
                    density_models,
                    samples_per_candidate,
                    PLANNING_SOURCE_REPEATS,
                    if planning.source_curve_all_parameters {
                        "all K x target combinations"
                    } else {
                        "selected K x target combination"
                    },
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
        "{} selected: full Frequency-domain algorithm, packed FFT, and FMM evaluation queued.",
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
