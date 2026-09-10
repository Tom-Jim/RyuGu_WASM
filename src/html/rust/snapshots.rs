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
    basilisk: Res<'w, BasiliskBridgeState>,
    probe: Res<'w, ProbeInitialConditions>,
    memory: Res<'w, GpuMemoryEstimate>,
    jacobi: Res<'w, JacobiHistory>,
    frequency_domain: Res<'w, FrequencyDomainTrajectoryBatchResult>,
    surface: Res<'w, SurfaceFieldState>,
    density_mode: Res<'w, DensityMode>,
}

pub(crate) fn browser_ui_publish_system(
    state: BrowserUiSnapshot,
    mut publish_state: Local<(
        u8,
        usize,
        u64,
        String,
        Option<bevy::platform::time::Instant>,
    )>,
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
            if publish_state
                .4
                .is_some_and(|last| now.duration_since(last).as_secs_f64() < 1.0)
            {
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
    let frequency_domain = state
        .frequency_domain
        .observations
        .iter()
        .map(|observation| {
            let jacobian_norm = (observation.transformed_jacobian.x_axis.length_squared()
                + observation.transformed_jacobian.y_axis.length_squared()
                + observation.transformed_jacobian.z_axis.length_squared())
            .sqrt();
            [
                observation.laplace_frequency,
                observation.transformed_field.length(),
                jacobian_norm,
                observation.transformed_potential,
            ]
        })
        .collect::<Vec<_>>();
    let performance_fps = state
        .performance
        .fps_history
        .iter()
        .map(|history| history.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let performance_diagnostics = state
        .performance
        .diagnostic_history
        .iter()
        .map(|history| {
            history
                .iter()
                .map(|sample| [sample.simulation_time_seconds, sample.value])
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
                    "pericenterError": (result.method != ActiveGravityMethod::FrequencyDomain)
                        .then_some(result.pericenter_error_m),
                    "minimumAltitude": result.minimum_altitude_m,
                    "separation": result.model_discrimination,
                    "objective": result.planning_objective,
                    "segments": result.segment_count,
                    "coldCandidates": result.cold_amortization_candidates,
                })
            })
        })
        .collect::<Vec<_>>();
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
            "implementation": "Discrete frequency-domain trajectory transform: 64-node finite reciprocal-space point-residue quadrature; GPU FFT (compensated f32), 56 GPU density bases + quintic evaluation; GPU order-2 FMM (P2M/M2M/M2L/P2P and 56-basis density mix). Independent f64 frequency-domain reference uses the same reciprocal-space operator and bounded CPU slices.",
            "timingDefinition": "Raw total = shared CPU preparation + method CPU preparation + GPU preparation/evaluation submission wall times + result processing. Cooperative gaps between submissions are excluded. Checked total = raw total + the additional full checked pass; fixed-target bases charged once, streamed FMM target windows charged whenever rebuilt. Warm calibration and shared f64 references are excluded. GPU views use only pass timestamps; no CPU or readback substitution. All methods share source counts, density rows and trajectory samples; the frequency-domain algorithm reports whole-trajectory Laplace observations while FFT/FMM report pointwise fields, so eligibility is checked against each observable's independent f64 reference rather than pretending the raw outputs are identical.",
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
            "workloadSelected": state.planning.workload_selected,
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
            "diagnosticHistory": performance_diagnostics,
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
        "jacobi": jacobi,
        "frequencyDomain": frequency_domain,
        "basilisk": {
            "protocol": state.basilisk.protocol,
            "version": state.basilisk.version,
            "fixedStepSeconds": state.basilisk.fixed_step_seconds,
            "snapshot": state.basilisk.snapshot.map(|value| json!({
                "algorithm": value.algorithm,
                "sequence": value.sequence,
                "protocolVersion": value.protocol_version,
                "simulationTimeNs": value.simulation_time_ns,
                "simulationTimeSeconds": value.simulation_time_seconds,
                "epoch": value.epoch,
                "positionM": value.position_m,
                "velocityMps": value.velocity_mps,
            })),
            "comparisons": state.basilisk.comparisons.iter().enumerate().map(|(index, value)| json!({
                "algorithm": crate::basilisk::BasiliskAlgorithm::ALL[index].key(),
                "label": crate::basilisk::BasiliskAlgorithm::ALL[index].label(),
                "observable": crate::basilisk::BasiliskAlgorithm::ALL[index].observable(),
                "sampleCount": value.sample_count,
                "relativeAccelerationError": value.relative_acceleration_error,
                "referenceAccelerationMps2": value.reference_acceleration_mps2,
                "measuredAccelerationMps2": value.measured_acceleration_mps2,
            })).collect::<Vec<_>>(),
        },
        "surfaceField": surface_snapshot(&state.surface, *state.density_mode),
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
        ActiveGravityMethod::FrequencyDomain => "frequency_domain",
        ActiveGravityMethod::MmfftCompressed => "fft",
        ActiveGravityMethod::Fmm => "fmm",
    }
}

fn method_from_key(key: &str) -> Option<ActiveGravityMethod> {
    Some(match key {
        "radial" => ActiveGravityMethod::RadialAnalytic,
        "werner" => ActiveGravityMethod::HomogeneousWerner,
        "frequency_domain" => ActiveGravityMethod::FrequencyDomain,
        "fft" => ActiveGravityMethod::MmfftCompressed,
        "fmm" => ActiveGravityMethod::Fmm,
        _ => return None,
    })
}

fn surface_metric_from_key(key: &str) -> Option<SurfaceFieldMetric> {
    Some(match key {
        "gravity" => SurfaceFieldMetric::Gravity,
        "gradient" => SurfaceFieldMetric::Gradient,
        "slope" => SurfaceFieldMetric::Slope,
        "error" => SurfaceFieldMetric::Error,
        _ => return None,
    })
}

fn surface_dataset_snapshot(dataset: &SurfaceFieldDataset) -> Value {
    json!({
        "method": method_key(dataset.method),
        "densityMode": dataset.density_mode.key(),
        "sampleCount": dataset.samples.len(),
        "gravityRange": dataset.gravity_range,
        "effectiveGravityRange": dataset.effective_gravity_range,
        "gradientRange": dataset.gradient_range,
        "slopeRange": dataset.slope_range,
    })
}

fn surface_snapshot(surface: &SurfaceFieldState, density_mode: DensityMode) -> Value {
    let selected = surface.latest.as_ref().and_then(|dataset| {
        surface.selected_patch.and_then(|index| {
            dataset
                .samples
                .get(index)
                .map(|sample| (dataset, index, sample))
        })
    });
    json!({
        "computing": surface.computing,
        "status": surface.status,
        "revision": surface.revision,
        "densityMode": density_mode.key(),
        "metric": surface.metric.key(),
        "metricLabel": surface.metric.label(),
        "baseline": method_key(surface.baseline_method),
        "comparison": method_key(surface.comparison_method),
        "selectedPatch": selected.map(|(dataset, index, sample)| json!({
            "index": index,
            "method": method_key(dataset.method),
            "densityMode": dataset.density_mode.key(),
            "position": sample.position.to_array(),
            "normal": sample.normal.to_array(),
            "gravity": sample.gravity.to_array(),
            "gravityMagnitude": sample.gravity_magnitude,
            "effectiveGravity": sample.effective_gravity.to_array(),
            "effectiveGravityMagnitude": sample.effective_gravity_magnitude,
            "gradientMagnitude": sample.gradient_magnitude,
            "slopeDegrees": sample.slope_degrees,
            "relativeError": surface.comparison.as_ref().and_then(|comparison|
                comparison.signed_errors.get(index).copied()),
        })),
        "latest": surface.latest.as_ref().map(surface_dataset_snapshot),
        "comparisonResult": surface.comparison.as_ref().map(|comparison| json!({
            "baseline": surface_dataset_snapshot(&comparison.baseline),
            "comparison": surface_dataset_snapshot(&comparison.comparison),
            "sampleCount": comparison.signed_errors.len(),
            "errorRange": comparison.error_range,
        })),
    })
}
