pub fn density_inversion_timing_ui_system(
    inversion: Res<TrajectoryInversionState>,
    active_method: Res<ActiveGravityMethod>,
    eq106_sensitivity: Res<Eq106SensitivityMatrix>,
    eq106_performance: Res<Eq106PerformanceMetrics>,
    planning: Res<PlanningComparisonState>,
    mut timing_labels: Query<(&DensityInversionTimingLabel, &mut Text)>,
    mut status_labels: Query<
        (&mut Text, &mut TextColor),
        (
            With<DensityInversionStatusLabel>,
            Without<DensityInversionTimingLabel>,
        ),
    >,
) {
    let names = ["Radial", "Werner", "Eq.106 GPU", "MMFFT GPU", "FMM GPU"];
    for (label, mut text) in timing_labels.iter_mut() {
        let marker = if label.0 == active_method.performance_index() {
            "*"
        } else {
            " "
        };
        **text = comparison_metric_text(
            marker,
            names[label.0],
            label.0,
            &inversion,
            &planning,
        );
    }
    for (mut text, mut color) in status_labels.iter_mut() {
        if planning.selected_metric.is_inversion() {
            update_density_inversion_status(
                &inversion,
                *active_method,
                &eq106_sensitivity,
                &eq106_performance,
                &mut text,
                &mut color,
            );
        } else {
            update_planning_comparison_status(&planning, &mut text, &mut color);
        }
    }
}

fn comparison_metric_text(
    marker: &str,
    name: &str,
    index: usize,
    inversion: &TrajectoryInversionState,
    planning: &PlanningComparisonState,
) -> String {
    if planning.selected_metric.is_inversion() {
        let Some(result) = inversion.results[index].as_ref() else {
            return format!("{marker}{name:<12} N/A");
        };
        return match planning.selected_metric {
            ComparisonMetric::DensityFit => {
                let best_fit = inversion.best_results[index]
                    .as_ref()
                    .map_or(result.model_fit, |best| best.model_fit);
                format!(
                    "{marker}{name:<12} current {:>7.4}% | best {:>7.4}%",
                    result.model_fit * 100.0,
                    best_fit * 100.0,
                )
            }
            ComparisonMetric::InversionTime => format!(
                "{marker}{name:<12} truth {:>7.2} ms | matrix {:>7.2} ms {} | QP {:>7.2} ms | verify {:>6.2} ms | total {:>8.2} ms",
                result.timing.truth_prepare_ms,
                result.timing.matrix_build_ms,
                if result.timing.matrix_cache_hit { "warm" } else { "cold" },
                result.timing.convex_solve_ms,
                result.timing.verification_ms,
                result.inversion_time_ms,
            ),
            _ => unreachable!(),
        };
    }

    let Some(result) = planning.results[index] else {
        if let Some(job) = planning.batch_job.as_ref()
            && job.method.performance_index() == index
        {
            let completed = (u64::from(job.density_model) * u64::from(job.candidate_count)
                + u64::from(job.candidate_start))
                * u64::from(job.samples_per_candidate);
            let percent = 100.0 * completed as f64 / job.total_evaluations.max(1) as f64;
            return format!(
                "{marker}{name:<12} running {:>6.2}% | {completed}/{} BxKxH evaluations",
                percent, job.total_evaluations,
            );
        }
        if planning.run_requested {
            return format!("{marker}{name:<12} pending shared workload");
        }
        return format!("{marker}{name:<12} N/A");
    };
    let (candidate_count, density_count, sample_count) = planning.workload_profile.dimensions();
    if (result.workload.candidate_count, result.workload.density_model_count, result.workload.samples_per_candidate)
        != (candidate_count, density_count, sample_count)
    {
        return format!("{marker}{name:<12} N/A (different workload)");
    }
    if !result.gpu_batch_verified {
        return format!("{marker}{name:<12} N/A (GPU verification failed)");
    }
    let total = result.total_ms;
    let value = match planning.selected_metric {
        ComparisonMetric::GravityRelativeError => {
            format!(
                "gravity {:.3e} | common {:.2} excluded | cold prep {:.2} + encode {:.2} + reduce {:.2} + readback {:.2} = {:.2} ms | verify {:.2} excluded | warm tile {:.2} | BxKxH {} | dispatch {} | kernels {}",
                result.relative_gravity_error,
                result.common_preparation_ms,
                result.preprocessing_ms,
                result.evaluation_ms,
                result.reduction_ms,
                result.readback_ms,
                total,
                result.verification_ms,
                result.warm_evaluation_ms,
                result.density_combinations,
                result.dispatch_count,
                result.forward_kernel_evaluations,
            )
        }
        ComparisonMetric::GradientRelativeError => {
            format!("gradient relative error {:.3e}", result.gradient_relative_error)
        }
        ComparisonMetric::PericenterError => {
            format!("pericenter error {:.4} m", result.pericenter_error_m)
        }
        ComparisonMetric::MinimumAltitude => {
            format!("minimum altitude {:.3} m", result.minimum_altitude_m)
        }
        ComparisonMetric::ModelDiscrimination => {
            format!("normalized model separation {:.3e}", result.model_discrimination)
        }
        ComparisonMetric::PlanningObjective => {
            format!("verified planning objective {:.3e}", result.planning_objective)
        }
        ComparisonMetric::SegmentCount if index == 2 => {
            format!("{} segments", result.segment_count)
        }
        ComparisonMetric::SpeedupVsGpuFmm => {
            if planning.shared_workload().is_none() {
                return format!("{marker}{name:<12} N/A (GPU fairness pending)");
            }
            let Some(fmm) = planning.results[4] else {
                return format!("{marker}{name:<12} N/A");
            };
            format!(
                "{:.3}x vs GPU FMM | {}",
                fmm.total_ms / result.total_ms.max(f64::MIN_POSITIVE),
                result.method.as_str(),
            )
        }
        ComparisonMetric::ColdStartAmortization if index == 2 => {
            if planning.shared_workload().is_none() || result.warm_evaluation_ms <= 0.0 {
                return format!("{marker}{name:<12} N/A (cold/warm GPU timing pending)");
            }
            format!("{} candidates", result.cold_amortization_candidates)
        }
        ComparisonMetric::SegmentCount | ComparisonMetric::ColdStartAmortization => "N/A".into(),
        ComparisonMetric::DensityFit | ComparisonMetric::InversionTime => unreachable!(),
    };
    format!("{marker}{name:<12} {value}")
}

fn update_planning_comparison_status(
    planning: &PlanningComparisonState,
    text: &mut Text,
    color: &mut TextColor,
) {
    let (candidate_count, density_count, sample_count) = planning.workload_profile.dimensions();
    let evaluations = u64::from(candidate_count) * u64::from(density_count) * u64::from(sample_count);
    let prefix = format!(
        "{} | B={} candidates, K={} density models, H={} samples | {} evaluations\nperiod={:.3}h, a={:.1}m, e={:.6}, rp={:.1}m, ra={:.1}m\nfrozen arc {:.1}s, segment <= {:.0}s, order {}, trust {:.0}m, transverse <= {:.0}m, remainder <= {:.1e}\n",
        planning.workload_profile.label(),
        candidate_count,
        density_count,
        sample_count,
        evaluations,
        NEAR_SYNC_ORBIT_PERIOD_SECONDS / 3600.0,
        NEAR_SYNC_SEMIMAJOR_AXIS_METERS,
        NEAR_SYNC_ECCENTRICITY,
        NEAR_SYNC_PERICENTER_RADIUS_METERS,
        NEAR_SYNC_APOCENTER_RADIUS_METERS,
        planning.reference_duration_seconds,
        NEAR_SYNC_SEGMENT_MAX_SECONDS,
        NEAR_SYNC_TAYLOR_ORDER,
        NEAR_SYNC_TRUST_RADIUS_METERS,
        NEAR_SYNC_TRANSVERSE_LIMIT_METERS,
        NEAR_SYNC_RELATIVE_REMAINDER_TARGET,
    );
    if planning.shared_workload().is_none() {
        **text = format!(
            "{prefix}{}\nMetrics are shown from the completed shared BxKxH validation run. Fair verdict withheld until identical method-specific GPU Eq.106, GPU MMFFT and GPU FMM batches are verified; preprocessing must be included.",
            planning.status,
        );
        color.0 = Color::srgb(1.0, 0.78, 0.25);
        return;
    }
    **text = format!(
        "{prefix}gravity <= 1e-3 | gradient <= 1e-2 | pericenter <= 1 m | Eq.106 segments <= 10 (hard 16) | cold amortization <= 256\n{}",
        planning
            .fair_verdict()
            .unwrap_or("Fair verdict withheld: incomplete planning results"),
    );
    color.0 = if planning
        .fair_verdict()
        .is_some_and(|verdict| verdict.contains("advantage"))
    {
        Color::srgb(0.45, 1.0, 0.62)
    } else {
        Color::srgb(1.0, 0.78, 0.25)
    };
}

fn update_density_inversion_status(
    inversion: &TrajectoryInversionState,
    active_method: ActiveGravityMethod,
    sensitivity: &Eq106SensitivityMatrix,
    performance: &Eq106PerformanceMetrics,
    text: &mut Text,
    color: &mut TextColor,
) {
    if let Some(job) = inversion.optimizer.as_ref() {
        if job.method == ActiveGravityMethod::CurvedArcEq106
            && sensitivity.columns.len() < job.voxels.len()
        {
            let source_ms = performance
                .inversion
                .map_or(0.0, |timing| timing.source_preparation_ms);
            **text = format!(
                "Eq.106 batched GPU matrix: {}/{} columns | common truth {:.2} ms | one dispatch/readback pending",
                sensitivity.columns.len(),
                job.voxels.len(),
                source_ms,
            );
        } else {
            let method = if job.method == ActiveGravityMethod::Fmm {
                "CPU quadrupole treecode"
            } else {
                job.method.as_str()
            };
            **text = format!("{} Clarabel 56x56 QP", method);
        }
        color.0 = Color::srgb(1.0, 0.78, 0.25);
        return;
    }
    if active_method == ActiveGravityMethod::RadialAnalytic {
        **text = "Radial forward-only: generating the shared ln-density truth trajectory".into();
        color.0 = Color::srgb(0.55, 0.82, 0.9);
        return;
    }
    if active_method == ActiveGravityMethod::HomogeneousWerner {
        **text = "Werner forward-only: density inversion disabled".into();
        color.0 = Color::srgb(0.55, 0.82, 0.9);
        return;
    }
    if active_method == ActiveGravityMethod::CurvedArcEq106
        && let Some(timing) = performance.inversion
        && timing.total_ms > 0.0
    {
        if timing.matrix_cache_hit {
            **text = format!(
                "Eq.106 ms common truth {:.2} (excluded) | matrix cache hit: GPU stages skipped | matrix {:.2} | Clarabel {:.2} | verify {:.2} | total {:.2} | dispatch 0",
                timing.source_preparation_ms,
                timing.design_matrix_assembly_ms,
                timing.convex_solve_ms,
                timing.verification_ms,
                timing.total_ms,
            );
        } else {
            **text = format!(
                "Eq.106 ms common truth {:.2} (excluded) | CPU encode spectrum {:.2} / evaluate {:.2} | GPU batch+map {:.2} | matrix {:.2} | Clarabel {:.2} | verify {:.2} | total {:.2} | dispatch {} | rebuild {} | cache miss",
                timing.source_preparation_ms,
                timing.spectrum_build_ms.unwrap_or(0.0),
                timing.target_evaluation_ms.unwrap_or(0.0),
                timing.gpu_readback_ms,
                timing.design_matrix_assembly_ms,
                timing.convex_solve_ms,
                timing.verification_ms,
                timing.total_ms,
                timing.dispatch_count,
                timing.spectrum_rebuild_count,
            );
        }
        color.0 = Color::srgb(0.55, 0.82, 0.9);
        return;
    }
    if let Some(result) = inversion.displayed_density.as_ref() {
        write_density_result_status(result, text);
        color.0 = Color::srgb(0.45, 1.0, 0.62);
    } else if let Some(error) = inversion.error.as_deref() {
        **text = error.to_owned();
        color.0 = Color::srgb(1.0, 0.4, 0.35);
    } else if !inversion.ready && inversion.certified_sample_streak > 0 {
        **text = format!(
            "Eq.106 certified warm-up: {}/30 consecutive samples",
            inversion.certified_sample_streak.min(30)
        );
        color.0 = Color::srgb(1.0, 0.78, 0.25);
    } else {
        **text = "Waiting for inversion".into();
        color.0 = Color::srgb(0.72, 0.72, 0.76);
    }
}

fn write_density_result_status(result: &DensityInversionResult, text: &mut Text) {
    let minimum = result
        .voxels
        .iter()
        .map(|voxel| voxel.density)
        .fold(f32::INFINITY, f32::min);
    let maximum = result
        .voxels
        .iter()
        .map(|voxel| voxel.density)
        .fold(f32::NEG_INFINITY, f32::max);
    let deviation = (result
        .voxels
        .iter()
        .map(|voxel| (voxel.density - result.density).powi(2))
        .sum::<f32>()
        / result.voxels.len().max(1) as f32)
        .sqrt();
    let model = if result.method == ActiveGravityMethod::HomogeneousWerner {
        "uniform start vs uniform Werner reference"
    } else {
        "uniform start vs ln density reference"
    };
    let convergence = if result.objective_improvement <= 1.0e-6 {
        "best remained at uniform start"
    } else {
        "Clarabel improved the trajectory objective"
    };
    **text = format!(
        "{}\ndensity fit={:.4}%, density RMSE={:.4}%\ntraining chi RMSE={:.4}, holdout chi RMSE={:.4}\n{}; objective gain={:.4}%\ncapture={:016x}, source={:016x}, epoch={}\nproblem={:016x}, J0={:.3e}, data scale={:.3e}\nnoise={:.2e}, {} seeded realizations\n{} Quintic track samples, {} voxels\nmean rho={:.5e}, range={:.4e}..{:.4e}\nsigma/mean={:.3}, mass scale={:.5}\nobjective={:.3e}, {} convex QP solve",
        model,
        result.model_fit * 100.0,
        result.model_deviation * 100.0,
        result.training_rmse,
        result.holdout_rmse,
        convergence,
        result.objective_improvement * 100.0,
        result.capture_id,
        result.source_hash,
        result.capture_epoch,
        result.problem_id,
        result.initial_objective,
        result.data_error_scale,
        result.observation_noise_fraction,
        result.observation_noise_realizations,
        result.trajectory_samples,
        result.voxels.len(),
        result.density,
        minimum,
        maximum,
        deviation / result.density.max(f32::MIN_POSITIVE),
        result.density_scale,
        result.objective,
        result.iterations,
    );
}
