pub fn density_inversion_timing_ui_system(
    inversion: Res<TrajectoryInversionState>,
    active_method: Res<ActiveGravityMethod>,
    eq106_sensitivity: Res<Eq106SensitivityMatrix>,
    eq106_performance: Res<Eq106PerformanceMetrics>,
    mut timing_labels: Query<(&DensityInversionTimingLabel, &mut Text)>,
    mut status_labels: Query<
        (&mut Text, &mut TextColor),
        (
            With<DensityInversionStatusLabel>,
            Without<DensityInversionTimingLabel>,
        ),
    >,
) {
    let names = ["Radial", "Werner", "Eq.106", "MMFFT", "FMM"];
    for (label, mut text) in timing_labels.iter_mut() {
        let marker = if label.0 == active_method.performance_index() {
            "*"
        } else {
            " "
        };
        **text = match inversion.best_results[label.0].as_ref() {
            Some(result) => format!(
                "{}{: <7} density fit {:>7.4}% | holdout {:>7.4}% | time {:>8.2} ms",
                marker,
                names[label.0],
                result.model_fit * 100.0,
                result.holdout_rmse * 100.0,
                result.inversion_time_ms,
            ),
            None => format!("{}{: <7}  --", marker, names[label.0]),
        };
    }
    for (mut text, mut color) in status_labels.iter_mut() {
        update_density_inversion_status(
            &inversion,
            *active_method,
            &eq106_sensitivity,
            &eq106_performance,
            &mut text,
            &mut color,
        );
    }
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
                "Eq.106 batched GPU matrix: {}/{} columns | source {:.2} ms | one dispatch/readback pending",
                sensitivity.columns.len(),
                job.voxels.len(),
                source_ms,
            );
        } else {
            **text = format!("{} Clarabel 56x56 QP", job.method.as_str());
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
                "Eq.106 ms source {:.2} | matrix cache hit: GPU stages skipped | matrix {:.2} | Clarabel {:.2} | verify {:.2} | total {:.2} | dispatch 0",
                timing.source_preparation_ms,
                timing.design_matrix_assembly_ms,
                timing.convex_solve_ms,
                timing.verification_ms,
                timing.total_ms,
            );
        } else {
            **text = format!(
                "Eq.106 ms source {:.2} | CPU encode spectrum {:.2} / evaluate {:.2} | GPU batch+map {:.2} | matrix {:.2} | Clarabel {:.2} | verify {:.2} | total {:.2} | dispatch {} | rebuild {} | cache miss",
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
