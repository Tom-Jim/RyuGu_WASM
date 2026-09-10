pub(crate) fn surface_field_compute_system(
    cpp: Res<crate::cpp_backend::CppBackendState>,
    density_mode: Res<DensityMode>,
    geometry: Res<SurfaceFieldGeometry>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    aggregated: Option<Res<AggregatedGravitySource>>,
    mut state: ResMut<SurfaceFieldState>,
    mut compute: ResMut<SurfaceFieldComputeState>,
) {
    if geometry.patches.is_empty() || !state.computing {
        return;
    }
    if !cpp.ready {
        state.status = "Waiting for C++ gravity geometry.".into();
        return;
    }
    let Some(topology) = topology else {
        state.status = "Waiting for the Ryugu topology to finish loading.".into();
        return;
    };
    let Some(job) = compute.job.as_mut() else {
        state.computing = false;
        state.status = "Surface calculation state was cleared; press Calculate again.".into();
        return;
    };

    if job.evaluator.is_none() {
        let method = job.methods[job.method_index];
        // The public surface model has one density profile: the default
        // logarithmic radial distribution. Werner is the sole exception and
        // always uses its homogeneous closed-polyhedron density.
        let method_density_mode = if method == ActiveGravityMethod::HomogeneousWerner {
            DensityMode::Constant
        } else {
            *density_mode
        };
        if method_density_mode == DensityMode::Variable && aggregated.is_none() {
            state.status = "Waiting for the variable-density source distribution...".into();
            return;
        }
        let sources = build_surface_sources(
            method_density_mode,
            &topology,
            aggregated.as_deref(),
            geometry.scale,
        );
        if sources.is_empty() {
            state.computing = false;
            state.status = "Could not build a finite surface source distribution.".into();
            compute.job = None;
            return;
        }
        job.evaluator = Some(build_evaluator(
            method,
            sources,
            &topology,
            geometry.scale,
            method_density_mode,
        ));
        job.datasets.push(SurfaceFieldDataset {
            method,
            density_mode: method_density_mode,
            samples: Vec::with_capacity(geometry.patches.len()),
            gravity_range: (f32::INFINITY, f32::NEG_INFINITY),
            effective_gravity_range: (f32::INFINITY, f32::NEG_INFINITY),
            gradient_range: (f32::INFINITY, f32::NEG_INFINITY),
            slope_range: (f32::INFINITY, f32::NEG_INFINITY),
        });
    }

    let start = job.patch_index;
    let method = job.methods[job.method_index];
    let chunk = matches!(
        method,
        ActiveGravityMethod::Fmm | ActiveGravityMethod::HomogeneousWerner
    )
    .then_some(SURFACE_EXPENSIVE_CHUNK)
    .unwrap_or(SURFACE_COMPUTE_CHUNK);
    let end = (start + chunk).min(geometry.patches.len());
    let evaluator = job
        .evaluator
        .as_ref()
        .expect("surface evaluator initialized");
    let dataset = job
        .datasets
        .last_mut()
        .expect("surface dataset initialized");
    let budget_start = Instant::now();
    let mut processed_end = start;
    for patch in &geometry.patches[start..end] {
        if processed_end > start
            && budget_start.elapsed().as_secs_f32() * 1_000.0 >= SURFACE_COMPUTE_BUDGET_MS
        {
            break;
        }
        let sample = evaluate_patch(evaluator, *patch);
        if !sample.gravity.is_finite()
            || !sample.effective_gravity.is_finite()
            || !sample.gravity_magnitude.is_finite()
            || !sample.effective_gravity_magnitude.is_finite()
            || !sample.gradient_magnitude.is_finite()
            || !sample.slope_degrees.is_finite()
        {
            state.computing = false;
            state.status = "Surface evaluator returned a non-finite field; calculation stopped. Adjust the sampling offset and calculate again.".into();
            compute.job = None;
            return;
        }
        dataset.gravity_range.0 = dataset.gravity_range.0.min(sample.gravity_magnitude);
        dataset.gravity_range.1 = dataset.gravity_range.1.max(sample.gravity_magnitude);
        dataset.effective_gravity_range.0 = dataset
            .effective_gravity_range
            .0
            .min(sample.effective_gravity_magnitude);
        dataset.effective_gravity_range.1 = dataset
            .effective_gravity_range
            .1
            .max(sample.effective_gravity_magnitude);
        dataset.gradient_range.0 = dataset.gradient_range.0.min(sample.gradient_magnitude);
        dataset.gradient_range.1 = dataset.gradient_range.1.max(sample.gradient_magnitude);
        dataset.slope_range.0 = dataset.slope_range.0.min(sample.slope_degrees);
        dataset.slope_range.1 = dataset.slope_range.1.max(sample.slope_degrees);
        dataset.samples.push(sample);
        processed_end += 1;
    }
    job.patch_index = processed_end;
    if processed_end < geometry.patches.len() {
        state.status = format!(
            "{}: {}/{} surface patches evaluated...",
            job.methods[job.method_index].as_str(),
            processed_end,
            geometry.patches.len()
        );
        return;
    }

    job.evaluator = None;
    job.patch_index = 0;
    job.method_index += 1;
    if job.method_index < job.methods.len() {
        state.status = format!(
            "{} complete; evaluating {}...",
            job.datasets
                .last()
                .expect("completed dataset")
                .method
                .as_str(),
            job.methods[job.method_index].as_str()
        );
        return;
    }

    let finished = compute.job.take().expect("surface job exists");
    state.computing = false;
    state.revision = state.revision.wrapping_add(1);
    if finished.datasets.len() == 1 {
        let dataset = finished.datasets.into_iter().next().expect("one dataset");
        state.latest = Some(dataset);
        state.comparison = None;
        state.selected_patch = state
            .latest
            .as_ref()
            .and_then(|dataset| (!dataset.samples.is_empty()).then_some(0));
        state.status = format!(
            "{} surface product ready: gravity, gradient, effective slope.",
            state
                .latest
                .as_ref()
                .expect("latest dataset")
                .method
                .as_str()
        );
    } else {
        let mut datasets = finished.datasets.into_iter();
        let baseline = datasets.next().expect("baseline dataset");
        let comparison = datasets.next().expect("comparison dataset");
        let signed_errors = baseline
            .samples
            .iter()
            .zip(&comparison.samples)
            .map(|(base, candidate)| {
                (candidate.effective_gravity_magnitude - base.effective_gravity_magnitude)
                    / base.effective_gravity_magnitude.max(1.0e-12)
            })
            .collect::<Vec<_>>();
        let error_range = signed_errors.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
        state.latest = Some(comparison.clone());
        state.selected_patch = (!comparison.samples.is_empty()).then_some(0);
        state.comparison = Some(SurfaceFieldComparison {
            baseline,
            comparison,
            signed_errors,
            error_range: finite_range(error_range),
        });
        state.status = format!(
            "Error map ready: {} compared with {}. Positive is overestimation; negative is underestimation.",
            state.comparison_method.as_str(),
            state.baseline_method.as_str()
        );
    }
}
