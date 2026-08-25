pub fn performance_comparison_system(
    time: Res<Time>,
    clock: Res<SimulationClock>,
    eq106_performance: Res<Eq106PerformanceMetrics>,
    planning: Res<PlanningComparisonState>,
    mut state: ResMut<PerformanceComparisonState>,
    active_method: Res<ActiveGravityMethod>,
    jacobi: Res<JacobiHistory>,
    mut nodes: ParamSet<(
        Query<&mut Node, With<PerformanceComparisonPanel>>,
        Query<&mut Node, (With<PerformanceViewButton>, Without<ThreeDViewButton>)>,
        Query<&mut Node, (With<ThreeDViewButton>, Without<PerformanceViewButton>)>,
        Query<&mut Node, With<PerformanceRepeatButton>>,
        Query<(&PerformanceChartSegment, &mut Node, &mut UiTransform)>,
        Query<&mut Node, With<TrajectoryInversionButton>>,
    )>,
    mut texts: ParamSet<(
        Query<&mut Text, With<PerformanceComparisonStatus>>,
        Query<(&mut Text, &PerformanceComparisonResult)>,
        Query<(&PerformanceJacobiAxisLabel, &mut Text)>,
        Query<(&PerformanceTimeAxisLabel, &mut Text)>,
    )>,
) {
    // A repeat request and an algorithm transition are intentionally handled
    // by separate ECS systems. Requeue the target if scheduling order or a
    // transient reset consumed the request before the active method changed;
    // otherwise the UI can remain indefinitely at 0 / 120 frames.
    if state.active && state.measuring {
        let target = method_for_phase(state.phase);
        if *active_method != target && state.pending_method.is_none() {
            state.pending_method = Some(target);
        }
    }

    let panel_display = if state.active {
        Display::Flex
    } else {
        Display::None
    };
    if let Some(mut panel) = nodes.p0().iter_mut().next() {
        panel.display = panel_display;
    }
    if let Some(mut button) = nodes.p1().iter_mut().next() {
        button.display = if state.active {
            Display::None
        } else {
            Display::Flex
        };
    }
    if let Some(mut button) = nodes.p2().iter_mut().next() {
        button.display = if state.active {
            Display::Flex
        } else {
            Display::None
        };
    }
    if let Some(mut button) = nodes.p3().iter_mut().next() {
        button.display = if state.active
            && !state.measuring
            && state.enabled_methods.iter().any(|enabled| *enabled)
        {
            Display::Flex
        } else {
            Display::None
        };
    }
    if let Some(mut button) = nodes.p5().iter_mut().next() {
        button.display = if state.active
            || !planning.selected_metric.is_inversion()
            || matches!(
                *active_method,
                ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner
            )
        {
            Display::None
        } else {
            Display::Flex
        };
    }

    if state.active
        && state.measuring
        && state
            .enabled_methods
            .get(state.phase)
            .copied()
            .unwrap_or(false)
        && *active_method == method_for_phase(state.phase)
        && clock.elapsed_seconds > 0.0
    {
        let dt = time.delta_secs_f64().max(f64::EPSILON);
        // Plot the active benchmark's frame interval. The browser RAF value
        // is a shared global average and makes every method look identical.
        let fps = (1.0 / dt).clamp(0.0, 240.0) as f32;
        let phase = state.phase;
        if let Some(history) = state.fps_history.get_mut(phase) {
            // Readback/phase hand-off frames are visible as single-frame FPS
            // spikes even though they are not representative of the method's
            // steady-state throughput. Keep the raw phase average below, but
            // plot a short EMA so the comparison remains readable.
            let plotted_fps = history
                .back()
                .map_or(fps, |previous| *previous * 0.82 + fps * 0.18);
            push_performance_sample(history, plotted_fps);
        }
        // The global Jacobi history may still contain the previous phase's
        // last sample while a new GPU readback is in flight. Do not splice
        // that stale value into the current algorithm's independent series.
        let jacobi_request_id = jacobi.last_request_id;
        if jacobi.last_sample_method == Some(*active_method)
            && jacobi_request_id.is_some()
            && state.jacobi_last_request_ids[phase] != jacobi_request_id
            && let Some(sample) = jacobi.samples.back()
        {
            let series = active_method.performance_index();
            if let Some(history) = state.jacobi_history.get_mut(series) {
                push_performance_sample(history, *sample);
            }
            state.jacobi_last_request_ids[phase] = jacobi_request_id;
        }
        state.phase_frames = state.phase_frames.saturating_add(1);
        state.phase_elapsed_seconds += time.delta_secs_f64();
        if clock.elapsed_seconds >= PERFORMANCE_PHASE_SIMULATION_SECONDS {
            let elapsed = state.phase_elapsed_seconds.max(f64::EPSILON);
            let phase = state.phase;
            let measured_frames = state.phase_frames;
            if let Some(result) = state.frames_per_second.get_mut(phase) {
                *result = measured_frames as f64 / elapsed;
            }
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
    }

    if let Some(mut text) = texts.p0().iter_mut().next() {
        *text = Text::new(if state.active && state.measuring {
            let mut status = format!(
                "Measuring {} ({:.0} / {:.0} simulation seconds)",
                method_for_phase(state.phase).as_str(),
                clock
                    .elapsed_seconds
                    .min(PERFORMANCE_PHASE_SIMULATION_SECONDS),
                PERFORMANCE_PHASE_SIMULATION_SECONDS,
            );
            if method_for_phase(state.phase) == ActiveGravityMethod::CurvedArcEq106
                && let Some(diagnostics) = state.jacobi_history[2]
                    .back()
                    .and_then(|sample| sample.eq106_diagnostics)
            {
                status.push_str(&format!(
                    " | seg {} origin=({:.0},{:.0},{:.0}) h/u/v=({:.1},{:.1},{:.1}) cert=[{:.1e},{:.1e},{:.1e},{:.1e}]",
                    diagnostics.segment_id,
                    diagnostics.line_origin.x,
                    diagnostics.line_origin.y,
                    diagnostics.line_origin.z,
                    diagnostics.h,
                    diagnostics.u,
                    diagnostics.v,
                    diagnostics.certificates[0],
                    diagnostics.certificates[1],
                    diagnostics.certificates[2],
                    diagnostics.certificates[3],
                ));
            }
            if method_for_phase(state.phase) == ActiveGravityMethod::CurvedArcEq106
                && let Some(timing) = eq106_performance.latest
            {
                let milliseconds = |value: Option<f64>| {
                    value.map_or_else(|| "--".to_owned(), |value| format!("{value:.3}"))
                };
                status.push_str(&format!(
                    " | GPU ms build/eval/copy={}/{}/{} map={:.3} targets={} elements={} inverse={}",
                    milliseconds(timing.spectrum_build_ms),
                    milliseconds(timing.target_evaluation_ms),
                    milliseconds(timing.gpu_readback_copy_ms),
                    timing.cpu_readback_wait_ms,
                    timing.target_count,
                    timing.spectral_element_count,
                    milliseconds(eq106_performance.full_inversion_iteration_ms),
                ));
            }
            status
        } else if state.active && state.enabled_methods.iter().any(|enabled| *enabled) {
            "Benchmark complete. Select 3D display to return.".to_owned()
        } else if state.active {
            "Select at least one algorithm to benchmark.".to_owned()
        } else {
            "Select 3D display to return.".to_owned()
        });
    }
    for (mut text, result) in texts.p1().iter_mut() {
        let fps = state
            .frames_per_second
            .get(result.0)
            .copied()
            .unwrap_or(0.0);
        *text = Text::new(format!(
            "{}: {} FPS",
            method_for_phase(result.0).as_str(),
            if fps > 0.0 {
                format!("{fps:.1}")
            } else {
                "--".to_owned()
            }
        ));
    }
    if let Some((minimum, maximum)) = performance_jacobi_bounds(&state) {
        let middle = (minimum + maximum) * 0.5;
        for (label, mut text) in texts.p2().iter_mut() {
            let value = match label.0 {
                0 => maximum,
                1 => middle,
                2 => minimum,
                _ => continue,
            };
            **text = format_axis_value(value);
        }
    } else {
        for (_, mut text) in texts.p2().iter_mut() {
            **text = "--".to_owned();
        }
    }

    for (label, mut text) in texts.p3().iter_mut() {
        **text = if label.jacobi {
            format!("{}%", label.slot as u32 * 25)
        } else {
            format!("{}%", label.slot as u32 * 25)
        };
    }

    update_performance_chart_segments(&state, &mut nodes.p4());
}
