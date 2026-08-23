fn push_performance_sample<T>(history: &mut std::collections::VecDeque<T>, value: T) {
    if history.len() == PERFORMANCE_HISTORY_CAPACITY {
        history.pop_front();
    }
    history.push_back(value);
}

fn update_performance_chart_segments(
    state: &PerformanceComparisonState,
    segments: &mut Query<(&PerformanceChartSegment, &mut Node, &mut UiTransform)>,
) {
    let fps_max = state
        .fps_history
        .iter()
        .flat_map(|series| series.iter().copied())
        .fold(1.0_f32, f32::max);
    let (jacobi_min, jacobi_max) = performance_jacobi_bounds(state).unwrap_or((0.0, 1.0));
    let jacobi_span = (jacobi_max - jacobi_min).max(1.0e-9);
    let jacobi_time_bounds = performance_jacobi_time_bounds(state).unwrap_or((0.0, 1.0));
    let jacobi_time_span = (jacobi_time_bounds.1 - jacobi_time_bounds.0).max(f64::EPSILON);

    for (segment, mut node, mut transform) in segments.iter_mut() {
        if !performance_chart_series_enabled(state, segment) {
            node.display = Display::None;
            continue;
        }
        let (x0, x1, y0, y1) = if segment.jacobi {
            let Some(history) = state.jacobi_history.get(segment.series) else {
                node.display = Display::None;
                continue;
            };
            let (Some(from), Some(to)) =
                (history.get(segment.index), history.get(segment.index + 1))
            else {
                node.display = Display::None;
                continue;
            };
            let Some(a_drift) = performance_jacobi_relative_drift(history, from) else {
                node.display = Display::None;
                continue;
            };
            let Some(b_drift) = performance_jacobi_relative_drift(history, to) else {
                node.display = Display::None;
                continue;
            };
            let a = ((a_drift - jacobi_min) / jacobi_span).clamp(0.0, 1.0) as f32;
            let b = ((b_drift - jacobi_min) / jacobi_span).clamp(0.0, 1.0) as f32;
            let x0 = ((from.simulation_time_seconds - jacobi_time_bounds.0) / jacobi_time_span)
                .clamp(0.0, 1.0) as f32
                * PERFORMANCE_CHART_CONTENT_WIDTH;
            let x1 = ((to.simulation_time_seconds - jacobi_time_bounds.0) / jacobi_time_span)
                .clamp(0.0, 1.0) as f32
                * PERFORMANCE_CHART_CONTENT_WIDTH;
            (x0, x1, (1.0 - a) * 220.0, (1.0 - b) * 220.0)
        } else {
            let Some(history) = state.fps_history.get(segment.series) else {
                node.display = Display::None;
                continue;
            };
            let (Some(from), Some(to)) =
                (history.get(segment.index), history.get(segment.index + 1))
            else {
                node.display = Display::None;
                continue;
            };
            let y0 = (1.0 - *from / fps_max) * 170.0;
            let y1 = (1.0 - *to / fps_max) * 170.0;
            let history_len = history.len();
            let width = if history_len > 1 {
                PERFORMANCE_CHART_CONTENT_WIDTH / (history_len - 1) as f32
            } else {
                1.0
            };
            (
                segment.index as f32 * width,
                (segment.index + 1) as f32 * width,
                y0,
                y1,
            )
        };
        let delta = Vec2::new(x1 - x0, y1 - y0);
        let length = delta.length();
        node.display = Display::Flex;
        node.left = px((x0 + x1) * 0.5 - length * 0.5);
        node.top = px((y0 + y1) * 0.5 - 1.0);
        node.width = px(length.max(0.5));
        transform.rotation = Rot2::radians(delta.y.atan2(delta.x));
    }
}

fn performance_jacobi_bounds(state: &PerformanceComparisonState) -> Option<(f64, f64)> {
    let mut values = state.jacobi_history.iter().flat_map(|series| {
        series
            .iter()
            .filter_map(|sample| performance_jacobi_relative_drift(series, sample))
    });
    let first = values.next()?.abs();
    let maximum_magnitude = values.fold(first, |maximum, value| maximum.max(value.abs()));
    // Jacobi drift is centered on the invariant value zero. A symmetric axis
    // makes methods directly comparable and prevents the chart from implying
    // that a one-sided numerical bias is the new baseline.
    let limit = (maximum_magnitude * 1.08).max(1.0e-6);
    Some((-limit, limit))
}

fn performance_jacobi_relative_drift(
    history: &VecDeque<JacobiSample>,
    sample: &JacobiSample,
) -> Option<f64> {
    let baseline = history.front()?.jacobi_constant;
    let denominator = baseline.abs().max(1.0e-12);
    let drift = (sample.jacobi_constant - baseline) / denominator;
    drift.is_finite().then_some(drift)
}

fn performance_jacobi_time_bounds(state: &PerformanceComparisonState) -> Option<(f64, f64)> {
    let mut times = state
        .jacobi_history
        .iter()
        .flat_map(|series| series.iter().map(|sample| sample.simulation_time_seconds))
        .filter(|time| time.is_finite());
    let first = times.next()?;
    let (minimum, maximum) = times.fold((first, first), |(minimum, maximum), time| {
        (minimum.min(time), maximum.max(time))
    });
    Some((minimum, maximum.max(minimum + f64::EPSILON)))
}

fn format_performance_time(seconds: f64) -> String {
    if seconds.abs() >= 3600.0 {
        format!("{:.2} h", seconds / 3600.0)
    } else {
        format!("{seconds:.0} s")
    }
}

fn format_axis_value(value: f64) -> String {
    if value.abs() >= 1.0e4 || (value != 0.0 && value.abs() < 1.0e-3) {
        format!("{value:.3e}")
    } else {
        format!("{value:.3}")
    }
}

fn clear_performance_method_history(state: &mut PerformanceComparisonState, method: usize) {
    if let Some(history) = state.fps_history.get_mut(method) {
        history.clear();
    }
    if let Some(result) = state.frames_per_second.get_mut(method) {
        *result = 0.0;
    }
    if let Some(request_id) = state.jacobi_last_request_ids.get_mut(method) {
        *request_id = None;
    }
    if let Some(history) = state.jacobi_history.get_mut(method) {
        history.clear();
    }
}

fn performance_chart_series_enabled(
    state: &PerformanceComparisonState,
    segment: &PerformanceChartSegment,
) -> bool {
    let method = if segment.jacobi {
        match segment.series {
            0..=4 => segment.series,
            _ => return false,
        }
    } else {
        segment.series
    };
    state.enabled_methods.get(method).copied().unwrap_or(false)
}

fn method_for_phase(phase: usize) -> ActiveGravityMethod {
    match phase {
        0 => ActiveGravityMethod::RadialAnalytic,
        1 => ActiveGravityMethod::HomogeneousWerner,
        2 => ActiveGravityMethod::CurvedArcEq106,
        3 => ActiveGravityMethod::MmfftCompressed,
        _ => ActiveGravityMethod::Fmm,
    }
}

pub fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    active_method: Res<ActiveGravityMethod>,
    memory: Res<GpuMemoryEstimate>,
    mut query: Query<&mut Text, With<FpsTextMarker>>,
    mut vram_query: Query<&mut Text, (With<VramTextMarker>, Without<FpsTextMarker>)>,
) {
    let fps = crate::browser_frame_rate().unwrap_or_else(|| {
        diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed())
            .unwrap_or(0.0)
    });
    if let Some(mut text) = query.iter_mut().next() {
        *text = Text::new(format!("FPS: {fps:.0}"));
    }
    if let Some(mut text) = vram_query.iter_mut().next() {
        *text = Text::new(format_vram_text(*active_method, *memory));
    }
}

/// Estimates the persistent GPU allocation for each algorithm from the exact
/// buffer sizes used by its render-world pipeline. WebGPU intentionally has no
/// portable API for driver-level VRAM usage, so this is labeled as an estimate.
pub fn update_gpu_memory_estimate_system(
    aggregated: Option<Res<crate::cpu::curved_arc::AggregatedGravitySource>>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    eq106_tensor: Option<Res<Eq106OperatorTensorResource>>,
    eq106_planner: Option<Res<CurvedArcPlannerState>>,
    eq106_performance: Res<Eq106PerformanceMetrics>,
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
    if let (Some(source), Some(tensor)) = (aggregated.as_ref(), eq106_tensor) {
        let order = eq106_planner
            .as_ref()
            .map_or(1, |planner| planner.taylor_order.clamp(1, 4));
        let coefficient_count = u64::from((order + 1) * (order + 2) / 2);
        let timing = eq106_performance.latest.unwrap_or_default();
        let target_count = u64::from(timing.target_count.max(1));
        // Timestamp query pools are disabled for browser/Metal stability.
        let timestamp_bytes = 0_u64;
        bytes[2] = source.sources.len() as u64 * 16
            + source.fourier_modes.len() as u64 * 16
            + 64 * 8
            + tensor.tensor.coefficients.len() as u64 * 4
            + tensor.psi.coefficients.len() as u64 * 4
            + 96
            + coefficient_count * 64 * 16
            + coefficient_count * 129 * 32
            + target_count * 16
            + 2 * target_count * 9 * 16
            + 2 * timestamp_bytes;
    }
    if let Some(source) = mmfft {
        bytes[3] = source.bytes.len() as u64 + 48 + 32;
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

fn format_vram_text(method: ActiveGravityMethod, memory: GpuMemoryEstimate) -> String {
    let labels = ["R", "W", "106", "MM", "FMM"];
    let total = memory.total_bytes().max(1);
    let active_index = method.performance_index();
    let active_bytes = memory.bytes[active_index];
    let active_share = active_bytes as f64 / total as f64 * 100.0;
    let details = labels
        .iter()
        .zip(memory.bytes)
        .map(|(label, bytes)| format!("{label} {}", format_bytes(bytes)))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "Active runtime VRAM: {} {} ({active_share:.1}% of total)\n{}",
        labels[active_index],
        format_bytes(active_bytes),
        details
    )
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    }
}

fn hint_text(
    mode: CameraMode,
    normals: bool,
    section: bool,
    method: ActiveGravityMethod,
) -> String {
    let mode_str = match mode {
        CameraMode::Overview => "Overview",
        CameraMode::FollowCassini => "Follow Cassini",
    };
    let n_str = if normals { "ON" } else { "OFF" };
    let s_str = if section { "ON" } else { "OFF" };
    format!(
        "Press 'S': View | 'F': Normals | 'D': Section | 'G': Method [{}] | Mode: [{}] | N: [{}] | S: [{}]",
        method.as_str(),
        mode_str,
        n_str,
        s_str
    )
}

pub fn normal_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    active_method: ResMut<ActiveGravityMethod>,
    mut show_normals: ResMut<ShowNormals>,
    show_section: Res<ShowSection>,
    mode: Res<CameraMode>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyF) {
        return;
    }
    show_normals.0 = !show_normals.0;
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}

pub fn section_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    active_method: ResMut<ActiveGravityMethod>,
    mut show_section: ResMut<ShowSection>,
    show_normals: Res<ShowNormals>,
    mode: Res<CameraMode>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyD) {
        return;
    }
    show_section.0 = !show_section.0;
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}
pub fn method_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut active_method: ResMut<ActiveGravityMethod>,
    mut performance: ResMut<PerformanceComparisonState>,
    probe_initial: Res<ProbeInitialConditions>,
    mut gravity_blend: ResMut<GravityBlendFactor>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut radial_potential: ResMut<GravityPotential>,
    mut werner_potential: Option<ResMut<WernerPotential>>,
    mut radial_samples: Option<ResMut<RadialGravityHistory>>,
    mut werner_samples: Option<ResMut<WernerGravityHistory>>,
    mut mmfft_samples: Option<ResMut<MmfftCompressedHistory>>,
    mut simulation_clock: ResMut<SimulationClock>,
    mut jacobi_history: ResMut<JacobiHistory>,
    mut curved_arc: ParamSet<(
        ResMut<CurvedArcPlannerState>,
        ResMut<CurvedArcResidualHistory>,
    )>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        With<CassiniMarker>,
    >,
    mut ryugu_query: Query<&mut Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
) {
    let requested_method = performance.pending_method.take().or_else(|| {
        (!performance.active && keyboard.just_pressed(KeyCode::KeyG)).then_some(
            match *active_method {
                ActiveGravityMethod::RadialAnalytic => ActiveGravityMethod::HomogeneousWerner,
                ActiveGravityMethod::HomogeneousWerner => ActiveGravityMethod::CurvedArcEq106,
                ActiveGravityMethod::CurvedArcEq106 => ActiveGravityMethod::MmfftCompressed,
                ActiveGravityMethod::MmfftCompressed => ActiveGravityMethod::Fmm,
                ActiveGravityMethod::Fmm => ActiveGravityMethod::RadialAnalytic,
            },
        )
    });
    let Some(next_method) = requested_method else {
        return;
    };
    *active_method = next_method;
    runtime_error.clear();
    // The newly selected GPU path may not have produced a sample for the reset
    // probe position yet. Physics remains paused until a fresh snapshot lands.
    gravity_blend.0 = 0.0;
    radial_potential.0 = None;
    if let Some(potential) = werner_potential.as_deref_mut() {
        potential.0 = None;
    }
    if let Some(samples) = radial_samples.as_deref_mut() {
        samples.0.clear();
    }
    if let Some(samples) = werner_samples.as_deref_mut() {
        samples.0.clear();
    }
    if let Some(samples) = mmfft_samples.as_deref_mut() {
        samples.0.clear();
    }
    simulation_clock.reset_state();
    // Every algorithm starts from a fresh physical-time origin. In particular,
    // do not let the first benchmark frame copy a same-method sample left over
    // from before the trajectory reset while the new GPU result is in flight.
    jacobi_history.reset();
    curved_arc.p0().reset();
    curved_arc.p1().reset();

    if let Ok((mut c_transform, mut c_velocity, mut c_history)) = cassini_query.single_mut()
        && let Some(mut r_transform) = ryugu_query.iter_mut().next()
    {
        c_transform.translation = probe_initial.position;
        c_velocity.0 = probe_initial.velocity();
        // Reset probe state so the new trajectory starts clean: drop the old
        // history line, undo accumulated spin, and keep Ryugu centered at CoM.
        c_history.0.clear();
        c_history.0.push_back(probe_initial.position);
        r_transform.rotation = Quat::IDENTITY;
        r_transform.translation = Vec3::ZERO;
    }
}

pub fn reset_inversion_on_method_change(
    active_method: Res<ActiveGravityMethod>,
    performance: Res<PerformanceComparisonState>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if !active_method.is_changed() || performance.active {
        return;
    }
    let preserve_truth_track =
        !inversion.truth_knots.is_empty() || !inversion.truth_orbit.is_empty();
    inversion.runtime_epoch = 0;
    inversion.capture_epoch = 0;
    inversion.last_capture_request_id = None;
    inversion.wall_elapsed_seconds = 0.0;
    inversion.raw_samples.clear();
    inversion.knots.clear();
    inversion.capture_id = None;
    inversion.capture_source_hash = 0;
    inversion.certified_sample_streak = 0;
    inversion.certified_segment_id = None;
    inversion.ready = false;
    inversion.knots_edited = false;
    inversion.inverted = false;
    inversion.selected = None;
    inversion.edit_buffer.clear();
    inversion.error = None;
    inversion.optimizer = None;
    inversion.preserve_truth_track = preserve_truth_track;
    if *active_method == ActiveGravityMethod::HomogeneousWerner {
        inversion.knots.clear();
        inversion.capture_id = None;
        inversion.ready = false;
    } else if !inversion.truth_knots.is_empty() {
        inversion.knots = inversion.truth_knots.clone();
        inversion.capture_id = inversion.truth_capture_id;
        inversion.capture_epoch = inversion.truth_capture_epoch;
        inversion.capture_source_hash = inversion.truth_source_hash;
        inversion.ready = true;
    }
    inversion.displayed_density = inversion.results[active_method.performance_index()]
        .clone()
        .or_else(|| inversion.best_results[active_method.performance_index()].clone());
}

pub fn update_hint_on_mode_change(
    active_method: Res<ActiveGravityMethod>,
    mode: Res<CameraMode>,
    show_normals: Res<ShowNormals>,
    show_section: Res<ShowSection>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !active_method.is_changed()
        && !mode.is_changed()
        && !show_normals.is_changed()
        && !show_section.is_changed()
    {
        return;
    }
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}

#[cfg(test)]
#[cfg(test)]
mod performance_chart_tests {
    use super::{
        PerformanceChartSegment, clear_performance_method_history, format_vram_text,
        performance_chart_series_count, performance_chart_series_enabled,
        performance_jacobi_relative_drift, performance_jacobi_time_bounds,
    };
    use crate::interface::components::{
        ActiveGravityMethod, GpuMemoryEstimate, JacobiSample, PerformanceComparisonState,
    };
    use std::collections::VecDeque;

    #[test]
    fn fps_and_jacobi_plot_series_have_matching_history_slots() {
        assert_eq!(performance_chart_series_count(false), 5);
        assert_eq!(performance_chart_series_count(true), 5);
    }

    #[test]
    fn disabling_method_hides_and_clears_all_related_series() {
        let mut state = PerformanceComparisonState::default();
        state.fps_history[1].push_back(60.0);
        state.jacobi_history[1].push_back(JacobiSample {
            simulation_time_seconds: 0.0,
            jacobi_constant: 1.0,
            eq106_diagnostics: None,
        });
        state.frames_per_second[1] = 60.0;
        clear_performance_method_history(&mut state, 1);
        state.enabled_methods[1] = false;

        assert!(state.fps_history[1].is_empty());
        assert!(state.jacobi_history[1].is_empty());
        assert_eq!(state.frames_per_second[1], 0.0);
        assert!(!performance_chart_series_enabled(
            &state,
            &PerformanceChartSegment {
                series: 1,
                index: 0,
                jacobi: false,
            }
        ));
    }

    #[test]
    fn disabling_eq106_clears_only_its_real_jacobi_series() {
        let mut state = PerformanceComparisonState::default();
        let sample = |time, value| JacobiSample {
            simulation_time_seconds: time,
            jacobi_constant: value,
            eq106_diagnostics: None,
        };
        state.jacobi_history[2] = VecDeque::from([sample(0.0, 1.0)]);
        state.jacobi_history[3] = VecDeque::from([sample(0.0, 2.0)]);
        clear_performance_method_history(&mut state, 2);
        state.enabled_methods[2] = false;

        assert!(state.jacobi_history[2].is_empty());
        assert_eq!(
            state.jacobi_history[3].front().unwrap().jacobi_constant,
            2.0
        );
        assert!(!performance_chart_series_enabled(
            &state,
            &PerformanceChartSegment {
                series: 2,
                index: 0,
                jacobi: true,
            }
        ));
    }

    #[test]
    fn jacobi_series_follow_the_five_algorithm_slots() {
        let state = PerformanceComparisonState::default();
        for series in 0..5 {
            assert!(performance_chart_series_enabled(
                &state,
                &PerformanceChartSegment {
                    series,
                    index: 0,
                    jacobi: true,
                }
            ));
        }
    }

    #[test]
    fn jacobi_plot_uses_physical_time_and_per_series_relative_drift() {
        let mut state = PerformanceComparisonState::default();
        state.jacobi_history[2] = VecDeque::from([
            JacobiSample {
                simulation_time_seconds: 5.0,
                jacobi_constant: 4.0,
                eq106_diagnostics: None,
            },
            JacobiSample {
                simulation_time_seconds: 25.0,
                jacobi_constant: 4.04,
                eq106_diagnostics: None,
            },
        ]);
        assert_eq!(performance_jacobi_time_bounds(&state), Some((5.0, 25.0)));
        let drift = performance_jacobi_relative_drift(
            &state.jacobi_history[2],
            state.jacobi_history[2].back().unwrap(),
        )
        .unwrap();
        assert!((drift - 0.01).abs() < 1.0e-12);
    }

    #[test]
    fn vram_label_follows_active_method_and_reports_all_slots() {
        let memory = GpuMemoryEstimate {
            bytes: [1024, 2048, 3 * 1024, 4 * 1024, 5 * 1024],
        };
        let text = format_vram_text(ActiveGravityMethod::Fmm, memory);
        assert!(text.starts_with("Active runtime VRAM: FMM 5.0 KB"));
        assert!(text.contains("R 1.0 KB"));
        assert!(text.contains("106 3.0 KB"));
    }
}
