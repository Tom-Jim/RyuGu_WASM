#[derive(Component, Clone, Copy)]
pub(crate) struct ProbeOrbitPresetButton(pub ProbeOrbitPreset);

#[derive(Component, Clone, Copy)]
pub(crate) struct ComparisonMetricButton(pub ComparisonMetric);

#[derive(Component, Clone, Copy)]
pub(crate) struct PlanningWorkloadButton(pub PlanningWorkloadProfile);

fn selection_button(label: &str, selected: bool, width: f32) -> impl Bundle {
    (
        Button,
        Node {
            width: px(width),
            height: px(24),
            padding: UiRect::horizontal(px(6)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
            ..default()
        },
        BackgroundColor(if selected {
            Color::srgb(0.08, 0.42, 0.46)
        } else {
            Color::srgb(0.04, 0.14, 0.18)
        }),
        children![(
            Text::new(label),
            TextFont {
                font_size: bevy::text::FontSize::Px(10.0),
                ..default()
            },
            TextColor(Color::srgb(0.84, 0.96, 1.0)),
        )],
    )
}

pub(crate) fn probe_orbit_preset_button(
    preset: ProbeOrbitPreset,
    current: ProbeOrbitPreset,
) -> impl Bundle {
    (
        selection_button(preset.label(), preset == current, 152.0),
        ProbeOrbitPresetButton(preset),
    )
}

pub fn probe_orbit_preset_system(
    mut commands: Commands,
    interactions: Query<
        (&Interaction, &ProbeOrbitPresetButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut probe_initial: ResMut<ProbeInitialConditions>,
    mut sliders: Query<(Entity, &mut ProbeSlider)>,
) {
    let Some(preset) = interactions.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Pressed).then_some(button.0)
    }) else {
        return;
    };
    *probe_initial = preset.conditions();
    for (entity, mut slider) in sliders.iter_mut() {
        slider.1 = false;
        let value = match slider.0 {
            ProbeParameter::X => probe_initial.position.x,
            ProbeParameter::Y => probe_initial.position.y,
            ProbeParameter::Z => probe_initial.position.z,
            ProbeParameter::SpeedFactor => probe_initial.speed_factor,
        };
        commands.entity(entity).insert(SliderValue(value));
    }
}

pub fn probe_orbit_preset_style_system(
    probe_initial: Res<ProbeInitialConditions>,
    mut buttons: Query<(&ProbeOrbitPresetButton, &mut BackgroundColor)>,
) {
    if !probe_initial.is_changed() {
        return;
    }
    for (button, mut color) in buttons.iter_mut() {
        color.0 = if button.0 == probe_initial.preset {
            Color::srgb(0.08, 0.42, 0.46)
        } else {
            Color::srgb(0.04, 0.14, 0.18)
        };
    }
}

pub fn setup_density_inversion_timing_panel(
    mut commands: Commands,
    planning: Res<PlanningComparisonState>,
) {
    let labels = [
        (2, "Eq.106"),
        (3, "FFT grid + GPU"),
        (4, "GPU treecode"),
    ];
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(15),
                top: px(180),
                width: px(650),
                padding: UiRect::all(px(10)),
                flex_direction: FlexDirection::Column,
                row_gap: px(5),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.06, 0.9)),
            BorderColor::all(Color::srgba(0.3, 0.7, 0.75, 0.65)),
            DensityInversionTimingPanel,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Inversion history and near-pericenter planning comparison"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::srgb(0.82, 0.96, 1.0)),
            ));
            panel
                .spawn(Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(4),
                    row_gap: px(4),
                    ..default()
                })
                .with_children(|row| {
                    for metric in ComparisonMetric::ALL {
                        row.spawn((
                            selection_button(
                                metric.label(),
                                metric == planning.selected_metric,
                                150.0,
                            ),
                            ComparisonMetricButton(metric),
                        ));
                    }
                });
            panel
                .spawn(Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    column_gap: px(5),
                    ..default()
                })
                .with_children(|row| {
                    for profile in [
                        PlanningWorkloadProfile::First,
                        PlanningWorkloadProfile::InteractiveStress,
                    ] {
                        row.spawn((
                            selection_button(
                                profile.label(),
                                profile == planning.workload_profile,
                                205.0,
                            ),
                            PlanningWorkloadButton(profile),
                        ));
                    }
                });
            for (method_index, label) in labels {
                panel.spawn((
                    Text::new(format!("{label:<8}  N/A")),
                    TextFont {
                        font_size: bevy::text::FontSize::Px(11.0),
                        ..default()
                    },
                    TextColor(Color::srgb(0.55, 0.82, 0.9)),
                    DensityInversionTimingLabel(method_index),
                ));
            }
            panel.spawn((
                Text::new("Waiting for inversion"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(10.0),
                    ..default()
                },
                TextColor(Color::srgb(0.72, 0.72, 0.76)),
                Node {
                    margin: UiRect::top(px(3)),
                    max_width: px(625),
                    ..default()
                },
                DensityInversionStatusLabel,
            ));
        });
}

pub fn planning_comparison_control_system(
    metric_interactions: Query<
        (&Interaction, &ComparisonMetricButton),
        (Changed<Interaction>, With<Button>),
    >,
    workload_interactions: Query<
        (&Interaction, &PlanningWorkloadButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut planning: ResMut<PlanningComparisonState>,
    mut request: ResMut<PlanningGpuRequest>,
    mut payload: ResMut<PlanningMethodPayload>,
    mut gpu_result: ResMut<PlanningGpuResult>,
    mut button_sets: ParamSet<(
        Query<(&ComparisonMetricButton, &mut BackgroundColor)>,
        Query<(&PlanningWorkloadButton, &mut BackgroundColor)>,
    )>,
) {
    if let Some(metric) = metric_interactions.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Pressed).then_some(button.0)
    }) {
        planning.selected_metric = metric;
        if metric.is_inversion() {
            planning.run_requested = false;
            planning.batch_job = None;
            planning.results = std::array::from_fn(|_| None);
            planning.reference_duration_seconds = 0.0;
            *request = PlanningGpuRequest::default();
            *payload = PlanningMethodPayload::default();
            gpu_result.0 = None;
            planning.status =
                "Density inversion is selected; use Invert trajectory to solve it.".into();
        } else if planning.completed_workload().is_none() && planning.batch_job.is_none() {
            planning.run_requested = true;
            planning.run_id = planning.run_id.wrapping_add(1);
            planning.results = std::array::from_fn(|_| None);
            planning.reference_duration_seconds = 0.0;
            *request = PlanningGpuRequest::default();
            *payload = PlanningMethodPayload::default();
            gpu_result.0 = None;
            planning.status = format!(
                "{} planning workload queued for the frozen capture.",
                planning.workload_profile.label()
            );
        }
    }
    if let Some(profile) = workload_interactions
        .iter()
        .find_map(|(interaction, button)| {
            (*interaction == Interaction::Pressed).then_some(button.0)
        })
    {
        planning.workload_profile = profile;
        planning.results = std::array::from_fn(|_| None);
        planning.batch_job = None;
        planning.reference_duration_seconds = 0.0;
        *request = PlanningGpuRequest::default();
        *payload = PlanningMethodPayload::default();
        gpu_result.0 = None;
        planning.run_requested = !planning.selected_metric.is_inversion();
        if planning.run_requested {
            planning.run_id = planning.run_id.wrapping_add(1);
            planning.status = format!(
                "{} selected: the planning workload is queued for the frozen capture.",
                profile.label()
            );
        } else {
            planning.status = format!(
                "{} selected for planning metrics; choose a non-inversion metric to run it.",
                profile.label()
            );
        }
    }
    for (button, mut color) in button_sets.p0().iter_mut() {
        color.0 = if button.0 == planning.selected_metric {
            Color::srgb(0.08, 0.42, 0.46)
        } else {
            Color::srgb(0.04, 0.14, 0.18)
        };
    }
    for (button, mut color) in button_sets.p1().iter_mut() {
        color.0 = if button.0 == planning.workload_profile {
            Color::srgb(0.08, 0.42, 0.46)
        } else {
            Color::srgb(0.04, 0.14, 0.18)
        };
    }
}

pub fn update_planning_results_from_inversion_system(
    mut commands: Commands,
    inversion: Res<TrajectoryInversionState>,
    radial: Option<Res<RadialGravitySource>>,
    aggregated: Option<Res<crate::cpu::curved_arc::AggregatedGravitySource>>,
    mut planning: ResMut<PlanningComparisonState>,
    mut batch_builder: Local<Option<crate::cpu::planning::PlanningBatchBuilder>>,
) {
    if !planning.run_requested || planning.batch_job.is_some() {
        return;
    }
    let Some(capture_id) = inversion.capture_id else {
        planning.status = format!(
            "{} planning queued: freeze a reference trajectory first.",
            planning.workload_profile.label()
        );
        return;
    };
    let source_hash = inversion.capture_source_hash;
    if source_hash == 0 {
        planning.status = "Planning queued: the frozen capture identity is incomplete.".into();
        return;
    }
    let dimensions = planning.workload_profile.dimensions();
    let builder_matches = batch_builder.as_ref().is_some_and(|builder| {
        builder.matches(
            planning.workload_profile,
            planning.run_id,
            capture_id,
            source_hash,
        )
    });
    if !builder_matches {
        let Some(radial) = radial else {
            planning.status = "Planning queued: the common radial volume source is not ready.".into();
            return;
        };
        let Some(aggregated) = aggregated else {
            planning.status = "Planning queued: the common 1024-source geometry is not ready.".into();
            return;
        };
        let Some((voxels, _)) = crate::cpu::inversion::build_density_voxels(
            &radial,
            ActiveGravityMethod::CurvedArcEq106,
        ) else {
            planning.status =
                "Planning batch could not build the independent 56-region truth geometry.".into();
            return;
        };
        let Some(builder) = crate::cpu::planning::PlanningBatchBuilder::new(
            planning.workload_profile,
            planning.run_id,
            capture_id,
            inversion.capture_epoch,
            source_hash,
            &inversion.knots,
            &voxels,
            &aggregated,
        ) else {
            planning.status =
                "Planning batch initialization failed its equal-mass or source checks.".into();
            planning.run_requested = false;
            return;
        };
        *batch_builder = Some(builder);
        planning.status = format!(
            "{} candidate preparation: 0 / {} trajectories.",
            planning.workload_profile.label(),
            dimensions.0
        );
        return;
    }
    let builder = batch_builder.as_mut().expect("matched planning builder");
    if !planning.workload_profile.is_compute_benchmark()
        && (crate::browser_frame_rate().is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS)
            || crate::browser_recent_frame_ms()
                .is_some_and(|milliseconds| milliseconds > PLANNING_MAX_RECENT_FRAME_MS))
    {
        planning.status = format!(
            "{} candidate preparation yielded to rendering at {:.1} FPS / {:.1} ms recent frame: {} / {} curves.",
            planning.workload_profile.label(),
            crate::browser_frame_rate().unwrap_or(0.0),
            crate::browser_recent_frame_ms().unwrap_or(0.0),
            builder.completed_candidates(),
            dimensions.0
        );
        return;
    }
    if !builder.advance(PLANNING_BUILD_CANDIDATES_PER_FRAME) {
        planning.status = "Planning candidate generation left the certified 15 m tube.".into();
        planning.run_requested = false;
        *batch_builder = None;
        return;
    }
    if !builder.is_complete() {
        planning.status = format!(
            "{} candidate preparation: {} / {} trajectories.",
            planning.workload_profile.label(),
            builder.completed_candidates(),
            dimensions.0
        );
        return;
    }
    let Some((batch, common_preparation_ms)) = batch_builder.take().and_then(|builder| builder.finish())
    else {
        planning.status = "Planning candidate batch could not be finalized.".into();
        planning.run_requested = false;
        return;
    };
    planning.reference_duration_seconds = inversion
        .knots
        .first()
        .zip(inversion.knots.last())
        .map_or(0.0, |(first, last)| {
            (last.simulation_time_seconds - first.simulation_time_seconds) as f32
        });
    let batch_id = batch.batch_id;
    let density_seed = batch.density_seed;
    let maximum_density_mass_relative_error = batch
        .density_model_masses
        .iter()
        .map(|mass| ((mass - batch.target_mass) / batch.target_mass).abs())
        .fold(0.0_f64, f64::max);
    commands.insert_resource(batch);
    commands.insert_resource(PlanningGpuRequest::default());
    commands.insert_resource(PlanningMethodPayload::default());
    let method_order = planning_method_order(planning.run_id);
    planning.batch_job = Some(PlanningBatchJob {
        run_id: planning.run_id,
        profile: planning.workload_profile,
        method: method_order[0],
        method_order,
        method_order_index: 0,
        batch_id,
        candidate_count: dimensions.0,
        density_model_count: dimensions.1,
        samples_per_candidate: dimensions.2,
        density_seed,
        maximum_density_mass_relative_error,
        request_id: planning.run_id.wrapping_shl(24),
        density_model: 0,
        candidate_start: 0,
        candidate_tile_size: PLANNING_GPU_TILE_INITIAL_CANDIDATES,
        minimum_tile_size_used: u32::MAX,
        maximum_tile_size_used: 0,
        gpu_request_count: 0,
        last_request_candidate_count: 0,
        awaiting_gpu: false,
        warm_repetition: false,
        total_evaluations: u64::from(dimensions.0)
            * u64::from(dimensions.1)
            * u64::from(dimensions.2),
        gravity_error_sum: 0.0,
        gravity_reference_sum: 0.0,
        gravity_samples: 0,
        gradient_error_sum: 0.0,
        gradient_reference_sum: 0.0,
        gradient_samples: 0,
        verification_sample_count: 0,
        maximum_gradient_self_fd_relative_error: 0.0,
        pericenter_error_m: 0.0,
        minimum_altitude_m: f32::INFINITY,
        discrimination_sum: 0.0,
        discrimination_reference_sum: 0.0,
        discrimination_samples: 0,
        gradient_information_sum: 0.0,
        candidate_discrimination_sum: vec![0.0; dimensions.0 as usize],
        candidate_reference_sum: vec![0.0; dimensions.0 as usize],
        candidate_gradient_sum: vec![0.0; dimensions.0 as usize],
        candidate_minimum_altitude_m: vec![f32::INFINITY; dimensions.0 as usize],
        candidate_valid: vec![true; dimensions.0 as usize],
        common_preparation_ms,
        preprocessing_ms: 0.0,
        command_submission_ms: 0.0,
        reduction_ms: 0.0,
        verification_ms: 0.0,
        gpu_completion_map_ms: 0.0,
        warm_evaluation_ms: 0.0,
        dispatch_count: 0,
        forward_kernel_evaluations: 0,
        spectral_element_count: 0,
    });
    planning.status = format!(
        "{} batch planning started: 0/{} evaluations, order {} -> {} -> {}.",
        planning.workload_profile.label(),
        planning.batch_job.as_ref().map_or(0, |job| job.total_evaluations),
        method_order[0].planning_label(),
        method_order[1].planning_label(),
        method_order[2].planning_label(),
    );
}

fn planning_method_order(_run_id: u64) -> [ActiveGravityMethod; 3] {
    // A visible benchmark always begins with Eq.106. This keeps First and
    // Interactive Stress predictable and prevents a new run from apparently
    // starting at the third method merely because its run id changed.
    [
        ActiveGravityMethod::CurvedArcEq106,
        ActiveGravityMethod::MmfftCompressed,
        ActiveGravityMethod::Fmm,
    ]
}

#[cfg(test)]
mod planning_method_order_tests {
    use super::planning_method_order;
    use crate::interface::components::ActiveGravityMethod;

    #[test]
    fn every_run_starts_with_eq106_and_uses_the_same_order() {
        let first = planning_method_order(0);
        let second = planning_method_order(1);
        let third = planning_method_order(2);
        assert_eq!(first[0], ActiveGravityMethod::CurvedArcEq106);
        assert_eq!(second, first);
        assert_eq!(third, first);
        for order in [first, second, third] {
            assert!(order.contains(&ActiveGravityMethod::CurvedArcEq106));
            assert!(order.contains(&ActiveGravityMethod::MmfftCompressed));
            assert!(order.contains(&ActiveGravityMethod::Fmm));
        }
    }
}

#[derive(Component)]
pub(crate) struct ProbeCrashOverlay;

#[derive(Component)]
pub(crate) struct ProbeCrashMessage;

pub fn setup_probe_crash_overlay(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                right: px(0),
                top: px(0),
                bottom: px(0),
                display: Display::None,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.08, 0.0, 0.0, 0.78)),
            GlobalZIndex(2_000_000),
            ProbeCrashOverlay,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("PROBE IMPACT\nResetting flight scene..."),
                TextFont {
                    font_size: bevy::text::FontSize::Px(32.0),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.35, 0.3)),
                ProbeCrashMessage,
            ));
        });
}

pub fn probe_collision_system(
    mut crash: ResMut<ProbeCrashState>,
    cassini_query: Query<&Transform, With<CassiniMarker>>,
    ryugu_query: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if crash.active || runtime_error.is_active() {
        return;
    }
    let Some(probe) = cassini_query.iter().next() else {
        return;
    };
    let body_position = ryugu_query
        .iter()
        .next()
        .map_or(Vec3::ZERO, |transform| transform.translation);
    let collision_radius = RYUGU_COLLISION_RADIUS_METERS + PROBE_COLLISION_RADIUS_METERS;
    if probe.translation.distance_squared(body_position) <= collision_radius * collision_radius {
        crash.trigger();
        runtime_error.raise("Probe collision detected; simulation paused for scene reset.");
    }
}

pub fn probe_crash_overlay_system(
    time: Res<Time>,
    mut crash: ResMut<ProbeCrashState>,
    mut overlays: Query<&mut Node, With<ProbeCrashOverlay>>,
    mut messages: Query<&mut Text, With<ProbeCrashMessage>>,
) {
    for mut node in overlays.iter_mut() {
        node.display = if crash.active {
            Display::Flex
        } else {
            Display::None
        };
    }
    if crash.active {
        let remaining = (ProbeCrashState::DISPLAY_SECONDS - crash.elapsed_seconds).max(0.0);
        for mut text in messages.iter_mut() {
            **text = format!("PROBE IMPACT\nResetting flight scene in {remaining:.1} s");
        }
        crash.elapsed_seconds += time.delta_secs();
    }
}

pub fn reset_after_probe_crash_scene_system(
    mut crash: ResMut<ProbeCrashState>,
    mut reset_request: ResMut<ProbeCrashResetRequest>,
    mut sliders: Query<(Entity, &ProbeSlider)>,
    mut commands: Commands,
    mut cassini_query: Query<(&mut Transform, &mut Velocity, &mut OrbitHistory), With<CassiniMarker>>,
    mut ryugu_query: Query<&mut Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
) {
    if !crash.active || crash.elapsed_seconds < ProbeCrashState::DISPLAY_SECONDS {
        return;
    }
    for (entity, slider) in sliders.iter_mut() {
        let value = match slider.0 {
            ProbeParameter::X => PROBE_R0.x,
            ProbeParameter::Y => PROBE_R0.y,
            ProbeParameter::Z => PROBE_R0.z,
            ProbeParameter::SpeedFactor => PROBE_SPEED_FACTOR,
        };
        commands.entity(entity).insert(SliderValue(value));
    }
    if let Ok((mut transform, mut velocity, mut history)) = cassini_query.single_mut() {
        transform.translation = PROBE_R0;
        velocity.0 = probe_initial_velocity(PROBE_R0, PROBE_SPEED_FACTOR);
        history.0.clear();
        history.0.push_back(PROBE_R0);
    }
    if let Some(mut transform) = ryugu_query.iter_mut().next() {
        transform.translation = Vec3::ZERO;
        transform.rotation = Quat::IDENTITY;
    }
    crash.clear();
    reset_request.0 = true;
}

pub fn reset_after_probe_crash_state_system(
    mut reset_request: ResMut<ProbeCrashResetRequest>,
    mut active_method: ResMut<ActiveGravityMethod>,
    mut performance: ResMut<PerformanceComparisonState>,
    mut inversion: ResMut<TrajectoryInversionState>,
    mut probe_initial: ResMut<ProbeInitialConditions>,
    mut clock: ResMut<SimulationClock>,
    mut blend: ResMut<GravityBlendFactor>,
    mut acceleration: ResMut<GravityAcceleration>,
    mut potential: ResMut<GravityPotential>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut jacobi: ResMut<JacobiHistory>,
    mut benchmark: ResMut<GravityBenchmarkTrajectory>,
    mut sensitivity: ResMut<DensitySensitivityCaches>,
    mut histories: ParamSet<(
        Option<ResMut<RadialGravityHistory>>,
        Option<ResMut<WernerGravityHistory>>,
        Option<ResMut<Eq106GpuHistory>>,
        Option<ResMut<MmfftCompressedHistory>>,
        Option<ResMut<FmmGravityHistory>>,
    )>,
) {
    if !reset_request.0 {
        return;
    }
    *reset_request = ProbeCrashResetRequest(false);
    *active_method = ActiveGravityMethod::RadialAnalytic;
    *performance = PerformanceComparisonState::default();
    *inversion = TrajectoryInversionState::default();
    *probe_initial = ProbeInitialConditions::default();
    clock.reset_state();
    blend.0 = 0.0;
    acceleration.0 = Vec3::ZERO;
    potential.0 = None;
    runtime_error.clear();
    jacobi.reset();
    benchmark.epoch = clock.epoch;
    benchmark.samples.clear();
    benchmark.capture_id = None;
    benchmark.complete = false;
    *sensitivity = DensitySensitivityCaches::default();
    if let Some(history) = histories.p0().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p1().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p2().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p3().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p4().as_deref_mut() {
        history.0.clear();
    }
}
