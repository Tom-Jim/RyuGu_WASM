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
    let labels = [(2, "Eq.106"), (3, "MMFFT"), (4, "GPU FMM")];
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
                    for profile in [PlanningWorkloadProfile::First, PlanningWorkloadProfile::Stress]
                    {
                        row.spawn((
                            selection_button(
                                profile.label(),
                                profile == planning.workload_profile,
                                200.0,
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
    mut button_sets: ParamSet<(
        Query<(&ComparisonMetricButton, &mut BackgroundColor)>,
        Query<(&PlanningWorkloadButton, &mut BackgroundColor)>,
    )>,
) {
    if let Some(metric) = metric_interactions.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Pressed).then_some(button.0)
    }) {
        planning.selected_metric = metric;
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
        planning.run_requested = true;
        planning.run_id = planning.run_id.wrapping_add(1);
        planning.status = format!(
            "{} selected and queued. Existing frozen method results will start the batch automatically.",
            profile.label()
        );
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
    inversion: Res<TrajectoryInversionState>,
    mut planning: ResMut<PlanningComparisonState>,
) {
    if !planning.run_requested || planning.batch_job.is_some() {
        return;
    }
    let methods = [
        ActiveGravityMethod::CurvedArcEq106,
        ActiveGravityMethod::MmfftCompressed,
        ActiveGravityMethod::Fmm,
    ];
    let Some(reference) = methods
        .iter()
        .filter_map(|method| inversion.results[method.performance_index()].as_ref())
        .next()
    else {
        planning.status = format!(
            "{} planning queued: complete a frozen-track inversion first.",
            planning.workload_profile.label()
        );
        return;
    };
    let completed = methods
        .iter()
        .filter(|method| inversion.results[method.performance_index()].is_some())
        .count();
    if completed < methods.len() {
        planning.status = format!(
            "{} planning queued: {}/3 frozen-track inversions complete.",
            planning.workload_profile.label(),
            completed
        );
        return;
    }
    let same_frozen_problem = methods.iter().all(|method| {
        inversion.results[method.performance_index()]
            .as_ref()
            .is_some_and(|result| {
                result.capture_id == reference.capture_id
                    && result.source_hash == reference.source_hash
                    && result.capture_epoch == reference.capture_epoch
            })
    });
    if !same_frozen_problem {
        planning.status = format!(
            "{} planning blocked: the three inversions do not share one capture_id, source hash and epoch.",
            planning.workload_profile.label()
        );
        return;
    }
    let dimensions = planning.workload_profile.dimensions();
    planning.batch_job = Some(PlanningBatchJob {
        run_id: planning.run_id,
        profile: planning.workload_profile,
        method: ActiveGravityMethod::CurvedArcEq106,
        capture_id: reference.capture_id,
        source_hash: reference.source_hash,
        voxels: reference.voxels.clone(),
        candidate_count: dimensions.0,
        density_model_count: dimensions.1,
        samples_per_candidate: dimensions.2,
        cursor: 0,
        total_evaluations: u64::from(dimensions.0)
            * u64::from(dimensions.1)
            * u64::from(dimensions.2),
        gravity_error_sum: 0.0,
        gradient_error_sum: 0.0,
        gradient_samples: 0,
        pericenter_error_m: 0.0,
        candidate_min_radius: f32::INFINITY,
        preprocessing_ms: reference.timing.truth_prepare_ms + reference.timing.matrix_build_ms,
        evaluation_ms: 0.0,
        baseline_gravity_error: reference.holdout_rmse,
    });
    planning.status = format!(
        "{} batch planning started: 0/{} evaluations, Eq.106.",
        planning.workload_profile.label(),
        planning.batch_job.as_ref().map_or(0, |job| job.total_evaluations)
    );
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
                Text::new("探测器撞毁\n正在重置场景…"),
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
            **text = format!("探测器撞毁\n{remaining:.1} s 后重置场景");
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
    mut planning: ResMut<PlanningComparisonState>,
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
    *planning = PlanningComparisonState::default();
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
