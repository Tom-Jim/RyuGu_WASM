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
            FocusPolicy::Pass,
            TextFont {
                font_size: bevy::text::FontSize::Px(10.0),
                ..default()
            },
            TextColor(Color::srgb(0.84, 0.96, 1.0)),
        )],
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = "
export function savePlanningCurve(text) {
  if (text) localStorage.setItem('ryugu-source-curve-v2', text);
  else localStorage.removeItem('ryugu-source-curve-v2');
}
export function loadPlanningCurve() {
  return localStorage.getItem('ryugu-source-curve-v2') || '';
}
export function downloadPlanningCurve(name, text, mime) {
  const url = URL.createObjectURL(new Blob([text], {type: mime}));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
")]
unsafe extern "C" {
    #[wasm_bindgen(js_name = savePlanningCurve)]
    fn save_planning_curve(text: &str);
    #[wasm_bindgen(js_name = loadPlanningCurve)]
    fn load_planning_curve() -> String;
    #[wasm_bindgen(js_name = downloadPlanningCurve)]
    fn download_planning_curve(name: &str, text: &str, mime: &str);
}

#[cfg(target_arch = "wasm32")]
fn source_curve_csv(samples: &[PlanningSourceCurveSample]) -> String {
    let mut text = String::from(
        "source_count,eq106_raw_ms,eq106_certified_ms,fft_ms,tree_ms,eq_build_ms,fft_build_ms,tree_build_ms,eq_query_ns,fft_query_ns,tree_query_ns,eq_raw_eligible,eq_cert_eligible,fft_eligible,tree_eligible\n",
    );
    for sample in samples {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            sample.source_count,
            sample.times_ms[0], sample.times_ms[1], sample.times_ms[2], sample.times_ms[3],
            sample.build_ms[0], sample.build_ms[1], sample.build_ms[2],
            sample.query_ns_per_target[0], sample.query_ns_per_target[1], sample.query_ns_per_target[2],
            sample.eligible[0], sample.eligible[1], sample.eligible[2], sample.eligible[3],
        ));
    }
    text
}

#[cfg(target_arch = "wasm32")]
fn source_curve_json(samples: &[PlanningSourceCurveSample]) -> String {
    let rows = samples
        .iter()
        .map(|sample| format!(
            "{{\"source_count\":{},\"times_ms\":{:?},\"build_ms\":{:?},\"query_ns_per_target\":{:?},\"eligible\":{:?}}}",
            sample.source_count, sample.times_ms, sample.build_ms,
            sample.query_ns_per_target, sample.eligible,
        ))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"schema\":2,\"samples\":[{rows}]}}")
}

pub(crate) fn persist_source_curve(samples: &[PlanningSourceCurveSample]) {
    #[cfg(target_arch = "wasm32")]
    save_planning_curve(&source_curve_csv(samples));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = samples;
}

pub fn restore_source_curve_system(mut planning: ResMut<PlanningComparisonState>) {
    #[cfg(target_arch = "wasm32")]
    {
        let restored = load_planning_curve();
        let mut samples = Vec::new();
        for line in restored.lines().skip(1) {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 15 {
                continue;
            }
            let number = |index: usize| fields[index].parse::<f64>().ok();
            let boolean = |index: usize| fields[index].parse::<bool>().ok();
            let Some(source_count) = fields[0].parse::<u32>().ok() else {
                continue;
            };
            let Some(sample) = (|| Some(PlanningSourceCurveSample {
                source_count,
                times_ms: [number(1)?, number(2)?, number(3)?, number(4)?],
                build_ms: [number(5)?, number(6)?, number(7)?],
                query_ns_per_target: [number(8)?, number(9)?, number(10)?],
                eligible: [boolean(11)?, boolean(12)?, boolean(13)?, boolean(14)?],
            }))() else {
                continue;
            };
            samples.push(sample);
        }
        if !samples.is_empty() {
            planning.source_curve_samples = samples;
            planning.source_curve_visible = true;
            planning.status = "Restored incremental source-crossover results.".into();
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = &mut planning;
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
        (2, "Eq.106 numeric proxy"),
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
                    row.spawn((
                        selection_button("Quadrature sources 1K->262K", false, 235.0),
                        SourceScaleCurveButton,
                    ));
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

    commands
        .spawn((
            SourceScaleCurveOverlay,
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
            BackgroundColor(Color::srgba(0.01, 0.015, 0.025, 0.92)),
            ZIndex(900),
        ))
        .with_children(|overlay| {
            overlay
                .spawn((
                    Node {
                        width: px(900),
                        height: px(610),
                        padding: UiRect::all(px(20)),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(10),
                        border: UiRect::all(px(1)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.025, 0.045, 0.07)),
                    BorderColor::all(Color::srgb(0.25, 0.75, 0.8)),
                ))
                .with_children(|card| {
                    card.spawn((
                        Text::new("Quadrature-source crossover - fixed 56 density unknowns"),
                        TextFont { font_size: bevy::text::FontSize::Px(18.0), ..default() },
                        TextColor(Color::srgb(0.88, 0.98, 1.0)),
                    ));
                    card.spawn(Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: px(8),
                        ..default()
                    }).with_children(|row| {
                        row.spawn((
                            selection_button("Export CSV", false, 130.0),
                            SourceScaleCurveExportButton(false),
                        ));
                        row.spawn((
                            selection_button("Export JSON", false, 130.0),
                            SourceScaleCurveExportButton(true),
                        ));
                    });
                    card.spawn((
                        Button,
                        SourceScaleCurveCloseButton,
                        Node {
                            position_type: PositionType::Absolute,
                            right: px(12),
                            top: px(10),
                            width: px(34),
                            height: px(30),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.35, 0.08, 0.1)),
                        children![(Text::new("X"), TextColor(Color::WHITE))],
                    ));
                    card.spawn((
                        Node {
                            position_type: PositionType::Relative,
                            width: px(840),
                            height: px(330),
                            border: UiRect::all(px(1)),
                            ..default()
                        },
                        BorderColor::all(Color::srgb(0.2, 0.38, 0.45)),
                    )).with_children(|plot| {
                        for method in 0..4 {
                            let color = match method {
                                0 => Color::srgb(0.2, 0.95, 1.0),
                                1 => Color::srgb(0.65, 0.45, 1.0),
                                2 => Color::srgb(1.0, 0.65, 0.2),
                                _ => Color::srgb(0.35, 0.95, 0.5),
                            };
                            for index in 0..(PLANNING_SOURCE_COUNTS.len() - 1) {
                                plot.spawn((
                                    SourceScaleCurveSegment { method, index },
                                    Node {
                                        position_type: PositionType::Absolute,
                                        display: Display::None,
                                        height: px(3),
                                        ..default()
                                    },
                                    UiTransform::IDENTITY,
                                    BackgroundColor(color),
                                ));
                            }
                        }
                    });
                    card.spawn((
                        SourceScaleCurveSummary,
                        Text::new("Waiting for source-count runs"),
                        TextFont { font_size: bevy::text::FontSize::Px(11.0), ..default() },
                        TextColor(Color::srgb(0.74, 0.9, 0.94)),
                        Node { max_width: px(840), ..default() },
                    ));
                });
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
    curve_interactions: Query<&Interaction, (Changed<Interaction>, With<SourceScaleCurveButton>)>,
    close_interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SourceScaleCurveCloseButton>),
    >,
    export_interactions: Query<
        (&Interaction, &SourceScaleCurveExportButton),
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
    if curve_interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        planning.workload_profile = PlanningWorkloadProfile::SourceCrossover;
        planning.selected_metric = ComparisonMetric::SpeedupVsGpuFmm;
        planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
        planning.source_curve_active = true;
        planning.source_curve_visible = true;
        planning.source_curve_index = 0;
        planning.source_curve_repeat = 0;
        planning.source_curve_samples.clear();
        persist_source_curve(&planning.source_curve_samples);
        planning.results = std::array::from_fn(|_| None);
        planning.batch_job = None;
        planning.run_requested = true;
        planning.run_id = planning.run_id.wrapping_add(1);
        *request = PlanningGpuRequest::default();
        *payload = PlanningMethodPayload::default();
        gpu_result.0 = None;
        planning.status = "Quadrature-source crossover queued: 1024 distinct positions, 56 density unknowns, repeat 1/10, Eq.106 proxy first.".into();
    }
    if close_interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        planning.source_curve_active = false;
        planning.source_curve_visible = false;
        planning.source_curve_index = 0;
        planning.source_curve_repeat = 0;
        planning.source_curve_samples.clear();
        persist_source_curve(&planning.source_curve_samples);
        planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
        planning.run_requested = false;
        planning.batch_job = None;
        planning.results = std::array::from_fn(|_| None);
        planning.reference_duration_seconds = 0.0;
        *request = PlanningGpuRequest::default();
        *payload = PlanningMethodPayload::default();
        gpu_result.0 = None;
        planning.status = "Source curve reset.".into();
    }
    if let Some(json) = export_interactions.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Pressed).then_some(button.0)
    }) {
        #[cfg(target_arch = "wasm32")]
        if json {
            download_planning_curve(
                "ryugu-source-crossover.json",
                &source_curve_json(&planning.source_curve_samples),
                "application/json",
            );
        } else {
            download_planning_curve(
                "ryugu-source-crossover.csv",
                &source_curve_csv(&planning.source_curve_samples),
                "text/csv",
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = json;
    }
    if let Some(metric) = metric_interactions.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Pressed).then_some(button.0)
    }) {
        planning.source_curve_active = false;
        planning.source_curve_visible = false;
        planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
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
        planning.source_curve_active = false;
        planning.source_curve_visible = false;
        planning.requested_source_count = PLANNING_SOURCE_COUNTS[0];
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
            planning.requested_source_count,
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
        let Some((voxels, voxel_size)) = crate::cpu::inversion::build_density_voxels(
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
            planning.requested_source_count,
            voxel_size,
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
            "{} candidate preparation: 0 / {} trajectories ({} sources).",
            planning.workload_profile.label(),
            dimensions.0,
            planning.requested_source_count,
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
    let order_rotation = if planning.source_curve_active {
        planning.source_curve_repeat as usize
    } else {
        0
    };
    let method_order = planning_method_order(order_rotation);
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
        raw_gpu_request_count: 0,
        last_request_candidate_count: 0,
        awaiting_gpu: false,
        warm_repetition: false,
        certified_repetition: false,
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
        raw_gravity_error_sum: 0.0,
        raw_gradient_error_sum: 0.0,
        pointwise_gravity_errors: Vec::new(),
        pointwise_gradient_errors: Vec::new(),
        certified_gravity_error_sum: 0.0,
        certified_gravity_reference_sum: 0.0,
        certified_gradient_error_sum: 0.0,
        certified_gradient_reference_sum: 0.0,
        certified_gravity_samples: 0,
        certified_gradient_samples: 0,
        certified_verification_sample_count: 0,
        certified_rejected_sample_count: 0,
        certified_candidate_valid: vec![true; dimensions.0 as usize],
        rejected_sample_count: 0,
        rejection_counts: [0; 6],
        self_fd_step_maxima: [0.0; 5],
        first_rejection: None,
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
        one_time_preparation_ms: 0.0,
        preprocessing_ms: 0.0,
        command_submission_ms: 0.0,
        reduction_ms: 0.0,
        verification_ms: 0.0,
        gpu_completion_map_ms: 0.0,
        warm_evaluation_ms: 0.0,
        certified_warm_evaluation_ms: 0.0,
        certified_full_pass_ms: 0.0,
        first_tile_ms: 0.0,
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

fn planning_method_order(rotation: usize) -> [ActiveGravityMethod; 3] {
    let mut order = [
        ActiveGravityMethod::CurvedArcEq106,
        ActiveGravityMethod::MmfftCompressed,
        ActiveGravityMethod::Fmm,
    ];
    order.rotate_left(rotation % 3);
    order
}

fn source_timing_summary(mut values: Vec<f64>) -> Option<(f64, f64, f64)> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return None;
    }
    let last = values.len() - 1;
    let percentile = |fraction: f64| values[(fraction * last as f64).round() as usize];
    let median = if values.len().is_multiple_of(2) {
        0.5 * (values[last / 2] + values[last / 2 + 1])
    } else {
        values[last / 2]
    };
    Some((median, percentile(0.1), percentile(0.9)))
}

pub fn source_scale_curve_ui_system(
    planning: Res<PlanningComparisonState>,
    mut roots: Query<
        &mut Node,
        (
            With<SourceScaleCurveOverlay>,
            Without<SourceScaleCurveSegment>,
        ),
    >,
    mut summaries: Query<&mut Text, With<SourceScaleCurveSummary>>,
    mut segments: Query<
        (&SourceScaleCurveSegment, &mut Node, &mut UiTransform),
        Without<SourceScaleCurveOverlay>,
    >,
) {
    for mut root in roots.iter_mut() {
        root.display = if planning.source_curve_visible {
            Display::Flex
        } else {
            Display::None
        };
    }
    if !planning.source_curve_visible {
        return;
    }
    let mut points = [[None; PLANNING_SOURCE_COUNTS.len()]; 4];
    let mut eligible_counts = [[0_u32; PLANNING_SOURCE_COUNTS.len()]; 4];
    for method in 0..4 {
        for (source_index, source_count) in PLANNING_SOURCE_COUNTS.iter().copied().enumerate() {
            let values = planning
                .source_curve_samples
                .iter()
                .filter(|sample| sample.source_count == source_count)
                .map(|sample| {
                    eligible_counts[method][source_index] += u32::from(sample.eligible[method]);
                    sample.times_ms[method]
                })
                .collect();
            points[method][source_index] = source_timing_summary(values);
        }
    }
    let maximum = points
        .iter()
        .flatten()
        .filter_map(|point| point.map(|value| value.2))
        .fold(1.0_f64, f64::max);
    let log_min = f64::from(PLANNING_SOURCE_COUNTS[0]).ln();
    let log_span = f64::from(*PLANNING_SOURCE_COUNTS.last().unwrap()).ln() - log_min;
    for (segment, mut node, mut transform) in segments.iter_mut() {
        let (Some(from), Some(to)) = (
            points[segment.method][segment.index],
            points[segment.method][segment.index + 1],
        ) else {
            node.display = Display::None;
            continue;
        };
        let x = |index: usize| {
            ((f64::from(PLANNING_SOURCE_COUNTS[index]).ln() - log_min) / log_span) as f32 * 800.0
                + 20.0
        };
        let y = |milliseconds: f64| (1.0 - milliseconds / maximum) as f32 * 290.0 + 20.0;
        let (x0, x1, y0, y1) = (x(segment.index), x(segment.index + 1), y(from.0), y(to.0));
        let delta = Vec2::new(x1 - x0, y1 - y0);
        let length = delta.length();
        node.display = Display::Flex;
        node.left = px((x0 + x1) * 0.5 - length * 0.5);
        node.top = px((y0 + y1) * 0.5 - 1.5);
        node.width = px(length.max(0.5));
        transform.rotation = Rot2::radians(delta.y.atan2(delta.x));
    }
    let names = [
        "Eq.106 proxy raw",
        "Eq.106 proxy certified estimate",
        "FFT-grid",
        "treecode",
    ];
    let mut lines = vec![
        format!(
            "{} | x: distinct quadrature points (log; density K fixed at 56) | y: measured total time ms | median; P10/P90 below",
            planning.status
        ),
    ];
    for method in 0..4 {
        let values = PLANNING_SOURCE_COUNTS
            .iter()
            .enumerate()
            .filter_map(|(index, source)| {
                points[method][index].map(|(median, p10, p90)| {
                    format!(
                        "{source}: {median:.1} [{p10:.1},{p90:.1}] {}/{} eligible",
                        eligible_counts[method][index], PLANNING_SOURCE_REPEATS
                    )
                })
            })
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("{} - {values}", names[method]));
    }
    if let Some(last) = planning.source_curve_samples.last() {
        lines.push(format!(
            "Latest build-est. ms [Eq/FFT/tree] {:.2}/{:.2}/{:.2}; hot ns/target {:.1}/{:.1}/{:.1}",
            last.build_ms[0], last.build_ms[1], last.build_ms[2],
            last.query_ns_per_target[0], last.query_ns_per_target[1], last.query_ns_per_target[2],
        ));
        let samples = planning
            .source_curve_samples
            .iter()
            .filter(|sample| sample.source_count == last.source_count)
            .collect::<Vec<_>>();
        let build = std::array::from_fn::<_, 3, _>(|method| {
            source_timing_summary(samples.iter().map(|sample| sample.build_ms[method]).collect())
                .map(|summary| summary.0)
                .unwrap_or(f64::NAN)
        });
        let query = std::array::from_fn::<_, 3, _>(|method| {
            source_timing_summary(
                samples
                    .iter()
                    .map(|sample| sample.query_ns_per_target[method])
                    .collect(),
            )
            .map(|summary| summary.0)
            .unwrap_or(f64::NAN)
        });
        lines.push(format!(
            "{} quadrature-point medians: diagnostic build allocation Eq/FFT/tree {:.2}/{:.2}/{:.2} ms; measured hot {:.1}/{:.1}/{:.1} ns/target. No extrapolated Qcross: only directly measured totals are admissible.",
            last.source_count, build[0], build[1], build[2], query[0], query[1], query[2],
        ));
    }
    for mut summary in summaries.iter_mut() {
        *summary = Text::new(lines.join("\n"));
    }
}

#[cfg(test)]
mod planning_method_order_tests {
    use super::planning_method_order;
    use crate::interface::components::ActiveGravityMethod;

    #[test]
    fn repeat_orders_rotate_without_changing_membership() {
        let first = planning_method_order(0);
        let second = planning_method_order(1);
        let third = planning_method_order(2);
        assert_eq!(first[0], ActiveGravityMethod::CurvedArcEq106);
        assert_eq!(second[0], ActiveGravityMethod::MmfftCompressed);
        assert_eq!(third[0], ActiveGravityMethod::Fmm);
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
