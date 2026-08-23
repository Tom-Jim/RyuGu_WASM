fn performance_time_axis() -> impl Bundle {
    performance_time_axis_at(224.0, false)
}

fn performance_time_axis_jacobi() -> impl Bundle {
    performance_time_axis_at(306.0, true)
}

fn performance_time_axis_text(value: &str, left: f32, jacobi: bool, slot: u8) -> impl Bundle {
    (
        Text::new(value),
        PerformanceTimeAxisLabel { jacobi, slot },
        TextFont {
            font_size: bevy::text::FontSize::Px(9.0),
            ..default()
        },
        TextColor(Color::srgb(0.58, 0.72, 0.78)),
        Node {
            position_type: PositionType::Absolute,
            left: px(left),
            top: px(0),
            ..default()
        },
    )
}

fn performance_time_axis_at(top: f32, jacobi: bool) -> impl Bundle {
    let label = if jacobi {
        "Simulation time"
    } else {
        "Benchmark progress"
    };
    (
        Node {
            position_type: PositionType::Absolute,
            left: px(48),
            top: px(top),
            width: px(1040),
            height: px(24),
            ..default()
        },
        children![
            performance_time_axis_text("0", 0.0, jacobi, 0),
            performance_time_axis_text("--", 252.0, jacobi, 1),
            performance_time_axis_text("--", 512.0, jacobi, 2),
            performance_time_axis_text("--", 772.0, jacobi, 3),
            performance_time_axis_text("--", 1010.0, jacobi, 4),
            axis_text(label, 454.0, 11.0),
        ],
    )
}

pub fn setup_performance_controls(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: px(38),
            left: percent(50),
            margin: UiRect::left(px(-325)),
            width: px(650),
            height: px(38),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            column_gap: px(8),
            ..default()
        },
        GlobalZIndex(1_000_001),
        children![
            (
                performance_button("Performance comparison", 190.0),
                PerformanceViewButton,
            ),
            (performance_button("3D display", 120.0), ThreeDViewButton,),
            (
                performance_button("Rotate 90 deg", 130.0),
                DisplayRotationButton,
            ),
            (
                performance_button("Invert trajectory", 140.0),
                TrajectoryInversionButton,
            ),
            (
                performance_button("Repeat benchmark", 150.0),
                PerformanceRepeatButton,
            ),
        ],
    ));

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            top: px(0),
            width: percent(100),
            height: percent(100),
            padding: UiRect {
                top: px(72),
                right: px(28),
                bottom: px(28),
                left: px(28),
            },
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: px(14),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgb(0.005, 0.012, 0.025)),
        GlobalZIndex(1_000_000),
        PerformanceComparisonPanel,
        PerformanceOverlay,
        children![
            (
                Text::new("Five-algorithm performance comparison"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(17.0),
                    ..default()
                },
                TextColor(Color::srgb(0.9, 1.0, 1.0)),
            ),
            (
                Text::new("Preparing benchmark..."),
                TextFont {
                    font_size: bevy::text::FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::srgb(0.7, 0.9, 0.9)),
                PerformanceComparisonStatus,
            ),
            performance_method_row(0, "Radial Analytic: -- FPS", Color::srgb(0.3, 1.0, 1.0)),
            performance_method_row(1, "Werner Polyhedron: -- FPS", Color::srgb(1.0, 0.35, 0.35)),
            performance_method_row(
                2,
                "Equation (106) Curved Arc: -- FPS",
                Color::srgb(0.85, 0.45, 1.0),
            ),
            performance_method_row(
                3,
                "MMFFT + VRAM Compression: -- FPS",
                Color::srgb(1.0, 0.72, 0.2),
            ),
            performance_method_row(
                4,
                "Fast Multipole Method: -- FPS",
                Color::srgb(0.25, 0.9, 0.55),
            ),
            (
                Text::new("Jacobi curves: radial | Werner | Eq.106 | MMFFT compressed | FMM"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(11.0),
                    ..default()
                },
                TextColor(Color::srgb(0.7, 0.82, 0.9)),
            ),
            (
                Node {
                    width: px(1120),
                    height: px(250),
                    border: UiRect::all(px(1)),
                    padding: UiRect::all(px(10)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.01, 0.02, 0.04, 0.95)),
                children![
                    (
                        Text::new("Frame rate by algorithm"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.75, 0.95, 1.0)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(14),
                            top: px(8),
                            ..default()
                        },
                    ),
                    axis_text("FPS", 8.0, 104.0),
                    axis_text("60", 16.0, 28.0),
                    axis_text("30", 16.0, 118.0),
                    axis_text("0", 20.0, 208.0),
                    performance_time_axis(),
                    (
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(48),
                            top: px(34),
                            width: px(1040),
                            height: px(190),
                            border: UiRect::all(px(1)),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.002, 0.006, 0.012, 0.95)),
                        PerformanceFpsPlot,
                        children![
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0),
                                    top: percent(50),
                                    width: percent(100),
                                    height: px(1),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.25))
                            ),
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0),
                                    top: percent(25),
                                    width: percent(100),
                                    height: px(1),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.18))
                            ),
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0),
                                    top: percent(75),
                                    width: percent(100),
                                    height: px(1),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.18))
                            ),
                        ]
                    )
                ]
            ),
            (
                Node {
                    width: px(1120),
                    height: px(330),
                    border: UiRect::all(px(1)),
                    padding: UiRect::all(px(10)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.01, 0.02, 0.04, 0.95)),
                children![
                    (
                        Text::new("Rotating-frame Jacobi relative drift"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.75, 0.95, 1.0)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(14),
                            top: px(8),
                            ..default()
                        },
                    ),
                    (
                        Text::new("Radial analytic"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.2, 0.95, 1.0)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(220),
                            top: px(30),
                            ..default()
                        },
                    ),
                    (
                        Text::new("Werner"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.3, 0.3)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(360),
                            top: px(30),
                            ..default()
                        },
                    ),
                    (
                        Text::new("Eq.106 near-straight"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.85, 0.45, 1.0)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(470),
                            top: px(30),
                            ..default()
                        },
                    ),
                    (
                        Text::new("MMFFT compressed"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.72, 0.2)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(690),
                            top: px(30),
                            ..default()
                        },
                    ),
                    (
                        Text::new("FMM"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.25, 0.9, 0.55)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(820),
                            top: px(30),
                            ..default()
                        },
                    ),
                    axis_text("dC/|C0|", 2.0, 168.0),
                    jacobi_axis_text(0, 2.0, 50.0),
                    jacobi_axis_text(1, 2.0, 174.0),
                    jacobi_axis_text(2, 2.0, 298.0),
                    performance_time_axis_jacobi(),
                    (
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(48),
                            top: px(58),
                            width: px(1040),
                            height: px(250),
                            border: UiRect::all(px(1)),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.002, 0.006, 0.012, 0.95)),
                        PerformanceJacobiPlot,
                        children![
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0),
                                    top: percent(50),
                                    width: percent(100),
                                    height: px(1),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.25))
                            ),
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0),
                                    top: percent(25),
                                    width: percent(100),
                                    height: px(1),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.18))
                            ),
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0),
                                    top: percent(75),
                                    width: percent(100),
                                    height: px(1),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.18))
                            ),
                        ]
                    )
                ]
            ),
        ],
    ));
}

pub fn setup_performance_chart_segments(
    mut commands: Commands,
    roots: Query<
        (
            Entity,
            Option<&PerformanceFpsPlot>,
            Option<&PerformanceJacobiPlot>,
        ),
        Or<(With<PerformanceFpsPlot>, With<PerformanceJacobiPlot>)>,
    >,
) {
    for (root, _fps_plot, jacobi_plot) in roots.iter() {
        let is_jacobi_plot = jacobi_plot.is_some();
        let series_count = performance_chart_series_count(is_jacobi_plot);
        commands.entity(root).with_children(|plot| {
            for series in 0..series_count {
                for index in 0..(PERFORMANCE_HISTORY_CAPACITY - 1) {
                    let color = match series {
                        0 => Color::srgb(0.2, 0.95, 1.0),
                        1 => Color::srgb(1.0, 0.3, 0.3),
                        2 => Color::srgb(0.85, 0.45, 1.0),
                        3 => Color::srgb(1.0, 0.72, 0.2),
                        4 => Color::srgb(0.25, 0.9, 0.55),
                        _ => Color::srgb(0.7, 0.7, 0.7),
                    };
                    plot.spawn((
                        PerformanceChartSegment {
                            series,
                            index,
                            // The plot marker, not the series count, defines
                            // which history array this segment addresses.
                            jacobi: is_jacobi_plot,
                        },
                        Node {
                            position_type: PositionType::Absolute,
                            display: Display::None,
                            height: px(2),
                            ..default()
                        },
                        UiTransform::IDENTITY,
                        BackgroundColor(color),
                    ));
                }
            }
        });
    }
}

pub fn performance_button_system(
    mut interactions: Query<
        (
            &Interaction,
            Option<&PerformanceViewButton>,
            Option<&ThreeDViewButton>,
            Option<&DisplayRotationButton>,
            Option<&PerformanceRepeatButton>,
        ),
        (Changed<Interaction>, With<Button>),
    >,
    active_method: Res<ActiveGravityMethod>,
    mut state: ResMut<PerformanceComparisonState>,
    mut display_rotation: ResMut<DisplayRotation>,
    mut simulation_acceleration: ResMut<SimulationAcceleration>,
) {
    for (interaction, performance_button, three_d_button, rotation_button, repeat_button) in
        interactions.iter_mut()
    {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if performance_button.is_some() && !state.active {
            state.return_simulation_acceleration = simulation_acceleration.0;
            simulation_acceleration.0 = MIN_SIMULATION_ACCELERATION;
            state.start(*active_method);
        } else if three_d_button.is_some() && state.active {
            state.stop();
            simulation_acceleration.0 = state.return_simulation_acceleration;
        } else if repeat_button.is_some() && state.active && !state.measuring {
            state.restart();
        } else if rotation_button.is_some() {
            crate::set_display_rotation(display_rotation.advance());
        }
    }
}

pub fn performance_method_checkbox_system(
    mut state: ResMut<PerformanceComparisonState>,
    checks: Query<(&PerformanceMethodCheckbox, Has<Checked>, &Children)>,
    mut marks: Query<&mut Text, With<PerformanceCheckboxMark>>,
) {
    for (checkbox, checked, children) in checks.iter() {
        let Some(was_enabled) = state.enabled_methods.get(checkbox.0).copied() else {
            continue;
        };
        if was_enabled != checked {
            if let Some(enabled) = state.enabled_methods.get_mut(checkbox.0) {
                *enabled = checked;
            }
            // A toggled method starts a fresh visual series. This prevents a
            // disabled algorithm's old samples from remaining visible or
            // affecting the shared chart scale.
            clear_performance_method_history(&mut state, checkbox.0);
            if state.active && state.measuring && !checked && state.phase == checkbox.0 {
                if let Some((next_phase, next_method)) =
                    state.next_uncompleted_enabled_method(checkbox.0)
                {
                    state.phase = next_phase;
                    state.pending_method = Some(next_method);
                    state.phase_frames = 0;
                    state.phase_elapsed_seconds = 0.0;
                    state.measuring = true;
                } else {
                    state.pending_method = None;
                    state.measuring = false;
                }
            } else if state.active
                && !state.measuring
                && checked
                && let Some((next_phase, next_method)) = state.first_enabled_method()
            {
                state.completed_methods = [false; 5];
                state.frames_per_second = [0.0; 5];
                for history in &mut state.fps_history {
                    history.clear();
                }
                for history in &mut state.jacobi_history {
                    history.clear();
                }
                state.phase = next_phase;
                state.pending_method = Some(next_method);
                state.phase_frames = 0;
                state.phase_elapsed_seconds = 0.0;
                state.measuring = true;
            }
        }
        for child in children.iter() {
            if let Ok(mut mark) = marks.get_mut(child) {
                **mark = if checked {
                    "x".to_owned()
                } else {
                    String::new()
                };
            }
        }
    }
}
fn axis_text(label: &'static str, left: f32, top: f32) -> impl Bundle {
    (
        Text::new(label),
        TextFont {
            font_size: bevy::text::FontSize::Px(9.0),
            ..default()
        },
        TextColor(Color::srgb(0.58, 0.72, 0.78)),
        Node {
            position_type: PositionType::Absolute,
            left: px(left),
            top: px(top),
            ..default()
        },
    )
}

fn jacobi_axis_text(slot: u8, left: f32, top: f32) -> impl Bundle {
    (
        Text::new("--"),
        PerformanceJacobiAxisLabel(slot),
        TextFont {
            font_size: bevy::text::FontSize::Px(9.0),
            ..default()
        },
        TextColor(Color::srgb(0.58, 0.72, 0.78)),
        Node {
            position_type: PositionType::Absolute,
            left: px(left),
            top: px(top),
            ..default()
        },
    )
}

