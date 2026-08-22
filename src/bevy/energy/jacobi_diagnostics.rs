
pub fn record_probe_jacobi_system(
    active_method: Res<ActiveGravityMethod>,
    radial_samples: Option<Res<RadialGravityHistory>>,
    werner_samples: Option<Res<WernerGravityHistory>>,
    eq106_samples: Option<Res<Eq106GpuHistory>>,
    mmfft_samples: Option<Res<MmfftCompressedHistory>>,
    fmm_samples: Option<Res<FmmGravityHistory>>,
    gravity_blend: Res<GravityBlendFactor>,
    clock: Res<SimulationClock>,
    curved_residual: Res<CurvedArcResidualHistory>,
    cassini: Query<(&Transform, &Velocity), With<CassiniMarker>>,
    ryugu: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut history: ResMut<JacobiHistory>,
) {
    if gravity_blend.0 < 1.0 {
        return;
    }
    let active_history = select_history(
        *active_method,
        radial_samples.as_deref(),
        werner_samples.as_deref(),
        eq106_samples.as_deref(),
        mmfft_samples.as_deref(),
        fmm_samples.as_deref(),
    );
    let sample = active_history.and_then(|samples| {
        if *active_method == ActiveGravityMethod::CurvedArcEq106 {
            samples.completed_at_or_before(clock.epoch, clock.elapsed_seconds)
        } else {
            samples.latest_for_epoch(clock.epoch)
        }
    });
    let Some(sample) = sample else {
        return;
    };
    if history.last_request_id == Some(sample.snapshot.request_id) {
        return;
    }

    // The CPU integrator advances the live state using interpolated GPU
    // fields. Accumulate the gravitational work along that same path for all
    // methods, then use the first spectral potential only as the integration
    // constant. This removes asynchronous snapshot lag and per-segment
    // potential gauge jumps from the Jacobi diagnostic.
    let (Ok((probe_transform, probe_velocity)), Ok(ryugu_transform)) =
        (cassini.single(), ryugu.single())
    else {
        return;
    };
    let world_to_body = ryugu_transform.rotation.inverse();
    let body_position = world_to_body * (probe_transform.translation - ryugu_transform.translation);
    let inertial_velocity_body = world_to_body * probe_velocity.0;
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let angular_velocity_body = world_to_body * angular_velocity_world;
    let base_potential = if *active_method == ActiveGravityMethod::CurvedArcEq106 {
        let Some(eq106_history) = eq106_samples.as_ref() else {
            return;
        };
        let Some(potential) = eq106_interpolated_positive_potential(
            &eq106_history.0,
            clock.epoch,
            clock.elapsed_seconds,
            body_position,
        ) else {
            return;
        };
        potential
    } else {
        sample.positive_potential
    };
    let positive_potential =
        if let Some(curve_work) = curved_residual.curve_work_at(clock.elapsed_seconds) {
            let origin_potential = *history
                .eq106_origin_potential
                .get_or_insert(base_potential as f64);
            let origin_curve_work = *history.eq106_origin_curve_work.get_or_insert(curve_work);
            (origin_potential + curve_work - origin_curve_work) as f32
        } else {
            base_potential
        };
    if !positive_potential.is_finite() || positive_potential <= 0.0 {
        return;
    }
    let Some(jacobi_constant) = rotating_frame_jacobi_constant(
        body_position,
        inertial_velocity_body,
        positive_potential,
        angular_velocity_body,
    ) else {
        return;
    };

    let origin = *history
        .origin_simulation_seconds
        .get_or_insert(clock.elapsed_seconds);
    history.elapsed_simulation_seconds = clock.elapsed_seconds - origin;
    history.last_request_id = Some(sample.snapshot.request_id);
    history.last_sample_method = Some(*active_method);
    if history.samples.len() == JACOBI_HISTORY_CAPACITY {
        history.samples.pop_front();
    }
    let simulation_time_seconds = history.elapsed_simulation_seconds;
    history.samples.push_back(JacobiSample {
        simulation_time_seconds,
        jacobi_constant,
        eq106_diagnostics: sample
            .eq106_diagnostics
            .map(|diagnostics| eq106_coordinates_at(diagnostics, body_position)),
    });
}

fn chart_text(value: &str, size: f32, color: Color, node: Node) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font_size: bevy::text::FontSize::Px(size),
            ..default()
        },
        TextColor(color),
        node,
    )
}

pub fn setup_jacobi_chart(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(15),
                bottom: px(15),
                width: px(450),
                height: px(270),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.06, 0.9)),
            BorderColor::all(Color::srgba(0.3, 0.7, 0.75, 0.65)),
        ))
        .with_children(|panel| {
            panel.spawn((
                chart_text(
                    "Rotating-frame Jacobi constant",
                    15.0,
                    Color::srgb(0.85, 0.95, 1.0),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(12),
                        top: px(9),
                        ..default()
                    },
                ),
                JacobiChartTitle,
            ));

            panel.spawn((
                chart_text(
                    "C_J = -- m^2/s^2",
                    11.0,
                    Color::srgb(0.5, 1.0, 0.65),
                    Node {
                        position_type: PositionType::Absolute,
                        right: px(12),
                        top: px(31),
                        ..default()
                    },
                ),
                JacobiChartLabel::Current,
            ));
            panel.spawn((
                chart_text(
                    "dC/|C0| = --",
                    10.0,
                    Color::srgb(0.65, 0.82, 0.9),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(12),
                        top: px(31),
                        ..default()
                    },
                ),
                JacobiChartLabel::RelativeDrift,
            ));

            panel.spawn((
                chart_text(
                    "--",
                    10.0,
                    Color::srgb(0.7, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(7),
                        top: px(49),
                        ..default()
                    },
                ),
                JacobiChartLabel::Maximum,
            ));
            panel.spawn((
                chart_text(
                    "--",
                    10.0,
                    Color::srgb(0.7, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(7),
                        top: px(211),
                        ..default()
                    },
                ),
                JacobiChartLabel::Minimum,
            ));
            panel.spawn((
                chart_text(
                    "C_J (m^2/s^2)",
                    10.0,
                    Color::srgb(0.65, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(7),
                        top: px(128),
                        ..default()
                    },
                ),
                JacobiChartAxisLabel,
            ));

            panel.spawn((
                chart_text(
                    "0 s",
                    10.0,
                    Color::srgb(0.7, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(70),
                        bottom: px(9),
                        ..default()
                    },
                ),
                JacobiChartLabel::TimeStart,
            ));
            panel.spawn((
                chart_text(
                    "--",
                    10.0,
                    Color::srgb(0.7, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        right: px(20),
                        bottom: px(9),
                        ..default()
                    },
                ),
                JacobiChartLabel::TimeEnd,
            ));
            panel.spawn(chart_text(
                "Simulation time",
                10.0,
                Color::srgb(0.65, 0.75, 0.8),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(202),
                    bottom: px(9),
                    ..default()
                },
            ));

            panel
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(73),
                        top: px(48),
                        width: px(CHART_WIDTH),
                        height: px(CHART_HEIGHT),
                        overflow: Overflow::clip(),
                        border: UiRect::all(px(1)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.005, 0.01, 0.02, 0.8)),
                    BorderColor::all(Color::srgba(0.4, 0.55, 0.6, 0.7)),
                ))
                .with_children(|plot| {
                    for fraction in [0.25, 0.5, 0.75] {
                        plot.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(0),
                                top: percent(fraction * 100.0),
                                width: percent(100),
                                height: px(1),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.3, 0.4, 0.45, 0.2)),
                        ));
                    }

                    for index in 0..(JACOBI_HISTORY_CAPACITY - 1) {
                        plot.spawn((
                            JacobiChartSegment(index),
                            Node {
                                position_type: PositionType::Absolute,
                                display: Display::None,
                                height: px(2),
                                ..default()
                            },
                            UiTransform::IDENTITY,
                            BackgroundColor(CHART_LINE_COLOR),
                        ));
                    }

                    plot.spawn((
                        JacobiLatestPoint,
                        Node {
                            position_type: PositionType::Absolute,
                            display: Display::None,
                            width: px(7),
                            height: px(7),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(1.0, 0.82, 0.2)),
                    ));
                });
        });
}

pub fn setup_eq106_residual_chart(mut commands: Commands) {
    commands
        .spawn((
            Eq106ResidualChartRoot,
            Visibility::Hidden,
            Node {
                position_type: PositionType::Absolute,
                right: px(15),
                bottom: px(300),
                width: px(450),
                height: px(270),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.06, 0.9)),
            BorderColor::all(Color::srgba(0.8, 0.45, 0.2, 0.7)),
        ))
        .with_children(|panel| {
            panel.spawn(chart_text(
                "Eq.106 Taylor residual",
                15.0,
                Color::srgb(1.0, 0.88, 0.75),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(12),
                    top: px(9),
                    ..default()
                },
            ));
            panel.spawn((
                chart_text(
                    "|epsilon_106| = --",
                    11.0,
                    Color::srgb(1.0, 0.65, 0.3),
                    Node {
                        position_type: PositionType::Absolute,
                        right: px(12),
                        top: px(31),
                        ..default()
                    },
                ),
                Eq106ResidualChartLabel::Current,
            ));
            panel.spawn((
                chart_text(
                    "Eq.106 warm-up",
                    10.0,
                    Color::srgb(0.65, 0.82, 0.9),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(12),
                        top: px(31),
                        ..default()
                    },
                ),
                Eq106ResidualChartLabel::Status,
            ));
            for (label, top) in [
                (Eq106ResidualChartLabel::Maximum, 49.0),
                (Eq106ResidualChartLabel::Minimum, 211.0),
            ] {
                panel.spawn((
                    chart_text(
                        "--",
                        10.0,
                        Color::srgb(0.7, 0.75, 0.8),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(7),
                            top: px(top),
                            ..default()
                        },
                    ),
                    label,
                ));
            }
            panel.spawn(chart_text(
                "|epsilon_106|",
                10.0,
                Color::srgb(0.65, 0.75, 0.8),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(7),
                    top: px(128),
                    ..default()
                },
            ));
            panel.spawn((
                chart_text(
                    "0 s",
                    10.0,
                    Color::srgb(0.7, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(70),
                        bottom: px(9),
                        ..default()
                    },
                ),
                Eq106ResidualChartLabel::TimeStart,
            ));
            panel.spawn((
                chart_text(
                    "--",
                    10.0,
                    Color::srgb(0.7, 0.75, 0.8),
                    Node {
                        position_type: PositionType::Absolute,
                        right: px(20),
                        bottom: px(9),
                        ..default()
                    },
                ),
                Eq106ResidualChartLabel::TimeEnd,
            ));
            panel.spawn(chart_text(
                "Simulation time",
                10.0,
                Color::srgb(0.65, 0.75, 0.8),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(202),
                    bottom: px(9),
                    ..default()
                },
            ));
            panel
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(73),
                        top: px(48),
                        width: px(CHART_WIDTH),
                        height: px(CHART_HEIGHT),
                        overflow: Overflow::clip(),
                        border: UiRect::all(px(1)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.005, 0.01, 0.02, 0.8)),
                    BorderColor::all(Color::srgba(0.55, 0.4, 0.3, 0.75)),
                ))
                .with_children(|plot| {
                    for fraction in [0.25, 0.5, 0.75] {
                        plot.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(0),
                                top: percent(fraction * 100.0),
                                width: percent(100),
                                height: px(1),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.45, 0.35, 0.3, 0.22)),
                        ));
                    }
                    for index in 0..(JACOBI_HISTORY_CAPACITY - 1) {
                        plot.spawn((
                            Eq106ResidualChartSegment(index),
                            Node {
                                position_type: PositionType::Absolute,
                                display: Display::None,
                                height: px(2),
                                ..default()
                            },
                            UiTransform::IDENTITY,
                            BackgroundColor(RESIDUAL_LINE_COLOR),
                        ));
                    }
                    plot.spawn((
                        Eq106ResidualLatestPoint,
                        Node {
                            position_type: PositionType::Absolute,
                            display: Display::None,
                            width: px(7),
                            height: px(7),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(1.0, 0.85, 0.25)),
                    ));
                });
        });
}

fn format_axis_time(seconds: f64) -> String {
    if seconds.abs() >= 3600.0 {
        format!("{:.1} h", seconds / 3600.0)
    } else {
        format!("{seconds:.0} s")
    }
}

fn jacobi_time_bounds(
    samples: &[JacobiSample],
    simulation_acceleration: SimulationAcceleration,
) -> (f64, f64) {
    let latest_time = samples
        .last()
        .map_or(0.0, |sample| sample.simulation_time_seconds);
    if samples.len() == JACOBI_HISTORY_CAPACITY {
        let first_time = samples
            .first()
            .map_or(latest_time, |sample| sample.simulation_time_seconds);
        (first_time, latest_time.max(first_time + f64::EPSILON))
    } else {
        let expected_window =
            JACOBI_BASE_WINDOW_SECONDS * simulation_acceleration.stable_steps() as f64;
        (0.0, expected_window.max(latest_time).max(f64::EPSILON))
    }
}

