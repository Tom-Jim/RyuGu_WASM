pub fn probe_slider_system(
    mut commands: Commands,
    mut sliders: Query<(Entity, &mut ProbeSlider, &SliderValue), Changed<SliderValue>>,
    mut labels: Query<(&ProbeValueLabel, &mut Text)>,
    mut probe_initial: ResMut<ProbeInitialConditions>,
    mut gravity_acceleration: ResMut<GravityAcceleration>,
    mut werner_acceleration: Option<ResMut<WernerAcceleration>>,
    mut gravity_blend: ResMut<GravityBlendFactor>,
    mut radial_potential: ResMut<GravityPotential>,
    mut werner_potential: Option<ResMut<WernerPotential>>,
    mut radial_samples: Option<ResMut<RadialGravityHistory>>,
    mut werner_samples: Option<ResMut<WernerGravityHistory>>,
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
    let mut next = *probe_initial;
    let mut received_value = false;
    for (slider_entity, mut slider, value) in sliders.iter_mut() {
        received_value = true;
        let snapped = if slider.1 {
            quantize_probe_value(slider.0, value.0)
        } else {
            slider.1 = true;
            value.0
        };
        if (value.0 - snapped).abs() > f32::EPSILON {
            commands.entity(slider_entity).insert(SliderValue(snapped));
        }
        match slider.0 {
            ProbeParameter::X => next.position.x = snapped,
            ProbeParameter::Y => next.position.y = snapped,
            ProbeParameter::Z => next.position.z = snapped,
            ProbeParameter::SpeedFactor => next.speed_factor = snapped,
        }
    }

    if !received_value {
        return;
    }

    let scalar_changed = next.position != probe_initial.position
        || next.speed_factor != probe_initial.speed_factor;
    if scalar_changed {
        next.preset = ProbeOrbitPreset::Custom;
    }

    for (label, mut text) in labels.iter_mut() {
        **text = probe_value_text(label.0, next);
    }

    *probe_initial = next;

    gravity_acceleration.0 = Vec3::ZERO;
    if let Some(acceleration) = werner_acceleration.as_deref_mut() {
        acceleration.0 = Vec3::ZERO;
    }
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
    simulation_clock.reset_state();
    jacobi_history.reset();
    curved_arc.p0().reset();
    curved_arc.p1().reset();

    if let Ok((mut transform, mut velocity, mut history)) = cassini_query.single_mut() {
        transform.translation = next.position;
        velocity.0 = next.velocity();
        history.0.clear();
        history.0.push_back(next.position);
    }
    if let Some(mut transform) = ryugu_query.iter_mut().next() {
        transform.rotation = Quat::IDENTITY;
        transform.translation = Vec3::ZERO;
    }
}

pub fn clear_runtime_error_on_probe_change(
    sliders: Query<(), (Changed<SliderValue>, With<ProbeSlider>)>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if !sliders.is_empty() {
        runtime_error.clear();
    }
}

fn default_probe_slider_value(parameter: ProbeParameter) -> f32 {
    let initial = ProbeInitialConditions::default();
    match parameter {
        ProbeParameter::X => initial.position.x,
        ProbeParameter::Y => initial.position.y,
        ProbeParameter::Z => initial.position.z,
        ProbeParameter::SpeedFactor => initial.speed_factor,
    }
}

/// Clears a failed Eq. (106) certificate through the same parameter-change
/// path used by the normal sliders. This keeps GPU histories, the trajectory,
/// and the simulation epoch reset atomically on the following Update tick.
pub fn runtime_error_reset_system(
    mut commands: Commands,
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<Button>,
            With<RuntimeErrorResetButton>,
        ),
    >,
    sliders: Query<(Entity, &ProbeSlider), With<ProbeSlider>>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    runtime_error.clear();
    for (entity, slider) in sliders.iter() {
        commands
            .entity(entity)
            .insert(SliderValue(default_probe_slider_value(slider.0)));
    }
}

pub fn setup_fps_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("FPS: --"),
        TextFont {
            font_size: bevy::text::FontSize::Px(16.0),
            ..default()
        },
        TextColor(Color::srgb(0.6, 1.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(15.0),
            right: Val::Px(15.0),
            ..default()
        },
        FpsTextMarker,
    ));
    commands.spawn((
        Text::new("VRAM estimate: --"),
        TextFont {
            font_size: bevy::text::FontSize::Px(11.0),
            ..default()
        },
        TextColor(Color::srgb(0.65, 0.85, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            // The acceleration panel occupies approximately 43..125 px on
            // the right edge; keep the VRAM readout below it.
            top: Val::Px(134.0),
            right: Val::Px(15.0),
            max_width: Val::Px(460.0),
            ..default()
        },
        VramTextMarker,
    ));
}

fn performance_button(label: &str, width: f32) -> impl Bundle {
    (
        Button,
        Node {
            width: px(width),
            height: px(34),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(6)),
            ..default()
        },
        BackgroundColor(Color::srgb(0.05, 0.25, 0.3)),
        children![(
            Text::new(label),
            TextFont {
                font_size: bevy::text::FontSize::Px(13.0),
                ..default()
            },
            TextColor(Color::srgb(0.9, 1.0, 1.0)),
        )],
    )
}

fn trajectory_vector_text(vector: Vec3) -> String {
    format!("{:.3}, {:.3}, {:.3}", vector.x, vector.y, vector.z)
}

fn parse_trajectory_vector(text: &str) -> Option<Vec3> {
    let values: Vec<f32> = text
        .split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
        .map(str::parse::<f32>)
        .collect::<Result<_, _>>()
        .ok()?;
    (values.len() == 3 && values.iter().all(|value| value.is_finite()))
        .then(|| Vec3::new(values[0], values[1], values[2]))
}

fn trajectory_inversion_field(index: usize, vector: TrajectoryVectorField) -> impl Bundle {
    let title = match vector {
        TrajectoryVectorField::Position => format!("{:02}  x, y, z", index + 1),
        TrajectoryVectorField::Velocity => format!("{:02}  vx, vy, vz", index + 1),
    };
    (
        Button,
        TrajectoryInversionField { index, vector },
        Node {
            width: px(300),
            height: px(22),
            padding: UiRect::horizontal(px(6)),
            align_items: AlignItems::Center,
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(3)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.025, 0.09, 0.14, 0.92)),
        children![(
            Text::new(title),
            TextFont {
                font_size: bevy::text::FontSize::Px(10.0),
                ..default()
            },
            TextColor(Color::srgb(0.7, 0.9, 1.0)),
            TrajectoryInversionFieldText { index, vector },
        )],
    )
}

pub fn setup_trajectory_inversion_controls(mut commands: Commands) {
    for (vector, left, heading) in [
        (
            TrajectoryVectorField::Position,
            22.0,
            "Position 16 uniform samples",
        ),
        (
            TrajectoryVectorField::Velocity,
            330.0,
            "Velocity 16 matching samples",
        ),
    ] {
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(left),
                    top: percent(50),
                    margin: UiRect::top(px(-216)),
                    width: px(300),
                    padding: UiRect::all(px(8)),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(2),
                    display: Display::None,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.005, 0.025, 0.055, 0.94)),
                BorderColor::all(Color::srgb(0.08, 0.5, 0.7)),
                GlobalZIndex(999_999),
                TrajectoryInversionPanel,
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text::new(heading),
                    TextFont {
                        font_size: bevy::text::FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(Color::srgb(0.5, 0.9, 1.0)),
                    Node {
                        margin: UiRect::bottom(px(4)),
                        ..default()
                    },
                ));
                parent.spawn((
                    Text::new("Click a row, type comma-separated values, press Enter."),
                    TextFont {
                        font_size: bevy::text::FontSize::Px(8.5),
                        ..default()
                    },
                    TextColor(Color::srgb(0.54, 0.65, 0.75)),
                    Node {
                        margin: UiRect::bottom(px(4)),
                        ..default()
                    },
                ));
                for index in 0..TRAJECTORY_INVERSION_SAMPLE_COUNT {
                    parent.spawn(trajectory_inversion_field(index, vector));
                }
            });
    }
}

pub fn trajectory_inversion_input_system(
    mut interactions: Query<
        (&Interaction, &TrajectoryInversionField),
        (Changed<Interaction>, With<Button>),
    >,
    mut keyboard: MessageReader<KeyboardInput>,
    active_method: Res<ActiveGravityMethod>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if matches!(
        *active_method,
        ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner
    ) {
        return;
    }
    if inversion.ready {
        for (interaction, field) in interactions.iter_mut() {
            if *interaction == Interaction::Pressed {
                if let Some((index, vector)) = inversion.selected
                    && let Some(value) = parse_trajectory_vector(&inversion.edit_buffer)
                    && let Some(knot) = inversion.knots.get_mut(index)
                {
                    match vector {
                        TrajectoryVectorField::Position => knot.position = value,
                        TrajectoryVectorField::Velocity => knot.velocity = value,
                    }
                    inversion.knots_edited = true;
                    inversion.capture_id = Some(crate::bevy_app::render::hash_trajectory_capture(
                        &inversion.knots,
                    ));
                    inversion.inverted = false;
                    inversion.optimizer = None;
                    inversion.batch_capture_id = None;
                    inversion.results = std::array::from_fn(|_| None);
                    inversion.best_results = std::array::from_fn(|_| None);
                    inversion.displayed_density = None;
                }
                inversion.selected = Some((field.index, field.vector));
                // A click selects the complete vector for immediate replacement.
                inversion.edit_buffer.clear();
                inversion.error = None;
            }
        }
    }
    let Some((index, vector)) = inversion.selected else {
        return;
    };
    for event in keyboard.read() {
        if event.state != ButtonState::Pressed {
            continue;
        }
        match &event.logical_key {
            Key::Backspace => {
                inversion.edit_buffer.pop();
            }
            Key::Escape => {
                inversion.selected = None;
                inversion.edit_buffer.clear();
                inversion.error = None;
            }
            Key::Enter => {
                if let Some(value) = parse_trajectory_vector(&inversion.edit_buffer) {
                    if let Some(knot) = inversion.knots.get_mut(index) {
                        match vector {
                            TrajectoryVectorField::Position => knot.position = value,
                            TrajectoryVectorField::Velocity => knot.velocity = value,
                        }
                        inversion.knots_edited = true;
                        inversion.capture_id = Some(
                            crate::bevy_app::render::hash_trajectory_capture(&inversion.knots),
                        );
                        inversion.inverted = false;
                        inversion.optimizer = None;
                        inversion.batch_capture_id = None;
                        inversion.results = std::array::from_fn(|_| None);
                        inversion.best_results = std::array::from_fn(|_| None);
                        inversion.displayed_density = None;
                        inversion.selected = None;
                        inversion.edit_buffer.clear();
                        inversion.error = None;
                    }
                } else {
                    inversion.error =
                        Some("Enter exactly three finite numbers, separated by commas.".into());
                }
            }
            Key::Character(text) => {
                if text.chars().all(|character| {
                    character.is_ascii_digit()
                        || matches!(character, '-' | '+' | '.' | ',' | ' ' | 'e' | 'E')
                }) {
                    inversion.edit_buffer.push_str(text);
                }
            }
            _ => {
                if let Some(text) = event.text.as_deref()
                    && text.chars().all(|character| {
                        character.is_ascii_digit()
                            || matches!(character, '-' | '+' | '.' | ',' | ' ' | 'e' | 'E')
                    })
                {
                    inversion.edit_buffer.push_str(text);
                }
            }
        }
    }
}

pub fn trajectory_inversion_ui_system(
    inversion: Res<TrajectoryInversionState>,
    active_method: Res<ActiveGravityMethod>,
    mut panels: Query<&mut Node, With<TrajectoryInversionPanel>>,
    mut labels: Query<(&TrajectoryInversionFieldText, &mut Text)>,
    mut fields: Query<(&TrajectoryInversionField, &mut BackgroundColor)>,
) {
    for mut panel in panels.iter_mut() {
        panel.display = if inversion.ready
            && !matches!(
                *active_method,
                ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner
            )
        {
            Display::Flex
        } else {
            Display::None
        };
    }
    for (field, mut text) in labels.iter_mut() {
        let display = if inversion.selected == Some((field.index, field.vector)) {
            inversion.edit_buffer.clone()
        } else if let Some(knot) = inversion.knots.get(field.index) {
            trajectory_vector_text(match field.vector {
                TrajectoryVectorField::Position => knot.position,
                TrajectoryVectorField::Velocity => knot.velocity,
            })
        } else {
            "waiting for 5 s capture…".into()
        };
        **text = format!("{:02}  {display}", field.index + 1);
    }
    for (field, mut color) in fields.iter_mut() {
        color.0 = if inversion.selected == Some((field.index, field.vector)) {
            Color::srgb(0.12, 0.38, 0.52)
        } else {
            Color::srgba(0.025, 0.09, 0.14, 0.92)
        };
    }
}

fn performance_method_row(index: usize, label: &str, color: Color) -> impl Bundle {
    (
        Node {
            display: Display::Flex,
            width: px(460),
            height: px(24),
            align_items: AlignItems::Center,
            column_gap: px(8),
            ..default()
        },
        children![
            (
                // `Button` keeps this native Checkbox in the UI picking path
                // used by this app. Retain `Checkbox` so a later input layer
                // can still drive it through Bevy's ValueChange events.
                Button,
                Checkbox,
                Checked,
                PerformanceMethodCheckbox(index),
                observe(checkbox_self_update),
                Node {
                    width: px(18),
                    height: px(18),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(px(1)),
                    border_radius: BorderRadius::all(px(3)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.08, 0.3, 0.32)),
                children![(
                    Text::new("x"),
                    PerformanceCheckboxMark,
                    TextFont {
                        font_size: bevy::text::FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(Color::srgb(0.95, 1.0, 1.0)),
                )],
            ),
            (
                Text::new(label),
                PerformanceComparisonResult(index),
                TextFont {
                    font_size: bevy::text::FontSize::Px(14.0),
                    ..default()
                },
                TextColor(color),
            ),
        ],
    )
}
