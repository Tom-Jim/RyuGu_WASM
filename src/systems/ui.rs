use crate::components::*;
use crate::systems::werner_pipeline::{WernerAcceleration, WernerPotential};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui_widgets::{
    Slider, SliderDragState, SliderRange, SliderStep, SliderThumb, SliderValue, TrackClick,
    observe, slider_self_update,
};

const SLIDER_TRACK_COLOR: Color = Color::srgb(0.12, 0.16, 0.2);
const SLIDER_THUMB_COLOR: Color = Color::srgb(0.1, 0.85, 0.9);

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbeParameter {
    X,
    Y,
    Z,
    SpeedFactor,
}

#[derive(Component)]
pub(crate) struct ProbeSlider(ProbeParameter, bool);

#[derive(Component)]
pub(crate) struct ProbeSliderThumb;

#[derive(Component)]
pub(crate) struct ProbeValueLabel(ProbeParameter);

#[derive(Component)]
pub(crate) struct SimulationAccelerationSlider;

#[derive(Component)]
pub(crate) struct SimulationAccelerationSliderThumb;

#[derive(Component)]
pub(crate) struct SimulationAccelerationValueLabel;

fn probe_value_text(parameter: ProbeParameter, conditions: ProbeInitialConditions) -> String {
    match parameter {
        ProbeParameter::X => format!("{:.0}", conditions.position.x),
        ProbeParameter::Y => format!("{:.0}", conditions.position.y),
        ProbeParameter::Z => format!("{:.0}", conditions.position.z),
        ProbeParameter::SpeedFactor => format!("{:.3}", conditions.speed_factor),
    }
}

fn quantize_probe_value(parameter: ProbeParameter, value: f32) -> f32 {
    let (minimum, maximum, step) = match parameter {
        ProbeParameter::X | ProbeParameter::Y | ProbeParameter::Z => (-2000.0, 2000.0, 40.0),
        ProbeParameter::SpeedFactor => (0.0, 2.0, 0.02),
    };
    (minimum + ((value - minimum) / step).round() * step).clamp(minimum, maximum)
}

fn probe_slider(
    parameter: ProbeParameter,
    value: f32,
    minimum: f32,
    maximum: f32,
    step: f32,
) -> impl Bundle {
    (
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Stretch,
            height: px(14),
            width: px(270),
            ..default()
        },
        ProbeSlider(parameter, false),
        Hovered::default(),
        Slider {
            track_click: TrackClick::Snap,
            ..default()
        },
        SliderValue(value),
        SliderRange::new(minimum, maximum),
        SliderStep(step),
        observe(slider_self_update),
        Children::spawn((
            Spawn((
                Node {
                    height: px(6),
                    border_radius: BorderRadius::all(px(3)),
                    ..default()
                },
                BackgroundColor(SLIDER_TRACK_COLOR),
            )),
            Spawn((
                Node {
                    display: Display::Flex,
                    position_type: PositionType::Absolute,
                    left: px(0),
                    right: px(14),
                    top: px(0),
                    bottom: px(0),
                    ..default()
                },
                children![(
                    ProbeSliderThumb,
                    SliderThumb,
                    Node {
                        width: px(14),
                        height: px(14),
                        position_type: PositionType::Absolute,
                        left: percent(0),
                        border_radius: BorderRadius::MAX,
                        ..default()
                    },
                    BackgroundColor(SLIDER_THUMB_COLOR),
                )],
            )),
        )),
    )
}

fn simulation_acceleration_slider(value: u32) -> impl Bundle {
    (
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Stretch,
            height: px(14),
            width: px(190),
            ..default()
        },
        SimulationAccelerationSlider,
        Hovered::default(),
        Slider {
            track_click: TrackClick::Snap,
            ..default()
        },
        SliderValue(value as f32),
        SliderRange::new(
            MIN_SIMULATION_ACCELERATION as f32,
            MAX_SIMULATION_ACCELERATION as f32,
        ),
        SliderStep(1.0),
        observe(slider_self_update),
        Children::spawn((
            Spawn((
                Node {
                    height: px(6),
                    border_radius: BorderRadius::all(px(3)),
                    ..default()
                },
                BackgroundColor(SLIDER_TRACK_COLOR),
            )),
            Spawn((
                Node {
                    display: Display::Flex,
                    position_type: PositionType::Absolute,
                    left: px(0),
                    right: px(14),
                    top: px(0),
                    bottom: px(0),
                    ..default()
                },
                children![(
                    SimulationAccelerationSliderThumb,
                    SliderThumb,
                    Node {
                        width: px(14),
                        height: px(14),
                        position_type: PositionType::Absolute,
                        left: percent(0),
                        border_radius: BorderRadius::MAX,
                        ..default()
                    },
                    BackgroundColor(SLIDER_THUMB_COLOR),
                )],
            )),
        )),
    )
}

pub fn setup_simulation_acceleration_control(
    mut commands: Commands,
    simulation_acceleration: Res<SimulationAcceleration>,
) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(15),
                top: px(43),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::SpaceEvenly,
                width: px(300),
                height: px(82),
                padding: UiRect::all(px(10)),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.06, 0.9)),
            BorderColor::all(Color::srgba(0.3, 0.7, 0.75, 0.65)),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Simulation acceleration"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.95, 1.0)),
            ));
            panel
                .spawn(Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn(simulation_acceleration_slider(
                        simulation_acceleration.stable_steps(),
                    ));
                    row.spawn((
                        Text::new(format!("{}x", simulation_acceleration.stable_steps())),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(13.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.75, 0.95, 0.95)),
                        Node {
                            width: px(40),
                            ..default()
                        },
                        SimulationAccelerationValueLabel,
                    ));
                });
        });
}

pub fn setup_probe_controls(mut commands: Commands, probe_initial: Res<ProbeInitialConditions>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(15),
                bottom: px(15),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::SpaceEvenly,
                width: px(450),
                height: px(270),
                padding: UiRect::all(px(12)),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.06, 0.9)),
            BorderColor::all(Color::srgba(0.3, 0.7, 0.75, 0.65)),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Probe initial conditions"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(15.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.95, 1.0)),
            ));

            let controls = [
                (
                    "X",
                    ProbeParameter::X,
                    probe_initial.position.x,
                    -2000.0,
                    2000.0,
                    40.0,
                ),
                (
                    "Y",
                    ProbeParameter::Y,
                    probe_initial.position.y,
                    -2000.0,
                    2000.0,
                    40.0,
                ),
                (
                    "Z",
                    ProbeParameter::Z,
                    probe_initial.position.z,
                    -2000.0,
                    2000.0,
                    40.0,
                ),
                (
                    "Speed",
                    ProbeParameter::SpeedFactor,
                    probe_initial.speed_factor,
                    0.0,
                    2.0,
                    0.02,
                ),
            ];

            for (name, parameter, value, minimum, maximum, step) in controls {
                panel
                    .spawn(Node {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: px(8),
                        height: px(18),
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Text::new(name),
                            TextFont {
                                font_size: bevy::text::FontSize::Px(13.0),
                                ..default()
                            },
                            TextColor(Color::srgb(0.8, 0.8, 0.85)),
                            Node {
                                width: px(48),
                                ..default()
                            },
                        ));
                        row.spawn(probe_slider(parameter, value, minimum, maximum, step));
                        row.spawn((
                            Text::new(probe_value_text(parameter, *probe_initial)),
                            TextFont {
                                font_size: bevy::text::FontSize::Px(13.0),
                                ..default()
                            },
                            TextColor(Color::srgb(0.75, 0.95, 0.95)),
                            Node {
                                width: px(70),
                                ..default()
                            },
                            ProbeValueLabel(parameter),
                        ));
                    });
            }
        });
}

pub fn probe_slider_visual_system(
    sliders: Query<
        (
            Entity,
            &SliderValue,
            &SliderRange,
            &Hovered,
            &SliderDragState,
        ),
        (
            Or<(
                Changed<SliderValue>,
                Changed<Hovered>,
                Changed<SliderDragState>,
            )>,
            With<ProbeSlider>,
        ),
    >,
    children: Query<&Children>,
    mut thumbs: Query<
        (&mut Node, &mut BackgroundColor),
        (With<ProbeSliderThumb>, Without<ProbeSlider>),
    >,
) {
    for (slider_entity, value, range, hovered, drag_state) in sliders.iter() {
        for descendant in children.iter_descendants(slider_entity) {
            if let Ok((mut thumb_node, mut thumb_color)) = thumbs.get_mut(descendant) {
                thumb_node.left = percent(range.thumb_position(value.0) * 100.0);
                thumb_color.0 = if hovered.0 || drag_state.dragging {
                    SLIDER_THUMB_COLOR.lighter(0.25)
                } else {
                    SLIDER_THUMB_COLOR
                };
            }
        }
    }
}

pub fn simulation_acceleration_slider_visual_system(
    sliders: Query<
        (
            Entity,
            &SliderValue,
            &SliderRange,
            &Hovered,
            &SliderDragState,
        ),
        (
            Or<(
                Changed<SliderValue>,
                Changed<Hovered>,
                Changed<SliderDragState>,
            )>,
            With<SimulationAccelerationSlider>,
        ),
    >,
    children: Query<&Children>,
    mut thumbs: Query<
        (&mut Node, &mut BackgroundColor),
        (
            With<SimulationAccelerationSliderThumb>,
            Without<SimulationAccelerationSlider>,
        ),
    >,
) {
    for (slider_entity, value, range, hovered, drag_state) in sliders.iter() {
        for descendant in children.iter_descendants(slider_entity) {
            if let Ok((mut thumb_node, mut thumb_color)) = thumbs.get_mut(descendant) {
                thumb_node.left = percent(range.thumb_position(value.0) * 100.0);
                thumb_color.0 = if hovered.0 || drag_state.dragging {
                    SLIDER_THUMB_COLOR.lighter(0.25)
                } else {
                    SLIDER_THUMB_COLOR
                };
            }
        }
    }
}

pub fn simulation_acceleration_slider_system(
    mut commands: Commands,
    sliders: Query<
        (Entity, &SliderValue),
        (Changed<SliderValue>, With<SimulationAccelerationSlider>),
    >,
    mut labels: Query<&mut Text, With<SimulationAccelerationValueLabel>>,
    mut simulation_acceleration: ResMut<SimulationAcceleration>,
) {
    for (entity, value) in sliders.iter() {
        let stable_steps = (value.0.round() as u32)
            .clamp(MIN_SIMULATION_ACCELERATION, MAX_SIMULATION_ACCELERATION);
        let snapped = stable_steps as f32;
        if (value.0 - snapped).abs() > f32::EPSILON {
            commands.entity(entity).insert(SliderValue(snapped));
        }
        if simulation_acceleration.0 != stable_steps {
            simulation_acceleration.0 = stable_steps;
        }
        for mut label in labels.iter_mut() {
            **label = format!("{stable_steps}x");
        }
    }
}

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

    for (label, mut text) in labels.iter_mut() {
        **text = probe_value_text(label.0, next);
    }

    if next == *probe_initial {
        return;
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

    if let Ok((mut transform, mut velocity, mut history)) = cassini_query.single_mut() {
        transform.translation = next.position;
        velocity.0 = next.velocity();
        history.0.clear();
    }
    if let Some(mut transform) = ryugu_query.iter_mut().next() {
        transform.rotation = Quat::IDENTITY;
        transform.translation = Vec3::ZERO;
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
}

pub fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsTextMarker>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    if let Some(mut text) = query.iter_mut().next() {
        *text = Text::new(format!("FPS: {fps:.0}"));
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
    probe_initial: Res<ProbeInitialConditions>,
    mode: Res<CameraMode>,
    show_normals: Res<ShowNormals>,
    show_section: Res<ShowSection>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
    mut gravity_blend: ResMut<GravityBlendFactor>,
    mut radial_potential: ResMut<GravityPotential>,
    mut werner_potential: Option<ResMut<WernerPotential>>,
    mut radial_samples: Option<ResMut<RadialGravityHistory>>,
    mut werner_samples: Option<ResMut<WernerGravityHistory>>,
    mut simulation_clock: ResMut<SimulationClock>,
    mut jacobi_history: ResMut<JacobiHistory>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        With<CassiniMarker>,
    >,
    mut ryugu_query: Query<&mut Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
) {
    if !keyboard.just_pressed(KeyCode::KeyG) {
        return;
    }
    *active_method = match *active_method {
        ActiveGravityMethod::RadialAnalytic => ActiveGravityMethod::HomogeneousWerner,
        ActiveGravityMethod::HomogeneousWerner => ActiveGravityMethod::RadialAnalytic,
    };
    // The newly selected GPU path may not have produced a sample for the reset
    // probe position yet. Warm it up from the Newtonian anchor again instead of
    // applying a full-strength stale readback from before the method switch.
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
    if let Ok((mut c_transform, mut c_velocity, mut c_history)) = cassini_query.single_mut()
        && let Some(mut r_transform) = ryugu_query.iter_mut().next()
    {
        c_transform.translation = probe_initial.position;
        c_velocity.0 = probe_initial.velocity();
        // Reset probe state so the new trajectory starts clean: drop the old
        // history line, undo accumulated spin, and keep Ryugu centered at CoM.
        c_history.0.clear();
        r_transform.rotation = Quat::IDENTITY;
        r_transform.translation = Vec3::ZERO;
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
pub fn update_hint_on_mode_change(
    active_method: ResMut<ActiveGravityMethod>,
    mode: Res<CameraMode>,
    show_normals: Res<ShowNormals>,
    show_section: Res<ShowSection>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !mode.is_changed() && !show_normals.is_changed() && !show_section.is_changed() {
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
