use crate::components::*;
use crate::systems::{
    curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory, PeriodicityDetector},
    eq106_operator::Eq106OperatorTensorResource,
    werner_pipeline::{WernerAcceleration, WernerPotential},
};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::Checked;
use bevy::ui_widgets::{
    Checkbox, Slider, SliderDragState, SliderRange, SliderStep, SliderThumb, SliderValue,
    TrackClick, checkbox_self_update, observe, slider_self_update,
};
use bevy::window::PrimaryWindow;
use std::collections::VecDeque;

const SLIDER_TRACK_COLOR: Color = Color::srgb(0.12, 0.16, 0.2);
const SLIDER_THUMB_COLOR: Color = Color::srgb(0.1, 0.85, 0.9);
const UI_REFERENCE_WIDTH: f32 = 1920.0;
const UI_REFERENCE_HEIGHT: f32 = 1080.0;
const PERFORMANCE_CHART_CONTENT_WIDTH: f32 = 1000.0;

/// Scale fixed-pixel UI values against the logical 16:9 presentation frame.
pub fn update_ui_scale_system(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut ui_scale: ResMut<UiScale>,
) {
    let Some(window) = windows.iter().next() else {
        return;
    };
    let width_scale = window.width() / UI_REFERENCE_WIDTH;
    let height_scale = window.height() / UI_REFERENCE_HEIGHT;
    let scale = width_scale.min(height_scale);
    if scale.is_finite() && scale > 0.0 {
        ui_scale.0 = scale;
    }
}

/// Shows numerical failures as a blocking modal. The simulation is stopped by
/// `physics_system`; this layer only presents the first diagnostic clearly.
pub fn setup_runtime_error_overlay(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                display: Display::None,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                padding: UiRect::all(Val::Px(32.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.02, 0.03, 0.96)),
            ZIndex(1000),
            RuntimeErrorOverlay,
        ))
        .with_children(|parent| {
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(18.0),
                    max_width: Val::Px(960.0),
                    ..default()
                },
                children![
                    (
                        Text::new("Gravity pipeline error\n\n"),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(20.0),
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.35, 0.35)),
                        Node {
                            max_width: Val::Px(900.0),
                            ..default()
                        },
                        RuntimeErrorMessage,
                    ),
                    (
                        Button,
                        RuntimeErrorResetButton,
                        Node {
                            width: Val::Px(210.0),
                            height: Val::Px(38.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            border: UiRect::all(Val::Px(1.0)),
                            border_radius: BorderRadius::all(Val::Px(6.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.08, 0.35, 0.38)),
                        children![(
                            Text::new("Reset parameters"),
                            TextFont {
                                font_size: bevy::text::FontSize::Px(14.0),
                                ..default()
                            },
                            TextColor(Color::srgb(0.92, 1.0, 1.0)),
                        ),],
                    ),
                ],
            ));
        });
}

pub fn runtime_error_overlay_system(
    runtime_error: Res<GravityRuntimeError>,
    mut overlays: Query<&mut Node, With<RuntimeErrorOverlay>>,
    mut messages: Query<&mut Text, With<RuntimeErrorMessage>>,
) {
    let display = if runtime_error.is_active() {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in overlays.iter_mut() {
        node.display = display;
    }
    if let Some(message) = runtime_error.message.as_deref() {
        for mut text in messages.iter_mut() {
            *text = Text::new(format!(
                "Gravity pipeline error\n\n{message}\n\nThe simulation has been paused because no mathematically valid force evaluation is available."
            ));
        }
    }
}

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

#[derive(Component)]
pub(crate) struct PerformanceViewButton;

#[derive(Component)]
pub(crate) struct ThreeDViewButton;

#[derive(Component)]
pub(crate) struct DisplayRotationButton;

#[derive(Component)]
pub(crate) struct PerformanceRepeatButton;

#[derive(Component)]
pub(crate) struct PerformanceComparisonPanel;

#[derive(Component)]
pub(crate) struct PerformanceComparisonStatus;

#[derive(Component, Clone, Copy)]
pub(crate) struct PerformanceJacobiAxisLabel(pub u8);

#[derive(Component)]
pub(crate) struct PerformanceComparisonResult(pub usize);

#[derive(Component, Clone, Copy)]
pub(crate) struct PerformanceMethodCheckbox(pub usize);

#[derive(Component)]
pub(crate) struct PerformanceCheckboxMark;

#[derive(Component)]
pub(crate) struct PerformanceOverlay;

#[derive(Component)]
pub(crate) struct PerformanceFpsPlot;

#[derive(Component)]
pub(crate) struct PerformanceJacobiPlot;

#[derive(Component)]
pub(crate) struct RuntimeErrorOverlay;

#[derive(Component)]
pub(crate) struct RuntimeErrorMessage;

#[derive(Component)]
pub(crate) struct RuntimeErrorResetButton;

#[derive(Component, Clone, Copy)]
pub(crate) struct PerformanceChartSegment {
    pub series: usize,
    pub index: usize,
    pub jacobi: bool,
}

fn performance_chart_series_count(_is_jacobi_plot: bool) -> usize {
    5
}

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
    mut curved_arc: ParamSet<(
        ResMut<CurvedArcPlannerState>,
        ResMut<PeriodicityDetector>,
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
    curved_arc.p0().reset();
    curved_arc.p1().reset();
    curved_arc.p2().reset();

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

fn axis_text_owned(label: String, left: f32, top: f32) -> impl Bundle {
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

fn performance_time_axis() -> impl Bundle {
    performance_time_axis_at(224.0)
}

fn performance_time_axis_jacobi() -> impl Bundle {
    performance_time_axis_at(306.0)
}

fn performance_time_axis_at(top: f32) -> impl Bundle {
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
            axis_text("0", 0.0, 0.0),
            axis_text("25", 252.0, 0.0),
            axis_text("50", 512.0, 0.0),
            axis_text("75", 772.0, 0.0),
            axis_text_owned(format!("{PERFORMANCE_TEST_DURATION_HOURS:.0}"), 1018.0, 0.0,),
            axis_text("Detector runtime (h)", 454.0, 11.0),
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
                        Text::new("Rotating-frame Jacobi constants"),
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
                    axis_text("C_J", 8.0, 168.0),
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
) {
    for (interaction, performance_button, three_d_button, rotation_button, repeat_button) in
        interactions.iter_mut()
    {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if performance_button.is_some() && !state.active {
            state.start(*active_method);
        } else if three_d_button.is_some() && state.active {
            state.stop();
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

pub fn performance_comparison_system(
    time: Res<Time>,
    mut state: ResMut<PerformanceComparisonState>,
    active_method: Res<ActiveGravityMethod>,
    jacobi: Res<JacobiHistory>,
    mut nodes: ParamSet<(
        Query<&mut Node, With<PerformanceComparisonPanel>>,
        Query<&mut Node, (With<PerformanceViewButton>, Without<ThreeDViewButton>)>,
        Query<&mut Node, (With<ThreeDViewButton>, Without<PerformanceViewButton>)>,
        Query<&mut Node, With<PerformanceRepeatButton>>,
        Query<(&PerformanceChartSegment, &mut Node, &mut UiTransform)>,
    )>,
    mut texts: ParamSet<(
        Query<&mut Text, With<PerformanceComparisonStatus>>,
        Query<(&mut Text, &PerformanceComparisonResult)>,
        Query<(&PerformanceJacobiAxisLabel, &mut Text)>,
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

    if state.active
        && state.measuring
        && state
            .enabled_methods
            .get(state.phase)
            .copied()
            .unwrap_or(false)
        && *active_method == method_for_phase(state.phase)
    {
        let dt = time.delta_secs_f64().max(f64::EPSILON);
        let fps = (1.0 / dt) as f32;
        let phase = state.phase;
        if let Some(history) = state.fps_history.get_mut(phase) {
            push_performance_sample(history, fps);
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
                push_performance_sample(history, sample.jacobi_constant);
            }
            state.jacobi_last_request_ids[phase] = jacobi_request_id;
        }
        state.phase_frames = state.phase_frames.saturating_add(1);
        state.phase_elapsed_seconds += time.delta_secs_f64();
        if state.phase_frames >= PERFORMANCE_PHASE_FRAMES {
            let elapsed = state.phase_elapsed_seconds.max(f64::EPSILON);
            let phase = state.phase;
            if let Some(result) = state.frames_per_second.get_mut(phase) {
                *result = PERFORMANCE_PHASE_FRAMES as f64 / elapsed;
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
            format!(
                "Measuring {} ({} / {} frames)",
                method_for_phase(state.phase).as_str(),
                state.phase_frames,
                PERFORMANCE_PHASE_FRAMES
            )
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

    update_performance_chart_segments(&state, &mut nodes.p4());
}

fn push_performance_sample<T>(history: &mut std::collections::VecDeque<T>, value: T) {
    if history.len() == PERFORMANCE_HISTORY_CAPACITY {
        history.pop_front();
    }
    history.push_back(value);
}

fn jacobi_series_visual_offset(series: usize) -> f32 {
    (series as f32 - 2.0) * 4.0
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

    for (segment, mut node, mut transform) in segments.iter_mut() {
        if !performance_chart_series_enabled(state, segment) {
            node.display = Display::None;
            continue;
        }
        let (y0, y1) = if segment.jacobi {
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
            let a = ((from - jacobi_min) / jacobi_span).clamp(0.0, 1.0) as f32;
            let b = ((to - jacobi_min) / jacobi_span).clamp(0.0, 1.0) as f32;
            (
                (1.0 - a) * 220.0 + jacobi_series_visual_offset(segment.series),
                (1.0 - b) * 220.0 + jacobi_series_visual_offset(segment.series),
            )
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
            (y0, y1)
        };
        let history_len = if segment.jacobi {
            state
                .jacobi_history
                .get(segment.series)
                .map_or(0, VecDeque::len)
        } else {
            state
                .fps_history
                .get(segment.series)
                .map_or(0, VecDeque::len)
        };
        // Each algorithm owns its own sample count, but every series is
        // stretched to the same detector-runtime axis: 0..100 hours.
        let width = if history_len > 1 {
            PERFORMANCE_CHART_CONTENT_WIDTH / (history_len - 1) as f32
        } else {
            1.0
        };
        let x0 = segment.index as f32 * width;
        let x1 = (segment.index + 1) as f32 * width;
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
    let mut values = state
        .jacobi_history
        .iter()
        .flat_map(|series| series.iter().copied())
        .filter(|value| value.is_finite());
    let first = values.next()?;
    let (minimum, maximum) = values.fold((first, first), |(minimum, maximum), value| {
        (minimum.min(value), maximum.max(value))
    });
    let span = maximum - minimum;
    let padding = if span > f64::EPSILON {
        0.08 * span
    } else {
        (maximum.abs() * 0.02).max(1.0e-9)
    };
    Some((minimum - padding, maximum + padding))
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
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
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
    radial: Option<Res<RadialGravitySource>>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    eq106_sources: Option<Res<crate::systems::curved_arc::Eq106SourceData>>,
    eq106_tensor: Option<Res<Eq106OperatorTensorResource>>,
    mmfft: Option<Res<MmfftCompressedSource>>,
    fmm: Option<Res<FmmSource>>,
    mut estimate: ResMut<GpuMemoryEstimate>,
) {
    let mut bytes = [0_u64; 5];
    if let Some(source) = radial {
        bytes[0] = source.bytes.len() as u64 + 32 + 2 * reduction_buffer_bytes(source.count);
    }
    if let Some(topology) = topology {
        let face_count = (topology.triangles.len() / 3) as u64;
        let edge_count = face_count * 3 / 2;
        let item_count = edge_count.max(face_count) as u32;
        bytes[1] = edge_count * 80 + face_count * 64 + 32 + 2 * reduction_buffer_bytes(item_count);
    }
    if let (Some(source), Some(tensor)) = (eq106_sources, eq106_tensor) {
        bytes[2] = source.sources.len() as u64 * 16
            + 64 * 8
            + tensor.tensor.coefficients.len() as u64 * 4
            + 64
            + 129 * 32
            + 2 * 8 * 9 * 16;
    }
    if let Some(source) = mmfft {
        bytes[3] = source.bytes.len() as u64 + 48 + 2 * reduction_buffer_bytes(source.count);
    }
    if let Some(source) = fmm {
        bytes[4] = source.bytes.len() as u64 + 32 + 2 * reduction_buffer_bytes(source.node_count);
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
        "VRAM estimate: {} {} ({active_share:.1}% of total)\n{}",
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
        ResMut<PeriodicityDetector>,
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
    // Manual method changes start a new Jacobi trace. During the performance
    // rotation, preserve the shared source long enough for the next method's
    // GPU readback to arrive; per-method benchmark histories remain isolated.
    if !performance.active {
        jacobi_history.reset();
    }
    curved_arc.p0().reset();
    curved_arc.p1().reset();
    curved_arc.p2().reset();
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
mod performance_chart_tests {
    use super::{
        PerformanceChartSegment, clear_performance_method_history, format_vram_text,
        jacobi_series_visual_offset, performance_chart_series_count,
        performance_chart_series_enabled,
    };
    use crate::components::{ActiveGravityMethod, GpuMemoryEstimate, PerformanceComparisonState};
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
        state.jacobi_history[1].push_back(1.0);
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
        state.jacobi_history[2] = VecDeque::from([1.0]);
        state.jacobi_history[3] = VecDeque::from([2.0]);
        clear_performance_method_history(&mut state, 2);
        state.enabled_methods[2] = false;

        assert!(state.jacobi_history[2].is_empty());
        assert_eq!(state.jacobi_history[3], VecDeque::from([2.0]));
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
    fn overlapping_jacobi_series_receive_distinct_visual_offsets() {
        let offsets: Vec<f32> = (0..5).map(jacobi_series_visual_offset).collect();
        assert_eq!(offsets, [-8.0, -4.0, 0.0, 4.0, 8.0]);
    }

    #[test]
    fn vram_label_follows_active_method_and_reports_all_slots() {
        let memory = GpuMemoryEstimate {
            bytes: [1024, 2048, 3 * 1024, 4 * 1024, 5 * 1024],
        };
        let text = format_vram_text(ActiveGravityMethod::Fmm, memory);
        assert!(text.starts_with("VRAM estimate: FMM 5.0 KB"));
        assert!(text.contains("R 1.0 KB"));
        assert!(text.contains("106 3.0 KB"));
    }
}
