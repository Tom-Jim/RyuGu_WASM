use crate::interface::components::*;
use crate::cpu::curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory};
use crate::cpu::eq106_operator::Eq106OperatorTensorResource;
use crate::gpu::werner::{WernerAcceleration, WernerPotential};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::{
    ButtonState,
    keyboard::{Key, KeyboardInput},
};
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
pub(crate) struct TrajectoryInversionButton;

#[derive(Component)]
pub(crate) struct TrajectoryInversionPanel;

#[derive(Component)]
pub(crate) struct DensityInversionTimingPanel;

#[derive(Component, Clone, Copy)]
pub(crate) struct DensityInversionTimingLabel(pub usize);

#[derive(Component)]
pub(crate) struct DensityInversionStatusLabel;

#[derive(Component, Clone, Copy)]
pub(crate) struct TrajectoryInversionField {
    pub index: usize,
    pub vector: TrajectoryVectorField,
}

#[derive(Component, Clone, Copy)]
pub(crate) struct TrajectoryInversionFieldText {
    pub index: usize,
    pub vector: TrajectoryVectorField,
}

#[derive(Component)]
pub(crate) struct PerformanceRepeatButton;

#[derive(Component)]
pub(crate) struct PerformanceComparisonPanel;

#[derive(Component)]
pub(crate) struct PerformanceComparisonStatus;

#[derive(Component, Clone, Copy)]
pub(crate) struct PerformanceJacobiAxisLabel(pub u8);

#[derive(Component, Clone, Copy)]
pub(crate) struct PerformanceTimeAxisLabel {
    pub jacobi: bool,
    pub slot: u8,
}

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
