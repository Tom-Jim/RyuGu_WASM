use crate::components::*;
use crate::systems::{
    curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory, PeriodicityDetector},
    werner_pipeline::{WernerAcceleration, WernerPotential},
};
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

#[derive(Component)]
pub(crate) struct PerformanceViewButton;

#[derive(Component)]
pub(crate) struct ThreeDViewButton;

#[derive(Component)]
pub(crate) struct PerformanceComparisonPanel;

#[derive(Component)]
pub(crate) struct PerformanceComparisonStatus;

#[derive(Component)]
pub(crate) struct PerformanceComparisonResult(pub usize);

#[derive(Component)]
pub(crate) struct PerformanceOverlay;

#[derive(Component)]
pub(crate) struct PerformanceFpsPlot;

#[derive(Component)]
pub(crate) struct PerformanceJacobiPlot;

#[derive(Component, Clone, Copy)]
pub(crate) struct PerformanceChartSegment {
    pub series: usize,
    pub index: usize,
    pub jacobi: bool,
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

fn performance_button(label: &str) -> impl Bundle {
    (
        Button,
        Node {
            width: px(230),
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

pub fn setup_performance_controls(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: px(38),
            left: percent(50),
            margin: UiRect::left(px(-240)),
            width: px(480),
            height: px(38),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        GlobalZIndex(1_000_001),
        children![
            (
                performance_button("Performance comparison"),
                PerformanceViewButton,
            ),
            (performance_button("3D display"), ThreeDViewButton,),
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
                Text::new("Three-algorithm performance comparison"),
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
            (
                Text::new("Radial Analytic: -- FPS"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.3, 1.0, 1.0)),
                PerformanceComparisonResult(0),
            ),
            (
                Text::new("Werner Polyhedron: -- FPS"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.35, 0.35)),
                PerformanceComparisonResult(1),
            ),
            (
                Text::new("Equation (106) Curved Arc: -- FPS"),
                TextFont {
                    font_size: bevy::text::FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.45, 1.0)),
                PerformanceComparisonResult(2),
            ),
            (
                Text::new("Jacobi curves: cyan radial | red Werner | purple Eq.106 near-straight | gold Eq.106 residual-adjusted"),
                TextFont { font_size: bevy::text::FontSize::Px(11.0), ..default() },
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
                        TextFont { font_size: bevy::text::FontSize::Px(11.0), ..default() },
                        TextColor(Color::srgb(0.2, 0.95, 1.0)),
                        Node { position_type: PositionType::Absolute, left: px(220), top: px(30), ..default() },
                    ),
                    (
                        Text::new("Werner"),
                        TextFont { font_size: bevy::text::FontSize::Px(11.0), ..default() },
                        TextColor(Color::srgb(1.0, 0.3, 0.3)),
                        Node { position_type: PositionType::Absolute, left: px(360), top: px(30), ..default() },
                    ),
                    (
                        Text::new("Eq.106 near-straight"),
                        TextFont { font_size: bevy::text::FontSize::Px(11.0), ..default() },
                        TextColor(Color::srgb(0.85, 0.45, 1.0)),
                        Node { position_type: PositionType::Absolute, left: px(470), top: px(30), ..default() },
                    ),
                    (
                        Text::new("Eq.106 residual-adjusted"),
                        TextFont { font_size: bevy::text::FontSize::Px(11.0), ..default() },
                        TextColor(Color::srgb(1.0, 0.78, 0.2)),
                        Node { position_type: PositionType::Absolute, left: px(690), top: px(30), ..default() },
                    ),
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
    for (root, fps_plot, jacobi_plot) in roots.iter() {
        let series_count = if fps_plot.is_some() && jacobi_plot.is_none() {
            3
        } else {
            4
        };
        commands.entity(root).with_children(|plot| {
            for series in 0..series_count {
                for index in 0..(PERFORMANCE_HISTORY_CAPACITY - 1) {
                    let color = match series {
                        0 => Color::srgb(0.2, 0.95, 1.0),
                        1 => Color::srgb(1.0, 0.3, 0.3),
                        2 => Color::srgb(0.85, 0.45, 1.0),
                        _ => Color::srgb(1.0, 0.78, 0.2),
                    };
                    plot.spawn((
                        PerformanceChartSegment {
                            series,
                            index,
                            jacobi: series_count == 4,
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
        ),
        (Changed<Interaction>, With<Button>),
    >,
    active_method: Res<ActiveGravityMethod>,
    mut state: ResMut<PerformanceComparisonState>,
) {
    for (interaction, performance_button, three_d_button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if performance_button.is_some() && !state.active {
            state.start(*active_method);
        } else if three_d_button.is_some() && state.active {
            state.stop();
        }
    }
}

pub fn performance_comparison_system(
    time: Res<Time>,
    mut state: ResMut<PerformanceComparisonState>,
    active_method: Res<ActiveGravityMethod>,
    jacobi: Res<JacobiHistory>,
    curved_history: Res<CurvedArcResidualHistory>,
    mut nodes: ParamSet<(
        Query<&mut Node, With<PerformanceComparisonPanel>>,
        Query<&mut Node, (With<PerformanceViewButton>, Without<ThreeDViewButton>)>,
        Query<&mut Node, (With<ThreeDViewButton>, Without<PerformanceViewButton>)>,
        Query<(&PerformanceChartSegment, &mut Node, &mut UiTransform)>,
    )>,
    mut texts: ParamSet<(
        Query<&mut Text, With<PerformanceComparisonStatus>>,
        Query<(&mut Text, &PerformanceComparisonResult)>,
    )>,
) {
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

    if state.active && state.measuring && *active_method == method_for_phase(state.phase) {
        let dt = time.delta_secs_f64().max(f64::EPSILON);
        let fps = (1.0 / dt) as f32;
        let phase = state.phase;
        push_performance_sample(&mut state.fps_history[phase], fps);
        if let Some(sample) = jacobi.samples.back() {
            let series = match *active_method {
                ActiveGravityMethod::RadialAnalytic => 0,
                ActiveGravityMethod::HomogeneousWerner => 1,
                ActiveGravityMethod::CurvedArcEq106 => 2,
            };
            push_performance_sample(&mut state.jacobi_history[series], sample.jacobi_constant);
            if *active_method == ActiveGravityMethod::CurvedArcEq106 {
                let residual = curved_history
                    .samples
                    .back()
                    .and_then(|sample| sample.dual_residual)
                    .unwrap_or(0.0);
                push_performance_sample(
                    &mut state.jacobi_history[3],
                    sample.jacobi_constant + 2.0 * residual,
                );
            }
        }
        state.phase_frames = state.phase_frames.saturating_add(1);
        state.phase_elapsed_seconds += time.delta_secs_f64();
        if state.phase_frames >= PERFORMANCE_PHASE_FRAMES {
            let elapsed = state.phase_elapsed_seconds.max(f64::EPSILON);
            let phase = state.phase;
            state.frames_per_second[phase] = PERFORMANCE_PHASE_FRAMES as f64 / elapsed;
            state.phase = (state.phase + 1) % 3;
            state.phase_frames = 0;
            state.phase_elapsed_seconds = 0.0;
            state.pending_method = Some(method_for_phase(state.phase));
        }
    }

    if let Some(mut text) = texts.p0().iter_mut().next() {
        *text = Text::new(if state.active {
            format!(
                "Measuring {} ({} / {} frames)",
                method_for_phase(state.phase).as_str(),
                state.phase_frames,
                PERFORMANCE_PHASE_FRAMES
            )
        } else {
            "Select 3D display to return.".to_owned()
        });
    }
    for (mut text, result) in texts.p1().iter_mut() {
        let fps = state.frames_per_second[result.0];
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

    update_performance_chart_segments(&state, &mut nodes.p3());
}

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
    let jacobi_values = state
        .jacobi_history
        .iter()
        .flat_map(|series| series.iter().copied());
    let (jacobi_min, jacobi_max) = jacobi_values
        .clone()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        });
    let jacobi_span = (jacobi_max - jacobi_min).max(1.0e-9);

    for (segment, mut node, mut transform) in segments.iter_mut() {
        let (y0, y1) = if segment.jacobi {
            let history = &state.jacobi_history[segment.series];
            let (Some(from), Some(to)) =
                (history.get(segment.index), history.get(segment.index + 1))
            else {
                node.display = Display::None;
                continue;
            };
            let a = ((from - jacobi_min) / jacobi_span).clamp(0.0, 1.0) as f32;
            let b = ((to - jacobi_min) / jacobi_span).clamp(0.0, 1.0) as f32;
            (
                (1.0 - a) * 220.0 + (segment.series as f32 - 1.5) * 2.0,
                (1.0 - b) * 220.0 + (segment.series as f32 - 1.5) * 2.0,
            )
        } else {
            let history = &state.fps_history[segment.series];
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
        let width = if segment.jacobi && state.jacobi_history[segment.series].len() > 1
            || !segment.jacobi && state.fps_history[segment.series].len() > 1
        {
            1000.0 / (PERFORMANCE_HISTORY_CAPACITY - 1) as f32
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

fn method_for_phase(phase: usize) -> ActiveGravityMethod {
    match phase {
        0 => ActiveGravityMethod::RadialAnalytic,
        1 => ActiveGravityMethod::HomogeneousWerner,
        _ => ActiveGravityMethod::CurvedArcEq106,
    }
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
    mut performance: ResMut<PerformanceComparisonState>,
    probe_initial: Res<ProbeInitialConditions>,
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
    let requested_method = performance.pending_method.take().or_else(|| {
        (!performance.active && keyboard.just_pressed(KeyCode::KeyG)).then_some(
            match *active_method {
                ActiveGravityMethod::RadialAnalytic => ActiveGravityMethod::HomogeneousWerner,
                ActiveGravityMethod::HomogeneousWerner => ActiveGravityMethod::CurvedArcEq106,
                ActiveGravityMethod::CurvedArcEq106 => ActiveGravityMethod::RadialAnalytic,
            },
        )
    });
    let Some(next_method) = requested_method else {
        return;
    };
    *active_method = next_method;
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
