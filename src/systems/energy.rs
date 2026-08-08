use crate::components::*;
use crate::systems::curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory};
use bevy::math::Rot2;
use bevy::prelude::*;

const CHART_WIDTH: f32 = 350.0;
const CHART_HEIGHT: f32 = 170.0;
const FIXED_UPDATES_PER_SECOND: f64 = 60.0;
const JACOBI_BASE_WINDOW_SECONDS: f64 =
    (JACOBI_HISTORY_CAPACITY - 1) as f64 * TIME_SCALE as f64 / FIXED_UPDATES_PER_SECOND;
const CHART_LINE_COLOR: Color = Color::srgb(0.2, 0.9, 0.45);

#[derive(Component)]
pub(crate) struct JacobiChartSegment(usize);

#[derive(Component)]
pub(crate) struct JacobiLatestPoint;

#[derive(Component)]
pub(crate) struct JacobiChartTitle;

#[derive(Component)]
pub(crate) struct JacobiChartAxisLabel;

#[derive(Component, Clone, Copy)]
pub(crate) enum JacobiChartLabel {
    Current,
    RelativeDrift,
    Minimum,
    Maximum,
    TimeStart,
    TimeEnd,
}

pub fn rotating_frame_jacobi_constant(
    body_position: Vec3,
    inertial_velocity_body_frame: Vec3,
    positive_gravitational_potential: f32,
    angular_velocity_body_frame: Vec3,
) -> Option<f64> {
    if !body_position.is_finite()
        || !inertial_velocity_body_frame.is_finite()
        || !positive_gravitational_potential.is_finite()
        || positive_gravitational_potential <= 0.0
        || !angular_velocity_body_frame.is_finite()
    {
        return None;
    }

    let frame_velocity =
        inertial_velocity_body_frame - angular_velocity_body_frame.cross(body_position);
    let centrifugal_speed = angular_velocity_body_frame.cross(body_position);
    let jacobi = 2.0 * positive_gravitational_potential as f64
        + centrifugal_speed.length_squared() as f64
        - frame_velocity.length_squared() as f64;
    jacobi.is_finite().then_some(jacobi)
}

pub fn record_probe_jacobi_system(
    active_method: Res<ActiveGravityMethod>,
    radial_samples: Option<Res<RadialGravityHistory>>,
    werner_samples: Option<Res<WernerGravityHistory>>,
    gravity_blend: Res<GravityBlendFactor>,
    clock: Res<SimulationClock>,
    mut history: ResMut<JacobiHistory>,
) {
    if gravity_blend.0 < 1.0 {
        return;
    }
    let sample = match *active_method {
        ActiveGravityMethod::RadialAnalytic => radial_samples
            .as_ref()
            .and_then(|samples| samples.0.latest_for_epoch(clock.epoch)),
        ActiveGravityMethod::HomogeneousWerner => werner_samples
            .as_ref()
            .and_then(|samples| samples.0.latest_for_epoch(clock.epoch)),
        // Eq. (106) starts from the same radial potential sample as the
        // near-straight operator; the second performance curve adds its dual
        // residual after this base Jacobi value is recorded.
        ActiveGravityMethod::CurvedArcEq106 => radial_samples
            .as_ref()
            .and_then(|samples| samples.0.latest_for_epoch(clock.epoch)),
    };
    let Some(sample) = sample else {
        return;
    };
    if history.last_request_id == Some(sample.snapshot.request_id) {
        return;
    }

    let world_to_body = sample.snapshot.ryugu_transform.rotation.inverse();
    let body_position = world_to_body
        * (sample.snapshot.probe_position - sample.snapshot.ryugu_transform.translation);
    debug_assert!(
        body_position.distance(sample.snapshot.body_position) < 1.0e-3,
        "gravity snapshot body/world positions diverged"
    );
    let inertial_velocity_body = world_to_body * sample.snapshot.probe_velocity;
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let angular_velocity_body = world_to_body * angular_velocity_world;
    let Some(jacobi_constant) = rotating_frame_jacobi_constant(
        body_position,
        inertial_velocity_body,
        sample.positive_potential,
        angular_velocity_body,
    ) else {
        return;
    };

    let origin = *history
        .origin_simulation_seconds
        .get_or_insert(sample.snapshot.simulation_time_seconds);
    history.elapsed_simulation_seconds = sample.snapshot.simulation_time_seconds - origin;
    history.last_request_id = Some(sample.snapshot.request_id);
    if history.samples.len() == JACOBI_HISTORY_CAPACITY {
        history.samples.pop_front();
    }
    let simulation_time_seconds = history.elapsed_simulation_seconds;
    history.samples.push_back(JacobiSample {
        simulation_time_seconds,
        jacobi_constant,
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

pub fn update_jacobi_chart_system(
    active_method: Res<ActiveGravityMethod>,
    history: Res<JacobiHistory>,
    curved_history: Res<CurvedArcResidualHistory>,
    curved_planner: Res<CurvedArcPlannerState>,
    simulation_acceleration: Res<SimulationAcceleration>,
    mut segments: Query<
        (&JacobiChartSegment, &mut Node, &mut UiTransform),
        Without<JacobiLatestPoint>,
    >,
    mut latest_point: Query<&mut Node, (With<JacobiLatestPoint>, Without<JacobiChartSegment>)>,
    mut labels: Query<
        (&JacobiChartLabel, &mut Text),
        (Without<JacobiChartTitle>, Without<JacobiChartAxisLabel>),
    >,
    mut titles: Query<
        &mut Text,
        (
            With<JacobiChartTitle>,
            Without<JacobiChartAxisLabel>,
            Without<JacobiChartLabel>,
        ),
    >,
    mut axis_labels: Query<
        &mut Text,
        (
            With<JacobiChartAxisLabel>,
            Without<JacobiChartTitle>,
            Without<JacobiChartLabel>,
        ),
    >,
) {
    if *active_method == ActiveGravityMethod::CurvedArcEq106 {
        update_curved_arc_chart(
            &curved_history,
            &curved_planner,
            &mut segments,
            &mut latest_point,
            &mut labels,
            &mut titles,
            &mut axis_labels,
        );
        return;
    }

    if !history.is_changed() && !simulation_acceleration.is_changed() && !active_method.is_changed()
    {
        return;
    }

    for mut title in titles.iter_mut() {
        **title = "Rotating-frame Jacobi constant".to_owned();
    }
    for mut axis in axis_labels.iter_mut() {
        **axis = "C_J (m^2/s^2)".to_owned();
    }

    let samples: Vec<JacobiSample> = history.samples.iter().copied().collect();
    if samples.is_empty() {
        for (_, mut node, _) in segments.iter_mut() {
            node.display = Display::None;
        }
        if let Some(mut node) = latest_point.iter_mut().next() {
            node.display = Display::None;
        }
        for (label, mut text) in labels.iter_mut() {
            **text = match label {
                JacobiChartLabel::Current => "C_J = -- m^2/s^2".to_owned(),
                JacobiChartLabel::RelativeDrift => "dC/|C0| = --".to_owned(),
                JacobiChartLabel::TimeStart => "0 s".to_owned(),
                _ => "--".to_owned(),
            };
        }
        return;
    }

    let raw_minimum = samples
        .iter()
        .map(|sample| sample.jacobi_constant)
        .fold(f64::INFINITY, f64::min);
    let raw_maximum = samples
        .iter()
        .map(|sample| sample.jacobi_constant)
        .fold(f64::NEG_INFINITY, f64::max);
    let raw_span = raw_maximum - raw_minimum;
    let padding = if raw_span > f64::EPSILON {
        raw_span * 0.08
    } else {
        (raw_maximum.abs() * 0.02).max(1.0e-6)
    };
    let minimum = raw_minimum - padding;
    let maximum = raw_maximum + padding;
    let jacobi_span = (maximum - minimum).max(f64::EPSILON);

    let (time_start, time_end) = jacobi_time_bounds(&samples, *simulation_acceleration);
    let time_span = (time_end - time_start).max(f64::EPSILON);

    let point_for = |sample: JacobiSample| {
        let x = ((sample.simulation_time_seconds - time_start) / time_span).clamp(0.0, 1.0) as f32
            * CHART_WIDTH;
        let y = (1.0 - ((sample.jacobi_constant - minimum) / jacobi_span).clamp(0.0, 1.0)) as f32
            * CHART_HEIGHT;
        Vec2::new(x, y)
    };

    for (segment, mut node, mut transform) in segments.iter_mut() {
        let Some((from, to)) = samples
            .get(segment.0)
            .zip(samples.get(segment.0 + 1))
            .map(|(from, to)| (point_for(*from), point_for(*to)))
        else {
            node.display = Display::None;
            continue;
        };

        let delta = to - from;
        let length = delta.length();
        let midpoint = (from + to) * 0.5;
        node.display = Display::Flex;
        node.left = px(midpoint.x - length * 0.5);
        node.top = px(midpoint.y - 1.0);
        node.width = px(length.max(0.5));
        transform.rotation = Rot2::radians(delta.y.atan2(delta.x));
    }

    if let Some(last) = samples.last()
        && let Some(mut node) = latest_point.iter_mut().next()
    {
        let point = point_for(*last);
        node.display = Display::Flex;
        node.left = px(point.x - 3.5);
        node.top = px(point.y - 3.5);
    }

    for (label, mut text) in labels.iter_mut() {
        let first = samples.first().unwrap().jacobi_constant;
        let latest = samples.last().unwrap().jacobi_constant;
        **text = match label {
            JacobiChartLabel::Current => {
                format!(
                    "C_J = {:.6e} m^2/s^2",
                    samples.last().unwrap().jacobi_constant
                )
            }
            JacobiChartLabel::RelativeDrift => {
                let denominator = first.abs().max(1.0e-12);
                format!("dC/|C0| = {:+.3e}%", 100.0 * (latest - first) / denominator)
            }
            JacobiChartLabel::Minimum => format!("{minimum:.3e}"),
            JacobiChartLabel::Maximum => format!("{maximum:.3e}"),
            JacobiChartLabel::TimeStart => format_axis_time(time_start),
            JacobiChartLabel::TimeEnd => format_axis_time(time_end),
        };
    }
}

fn update_curved_arc_chart(
    history: &CurvedArcResidualHistory,
    planner: &CurvedArcPlannerState,
    segments: &mut Query<
        (&JacobiChartSegment, &mut Node, &mut UiTransform),
        Without<JacobiLatestPoint>,
    >,
    latest_point: &mut Query<&mut Node, (With<JacobiLatestPoint>, Without<JacobiChartSegment>)>,
    labels: &mut Query<
        (&JacobiChartLabel, &mut Text),
        (Without<JacobiChartTitle>, Without<JacobiChartAxisLabel>),
    >,
    titles: &mut Query<
        &mut Text,
        (
            With<JacobiChartTitle>,
            Without<JacobiChartAxisLabel>,
            Without<JacobiChartLabel>,
        ),
    >,
    axis_labels: &mut Query<
        &mut Text,
        (
            With<JacobiChartAxisLabel>,
            Without<JacobiChartTitle>,
            Without<JacobiChartLabel>,
        ),
    >,
) {
    for mut title in titles.iter_mut() {
        **title = "Eq.106 curved-path residual".to_owned();
    }
    for mut axis in axis_labels.iter_mut() {
        **axis = "|r_dual| (m^2/s^2)".to_owned();
    }

    let samples: Vec<_> = history.samples.iter().copied().collect();
    if samples.is_empty() {
        for (_, mut node, _) in segments.iter_mut() {
            node.display = Display::None;
        }
        if let Some(mut node) = latest_point.iter_mut().next() {
            node.display = Display::None;
        }
        for (label, mut text) in labels.iter_mut() {
            **text = match label {
                JacobiChartLabel::Current => "|r_dual| = --".to_owned(),
                JacobiChartLabel::RelativeDrift => format!(
                    "{} | segments: {} | closures: {}/10",
                    planner.mode.as_str(),
                    planner.segments.len(),
                    planner.stable_closures,
                ),
                JacobiChartLabel::TimeStart => "0 s".to_owned(),
                _ => "--".to_owned(),
            };
        }
        return;
    }

    let residual_value = |sample: crate::systems::curved_arc::CurvedArcResidualSample| {
        sample.dual_residual.unwrap_or(0.0).abs()
    };
    let raw_minimum = samples
        .iter()
        .copied()
        .map(residual_value)
        .fold(f64::INFINITY, f64::min);
    let raw_maximum = samples
        .iter()
        .copied()
        .map(residual_value)
        .fold(f64::NEG_INFINITY, f64::max);
    let maximum = if raw_maximum > f64::EPSILON {
        raw_maximum * 1.08
    } else {
        1.0e-9
    };
    let minimum = 0.0_f64.min(raw_minimum);
    let span = (maximum - minimum).max(f64::EPSILON);
    let time_start = samples
        .first()
        .map_or(0.0, |sample| sample.simulation_time_seconds);
    let time_end = samples
        .last()
        .map_or(time_start + 1.0, |sample| sample.simulation_time_seconds)
        .max(time_start + f64::EPSILON);
    let time_span = time_end - time_start;
    let point_for = |sample: crate::systems::curved_arc::CurvedArcResidualSample| {
        let x = ((sample.simulation_time_seconds - time_start) / time_span).clamp(0.0, 1.0) as f32
            * CHART_WIDTH;
        let y = (1.0 - ((residual_value(sample) - minimum) / span).clamp(0.0, 1.0)) as f32
            * CHART_HEIGHT;
        Vec2::new(x, y)
    };

    for (segment, mut node, mut transform) in segments.iter_mut() {
        let Some((from, to)) = samples
            .get(segment.0)
            .zip(samples.get(segment.0 + 1))
            .map(|(from, to)| (point_for(*from), point_for(*to)))
        else {
            node.display = Display::None;
            continue;
        };
        let delta = to - from;
        let length = delta.length();
        let midpoint = (from + to) * 0.5;
        node.display = Display::Flex;
        node.left = px(midpoint.x - length * 0.5);
        node.top = px(midpoint.y - 1.0);
        node.width = px(length.max(0.5));
        transform.rotation = Rot2::radians(delta.y.atan2(delta.x));
    }

    if let Some(last) = samples.last()
        && let Some(mut node) = latest_point.iter_mut().next()
    {
        let point = point_for(*last);
        node.display = Display::Flex;
        node.left = px(point.x - 3.5);
        node.top = px(point.y - 3.5);
    }

    for (label, mut text) in labels.iter_mut() {
        let latest = residual_value(*samples.last().unwrap());
        **text = match label {
            JacobiChartLabel::Current => {
                format!("|r_dual| = {latest:.3e} m^2/s^2")
            }
            JacobiChartLabel::RelativeDrift => {
                let order = samples
                    .last()
                    .map_or(planner.taylor_order, |sample| sample.taylor_order);
                let epsilon = samples.last().map_or(0.0, |sample| sample.epsilon_max);
                let remainder = planner
                    .active_segment
                    .as_ref()
                    .map_or(f64::INFINITY, |segment| segment.remainder_bound);
                format!(
                    "{} A{} e={:.2} R={:.1e} seg={} c={}/10",
                    planner.mode.short_str(),
                    order,
                    epsilon,
                    remainder,
                    planner.segments.len(),
                    planner.stable_closures,
                )
            }
            JacobiChartLabel::Minimum => format!("{minimum:.3e}"),
            JacobiChartLabel::Maximum => format!("{maximum:.3e}"),
            JacobiChartLabel::TimeStart => format_axis_time(time_start),
            JacobiChartLabel::TimeEnd => format_axis_time(time_end),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn co_rotating_stationary_probe_has_expected_jacobi_constant() {
        let position = Vec3::new(1000.0, 0.0, 0.0);
        let omega = Vec3::Y * 2.0e-4;
        let inertial_velocity = omega.cross(position);
        let potential = 0.03;
        let actual =
            rotating_frame_jacobi_constant(position, inertial_velocity, potential, omega).unwrap();
        let expected = 2.0 * potential as f64 + omega.cross(position).length_squared() as f64;
        assert!((actual - expected).abs() < 1.0e-8);
    }

    #[test]
    fn invalid_jacobi_input_is_rejected() {
        assert!(rotating_frame_jacobi_constant(Vec3::NAN, Vec3::ZERO, 1.0, Vec3::Y).is_none());
    }

    #[test]
    fn incomplete_chart_window_scales_with_simulation_acceleration() {
        let samples = [JacobiSample {
            simulation_time_seconds: 8.0,
            jacobi_constant: 1.0,
        }];
        let (_, time_end_1x) = jacobi_time_bounds(&samples, SimulationAcceleration(1));
        let (_, time_end_8x) = jacobi_time_bounds(&samples, SimulationAcceleration(8));
        assert!((time_end_8x - 8.0 * time_end_1x).abs() < f64::EPSILON);
    }
}
