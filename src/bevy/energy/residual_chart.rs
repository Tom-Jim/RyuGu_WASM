fn finite_chart_sample(simulation_time_seconds: f64, value: f64) -> bool {
    simulation_time_seconds.is_finite() && value.is_finite()
}

pub fn update_jacobi_chart_system(
    active_method: Res<ActiveGravityMethod>,
    history: Res<JacobiHistory>,
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

    let samples: Vec<JacobiSample> = history
        .samples
        .iter()
        .copied()
        .filter(|sample| {
            finite_chart_sample(sample.simulation_time_seconds, sample.jacobi_constant)
        })
        .collect();
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
        if !delta.is_finite() || !length.is_finite() {
            node.display = Display::None;
            continue;
        }
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

/// Displays the production Eq. (106) Taylor convergence residual. When the
/// optional Eq.157 certificate is enabled, its dual residual remains available
/// in the sample data but does not gate this always-on chart.
pub fn update_eq106_residual_chart_system(
    active_method: Res<ActiveGravityMethod>,
    history: Res<CurvedArcResidualHistory>,
    planner: Res<CurvedArcPlannerState>,
    propagation: Res<VolterraPropagationStatus>,
    mut roots: Query<&mut Visibility, With<Eq106ResidualChartRoot>>,
    mut segments: Query<
        (&Eq106ResidualChartSegment, &mut Node, &mut UiTransform),
        Without<Eq106ResidualLatestPoint>,
    >,
    mut latest_point: Query<
        &mut Node,
        (
            With<Eq106ResidualLatestPoint>,
            Without<Eq106ResidualChartSegment>,
        ),
    >,
    mut labels: Query<(&Eq106ResidualChartLabel, &mut Text)>,
) {
    let visible = *active_method == ActiveGravityMethod::CurvedArcEq106;
    for mut visibility in roots.iter_mut() {
        *visibility = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if !visible {
        return;
    }

    // A failed or not-yet-ready Eq.106 step may temporarily publish an
    // infinite convergence bound.  Such a diagnostic is meaningful to the
    // planner, but it is not a drawable chart coordinate: passing the
    // resulting NaN angle to `Rot2` panics in Bevy and aborts the WASM event
    // loop.  Keep the diagnostic history intact and omit only non-finite
    // points from the visualization.
    let samples: Vec<_> = history
        .samples
        .iter()
        .copied()
        .filter(|sample| {
            finite_chart_sample(sample.simulation_time_seconds, sample.epsilon_max)
        })
        .collect();
    if samples.is_empty() {
        for (_, mut node, _) in &mut segments {
            node.display = Display::None;
        }
        if let Some(mut node) = latest_point.iter_mut().next() {
            node.display = Display::None;
        }
        for (label, mut text) in &mut labels {
            **text = match label {
                Eq106ResidualChartLabel::Current => "|epsilon_106| = --".to_owned(),
                Eq106ResidualChartLabel::Status => planner.reject_status.clone().unwrap_or_else(|| {
                    format!(
                        "{} | A{} | segments: {}",
                        planner.mode.as_str(),
                        planner.taylor_order,
                        planner.segments.len(),
                    )
                }),
                Eq106ResidualChartLabel::TimeStart => "0 s".to_owned(),
                _ => "--".to_owned(),
            };
        }
        return;
    }

    let residual_value =
        |sample: crate::cpu::curved_arc::CurvedArcResidualSample| sample.epsilon_max.abs();
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
        // `time_start + f64::EPSILON` can round straight back to time_start
        // once the clock is above one second, producing 0/0 chart points.
        .max(time_start + 1.0e-6);
    let time_span = time_end - time_start;
    let point_for = |sample: crate::cpu::curved_arc::CurvedArcResidualSample| {
        let x = ((sample.simulation_time_seconds - time_start) / time_span).clamp(0.0, 1.0) as f32
            * CHART_WIDTH;
        let y = (1.0 - ((residual_value(sample) - minimum) / span).clamp(0.0, 1.0)) as f32
            * CHART_HEIGHT;
        Vec2::new(x, y)
    };

    for (segment, mut node, mut transform) in &mut segments {
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
        if !delta.is_finite() || !length.is_finite() {
            node.display = Display::None;
            continue;
        }
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

    for (label, mut text) in &mut labels {
        let latest = residual_value(*samples.last().unwrap());
        **text = match label {
            Eq106ResidualChartLabel::Current => {
                format!("|epsilon_106| = {latest:.3e}")
            }
            Eq106ResidualChartLabel::Status => {
                let order = samples
                    .last()
                    .map_or(planner.taylor_order, |sample| sample.taylor_order);
                let epsilon = samples.last().map_or(0.0, |sample| sample.epsilon_max);
                let remainder = planner
                    .active_segment
                    .as_ref()
                    .map_or(f64::INFINITY, |segment| segment.remainder_bound);
                if let Some(solve) = propagation.latest {
                    format!(
                        "{} A{} e={:.2} R={:.1e} V{}/{} r={:.1e} y={:.1e} seg={}",
                        planner.mode.short_str(),
                        order,
                        epsilon,
                        remainder,
                        solve.picard_iterations,
                        solve.endpoint_iterations,
                        solve.relative_residual,
                        solve.maximum_transverse_distance,
                        planner.segments.len(),
                    )
                } else {
                    format!(
                        "{} A{} e={:.2} R={:.1e} seg={}",
                        planner.mode.short_str(),
                        order,
                        epsilon,
                        remainder,
                        planner.segments.len(),
                    )
                }
            }
            Eq106ResidualChartLabel::Minimum => format!("{minimum:.3e}"),
            Eq106ResidualChartLabel::Maximum => format!("{maximum:.3e}"),
            Eq106ResidualChartLabel::TimeStart => format_axis_time(time_start),
            Eq106ResidualChartLabel::TimeEnd => format_axis_time(time_end),
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
    fn chart_samples_require_finite_time_and_value() {
        assert!(finite_chart_sample(1.0, 2.0));
        assert!(!finite_chart_sample(f64::NAN, 2.0));
        assert!(!finite_chart_sample(1.0, f64::INFINITY));
    }

    #[test]
    fn eq106_local_potential_gradient_is_the_integrated_local_force() {
        let hessian = Mat3::from_cols(
            Vec3::new(2.0e-7, 3.0e-8, -1.0e-8),
            Vec3::new(3.0e-8, -1.5e-7, 2.0e-8),
            Vec3::new(-1.0e-8, 2.0e-8, 0.5e-7),
        );
        let sample = GravityFieldSample {
            snapshot: GravityRequestSnapshot {
                request_id: 1,
                epoch: 1,
                simulation_time_seconds: 0.0,
                body_position: Vec3::new(10.0, -20.0, 5.0),
                ryugu_transform: Transform::IDENTITY,
                probe_position: Vec3::ZERO,
                probe_velocity: Vec3::ZERO,
            },
            predictive: false,
            body_acceleration: Vec3::new(1.0e-4, -2.0e-5, 3.0e-5),
            positive_potential: 0.04,
            #[cfg(feature = "eq106-dual-certificate")]
            independent_positive_potential: None,
            body_acceleration_jacobian: Some(hessian),
            eq106_diagnostics: None,
        };
        let position = sample.snapshot.body_position + Vec3::new(4.0, -3.0, 2.0);
        let expected =
            sample.body_acceleration + hessian * (position - sample.snapshot.body_position);
        let step = 0.01;
        let mut numerical = Vec3::ZERO;
        for axis in 0..3 {
            let direction = [Vec3::X, Vec3::Y, Vec3::Z][axis];
            let plus =
                eq106_local_positive_potential(&sample, position + step * direction).unwrap();
            let minus =
                eq106_local_positive_potential(&sample, position - step * direction).unwrap();
            numerical[axis] = (plus - minus) / (2.0 * step);
        }
        assert!((numerical - expected).length() < 3.0e-7);
    }

    #[test]
    fn incomplete_chart_window_scales_with_simulation_acceleration() {
        let samples = [JacobiSample {
            simulation_time_seconds: 8.0,
            jacobi_constant: 1.0,
            eq106_diagnostics: None,
        }];
        let (_, time_end_1x) = jacobi_time_bounds(&samples, SimulationAcceleration(1));
        let (_, time_end_8x) = jacobi_time_bounds(&samples, SimulationAcceleration(8));
        assert!((time_end_8x - 8.0 * time_end_1x).abs() < f64::EPSILON);
    }
}
