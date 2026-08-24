
fn taylor_remainder_bound(epsilon_max: f64, taylor_order: u32) -> Option<f64> {
    if !epsilon_max.is_finite() || epsilon_max >= 1.0 {
        return None;
    }
    let next_term = epsilon_max.powi((taylor_order + 1) as i32);
    let bound = next_term / (1.0 - epsilon_max).max(f64::EPSILON);
    bound.is_finite().then_some(bound)
}

fn taylor_gradient_remainder_bound(
    epsilon_max: f64,
    taylor_order: u32,
) -> Option<f64> {
    if !epsilon_max.is_finite() || epsilon_max >= 1.0 {
        return None;
    }
    let denominator = (1.0 - epsilon_max).max(f64::EPSILON).powi(2);
    let bound = f64::from(taylor_order + 1)
        * epsilon_max.powi(taylor_order as i32)
        / denominator;
    bound.is_finite().then_some(bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_segment_has_zero_offset() {
        let points = [
            Vec3::Y * 100.0,
            Vec3::Y * 100.0 + Vec3::X,
            Vec3::Y * 100.0 + Vec3::X * 2.0,
            Vec3::Y * 100.0 + Vec3::X * 3.0,
        ];
        let segment = evaluate_segment(&points, 0, points.len() - 1, 0.5);
        assert_eq!(segment.epsilon_max, 0.0);
    }

    #[test]
    fn interior_sphere_marks_segment_as_unusable() {
        let points = [Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::X * 3.0];
        let segment = evaluate_segment(&points, 0, points.len() - 1, 10.0);
        assert!(segment.distance_lower_bound <= 0.0);
        assert!(segment.epsilon_max.is_infinite());
    }

    #[test]
    fn taylor_order_and_dual_residual_follow_convergence_ratio() {
        assert_eq!(select_taylor_order(0.0), Some(1));
        assert!(select_taylor_order(0.1).is_some_and(|order| order >= 2));
        assert!(select_taylor_order(0.7).is_none());
        assert!(taylor_remainder_bound(0.2, 4).is_some_and(|value| value < 1.0e-3));
        assert!(
            taylor_gradient_remainder_bound(0.2, 4)
                .is_some_and(|value| value < TAYLOR_GRADIENT_REMAINDER_TARGET)
        );
        assert!(taylor_remainder_bound(1.0, 2).is_none());
    }

    #[test]
    fn curvature_bound_forces_tighter_segments() {
        let gentle = [
            Vec3::new(0.0, 1_000.0, 0.0),
            Vec3::new(10.0, 1_000.1, 0.0),
            Vec3::new(20.0, 1_000.4, 0.0),
            Vec3::new(30.0, 1_000.9, 0.0),
        ];
        let sharp = [
            Vec3::new(0.0, 1_000.0, 0.0),
            Vec3::new(10.0, 1_010.0, 0.0),
            Vec3::new(20.0, 1_000.0, 0.0),
            Vec3::new(30.0, 990.0, 0.0),
        ];
        let gentle = evaluate_segment(&gentle, 0, 3, 400.0);
        let sharp = evaluate_segment(&sharp, 0, 3, 400.0);
        assert!(sharp.maximum_curvature > gentle.maximum_curvature);
        assert!(sharp.epsilon_max > gentle.epsilon_max);
    }

    #[cfg(feature = "eq106-dual-certificate")]
    #[test]
    fn eq157_residual_compares_curve_work_with_independent_potential() {
        let mut history = CurvedArcResidualHistory::default();
        history.accumulate_curve_work(0.0, 1.0, Vec3::ZERO, Vec3::X, Vec3::X, Vec3::X);

        let sample = |request_id, time, potential| GravityFieldSample {
            snapshot: GravityRequestSnapshot {
                request_id,
                epoch: 0,
                simulation_time_seconds: time,
                body_position: Vec3::ZERO,
                ryugu_transform: Transform::IDENTITY,
                probe_position: Vec3::ZERO,
                probe_velocity: Vec3::ZERO,
            },
            predictive: false,
            body_acceleration: Vec3::ZERO,
            positive_potential: potential,
            #[cfg(feature = "eq106-dual-certificate")]
            independent_positive_potential: Some(potential),
            body_acceleration_jacobian: None,
            eq106_diagnostics: None,
        };

        assert_eq!(history.dual_residual_for(&sample(1, 0.0, 4.0)), Some(0.0));
        assert!(
            history
                .dual_residual_for(&sample(2, 1.0, 5.0))
                .is_some_and(|residual| residual.abs() <= f64::EPSILON)
        );
        // A repeated GPU request must not be plotted twice.
        assert_eq!(history.dual_residual_for(&sample(2, 1.0, 5.0)), None);
    }
}

    #[test]
    fn thirty_two_bin_density_fourier_modes_reconstruct_without_alias_loss() {
        let density = std::array::from_fn::<_, EQ106_AZIMUTH_BINS, _>(|index| {
            let phi = std::f64::consts::TAU * index as f64 / EQ106_AZIMUTH_BINS as f64;
            3.0 + 0.7 * (5.0 * phi).cos() - 0.2 * (9.0 * phi).sin()
        });
        let coefficients = std::array::from_fn::<_, 17, _>(|mode| {
            density
                .iter()
                .enumerate()
                .fold([0.0, 0.0], |mut sum, (index, value)| {
                    let phi = std::f64::consts::TAU * index as f64 / EQ106_AZIMUTH_BINS as f64;
                    sum[0] += value * (-(mode as f64) * phi).cos();
                    sum[1] += value * (-(mode as f64) * phi).sin();
                    sum
                })
        });
        for (index, expected) in density.into_iter().enumerate() {
            let phi = std::f64::consts::TAU * index as f64 / EQ106_AZIMUTH_BINS as f64;
            let mut reconstructed = coefficients[0][0] + coefficients[16][0] * (16.0 * phi).cos();
            for (mode, coefficient) in coefficients.iter().enumerate().take(16).skip(1) {
                reconstructed += 2.0
                    * (coefficient[0] * (mode as f64 * phi).cos()
                        - coefficient[1] * (mode as f64 * phi).sin());
            }
            reconstructed /= EQ106_AZIMUTH_BINS as f64;
            assert!((reconstructed - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn runtime_frequency_certificate_accepts_external_line() {
        let line = Eq106ReferenceLine::new(DVec3::new(0.0, 0.0, 1_000.0), DVec3::X).unwrap();
        let sources = vec![Eq106PointSource {
            position: DVec3::ZERO,
            mass: RYUGU_MASS as f64,
        }];
        let result = certify_runtime_line(line, &sources, 1_000.0);
        assert!(result.is_ok(), "certificate failed: {result:?}");
    }
