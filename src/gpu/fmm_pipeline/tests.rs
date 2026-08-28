#[cfg(test)]
mod tests {
    use super::*;

    fn direct_field(observer: DVec3, sources: &[(DVec3, f64)]) -> (DVec3, f64) {
        sources.iter().fold(
            (DVec3::ZERO, 0.0),
            |(acceleration, potential), &(position, mass)| {
                let d = position - observer;
                let r = d.length();
                (acceleration + mass * d / r.powi(3), potential + mass / r)
            },
        )
    }

    fn m2l_is_acceptable(source_half: f64, target_half: f64, distance: f64, theta: f64) -> bool {
        let expansion_radius = 3.0_f64.sqrt() * (source_half + target_half);
        distance > expansion_radius && expansion_radius / distance < theta
    }

    #[test]
    fn m2l_rejects_a_large_target_box_even_when_the_source_is_small() {
        // The previous source-only test accepted this coarse target box:
        // source_half / distance = 0.037. Its actual L2L radius ratio is 0.63,
        // so an order-two local expansion there is not convergent enough.
        assert!(!m2l_is_acceptable(232.0, 3_720.0, 10_800.0, THETA as f64));
        assert!(m2l_is_acceptable(14.5, 100.0, 2_500.0, THETA as f64));
    }

    #[test]
    fn quadrupole_far_field_matches_direct_sources() {
        let sources = [
            (DVec3::new(-12.0, 3.0, 1.0), 2.0),
            (DVec3::new(8.0, -4.0, 5.0), 3.0),
            (DVec3::new(2.0, 7.0, -6.0), 4.0),
            (DVec3::new(-1.0, -5.0, 2.0), 5.0),
        ];
        let mut moment = MomentAccumulator::default();
        for &(position, mass) in &sources {
            moment.add(position, mass);
        }
        let com = moment.first / moment.mass;
        let [x, y, z] = com.to_array();
        let central = [
            moment.second[0] - moment.mass * x * x,
            moment.second[1] - moment.mass * x * y,
            moment.second[2] - moment.mass * x * z,
            moment.second[3] - moment.mass * y * y,
            moment.second[4] - moment.mass * y * z,
            moment.second[5] - moment.mass * z * z,
        ];
        let trace = central[0] + central[3] + central[5];
        let q = [
            [3.0 * central[0] - trace, 3.0 * central[1], 3.0 * central[2]],
            [3.0 * central[1], 3.0 * central[3] - trace, 3.0 * central[4]],
            [3.0 * central[2], 3.0 * central[4], 3.0 * central[5] - trace],
        ];
        let observer = DVec3::new(1_200.0, -800.0, 600.0);
        let d = com - observer;
        let qd = DVec3::new(
            q[0][0] * d.x + q[0][1] * d.y + q[0][2] * d.z,
            q[1][0] * d.x + q[1][1] * d.y + q[1][2] * d.z,
            q[2][0] * d.x + q[2][1] * d.y + q[2][2] * d.z,
        );
        let r2 = d.length_squared();
        let r = r2.sqrt();
        let scalar = d.dot(qd);
        let multipole_acceleration =
            moment.mass * d / r.powi(3) - qd / r.powi(5) + 2.5 * scalar * d / r.powi(7);
        let multipole_potential = moment.mass / r + 0.5 * scalar / r.powi(5);
        let (direct_acceleration, direct_potential) = direct_field(observer, &sources);

        let acceleration_error =
            (multipole_acceleration - direct_acceleration).length() / direct_acceleration.length();
        let potential_error = (multipole_potential - direct_potential).abs() / direct_potential;
        assert!(acceleration_error < 1.0e-6, "{acceleration_error:.3e}");
        assert!(potential_error < 1.0e-7, "{potential_error:.3e}");
    }

    #[test]
    fn local_m2l_expansion_reuses_analytic_field_and_jacobian() {
        let sources = [
            (DVec3::new(-12.0, 3.0, 1.0), 2.0),
            (DVec3::new(8.0, -4.0, 5.0), 3.0),
            (DVec3::new(2.0, 7.0, -6.0), 4.0),
        ];
        let mut moment = MomentAccumulator::default();
        for &(position, mass) in &sources {
            moment.add(position, mass);
        }
        let center = DVec3::new(1_500.0, -900.0, 700.0);
        let delta = DVec3::new(0.05, -0.03, 0.02);
        let mut local = LocalExpansion::default();
        accumulate_multipole(moment, center, &mut local).unwrap();
        let translated = local.acceleration + local.jacobian * delta;
        let (direct, _) = direct_field(center + delta, &sources);
        let relative = (translated - direct).length() / direct.length();
        assert!(relative < 2.0e-6, "local translation error {relative:.3e}");
        assert!((local.jacobian - local.jacobian.transpose()).abs_diff_eq(bevy::math::DMat3::ZERO, 1.0e-12));
    }
}
