#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_ifft_round_trip() {
        let padded_size = 2 * LEVEL_GRID_SIZES[1];
        let mut values = (0..padded_size)
            .map(|index| Complex64 {
                re: index as f64,
                im: 0.0,
            })
            .collect::<Vec<_>>();
        let expected = values.clone();
        fft_1d(&mut values, false);
        fft_1d(&mut values, true);
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual.re - expected.re).abs() < 1.0e-9);
        }
    }

    #[test]
    fn zero_padding_prevents_circular_aliasing() {
        for grid_size in LEVEL_GRID_SIZES {
            assert!((2 * grid_size).is_power_of_two());
        }
    }

    #[test]
    fn potential_grid_is_packed_to_two_bytes_per_sample_with_bounded_error() {
        let field = (0..257)
            .map(|index| [0.0, 0.0, 0.0, (index as f32 - 128.0) * 0.001_25])
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        let scale = append_compressed_potential_level(&mut bytes, &field);
        assert_eq!(bytes.len(), field.len().div_ceil(2) * 4);
        for (index, sample) in field.iter().enumerate() {
            let pair = u32::from_le_bytes(
                bytes[index / 2 * 4..index / 2 * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let bits = if index.is_multiple_of(2) {
                pair as u16
            } else {
                (pair >> 16) as u16
            };
            let decoded = half::f16::from_bits(bits).to_f32() * scale;
            assert!((decoded - sample[3]).abs() <= scale * 5.0e-4);
        }
    }

    #[test]
    fn three_dimensional_convolution_matches_exterior_direct_field() {
        let sources = [
            (DVec3::new(-80.0, 20.0, 35.0), 2.0e11),
            (DVec3::new(60.0, -45.0, -10.0), 3.0e11),
            (DVec3::new(10.0, 70.0, -55.0), 5.0e11),
        ];
        let grid_size = LEVEL_GRID_SIZES[0];
        let grid = build_level(&sources, LEVEL_HALF_EXTENTS[0], grid_size);
        let observer = DVec3::new(700.0, 100.0, -100.0);
        let spacing = 2.0 * LEVEL_HALF_EXTENTS[0] / grid_size as f64;
        let coordinate =
            (observer + DVec3::splat(LEVEL_HALF_EXTENTS[0])) / spacing - DVec3::splat(0.5);
        let center = coordinate.floor().as_uvec3();
        let fraction = coordinate - center.as_dvec3();
        let base = center - UVec3::ONE;
        let weights = |t: f64| {
            let t2 = t * t;
            let t3 = t2 * t;
            [
                -0.5 * t + t2 - 0.5 * t3,
                1.0 - 2.5 * t2 + 1.5 * t3,
                0.5 * t + 2.0 * t2 - 1.5 * t3,
                -0.5 * t2 + 0.5 * t3,
            ]
        };
        let derivatives = |t: f64| {
            let t2 = t * t;
            [
                -0.5 + 2.0 * t - 1.5 * t2,
                -5.0 * t + 4.5 * t2,
                0.5 + 4.0 * t - 4.5 * t2,
                -t + 1.5 * t2,
            ]
        };
        let wx = weights(fraction.x);
        let wy = weights(fraction.y);
        let wz = weights(fraction.z);
        let dxw = derivatives(fraction.x);
        let dyw = derivatives(fraction.y);
        let dzw = derivatives(fraction.z);
        let mut interpolated_potential = 0.0;
        let mut interpolated_acceleration = DVec3::ZERO;
        for dz in 0..4 {
            for dy in 0..4 {
                for dx in 0..4 {
                    let corner_potential = grid[grid_index(
                        base.x as usize + dx,
                        base.y as usize + dy,
                        base.z as usize + dz,
                        grid_size,
                    )][3] as f64;
                    interpolated_potential += wx[dx] * wy[dy] * wz[dz] * corner_potential;
                    interpolated_acceleration += corner_potential / spacing
                        * DVec3::new(
                            dxw[dx] * wy[dy] * wz[dz],
                            dyw[dy] * wx[dx] * wz[dz],
                            dzw[dz] * wx[dx] * wy[dy],
                        );
                }
            }
        }
        let (direct_acceleration, direct_potential) = sources.iter().fold(
            (DVec3::ZERO, 0.0),
            |(acceleration, potential), &(position, mass)| {
                let displacement = position - observer;
                let distance = displacement.length();
                (
                    acceleration + G as f64 * mass * displacement / distance.powi(3),
                    potential + G as f64 * mass / distance,
                )
            },
        );
        let acceleration_error = (interpolated_acceleration - direct_acceleration).length()
            / direct_acceleration.length();
        let potential_error = (interpolated_potential - direct_potential).abs() / direct_potential;
        assert!(
            acceleration_error < GRAVITY_BENCHMARK_RELATIVE_TOLERANCE as f64,
            "acceleration error {acceleration_error:.3e}"
        );
        assert!(
            potential_error < 0.02,
            "potential error {potential_error:.3e}"
        );
    }
}
