pub fn simulated_annealing_system(
    mut inversion: ResMut<TrajectoryInversionState>,
    mut performance: ResMut<Eq106PerformanceMetrics>,
) {
    let Some(mut job) = inversion.annealing.take() else {
        return;
    };
    let iteration_started = Instant::now();
    let starting_iteration = job.iteration;
    let voxel_count = job.voxels.len();
    let total_volume = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mean_density = (job.current_mass as f32 / total_volume).max(f32::MIN_POSITIVE);
    let maximum_radius = job
        .voxels
        .iter()
        .map(|voxel| voxel.center.length())
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    let mean_normalized_radius = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume * voxel.center.length() / maximum_radius)
        .sum::<f32>()
        / total_volume;
    let mean_squared_radius = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume * (voxel.center.length() / maximum_radius).powi(2))
        .sum::<f32>()
        / total_volume;
    let (linear_variance, quadratic_covariance) =
        job.voxels
            .iter()
            .fold((0.0_f32, 0.0_f32), |(variance, covariance), voxel| {
                let q = voxel.center.length() / maximum_radius;
                let linear = q - mean_normalized_radius;
                let quadratic = q * q - mean_squared_radius;
                (
                    variance + voxel.volume * linear * linear,
                    covariance + voxel.volume * linear * quadratic,
                )
            });
    let quadratic_projection = quadratic_covariance / linear_variance.max(f32::MIN_POSITIVE);
    let mut proposal_deltas = vec![0.0_f32; voxel_count];
    let homogeneous_werner = job.method == ActiveGravityMethod::HomogeneousWerner;
    for _ in 0..ITERATIONS_PER_FRAME {
        if job.iteration >= job.iterations || voxel_count == 0 {
            break;
        }
        if homogeneous_werner {
            // The Werner inverse is not a free 56-voxel density inversion.
            // Its admissible model is one homogeneous density, already fixed
            // by total mass and volume; accepting voxel-wise proposals would
            // turn it into a different, non-Werner inverse problem.
            job.iteration = job.iterations;
            break;
        }
        let progress = job.iteration as f64 / job.iterations as f64;
        let temperature = 0.035 * (1.0 - progress).powi(2) + 2.0e-6;
        let log_width = 0.65 * (1.0 - progress) + 0.008;
        proposal_deltas.fill(0.0);

        let proposal_kind = next_random(&mut job.rng);
        if proposal_kind < 0.62 {
            // Low-frequency mass-conserving modes efficiently discover the
            // smooth radial component without restricting the final model to
            // radial symmetry.
            let use_curvature = next_random(&mut job.rng) < 0.38;
            let signed_width = (next_random(&mut job.rng) * 2.0 - 1.0) * log_width * 0.28;
            for (delta, voxel) in proposal_deltas.iter_mut().zip(&job.voxels) {
                let q = voxel.center.length() / maximum_radius;
                let linear = q - mean_normalized_radius;
                let radial_mode = if use_curvature {
                    q * q - mean_squared_radius - quadratic_projection * linear
                } else {
                    linear
                };
                *delta = mean_density * signed_width as f32 * radial_mode;
            }
        } else {
            // The remaining proposals retain the full three-dimensional voxel
            // freedom. Moving equal mass along a real adjacency edge preserves
            // total mass exactly and lets the dense interpolated trajectory
            // resolve supported lateral structure without arbitrary distant
            // swaps.
            let edge = ((next_random(&mut job.rng) * job.neighbours.len() as f64) as usize)
                .min(job.neighbours.len().saturating_sub(1));
            let (first, second) = if let Some(pair) = job.neighbours.get(edge) {
                *pair
            } else {
                (0, voxel_count.saturating_sub(1))
            };
            if first != second {
                let signed_width = (next_random(&mut job.rng) * 2.0 - 1.0) * log_width * 0.12;
                proposal_deltas[first] = mean_density * signed_width as f32;
                proposal_deltas[second] =
                    -proposal_deltas[first] * job.voxels[first].volume / job.voxels[second].volume;
            }
        }

        if proposal_deltas.iter().enumerate().any(|(index, delta)| {
            let proposed = job.current_densities[index] + *delta;
            let baseline = job.voxels[index].baseline_density.max(f32::MIN_POSITIVE);
            proposed < 0.02 * baseline || proposed > 8.0 * baseline
        }) {
            job.iteration += 1;
            continue;
        }

        for (density, delta) in job.current_densities.iter_mut().zip(&proposal_deltas) {
            *density += *delta;
        }
        for observation in 0..job.predicted_accelerations.len() {
            for (voxel, delta) in proposal_deltas.iter().enumerate() {
                job.predicted_accelerations[observation] +=
                    job.sensitivities[observation * voxel_count + voxel] * *delta;
            }
        }
        let proposal_objective = objective(&job);
        let delta = proposal_objective - job.current_objective;
        let accept = delta <= 0.0
            || next_random(&mut job.rng) < (-delta / temperature).exp().clamp(0.0, 1.0);
        if accept {
            job.current_objective = proposal_objective;
            if proposal_objective < job.best_objective {
                job.best_objective = proposal_objective;
                job.best_densities.clone_from(&job.current_densities);
            }
        } else {
            for (density, delta) in job.current_densities.iter_mut().zip(&proposal_deltas) {
                *density -= *delta;
            }
            for observation in 0..job.predicted_accelerations.len() {
                for (voxel, delta) in proposal_deltas.iter().enumerate() {
                    job.predicted_accelerations[observation] -=
                        job.sensitivities[observation * voxel_count + voxel] * *delta;
                }
            }
        }
        job.iteration += 1;
    }

    let completed_iterations = job.iteration.saturating_sub(starting_iteration).max(1);
    performance.full_inversion_iteration_ms =
        Some(iteration_started.elapsed().as_secs_f64() * 1.0e3 / completed_iterations as f64);

    if job.iteration < job.iterations {
        inversion.displayed_density = Some(density_result_from_job(
            &job,
            &job.best_densities,
            job.best_objective,
        ));
        inversion.annealing = Some(job);
        return;
    }
    for (voxel, density) in job.voxels.iter_mut().zip(&job.best_densities) {
        voxel.density = *density;
    }
    let completed_method = job.method;
    let result = density_result_from_job(&job, &job.best_densities, job.best_objective);
    let index = completed_method.performance_index();
    inversion.results[index] = Some(result.clone());
    // The result shown in the central section is the annealer's prediction for
    // the one method that was selected when the button was pressed.
    inversion.displayed_density = Some(result);
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inversion_seed_depends_only_on_frozen_data_and_source() {
        let first = inversion_rng_seed(0x1234, 0x5678);
        assert_eq!(first, inversion_rng_seed(0x1234, 0x5678));
        assert_ne!(first, inversion_rng_seed(0x1235, 0x5678));
        assert_ne!(first, inversion_rng_seed(0x1234, 0x5679));
    }

    #[test]
    fn comparison_batch_preserves_the_mmfft_discrete_green_scale() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes, count: 2 };
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::new(1000.0, 1200.0, 100.0),
                velocity: Vec3::Y,
                simulation_time_seconds: 0.0,
                baseline_acceleration: Vec3::new(-1.0e-5, -2.0e-5, 0.0),
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: Vec3::new(1000.0, 1201.0, 100.0),
                velocity: Vec3::Y,
                simulation_time_seconds: 1.0,
                baseline_acceleration: Vec3::new(-1.0e-5, -2.0e-5, 0.0),
                body_rotation: Quat::IDENTITY,
            },
        ];
        let sensitivities_for = |method| {
            let (voxels, _) = build_density_voxels(&source, method).unwrap();
            build_observations_and_sensitivities(&knots, &voxels, method, true)
                .unwrap()
                .1
        };
        let direct = sensitivities_for(ActiveGravityMethod::RadialAnalytic);
        assert_eq!(
            direct,
            sensitivities_for(ActiveGravityMethod::CurvedArcEq106)
        );
        assert_eq!(direct, sensitivities_for(ActiveGravityMethod::Fmm));
        assert_ne!(
            direct,
            sensitivities_for(ActiveGravityMethod::MmfftCompressed)
        );
    }

    fn source_record(direction: Vec3, density: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for value in [
            direction.x,
            direction.y,
            direction.z,
            1.0, // solid angle
            0.0, // inner radius
            100.0,
            density,
            0.0, // record padding
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn voxel_prior_does_not_copy_forward_density() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes, count: 2 };

        let (voxels, _) = build_density_voxels(&source, ActiveGravityMethod::HomogeneousWerner)
            .expect("valid voxel source");

        assert_eq!(voxels.len(), 2);
        assert_eq!(voxels[0].density, voxels[1].density);
        assert_eq!(voxels[0].baseline_density, voxels[1].baseline_density);
    }

    #[test]
    fn werner_reference_field_is_uniform_and_scores_one_hundred_percent() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes, count: 2 };

        let (voxels, _) = build_density_voxels(&source, ActiveGravityMethod::HomogeneousWerner)
            .expect("valid voxel source");

        assert!(voxels.iter().all(|voxel| {
            voxel.reference_density == voxel.baseline_density
                && voxel.density == voxel.reference_density
        }));
        assert_eq!(density_model_deviation(&voxels), 0.0);
    }

    #[test]
    fn logarithmic_reference_is_separate_from_uniform_annealing_prior() {
        let mut bytes = source_record(Vec3::X, 100.0);
        let mut outer = source_record(Vec3::NEG_X, 10_000.0);
        outer[20..24].copy_from_slice(&200.0_f32.to_le_bytes());
        bytes.extend(outer);
        let source = RadialGravitySource { bytes, count: 2 };

        let (voxels, _) =
            build_density_voxels(&source, ActiveGravityMethod::RadialAnalytic).unwrap();

        let mut ordered = voxels.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.center.length().total_cmp(&b.center.length()));
        assert_eq!(ordered[0].density, ordered[1].density);
        assert_eq!(ordered[0].density, ordered[0].baseline_density);
        assert!(ordered[0].reference_density < ordered[1].reference_density);
        assert!(density_model_deviation(&voxels) > 0.0);
    }

    #[test]
    fn model_deviation_is_volume_weighted_relative_rmse() {
        let voxels = [
            InvertedDensityVoxel {
                center: Vec3::ZERO,
                volume: 1.0,
                density: 2.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [0, 0, 0],
            },
            InvertedDensityVoxel {
                center: Vec3::X,
                volume: 3.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [1, 0, 0],
            },
        ];
        assert!((density_model_deviation(&voxels) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn spatial_sensitivity_distinguishes_voxel_locations() {
        let start = TrajectoryInversionKnot {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            simulation_time_seconds: 0.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let end = TrajectoryInversionKnot {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            simulation_time_seconds: 1.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let voxels = [
            InvertedDensityVoxel {
                center: Vec3::new(1.0, 0.0, 0.0),
                volume: 1.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [0, 0, 0],
            },
            InvertedDensityVoxel {
                center: Vec3::new(-1.0, 0.0, 0.0),
                volume: 1.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [1, 0, 0],
            },
        ];
        let (_, sensitivities, _) = build_observations_and_sensitivities(
            &[start, end],
            &voxels,
            ActiveGravityMethod::RadialAnalytic,
            false,
        )
        .unwrap();
        assert_ne!(sensitivities[0], sensitivities[1]);
        assert!(sensitivities[0].length() > sensitivities[1].length());
    }

    #[test]
    fn quintic_track_is_densely_sampled_and_does_not_reuse_captured_acceleration() {
        let acceleration = Vec3::new(2.0, -1.0, 0.5);
        let initial_velocity = Vec3::new(1.0, 2.0, 3.0);
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::ZERO,
                velocity: initial_velocity,
                simulation_time_seconds: 0.0,
                baseline_acceleration: Vec3::splat(99.0),
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: initial_velocity + 0.5 * acceleration,
                velocity: initial_velocity + acceleration,
                simulation_time_seconds: 1.0,
                baseline_acceleration: Vec3::splat(-99.0),
                body_rotation: Quat::IDENTITY,
            },
        ];
        let voxels = [InvertedDensityVoxel {
            center: Vec3::ZERO,
            volume: 1.0,
            density: 1.0,
            baseline_density: 1.0,
            reference_density: 2.0,
            grid: [0, 0, 0],
        }];

        let (observations, sensitivities, predictions) = build_observations_and_sensitivities(
            &knots,
            &voxels,
            ActiveGravityMethod::RadialAnalytic,
            false,
        )
        .unwrap();

        assert_eq!(observations.len(), TRAJECTORY_SAMPLES_PER_SEGMENT + 1);
        assert_eq!(sensitivities.len(), observations.len());
        assert_eq!(predictions.len(), observations.len());
        assert!(observations.iter().all(|sample| {
            (*sample - acceleration).length() <= 2.0e-5
                && *sample != knots[0].baseline_acceleration
                && *sample != knots[1].baseline_acceleration
        }));
    }

    #[test]
    fn unedited_capture_uses_snapshot_aligned_acceleration() {
        let captured = Vec3::new(-2.0e-5, 3.0e-5, 1.0e-5);
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
                simulation_time_seconds: 0.0,
                baseline_acceleration: captured,
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: captured * 0.5,
                velocity: captured,
                simulation_time_seconds: 1.0,
                baseline_acceleration: captured,
                body_rotation: Quat::IDENTITY,
            },
        ];
        let voxels = [InvertedDensityVoxel {
            center: Vec3::ZERO,
            volume: 1.0,
            density: 1.0,
            baseline_density: 1.0,
            reference_density: 1.0,
            grid: [0, 0, 0],
        }];

        let (observations, _, _) = build_observations_and_sensitivities(
            &knots,
            &voxels,
            ActiveGravityMethod::RadialAnalytic,
            true,
        )
        .unwrap();

        assert!(
            observations
                .iter()
                .all(|observation| (*observation - captured).length() < 1.0e-7)
        );
    }
}
