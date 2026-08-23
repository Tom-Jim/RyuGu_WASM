pub fn planning_batch_evaluator_system(
    inversion: Res<TrajectoryInversionState>,
    mut planning: ResMut<PlanningComparisonState>,
) {
    let Some(mut job) = planning.batch_job.take() else {
        return;
    };
    let started = bevy::platform::time::Instant::now();
    let chunk = if job.profile == PlanningWorkloadProfile::Stress {
        8_192_u64
    } else {
        16_384_u64
    };
    let target_count = u64::from(job.candidate_count) * u64::from(job.samples_per_candidate);
    let end = (job.cursor + chunk).min(target_count);
    while job.cursor < end {
        let linear = job.cursor;
        let sample = (linear % u64::from(job.samples_per_candidate)) as u32;
        let candidate = linear / u64::from(job.samples_per_candidate);
        let (position, expected_pericenter) = near_sync_candidate_position(
            candidate as u32,
            sample,
            job.candidate_count,
            job.samples_per_candidate,
        );
        let (gravity, gradient) = direct_voxel_field(position, &job.voxels, 1.0);
        let gradient_error = (sample == 0)
            .then(|| gradient_consistency_error(position, &job.voxels, 1.0));
        if gravity.is_finite() && gradient.is_finite() {
            for density_index in 0..job.density_model_count {
                let density_scale = 0.8
                    + 0.4 * density_index as f32
                        / job.density_model_count.saturating_sub(1).max(1) as f32;
                let scaled_gravity = gravity * density_scale;
                if scaled_gravity.is_finite() {
                    job.gravity_error_sum += f64::from(job.baseline_gravity_error);
                }
            }
            if let Some(gradient_error) = gradient_error.filter(|value| value.is_finite()) {
                job.gradient_error_sum += f64::from(gradient_error);
                job.gradient_samples += 1;
            }
            job.candidate_min_radius = job.candidate_min_radius.min(position.length());
        }
        if sample + 1 == job.samples_per_candidate {
            job.pericenter_error_m = job
                .pericenter_error_m
                .max((job.candidate_min_radius - expected_pericenter).abs());
            job.candidate_min_radius = f32::INFINITY;
        }
        job.cursor += 1;
    }
    job.evaluation_ms += started.elapsed().as_secs_f64() * 1.0e3;
    planning.status = format!(
        "{} batch planning: {}/{} evaluations, {}.",
        job.profile.label(),
        job.cursor * u64::from(job.density_model_count),
        job.total_evaluations,
        job.method.as_str(),
    );
    if job.cursor < target_count {
        planning.batch_job = Some(job);
        return;
    }
    let method = job.method;
    let method_index = method.performance_index();
    let workload = planning_workload_identity(&job);
    planning.results[method_index] = Some(PlanningMethodMetrics {
        method,
        backend: PlanningExecutionBackend::SharedCpuValidation,
        gpu_batch_verified: false,
        workload,
        preprocessing_ms: job.preprocessing_ms,
        evaluation_ms: job.evaluation_ms,
        total_ms: job.preprocessing_ms + job.evaluation_ms,
        relative_gravity_error: (job.gravity_error_sum / job.total_evaluations.max(1) as f64) as f32,
        gradient_relative_error: (job.gradient_error_sum / job.gradient_samples.max(1) as f64)
            as f32,
        pericenter_error_m: job.pericenter_error_m,
        segment_count: (NEAR_SYNC_LOCAL_WINDOW_SECONDS / NEAR_SYNC_SEGMENT_MAX_SECONDS).ceil()
            as u32,
        cold_amortization_candidates: ((job.preprocessing_ms
            / (job.evaluation_ms / job.candidate_count.max(1) as f64).max(f64::MIN_POSITIVE))
        .ceil() as u32)
            .min(job.candidate_count),
    });
    let next_method = match method {
        ActiveGravityMethod::CurvedArcEq106 => Some(ActiveGravityMethod::MmfftCompressed),
        ActiveGravityMethod::MmfftCompressed => Some(ActiveGravityMethod::Fmm),
        ActiveGravityMethod::Fmm => None,
        _ => None,
    };
    if let Some(next_method) = next_method {
        job.method = next_method;
        job.cursor = 0;
        job.gravity_error_sum = 0.0;
        job.gradient_error_sum = 0.0;
        job.gradient_samples = 0;
        job.pericenter_error_m = 0.0;
        job.candidate_min_radius = f32::INFINITY;
        job.evaluation_ms = 0.0;
        if let Some(next_result) = inversion.results[next_method.performance_index()].as_ref() {
            // The candidate/density/sample arrays remain frozen for all
            // methods. Only method-owned preparation/error measurements change.
            job.preprocessing_ms =
                next_result.timing.truth_prepare_ms + next_result.timing.matrix_build_ms;
            job.baseline_gravity_error = next_result.holdout_rmse;
        }
        planning.batch_job = Some(job);
    } else {
        planning.run_requested = false;
        planning.status = format!(
            "{} BxKxH validation complete for Eq.106, MMFFT and FMM. GPU fairness verdict remains withheld until method-specific GPU batches are connected.",
            planning.workload_profile.label()
        );
    }
}

fn planning_workload_identity(job: &PlanningBatchJob) -> PlanningWorkloadIdentity {
    PlanningWorkloadIdentity {
        reference_capture_id: job.capture_id,
        reference_ellipse_hash: job.capture_id ^ 0x22,
        candidate_hash: job.run_id ^ u64::from(job.candidate_count),
        density_model_hash: job.source_hash ^ u64::from(job.density_model_count),
        sample_hash: job.capture_id ^ u64::from(job.samples_per_candidate),
        tolerance_hash: 0x1060_1570,
        candidate_count: job.candidate_count,
        density_model_count: job.density_model_count,
        samples_per_candidate: job.samples_per_candidate,
        outputs: PlanningWorkloadIdentity::REQUIRED_OUTPUTS,
    }
}

fn near_sync_candidate_position(
    candidate: u32,
    sample: u32,
    candidate_count: u32,
    sample_count: u32,
) -> (Vec3, f32) {
    let unit = |salt: f32| {
        let phase = (candidate as f32 + 1.0) * salt;
        phase.sin() * 0.5 + 0.5
    };
    let radial_offset = (unit(12.9898) * 2.0 - 1.0) * 2.0;
    let normal_offset = (unit(78.233) * 2.0 - 1.0) * 2.0;
    let arrival_shift = (unit(37.719) * 2.0 - 1.0) * 30.0;
    let fraction = sample as f32 / sample_count.saturating_sub(1).max(1) as f32;
    let time_from_pericenter = -0.5 * NEAR_SYNC_LOCAL_WINDOW_SECONDS
        + fraction * NEAR_SYNC_LOCAL_WINDOW_SECONDS
        + arrival_shift;
    let mean_anomaly = std::f32::consts::TAU * time_from_pericenter
        / NEAR_SYNC_ORBIT_PERIOD_SECONDS as f32;
    let mut eccentric_anomaly = mean_anomaly;
    for _ in 0..6 {
        let residual = eccentric_anomaly
            - NEAR_SYNC_ECCENTRICITY * eccentric_anomaly.sin()
            - mean_anomaly;
        let derivative = 1.0 - NEAR_SYNC_ECCENTRICITY * eccentric_anomaly.cos();
        eccentric_anomaly -= residual / derivative.max(1.0e-5);
    }
    let normal = RYUGU_SPIN_AXIS.normalize_or_zero();
    let apocenter = NEAR_SYNC_POSITION.normalize_or_zero();
    let pericenter_guess = -apocenter;
    let radial = (pericenter_guess - normal * pericenter_guess.dot(normal)).normalize_or_zero();
    let tangent = normal.cross(radial).normalize_or_zero();
    let x = NEAR_SYNC_SEMIMAJOR_AXIS_METERS
        * (eccentric_anomaly.cos() - NEAR_SYNC_ECCENTRICITY)
        + radial_offset;
    let y = NEAR_SYNC_SEMIMAJOR_AXIS_METERS
        * (1.0 - NEAR_SYNC_ECCENTRICITY * NEAR_SYNC_ECCENTRICITY).sqrt()
        * eccentric_anomaly.sin();
    let trust_fraction = candidate as f32 / candidate_count.saturating_sub(1).max(1) as f32;
    let along_track_offset = (trust_fraction * std::f32::consts::TAU).sin()
        * (NEAR_SYNC_TRUST_RADIUS_METERS - 2.0);
    (
        radial * x + tangent * (y + along_track_offset) + normal * normal_offset,
        NEAR_SYNC_PERICENTER_RADIUS_METERS + radial_offset,
    )
}

fn direct_voxel_field(
    position: Vec3,
    voxels: &[InvertedDensityVoxel],
    density_scale: f32,
) -> (Vec3, Mat3) {
    let mut acceleration = Vec3::ZERO;
    let mut gradient = Mat3::ZERO;
    for voxel in voxels {
        let displacement = position - voxel.center;
        let radius_squared = displacement.length_squared().max(1.0e-6);
        let inverse_radius = radius_squared.sqrt().recip();
        let inverse_radius3 = inverse_radius / radius_squared;
        let mass = voxel.volume * voxel.density * density_scale;
        acceleration += -G * mass * displacement * inverse_radius3;
        let outer = Mat3::from_cols(
            displacement * displacement.x,
            displacement * displacement.y,
            displacement * displacement.z,
        );
        gradient += -G
            * mass
            * (Mat3::IDENTITY * inverse_radius3
                - outer * (3.0 * inverse_radius3 / radius_squared));
    }
    (acceleration, gradient)
}

fn gradient_consistency_error(
    position: Vec3,
    voxels: &[InvertedDensityVoxel],
    density_scale: f32,
) -> f32 {
    let (_, analytic) = direct_voxel_field(position, voxels, density_scale);
    let step = 0.25;
    let mut columns = [Vec3::ZERO; 3];
    for (axis, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let plus = direct_voxel_field(position + direction * step, voxels, density_scale).0;
        let minus = direct_voxel_field(position - direction * step, voxels, density_scale).0;
        columns[axis] = (plus - minus) / (2.0 * step);
    }
    let finite_difference = Mat3::from_cols(columns[0], columns[1], columns[2]);
    let frobenius = |matrix: Mat3| {
        (matrix.x_axis.length_squared()
            + matrix.y_axis.length_squared()
            + matrix.z_axis.length_squared())
        .sqrt()
    };
    frobenius(analytic - finite_difference) / frobenius(analytic).max(1.0e-20)
}
