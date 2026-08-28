pub fn record_probe_jacobi_system(
    active_method: Res<ActiveGravityMethod>,
    radial_samples: Option<Res<RadialGravityHistory>>,
    werner_samples: Option<Res<WernerGravityHistory>>,
    eq106_samples: Option<Res<Eq106GpuHistory>>,
    mmfft_samples: Option<Res<MmfftCompressedHistory>>,
    fmm_samples: Option<Res<FmmGravityHistory>>,
    gravity_blend: Res<GravityBlendFactor>,
    clock: Res<SimulationClock>,
    curved_residual: Res<CurvedArcResidualHistory>,
    cassini: Query<(&Transform, &Velocity), With<CassiniMarker>>,
    ryugu: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut history: ResMut<JacobiHistory>,
) {
    if gravity_blend.0 < 1.0 {
        return;
    }
    let active_history = select_history(
        *active_method,
        radial_samples.as_deref(),
        werner_samples.as_deref(),
        eq106_samples.as_deref(),
        mmfft_samples.as_deref(),
        fmm_samples.as_deref(),
    );
    let sample = active_history.and_then(|samples| {
        if *active_method == ActiveGravityMethod::CurvedArcEq106 {
            samples.completed_at_or_before(clock.epoch, clock.elapsed_seconds)
        } else {
            samples.latest_for_epoch(clock.epoch)
        }
    });
    let Some(sample) = sample else {
        return;
    };
    if history.last_request_id == Some(sample.snapshot.request_id) {
        return;
    }

    // The CPU integrator advances the live state using interpolated GPU
    // fields. Accumulate the gravitational work along that same path for all
    // methods, then use the first spectral potential only as the integration
    // constant. This removes asynchronous snapshot lag and per-segment
    // potential gauge jumps from the Jacobi diagnostic.
    let (Ok((probe_transform, probe_velocity)), Ok(ryugu_transform)) =
        (cassini.single(), ryugu.single())
    else {
        return;
    };
    let world_to_body = ryugu_transform.rotation.inverse();
    let body_position = world_to_body * (probe_transform.translation - ryugu_transform.translation);
    let inertial_velocity_body = world_to_body * probe_velocity.0;
    let angular_velocity_world =
        RYUGU_SPIN_AXIS.normalize() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let angular_velocity_body = world_to_body * angular_velocity_world;
    let base_potential = if *active_method == ActiveGravityMethod::CurvedArcEq106 {
        let Some(eq106_history) = eq106_samples.as_ref() else {
            return;
        };
        let Some(potential) = eq106_interpolated_positive_potential(
            &eq106_history.0,
            clock.epoch,
            clock.elapsed_seconds,
            body_position,
        ) else {
            return;
        };
        potential
    } else {
        sample.positive_potential
    };
    let positive_potential =
        if let Some(curve_work) = curved_residual.curve_work_at(clock.elapsed_seconds) {
            let origin_potential = *history
                .eq106_origin_potential
                .get_or_insert(base_potential as f64);
            let origin_curve_work = *history.eq106_origin_curve_work.get_or_insert(curve_work);
            (origin_potential + curve_work - origin_curve_work) as f32
        } else {
            base_potential
        };
    if !positive_potential.is_finite() || positive_potential <= 0.0 {
        return;
    }
    let Some(jacobi_constant) = rotating_frame_jacobi_constant(
        body_position,
        inertial_velocity_body,
        positive_potential,
        angular_velocity_body,
    ) else {
        return;
    };

    let origin = *history
        .origin_simulation_seconds
        .get_or_insert(clock.elapsed_seconds);
    history.elapsed_simulation_seconds = clock.elapsed_seconds - origin;
    history.last_request_id = Some(sample.snapshot.request_id);
    history.last_sample_method = Some(*active_method);
    if history.samples.len() == JACOBI_HISTORY_CAPACITY {
        history.samples.pop_front();
    }
    let simulation_time_seconds = history.elapsed_simulation_seconds;
    history.samples.push_back(JacobiSample {
        simulation_time_seconds,
        jacobi_constant,
    });
}
