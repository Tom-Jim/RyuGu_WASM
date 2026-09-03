pub fn record_probe_jacobi_system(
    active_method: Res<ActiveGravityMethod>,
    radial_samples: Option<Res<RadialGravityHistory>>,
    werner_samples: Option<Res<WernerGravityHistory>>,
    mmfft_samples: Option<Res<MmfftCompressedHistory>>,
    fmm_samples: Option<Res<FmmGravityHistory>>,
    gravity_blend: Res<GravityBlendFactor>,
    clock: Res<SimulationClock>,
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
        mmfft_samples.as_deref(),
        fmm_samples.as_deref(),
    );
    let sample = active_history.and_then(|samples| samples.latest_for_epoch(clock.epoch));
    let Some(sample) = sample else {
        return;
    };
    if history.last_request_id == Some(sample.snapshot.request_id) {
        return;
    }

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
    let positive_potential = sample.positive_potential;
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
