pub fn apply_probe_input_system(
    probe: Res<ProbeInitialConditions>,
    mut gravity_acceleration: ResMut<GravityAcceleration>,
    mut werner_acceleration: Option<ResMut<WernerAcceleration>>,
    mut gravity_blend: ResMut<GravityBlendFactor>,
    mut radial_potential: ResMut<GravityPotential>,
    mut werner_potential: Option<ResMut<WernerPotential>>,
    mut radial_samples: Option<ResMut<RadialGravityHistory>>,
    mut werner_samples: Option<ResMut<WernerGravityHistory>>,
    mut mmfft_samples: Option<ResMut<MmfftCompressedHistory>>,
    mut fmm_samples: Option<ResMut<FmmGravityHistory>>,
    mut simulation_clock: ResMut<SimulationClock>,
    mut jacobi_history: ResMut<JacobiHistory>,
    mut frequency_domain_result: ResMut<FrequencyDomainTrajectoryBatchResult>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        With<CassiniMarker>,
    >,
    mut ryugu_query: Query<&mut Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
) {
    if !probe.is_changed() {
        return;
    }
    gravity_acceleration.0 = Vec3::ZERO;
    if let Some(acceleration) = werner_acceleration.as_deref_mut() {
        acceleration.0 = Vec3::ZERO;
    }
    gravity_blend.0 = 0.0;
    radial_potential.0 = None;
    if let Some(potential) = werner_potential.as_deref_mut() {
        potential.0 = None;
    }
    for history in [
        radial_samples.as_deref_mut().map(|history| &mut history.0),
        werner_samples.as_deref_mut().map(|history| &mut history.0),
        mmfft_samples.as_deref_mut().map(|history| &mut history.0),
        fmm_samples.as_deref_mut().map(|history| &mut history.0),
    ]
    .into_iter()
    .flatten()
    {
        history.clear();
    }
    simulation_clock.reset_state();
    jacobi_history.reset();
    frequency_domain_result.capture_id = None;
    frequency_domain_result.observations.clear();

    if let Ok((mut transform, mut velocity, mut history)) = cassini_query.single_mut() {
        transform.translation = probe.position;
        velocity.0 = probe.velocity();
        history.0.clear();
        history.0.push_back(probe.position);
    }
    if let Some(mut transform) = ryugu_query.iter_mut().next() {
        transform.rotation = Quat::IDENTITY;
        transform.translation = Vec3::ZERO;
    }
}

pub fn clear_inversion_request_on_probe_change(
    probe: Res<ProbeInitialConditions>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if probe.is_changed() {
        inversion.start_requested = false;
    }
}

pub fn clear_runtime_error_on_probe_change(
    probe: Res<ProbeInitialConditions>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if probe.is_changed() {
        runtime_error.clear();
    }
}

pub fn probe_collision_system(
    mut crash: ResMut<ProbeCrashState>,
    cassini_query: Query<&Transform, With<CassiniMarker>>,
    ryugu_query: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if crash.active || runtime_error.is_active() {
        return;
    }
    let Some(probe) = cassini_query.iter().next() else {
        return;
    };
    let body_position = ryugu_query
        .iter()
        .next()
        .map_or(Vec3::ZERO, |transform| transform.translation);
    let collision_radius = RYUGU_COLLISION_RADIUS_METERS + PROBE_COLLISION_RADIUS_METERS;
    if probe.translation.distance_squared(body_position) <= collision_radius * collision_radius {
        crash.trigger();
        runtime_error.raise("Probe collision detected; simulation paused for scene reset.");
    }
}

pub fn reset_after_probe_crash_scene_system(
    time: Res<Time>,
    mut crash: ResMut<ProbeCrashState>,
    mut reset_request: ResMut<ProbeCrashResetRequest>,
    mut probe: ResMut<ProbeInitialConditions>,
) {
    if !crash.active {
        return;
    }
    crash.elapsed_seconds += time.delta_secs();
    if crash.elapsed_seconds < ProbeCrashState::DISPLAY_SECONDS {
        return;
    }
    *probe = ProbeInitialConditions::default();
    crash.clear();
    reset_request.0 = true;
}

pub fn reset_after_probe_crash_state_system(
    mut reset_request: ResMut<ProbeCrashResetRequest>,
    mut active_method: ResMut<ActiveGravityMethod>,
    mut performance: ResMut<PerformanceComparisonState>,
    mut inversion: ResMut<TrajectoryInversionState>,
    mut clock: ResMut<SimulationClock>,
    mut blend: ResMut<GravityBlendFactor>,
    mut acceleration: ResMut<GravityAcceleration>,
    mut potential: ResMut<GravityPotential>,
    mut runtime_error: ResMut<GravityRuntimeError>,
    mut jacobi: ResMut<JacobiHistory>,
    mut benchmark: ResMut<GravityBenchmarkTrajectory>,
    mut sensitivity: ResMut<DensitySensitivityCaches>,
    mut frequency_domain_result: ResMut<FrequencyDomainTrajectoryBatchResult>,
    mut histories: ParamSet<(
        Option<ResMut<RadialGravityHistory>>,
        Option<ResMut<WernerGravityHistory>>,
        Option<ResMut<MmfftCompressedHistory>>,
        Option<ResMut<FmmGravityHistory>>,
    )>,
) {
    if !reset_request.0 {
        return;
    }
    *reset_request = ProbeCrashResetRequest(false);
    *active_method = ActiveGravityMethod::RadialAnalytic;
    *performance = PerformanceComparisonState::default();
    *inversion = TrajectoryInversionState::default();
    clock.reset_state();
    blend.0 = 0.0;
    acceleration.0 = Vec3::ZERO;
    potential.0 = None;
    runtime_error.clear();
    jacobi.reset();
    benchmark.epoch = clock.epoch;
    benchmark.samples.clear();
    benchmark.capture_id = None;
    benchmark.complete = false;
    *sensitivity = DensitySensitivityCaches::default();
    frequency_domain_result.capture_id = None;
    frequency_domain_result.observations.clear();
    if let Some(history) = histories.p0().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p1().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p2().as_deref_mut() {
        history.0.clear();
    }
    if let Some(history) = histories.p3().as_deref_mut() {
        history.0.clear();
    }
}
