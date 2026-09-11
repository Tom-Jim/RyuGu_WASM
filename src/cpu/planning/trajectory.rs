fn central_reference_states(
    reference: &[TrajectoryInversionKnot],
) -> Option<Vec<PlanningCandidateState>> {
    let angular_velocity =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    reference
        .iter()
        .enumerate()
        .map(|(sample, state)| {
            if !state.position.is_finite()
                || !state.velocity.is_finite()
                || !state.body_rotation.is_finite()
                || !state.simulation_time_seconds.is_finite()
            {
                return None;
            }
            let body_position = state.body_rotation.inverse() * state.position;
            let body_velocity = state.body_rotation.inverse()
                * (state.velocity - angular_velocity.cross(state.position));
            Some(PlanningCandidateState {
                position_time: [
                    body_position.x,
                    body_position.y,
                    body_position.z,
                    state.simulation_time_seconds as f32,
                ],
                velocity_distance: [body_velocity.x, body_velocity.y, body_velocity.z, 0.0],
                body_rotation: state.body_rotation.to_array(),
                identity: [u32::MAX, sample as u32, 0, 0],
            })
        })
        .collect()
}

fn write_dynamical_candidate_sample(
    candidate: u32,
    sample: usize,
    reference: &[TrajectoryInversionKnot],
    values: &[f64],
    states: &mut [PlanningCandidateState],
    gpu_position_bytes: &mut [u8],
) -> Option<()> {
    if sample >= reference.len() || values.len() != 6 {
        return None;
    }
    let angular_velocity =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let first_time = reference.first()?.simulation_time_seconds;
    let reference_state = reference[sample];
    let world_position = DVec3::from_slice(values).as_vec3();
    let world_velocity = DVec3::from_slice(&values[3..]).as_vec3();
    let transverse_distance = world_position.distance(reference_state.position);
    if !transverse_distance.is_finite() {
        return None;
    }
    let rotation = reference_state.body_rotation;
    let body_position = rotation.inverse() * world_position;
    let body_velocity =
        rotation.inverse() * (world_velocity - angular_velocity.cross(world_position));
    let position_time = [
        body_position.x,
        body_position.y,
        body_position.z,
        reference_state.simulation_time_seconds as f32,
    ];
    let state_index = candidate as usize * reference.len() + sample;
    let byte_start = state_index * 16;
    let byte_end = byte_start + 16;
    let state = states.get_mut(state_index)?;
    let bytes = gpu_position_bytes.get_mut(byte_start..byte_end)?;
    bytes.copy_from_slice(bytemuck::cast_slice(&position_time));
    *state = PlanningCandidateState {
        position_time,
        velocity_distance: [
            body_velocity.x,
            body_velocity.y,
            body_velocity.z,
            transverse_distance,
        ],
        body_rotation: rotation.to_array(),
        identity: [
            candidate,
            sample as u32,
            ((reference_state.simulation_time_seconds - first_time) as f32)
                .max(0.0)
                .div_euclid(NEAR_SYNC_SEGMENT_MAX_SECONDS) as u32,
            1,
        ],
    };
    Some(())
}

fn candidate_initial_offset(
    reference_state: TrajectoryInversionKnot,
    sample: u32,
    sample_count: u32,
    radius: f32,
    phase: f32,
    harmonic: f32,
    phase_rate: f32,
) -> Option<Vec3> {
    let tangent = reference_state.velocity.normalize_or_zero();
    let normal_hint = RYUGU_SPIN_AXIS.normalize_or_zero();
    let normal = (normal_hint - tangent * normal_hint.dot(tangent)).normalize_or_zero();
    let binormal = tangent.cross(normal).normalize_or_zero();
    if tangent == Vec3::ZERO || normal == Vec3::ZERO || binormal == Vec3::ZERO {
        return None;
    }
    let normalized_time = sample as f32 / sample_count.saturating_sub(1) as f32 - 0.5;
    let angle = phase + harmonic * std::f32::consts::TAU * normalized_time + phase_rate;
    let envelope = 0.82 + 0.18 * (std::f32::consts::TAU * normalized_time + phase).cos();
    let offset_radius = (radius * envelope).min(PLANNING_PERTURBATION_RADIUS_METERS);
    Some(normal * (offset_radius * angle.cos()) + binormal * (offset_radius * angle.sin()))
}

fn build_planning_reference_jets(
    reference: &[TrajectoryInversionKnot],
) -> Vec<PlanningReferenceJet> {
    reference
        .iter()
        .map(|state| {
            let rotation = DQuat::from_xyzw(
                f64::from(state.body_rotation.x),
                f64::from(state.body_rotation.y),
                f64::from(state.body_rotation.z),
                f64::from(state.body_rotation.w),
            )
            .normalize();
            let world_position = state.position.as_dvec3();
            PlanningReferenceJet {
                simulation_time_seconds: state.simulation_time_seconds,
                body_rotation: rotation,
                world_position,
                world_acceleration: state.baseline_acceleration.as_dvec3(),
                world_jacobian: DMat3::ZERO,
            }
        })
        .collect()
}
