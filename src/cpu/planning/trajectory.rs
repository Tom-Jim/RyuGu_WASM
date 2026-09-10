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

#[cfg(not(target_arch = "wasm32"))]
fn append_dynamical_candidate_states(
    candidate: u32,
    candidate_count: u32,
    reference: &[TrajectoryInversionKnot],
    reference_jets: &[PlanningReferenceJet],
    dynamics_tree: Option<&PlanningDynamicsTree>,
    states: &mut Vec<PlanningCandidateState>,
    gpu_position_bytes: &mut Vec<u8>,
) -> Option<()> {
    let sample_count = reference.len() as u32;
    if sample_count < 2 || reference_jets.len() != reference.len() {
        return None;
    }
    let (requested_radius, phase, harmonic, phase_rate) =
        candidate_perturbation_parameters(candidate, candidate_count);
    append_dynamical_candidate_at_radius(
        candidate,
        reference,
        reference_jets,
        dynamics_tree,
        requested_radius,
        phase,
        harmonic,
        phase_rate,
        states,
        gpu_position_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "wasm32"))]
fn append_dynamical_candidate_at_radius(
    candidate: u32,
    reference: &[TrajectoryInversionKnot],
    reference_jets: &[PlanningReferenceJet],
    dynamics_tree: Option<&PlanningDynamicsTree>,
    radius: f32,
    phase: f32,
    harmonic: f32,
    phase_rate: f32,
    states: &mut Vec<PlanningCandidateState>,
    gpu_position_bytes: &mut Vec<u8>,
) -> Option<()> {
    let sample_count = reference.len() as u32;
    let first = *reference.first()?;
    let first_offset =
        candidate_initial_offset(first, 0, sample_count, radius, phase, harmonic, phase_rate)?;
    let jets: Vec<_> = reference_jets
        .iter()
        .map(|jet| {
            serde_json::json!({
                "time": jet.simulation_time_seconds,
                "rotation": jet.body_rotation.to_array(),
                "position": jet.world_position.to_array(),
                "acceleration": jet.world_acceleration.to_array(),
                "jacobian": jet.world_jacobian.to_cols_array(),
            })
        })
        .collect();
    let sources: Vec<_> = dynamics_tree
        .map(|tree| tree.sources())
        .unwrap_or(&[])
        .iter()
        .map(|(p, m)| (p.to_array(), *m))
        .collect();
    let request = serde_json::json!({
        "positions": [(first.position + first_offset).as_dvec3().to_array()],
        "velocities": [first.velocity.as_dvec3().to_array()],
        "jets": jets,
        "sources": sources,
    });
    let trajectory = crate::cpp_backend::propagate_candidates(&request.to_string()).ok()?;
    if trajectory.len() != reference.len() * 6 || !trajectory.iter().all(|v| v.is_finite()) {
        return None;
    }

    let angular_velocity =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let first_time = first.simulation_time_seconds;
    for sample in 0..sample_count {
        let reference_state = reference[sample as usize];
        let values = &trajectory[sample as usize * 6..];
        let world_position_f32 = DVec3::from_slice(values).as_vec3();
        let world_velocity_f32 = DVec3::from_slice(&values[3..]).as_vec3();
        let transverse_distance = world_position_f32.distance(reference_state.position);
        if !transverse_distance.is_finite() {
            return None;
        }
        let time = reference_state.simulation_time_seconds as f32;
        let rotation = reference_state.body_rotation;
        let body_position = rotation.inverse() * world_position_f32;
        let body_velocity =
            rotation.inverse() * (world_velocity_f32 - angular_velocity.cross(world_position_f32));
        let position_time = [body_position.x, body_position.y, body_position.z, time];
        gpu_position_bytes.extend_from_slice(bytemuck::cast_slice(&position_time));
        states.push(PlanningCandidateState {
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
                sample,
                ((reference_state.simulation_time_seconds - first_time) as f32)
                    .max(0.0)
                    .div_euclid(NEAR_SYNC_SEGMENT_MAX_SECONDS) as u32,
                1,
            ],
        });
    }
    Some(())
}

#[cfg(target_arch = "wasm32")]
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
