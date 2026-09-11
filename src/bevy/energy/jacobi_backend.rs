/// Prevents a very fast Worker from turning Jacobi submits into a busy poll.
/// New samples are otherwise driven by physics `request_id`, not wall-clock.
const LIVE_JACOBI_SUBMIT_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Default)]
pub(crate) struct LiveJacobiPacer {
    last_submit: Option<Instant>,
}

impl LiveJacobiPacer {
    fn allow(&mut self) -> bool {
        let now = Instant::now();
        if self
            .last_submit
            .is_some_and(|last| now.duration_since(last) < LIVE_JACOBI_SUBMIT_INTERVAL)
        {
            return false;
        }
        self.last_submit = Some(now);
        true
    }
}

/// One outstanding pointwise field request.
///
/// The kinematics travel with the request so a late answer is still paired with
/// the position it was measured at. Reusing the live `Transform` instead would
/// mix a stale potential with a newer position and silently change the recorded
/// Jacobi constant.
struct PendingJacobiField {
    snapshot: crate::cpp_backend::BackendEvaluateSnapshot,
    method: ActiveGravityMethod,
    body_position: Vec3,
    inertial_velocity_body: Vec3,
    angular_velocity_body: Vec3,
    simulation_time_seconds: f64,
    clock_request_id: u64,
}

#[derive(Default)]
pub(crate) struct LiveJacobiWorker {
    pending: Option<PendingJacobiField>,
    next_request_id: u64,
}

pub fn record_probe_jacobi_system(
    active_method: Res<ActiveGravityMethod>,
    gravity_blend: Res<GravityBlendFactor>,
    clock: Res<SimulationClock>,
    cassini: Query<(&Transform, &Velocity), With<CassiniMarker>>,
    ryugu: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut history: ResMut<JacobiHistory>,
    cpp_backend: Res<crate::cpp_backend::CppBackendState>,
    evaluate_channel: Res<crate::cpp_backend::BackendEvaluateChannel>,
    advance_channel: Res<crate::cpp_backend::BackendAdvanceChannel>,
    mut live_pacer: Local<LiveJacobiPacer>,
    mut live_worker: Local<LiveJacobiWorker>,
    inversion: Res<TrajectoryInversionState>,
    planning: Res<PlanningComparisonState>,
) {
    if gravity_blend.0 < 1.0 {
        return;
    }
    if inversion.preparing {
        return;
    }
    let first_or_stress = planning.run_requested
        && planning.workload_profile != PlanningWorkloadProfile::SourceCrossover;
    if first_or_stress {
        return;
    }

    // Pointwise Jacobi, including frequency-domain: live FD force is FLUPS
    // Eq.(121) (inverse Laplace of Eq.(106) at q_B). GPU Eq.(184) stays the
    // spectral chart; do not feed 64-node Eq.(106) stamps into C_J.
    if !cpp_backend.ready {
        // Keep the invariant "no pending request implies nothing in
        // flight" so a re-configured backend can always start again.
        if live_worker.pending.take().is_some() {
            evaluate_channel.reset();
        }
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

    let completed = evaluate_channel.take();
    if let Some(packet) = completed {
        let Some(pending) = live_worker.pending.take() else {
            return;
        };
        // Method switches and probe edits reset the channel, so anything
        // that is not the answer to the outstanding request is stale.
        if packet.snapshot != pending.snapshot
            || pending.snapshot.epoch != clock.epoch
            || pending.method != *active_method
        {
            return;
        }
        let Ok((_, positive_potential)) =
            packet.result.and_then(crate::cpp_backend::decode_field_response)
        else {
            return;
        };
        let Some(jacobi_constant) = rotating_frame_jacobi_constant(
            pending.body_position,
            pending.inertial_velocity_body,
            positive_potential,
            pending.angular_velocity_body,
        ) else {
            return;
        };
        let origin = *history
            .origin_simulation_seconds
            .get_or_insert(pending.simulation_time_seconds);
        history.elapsed_simulation_seconds = pending.simulation_time_seconds - origin;
        history.last_request_id = Some(pending.clock_request_id);
        history.last_sample_method = Some(pending.method);
        if history.samples.len() == JACOBI_HISTORY_CAPACITY {
            history.samples.pop_front();
        }
        let simulation_time_seconds = history.elapsed_simulation_seconds;
        history.samples.push_back(JacobiSample {
            simulation_time_seconds,
            jacobi_constant,
        });
        return;
    }
    if live_worker.pending.is_some() {
        // A cancelled request never delivers a packet, so `in_flight`
        // falling back to false means the answer was discarded.
        if !evaluate_channel.is_idle() {
            return;
        }
        live_worker.pending = None;
    }
    if history.last_request_id == Some(clock.request_id) {
        return;
    }
    if !advance_channel.is_idle() {
        return;
    }
    if !live_pacer.allow() {
        return;
    }
    live_worker.next_request_id = live_worker.next_request_id.wrapping_add(1).max(1);
    let snapshot = crate::cpp_backend::BackendEvaluateSnapshot {
        request_id: live_worker.next_request_id,
        epoch: clock.epoch,
    };
    match crate::cpp_backend::request_evaluate(
        &evaluate_channel,
        snapshot,
        *active_method,
        body_position,
    ) {
        Ok(true) => {
            live_worker.pending = Some(PendingJacobiField {
                snapshot,
                method: *active_method,
                body_position,
                inertial_velocity_body,
                angular_velocity_body,
                simulation_time_seconds: clock.elapsed_seconds,
                clock_request_id: clock.request_id,
            });
        }
        Ok(false) => {}
        Err(error) => bevy::log::warn!("Probe Jacobi field evaluation failed: {error}"),
    }
}

#[cfg(test)]
mod live_jacobi_pacer_tests {
    use super::*;

    #[test]
    fn live_jacobi_submit_floor_rejects_immediate_retry() {
        let mut pacer = LiveJacobiPacer::default();
        assert!(pacer.allow());
        assert!(!pacer.allow());
    }
}
