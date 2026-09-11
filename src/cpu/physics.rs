use crate::interface::components::*;
use bevy::prelude::*;

fn hash_benchmark_trajectory(samples: &[GravityBenchmarkSample]) -> u64 {
    samples
        .iter()
        .fold(1469598103934665603_u64, |hash, sample| {
            [
                sample.simulation_time_seconds.to_bits(),
                sample.position.x.to_bits() as u64,
                sample.position.y.to_bits() as u64,
                sample.position.z.to_bits() as u64,
                sample.velocity.x.to_bits() as u64,
                sample.velocity.y.to_bits() as u64,
                sample.velocity.z.to_bits() as u64,
            ]
            .into_iter()
            .fold(hash, |hash, value| {
                (hash ^ value).wrapping_mul(1099511628211_u64)
            })
        })
}

fn append_capture_trace_record(
    inversion: &mut TrajectoryInversionState,
    record: &[f64; 14],
    position: impl Fn(&[f64; 14]) -> Vec3,
    speed: impl Fn(&[f64; 14]) -> Vec3,
    field: impl Fn(&[f64; 14]) -> Vec3,
    attitude: impl Fn(&[f64; 14]) -> Quat,
) {
    let simulation_time_seconds = record[0];
    if inversion
        .capture_trace
        .last()
        .is_some_and(|last| simulation_time_seconds <= last.simulation_time_seconds + 1e-12)
    {
        return;
    }
    // Bound memory while waiting for a non-degenerate wall-clock arc.
    const CAPTURE_TRACE_CAPACITY: usize = 65_536;
    if inversion.capture_trace.len() >= CAPTURE_TRACE_CAPACITY {
        let drop = CAPTURE_TRACE_CAPACITY / 4;
        inversion.capture_trace.drain(..drop);
    }
    inversion.capture_trace.push(TrajectoryInversionKnot {
        position: position(record),
        velocity: speed(record),
        simulation_time_seconds,
        baseline_acceleration: field(record),
        body_rotation: attitude(record),
    });
}

pub fn physics_system(
    mut frame_pacer: Local<crate::cpp_backend::BackendFramePacer>,
    ready: Res<crate::cpp_backend::CppBackendState>,
    advance_channel: Res<crate::cpp_backend::BackendAdvanceChannel>,
    body: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut probe: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    active: Res<ActiveGravityMethod>,
    planning: Res<PlanningComparisonState>,
    acceleration: Res<SimulationAcceleration>,
    mut blend: ResMut<GravityBlendFactor>,
    mut error: ResMut<GravityRuntimeError>,
    mut clock: ResMut<SimulationClock>,
    mut visual: ResMut<ProbeVisualState>,
    mut benchmark: ResMut<GravityBenchmarkTrajectory>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    if !ready.ready || error.is_active() || planning.blocks_live_verlet(inversion.ready) {
        return;
    }
    let (Ok(body), Ok((mut transform, mut velocity, mut orbit))) =
        (body.single(), probe.single_mut())
    else {
        return;
    };
    let packet = advance_channel.take();
    let trace = if let Some(packet) = packet {
        if packet.snapshot.epoch != clock.epoch || packet.snapshot.request_id != clock.request_id {
            return;
        }
        match packet.result {
            Ok(trace) => trace,
            Err(message) => {
                if recoverable_live_field_miss(&message) {
                    // Soft status only — do not freeze Invert / First / Stress.
                    // Pause the capture clock until the Worker can advance again.
                    inversion.capture_note = Some(if message.contains("Waiting for frequency-domain") {
                        "Waiting for Eq.121 frequency-domain modes…".into()
                    } else {
                        "Live field used the far-field fallback; continuing the orbit…".into()
                    });
                    inversion.capture_last_advance_at = None;
                    return;
                }
                error.raise(message);
                return;
            }
        }
    } else {
        if inversion.preparing {
            // FMM/FFT invert's `source_sets` share this Worker. Queueing
            // another 64× advance while Preparing is showing starves it.
            return;
        }
        if !crate::cpp_backend::should_advance_backend(&mut frame_pacer, &advance_channel) {
            return;
        }
        let first_or_stress = planning.run_requested
            && planning.workload_profile != PlanningWorkloadProfile::SourceCrossover;
        // First/Stress keep the orbit alive with 1-step ticks so the Worker
        // can spend most of its time on the batch. Quadrature-only still uses
        // the user acceleration, including 64×.
        let steps = if first_or_stress {
            1
        } else {
            acceleration.stable_steps()
        };
        // Live FD force is Worker FLUPS Eq.(121) at q_B (`mathtidy.md`).
        // GPU Eq.184 stamps stay diagnostics; packing them here can abort
        // advance_frame on a non-monotonic diagnostic clock.
        let history = Vec::new();
        let initial: Vec<f64> = transform
            .translation
            .to_array()
            .into_iter()
            .chain(velocity.0.to_array())
            .chain(body.rotation.to_array())
            .map(f64::from)
            .collect();
        let snapshot = crate::cpp_backend::BackendAdvanceSnapshot {
            request_id: clock.request_id,
            epoch: clock.epoch,
        };
        if let Err(message) = crate::cpp_backend::request_advance(
            &advance_channel,
            snapshot,
            *active,
            &initial,
            clock.fixed_step_seconds,
            steps,
            &history,
        ) {
            error.raise(message);
        }
        return;
    };
    if trace.len() < 28 || !trace.len().is_multiple_of(14) || !trace.iter().all(|v| v.is_finite()) {
        error.raise("Invalid simulation snapshot sequence");
        return;
    }
    let records = trace.as_chunks::<14>().0;
    let position = |r: &[f64; 14]| Vec3::new(r[1] as f32, r[2] as f32, r[3] as f32);
    let speed = |r: &[f64; 14]| Vec3::new(r[4] as f32, r[5] as f32, r[6] as f32);
    let field = |r: &[f64; 14]| Vec3::new(r[7] as f32, r[8] as f32, r[9] as f32);
    let attitude =
        |r: &[f64; 14]| Quat::from_xyzw(r[10] as f32, r[11] as f32, r[12] as f32, r[13] as f32);
    if benchmark.epoch != clock.epoch {
        benchmark.epoch = clock.epoch;
        benchmark.samples.clear();
        benchmark.capture_id = None;
        benchmark.complete = false;
    }
    if benchmark.samples.is_empty() {
        benchmark.samples.push(GravityBenchmarkSample {
            simulation_time_seconds: records[0][0],
            position: position(&records[0]),
            velocity: speed(&records[0]),
        });
    }
    let capturing =
        !inversion.ready && needs_live_observation_arc(*active, planning.run_requested);
    if capturing {
        let now = bevy::platform::time::Instant::now();
        // Accrue only while advances are actually delivering. Gaps above a
        // few frames mean the Worker was blocked (modes / First / Stress);
        // pause the capture clock across that wall time.
        if inversion.capture_started_at.is_none() {
            inversion.capture_started_at = Some(now);
            inversion.capture_last_advance_at = Some(now);
            inversion.wall_elapsed_seconds = 0.0;
            inversion.capture_note = None;
            inversion.capture_trace.clear();
            inversion.knots.clear();
        } else if let Some(last) = inversion.capture_last_advance_at {
            let gap = now.saturating_duration_since(last).as_secs_f64();
            if let Some(accrued) = accrue_live_capture_gap(gap) {
                inversion.wall_elapsed_seconds += accrued;
            }
            inversion.capture_last_advance_at = Some(now);
        } else {
            // Resuming after an explicit pause (e.g. waiting for modes).
            inversion.capture_last_advance_at = Some(now);
        }
        // Accumulate every Worker state in the effective capture window.
        for record in records {
            append_capture_trace_record(
                &mut inversion,
                record,
                position,
                speed,
                field,
                attitude,
            );
        }
    }
    for pair in records.windows(2) {
        let b = &pair[1];
        if *active == ActiveGravityMethod::RadialAnalytic
            && inversion.truth_orbit.len() < ORBIT_HISTORY_LEN
        {
            inversion.truth_orbit.push(position(b));
        }
        if !benchmark.complete && b[0] <= BENCHMARK_DURATION_SECONDS + 1e-9 {
            benchmark.samples.push(GravityBenchmarkSample {
                simulation_time_seconds: b[0],
                position: position(b),
                velocity: speed(b),
            });
            if b[0] + 1e-9 >= BENCHMARK_DURATION_SECONDS {
                benchmark.complete = true;
                benchmark.capture_id = Some(hash_benchmark_trajectory(&benchmark.samples));
            }
        }
    }
    let last = records.last().expect("validated trace");
    // The independent backend may return fewer records than the legacy GPU
    // substep batch. Retain the latest authoritative state on every tick so
    // the visible orbit reflects real propagation rather than an empty trail.
    for record in records {
        if orbit.0.len() >= ORBIT_HISTORY_LEN {
            orbit.0.pop_front();
        }
        orbit.0.push_back(position(record));
    }
    transform.translation = position(last);
    velocity.0 = speed(last);
    clock.elapsed_seconds = (last[0] * 1e9).round() * 1e-9;
    clock.request_id = clock.request_id.wrapping_add(1);
    visual.accept_authoritative_sample(
        transform.translation,
        velocity.0,
        clock.epoch,
        bevy::platform::time::Instant::now(),
    );
    blend.0 = 1.0;
}

pub fn ryugu_rotation_system(
    mut ryugu_query: Query<&mut Transform, With<RyuguMarker>>,
    clock: Res<SimulationClock>,
) {
    let angular_speed = std::f64::consts::TAU / RYUGU_ROTATION_PERIOD_SECS as f64;
    let rotation = Quat::from_axis_angle(
        RYUGU_SPIN_AXIS.normalize(),
        (angular_speed * clock.elapsed_seconds) as f32,
    );
    for mut transform in ryugu_query.iter_mut() {
        // Derive body attitude from the authoritative simulation clock. If
        // physics is waiting for a GPU readback, both clock and body frame now
        // remain frozen instead of silently diverging.
        transform.rotation = rotation;
    }
}

fn recoverable_live_field_miss(message: &str) -> bool {
    message.contains("Waiting for frequency-domain")
        || message.contains("outside FLUPS grid")
        || message.contains("Mass source outside FLUPS")
}

/// Accrue wall time for a delivered live advance. Slow FMM/FFT ticks (often
/// >350 ms) still count; only an explicit pause (`capture_last_advance_at =
/// None`, e.g. waiting for Eq.121 modes) skips the gap. Gaps longer than the
/// stall cap are treated as a blocked Worker, not live integration.
pub(crate) fn accrue_live_capture_gap(gap_secs: f64) -> Option<f64> {
    const CAPTURE_IDLE_STALL_SECS: f64 = 8.0;
    (gap_secs.is_finite() && gap_secs > 0.0 && gap_secs <= CAPTURE_IDLE_STALL_SECS)
        .then_some(gap_secs)
}

#[cfg(test)]
mod live_capture_gap_tests {
    use super::{accrue_live_capture_gap, recoverable_live_field_miss};

    #[test]
    fn slow_fmm_ticks_still_accrue() {
        assert_eq!(accrue_live_capture_gap(0.2), Some(0.2));
        assert_eq!(accrue_live_capture_gap(0.5), Some(0.5));
        assert_eq!(accrue_live_capture_gap(2.0), Some(2.0));
        assert_eq!(accrue_live_capture_gap(8.0), Some(8.0));
        assert_eq!(accrue_live_capture_gap(30.0), None);
        assert_eq!(accrue_live_capture_gap(0.0), None);
    }

    #[test]
    fn flups_grid_miss_is_recoverable() {
        assert!(recoverable_live_field_miss("Gravity target outside FLUPS grid"));
        assert!(recoverable_live_field_miss("Mass source outside FLUPS grid"));
        assert!(recoverable_live_field_miss(
            "Waiting for frequency-domain Eq.121 modes"
        ));
        assert!(!recoverable_live_field_miss("WebGPU device lost"));
    }
}
