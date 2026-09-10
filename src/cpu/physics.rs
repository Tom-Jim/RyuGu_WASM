use crate::interface::components::*;
use bevy::prelude::*;

fn hermite_vector(a: Vec3, b: Vec3, tangent_a: Vec3, tangent_b: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * a
        + (t3 - 2.0 * t2 + t) * tangent_a
        + (-2.0 * t3 + 3.0 * t2) * b
        + (t3 - t2) * tangent_b
}

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

pub fn physics_system(
    mut frame_pacer: Local<crate::cpp_backend::BackendFramePacer>,
    ready: Res<crate::cpp_backend::CppBackendState>,
    advance_channel: Res<crate::cpp_backend::BackendAdvanceChannel>,
    body: Query<&Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
    mut probe: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        (With<CassiniMarker>, Without<RyuguMarker>),
    >,
    frequency: Option<Res<crate::gpu::equation106::Equation106History>>,
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
    if !ready.ready || !ready.worker_ready || error.is_active() || planning.blocks_realtime_gpu() {
        return;
    }
    let (Ok(body), Ok((mut transform, mut velocity, mut orbit))) =
        (body.single(), probe.single_mut())
    else {
        return;
    };
    let packet = advance_channel
        .data
        .lock()
        .expect("backend advance result channel poisoned")
        .take();
    let trace = if let Some(packet) = packet {
        if packet.snapshot.epoch != clock.epoch || packet.snapshot.request_id != clock.request_id {
            return;
        }
        match packet.result {
            Ok(trace) => trace,
            Err(message) => {
                if message.contains("Waiting for frequency-domain samples") {
                    return;
                }
                error.raise(message);
                return;
            }
        }
    } else {
        if advance_channel
            .in_flight
            .load(std::sync::atomic::Ordering::Acquire)
            || !crate::cpp_backend::should_advance_backend(&mut frame_pacer, *active)
        {
            return;
        }
        let mut steps = acceleration.stable_steps();
        let mut history = Vec::new();
        if *active == ActiveGravityMethod::FrequencyDomain {
            let Some(frequency) = frequency else { return };
            for sample in &frequency.0.samples {
                if sample.snapshot.epoch != clock.epoch {
                    continue;
                }
                history.push(sample.snapshot.simulation_time_seconds);
                history.extend(sample.body_acceleration.to_array().map(f64::from));
            }
            let samples = history.as_chunks::<4>().0;
            let Some(last) = samples.last() else { return };
            let horizon = if samples.len() > 1 {
                2.0 * (last[0] - samples[samples.len() - 2][0])
            } else {
                clock.fixed_step_seconds
            };
            let available = ((last[0] + horizon - clock.elapsed_seconds) / clock.fixed_step_seconds
                + 1e-6)
                .floor()
                .max(0.0) as u32;
            steps = steps.min(available);
            if steps == 0 {
                return;
            }
        }
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
    if *active == ActiveGravityMethod::FrequencyDomain
        && !inversion.ready
        && inversion.knots.is_empty()
    {
        let r = &records[0];
        inversion.knots.push(TrajectoryInversionKnot {
            position: position(r),
            velocity: speed(r),
            simulation_time_seconds: r[0],
            baseline_acceleration: field(r),
            body_rotation: attitude(r),
        });
    }
    for pair in records.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if *active == ActiveGravityMethod::FrequencyDomain && !inversion.ready {
            let interval = TRAJECTORY_INVERSION_CAPTURE_SECONDS
                / (TRAJECTORY_INVERSION_SAMPLE_COUNT - 1) as f64;
            while inversion.knots.len() < TRAJECTORY_INVERSION_SAMPLE_COUNT {
                let time = inversion.knots.len() as f64 * interval;
                if time > b[0] + 1e-9 {
                    break;
                }
                let dt = (b[0] - a[0]) as f32;
                let fraction = ((time - a[0]) / (b[0] - a[0])).clamp(0.0, 1.0) as f32;
                inversion.knots.push(TrajectoryInversionKnot {
                    position: hermite_vector(
                        position(a),
                        position(b),
                        speed(a) * dt,
                        speed(b) * dt,
                        fraction,
                    ),
                    velocity: speed(a).lerp(speed(b), fraction),
                    simulation_time_seconds: time,
                    baseline_acceleration: field(a).lerp(field(b), fraction),
                    body_rotation: attitude(a).slerp(attitude(b), fraction),
                });
            }
        }
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
    if orbit.0.len() >= ORBIT_HISTORY_LEN {
        orbit.0.pop_front();
    }
    orbit.0.push_back(position(last));
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
