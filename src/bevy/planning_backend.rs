pub fn update_planning_results_from_inversion_system(
    mut commands: Commands,
    inversion: Res<TrajectoryInversionState>,
    radial: Option<Res<RadialGravitySource>>,
    aggregated: Option<Res<crate::cpu::curved_arc::AggregatedGravitySource>>,
    mut planning: ResMut<PlanningComparisonState>,
    mut batch_builder: Local<Option<crate::cpu::planning::PlanningBatchBuilder>>,
) {
    if !planning.run_requested {
        // A UI cancellation must release the CPU-side candidate builder too;
        // otherwise a hidden quadrature page would retain a large work queue.
        *batch_builder = None;
        return;
    }
    if planning.batch_job.is_some() {
        return;
    }
    let Some(capture_id) = inversion.capture_id else {
        planning.status = format!(
            "{} planning queued: freeze a reference trajectory first.",
            planning.workload_profile.label()
        );
        return;
    };
    let source_hash = inversion.capture_source_hash;
    if source_hash == 0 {
        planning.status = "Planning queued: the frozen capture identity is incomplete.".into();
        return;
    }
    let dimensions = planning.workload_profile.dimensions();
    let builder_matches = batch_builder.as_ref().is_some_and(|builder| {
        builder.matches(
            planning.workload_profile,
            planning.run_id,
            capture_id,
            source_hash,
            planning.requested_source_count,
        )
    });
    if !builder_matches {
        let Some(radial) = radial else {
            planning.status =
                "Planning queued: the common radial volume source is not ready.".into();
            return;
        };
        let Some(aggregated) = aggregated else {
            planning.status =
                "Planning queued: the common 1024-source geometry is not ready.".into();
            return;
        };
        let Some((voxels, voxel_size)) = crate::cpu::inversion::build_density_voxels(
            &radial,
            ActiveGravityMethod::CurvedArcEq106,
        ) else {
            planning.status =
                "Planning batch could not build the independent 56-region truth geometry.".into();
            return;
        };
        let Some(builder) = crate::cpu::planning::PlanningBatchBuilder::new(
            planning.workload_profile,
            planning.run_id,
            capture_id,
            inversion.capture_epoch,
            source_hash,
            planning.requested_source_count,
            voxel_size,
            &inversion.knots,
            &voxels,
            &aggregated,
        ) else {
            planning.status =
                "Planning batch initialization failed its equal-mass or source checks.".into();
            planning.run_requested = false;
            return;
        };
        *batch_builder = Some(builder);
        planning.status = format!(
            "{} candidate preparation: 0 / {} trajectories ({} sources).",
            planning.workload_profile.label(),
            dimensions.0,
            planning.requested_source_count,
        );
        return;
    }
    let builder = batch_builder.as_mut().expect("matched planning builder");
    if crate::browser_frame_rate().is_some_and(|fps| fps < PLANNING_MIN_INTERACTIVE_FPS)
        || crate::browser_recent_frame_ms()
            .is_some_and(|milliseconds| milliseconds > PLANNING_MAX_RECENT_FRAME_MS)
    {
        planning.status = format!(
            "{} candidate preparation yielded to rendering at {:.1} FPS / {:.1} ms recent frame: {} / {} curves.",
            planning.workload_profile.label(),
            crate::browser_frame_rate().unwrap_or(0.0),
            crate::browser_recent_frame_ms().unwrap_or(0.0),
            builder.completed_candidates(),
            dimensions.0
        );
        return;
    }
    let candidate_budget = if cfg!(target_arch = "wasm32") {
        PLANNING_BUILD_CANDIDATES_PER_FRAME
    } else {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .saturating_mul(2)
            .min(u32::MAX as usize) as u32
    };
    if !builder.advance(candidate_budget) {
        planning.status = "Planning candidate generation left the certified 15 m tube.".into();
        planning.run_requested = false;
        *batch_builder = None;
        return;
    }
    if !builder.is_complete() {
        planning.status = format!(
            "{} candidate preparation: {} / {} trajectories.",
            planning.workload_profile.label(),
            builder.completed_candidates(),
            dimensions.0
        );
        return;
    }
    let Some((batch, _common_preparation_ms)) =
        batch_builder.take().and_then(|builder| builder.finish())
    else {
        planning.status = "Planning candidate batch could not be finalized.".into();
        planning.run_requested = false;
        return;
    };
    planning.reference_duration_seconds = inversion
        .knots
        .first()
        .zip(inversion.knots.last())
        .map_or(0.0, |(first, last)| {
            (last.simulation_time_seconds - first.simulation_time_seconds) as f32
        });
    let batch_id = batch.batch_id;
    let density_seed = batch.density_seed;
    let maximum_density_mass_relative_error = batch
        .density_model_masses
        .iter()
        .map(|mass| ((mass - batch.target_mass) / batch.target_mass).abs())
        .fold(0.0_f64, f64::max);
    commands.insert_resource(batch);
    commands.insert_resource(PlanningGpuRequest::default());
    commands.insert_resource(PlanningMethodPayload::default());
    let order_rotation = if planning.source_curve_active {
        planning.source_curve_repeat as usize
    } else {
        0
    };
    let method_order = planning_method_order(order_rotation);
    planning.batch_job = Some(PlanningBatchJob {
        run_id: planning.run_id,
        profile: planning.workload_profile,
        method: method_order[0],
        method_order,
        method_order_index: 0,
        batch_id,
        candidate_count: dimensions.0,
        density_model_count: dimensions.1,
        samples_per_candidate: dimensions.2,
        density_seed,
        maximum_density_mass_relative_error,
        request_id: planning.run_id.wrapping_shl(24),
        density_model: 0,
        candidate_start: 0,
        candidate_tile_size: PLANNING_GPU_TILE_INITIAL_CANDIDATES,
        minimum_tile_size_used: u32::MAX,
        maximum_tile_size_used: 0,
        gpu_request_count: 0,
        raw_gpu_request_count: 0,
        last_request_candidate_count: 0,
        awaiting_gpu: false,
        awaiting_gpu_frames: 0,
        warm_repetition: false,
        certified_repetition: false,
        total_evaluations: u64::from(dimensions.0)
            * u64::from(dimensions.1)
            * u64::from(dimensions.2),
        gravity_error_sum: 0.0,
        gravity_reference_sum: 0.0,
        gravity_samples: 0,
        gradient_error_sum: 0.0,
        gradient_reference_sum: 0.0,
        gradient_samples: 0,
        verification_sample_count: 0,
        raw_gravity_error_sum: 0.0,
        raw_gradient_error_sum: 0.0,
        pointwise_gravity_errors: Vec::new(),
        pointwise_gradient_errors: Vec::new(),
        certified_gravity_error_sum: 0.0,
        certified_gravity_reference_sum: 0.0,
        certified_gradient_error_sum: 0.0,
        certified_gradient_reference_sum: 0.0,
        certified_gravity_samples: 0,
        certified_gradient_samples: 0,
        certified_verification_sample_count: 0,
        certified_rejected_sample_count: 0,
        certified_candidate_valid: vec![true; dimensions.0 as usize],
        rejected_sample_count: 0,
        rejection_counts: [0; 6],
        self_fd_step_maxima: [0.0; 5],
        first_rejection: None,
        maximum_gradient_self_fd_relative_error: 0.0,
        pericenter_error_m: 0.0,
        minimum_altitude_m: f32::INFINITY,
        discrimination_sum: 0.0,
        discrimination_reference_sum: 0.0,
        discrimination_samples: 0,
        gradient_information_sum: 0.0,
        candidate_discrimination_sum: vec![0.0; dimensions.0 as usize],
        candidate_reference_sum: vec![0.0; dimensions.0 as usize],
        candidate_gradient_sum: vec![0.0; dimensions.0 as usize],
        candidate_minimum_altitude_m: vec![f32::INFINITY; dimensions.0 as usize],
        candidate_valid: vec![true; dimensions.0 as usize],
        one_time_preparation_ms: 0.0,
        preprocessing_ms: 0.0,
        command_submission_ms: 0.0,
        reduction_ms: 0.0,
        verification_ms: 0.0,
        gpu_completion_map_ms: 0.0,
        warm_evaluation_ms: 0.0,
        certified_warm_evaluation_ms: 0.0,
        certified_full_pass_ms: 0.0,
        first_tile_ms: 0.0,
        dispatch_count: 0,
        forward_kernel_evaluations: 0,
        spectral_element_count: 0,
    });
    planning.status = format!(
        "{} batch planning started: 0/{} evaluations, order {} -> {} -> {}.",
        planning.workload_profile.label(),
        planning
            .batch_job
            .as_ref()
            .map_or(0, |job| job.total_evaluations),
        method_order[0].planning_label(),
        method_order[1].planning_label(),
        method_order[2].planning_label(),
    );
}

fn planning_method_order(rotation: usize) -> [ActiveGravityMethod; 3] {
    let mut order = [
        ActiveGravityMethod::CurvedArcEq106,
        ActiveGravityMethod::MmfftCompressed,
        ActiveGravityMethod::Fmm,
    ];
    order.rotate_left(rotation % 3);
    order
}
