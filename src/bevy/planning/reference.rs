fn reference_key(
    target: DVec3,
    batch: &PlanningCandidateBatch,
    model: u32,
) -> (u64, u64, u32, [u32; 3]) {
    (
        batch.basis_hash,
        batch.density_model_hash,
        model,
        [
            (target.x as f32).to_bits(),
            (target.y as f32).to_bits(),
            (target.z as f32).to_bits(),
        ],
    )
}

/// Reference points requested from the Worker per round trip.
/// First verifies every sample in a tile (up to 8×241); a small batch turns
/// each GPU packet into dozens of Worker round-trips. Larger batches keep the
/// numerical contract identical while amortizing source uploads.
const PLANNING_REFERENCE_TARGETS_PER_REQUEST: usize = 256;

/// Drives the independent f64 reference for one GPU packet through the
/// reference channel. Returns `true` once every requested target has a cached
/// field (finite or `NaN` for an invalid one).
fn prepare_planning_references(
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    cache: &mut PlanningReferenceCache,
    channel: &crate::cpp_backend::BackendReferenceChannel,
) -> bool {
    if packet.request.method == Some(ActiveGravityMethod::FrequencyDomain) {
        return prepare_frequency_domain_reference(batch, packet, cache);
    }
    let identity = (
        batch.basis_hash,
        batch.density_model_hash,
        batch.sample_hash,
    );
    if cache.identity != Some(identity) {
        cache.fields.clear();
        cache.identity = Some(identity);
        cache.packet_id = None;
    }
    if cache.packet_id != Some(packet.request.request_id) {
        cache.packet_id = Some(packet.request.request_id);
        cache.target_indices = packet.state_indices.clone();
        cache.target_cursor = 0;
        // A request issued for the previous packet must not be applied to
        // this one even if its keys happen to overlap.
        if cache.pending.take().is_some() {
            channel.reset();
        }
    }
    let global_start =
        packet.request.candidate_start as usize * batch.samples_per_candidate as usize;
    let density_model = packet.request.density_model;

    if let Some(pending) = cache.pending.take() {
        match channel.take() {
            None if !channel.is_idle() => {
                cache.pending = Some(pending);
                return false;
            }
            // An experiment reset discarded the request; it is re-issued below.
            None => {}
            Some(delivered)
                if delivered.snapshot == pending.snapshot
                    && delivered.snapshot.epoch == batch.capture_epoch =>
            {
                let chunk_count = pending
                    .source_count
                    .div_ceil(PLANNING_REFERENCE_SOURCE_CHUNK as usize);
                let points = pending.keys.len() * PLANNING_REFERENCE_STENCIL;
                let records = delivered
                    .result
                    .ok()
                    .filter(|values| values.len() == chunk_count * points * 4)
                    .map(|values| values.as_chunks::<4>().0.to_vec());
                for (point, key) in pending.keys.iter().enumerate() {
                    let value = records
                        .as_deref()
                        .and_then(|records| {
                            accumulate_planning_reference((0..chunk_count).map(|chunk| {
                                let start = chunk * points + point * PLANNING_REFERENCE_STENCIL;
                                &records[start..start + PLANNING_REFERENCE_STENCIL]
                            }))
                        })
                        .unwrap_or((DVec3::NAN, DMat3::NAN));
                    cache.fields.insert(*key, value);
                }
            }
            // A mismatching answer belongs to a superseded request: drop it.
            Some(_) => {}
        }
    }

    let mut keys = Vec::new();
    let mut targets = Vec::new();
    let mut index = cache.target_cursor;
    let mut cursor_settled = false;
    while index < cache.target_indices.len() && keys.len() < PLANNING_REFERENCE_TARGETS_PER_REQUEST
    {
        let state_index = global_start + cache.target_indices[index] as usize;
        let Some(state) = batch.states.get(state_index) else {
            return true;
        }; // reduction rejects malformed output
        let target = state.body_position().as_dvec3();
        let key = reference_key(target, batch, density_model);
        index += 1;
        if cache.fields.contains_key(&key) {
            if !cursor_settled {
                cache.target_cursor = index;
            }
            continue;
        }
        cursor_settled = true;
        if keys.contains(&key) {
            continue;
        }
        if !push_planning_reference_stencil(target, &mut targets) {
            cache.fields.insert(key, (DVec3::NAN, DMat3::NAN));
            continue;
        }
        keys.push(key);
    }
    if keys.is_empty() {
        return index >= cache.target_indices.len();
    }
    let row = density_model as usize * 56;
    let sources = batch
        .density_models
        .get(row..row + 56)
        .and_then(|densities| planning_reference_sources(&batch.basis_records, densities));
    let Some(sources) = sources else {
        for key in keys {
            cache.fields.insert(key, (DVec3::NAN, DMat3::NAN));
        }
        return false;
    };
    cache.next_request_id = cache.next_request_id.wrapping_add(1).max(1);
    let snapshot = crate::cpp_backend::BackendReferenceSnapshot {
        request_id: packet.request.request_id.rotate_left(20) ^ cache.next_request_id,
        epoch: batch.capture_epoch,
    };
    match crate::cpp_backend::request_reference_sources(
        channel,
        snapshot,
        &sources,
        &targets,
        PLANNING_REFERENCE_SOURCE_CHUNK,
    ) {
        Ok(true) => {
            cache.pending = Some(PendingPlanningReference {
                snapshot,
                keys,
                source_count: sources.len(),
            });
        }
        // Worker busy or not ready yet: try again next frame.
        Ok(false) => {}
        Err(_) => {
            for key in keys {
                cache.fields.insert(key, (DVec3::NAN, DMat3::NAN));
            }
        }
    }
    false
}

fn prepare_frequency_domain_reference(
    batch: &PlanningCandidateBatch,
    packet: &PlanningGpuPacket,
    cache: &mut PlanningReferenceCache,
) -> bool {
    let identity = (
        batch.basis_hash,
        batch.density_model_hash,
        packet.request.density_model,
    );
    if cache.frequency_domain_identity != Some(identity) {
        let quadrature = (0..EQ184_QUADRATURE_COUNT)
            .map(|index| {
                let (wave_vector, volume_weight) =
                    eq184_quadrature_node(index, f64::from(batch.frequency_domain_source_radius))?;
                let coefficient =
                    f64::from(crate::interface::components::G) * 4.0 * std::f64::consts::PI
                        / std::f64::consts::TAU.powi(3)
                        * volume_weight
                        / wave_vector.length_squared().max(1.0e-18);
                Some((wave_vector, coefficient.clamp(-1.0e20, 1.0e20)))
            })
            .collect::<Option<Vec<_>>>();
        let Some(quadrature) = quadrature else {
            return false;
        };
        cache.frequency_domain_identity = Some(identity);
        cache.frequency_domain_quadrature = quadrature;
        cache.frequency_domain_density_spectrum = Vec::new();
        cache.frequency_domain_partial_density_spectrum =
            vec![Complex64::new(0.0, 0.0); crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT];
        cache.frequency_domain_source_cursor = 0;
        cache.frequency_domain_observations.clear();
    }

    let density_row = packet.request.density_model as usize * 56;
    let Some(densities) = batch.density_models.get(density_row..density_row + 56) else {
        return false;
    };
    let started = bevy::platform::time::Instant::now();
    while cache.frequency_domain_source_cursor < batch.basis_records.len() {
        let end = (cache.frequency_domain_source_cursor + 2_048).min(batch.basis_records.len());
        for source in &batch.basis_records[cache.frequency_domain_source_cursor..end] {
            let voxel_density = f64::from(
                *densities
                    .get(source.voxel_index as usize)
                    .unwrap_or(&f32::NAN),
            );
            let volume_density = f64::from(source.position_volume[3]) * voxel_density;
            let position = DVec3::new(
                f64::from(source.position_volume[0]),
                f64::from(source.position_volume[1]),
                f64::from(source.position_volume[2]),
            );
            if !position.is_finite() || !volume_density.is_finite() {
                return false;
            }
            for (spectrum, (wave_vector, _)) in cache
                .frequency_domain_partial_density_spectrum
                .iter_mut()
                .zip(&cache.frequency_domain_quadrature)
            {
                *spectrum += Complex64::from_polar(volume_density, -wave_vector.dot(position));
            }
        }
        cache.frequency_domain_source_cursor = end;
        // Planning already owns the exclusive compute slot; a slightly longer
        // slice finishes the spectrum in far fewer frames without changing
        // the accumulated operator.
        if started.elapsed().as_secs_f64() >= 0.008 {
            return false;
        }
    }
    if cache.frequency_domain_density_spectrum.is_empty() {
        cache.frequency_domain_density_spectrum =
            std::mem::take(&mut cache.frequency_domain_partial_density_spectrum);
    }
    true
}

fn direct_planning_reference_cached(
    target: DVec3,
    batch: &PlanningCandidateBatch,
    density_model: u32,
    cache: &mut PlanningReferenceCache,
) -> (DVec3, DMat3) {
    // Preflight populated every requested reference in bounded frame slices.
    // Missing entries fail accuracy; never hide a synchronous full-source solve
    // here, and never validate one GPU algorithm against its own approximation.
    cache
        .fields
        .get(&reference_key(target, batch, density_model))
        .copied()
        .unwrap_or((DVec3::NAN, DMat3::NAN))
}

/// Independent f64 reference for one discrete frequency-domain observation.
/// This mirrors the shader's rho-hat(k) * T_gamma(s,k) reciprocal-space
/// operator, including its quadrature, phase convention, Laplace attenuation,
/// Newton multiplier, and Jacobian column layout. Callers time-slice the
/// outer reduction (~8 ms) so one GPU callback cannot run unbounded here.
fn frequency_domain_reference_integral(
    batch: &PlanningCandidateBatch,
    candidate_index: usize,
    observation_index: usize,
    cache: &mut PlanningReferenceCache,
) -> (DVec3, DMat3) {
    let key = (candidate_index, observation_index);
    if let Some(result) = cache.frequency_domain_observations.get(&key) {
        return *result;
    }
    let samples = batch.samples_per_candidate as usize;
    let start = candidate_index.saturating_mul(samples);
    if start + samples > batch.states.len() {
        return (DVec3::NAN, DMat3::NAN);
    }
    if cache.frequency_domain_density_spectrum.len()
        != crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT
        || cache.frequency_domain_quadrature.len()
            != crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT
    {
        return (DVec3::NAN, DMat3::NAN);
    }
    let laplace_frequency = eq184_laplace_sigma(observation_index, samples);
    let mut result_field = DVec3::ZERO;
    let mut result_gradient = DMat3::ZERO;
    for (index, (wave_vector, coefficient)) in cache.frequency_domain_quadrature.iter().enumerate()
    {
        let trajectory = (0..samples).try_fold(Complex64::new(0.0, 0.0), |sum, sample_index| {
            let sample = batch.states[start + sample_index];
            let previous = if sample_index > 0 {
                batch.states[start + sample_index - 1]
            } else {
                sample
            };
            let next = if sample_index + 1 < samples {
                batch.states[start + sample_index + 1]
            } else {
                sample
            };
            Some(
                sum + eq184_trajectory_term(
                    *wave_vector,
                    sample.body_position().as_dvec3(),
                    f64::from(previous.position_time[3]),
                    f64::from(sample.position_time[3]),
                    f64::from(next.position_time[3]),
                    sample_index,
                    samples,
                    laplace_frequency,
                )?,
            )
        });
        let Some(trajectory) = trajectory else {
            return (DVec3::NAN, DMat3::NAN);
        };
        let product = cache.frequency_domain_density_spectrum[index] * trajectory;
        result_field += -*coefficient * product.im * *wave_vector;
        let hessian_scale = -*coefficient * product.re;
        let jacobian_x = hessian_scale * *wave_vector * wave_vector.x;
        let jacobian_y = hessian_scale * *wave_vector * wave_vector.y;
        let jacobian_z = hessian_scale * *wave_vector * wave_vector.z;
        result_gradient += DMat3::from_cols(jacobian_x, jacobian_y, jacobian_z);
    }
    let result = (result_field, result_gradient);
    if result_field.is_finite() && result_gradient.is_finite() {
        cache.frequency_domain_observations.insert(key, result);
    }
    result
}
