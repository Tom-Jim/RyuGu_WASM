pub(crate) fn build_planning_eq106_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    // The first 56 vec4 records are immutable `(source_start, source_count)`
    // ranges. The next 56 contain the current model's density. Geometry and
    // volume stay in `primary`, so switching models is O(56), not O(Nsource).
    let weighted_acceleration_coefficient = batch
        .eq106_compression_acceleration_coefficients
        .iter()
        .zip(densities)
        .try_fold(0.0_f64, |sum, (moment, density)| {
            let contribution = *moment * f64::from(*density);
            contribution.is_finite().then_some(sum + contribution)
        })?;
    let compression_relative_bound = eq106_compression_relative_bound(
        batch,
        weighted_acceleration_coefficient,
        *batch.density_model_masses.get(model as usize)?,
    );
    let worst_acceleration_coefficient = batch
        .density_models
        .as_chunks::<56>()
        .0
        .iter()
        .map(|row| {
            batch
                .eq106_compression_acceleration_coefficients
                .iter()
                .zip(row)
                .map(|(coefficient, density)| coefficient * f64::from(*density))
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    let compression_certified = eq106_compression_relative_bound(
        batch,
        worst_acceleration_coefficient,
        batch
            .density_model_masses
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
    )
        .is_some_and(|bound| bound <= 0.25 * f64::from(PLANNING_GRAVITY_ERROR_LIMIT));
    let (primary, ranges, certified_acceleration_coefficient) = if compression_certified {
        (
            Arc::clone(&batch.eq106_volume_source_bytes),
            batch.eq106_voxel_source_ranges.to_vec(),
            weighted_acceleration_coefficient,
        )
    } else {
        let (bytes, ranges) = uncompressed_eq106_geometry(batch)?;
        (bytes, ranges.to_vec(), 0.0)
    };
    let mut metadata = Vec::with_capacity(113 * 16);
    if batch.eq106_voxel_source_ranges.len() != 56 {
        return None;
    }
    for range in ranges.iter().copied() {
        let record = [range[0] as f32, range[1] as f32, 0.0, 0.0];
        metadata.extend_from_slice(bytemuck::cast_slice(&record));
    }
    for density in densities {
        let record = [*density, 0.0, 0.0, 0.0];
        metadata.extend_from_slice(bytemuck::cast_slice(&record));
    }
    let compression = [
        0.0_f32,
        certified_acceleration_coefficient as f32,
        compression_relative_bound.unwrap_or(f64::INFINITY) as f32,
        if compression_certified { 1.0 } else { 0.0 },
    ];
    metadata.extend_from_slice(bytemuck::cast_slice(&compression));
    let total_mass = *batch.density_model_masses.get(model as usize)?;
    if primary.is_empty() || !primary.len().is_multiple_of(16) {
        return None;
    }
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return None;
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::CurvedArcEq106),
        density_model: model,
        primary,
        secondary: Arc::from(metadata),
        item_count: u32::try_from(ranges.iter().map(|range| u64::from(range[1])).sum::<u64>())
            .ok()?,
        secondary_count: 56,
        total_mass: total_mass as f32,
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
        ..default()
    })
}

fn eq106_compression_relative_bound(
    batch: &PlanningCandidateBatch,
    density_weighted_acceleration_coefficient: f64,
    total_mass: f64,
) -> Option<f64> {
    if !density_weighted_acceleration_coefficient.is_finite()
        || density_weighted_acceleration_coefficient < 0.0
        || !total_mass.is_finite()
        || total_mass <= 0.0
    {
        return None;
    }
    let target_radius = batch
        .states
        .iter()
        .chain(batch.reference_states.iter())
        .map(|state| f64::from(state.body_position().length()))
        .filter(|radius| radius.is_finite())
        .fold(f64::INFINITY, f64::min);
    let source_radius = f64::from(batch.eq106_source_radius);
    let separation = target_radius - source_radius;
    if !target_radius.is_finite() || separation <= 0.0 {
        return None;
    }
    // The per-cluster 1/r^6 factors were accumulated against the complete
    // trajectory tube during compression. The fourth-order multivariate
    // remainder for r/|r|^3 is bounded conservatively by the factor 64
    // (the fifth derivative of 1/r divided by 4!, with vector norm margin).
    let absolute_error = 64.0 * G as f64 * density_weighted_acceleration_coefficient;
    // Positive mass inside the enclosing sphere has this strictly inward
    // radial lower bound, avoiding an empirical field-scale estimate.
    let field_lower_bound = G as f64 * total_mass * separation
        / (target_radius + source_radius).powi(3);
    (field_lower_bound > 0.0).then_some(absolute_error / field_lower_bound)
}

fn uncompressed_eq106_geometry(
    batch: &PlanningCandidateBatch,
) -> Option<(Arc<[u8]>, [[u32; 2]; 56])> {
    if batch.basis_records.is_empty() || batch.basis_records.len() > u32::MAX as usize {
        return None;
    }
    let mut bytes = Vec::with_capacity(batch.basis_records.len() * 16);
    let mut ranges = [[0_u32; 2]; 56];
    let mut cursor = 0_usize;
    for (voxel, range) in ranges.iter_mut().enumerate() {
        let start = cursor;
        while cursor < batch.basis_records.len()
            && batch.basis_records[cursor].voxel_index as usize == voxel
        {
            bytes.extend_from_slice(bytemuck::cast_slice(
                &batch.basis_records[cursor].position_volume,
            ));
            cursor += 1;
        }
        *range = [start as u32, (cursor - start) as u32];
    }
    (cursor == batch.basis_records.len() && ranges.iter().all(|range| range[1] > 0))
        .then_some((Arc::from(bytes), ranges))
}

#[cfg(test)]
mod planning_voxel_spectrum_payload_tests {
    use super::*;

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn density_switch_reuses_geometry_and_uploads_only_56_weights() {
        let mut batch = PlanningCandidateBatch::default();
        let records = (0..56)
            .map(|voxel| PlanningBasisRecord {
                position_volume: [voxel as f32, 0.0, 0.0, voxel as f32 + 1.0],
                voxel_index: voxel,
            })
            .collect::<Vec<_>>();
        let mut geometry = Vec::with_capacity(56 * 16);
        for record in &records {
            for value in record.position_volume {
                geometry.extend_from_slice(&value.to_le_bytes());
            }
        }
        batch.basis_records = Arc::from(records);
        batch.eq106_volume_source_bytes = Arc::from(geometry);
        batch.eq106_voxel_source_ranges =
            Arc::from((0..56).map(|voxel| [voxel, 1]).collect::<Vec<[u32; 2]>>());
        batch.eq106_compression_acceleration_coefficients = Arc::from([0.0_f64; 56]);
        batch.eq106_source_radius = 100.0;
        batch.states = Arc::from([PlanningCandidateState {
            position_time: [1_000.0, 0.0, 0.0, 0.0],
            ..default()
        }]);
        batch.density_models = Arc::from(
            (0..112)
                .map(|index| if index < 56 { 2.0 } else { 3.0 })
                .collect::<Vec<f32>>(),
        );
        batch.density_model_masses = Arc::from([1.0_f64, 2.0]);

        let first = build_planning_eq106_payload(&batch, 0, 7).unwrap();
        let second = build_planning_eq106_payload(&batch, 1, 8).unwrap();
        assert!(Arc::ptr_eq(
            &first.primary,
            &batch.eq106_volume_source_bytes
        ));
        assert!(Arc::ptr_eq(&first.primary, &second.primary));
        assert_eq!(first.secondary.len(), 113 * 16);
        assert_eq!(second.secondary.len(), 113 * 16);
        for voxel in 0..56 {
            let range_offset = voxel * 16;
            assert_eq!(read_f32(&first.secondary, range_offset), voxel as f32);
            assert_eq!(read_f32(&first.secondary, range_offset + 4), 1.0);
            let density_offset = (56 + voxel) * 16;
            assert_eq!(read_f32(&first.secondary, density_offset), 2.0);
            assert_eq!(read_f32(&second.secondary, density_offset), 3.0);
        }
    }
}
