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
    let mut metadata = Vec::with_capacity(112 * 16);
    if batch.eq106_voxel_source_ranges.len() != 56 {
        return None;
    }
    for range in batch.eq106_voxel_source_ranges.iter().copied() {
        let record = [range[0] as f32, range[1] as f32, 0.0, 0.0];
        metadata.extend_from_slice(bytemuck::cast_slice(&record));
    }
    for density in densities {
        let record = [*density, 0.0, 0.0, 0.0];
        metadata.extend_from_slice(bytemuck::cast_slice(&record));
    }
    let total_mass = *batch.density_model_masses.get(model as usize)?;
    if batch.eq106_volume_source_bytes.len() != batch.basis_records.len() * 16 {
        return None;
    }
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return None;
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::CurvedArcEq106),
        density_model: model,
        primary: Arc::clone(&batch.eq106_volume_source_bytes),
        secondary: Arc::from(metadata),
        item_count: batch.basis_records.len() as u32,
        secondary_count: 56,
        total_mass: total_mass as f32,
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
        ..default()
    })
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
        assert_eq!(first.secondary.len(), 112 * 16);
        assert_eq!(second.secondary.len(), 112 * 16);
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
