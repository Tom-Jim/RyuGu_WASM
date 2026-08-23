pub(crate) fn build_planning_eq106_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    let mut bytes = Vec::with_capacity(batch.basis_records.len() * 16);
    let mut total_mass = 0.0_f64;
    for record in batch.basis_records.iter() {
        let density = f64::from(*densities.get(record.voxel_index as usize)?);
        let mass = f64::from(record.position_volume[3]) * density;
        total_mass += mass;
        for value in [
            record.position_volume[0],
            record.position_volume[1],
            record.position_volume[2],
            mass as f32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::CurvedArcEq106),
        density_model: model,
        primary: Arc::from(bytes),
        item_count: batch.basis_records.len() as u32,
        total_mass: total_mass as f32,
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
        ..default()
    })
}
