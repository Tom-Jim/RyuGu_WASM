pub(crate) struct PlanningMmfftWorkspace {
    batch_id: u64,
    levels: Vec<MmfftLevelWorkspace>,
}

impl Default for PlanningMmfftWorkspace {
    fn default() -> Self {
        Self {
            batch_id: 0,
            levels: Vec::new(),
        }
    }
}

pub(crate) fn build_planning_mmfft_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningMmfftWorkspace,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    let records = batch
        .basis_records
        .iter()
        .filter_map(|record| {
            let density = f64::from(*densities.get(record.voxel_index as usize)?);
            let position = DVec3::new(
                f64::from(record.position_volume[0]),
                f64::from(record.position_volume[1]),
                f64::from(record.position_volume[2]),
            );
            let mass = f64::from(record.position_volume[3]) * density;
            (position.is_finite() && mass.is_finite() && mass > 0.0).then_some((position, mass))
        })
        .collect::<Vec<_>>();
    if records.is_empty() {
        return None;
    }
    if cache.batch_id != batch.batch_id || cache.levels.len() != LEVEL_GRID_SIZES.len() {
        cache.batch_id = batch.batch_id;
        cache.levels = LEVEL_GRID_SIZES
            .into_iter()
            .zip(LEVEL_HALF_EXTENTS)
            .map(|(n, half)| MmfftLevelWorkspace::new(n, half))
            .collect();
    }
    let mut bytes =
        Vec::with_capacity(LEVEL_GRID_SIZES.iter().map(|n| n.pow(3)).sum::<usize>() * 16);
    for workspace in &mut cache.levels {
        for sample in workspace.build(&records) {
            for value in *sample {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::MmfftCompressed),
        density_model: model,
        primary: Arc::from(bytes),
        item_count: LEVEL_GRID_SIZES.iter().map(|n| n.pow(3)).sum::<usize>() as u32,
        grid_sizes: LEVEL_GRID_SIZES.map(|value| value as u32),
        half_extents: LEVEL_HALF_EXTENTS.map(|value| value as f32),
        total_mass: records.iter().map(|record| record.1).sum::<f64>() as f32,
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
        ..default()
    })
}
