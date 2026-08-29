pub(crate) struct PlanningEq106Workspace {
    batch_id: u64,
    primary: Arc<[u8]>,
    ranges: [[u32; 2]; 56],
}

impl Default for PlanningEq106Workspace {
    fn default() -> Self {
        Self {
            batch_id: 0,
            primary: Arc::from([]),
            ranges: [[0_u32; 2]; 56],
        }
    }
}

pub(crate) fn build_planning_eq106_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningEq106Workspace,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let mut geometry_basis_preparation_ms = 0.0;
    if cache.batch_id != batch.batch_id || cache.primary.is_empty() {
        let one_time_started = bevy::platform::time::Instant::now();
        let (primary, ranges) = uncompressed_eq106_geometry(batch)?;
        cache.batch_id = batch.batch_id;
        cache.primary = primary;
        cache.ranges = ranges;
        geometry_basis_preparation_ms = one_time_started.elapsed().as_secs_f64() * 1.0e3;
    }
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    // The first 56 vec4 records are immutable `(source_start, source_count)`
    // ranges. The next 56 contain the current model's density. Geometry and
    // volume stay in `primary`, so switching models is O(56), not O(Nsource).
    let primary = Arc::clone(&cache.primary);
    let ranges = cache.ranges;
    let mut metadata = Vec::with_capacity(113 * 16);
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
        0.0,
        0.0,
        0.0,
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
        geometry_basis_preparation_ms,
        density_payload_preparation_ms: (started.elapsed().as_secs_f64() * 1.0e3
            - geometry_basis_preparation_ms)
            .max(0.0),
        ..default()
    })
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
