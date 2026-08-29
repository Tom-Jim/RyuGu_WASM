#[derive(Default)]
pub(crate) struct PlanningMmfftWorkspace {
    levels: Vec<MmfftLevelWorkspace>,
}

pub(crate) fn build_planning_mmfft_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningMmfftWorkspace,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let mut program_setup_ms = 0.0;
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    if batch.basis_records.is_empty() {
        return None;
    }
    if cache.levels.len() != LEVEL_GRID_SIZES.len() {
        let program_setup_started = bevy::platform::time::Instant::now();
        cache.levels = LEVEL_GRID_SIZES
            .into_iter()
            .zip(LEVEL_HALF_EXTENTS)
            .map(|(n, half)| MmfftLevelWorkspace::new(n, half))
            .collect();
        program_setup_ms = program_setup_started.elapsed().as_secs_f64() * 1.0e3;
    }
    // Compute deposition stencils directly into the fixed FFT workspaces.
    // Caching eight heap records per quadrature point exceeded a gigabyte at
    // the 8192K endpoint and provided no arithmetic reduction across grids.
    let total_mass = batch.basis_records.iter().try_fold(0.0_f64, |sum, record| {
        let density = f64::from(*densities.get(record.voxel_index as usize)?);
        let mass = f64::from(record.position_volume[3]) * density;
        (mass.is_finite() && mass > 0.0).then_some(sum + mass)
    })?;
    let sample_count = LEVEL_GRID_SIZES.iter().map(|n| n.pow(3)).sum::<usize>();
    let mut bytes = Vec::with_capacity(sample_count.div_ceil(2) * 4);
    let mut grid_scales = [1.0_f32; 2];
    for (level, workspace) in cache.levels.iter_mut().enumerate() {
        grid_scales[level] = append_compressed_potential_level(
            &mut bytes,
            workspace.build_from_basis(&batch.basis_records, densities)?,
        );
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::MmfftCompressed),
        density_model: model,
        primary: Arc::from(bytes),
        item_count: sample_count as u32,
        grid_sizes: LEVEL_GRID_SIZES.map(|value| value as u32),
        half_extents: LEVEL_HALF_EXTENTS.map(|value| value as f32),
        grid_scales,
        total_mass: total_mass as f32,
        geometry_basis_preparation_ms: 0.0,
        density_payload_preparation_ms: (started.elapsed().as_secs_f64() * 1.0e3
            - program_setup_ms)
            .max(0.0),
        ..default()
    })
}
