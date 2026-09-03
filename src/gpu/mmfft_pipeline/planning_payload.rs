// Planning keeps only metadata on the CPU. All 56 density columns, both
// padded FFT convolutions and every RHS combination live in render-world GPU
// buffers (planning_fft_gpu.rs). The CPU FFT workspace below remains an
// independent small-grid numerical oracle / legacy runtime implementation.
#[derive(Default)]
pub(crate) struct PlanningMmfftWorkspace {
    batch_id: Option<u64>,
    half_extents: [f64; 2],
}

pub(crate) fn build_planning_mmfft_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningMmfftWorkspace,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let densities = batch.density_models.get(model as usize * 56..(model as usize + 1) * 56)?;
    if batch.basis_records.is_empty()
        || densities.iter().any(|density| !density.is_finite() || *density < 0.0)
    { return None; }
    let mut geometry_ms = 0.0;
    if cache.batch_id != Some(batch.batch_id) {
        cache.half_extents = planning_fft_half_extents(batch)?;
        cache.batch_id = Some(batch.batch_id);
        geometry_ms = started.elapsed().as_secs_f64() * 1.0e3;
    }
    let total_mass = *batch.density_model_masses.get(model as usize)?;
    if !total_mass.is_finite() || total_mass <= 0.0 { return None; }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::MmfftCompressed),
        density_model: model,
        // No potential grid upload: the render-world GPU builder provides it.
        item_count: LEVEL_GRID_SIZES.iter().map(|n| n.pow(3)).sum::<usize>() as u32,
        grid_sizes: LEVEL_GRID_SIZES.map(|value| value as u32),
        half_extents: cache.half_extents.map(|value| value as f32),
        grid_scales: [1.0; 2],
        total_mass: total_mass as f32,
        geometry_basis_preparation_ms: geometry_ms,
        density_payload_preparation_ms: (started.elapsed().as_secs_f64() * 1.0e3 - geometry_ms).max(0.0),
        ..default()
    })
}

fn planning_fft_half_extents(batch: &PlanningCandidateBatch) -> Option<[f64; 2]> {
    // Keep the 64^3 / 16^3 FFT sizes. Fit every source/target, with 4.5 cells
    // of padding (including a half-cell safety margin against f32 rounding).
    if !batch.frequency_domain_source_radius.is_finite() { return None; }
    let mut extent = f64::from(batch.frequency_domain_source_radius).max(1.0);
    for state in batch.states.iter() {
        for value in state.body_position().to_array() {
            if !value.is_finite() { return None; }
            extent = extent.max(f64::from(value.abs()));
        }
    }
    let inner = extent / (1.0 - 9.0 / LEVEL_GRID_SIZES[0] as f64);
    Some([inner, 4.0 * inner])
}

#[cfg(test)]
mod planning_potential_tests {
    use super::*;

    #[test]
    fn cached_unit_potentials_match_independent_density_convolution() {
        // Test the linear basis cache against the former per-RHS convolution,
        // with off-grid sources, two populated columns and empty voxel columns.
        let records = [
            PlanningBasisRecord { position_volume: [-1.3, 0.4, 0.8, 2.0], voxel_index: 0 },
            PlanningBasisRecord { position_volume: [0.6, -1.1, 1.4, 0.7], voxel_index: 55 },
            PlanningBasisRecord { position_volume: [0.2, 0.9, -0.3, 1.1], voxel_index: 0 },
        ];
        let mut workspace = MmfftLevelWorkspace::new(8, 8.0);
        let basis = workspace.unit_density_potentials(&records).unwrap();
        for (first, last) in [(1.0_f32, 1.0_f32), (0.25, 3.7), (2.1, 0.01)] {
            let mut densities = [1.0_f32; 56];
            densities[0] = first;
            densities[55] = last;
            let reference = workspace.build_from_basis(&records, &densities).unwrap();
            for (index, sample) in reference.iter().enumerate() {
                let mixed = f64::from(first) * basis[index] + f64::from(last) * basis[55 * 8usize.pow(3) + index];
                let expected = f64::from(sample[3]);
                assert!((mixed - expected).abs() <= expected.abs() * 2.0e-6);
            }
        }
    }

    #[test]
    fn fitted_grid_covers_sources_and_targets_with_full_stencil_margin() {
        for distance in [620.0_f32, 2000.0, 16000.0] {
            let mut batch = PlanningCandidateBatch {
                frequency_domain_source_radius: 650.0,
                basis_records: Arc::from(vec![PlanningBasisRecord {
                    position_volume: [-500.0, 400.0, 50.0, 1.0],
                    voxel_index: 0,
                }]),
                states: Arc::from(vec![PlanningCandidateState {
                    position_time: [distance, -distance, 0.0, 0.0],
                    ..Default::default()
                }]),
                ..Default::default()
            };
            let half = planning_fft_half_extents(&batch).unwrap()[0] as f32;
            let spacing = 2.0 * half / LEVEL_GRID_SIZES[0] as f32;
            assert!(distance < half - 4.0 * spacing);
            assert!(500.0 < half - 4.0 * spacing);
            batch.states = Arc::from(vec![PlanningCandidateState {
                position_time: [f32::NAN, 0.0, 0.0, 0.0],
                ..Default::default()
            }]);
            assert!(planning_fft_half_extents(&batch).is_none());
        }
    }

}
