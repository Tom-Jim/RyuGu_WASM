fn candidate_perturbation_parameters(candidate: u32, candidate_count: u32) -> (f32, f32, f32, f32) {
    let golden = 0.618_033_95_f32;
    let radial_fraction = ((candidate as f32 + 0.5) / candidate_count.max(1) as f32).sqrt();
    // Reserve part of the perturbation radius for differential-force drift over
    // the complete propagated trajectory.
    let radius = PLANNING_PERTURBATION_RADIUS_METERS
        * PLANNING_INITIAL_PERTURBATION_FRACTION
        * radial_fraction;
    let phase = std::f32::consts::TAU * ((candidate as f32 * golden).fract());
    let harmonic = 1.0 + (candidate % 5) as f32;
    let phase_rate = ((candidate.wrapping_mul(747_796_405) ^ 2_891_336_453) as f32) * f32::EPSILON;
    (radius, phase, harmonic, phase_rate)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn uniform_unit_random(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
}

/// Generates independent voxel densities from a uniform distribution and
/// then applies one scalar normalization per model. The spatial randomness is
/// therefore preserved while every row represents exactly the same asteroid
/// mass to f32 storage precision.
fn uniform_random_equal_mass_models(
    voxels: &[InvertedDensityVoxel],
    target_mass: f64,
    model_count: u32,
    seed: u64,
) -> Option<(Vec<f32>, Vec<f64>)> {
    if voxels.is_empty()
        || model_count == 0
        || !target_mass.is_finite()
        || target_mass <= 0.0
        || voxels
            .iter()
            .any(|voxel| !voxel.volume.is_finite() || voxel.volume <= 0.0)
    {
        return None;
    }
    let total_volume = voxels
        .iter()
        .map(|voxel| f64::from(voxel.volume))
        .sum::<f64>();
    let mean_density = target_mass / total_volume;
    let correction_index = voxels
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.volume.total_cmp(&right.volume))?
        .0;
    let mut random_state = seed;
    let mut models = Vec::with_capacity(model_count as usize * voxels.len());
    let mut masses = Vec::with_capacity(model_count as usize);
    for _ in 0..model_count {
        let mut row = voxels
            .iter()
            .map(|_| (mean_density * (0.35 + 1.30 * uniform_unit_random(&mut random_state))) as f32)
            .collect::<Vec<_>>();
        let mass = voxels
            .iter()
            .zip(&row)
            .map(|(voxel, density)| voxel.volume as f64 * f64::from(*density))
            .sum::<f64>();
        let scale = target_mass / mass.max(f64::MIN_POSITIVE);
        for density in &mut row {
            *density = (f64::from(*density) * scale) as f32;
        }
        // Correct the f32 rounding residual in the largest voxel. Two passes
        // are enough to reach the representable mass nearest to target_mass.
        for _ in 0..2 {
            let corrected_mass = voxels
                .iter()
                .zip(&row)
                .map(|(voxel, density)| voxel.volume as f64 * f64::from(*density))
                .sum::<f64>();
            let correction =
                (target_mass - corrected_mass) / f64::from(voxels[correction_index].volume);
            row[correction_index] = (f64::from(row[correction_index]) + correction) as f32;
        }
        let final_mass = voxels
            .iter()
            .zip(&row)
            .map(|(voxel, density)| voxel.volume as f64 * f64::from(*density))
            .sum::<f64>();
        if row
            .iter()
            .any(|density| !density.is_finite() || *density <= 0.0)
            || ((final_mass - target_mass) / target_mass).abs() > 2.0e-7
        {
            return None;
        }
        models.extend(row);
        masses.push(final_mass);
    }
    Some((models, masses))
}

fn hash_reference_samples(samples: &[TrajectoryInversionKnot]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for sample in samples {
        for value in sample
            .position
            .to_array()
            .into_iter()
            .chain(sample.velocity.to_array())
            .chain(sample.body_rotation.to_array())
            .chain([sample.simulation_time_seconds as f32])
        {
            hash = mix_hash(hash, u64::from(value.to_bits()));
        }
    }
    hash
}

fn hash_candidate_states(states: &[PlanningCandidateState]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for state in states {
        for value in state
            .position_time
            .into_iter()
            .chain(state.velocity_distance)
            .chain(state.body_rotation)
        {
            hash = mix_hash(hash, u64::from(value.to_bits()));
        }
        for value in state.identity {
            hash = mix_hash(hash, u64::from(value));
        }
    }
    hash
}

fn hash_f32_iter(values: impl IntoIterator<Item = f32>) -> u64 {
    values
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            mix_hash(hash, u64::from(value.to_bits()))
        })
}

fn mix_hash(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
}
