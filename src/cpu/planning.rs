use crate::cpu::curved_arc::AggregatedGravitySource;
use crate::cpu::inversion::build_voxel_basis_sources;
use crate::interface::components::*;
use bevy::prelude::*;
use std::sync::Arc;

pub(crate) struct PlanningBatchBuilder {
    profile: PlanningWorkloadProfile,
    run_id: u64,
    capture_id: u64,
    capture_epoch: u64,
    source_hash: u64,
    source_count: u32,
    body_radius: f32,
    candidate_count: u32,
    density_model_count: u32,
    samples_per_candidate: u32,
    next_candidate: u32,
    preparation_ms: f64,
    reference_samples: Vec<TrajectoryInversionKnot>,
    reference_states: Vec<PlanningCandidateState>,
    states: Vec<PlanningCandidateState>,
    gpu_position_bytes: Vec<u8>,
    density_models: Vec<f32>,
    density_model_masses: Vec<f64>,
    density_seed: u64,
    target_mass: f64,
    basis_records: Vec<PlanningBasisRecord>,
    reference_basis_records: Vec<PlanningBasisRecord>,
    basis_hash: u64,
}

impl PlanningBatchBuilder {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        profile: PlanningWorkloadProfile,
        run_id: u64,
        capture_id: u64,
        capture_epoch: u64,
        source_hash: u64,
        requested_source_count: u32,
        reference_knots: &[TrajectoryInversionKnot],
        voxels: &[InvertedDensityVoxel],
        source: &AggregatedGravitySource,
    ) -> Option<Self> {
        let started = bevy::platform::time::Instant::now();
        let (candidate_count, density_model_count, samples_per_candidate) = profile.dimensions();
        if candidate_count == 0 || candidate_count > PLANNING_CANDIDATE_COUNT || voxels.len() != 56
        {
            return None;
        }
        let basis = build_voxel_basis_sources(voxels, source)?;
        let reference_samples = crate::cpu::inversion::sample_frozen_trajectory_at_count(
            reference_knots,
            samples_per_candidate as usize,
        )?;
        let reference_states = central_reference_states(&reference_samples)?;
        let canonical_basis_records = basis
            .columns
            .iter()
            .enumerate()
            .flat_map(|(voxel_index, column)| {
                column.iter().map(move |source| PlanningBasisRecord {
                    position_volume: [
                        source.position.x as f32,
                        source.position.y as f32,
                        source.position.z as f32,
                        source.volume as f32,
                    ],
                    voxel_index: voxel_index as u32,
                    _padding: [0; 3],
                })
            })
            .collect::<Vec<_>>();
        let basis_records = refine_basis_records(&canonical_basis_records, requested_source_count)?;
        let basis_hash = mix_hash(basis.hash, u64::from(requested_source_count));
        let density_seed = mix_hash(
            mix_hash(source_hash, basis.hash),
            mix_hash(capture_id, 0x1060_d315_7a11_5eed),
        );
        let target_mass = source.total_mass;
        let (density_models, density_model_masses) = uniform_random_equal_mass_models(
            voxels,
            target_mass,
            density_model_count,
            density_seed,
        )?;
        Some(Self {
            profile,
            run_id,
            capture_id,
            capture_epoch,
            source_hash,
            source_count: requested_source_count,
            body_radius: source.radius as f32,
            candidate_count,
            density_model_count,
            samples_per_candidate,
            next_candidate: 0,
            preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
            reference_samples,
            reference_states,
            states: Vec::with_capacity(candidate_count as usize * samples_per_candidate as usize),
            gpu_position_bytes: Vec::with_capacity(
                candidate_count as usize * samples_per_candidate as usize * 16,
            ),
            density_models,
            density_model_masses,
            density_seed,
            target_mass,
            basis_records,
            reference_basis_records: canonical_basis_records,
            basis_hash,
        })
    }

    pub(crate) fn matches(
        &self,
        profile: PlanningWorkloadProfile,
        run_id: u64,
        capture_id: u64,
        source_hash: u64,
        requested_source_count: u32,
    ) -> bool {
        self.profile == profile
            && self.run_id == run_id
            && self.capture_id == capture_id
            && self.source_hash == source_hash
            && self.source_count == requested_source_count
    }

    pub(crate) fn advance(&mut self, candidate_budget: u32) -> bool {
        let started = bevy::platform::time::Instant::now();
        let end = (self.next_candidate + candidate_budget.max(1)).min(self.candidate_count);
        while self.next_candidate < end {
            if append_tube_candidate_states(
                self.next_candidate,
                self.candidate_count,
                &self.reference_samples,
                &mut self.states,
                &mut self.gpu_position_bytes,
            )
            .is_none()
            {
                return false;
            }
            self.next_candidate += 1;
        }
        self.preparation_ms += started.elapsed().as_secs_f64() * 1.0e3;
        true
    }

    pub(crate) fn completed_candidates(&self) -> u32 {
        self.next_candidate
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.next_candidate == self.candidate_count
    }

    pub(crate) fn finish(self) -> Option<(PlanningCandidateBatch, f64)> {
        if !self.is_complete() {
            return None;
        }
        let reference_arc_hash = hash_reference_samples(&self.reference_samples);
        let candidate_hash = hash_candidate_states(&self.states);
        let density_model_hash = hash_f32_iter(self.density_models.iter().copied());
        let sample_hash = hash_f32_iter(self.states.iter().flat_map(|state| {
            state
                .position_time
                .into_iter()
                .chain(state.velocity_distance)
                .chain(state.body_rotation)
        }));
        let batch_id = mix_hash(
            mix_hash(mix_hash(self.run_id, self.capture_id), self.capture_epoch),
            mix_hash(candidate_hash, density_model_hash),
        );
        let maximum_relative_mass_error = self
            .density_model_masses
            .iter()
            .map(|mass| ((mass - self.target_mass) / self.target_mass).abs())
            .fold(0.0_f64, f64::max);
        info!(
            target: "planning::density",
            seed = self.density_seed,
            model_count = self.density_model_count,
            voxel_count = 56,
            target_mass = self.target_mass,
            maximum_relative_mass_error,
            "generated uniformly randomized positive voxel-density models with conserved asteroid mass"
        );
        Some((
            PlanningCandidateBatch {
                batch_id,
                capture_id: self.capture_id,
                capture_epoch: self.capture_epoch,
                source_hash: self.source_hash,
                source_count: self.source_count,
                candidate_count: self.candidate_count,
                density_model_count: self.density_model_count,
                samples_per_candidate: self.samples_per_candidate,
                body_radius: self.body_radius,
                reference_states: Arc::from(self.reference_states),
                states: Arc::from(self.states),
                gpu_position_bytes: Arc::from(self.gpu_position_bytes),
                density_models: Arc::from(self.density_models),
                density_model_masses: Arc::from(self.density_model_masses),
                density_seed: self.density_seed,
                target_mass: self.target_mass,
                basis_records: Arc::from(self.basis_records),
                reference_basis_records: Arc::from(self.reference_basis_records),
                reference_arc_hash,
                candidate_hash,
                density_model_hash,
                sample_hash,
                basis_hash: self.basis_hash,
            },
            self.preparation_ms,
        ))
    }
}

/// Split each canonical aggregate into colocated equal-volume records. All
/// backends therefore see the same requested source count, geometry, density
/// assignment, and conserved asteroid mass.
fn refine_basis_records(
    canonical: &[PlanningBasisRecord],
    requested: u32,
) -> Option<Vec<PlanningBasisRecord>> {
    let requested = requested as usize;
    if canonical.is_empty() || requested < canonical.len() {
        return None;
    }
    let base = requested / canonical.len();
    let remainder = requested % canonical.len();
    let mut refined = Vec::with_capacity(requested);
    for (index, record) in canonical.iter().copied().enumerate() {
        let copies = base + usize::from(index < remainder);
        let mut split = record;
        split.position_volume[3] /= copies as f32;
        refined.extend(std::iter::repeat_n(split, copies));
    }
    (refined.len() == requested).then_some(refined)
}

fn central_reference_states(
    reference: &[TrajectoryInversionKnot],
) -> Option<Vec<PlanningCandidateState>> {
    let angular_velocity =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    reference
        .iter()
        .enumerate()
        .map(|(sample, state)| {
            if !state.position.is_finite()
                || !state.velocity.is_finite()
                || !state.body_rotation.is_finite()
                || !state.simulation_time_seconds.is_finite()
            {
                return None;
            }
            let body_position = state.body_rotation.inverse() * state.position;
            let body_velocity = state.body_rotation.inverse()
                * (state.velocity - angular_velocity.cross(state.position));
            Some(PlanningCandidateState {
                position_time: [
                    body_position.x,
                    body_position.y,
                    body_position.z,
                    state.simulation_time_seconds as f32,
                ],
                velocity_distance: [body_velocity.x, body_velocity.y, body_velocity.z, 0.0],
                body_rotation: state.body_rotation.to_array(),
                identity: [u32::MAX, sample as u32, 0, 0],
            })
        })
        .collect()
}

fn append_tube_candidate_states(
    candidate: u32,
    candidate_count: u32,
    reference: &[TrajectoryInversionKnot],
    states: &mut Vec<PlanningCandidateState>,
    gpu_position_bytes: &mut Vec<u8>,
) -> Option<()> {
    let sample_count = reference.len() as u32;
    if sample_count < 2 {
        return None;
    }
    let mut world_positions = Vec::with_capacity(sample_count as usize);
    let (radius, phase, harmonic, phase_rate) =
        candidate_tube_parameters(candidate, candidate_count);
    for sample in 0..sample_count {
        let reference_state = reference[sample as usize];
        let tangent = reference_state.velocity.normalize_or_zero();
        let normal_hint = RYUGU_SPIN_AXIS.normalize_or_zero();
        let normal = (normal_hint - tangent * normal_hint.dot(tangent)).normalize_or_zero();
        let binormal = tangent.cross(normal).normalize_or_zero();
        if tangent == Vec3::ZERO || normal == Vec3::ZERO || binormal == Vec3::ZERO {
            return None;
        }
        let normalized_time = sample as f32 / sample_count.saturating_sub(1) as f32 - 0.5;
        let angle = phase + harmonic * std::f32::consts::TAU * normalized_time + phase_rate;
        let envelope = 0.82 + 0.18 * (std::f32::consts::TAU * normalized_time + phase).cos();
        let offset_radius = (radius * envelope).min(PLANNING_TRAJECTORY_TUBE_RADIUS_METERS);
        let offset =
            normal * (offset_radius * angle.cos()) + binormal * (offset_radius * angle.sin());
        if offset.length() > PLANNING_TRAJECTORY_TUBE_RADIUS_METERS + 1.0e-4 {
            return None;
        }
        world_positions.push(reference_state.position + offset);
    }

    let angular_velocity =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let first_time = reference.first()?.simulation_time_seconds;
    for sample in 0..sample_count {
        let world_position = world_positions[sample as usize];
        let previous = world_positions[sample.saturating_sub(1) as usize];
        let next = world_positions[(sample + 1).min(sample_count - 1) as usize];
        let previous_time = reference[sample.saturating_sub(1) as usize].simulation_time_seconds;
        let next_time =
            reference[(sample + 1).min(sample_count - 1) as usize].simulation_time_seconds;
        let denominator = (next_time - previous_time) as f32;
        let world_velocity = (next - previous) / denominator.max(f32::MIN_POSITIVE);
        let reference_state = reference[sample as usize];
        let time = reference_state.simulation_time_seconds as f32;
        let rotation = reference_state.body_rotation;
        let body_position = rotation.inverse() * world_position;
        let body_velocity =
            rotation.inverse() * (world_velocity - angular_velocity.cross(world_position));
        let transverse_distance = world_position.distance(reference_state.position);
        let position_time = [body_position.x, body_position.y, body_position.z, time];
        for value in position_time {
            gpu_position_bytes.extend_from_slice(&value.to_le_bytes());
        }
        states.push(PlanningCandidateState {
            position_time,
            velocity_distance: [
                body_velocity.x,
                body_velocity.y,
                body_velocity.z,
                transverse_distance,
            ],
            body_rotation: rotation.to_array(),
            identity: [
                candidate,
                sample,
                ((reference_state.simulation_time_seconds - first_time) as f32)
                    .max(0.0)
                    .div_euclid(NEAR_SYNC_SEGMENT_MAX_SECONDS) as u32,
                0,
            ],
        });
    }
    Some(())
}

fn candidate_tube_parameters(candidate: u32, candidate_count: u32) -> (f32, f32, f32, f32) {
    let golden = 0.618_033_95_f32;
    let radial_fraction = ((candidate as f32 + 0.5) / candidate_count.max(1) as f32).sqrt();
    let radius = PLANNING_TRAJECTORY_TUBE_RADIUS_METERS * radial_fraction;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_voxels() -> Vec<InvertedDensityVoxel> {
        (0..56)
            .map(|index| InvertedDensityVoxel {
                center: Vec3::new(index as f32, 0.0, 0.0),
                volume: 1_000.0 + 17.0 * index as f32,
                density: 1_700.0,
                baseline_density: 1_700.0,
                reference_density: 1_700.0,
                grid: [index as u8, 0, 0],
            })
            .collect()
    }

    #[test]
    fn randomized_density_rows_are_distinct_positive_and_mass_preserving() {
        let voxels = test_voxels();
        let target_mass = 2.45e8;
        let (models, masses) =
            uniform_random_equal_mass_models(&voxels, target_mass, 32, 0x1065).unwrap();
        assert_eq!(models.len(), 32 * 56);
        assert_eq!(masses.len(), 32);
        assert!(
            models
                .iter()
                .all(|density| density.is_finite() && *density > 0.0)
        );
        assert!(
            masses
                .iter()
                .all(|mass| ((mass - target_mass) / target_mass).abs() <= 2.0e-7)
        );
        for pair in models.chunks_exact(56).collect::<Vec<_>>().windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
    }

    #[test]
    fn first_random_models_are_the_prefix_of_stress_for_the_same_capture() {
        let voxels = test_voxels();
        let (first, _) = uniform_random_equal_mass_models(&voxels, 2.45e8, 4, 7).unwrap();
        let (stress, _) = uniform_random_equal_mass_models(&voxels, 2.45e8, 32, 7).unwrap();
        assert_eq!(first, stress[..first.len()]);
    }
}
