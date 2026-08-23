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
    body_radius: f32,
    candidate_count: u32,
    density_model_count: u32,
    samples_per_candidate: u32,
    next_candidate: u32,
    preparation_ms: f64,
    reference_samples: Vec<TrajectoryInversionKnot>,
    states: Vec<PlanningCandidateState>,
    gpu_position_bytes: Vec<u8>,
    density_models: Vec<f32>,
    basis_records: Vec<PlanningBasisRecord>,
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
        reference_knots: &[TrajectoryInversionKnot],
        voxels: &[InvertedDensityVoxel],
        source: &AggregatedGravitySource,
    ) -> Option<Self> {
        let started = bevy::platform::time::Instant::now();
        let (candidate_count, density_model_count, samples_per_candidate) = profile.dimensions();
        if candidate_count != PLANNING_CANDIDATE_COUNT || voxels.len() != 56 {
            return None;
        }
        let basis = build_voxel_basis_sources(voxels, source)?;
        let reference = voxels
            .iter()
            .map(|voxel| voxel.reference_density)
            .collect::<Vec<_>>();
        let density_models = structured_equal_mass_models(voxels, &reference, density_model_count)?;
        let reference_samples = crate::cpu::inversion::sample_frozen_trajectory_at_count(
            reference_knots,
            samples_per_candidate as usize,
        )?;
        let basis_records = basis
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
        Some(Self {
            profile,
            run_id,
            capture_id,
            capture_epoch,
            source_hash,
            body_radius: source.radius as f32,
            candidate_count,
            density_model_count,
            samples_per_candidate,
            next_candidate: 0,
            preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
            reference_samples,
            states: Vec::with_capacity(candidate_count as usize * samples_per_candidate as usize),
            gpu_position_bytes: Vec::with_capacity(
                candidate_count as usize * samples_per_candidate as usize * 16,
            ),
            density_models,
            basis_records,
            basis_hash: basis.hash,
        })
    }

    pub(crate) fn matches(
        &self,
        profile: PlanningWorkloadProfile,
        run_id: u64,
        capture_id: u64,
        source_hash: u64,
    ) -> bool {
        self.profile == profile
            && self.run_id == run_id
            && self.capture_id == capture_id
            && self.source_hash == source_hash
    }

    pub(crate) fn advance(&mut self, candidate_budget: u32) -> bool {
        let started = bevy::platform::time::Instant::now();
        let end = (self.next_candidate + candidate_budget.max(1)).min(self.candidate_count);
        while self.next_candidate < end {
            if append_tube_candidate_states(
                self.next_candidate,
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
        Some((
            PlanningCandidateBatch {
                batch_id,
                capture_id: self.capture_id,
                capture_epoch: self.capture_epoch,
                source_hash: self.source_hash,
                candidate_count: self.candidate_count,
                density_model_count: self.density_model_count,
                samples_per_candidate: self.samples_per_candidate,
                body_radius: self.body_radius,
                states: Arc::from(self.states),
                gpu_position_bytes: Arc::from(self.gpu_position_bytes),
                density_models: Arc::from(self.density_models),
                basis_records: Arc::from(self.basis_records),
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

fn append_tube_candidate_states(
    candidate: u32,
    reference: &[TrajectoryInversionKnot],
    states: &mut Vec<PlanningCandidateState>,
    gpu_position_bytes: &mut Vec<u8>,
) -> Option<()> {
    let sample_count = reference.len() as u32;
    if sample_count < 2 {
        return None;
    }
    let mut world_positions = Vec::with_capacity(sample_count as usize);
    let (radius, phase, harmonic, phase_rate) = candidate_tube_parameters(candidate);
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

fn candidate_tube_parameters(candidate: u32) -> (f32, f32, f32, f32) {
    let golden = 0.618_033_95_f32;
    let radial_fraction = ((candidate as f32 + 0.5) / PLANNING_CANDIDATE_COUNT as f32).sqrt();
    let radius = PLANNING_TRAJECTORY_TUBE_RADIUS_METERS * radial_fraction;
    let phase = std::f32::consts::TAU * ((candidate as f32 * golden).fract());
    let harmonic = 1.0 + (candidate % 5) as f32;
    let phase_rate = ((candidate.wrapping_mul(747_796_405) ^ 2_891_336_453) as f32) * f32::EPSILON;
    (radius, phase, harmonic, phase_rate)
}

fn structured_equal_mass_models(
    voxels: &[InvertedDensityVoxel],
    reference: &[f32],
    model_count: u32,
) -> Option<Vec<f32>> {
    let target_mass = voxels
        .iter()
        .zip(reference)
        .map(|(voxel, density)| voxel.volume as f64 * f64::from(*density))
        .sum::<f64>();
    if !target_mass.is_finite() || target_mass <= 0.0 {
        return None;
    }
    let radius = voxels
        .iter()
        .map(|voxel| voxel.center.length())
        .fold(0.0_f32, f32::max)
        .max(1.0);
    let mut models = Vec::with_capacity(model_count as usize * voxels.len());
    for model in 0..model_count {
        let phase = std::f32::consts::TAU * model as f32 / model_count.max(1) as f32;
        let axis = Vec3::new(
            (phase * 1.7).cos(),
            (phase * 2.3).sin(),
            (phase * 0.7 + 0.4).cos(),
        )
        .normalize_or_zero();
        let mut row = voxels
            .iter()
            .zip(reference)
            .map(|(voxel, base)| {
                let position = voxel.center / radius;
                let radial = position.length().clamp(0.0, 1.0);
                let lobe = position.dot(axis);
                let shell = (std::f32::consts::PI * radial).cos();
                let rubble = (position.x * 17.0 + phase).sin()
                    * (position.y * 13.0 - phase * 0.7).cos()
                    * (position.z * 11.0 + phase * 1.3).sin();
                let pattern = match model % 8 {
                    0 => 0.0,
                    1 => 0.28 * (1.0 - radial),
                    2 => -0.24 * (1.0 - radial),
                    3 => 0.22 * shell,
                    4 => 0.26 * lobe,
                    5 => 0.20 * (2.0 * lobe * lobe - 0.5),
                    6 => 0.18 * rubble,
                    _ => 0.14 * shell + 0.16 * lobe + 0.10 * rubble,
                };
                (*base * (1.0 + pattern)).max(250.0)
            })
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
        models.extend(row);
    }
    Some(models)
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
