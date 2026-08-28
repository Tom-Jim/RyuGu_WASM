use crate::cpu::curved_arc::AggregatedGravitySource;
use crate::cpu::inversion::{
    PlanningDynamicsTree, build_planning_dynamics_tree, build_voxel_basis_sources,
};
use crate::cpu::volterra::{VolterraConfig, VolterraForceInput, propagate_reference_line_batched};
use crate::interface::components::*;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use std::sync::Arc;

const PLANNING_INITIAL_TUBE_FRACTION: f32 = 0.70;

pub(crate) struct PlanningBatchBuilder {
    profile: PlanningWorkloadProfile,
    run_id: u64,
    capture_id: u64,
    capture_epoch: u64,
    source_hash: u64,
    source_count: u32,
    body_radius: f32,
    eq106_source_radius: f32,
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
    basis_hash: u64,
    reference_jets: Vec<PlanningReferenceJet>,
    dynamics_tree: PlanningDynamicsTree,
}

#[derive(Clone, Copy, Debug)]
struct PlanningReferenceJet {
    simulation_time_seconds: f64,
    body_rotation: DQuat,
    world_position: DVec3,
    world_acceleration: DVec3,
    world_jacobian: DMat3,
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
        voxel_size: f32,
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
        let basis = build_voxel_basis_sources(voxels, source, voxel_size)?;
        let reference_samples = crate::cpu::inversion::sample_frozen_trajectory_at_count(
            reference_knots,
            samples_per_candidate as usize,
        )?;
        let reference_states = central_reference_states(&reference_samples)?;
        let mut canonical_basis_records = basis
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
                })
            })
            .collect::<Vec<_>>();
        // Every one of the 56 voxel columns must remain addressable for the
        // basis-spectrum cache.  Empty voxels therefore contribute a single
        // zero/nominal-volume representative, which can make the raw
        // canonical list slightly larger than a requested low source count
        // capture.  Coalesce only records belonging to the same voxel before
        // refinement so the requested source count remains exact while each
        // voxel range is still present and its volume/centroid are conserved.
        if canonical_basis_records.len() > requested_source_count as usize {
            canonical_basis_records =
                coalesce_basis_records(&canonical_basis_records, requested_source_count as usize)?;
        }
        let basis_records = spatially_refine_basis_records(
            &canonical_basis_records,
            voxels,
            voxel_size,
            source.radius as f32,
            requested_source_count,
        )?;
        let eq106_source_radius = basis_records.iter().fold(0.0_f32, |radius, record| {
            radius.max(
                Vec3::new(
                    record.position_volume[0],
                    record.position_volume[1],
                    record.position_volume[2],
                )
                .length(),
            )
        });
        if !eq106_source_radius.is_finite() || eq106_source_radius <= 0.0 {
            return None;
        }
        let basis_hash = basis_records.iter().fold(
            mix_hash(basis.hash, u64::from(requested_source_count)),
            |hash, record| {
                let hash = record
                    .position_volume
                    .into_iter()
                    .fold(hash, |hash, value| {
                        mix_hash(hash, u64::from(value.to_bits()))
                    });
                mix_hash(hash, u64::from(record.voxel_index))
            },
        );
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
        // Candidate dynamics must not change merely because the crossover
        // benchmark refines the same mass distribution from 32K to 8192K
        // quadrature records. Build one nonlinear FMM field from the canonical
        // mass/centroid representation and close every Picard sweep against it.
        let dynamics_tree =
            build_planning_dynamics_tree(&canonical_basis_records, density_models.get(..56)?)?;
        let reference_jets = build_planning_reference_jets(&reference_samples);
        Some(Self {
            profile,
            run_id,
            capture_id,
            capture_epoch,
            source_hash,
            source_count: requested_source_count,
            body_radius: source.radius as f32,
            eq106_source_radius,
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
            basis_hash,
            reference_jets,
            dynamics_tree,
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            let generated = generate_candidate_range_parallel(
                self.next_candidate,
                end,
                self.candidate_count,
                &self.reference_samples,
                &self.reference_jets,
                &self.dynamics_tree,
            );
            let Some(mut generated) = generated else {
                return false;
            };
            generated.sort_unstable_by_key(|(candidate, _, _)| *candidate);
            for (_, states, bytes) in generated {
                self.states.extend(states);
                self.gpu_position_bytes.extend(bytes);
            }
            self.next_candidate = end;
        }
        #[cfg(target_arch = "wasm32")]
        while self.next_candidate < end {
            if append_dynamical_candidate_states(
                self.next_candidate,
                self.candidate_count,
                &self.reference_samples,
                &self.reference_jets,
                Some(&self.dynamics_tree),
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
        let (eq106_volume_source_bytes, eq106_voxel_source_ranges) =
            eq106_geometry_buffers(&self.basis_records)?;
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
                eq106_source_radius: self.eq106_source_radius,
                reference_states: Arc::from(self.reference_states),
                states: Arc::from(self.states),
                gpu_position_bytes: Arc::from(self.gpu_position_bytes),
                density_models: Arc::from(self.density_models),
                density_model_masses: Arc::from(self.density_model_masses),
                density_seed: self.density_seed,
                target_mass: self.target_mass,
                basis_records: Arc::from(self.basis_records),
                eq106_volume_source_bytes,
                eq106_voxel_source_ranges: Arc::from(eq106_voxel_source_ranges),
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

/// Native planning uses a bounded work queue so trajectory propagation does
/// not serialize behind GPU submission or UI rendering. Results are sorted by
/// candidate index before they are appended, preserving the deterministic GPU
/// buffer layout used by the WASM build. Browser WASM intentionally keeps the
/// same algorithm cooperative: a web worker/atomics build is an opt-in deploy
/// target, while the default page must remain responsive without it.
#[cfg(not(target_arch = "wasm32"))]
fn generate_candidate_range_parallel(
    start: u32,
    end: u32,
    candidate_count: u32,
    reference: &[TrajectoryInversionKnot],
    reference_jets: &[PlanningReferenceJet],
    dynamics_tree: &PlanningDynamicsTree,
) -> Option<Vec<(u32, Vec<PlanningCandidateState>, Vec<u8>)>> {
    use crossbeam_channel::bounded;

    if start >= end {
        return Some(Vec::new());
    }
    let worker_count = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min((end - start) as usize)
        .max(1);
    let (work_tx, work_rx) = bounded::<u32>(worker_count);
    // Workers must never block publishing completion while the scheduler is
    // still filling the bounded work queue; otherwise a full two-way queue
    // can deadlock before the main thread begins collection.
    let (result_tx, result_rx) =
        crossbeam_channel::unbounded::<Option<(u32, Vec<PlanningCandidateState>, Vec<u8>)>>();
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            scope.spawn(move || {
                while let Ok(candidate) = work_rx.recv() {
                    let mut states = Vec::with_capacity(reference.len());
                    let mut bytes = Vec::with_capacity(reference.len() * 16);
                    let result = append_dynamical_candidate_states(
                        candidate,
                        candidate_count,
                        reference,
                        reference_jets,
                        Some(dynamics_tree),
                        &mut states,
                        &mut bytes,
                    )
                    .map(|()| (candidate, states, bytes));
                    if result_tx.send(result).is_err() {
                        return;
                    }
                }
            });
        }
        drop(result_tx);
        for candidate in start..end {
            if work_tx.send(candidate).is_err() {
                return None;
            }
        }
        drop(work_tx);
        let mut generated = Vec::with_capacity((end - start) as usize);
        for _ in start..end {
            generated.push(result_rx.recv().ok()??);
        }
        Some(generated)
    })
}

fn coalesce_basis_records(
    records: &[PlanningBasisRecord],
    requested: usize,
) -> Option<Vec<PlanningBasisRecord>> {
    if records.is_empty() || requested == 0 || requested > records.len() {
        return None;
    }
    let mut groups = std::collections::BTreeMap::<u32, Vec<PlanningBasisRecord>>::new();
    for record in records.iter().copied() {
        if !record.position_volume.iter().all(|value| value.is_finite())
            || record.position_volume[3] <= 0.0
        {
            return None;
        }
        groups.entry(record.voxel_index).or_default().push(record);
    }
    if requested < groups.len() {
        return None;
    }
    let mut total = records.len();
    while total > requested {
        let (voxel, group) = groups
            .iter_mut()
            .filter(|(_, group)| group.len() > 1)
            .max_by_key(|(_, group)| group.len())?;
        let right = group.pop()?;
        let left = group.pop()?;
        let left_volume = f64::from(left.position_volume[3]);
        let right_volume = f64::from(right.position_volume[3]);
        let volume = left_volume + right_volume;
        if !volume.is_finite() || volume <= 0.0 {
            return None;
        }
        let position = (Vec3::new(
            left.position_volume[0],
            left.position_volume[1],
            left.position_volume[2],
        ) * left_volume as f32
            + Vec3::new(
                right.position_volume[0],
                right.position_volume[1],
                right.position_volume[2],
            ) * right_volume as f32)
            / volume as f32;
        group.push(PlanningBasisRecord {
            position_volume: [position.x, position.y, position.z, volume as f32],
            voxel_index: *voxel,
        });
        total -= 1;
    }
    Some(groups.into_values().flatten().collect())
}

fn eq106_geometry_buffers(records: &[PlanningBasisRecord]) -> Option<(Arc<[u8]>, [[u32; 2]; 56])> {
    if records.is_empty() || records.len() > u32::MAX as usize {
        return None;
    }
    let mut bytes = Vec::with_capacity(records.len() * 16);
    let mut ranges = [[0_u32; 2]; 56];
    let mut cursor = 0_usize;
    for (voxel, range) in ranges.iter_mut().enumerate() {
        let start = cursor;
        while cursor < records.len() && records[cursor].voxel_index as usize == voxel {
            bytes.extend_from_slice(bytemuck::cast_slice(&records[cursor].position_volume));
            cursor += 1;
        }
        *range = [start as u32, (cursor - start) as u32];
    }
    if cursor != records.len() || ranges.iter().any(|range| range[1] == 0) {
        return None;
    }
    Some((Arc::from(bytes), ranges))
}

/// Deterministically replace every parent quadrature source with an
/// antithetic cloud inside a representative micro-voxel centred on the
/// original quadrature source. The micro-voxel side is bounded by both the
/// density-grid voxel size and the cube root of the parent's volume. Each cloud
/// preserves the parent's total volume and centre of mass, while additional
/// pairs sample genuinely distinct spatial positions. All backends and the
/// independent f64 reference consume this exact same refined point set.
fn spatially_refine_basis_records(
    canonical: &[PlanningBasisRecord],
    voxels: &[InvertedDensityVoxel],
    voxel_size: f32,
    body_radius: f32,
    requested: u32,
) -> Option<Vec<PlanningBasisRecord>> {
    let requested = requested as usize;
    if canonical.is_empty()
        || requested < canonical.len()
        || voxels.is_empty()
        || !voxel_size.is_finite()
        || voxel_size <= 0.0
        || !body_radius.is_finite()
        || body_radius <= 0.0
    {
        return None;
    }
    let base = requested / canonical.len();
    let remainder = requested % canonical.len();
    let mut refined = Vec::with_capacity(requested);
    for (index, record) in canonical.iter().copied().enumerate() {
        let copies = base + usize::from(index < remainder);
        let voxel = voxels.get(record.voxel_index as usize)?;
        let parent_volume = record.position_volume[3];
        if !parent_volume.is_finite() || parent_volume <= 0.0 {
            return None;
        }
        let centre = Vec3::from_array(record.position_volume[..3].try_into().ok()?);
        if !centre.is_finite() {
            return None;
        }
        let micro_voxel_side = voxel_size.min(parent_volume.cbrt());
        let safe_extent = 0.45 * micro_voxel_side;
        let pair_count = copies / 2;
        let has_centre = !copies.is_multiple_of(2);
        let nominal_volume = f64::from(parent_volume) / copies as f64;
        for pair in 0..pair_count {
            let direction = refinement_direction(index, pair);
            let radial_fraction = 0.35 + 0.55 * radical_inverse_vdc((pair + 1) as u32);
            let requested_extent = safe_extent * radial_fraction;
            let cell_min = Vec3::splat(-body_radius)
                + Vec3::new(
                    f32::from(voxel.grid[0]),
                    f32::from(voxel.grid[1]),
                    f32::from(voxel.grid[2]),
                ) * voxel_size;
            let cell_max = cell_min + Vec3::splat(voxel_size);
            let cell_margin = (centre - cell_min).min(cell_max - centre).max(Vec3::ZERO);
            let mut extent = requested_extent;
            for (component, margin) in direction
                .abs()
                .to_array()
                .into_iter()
                .zip(cell_margin.to_array())
            {
                if component > 1.0e-6 {
                    extent = extent.min(margin / component);
                }
            }
            // Both antithetic children remain inside the conservative body
            // sphere as well as the occupied density cell. The exact radius
            // is nevertheless recomputed after refinement and used by every
            // Eq.106 Taylor certificate.
            let centre2 = centre.length_squared();
            let projected = centre.dot(direction).abs();
            let sphere_extent = (-projected
                + (projected * projected + body_radius * body_radius - centre2)
                    .max(0.0)
                    .sqrt())
            .max(0.0);
            extent = extent.min(sphere_extent);
            let offset = direction * extent;
            let pair_volume = if !has_centre && pair + 1 == pair_count {
                0.5 * (f64::from(parent_volume)
                    - 2.0 * nominal_volume * (pair_count.saturating_sub(1)) as f64)
            } else {
                nominal_volume
            } as f32;
            for position in [centre + offset, centre - offset] {
                let mut child = record;
                child.position_volume = [position.x, position.y, position.z, pair_volume];
                refined.push(child);
            }
        }
        if has_centre {
            let used = 2.0 * nominal_volume * pair_count as f64;
            let mut child = record;
            child.position_volume[3] = (f64::from(parent_volume) - used).max(0.0) as f32;
            refined.push(child);
        }
    }
    (refined.len() == requested).then_some(refined)
}

fn radical_inverse_vdc(value: u32) -> f32 {
    value.reverse_bits() as f32 * 2.328_306_4e-10
}

fn refinement_direction(parent: usize, pair: usize) -> Vec3 {
    let sequence = (parent as u32)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(pair as u32 + 1);
    let z = 1.0 - 2.0 * radical_inverse_vdc(sequence);
    let azimuth = std::f32::consts::TAU * radical_inverse_vdc(sequence.wrapping_mul(0x85eb_ca6b));
    let radius = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(radius * azimuth.cos(), radius * azimuth.sin(), z).normalize_or_zero()
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

fn append_dynamical_candidate_states(
    candidate: u32,
    candidate_count: u32,
    reference: &[TrajectoryInversionKnot],
    reference_jets: &[PlanningReferenceJet],
    dynamics_tree: Option<&PlanningDynamicsTree>,
    states: &mut Vec<PlanningCandidateState>,
    gpu_position_bytes: &mut Vec<u8>,
) -> Option<()> {
    let sample_count = reference.len() as u32;
    if sample_count < 2 || reference_jets.len() != reference.len() {
        return None;
    }
    let (requested_radius, phase, harmonic, phase_rate) =
        candidate_tube_parameters(candidate, candidate_count);
    // A fixed initial offset can leave the 15 m Taylor tube on a captured arc
    // whose transverse variational dynamics are locally unstable.  That is
    // not a reason to cancel the entire quadrature sweep.  Contract only this
    // candidate's initial perturbation, re-propagating it from scratch each
    // time, until the complete Volterra trajectory is certified inside the
    // same tube.  The zero-radius final attempt also distinguishes a bad
    // perturbation from a genuinely unusable reference arc.
    for contraction in [1.0_f32, 0.5, 0.25, 0.125, 0.0625, 0.0] {
        let mut candidate_states = Vec::with_capacity(reference.len());
        let mut candidate_bytes = Vec::with_capacity(reference.len() * 16);
        if append_dynamical_candidate_at_radius(
            candidate,
            reference,
            reference_jets,
            dynamics_tree,
            requested_radius * contraction,
            phase,
            harmonic,
            phase_rate,
            &mut candidate_states,
            &mut candidate_bytes,
        )
        .is_some()
        {
            states.extend(candidate_states);
            gpu_position_bytes.extend(candidate_bytes);
            return Some(());
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn append_dynamical_candidate_at_radius(
    candidate: u32,
    reference: &[TrajectoryInversionKnot],
    reference_jets: &[PlanningReferenceJet],
    dynamics_tree: Option<&PlanningDynamicsTree>,
    radius: f32,
    phase: f32,
    harmonic: f32,
    phase_rate: f32,
    states: &mut Vec<PlanningCandidateState>,
    gpu_position_bytes: &mut Vec<u8>,
) -> Option<()> {
    let sample_count = reference.len() as u32;
    let first = *reference.first()?;
    let first_offset =
        candidate_initial_offset(first, 0, sample_count, radius, phase, harmonic, phase_rate)?;
    let initial_position = (first.position + first_offset).as_dvec3();
    // Candidates differ through their initial transverse displacement.  The
    // velocity is not differentiated from an arbitrary drawn tube; it is the
    // captured physical velocity and the propagated field determines every
    // subsequent state.
    let initial_velocity = first.velocity.as_dvec3();
    let final_time = reference.last()?.simulation_time_seconds;
    let duration = final_time - first.simulation_time_seconds;
    let solution = propagate_reference_line_batched(
        initial_position,
        initial_velocity,
        initial_velocity,
        duration,
        VolterraConfig {
            node_count: reference.len().max(33),
            maximum_picard_iterations: 24,
            maximum_endpoint_iterations: 8,
            damping: 0.70,
            relative_tolerance: 1.0e-6,
            minimum_longitudinal_speed: 1.0e-5,
            maximum_transverse_distance: f64::INFINITY,
        },
        |inputs, accelerations| {
            fill_planning_reference_accelerations(
                reference_jets,
                dynamics_tree,
                first.simulation_time_seconds,
                inputs,
                accelerations,
            )
        },
    )
    .ok()?;

    let angular_velocity =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    let first_time = first.simulation_time_seconds;
    let mut solution_cursor = 1;
    for sample in 0..sample_count {
        let reference_state = reference[sample as usize];
        let propagated = solution.sample_at_ordered(
            reference_state.simulation_time_seconds - first.simulation_time_seconds,
            &mut solution_cursor,
        )?;
        let world_position = propagated.position.as_vec3();
        let world_velocity = propagated.velocity.as_vec3();
        let transverse_distance = world_position.distance(reference_state.position);
        if !transverse_distance.is_finite()
            || transverse_distance > PLANNING_TRAJECTORY_TUBE_RADIUS_METERS + 1.0e-3
        {
            return None;
        }
        let time = reference_state.simulation_time_seconds as f32;
        let rotation = reference_state.body_rotation;
        let body_position = rotation.inverse() * world_position;
        let body_velocity =
            rotation.inverse() * (world_velocity - angular_velocity.cross(world_position));
        let position_time = [body_position.x, body_position.y, body_position.z, time];
        gpu_position_bytes.extend_from_slice(bytemuck::cast_slice(&position_time));
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
                1,
            ],
        });
    }
    Some(())
}

fn candidate_initial_offset(
    reference_state: TrajectoryInversionKnot,
    sample: u32,
    sample_count: u32,
    radius: f32,
    phase: f32,
    harmonic: f32,
    phase_rate: f32,
) -> Option<Vec3> {
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
    Some(normal * (offset_radius * angle.cos()) + binormal * (offset_radius * angle.sin()))
}

fn fill_planning_reference_accelerations(
    jets: &[PlanningReferenceJet],
    dynamics_tree: Option<&PlanningDynamicsTree>,
    start_time_seconds: f64,
    inputs: &[VolterraForceInput],
    accelerations: &mut [DVec3],
) -> Result<(), ()> {
    if inputs.len() != accelerations.len() || jets.is_empty() {
        return Err(());
    }
    let first = jets[0];
    let last = jets[jets.len() - 1];
    let mut upper_index = usize::from(jets.len() > 1);
    for (input, acceleration) in inputs.iter().zip(accelerations) {
        let simulation_time_seconds = start_time_seconds + input.elapsed_seconds;
        let jet = if simulation_time_seconds <= first.simulation_time_seconds {
            first
        } else if simulation_time_seconds >= last.simulation_time_seconds {
            last
        } else {
            // Picard's h grid is monotone, so its elapsed times are monotone as
            // well.  Advance one shared cursor for the whole sweep instead of
            // performing M independent binary searches.
            while upper_index + 1 < jets.len()
                && jets[upper_index].simulation_time_seconds < simulation_time_seconds
            {
                upper_index += 1;
            }
            let lower = jets[upper_index - 1];
            let upper = jets[upper_index];
            let interval = (upper.simulation_time_seconds - lower.simulation_time_seconds)
                .max(f64::MIN_POSITIVE);
            let weight = ((simulation_time_seconds - lower.simulation_time_seconds) / interval)
                .clamp(0.0, 1.0);
            PlanningReferenceJet {
                simulation_time_seconds,
                body_rotation: lower.body_rotation.slerp(upper.body_rotation, weight),
                world_position: lower.world_position.lerp(upper.world_position, weight),
                world_acceleration: lower
                    .world_acceleration
                    .lerp(upper.world_acceleration, weight),
                world_jacobian: lower.world_jacobian * (1.0 - weight)
                    + upper.world_jacobian * weight,
            }
        };
        *acceleration = if let Some(tree) = dynamics_tree {
            // Nonlinear Picard closure: every updated world position is
            // transformed into the rotating density frame and reevaluated by
            // the FMM field, rather than being frozen to a+J*delta_r.
            let body_position = jet.body_rotation.inverse() * input.position;
            jet.body_rotation * tree.acceleration(body_position).ok_or(())?
        } else {
            // Retained only for focused affine-field unit tests.
            jet.world_acceleration + jet.world_jacobian * (input.position - jet.world_position)
        };
    }
    Ok(())
}

fn build_planning_reference_jets(
    reference: &[TrajectoryInversionKnot],
) -> Vec<PlanningReferenceJet> {
    reference
        .iter()
        .map(|state| {
            let rotation = DQuat::from_xyzw(
                f64::from(state.body_rotation.x),
                f64::from(state.body_rotation.y),
                f64::from(state.body_rotation.z),
                f64::from(state.body_rotation.w),
            )
            .normalize();
            let world_position = state.position.as_dvec3();
            PlanningReferenceJet {
                simulation_time_seconds: state.simulation_time_seconds,
                body_rotation: rotation,
                world_position,
                world_acceleration: state.baseline_acceleration.as_dvec3(),
                world_jacobian: DMat3::ZERO,
            }
        })
        .collect()
}

/// Direct source sum retained as the independent certified reference.  It is
/// intentionally not used by Picard propagation: certification needs an
/// exact, method-independent field/gradient, while candidate dynamics use the
/// source-count-independent FMM tree and reevaluate every updated position.
pub(crate) fn evaluate_planning_reference_field(
    target: DVec3,
    basis_records: &[PlanningBasisRecord],
    densities: &[f32],
) -> Option<(DVec3, DMat3)> {
    if densities.len() != 56 || !target.is_finite() {
        return None;
    }
    let mut acceleration = DVec3::ZERO;
    let mut gradient = DMat3::ZERO;
    for source in basis_records {
        let density = f64::from(*densities.get(source.voxel_index as usize)?);
        let position = DVec3::new(
            f64::from(source.position_volume[0]),
            f64::from(source.position_volume[1]),
            f64::from(source.position_volume[2]),
        );
        let displacement = position - target;
        let radius_squared = displacement.length_squared().max(1.0e-16);
        let inverse_radius = radius_squared.sqrt().recip();
        let inverse_radius_cubed = inverse_radius / radius_squared;
        let mass = f64::from(source.position_volume[3]) * density;
        acceleration += f64::from(G) * mass * displacement * inverse_radius_cubed;
        let outer = DMat3::from_cols(
            displacement * displacement.x,
            displacement * displacement.y,
            displacement * displacement.z,
        );
        gradient += f64::from(G)
            * mass
            * (-DMat3::IDENTITY * inverse_radius_cubed
                + outer * (3.0 * inverse_radius_cubed / radius_squared));
    }
    (acceleration.is_finite() && gradient.is_finite()).then_some((acceleration, gradient))
}

fn candidate_tube_parameters(candidate: u32, candidate_count: u32) -> (f32, f32, f32, f32) {
    let golden = 0.618_033_95_f32;
    let radial_fraction = ((candidate as f32 + 0.5) / candidate_count.max(1) as f32).sqrt();
    // Reserve part of the certified tube for differential-force drift over the
    // complete propagated arc. Final states are still checked against the full
    // 15 m trust radius before they enter any GPU batch.
    let radius =
        PLANNING_TRAJECTORY_TUBE_RADIUS_METERS * PLANNING_INITIAL_TUBE_FRACTION * radial_fraction;
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
    use bevy::math::DVec3;

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
        for pair in models.as_chunks::<56>().0.windows(2) {
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

    #[test]
    fn spatial_refinement_is_distinct_and_preserves_parent_mass_and_centroid() {
        let voxels = vec![InvertedDensityVoxel {
            center: Vec3::new(12.0, -4.0, 8.0),
            volume: 1_000.0,
            density: 1_700.0,
            baseline_density: 1_700.0,
            reference_density: 1_700.0,
            grid: [2, 1, 2],
        }];
        let parent = PlanningBasisRecord {
            position_volume: [12.0, -4.0, 8.0, 1_000.0],
            voxel_index: 0,
        };
        let refined = spatially_refine_basis_records(&[parent], &voxels, 20.0, 40.0, 8).unwrap();
        assert_eq!(refined.len(), 8);

        let mut positions = refined
            .iter()
            .map(|record| {
                (
                    record.position_volume[0].to_bits(),
                    record.position_volume[1].to_bits(),
                    record.position_volume[2].to_bits(),
                )
            })
            .collect::<Vec<_>>();
        positions.sort_unstable();
        positions.dedup();
        assert_eq!(positions.len(), refined.len());

        let volume = refined
            .iter()
            .map(|record| f64::from(record.position_volume[3]))
            .sum::<f64>();
        let centroid = refined.iter().fold(DVec3::ZERO, |moment, record| {
            moment
                + DVec3::new(
                    f64::from(record.position_volume[0]),
                    f64::from(record.position_volume[1]),
                    f64::from(record.position_volume[2]),
                ) * f64::from(record.position_volume[3])
        }) / volume;
        assert!((volume - 1_000.0).abs() <= 1.0e-5);
        assert!(centroid.distance(DVec3::new(12.0, -4.0, 8.0)) <= 1.0e-5);
        assert!(refined.iter().all(|record| {
            let position = Vec3::from_array(record.position_volume[..3].try_into().unwrap());
            position.distance(Vec3::new(12.0, -4.0, 8.0)) < 9.0
                && position.length() <= 40.0
                && (0.0..=20.0).contains(&position.x)
                && (-20.0..=0.0).contains(&position.y)
                && (0.0..=20.0).contains(&position.z)
        }));
    }

    #[test]
    fn canonical_coalescing_hits_requested_count_per_voxel_and_preserves_moments() {
        let records = vec![
            PlanningBasisRecord {
                position_volume: [0.0, 0.0, 0.0, 2.0],
                voxel_index: 0,
            },
            PlanningBasisRecord {
                position_volume: [2.0, 0.0, 0.0, 3.0],
                voxel_index: 0,
            },
            PlanningBasisRecord {
                position_volume: [10.0, 0.0, 0.0, 4.0],
                voxel_index: 1,
            },
        ];
        let coalesced = coalesce_basis_records(&records, 2).unwrap();
        assert_eq!(coalesced.len(), 2);
        assert_eq!(
            coalesced
                .iter()
                .map(|record| record.voxel_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let merged = coalesced[0];
        assert!((merged.position_volume[0] - 1.2).abs() <= 1.0e-6);
        assert!((merged.position_volume[3] - 5.0).abs() <= 1.0e-6);
    }

    #[test]
    fn spatial_refinement_supports_non_power_of_two_source_counts() {
        let voxels = vec![InvertedDensityVoxel {
            center: Vec3::ZERO,
            volume: 2.0,
            density: 1.0,
            baseline_density: 1.0,
            reference_density: 1.0,
            grid: [0; 3],
        }];
        let parents = [
            PlanningBasisRecord {
                position_volume: [-2.0, 0.0, 0.0, 2.0],
                voxel_index: 0,
            },
            PlanningBasisRecord {
                position_volume: [2.0, 0.0, 0.0, 3.0],
                voxel_index: 0,
            },
        ];
        let refined = spatially_refine_basis_records(&parents, &voxels, 2.0, 4.0, 9).unwrap();
        assert_eq!(refined.len(), 9);
        for (parent_index, expected_volume) in [2.0_f64, 3.0].into_iter().enumerate() {
            let actual = refined
                .iter()
                .filter(|record| {
                    if parent_index == 0 {
                        record.position_volume[0] < 0.0
                    } else {
                        record.position_volume[0] > 0.0
                    }
                })
                .map(|record| f64::from(record.position_volume[3]))
                .sum::<f64>();
            assert!((actual - expected_volume).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn planning_candidates_are_propagated_from_initial_conditions() {
        let reference = [
            TrajectoryInversionKnot {
                position: Vec3::ZERO,
                velocity: Vec3::X * 10.0,
                simulation_time_seconds: 0.0,
                baseline_acceleration: Vec3::ZERO,
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: Vec3::X * 10.0,
                velocity: Vec3::X * 10.0,
                simulation_time_seconds: 1.0,
                baseline_acceleration: Vec3::ZERO,
                body_rotation: Quat::IDENTITY,
            },
        ];
        let jets = reference
            .iter()
            .map(|sample| PlanningReferenceJet {
                simulation_time_seconds: sample.simulation_time_seconds,
                body_rotation: DQuat::IDENTITY,
                world_position: sample.position.as_dvec3(),
                world_acceleration: DVec3::ZERO,
                world_jacobian: DMat3::ZERO,
            })
            .collect::<Vec<_>>();
        let mut states = Vec::new();
        let mut bytes = Vec::new();
        append_dynamical_candidate_states(0, 32, &reference, &jets, None, &mut states, &mut bytes)
            .unwrap();

        assert_eq!(states.len(), reference.len());
        assert_eq!(bytes.len(), reference.len() * 16);
        assert!(states.iter().all(|state| state.identity[3] == 1));
        let first_offset = states[0].body_position().distance(reference[0].position);
        let last_offset = states[1].body_position().distance(reference[1].position);
        assert!((first_offset - last_offset).abs() < 1.0e-5);
        assert!(last_offset <= PLANNING_TRAJECTORY_TUBE_RADIUS_METERS);
    }

    #[test]
    fn unstable_transverse_candidate_contracts_instead_of_cancelling_batch() {
        let reference = (0..=2)
            .map(|second| TrajectoryInversionKnot {
                position: Vec3::X * (10.0 * second as f32),
                velocity: Vec3::X * 10.0,
                simulation_time_seconds: second as f64,
                baseline_acceleration: Vec3::ZERO,
                body_rotation: Quat::IDENTITY,
            })
            .collect::<Vec<_>>();
        let jets = reference
            .iter()
            .map(|sample| PlanningReferenceJet {
                simulation_time_seconds: sample.simulation_time_seconds,
                body_rotation: DQuat::IDENTITY,
                world_position: sample.position.as_dvec3(),
                world_acceleration: DVec3::ZERO,
                // y'' = y and z'' = z amplify the requested outer offset
                // beyond 15 m, forcing the adaptive contraction path.
                world_jacobian: DMat3::from_diagonal(DVec3::new(0.0, 1.0, 1.0)),
            })
            .collect::<Vec<_>>();
        let requested_radius = candidate_tube_parameters(31, 32).0;
        let mut states = Vec::new();
        let mut bytes = Vec::new();

        append_dynamical_candidate_states(31, 32, &reference, &jets, None, &mut states, &mut bytes)
            .expect("an unstable candidate should contract, not cancel the complete sweep");

        assert_eq!(states.len(), reference.len());
        assert_eq!(bytes.len(), reference.len() * 16);
        assert!(states[0].velocity_distance[3] < requested_radius * 0.75);
        assert!(states.iter().all(|state| {
            state.velocity_distance[3].is_finite()
                && state.velocity_distance[3] <= PLANNING_TRAJECTORY_TUBE_RADIUS_METERS + 1.0e-3
        }));
    }

    #[test]
    fn long_arc_outer_candidate_remains_inside_the_certified_tube() {
        let sample_count = 241;
        let duration = BENCHMARK_DURATION_SECONDS;
        let radius = 1_000.0_f64;
        let gravitational_parameter = f64::from(G) * f64::from(RYUGU_MASS);
        let angular_speed = (gravitational_parameter / radius.powi(3)).sqrt();
        let normal = RYUGU_SPIN_AXIS.as_dvec3().normalize();
        let axis_x = normal.cross(DVec3::X).normalize();
        let axis_y = normal.cross(axis_x).normalize();
        let mut reference = Vec::with_capacity(sample_count);
        let mut jets = Vec::with_capacity(sample_count);
        for index in 0..sample_count {
            let time = duration * index as f64 / (sample_count - 1) as f64;
            let angle = angular_speed * time;
            let radial = axis_x * angle.cos() + axis_y * angle.sin();
            let tangent = -axis_x * angle.sin() + axis_y * angle.cos();
            let position = radial * radius;
            let velocity = tangent * (radius * angular_speed);
            let inverse_radius3 = radius.powi(-3);
            let acceleration = -gravitational_parameter * position * inverse_radius3;
            let outer = DMat3::from_cols(
                position * position.x,
                position * position.y,
                position * position.z,
            );
            let jacobian = gravitational_parameter
                * (outer * (3.0 / radius.powi(5)) - DMat3::IDENTITY * inverse_radius3);
            reference.push(TrajectoryInversionKnot {
                position: position.as_vec3(),
                velocity: velocity.as_vec3(),
                simulation_time_seconds: time,
                baseline_acceleration: acceleration.as_vec3(),
                body_rotation: Quat::IDENTITY,
            });
            jets.push(PlanningReferenceJet {
                simulation_time_seconds: time,
                body_rotation: DQuat::IDENTITY,
                world_position: position,
                world_acceleration: acceleration,
                world_jacobian: jacobian,
            });
        }

        let (candidate_radius, phase, harmonic, phase_rate) = candidate_tube_parameters(31, 32);
        let initial_offset = candidate_initial_offset(
            reference[0],
            0,
            sample_count as u32,
            candidate_radius,
            phase,
            harmonic,
            phase_rate,
        )
        .unwrap();
        let direct_solve = propagate_reference_line_batched(
            (reference[0].position + initial_offset).as_dvec3(),
            reference[0].velocity.as_dvec3(),
            reference[0].velocity.as_dvec3(),
            duration,
            VolterraConfig {
                node_count: sample_count,
                maximum_picard_iterations: 24,
                maximum_endpoint_iterations: 8,
                damping: 0.70,
                relative_tolerance: 1.0e-6,
                minimum_longitudinal_speed: 1.0e-5,
                maximum_transverse_distance: f64::INFINITY,
            },
            |inputs, accelerations| {
                fill_planning_reference_accelerations(&jets, None, 0.0, inputs, accelerations)
            },
        );
        assert!(direct_solve.is_ok(), "{direct_solve:?}");

        let mut states = Vec::new();
        let mut bytes = Vec::new();
        append_dynamical_candidate_states(31, 32, &reference, &jets, None, &mut states, &mut bytes)
            .unwrap();
        assert_eq!(states.len(), sample_count);
        assert!(states.iter().all(|state| {
            state.velocity_distance[3].is_finite()
                && state.velocity_distance[3] <= PLANNING_TRAJECTORY_TUBE_RADIUS_METERS + 1.0e-3
        }));
    }
}
