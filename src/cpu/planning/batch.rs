use crate::cpu::frequency_domain::AggregatedGravitySource;
use crate::cpu::inversion::{build_planning_dynamics_tree, build_voxel_basis_sources};
use crate::interface::components::*;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use std::sync::Arc;

const PLANNING_INITIAL_PERTURBATION_FRACTION: f32 = 0.70;

pub(crate) struct PlanningBatchBuilder {
    profile: PlanningWorkloadProfile,
    run_id: u64,
    capture_id: u64,
    capture_epoch: u64,
    source_hash: u64,
    source_count: u32,
    body_radius: f32,
    frequency_domain_source_radius: f32,
    candidate_count: u32,
    density_model_count: u32,
    samples_per_candidate: u32,
    next_sample: usize,
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
    /// Latest propagated state of every candidate; the next slice starts here.
    candidate_positions: Vec<DVec3>,
    candidate_velocities: Vec<DVec3>,
    /// Point sources of the canonical FMM dynamics field. They are uploaded to
    /// the Worker once per planning run; slices only carry candidate states.
    dynamics_sources: Vec<(DVec3, f64)>,
    /// The Worker acknowledged this run's source upload and no channel reset
    /// has happened since, so slices may reference the uploaded sources.
    sources_prepared: bool,
    pending: Option<PendingCandidateRequest>,
    worker_request_id: u64,
    /// Candidate window for the current sample span when the integration cap
    /// cannot cover every candidate in one `propagate_candidates` call.
    next_candidate: usize,
    /// Last Worker/slice failure detail for the planning status line.
    pub(crate) last_error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct PlanningReferenceJet {
    simulation_time_seconds: f64,
    body_rotation: DQuat,
    world_position: DVec3,
    world_acceleration: DVec3,
    world_jacobian: DMat3,
}

/// Outstanding request of this builder on the candidates channel.
enum PendingCandidateRequest {
    /// `prepare_candidate_sources` upload; must complete before any slice.
    Prepare {
        snapshot: crate::cpp_backend::BackendCandidatesSnapshot,
    },
    /// `propagate_candidates` slice against the uploaded sources.
    Slice {
        snapshot: crate::cpp_backend::BackendCandidatesSnapshot,
        start_sample: usize,
        end_sample: usize,
        start_candidate: usize,
        end_candidate: usize,
        started: bevy::platform::time::Instant,
    },
}

impl PendingCandidateRequest {
    fn snapshot(&self) -> crate::cpp_backend::BackendCandidatesSnapshot {
        match self {
            Self::Prepare { snapshot } | Self::Slice { snapshot, .. } => *snapshot,
        }
    }
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
        dimensions: (u32, u32, u32),
        voxel_size: f32,
        reference_knots: &[TrajectoryInversionKnot],
        voxels: &[InvertedDensityVoxel],
        source: &AggregatedGravitySource,
    ) -> Option<Self> {
        let started = bevy::platform::time::Instant::now();
        let (candidate_count, density_model_count, samples_per_candidate) = dimensions;
        if candidate_count == 0
            || candidate_count > PLANNING_CANDIDATE_COUNT
            || voxels.len() != 56
            || density_model_count == 0
            || samples_per_candidate < 2
        {
            return None;
        }
        let basis = build_voxel_basis_sources(voxels, source, voxel_size)?;
        // All planning backends use the same fixed-length samples generated
        // from the equation-(185) reference arc.  Candidates are perturbed
        // copies of this arc; no backend-specific radial sampling is involved.
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
        let frequency_domain_source_radius =
            basis_records.iter().fold(0.0_f32, |radius, record| {
                radius.max(
                    Vec3::new(
                        record.position_volume[0],
                        record.position_volume[1],
                        record.position_volume[2],
                    )
                    .length(),
                )
            });
        if !frequency_domain_source_radius.is_finite() || frequency_domain_source_radius <= 0.0 {
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
            mix_hash(capture_id, 0x1840_d315_7a11_5eed),
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
        // mass/centroid representation for candidate propagation.
        let dynamics_tree =
            build_planning_dynamics_tree(&canonical_basis_records, density_models.get(..56)?)?;
        let reference_jets = build_planning_reference_jets(&reference_samples);
        let first = *reference_samples.first()?;
        let mut candidate_positions = Vec::with_capacity(candidate_count as usize);
        for candidate in 0..candidate_count {
            let (radius, phase, harmonic, phase_rate) =
                candidate_perturbation_parameters(candidate, candidate_count);
            let offset = candidate_initial_offset(
                first,
                0,
                samples_per_candidate,
                radius,
                phase,
                harmonic,
                phase_rate,
            )?;
            candidate_positions.push((first.position + offset).as_dvec3());
        }
        let candidate_velocities = vec![first.velocity.as_dvec3(); candidate_count as usize];
        let dynamics_sources = dynamics_tree.sources().to_vec();
        let state_count = candidate_count as usize * samples_per_candidate as usize;
        Some(Self {
            profile,
            run_id,
            capture_id,
            capture_epoch,
            source_hash,
            source_count: requested_source_count,
            body_radius: source.radius as f32,
            frequency_domain_source_radius,
            candidate_count,
            density_model_count,
            samples_per_candidate,
            next_sample: 0,
            preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
            reference_samples,
            reference_states,
            states: vec![PlanningCandidateState::default(); state_count],
            gpu_position_bytes: vec![0; state_count * 16],
            density_models,
            density_model_masses,
            density_seed,
            target_mass,
            basis_records,
            basis_hash,
            reference_jets,
            candidate_positions,
            candidate_velocities,
            dynamics_sources,
            sources_prepared: false,
            pending: None,
            worker_request_id: 0,
            next_candidate: 0,
            last_error: None,
        })
    }

    pub(crate) fn matches(
        &self,
        profile: PlanningWorkloadProfile,
        run_id: u64,
        capture_id: u64,
        source_hash: u64,
        requested_source_count: u32,
        dimensions: (u32, u32, u32),
    ) -> bool {
        self.profile == profile
            && self.run_id == run_id
            && self.capture_id == capture_id
            && self.source_hash == source_hash
            && self.source_count == requested_source_count
            && (
                self.candidate_count,
                self.density_model_count,
                self.samples_per_candidate,
            ) == dimensions
    }

    /// Uploads the dynamics sources once, then submits or collects one bounded
    /// propagation slice per call.
    ///
    /// All candidates share one C++ FMM time slice on the numerical Worker.
    /// Returns `false` only when the batch is unrecoverable; an idle Worker
    /// (or a native build, which has no Worker) simply reports no progress.
    pub(crate) fn advance(
        &mut self,
        propagation_budget: u32,
        channel: &crate::cpp_backend::BackendCandidatesChannel,
    ) -> bool {
        if let Some(pending) = self.pending.take() {
            match channel.take() {
                None if !channel.is_idle() => {
                    // The Worker is still working on it.
                    self.pending = Some(pending);
                    return true;
                }
                // Cancelling an experiment clears the channel, so a free
                // channel with nothing delivered means this request was
                // discarded. The upload is repeated before the next slice so a
                // slice never depends on an acknowledgement from before the
                // reset; fall through and re-issue below.
                None => self.sources_prepared = false,
                Some(packet) => {
                    if packet.snapshot != pending.snapshot()
                        || packet.snapshot.epoch != self.capture_epoch
                    {
                        return false;
                    }
                    match pending {
                        PendingCandidateRequest::Prepare { .. } => {
                            if let Err(message) = packet.result {
                                self.last_error = Some(message);
                                return false;
                            }
                            self.sources_prepared = true;
                            return true;
                        }
                        PendingCandidateRequest::Slice {
                            start_sample,
                            end_sample,
                            start_candidate,
                            end_candidate,
                            started,
                            ..
                        } => {
                            let trajectory = match packet.result {
                                Ok(values)
                                    if values.len()
                                        == (end_candidate - start_candidate)
                                            * (end_sample - start_sample + 1)
                                            * 6
                                        && values.iter().all(|value| value.is_finite()) =>
                                {
                                    values
                                }
                                Ok(_) => {
                                    self.last_error = Some(
                                        "Planning candidate slice returned an invalid trajectory layout."
                                            .into(),
                                    );
                                    return false;
                                }
                                Err(message) => {
                                    self.last_error = Some(message);
                                    return false;
                                }
                            };
                            if !self.apply_candidate_slice(
                                start_sample,
                                end_sample,
                                start_candidate,
                                end_candidate,
                                &trajectory,
                            ) {
                                self.last_error = Some(
                                    "Planning candidate slice could not be applied to the batch."
                                        .into(),
                                );
                                return false;
                            }
                            self.preparation_ms += started.elapsed().as_secs_f64() * 1.0e3;
                            return true;
                        }
                    }
                }
            }
        }

        let sample_count = self.reference_samples.len();
        if sample_count < 2 {
            return false;
        }
        if !self.sources_prepared {
            let snapshot = self.next_snapshot();
            return match crate::cpp_backend::request_prepare_candidate_sources(
                channel,
                snapshot,
                &self.dynamics_sources,
            ) {
                Ok(true) => {
                    self.pending = Some(PendingCandidateRequest::Prepare { snapshot });
                    true
                }
                Ok(false) => true,
                Err(message) => {
                    self.last_error = Some(message);
                    false
                }
            };
        }
        let start_sample = self.next_sample.min(sample_count - 1);
        let end_sample =
            (start_sample + propagation_budget.max(1) as usize).min(sample_count - 1);
        if start_sample == end_sample {
            self.next_sample = end_sample;
            return true;
        }
        let step_count = (end_sample - start_sample).max(1);
        let candidate_limit = (PLANNING_MAX_PROPAGATION_INTEGRATIONS as usize / step_count).max(1);
        let start_candidate = self.next_candidate.min(self.candidate_count as usize);
        let end_candidate =
            (start_candidate + candidate_limit).min(self.candidate_count as usize);
        if start_candidate >= end_candidate {
            self.next_candidate = 0;
            return true;
        }
        let Some(request) =
            self.serialize_candidate_slice(start_sample, end_sample, start_candidate, end_candidate)
        else {
            self.last_error = Some("Planning candidate slice could not be serialized.".into());
            return false;
        };
        let snapshot = self.next_snapshot();
        match crate::cpp_backend::request_candidates(channel, snapshot, &request) {
            Ok(true) => {
                self.pending = Some(PendingCandidateRequest::Slice {
                    snapshot,
                    start_sample,
                    end_sample,
                    start_candidate,
                    end_candidate,
                    started: bevy::platform::time::Instant::now(),
                });
                true
            }
            Ok(false) => true,
            Err(message) => {
                self.last_error = Some(message);
                false
            }
        }
    }

    fn next_snapshot(&mut self) -> crate::cpp_backend::BackendCandidatesSnapshot {
        self.worker_request_id = self.worker_request_id.wrapping_add(1).max(1);
        crate::cpp_backend::BackendCandidatesSnapshot {
            request_id: self.run_id.rotate_left(23) ^ self.worker_request_id,
            epoch: self.capture_epoch,
        }
    }

    fn serialize_candidate_slice(
        &self,
        start_sample: usize,
        end_sample: usize,
        start_candidate: usize,
        end_candidate: usize,
    ) -> Option<String> {
        let candidate_positions: Vec<_> = self
            .candidate_positions
            .get(start_candidate..end_candidate)?
            .iter()
            .map(|position| position.to_array())
            .collect();
        let candidate_velocities: Vec<_> = self
            .candidate_velocities
            .get(start_candidate..end_candidate)?
            .iter()
            .map(|velocity| velocity.to_array())
            .collect();
        let reference_jets = self.reference_jets.get(start_sample..=end_sample)?;
        #[derive(serde::Serialize)]
        struct CandidateJet {
            time: f64,
            rotation: [f64; 4],
            position: [f64; 3],
            acceleration: [f64; 3],
            jacobian: [f64; 9],
        }
        let jets: Vec<_> = reference_jets
            .iter()
            .map(|jet| CandidateJet {
                time: jet.simulation_time_seconds,
                rotation: jet.body_rotation.to_array(),
                position: jet.world_position.to_array(),
                acceleration: jet.world_acceleration.to_array(),
                jacobian: jet.world_jacobian.to_cols_array(),
            })
            .collect();
        // An empty `sources` list tells the backend to use the sources this
        // run uploaded with `prepare_candidate_sources`.
        #[derive(serde::Serialize)]
        struct CandidateRequest<'a> {
            positions: &'a [[f64; 3]],
            velocities: &'a [[f64; 3]],
            jets: &'a [CandidateJet],
            sources: [([f64; 3], f64); 0],
        }
        serde_json::to_string(&CandidateRequest {
            positions: &candidate_positions,
            velocities: &candidate_velocities,
            jets: &jets,
            sources: [],
        })
        .ok()
    }

    fn apply_candidate_slice(
        &mut self,
        start_sample: usize,
        end_sample: usize,
        start_candidate: usize,
        end_candidate: usize,
        trajectory: &[f64],
    ) -> bool {
        let local_sample_count = end_sample - start_sample + 1;
        let local_candidate_count = end_candidate - start_candidate;
        for local_candidate in 0..local_candidate_count {
            let candidate = start_candidate + local_candidate;
            let candidate_start = local_candidate * local_sample_count * 6;
            for local_sample in 0..local_sample_count {
                if start_sample != 0 && local_sample == 0 {
                    continue;
                }
                let values_start = candidate_start + local_sample * 6;
                if write_dynamical_candidate_sample(
                    candidate as u32,
                    start_sample + local_sample,
                    &self.reference_samples,
                    &trajectory[values_start..values_start + 6],
                    &mut self.states,
                    &mut self.gpu_position_bytes,
                )
                .is_none()
                {
                    return false;
                }
            }
            let final_start = candidate_start + (local_sample_count - 1) * 6;
            self.candidate_positions[candidate] =
                DVec3::from_slice(&trajectory[final_start..final_start + 3]);
            self.candidate_velocities[candidate] =
                DVec3::from_slice(&trajectory[final_start + 3..final_start + 6]);
        }
        if end_candidate >= self.candidate_count as usize {
            self.next_candidate = 0;
            self.next_sample = end_sample;
        } else {
            self.next_candidate = end_candidate;
        }
        true
    }

    pub(crate) fn preparation_progress(&self) -> f64 {
        let samples = self.reference_samples.len() as f64;
        let candidates = f64::from(self.candidate_count.max(1));
        ((self.next_sample as f64) * candidates + self.next_candidate as f64)
            / (samples * candidates).max(1.0)
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.next_candidate == 0 && self.next_sample + 1 >= self.reference_samples.len()
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
                frequency_domain_source_radius: self.frequency_domain_source_radius,
                reference_states: Arc::from(self.reference_states),
                states: Arc::from(self.states),
                gpu_position_bytes: Arc::from(self.gpu_position_bytes),
                density_models: Arc::from(self.density_models),
                density_model_masses: Arc::from(self.density_model_masses),
                density_seed: self.density_seed,
                target_mass: self.target_mass,
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
