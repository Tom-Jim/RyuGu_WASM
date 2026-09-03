/// No source traversal or target solve here. The render world builds the
/// topology, 56 moment/response banks and RHS combination on GPU.
pub(crate) fn build_planning_fmm_gpu_payload(
    batch: &PlanningCandidateBatch, model: u32, request_id: u64,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let densities = batch.density_models.get(model as usize*56..(model as usize+1)*56)?;
    let mass = *batch.density_model_masses.get(model as usize)?;
    if batch.basis_records.is_empty() || !batch.frequency_domain_source_radius.is_finite() || batch.frequency_domain_source_radius <= 0.0
        || !mass.is_finite() || mass <= 0.0
        || densities.iter().any(|d| !d.is_finite() || *d < 0.0) { return None; }
    Some(PlanningMethodPayload {
        request_id, method: Some(ActiveGravityMethod::Fmm), density_model: model,
        maximum_level: 56, total_mass: mass as f32,
        density_payload_preparation_ms: started.elapsed().as_secs_f64()*1e3,
        ..default()
    })
}

#[cfg(test)]
mod cpu_reference {
use super::*;
type PlanningCellKey = (u32, u32, u32);
type PlanningTargetKey = (i32, i32, i32);

/// Geometry cache shared by every density row in one planning batch.
/// Source topology, target leaf boxes and the state-to-target mapping are
/// immutable. Bounded fixed-target batches also cache all 56 unit-density
/// responses, including L2P and near-field P2P, before combining any RHS.
#[derive(Default)]
pub(crate) struct PlanningFmmWorkspace {
    batch_id: u64,
    radius: f64,
    levels: Vec<Vec<PlanningCellKey>>,
    level_offsets: Vec<u32>,
    index_maps: Vec<HashMap<PlanningCellKey, u32>>,
    particle_order: Vec<usize>,
    leaf_ranges: HashMap<PlanningCellKey, (u32, u32)>,
    target_keys: Vec<PlanningTargetKey>,
    state_target_indices: Vec<u32>,
    /// State-major [state][voxel][g, potential, Jacobian columns], in f64.
    response_basis: Vec<[f64; 13]>,
    basis_volumes: Vec<f64>,
    response_state_map: Arc<[u8]>,
}

#[derive(Clone, Debug)]
struct LocalExpansion {
    potential: f64,
    acceleration: DVec3,
    jacobian: bevy::math::DMat3,
    near_sources: Vec<usize>,
}

impl Default for LocalExpansion {
    fn default() -> Self {
        Self {
            potential: 0.0,
            acceleration: DVec3::ZERO,
            jacobian: bevy::math::DMat3::ZERO,
            near_sources: Vec::new(),
        }
    }
}

impl PlanningFmmWorkspace {
    fn rebuild(&mut self, batch: &PlanningCandidateBatch) -> Option<()> {
        self.response_basis.clear();
        self.basis_volumes.clear();
        self.response_state_map = Arc::from([]);
        self.batch_id = batch.batch_id;
        self.radius = batch
            .basis_records
            .iter()
            .map(|record| record_position(*record).length())
            .fold(0.0_f64, f64::max);
        if !self.radius.is_finite() || self.radius <= 0.0 || batch.basis_records.is_empty() {
            return None;
        }

        let leaf_grid = 1u32 << MAXIMUM_LEVEL;
        let source_leaf_key = |record: &PlanningBasisRecord| {
            let normalized = (record_position(*record) / self.radius + DVec3::ONE) * 0.5;
            let coordinate = |value: f64| {
                ((value.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32)
                    .min(leaf_grid - 1)
            };
            (
                coordinate(normalized.x),
                coordinate(normalized.y),
                coordinate(normalized.z),
            )
        };
        let leaf_keys = batch
            .basis_records
            .iter()
            .map(source_leaf_key)
            .collect::<Vec<_>>();
        self.levels = (0..=MAXIMUM_LEVEL)
            .map(|level| {
                let shift = MAXIMUM_LEVEL - level;
                let mut keys = leaf_keys
                    .iter()
                    .map(|key| (key.0 >> shift, key.1 >> shift, key.2 >> shift))
                    .collect::<Vec<_>>();
                keys.sort_unstable();
                keys.dedup();
                keys
            })
            .collect();
        self.level_offsets.clear();
        let mut node_count = 0_u32;
        for level in &self.levels {
            self.level_offsets.push(node_count);
            node_count = node_count.checked_add(level.len() as u32)?;
        }
        self.index_maps = self
            .levels
            .iter()
            .enumerate()
            .map(|(level, keys)| {
                keys.iter()
                    .enumerate()
                    .map(|(index, key)| (*key, self.level_offsets[level] + index as u32))
                    .collect()
            })
            .collect();

        self.particle_order = (0..batch.basis_records.len()).collect();
        self.particle_order.sort_by_key(|index| leaf_keys[*index]);
        self.leaf_ranges.clear();
        for (ordered, source_index) in self.particle_order.iter().copied().enumerate() {
            let key = leaf_keys[source_index];
            self.leaf_ranges
                .entry(key)
                .and_modify(|range| range.1 += 1)
                .or_insert((ordered as u32, 1));
        }

        let leaf_width = 2.0 * self.radius / leaf_grid as f64;
        let mut target_indices = HashMap::<PlanningTargetKey, u32>::new();
        self.target_keys.clear();
        self.state_target_indices.clear();
        self.state_target_indices.reserve(batch.states.len());
        for state in batch.states.iter().copied() {
            let position = state.body_position().as_dvec3();
            if !position.is_finite() {
                return None;
            }
            let coordinate = |value: f64| ((value + self.radius) / leaf_width).floor() as i32;
            let key = (
                coordinate(position.x),
                coordinate(position.y),
                coordinate(position.z),
            );
            let next = self.target_keys.len() as u32;
            let index = *target_indices.entry(key).or_insert_with(|| {
                self.target_keys.push(key);
                next
            });
            self.state_target_indices.push(index);
        }
        Some(())
    }
}

// Kept as an independent CPU oracle for the existing small numerical tests;
// production planning calls the GPU metadata builder below.
#[cfg(test)]
fn build_planning_fmm_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningFmmWorkspace,
) -> Option<PlanningMethodPayload> {
    // Every source/K/target sweep cell fits this bound (<=8192 targets).
    // Keep the million-state interactive stress workload on the streaming
    // path instead of allocating a multi-gigabyte response matrix.
    if batch.state_count() > 8192 {
        return build_planning_fmm_streaming_payload(batch, model, request_id, cache);
    }
    let started = bevy::platform::time::Instant::now();
    let mut geometry_basis_preparation_ms = 0.0;
    if cache.batch_id != batch.batch_id || cache.response_basis.is_empty() {
        cache.rebuild(batch)?;
        cache.build_response_basis(batch)?;
        geometry_basis_preparation_ms = started.elapsed().as_secs_f64() * 1.0e3;
    }
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    if densities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return None;
    }
    let mut primary = Vec::with_capacity(batch.state_count() * 96);
    for (state, basis) in batch
        .states
        .iter()
        .zip(cache.response_basis.chunks_exact(56))
    {
        let mut response = [0.0_f64; 13];
        for (column, density) in basis.iter().zip(densities) {
            for (total, value) in response.iter_mut().zip(column) {
                *total += f64::from(*density) * value;
            }
        }
        if response.iter().any(|value| !(*value as f32).is_finite()) {
            return None;
        }
        // Use the existing GPU L2P ABI with the expansion centered exactly on
        // its target. P2P is already in the basis; no density-dependent source
        // scans, tree moments, translations or near-particle packing remain.
        let position = state.body_position();
        push_f32s(&mut primary, [position.x, position.y, position.z, 0.0]);
        push_f32s(
            &mut primary,
            std::array::from_fn::<_, 4, _>(|i| response[i] as f32),
        );
        for column in response[4..].chunks_exact(3) {
            push_f32s(
                &mut primary,
                [column[0] as f32, column[1] as f32, column[2] as f32, 0.0],
            );
        }
        primary.extend_from_slice(&[0u8; 16]);
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::Fmm),
        density_model: model,
        primary: Arc::from(primary),
        secondary: Arc::clone(&cache.response_state_map),
        item_count: batch.state_count() as u32,
        secondary_count: 0,
        maximum_level: 1, // One GPU L2P read; the 56-term CPU mix is timed below.
        total_mass: cache
            .basis_volumes
            .iter()
            .zip(densities)
            .map(|(volume, density)| volume * f64::from(*density))
            .sum::<f64>() as f32,
        geometry_basis_preparation_ms,
        density_payload_preparation_ms: (started.elapsed().as_secs_f64() * 1.0e3
            - geometry_basis_preparation_ms)
            .max(0.0),
        ..default()
    })
}

impl PlanningFmmWorkspace {
    fn build_response_basis(&mut self, batch: &PlanningCandidateBatch) -> Option<()> {
        let mut voxel_sources = vec![Vec::new(); 56];
        self.basis_volumes = vec![0.0; 56];
        // Visit the source bank once to partition P2M work by basis column.
        for (index, record) in batch.basis_records.iter().enumerate() {
            let voxel = record.voxel_index as usize;
            let volume = f64::from(record.position_volume[3]);
            if !volume.is_finite() || volume < 0.0 {
                return None;
            }
            voxel_sources.get_mut(voxel)?.push(index);
            self.basis_volumes[voxel] += volume;
        }
        let mut responses = vec![[0.0_f64; 13]; batch.state_count().checked_mul(56)?];
        let leaf_level = MAXIMUM_LEVEL as usize;
        let leaf_grid = 1u32 << MAXIMUM_LEVEL;
        for (voxel, indices) in voxel_sources.iter().enumerate() {
            let mut moments = self
                .levels
                .iter()
                .map(|level| vec![MomentAccumulator::default(); level.len()])
                .collect::<Vec<_>>();
            for &index in indices {
                let record = batch.basis_records[index];
                let position = record_position(record);
                let normalized = (position / self.radius + DVec3::ONE) * 0.5;
                let coordinate = |value: f64| {
                    ((value.clamp(0.0, 1.0 - f64::EPSILON) * f64::from(leaf_grid)) as u32)
                        .min(leaf_grid - 1)
                };
                let key = (
                    coordinate(normalized.x),
                    coordinate(normalized.y),
                    coordinate(normalized.z),
                );
                let global = *self.index_maps[leaf_level].get(&key)?;
                moments[leaf_level][(global - self.level_offsets[leaf_level]) as usize]
                    .add(position, f64::from(record.position_volume[3]));
            }
            for level in (1..=leaf_level).rev() {
                let (parents, children) = moments.split_at_mut(level);
                for (index, &child) in children[0].iter().enumerate() {
                    let key = self.levels[level][index];
                    let parent =
                        *self.index_maps[level - 1].get(&(key.0 / 2, key.1 / 2, key.2 / 2))?;
                    parents[level - 1][(parent - self.level_offsets[level - 1]) as usize]
                        .merge(child);
                }
            }
            for (state, target) in batch.states.iter().enumerate() {
                let observer = target.body_position().as_dvec3();
                let mut local = LocalExpansion::default();
                // Center each fixed-target basis expansion on the actual
                // observer. A quadratic potential at a shared leaf center has
                // a constant Hessian, leaving an O(target offset / distance)
                // gradient error even as source quadrature is refined.
                // This extra M2L work is paid only during the 56-basis build.
                accumulate_target_local(self, &moments, 0, (0, 0, 0), observer, 0.0, &mut local)?;
                let mut acceleration = local.acceleration;
                let mut potential = local.potential;
                let mut jacobian = local.jacobian;
                for &source in &local.near_sources {
                    let record = batch.basis_records[source];
                    if record.voxel_index as usize != voxel {
                        continue;
                    }
                    let displacement = record_position(record) - observer;
                    let radius2 = displacement.length_squared().max(1.0e-8);
                    let inverse_radius = radius2.sqrt().recip();
                    let mass = f64::from(record.position_volume[3]);
                    let inverse_radius3 = inverse_radius / radius2;
                    acceleration += mass * displacement * inverse_radius3;
                    potential += mass * inverse_radius;
                    let column = |axis: DVec3, component: f64| {
                        -mass * inverse_radius3 * axis
                            + 3.0 * mass * inverse_radius3 / radius2 * displacement * component
                    };
                    jacobian += bevy::math::DMat3::from_cols(
                        column(DVec3::X, displacement.x),
                        column(DVec3::Y, displacement.y),
                        column(DVec3::Z, displacement.z),
                    );
                }
                let response = &mut responses[state * 56 + voxel];
                response[..3].copy_from_slice(&acceleration.to_array());
                response[3] = potential;
                response[4..].copy_from_slice(&jacobian.to_cols_array());
                for value in response {
                    *value *= f64::from(G);
                    if !value.is_finite() {
                        return None;
                    }
                }
            }
        }
        self.response_basis = responses;
        self.response_state_map = Arc::from(
            (0..batch.state_count() as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        Some(())
    }
}

fn build_planning_fmm_streaming_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningFmmWorkspace,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let mut geometry_basis_preparation_ms = 0.0;
    if cache.batch_id != batch.batch_id || cache.levels.len() != MAXIMUM_LEVEL as usize + 1 {
        let one_time_started = bevy::platform::time::Instant::now();
        cache.rebuild(batch)?;
        geometry_basis_preparation_ms = one_time_started.elapsed().as_secs_f64() * 1.0e3;
    }
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    let masses = batch
        .basis_records
        .iter()
        .map(|record| {
            let density = f64::from(*densities.get(record.voxel_index as usize)?);
            let mass = f64::from(record.position_volume[3]) * density;
            (mass.is_finite() && mass > 0.0).then_some(mass)
        })
        .collect::<Option<Vec<_>>>()?;

    // P2M at occupied leaves followed by exact raw-moment M2M aggregation.
    let mut moments = cache
        .levels
        .iter()
        .map(|level| vec![MomentAccumulator::default(); level.len()])
        .collect::<Vec<_>>();
    let leaf_level = MAXIMUM_LEVEL as usize;
    let leaf_grid = 1u32 << MAXIMUM_LEVEL;
    for (record, mass) in batch.basis_records.iter().zip(masses.iter().copied()) {
        let position = record_position(*record);
        let normalized = (position / cache.radius + DVec3::ONE) * 0.5;
        let coordinate = |value: f64| {
            ((value.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32).min(leaf_grid - 1)
        };
        let key = (
            coordinate(normalized.x),
            coordinate(normalized.y),
            coordinate(normalized.z),
        );
        let global_index = *cache.index_maps[leaf_level].get(&key)?;
        moments[leaf_level][(global_index - cache.level_offsets[leaf_level]) as usize]
            .add(position, mass);
    }
    for level in (1..=leaf_level).rev() {
        let children = moments[level].clone();
        for (index, child) in children.into_iter().enumerate() {
            let key = cache.levels[level][index];
            let parent_key = (key.0 / 2, key.1 / 2, key.2 / 2);
            let global_parent = *cache.index_maps[level - 1].get(&parent_key)?;
            moments[level - 1][(global_parent - cache.level_offsets[level - 1]) as usize]
                .merge(child);
        }
    }

    // One M2L build per occupied target leaf; all states in that box reuse it.
    let leaf_half = cache.radius / leaf_grid as f64;
    let mut local_bytes = Vec::with_capacity(cache.target_keys.len() * 96);
    let mut near_particle_bytes = Vec::new();
    let mut total_interactions = 0_u64;
    for &target_key in &cache.target_keys {
        let target_center = target_cell_center(target_key, cache.radius, leaf_grid);
        let mut local = LocalExpansion::default();
        accumulate_target_local(
            cache,
            &moments,
            0,
            (0, 0, 0),
            target_center,
            leaf_half,
            &mut local,
        )?;
        let near_start = (near_particle_bytes.len() / 16) as u32;
        for source_index in local.near_sources.iter().copied() {
            let record = *batch.basis_records.get(source_index)?;
            let position = record_position(record);
            push_f32s(
                &mut near_particle_bytes,
                [
                    position.x as f32,
                    position.y as f32,
                    position.z as f32,
                    masses[source_index] as f32,
                ],
            );
        }
        let near_count = local.near_sources.len() as u32;
        total_interactions = total_interactions.saturating_add(u64::from(near_count) + 1);
        let acceleration = local.acceleration * f64::from(G);
        let potential = local.potential * f64::from(G);
        let jacobian = local.jacobian * f64::from(G);
        push_f32s(
            &mut local_bytes,
            [
                target_center.x as f32,
                target_center.y as f32,
                target_center.z as f32,
                leaf_half as f32,
            ],
        );
        push_f32s(
            &mut local_bytes,
            [
                acceleration.x as f32,
                acceleration.y as f32,
                acceleration.z as f32,
                potential as f32,
            ],
        );
        for column in [jacobian.x_axis, jacobian.y_axis, jacobian.z_axis] {
            push_f32s(
                &mut local_bytes,
                [column.x as f32, column.y as f32, column.z as f32, 0.0],
            );
        }
        for value in [near_start, near_count, 0, 0] {
            local_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    // One raw-u32 buffer keeps the state map and packed P2P records under a
    // single browser storage binding.
    let mut secondary =
        Vec::with_capacity(cache.state_target_indices.len() * 4 + near_particle_bytes.len());
    for index in cache.state_target_indices.iter().copied() {
        secondary.extend_from_slice(&index.to_le_bytes());
    }
    secondary.extend_from_slice(&near_particle_bytes);
    let average_interactions = total_interactions
        .div_ceil(cache.target_keys.len().max(1) as u64)
        .min(u64::from(u32::MAX)) as u32;

    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::Fmm),
        density_model: model,
        primary: Arc::from(local_bytes),
        secondary: Arc::from(secondary),
        item_count: cache.target_keys.len() as u32,
        secondary_count: (near_particle_bytes.len() / 16) as u32,
        // Planning uses this otherwise-unused field to report the measured
        // target-cell average (one local expansion plus P2P interactions).
        maximum_level: average_interactions,
        total_mass: masses.iter().sum::<f64>() as f32,
        geometry_basis_preparation_ms,
        density_payload_preparation_ms: (started.elapsed().as_secs_f64() * 1.0e3
            - geometry_basis_preparation_ms)
            .max(0.0),
        ..default()
    })
}

#[allow(clippy::too_many_arguments)]
fn accumulate_target_local(
    cache: &PlanningFmmWorkspace,
    moments: &[Vec<MomentAccumulator>],
    level: usize,
    key: PlanningCellKey,
    target_center: DVec3,
    target_half: f64,
    local: &mut LocalExpansion,
) -> Option<()> {
    let global = *cache.index_maps.get(level)?.get(&key)?;
    let moment = *moments
        .get(level)?
        .get((global - cache.level_offsets[level]) as usize)?;
    if !moment.mass.is_finite() || moment.mass <= 0.0 {
        return Some(());
    }
    let grid = 1u32 << level;
    let source_half = cache.radius / grid as f64;
    let source_center = DVec3::new(
        -cache.radius + (key.0 as f64 + 0.5) * 2.0 * source_half,
        -cache.radius + (key.1 as f64 + 0.5) * 2.0 * source_half,
        -cache.radius + (key.2 as f64 + 0.5) * 2.0 * source_half,
    );
    let distance = source_center.distance(target_center);
    let expansion_radius = 3.0_f64.sqrt() * (source_half + target_half);
    // Exact-target basis locals use a tighter source opening ratio for the
    // Hessian; streaming target-cell payloads retain their original setting.
    let opening_ratio = if target_half == 0.0 {
        0.05
    } else {
        f64::from(THETA)
    };
    let accepted = level > 0
        && distance > 1.01 * expansion_radius
        && expansion_radius / distance.max(1.0e-12) < opening_ratio;
    if accepted {
        accumulate_multipole(moment, target_center, local)?;
        return Some(());
    }
    if level == MAXIMUM_LEVEL as usize {
        let (start, count) = cache.leaf_ranges.get(&key).copied().unwrap_or((0, 0));
        local.near_sources.extend(
            cache
                .particle_order
                .get(start as usize..start.saturating_add(count) as usize)?
                .iter()
                .copied(),
        );
        return Some(());
    }
    let child_level = level + 1;
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let child = (key.0 * 2 + dx, key.1 * 2 + dy, key.2 * 2 + dz);
                if cache.index_maps[child_level].contains_key(&child) {
                    accumulate_target_local(
                        cache,
                        moments,
                        child_level,
                        child,
                        target_center,
                        target_half,
                        local,
                    )?;
                }
            }
        }
    }
    Some(())
}

fn accumulate_multipole(
    moment: MomentAccumulator,
    observer: DVec3,
    local: &mut LocalExpansion,
) -> Option<()> {
    let mass = moment.mass;
    let com = moment.first / mass;
    let [x, y, z] = com.to_array();
    let central = [
        moment.second[0] - mass * x * x,
        moment.second[1] - mass * x * y,
        moment.second[2] - mass * x * z,
        moment.second[3] - mass * y * y,
        moment.second[4] - mass * y * z,
        moment.second[5] - mass * z * z,
    ];
    let trace = central[0] + central[3] + central[5];
    let qx = DVec3::new(3.0 * central[0] - trace, 3.0 * central[1], 3.0 * central[2]);
    let qy = DVec3::new(3.0 * central[1], 3.0 * central[3] - trace, 3.0 * central[4]);
    let qz = DVec3::new(3.0 * central[2], 3.0 * central[4], 3.0 * central[5] - trace);
    let displacement = com - observer;
    let radius2 = displacement.length_squared().max(1.0e-16);
    let inverse_radius = radius2.sqrt().recip();
    let inverse_radius3 = inverse_radius / radius2;
    let inverse_radius5 = inverse_radius3 / radius2;
    let inverse_radius7 = inverse_radius5 / radius2;
    let inverse_radius9 = inverse_radius7 / radius2;
    let qd = qx * displacement.x + qy * displacement.y + qz * displacement.z;
    let scalar = displacement.dot(qd);
    let acceleration = mass * displacement * inverse_radius3 - qd * inverse_radius5
        + 2.5 * scalar * displacement * inverse_radius7;
    let potential = mass * inverse_radius + 0.5 * scalar * inverse_radius5;
    let diagonal = -mass * inverse_radius3 - 2.5 * scalar * inverse_radius7;
    let outer_scale = 3.0 * mass * inverse_radius5 + 17.5 * scalar * inverse_radius9;
    let mixed_scale = -5.0 * inverse_radius7;
    let column = |axis: DVec3, q: DVec3, component: f64, qd_component: f64| {
        axis * diagonal
            + displacement * (outer_scale * component)
            + q * inverse_radius5
            + (qd * component + displacement * qd_component) * mixed_scale
    };
    let jacobian = bevy::math::DMat3::from_cols(
        column(DVec3::X, qx, displacement.x, qd.x),
        column(DVec3::Y, qy, displacement.y, qd.y),
        column(DVec3::Z, qz, displacement.z, qd.z),
    );
    if !acceleration.is_finite() || !potential.is_finite() || !jacobian.is_finite() {
        return None;
    }
    local.acceleration += acceleration;
    local.potential += potential;
    local.jacobian += jacobian;
    Some(())
}

fn target_cell_center(key: PlanningTargetKey, radius: f64, leaf_grid: u32) -> DVec3 {
    let width = 2.0 * radius / leaf_grid as f64;
    DVec3::new(
        -radius + (key.0 as f64 + 0.5) * width,
        -radius + (key.1 as f64 + 0.5) * width,
        -radius + (key.2 as f64 + 0.5) * width,
    )
}

fn record_position(record: PlanningBasisRecord) -> DVec3 {
    DVec3::new(
        f64::from(record.position_volume[0]),
        f64::from(record.position_volume[1]),
        f64::from(record.position_volume[2]),
    )
}

fn push_f32s<const N: usize>(bytes: &mut Vec<u8>, values: [f32; N]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod planning_basis_tests {
    use super::*;

    fn fixture() -> PlanningCandidateBatch {
        let records = (0..56)
            .flat_map(|voxel| {
                (0..2).map(move |source| PlanningBasisRecord {
                    position_volume: [
                        (voxel % 7) as f32 * 2.0 - 6.0 + source as f32 * 0.1,
                        (voxel / 7) as f32 * 2.0 - 7.0,
                        source as f32 - 0.5,
                        1.0 + voxel as f32 / 56.0,
                    ],
                    voxel_index: voxel,
                })
            })
            .collect::<Vec<_>>();
        let states = [[0.1, 0.2, 15.0, 0.0], [1500.0, 400.0, 800.0, 1.0]].map(|position_time| {
            PlanningCandidateState {
                position_time,
                ..default()
            }
        });
        let densities = (0..4)
            .flat_map(|model| {
                (0..56).map(move |voxel| {
                    let a = 0.5 + voxel as f32 / 56.0;
                    let b = 2.0 - voxel as f32 / 112.0;
                    match model {
                        0 => a,
                        1 => b,
                        2 => a + b,
                        _ => 0.0,
                    }
                })
            })
            .collect::<Vec<_>>();
        PlanningCandidateBatch {
            batch_id: 1,
            candidate_count: 1,
            samples_per_candidate: 2,
            density_model_count: 4,
            basis_records: Arc::from(records),
            states: Arc::from(states),
            density_models: Arc::from(densities),
            ..default()
        }
    }

    fn response(payload: &PlanningMethodPayload, state: usize) -> [f64; 13] {
        let bytes = &payload.primary[state * 96 + 16..state * 96 + 80];
        let floats = bytes
            .chunks_exact(4)
            .map(|bytes| f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
            .collect::<Vec<_>>();
        [
            floats[0], floats[1], floats[2], floats[3], floats[4], floats[5], floats[6], floats[8],
            floats[9], floats[10], floats[12], floats[13], floats[14],
        ]
    }

    #[test]
    fn cached_fmm_is_linear_and_agrees_with_direct_gravity_and_jacobian() {
        let batch = fixture();
        let mut cache = PlanningFmmWorkspace::default();
        let a = build_planning_fmm_payload(&batch, 0, 1, &mut cache).unwrap();
        let pointer = cache.response_basis.as_ptr();
        let b = build_planning_fmm_payload(&batch, 1, 2, &mut cache).unwrap();
        let sum = build_planning_fmm_payload(&batch, 2, 3, &mut cache).unwrap();
        let zero = build_planning_fmm_payload(&batch, 3, 4, &mut cache).unwrap();
        assert_eq!(cache.response_basis.as_ptr(), pointer);
        assert_eq!(b.geometry_basis_preparation_ms, 0.0);
        assert_eq!(sum.secondary_count, 0);
        for state in 0..batch.state_count() {
            let (ra, rb, rs) = (
                response(&a, state),
                response(&b, state),
                response(&sum, state),
            );
            for i in 0..13 {
                assert!(
                    (ra[i] + rb[i] - rs[i]).abs() < 3e-7 * (ra[i].abs() + rb[i].abs()).max(1e-25)
                );
            }
            assert_eq!(response(&zero, state), [0.0; 13]);
            let (gravity, jacobian) = crate::cpu::planning::evaluate_planning_reference_field(
                batch.states[state].body_position().as_dvec3(),
                &batch.basis_records,
                &batch.density_models[..56],
            )
            .unwrap();
            let actual_g = DVec3::from_array(ra[..3].try_into().unwrap());
            let actual_j = bevy::math::DMat3::from_cols_array(&ra[4..].try_into().unwrap());
            assert!(actual_g.distance(gravity) / gravity.length() < 1e-3);
            let squared_norm = |matrix: bevy::math::DMat3| {
                matrix.to_cols_array().iter().map(|x| x * x).sum::<f64>()
            };
            assert!((squared_norm(actual_j - jacobian) / squared_norm(jacobian)).sqrt() < 1e-2);
        }
    }

    #[test]
    fn new_batch_invalidates_response_basis_and_negative_density_is_rejected() {
        let mut batch = fixture();
        let mut cache = PlanningFmmWorkspace::default();
        let old = build_planning_fmm_payload(&batch, 0, 1, &mut cache).unwrap();
        Arc::make_mut(&mut batch.states)[0].position_time[2] += 4.0;
        batch.batch_id += 1;
        let new = build_planning_fmm_payload(&batch, 0, 2, &mut cache).unwrap();
        assert_ne!(response(&old, 0), response(&new, 0));
        Arc::make_mut(&mut batch.density_models)[0] = -1.0;
        assert!(build_planning_fmm_payload(&batch, 0, 3, &mut cache).is_none());
    }

    #[test]
    fn jacobian_varies_between_observers_in_the_same_target_leaf() {
        let batch = PlanningCandidateBatch {
            batch_id: 1,
            candidate_count: 1,
            samples_per_candidate: 2,
            density_model_count: 1,
            basis_records: Arc::from([PlanningBasisRecord {
                position_volume: [100.0, 0.0, 0.0, 2.0],
                voxel_index: 0,
            }]),
            density_models: Arc::from(vec![1.0; 56]),
            states: Arc::from([-140.0, -139.0].map(|x| PlanningCandidateState {
                position_time: [x, 0.0, 0.0, 0.0],
                ..default()
            })),
            ..default()
        };
        let mut cache = PlanningFmmWorkspace::default();
        let payload = build_planning_fmm_payload(&batch, 0, 1, &mut cache).unwrap();
        assert_eq!(cache.state_target_indices[0], cache.state_target_indices[1]);
        assert_ne!(response(&payload, 0)[4], response(&payload, 1)[4]);
        for state in 0..2 {
            let (_, reference) = crate::cpu::planning::evaluate_planning_reference_field(
                batch.states[state].body_position().as_dvec3(),
                &batch.basis_records,
                &batch.density_models,
            )
            .unwrap();
            assert!((response(&payload, state)[4] / reference.x_axis.x - 1.0).abs() < 2e-7);
        }
    }
}


}
