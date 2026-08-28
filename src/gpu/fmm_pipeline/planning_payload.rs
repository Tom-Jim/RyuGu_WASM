type PlanningCellKey = (u32, u32, u32);
type PlanningTargetKey = (i32, i32, i32);

/// Geometry cache shared by every density row in one planning batch.
/// Source topology, target leaf boxes and the state-to-target mapping are
/// immutable; only P2M/M2M moments and M2L coefficients change with density.
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

pub(crate) fn build_planning_fmm_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
    cache: &mut PlanningFmmWorkspace,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    if cache.batch_id != batch.batch_id || cache.levels.len() != MAXIMUM_LEVEL as usize + 1 {
        cache.rebuild(batch)?;
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
            ((value.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32)
                .min(leaf_grid - 1)
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
            moments[level - 1]
                [(global_parent - cache.level_offsets[level - 1]) as usize]
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
    let mut secondary = Vec::with_capacity(
        cache.state_target_indices.len() * 4 + near_particle_bytes.len(),
    );
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
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
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
    let accepted = level > 0
        && distance > 1.01 * expansion_radius
        && expansion_radius / distance.max(1.0e-12) < f64::from(THETA);
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
    let qx = DVec3::new(
        3.0 * central[0] - trace,
        3.0 * central[1],
        3.0 * central[2],
    );
    let qy = DVec3::new(
        3.0 * central[1],
        3.0 * central[3] - trace,
        3.0 * central[4],
    );
    let qz = DVec3::new(
        3.0 * central[2],
        3.0 * central[4],
        3.0 * central[5] - trace,
    );
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
