type PlanningCellKey = (u32, u32, u32);

/// Geometry-only octree cache. Node keys, parent links, leaf ranges and
/// particle ordering are invariant across density models; only moments and
/// particle masses are rebuilt for each 56-weight row.
#[derive(Default)]
pub(crate) struct PlanningFmmWorkspace {
    batch_id: u64,
    radius: f64,
    levels: Vec<Vec<PlanningCellKey>>,
    level_offsets: Vec<u32>,
    index_maps: Vec<HashMap<PlanningCellKey, u32>>,
    particle_order: Vec<usize>,
    leaf_ranges: HashMap<PlanningCellKey, (u32, u32)>,
}

impl PlanningFmmWorkspace {
    fn rebuild(&mut self, batch: &PlanningCandidateBatch) -> Option<()> {
        self.batch_id = batch.batch_id;
        self.radius = batch
            .basis_records
            .iter()
            .map(|record| {
                DVec3::new(
                    f64::from(record.position_volume[0]),
                    f64::from(record.position_volume[1]),
                    f64::from(record.position_volume[2]),
                )
                .length()
            })
            .fold(0.0_f64, f64::max);
        if !self.radius.is_finite() || self.radius <= 0.0 || batch.basis_records.is_empty() {
            return None;
        }
        let leaf_grid = 1u32 << MAXIMUM_LEVEL;
        let leaf_key = |record: &PlanningBasisRecord| {
            let position = DVec3::new(
                f64::from(record.position_volume[0]),
                f64::from(record.position_volume[1]),
                f64::from(record.position_volume[2]),
            );
            let normalized = (position / self.radius + DVec3::ONE) * 0.5;
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
        let leaf_keys = batch.basis_records.iter().map(leaf_key).collect::<Vec<_>>();
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
        for (ordered, index) in self.particle_order.iter().copied().enumerate() {
            let key = leaf_keys[index];
            self.leaf_ranges
                .entry(key)
                .and_modify(|range| range.1 += 1)
                .or_insert((ordered as u32, 1));
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
    let mut moments = cache
        .levels
        .iter()
        .map(|level| vec![MomentAccumulator::default(); level.len()])
        .collect::<Vec<_>>();
    let leaf_level = MAXIMUM_LEVEL as usize;
    let leaf_grid = 1u32 << MAXIMUM_LEVEL;
    for (record, mass) in batch.basis_records.iter().zip(masses.iter().copied()) {
        let position = DVec3::new(
            f64::from(record.position_volume[0]),
            f64::from(record.position_volume[1]),
            f64::from(record.position_volume[2]),
        );
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
        let local_index = (global_index - cache.level_offsets[leaf_level]) as usize;
        moments[leaf_level][local_index].add(position, mass);
    }
    for level in (1..=leaf_level).rev() {
        let children = moments[level].clone();
        for (index, child) in children.into_iter().enumerate() {
            let key = cache.levels[level][index];
            let parent_key = (key.0 / 2, key.1 / 2, key.2 / 2);
            let global_parent = *cache.index_maps[level - 1].get(&parent_key)?;
            let parent = (global_parent - cache.level_offsets[level - 1]) as usize;
            moments[level - 1][parent].merge(child);
        }
    }
    let mut particle_bytes = Vec::with_capacity(batch.basis_records.len() * 16);
    for index in cache.particle_order.iter().copied() {
        let record = batch.basis_records[index];
        push_f32s(
            &mut particle_bytes,
            [
                record.position_volume[0],
                record.position_volume[1],
                record.position_volume[2],
                masses[index] as f32,
            ],
        );
    }
    let node_count = cache.levels.iter().map(Vec::len).sum::<usize>();
    let mut node_bytes = Vec::with_capacity(node_count * 80);
    for (level_index, keys) in cache.levels.iter().enumerate() {
        let grid = 1u32 << level_index;
        let cell_width = 2.0 * cache.radius / grid as f64;
        for (key, moment) in keys.iter().zip(&moments[level_index]) {
            let center = DVec3::new(
                -cache.radius + (key.0 as f64 + 0.5) * cell_width,
                -cache.radius + (key.1 as f64 + 0.5) * cell_width,
                -cache.radius + (key.2 as f64 + 0.5) * cell_width,
            );
            let com = moment.first / moment.mass.max(f64::MIN_POSITIVE);
            let [x, y, z] = com.to_array();
            let central = [
                moment.second[0] - moment.mass * x * x,
                moment.second[1] - moment.mass * x * y,
                moment.second[2] - moment.mass * x * z,
                moment.second[3] - moment.mass * y * y,
                moment.second[4] - moment.mass * y * z,
                moment.second[5] - moment.mass * z * z,
            ];
            let trace = central[0] + central[3] + central[5];
            push_f32s(&mut node_bytes, [center.x as f32, center.y as f32, center.z as f32, (0.5 * cell_width) as f32]);
            push_f32s(&mut node_bytes, [com.x as f32, com.y as f32, com.z as f32, moment.mass as f32]);
            push_f32s(&mut node_bytes, [(3.0 * central[0] - trace) as f32, (3.0 * central[1]) as f32, (3.0 * central[2]) as f32, 0.0]);
            push_f32s(&mut node_bytes, [(3.0 * central[3] - trace) as f32, (3.0 * central[4]) as f32, (3.0 * central[5] - trace) as f32, 0.0]);
            let parent = if level_index == 0 {
                INVALID_PARENT
            } else {
                cache.index_maps[level_index - 1][&(key.0 / 2, key.1 / 2, key.2 / 2)]
            };
            let (particle_start, particle_count) = if level_index == leaf_level {
                cache.leaf_ranges.get(key).copied().unwrap_or((0, 0))
            } else {
                (0, 0)
            };
            for value in [parent, level_index as u32, particle_start, particle_count] {
                node_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Some(PlanningMethodPayload {
        request_id,
        method: Some(ActiveGravityMethod::Fmm),
        density_model: model,
        primary: Arc::from(node_bytes),
        secondary: Arc::from(particle_bytes),
        item_count: node_count as u32,
        secondary_count: batch.basis_records.len() as u32,
        maximum_level: MAXIMUM_LEVEL,
        total_mass: masses.iter().sum::<f64>() as f32,
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
        ..default()
    })
}

fn push_f32s<const N: usize>(bytes: &mut Vec<u8>, values: [f32; N]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
