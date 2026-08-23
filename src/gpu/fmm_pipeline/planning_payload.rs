pub(crate) fn build_planning_fmm_payload(
    batch: &PlanningCandidateBatch,
    model: u32,
    request_id: u64,
) -> Option<PlanningMethodPayload> {
    let started = bevy::platform::time::Instant::now();
    let row_start = model as usize * 56;
    let densities = batch.density_models.get(row_start..row_start + 56)?;
    let records = batch
        .basis_records
        .iter()
        .filter_map(|record| {
            let density = f64::from(*densities.get(record.voxel_index as usize)?);
            let position = DVec3::new(
                f64::from(record.position_volume[0]),
                f64::from(record.position_volume[1]),
                f64::from(record.position_volume[2]),
            );
            let mass = f64::from(record.position_volume[3]) * density;
            (position.is_finite() && mass.is_finite() && mass > 0.0).then_some((position, mass))
        })
        .collect::<Vec<_>>();
    let radius = records
        .iter()
        .map(|record| record.0.length())
        .fold(0.0_f64, f64::max);
    if records.is_empty() || radius <= 0.0 {
        return None;
    }
    let leaf_grid = 1u32 << MAXIMUM_LEVEL;
    let mut level_maps = vec![HashMap::new(); MAXIMUM_LEVEL as usize + 1];
    let mut leaf_particles: HashMap<(u32, u32, u32), Vec<(DVec3, f64)>> = HashMap::new();
    for &(position, mass) in &records {
        let normalized = (position / radius + DVec3::ONE) * 0.5;
        let coordinate = |value: f64| {
            ((value.clamp(0.0, 1.0 - f64::EPSILON) * leaf_grid as f64) as u32)
                .min(leaf_grid - 1)
        };
        let key = (
            coordinate(normalized.x),
            coordinate(normalized.y),
            coordinate(normalized.z),
        );
        level_maps[MAXIMUM_LEVEL as usize]
            .entry(key)
            .or_insert_with(MomentAccumulator::default)
            .add(position, mass);
        leaf_particles.entry(key).or_default().push((position, mass));
    }
    for level in (1..=MAXIMUM_LEVEL as usize).rev() {
        let children = level_maps[level]
            .iter()
            .map(|(key, moment)| (*key, *moment))
            .collect::<Vec<_>>();
        for (key, child) in children {
            level_maps[level - 1]
                .entry((key.0 / 2, key.1 / 2, key.2 / 2))
                .or_insert_with(MomentAccumulator::default)
                .merge(child);
        }
    }
    let mut levels = level_maps
        .into_iter()
        .map(|cells| {
            let mut sorted = cells.into_iter().collect::<Vec<_>>();
            sorted.sort_by_key(|(key, _)| *key);
            sorted
        })
        .collect::<Vec<_>>();
    let mut level_offsets = Vec::with_capacity(levels.len());
    let mut node_count = 0_u32;
    for level in &levels {
        level_offsets.push(node_count);
        node_count += level.len() as u32;
    }
    let index_maps = levels
        .iter()
        .enumerate()
        .map(|(level_index, level)| {
            level
                .iter()
                .enumerate()
                .map(|(index, (key, _))| (*key, level_offsets[level_index] + index as u32))
                .collect::<HashMap<_, _>>()
        })
        .collect::<Vec<_>>();
    let mut particle_bytes = Vec::with_capacity(records.len() * 16);
    let mut leaf_ranges = HashMap::new();
    let mut leaf_keys = leaf_particles.keys().copied().collect::<Vec<_>>();
    leaf_keys.sort_unstable();
    for key in leaf_keys {
        let particles = &leaf_particles[&key];
        let start = (particle_bytes.len() / 16) as u32;
        for &(position, mass) in particles {
            push_f32s(
                &mut particle_bytes,
                [position.x as f32, position.y as f32, position.z as f32, mass as f32],
            );
        }
        leaf_ranges.insert(key, (start, particles.len() as u32));
    }
    let mut node_bytes = Vec::with_capacity(node_count as usize * 80);
    for (level_index, level) in levels.iter_mut().enumerate() {
        let grid = 1u32 << level_index;
        let cell_width = 2.0 * radius / grid as f64;
        for (key, moment) in level {
            let center = DVec3::new(
                -radius + (key.0 as f64 + 0.5) * cell_width,
                -radius + (key.1 as f64 + 0.5) * cell_width,
                -radius + (key.2 as f64 + 0.5) * cell_width,
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
                index_maps[level_index - 1][&(key.0 / 2, key.1 / 2, key.2 / 2)]
            };
            let (particle_start, particle_count) = if level_index == MAXIMUM_LEVEL as usize {
                leaf_ranges.get(key).copied().unwrap_or((0, 0))
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
        item_count: node_count,
        secondary_count: records.len() as u32,
        maximum_level: MAXIMUM_LEVEL,
        total_mass: records.iter().map(|record| record.1).sum::<f64>() as f32,
        preparation_ms: started.elapsed().as_secs_f64() * 1.0e3,
        ..default()
    })
}

fn push_f32s<const N: usize>(bytes: &mut Vec<u8>, values: [f32; N]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
