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
            // is nevertheless recomputed after refinement and used by the
            // reciprocal-space quadrature.
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
