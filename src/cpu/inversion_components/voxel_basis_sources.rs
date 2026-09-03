pub(crate) fn build_voxel_basis_sources(
    voxels: &[InvertedDensityVoxel],
    source: &AggregatedGravitySource,
    voxel_size: f32,
) -> Option<VoxelBasisSources> {
    if voxels.is_empty()
        || source.sources.is_empty()
        || !voxel_size.is_finite()
        || voxel_size <= 0.0
    {
        return None;
    }
    let grid_radius = 0.5 * VOXEL_SIDE as f64 * f64::from(voxel_size);
    let mut groups = vec![Vec::<(DVec3, f64)>::new(); voxels.len()];
    for point in &source.sources {
        if !point.position.is_finite() || !point.mass.is_finite() || point.mass <= 0.0 {
            continue;
        }
        let coordinate = |value: f64| {
            (((value + grid_radius) / f64::from(voxel_size)).floor() as isize)
                .clamp(0, VOXEL_SIDE as isize - 1) as u8
        };
        let grid = [
            coordinate(point.position.x),
            coordinate(point.position.y),
            coordinate(point.position.z),
        ];
        let index = voxels
            .iter()
            .position(|voxel| voxel.grid == grid)
            .or_else(|| {
                voxels
                    .iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        left.center
                            .as_dvec3()
                            .distance_squared(point.position)
                            .total_cmp(&right.center.as_dvec3().distance_squared(point.position))
                    })
                    .map(|(index, _)| index)
            })?;
        groups[index].push((point.position, point.mass));
    }

    let columns = groups
        .into_iter()
        .zip(voxels)
        .map(|(points, voxel)| {
            let total_weight = points.iter().map(|(_, weight)| *weight).sum::<f64>();
            if total_weight <= f64::MIN_POSITIVE {
                return vec![VoxelBasisSource {
                    position: voxel.center.as_dvec3(),
                    volume: f64::from(voxel.volume),
                }];
            }
            let scale = f64::from(voxel.volume) / total_weight;
            points
                .into_iter()
                .map(|(position, weight)| VoxelBasisSource {
                    position,
                    volume: weight * scale,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut hasher = DefaultHasher::new();
    for (column, sources) in columns.iter().enumerate() {
        hasher.write_usize(column);
        hasher.write_usize(sources.len());
        for source in sources {
            hasher.write_u64(source.position.x.to_bits());
            hasher.write_u64(source.position.y.to_bits());
            hasher.write_u64(source.position.z.to_bits());
            hasher.write_u64(source.volume.to_bits());
        }
    }
    Some(VoxelBasisSources {
        columns,
        hash: hasher.finish(),
    })
}
