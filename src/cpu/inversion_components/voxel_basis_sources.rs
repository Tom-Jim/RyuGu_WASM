pub(crate) fn build_voxel_basis_sources(
    voxels: &[InvertedDensityVoxel],
    source: &AggregatedGravitySource,
) -> Option<VoxelBasisSources> {
    if voxels.is_empty() || source.sources.is_empty() {
        return None;
    }
    let mut groups = vec![Vec::<(DVec3, f64)>::new(); voxels.len()];
    for point in &source.sources {
        if !point.position.is_finite() || !point.mass.is_finite() || point.mass <= 0.0 {
            continue;
        }
        let index = voxels
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                left.center
                    .as_dvec3()
                    .distance_squared(point.position)
                    .total_cmp(&right.center.as_dvec3().distance_squared(point.position))
            })
            .map(|(index, _)| index)?;
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

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for (column, sources) in columns.iter().enumerate() {
        hash = fnv_mix(hash, column as u64);
        hash = fnv_mix(hash, sources.len() as u64);
        for source in sources {
            hash = fnv_mix(hash, source.position.x.to_bits());
            hash = fnv_mix(hash, source.position.y.to_bits());
            hash = fnv_mix(hash, source.position.z.to_bits());
            hash = fnv_mix(hash, source.volume.to_bits());
        }
    }
    Some(VoxelBasisSources { columns, hash })
}

fn fnv_mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
}
