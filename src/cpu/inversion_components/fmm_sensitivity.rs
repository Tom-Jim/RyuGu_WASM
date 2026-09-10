const REFERENCE_THETA: f64 = 0.025;

/// Source-count-independent nonlinear field used for planning propagation. It
/// stores source nodes for the independent backend's ExaFMM solver.
pub(crate) type PlanningDynamicsTree = FmmNode;

impl FmmNode {
    pub(crate) fn sources(&self) -> &[(DVec3, f64)] {
        &self.0
    }
}

pub(crate) fn build_planning_dynamics_tree(
    records: &[PlanningBasisRecord],
    densities: &[f32],
) -> Option<PlanningDynamicsTree> {
    if records.is_empty() || densities.len() != EXPECTED_VOXEL_COUNT {
        return None;
    }
    let points = records
        .iter()
        .filter_map(|record| {
            let density = f64::from(*densities.get(record.voxel_index as usize)?);
            let mass = f64::from(record.position_volume[3]) * density;
            let position = DVec3::new(
                f64::from(record.position_volume[0]),
                f64::from(record.position_volume[1]),
                f64::from(record.position_volume[2]),
            );
            (position.is_finite() && mass.is_finite() && mass > 0.0).then_some((position, mass))
        })
        .collect::<Vec<_>>();
    (!points.is_empty()).then_some(FmmNode(points))
}

pub(crate) struct FmmNode(Vec<(DVec3, f64)>);

fn evaluate_fmm(node: &FmmNode, target: DVec3, theta: f64) -> DVec3 {
    let method = if theta <= REFERENCE_THETA { "direct" } else { "fmm" };
    match crate::cpp_backend::evaluate_sources(method, &node.0, &[target]) {
        Ok(values) => DVec3::new(values[0][0], values[0][1], values[0][2]) / f64::from(G),
        Err(_) => DVec3::splat(f64::NAN),
    }
}

fn high_resolution_reference_tree(source: &DensityQuadratureSource) -> Option<FmmNode> {
    let points = source
        .bytes
        .as_chunks::<32>()
        .0
        .iter()
        .filter_map(|record| {
            let direction = DVec3::new(
                f64::from(read_f32(record, 0)),
                f64::from(read_f32(record, 4)),
                f64::from(read_f32(record, 8)),
            )
            .normalize_or_zero();
            let solid_angle = f64::from(read_f32(record, 12).max(0.0));
            let inner = f64::from(read_f32(record, 16).max(0.0));
            let outer = f64::from(read_f32(record, 20).max(0.0));
            let density = f64::from(read_f32(record, 24).max(0.0));
            let volume = solid_angle * (outer.powi(3) - inner.powi(3)).max(0.0) / 3.0;
            let denominator = (outer.powi(3) - inner.powi(3)).max(f64::MIN_POSITIVE);
            let centroid_radius = 0.75 * (outer.powi(4) - inner.powi(4)) / denominator;
            let mass = volume * density;
            (direction != DVec3::ZERO && mass > 0.0 && mass.is_finite())
                .then_some((direction * centroid_radius, mass))
        })
        .collect::<Vec<_>>();
    (!points.is_empty()).then_some(FmmNode(points))
}

fn evaluate_reference_tree(tree: &FmmNode, sample: &TrajectoryInversionKnot) -> Option<Vec3> {
    let body_position = sample.body_rotation.inverse() * sample.position;
    let acceleration_body =
        evaluate_fmm(tree, body_position.as_dvec3(), REFERENCE_THETA) * G as f64;
    let acceleration = sample.body_rotation * acceleration_body.as_vec3();
    acceleration.is_finite().then_some(acceleration)
}

fn high_resolution_reference_basis_trees(
    voxels: &[InvertedDensityVoxel],
    source: &DensityQuadratureSource,
) -> Option<Vec<FmmNode>> {
    let radius = source
        .bytes
        .as_chunks::<32>()
        .0
        .iter()
        .map(|record| read_f32(record, 20))
        .filter(|value| value.is_finite())
        .fold(0.0_f32, f32::max);
    if radius <= 0.0 || voxels.is_empty() {
        return None;
    }
    let voxel_size = 2.0 * radius / VOXEL_SIDE as f32;
    let mut grid_lookup = [usize::MAX; VOXEL_SIDE * VOXEL_SIDE * VOXEL_SIDE];
    for (index, voxel) in voxels.iter().enumerate() {
        let [x, y, z] = voxel.grid.map(usize::from);
        grid_lookup[(z * VOXEL_SIDE + y) * VOXEL_SIDE + x] = index;
    }
    let mut groups = vec![Vec::<(DVec3, f64)>::new(); voxels.len()];
    for record in source.bytes.as_chunks::<32>().0 {
        let direction = DVec3::new(
            f64::from(read_f32(record, 0)),
            f64::from(read_f32(record, 4)),
            f64::from(read_f32(record, 8)),
        )
        .normalize_or_zero();
        let solid_angle = f64::from(read_f32(record, 12).max(0.0));
        let inner = f64::from(read_f32(record, 16).max(0.0));
        let outer = f64::from(read_f32(record, 20).max(0.0));
        let volume = solid_angle * (outer.powi(3) - inner.powi(3)).max(0.0) / 3.0;
        let denominator = (outer.powi(3) - inner.powi(3)).max(f64::MIN_POSITIVE);
        let position = direction * (0.75 * (outer.powi(4) - inner.powi(4)) / denominator);
        if direction == DVec3::ZERO || volume <= 0.0 || !volume.is_finite() {
            continue;
        }
        let coordinate = |value: f64| {
            (((value + f64::from(radius)) / f64::from(voxel_size)).floor() as isize)
                .clamp(0, VOXEL_SIDE as isize - 1) as usize
        };
        let grid = [
            coordinate(position.x),
            coordinate(position.y),
            coordinate(position.z),
        ];
        let voxel = grid_lookup[(grid[2] * VOXEL_SIDE + grid[1]) * VOXEL_SIDE + grid[0]];
        if voxel != usize::MAX {
            groups[voxel].push((position, volume));
        }
    }
    Some(
        groups
            .into_iter()
            .zip(voxels)
            .map(|(mut points, voxel)| {
                if points.is_empty() {
                    points.push((voxel.center.as_dvec3(), voxel.volume as f64));
                }
                FmmNode(points)
            })
            .collect(),
    )
}

fn evaluate_reference_basis(trees: &[FmmNode], samples: &[TrajectoryInversionKnot]) -> Vec<Vec3> {
    let mut result = Vec::with_capacity(trees.len() * samples.len());
    for sample in samples {
        let body_position = sample.body_rotation.inverse() * sample.position;
        for tree in trees {
            let acceleration_body =
                evaluate_fmm(tree, body_position.as_dvec3(), REFERENCE_THETA) * G as f64;
            result.push(sample.body_rotation * acceleration_body.as_vec3());
        }
    }
    result
}
