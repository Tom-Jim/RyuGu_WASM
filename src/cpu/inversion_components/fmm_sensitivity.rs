const FMM_THETA: f64 = 0.10;
const REFERENCE_THETA: f64 = 0.025;
const FMM_LEAF_CAPACITY: usize = 8;
const FMM_METHOD_MAX_DEPTH: u8 = 5;
const REFERENCE_MAX_DEPTH: u8 = 8;

#[derive(Clone, Copy, Default)]
struct FmmMoment {
    mass: f64,
    first: DVec3,
    second: [f64; 6],
}

impl FmmMoment {
    fn add(&mut self, position: DVec3, mass: f64) {
        self.mass += mass;
        self.first += position * mass;
        let [x, y, z] = position.to_array();
        self.second[0] += mass * x * x;
        self.second[1] += mass * x * y;
        self.second[2] += mass * x * z;
        self.second[3] += mass * y * y;
        self.second[4] += mass * y * z;
        self.second[5] += mass * z * z;
    }

    fn center(self) -> DVec3 {
        self.first / self.mass.max(f64::MIN_POSITIVE)
    }
}

struct FmmNode {
    half_width: f64,
    moment: FmmMoment,
    center_of_mass: DVec3,
    children: Vec<FmmNode>,
    points: Vec<(DVec3, f64)>,
}

/// Source-count-independent nonlinear field used to close planning Picard
/// trajectories.  It reuses the same quadrupole FMM implementation as the FMM
/// sensitivity path instead of maintaining another hand-written tree.
pub(crate) struct PlanningDynamicsTree(FmmNode);

impl PlanningDynamicsTree {
    pub(crate) fn acceleration(&self, body_position: DVec3) -> Option<DVec3> {
        // Candidate generation needs a closed nonlinear field, but it is not
        // the independent certification oracle below.  Use the production FMM
        // opening criterion here; the stricter reference criterion would
        // collapse most near-body nodes to direct sums on every Picard sweep.
        let acceleration = evaluate_fmm(&self.0, body_position, FMM_THETA) * f64::from(G);
        acceleration.is_finite().then_some(acceleration)
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
            (position.is_finite() && mass.is_finite() && mass > 0.0)
                .then_some((position, mass))
        })
        .collect::<Vec<_>>();
    (!points.is_empty()).then(|| {
        PlanningDynamicsTree(build_fmm_node(points, 0, REFERENCE_MAX_DEPTH))
    })
}

fn build_fmm_node(points: Vec<(DVec3, f64)>, depth: u8, maximum_depth: u8) -> FmmNode {
    let mut moment = FmmMoment::default();
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for &(position, mass) in &points {
        moment.add(position, mass);
        min = min.min(position);
        max = max.max(position);
    }
    let extent = (max - min).abs().max_element().max(1.0e-6);
    let center = (min + max) * 0.5;
    let center_of_mass = moment.center();
    if points.len() <= FMM_LEAF_CAPACITY || depth >= maximum_depth || extent <= 1.0e-6 {
        return FmmNode {
            half_width: extent * 0.5,
            moment,
            center_of_mass,
            children: Vec::new(),
            points,
        };
    }

    let mut buckets = (0..8).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
    for point in points {
        let index = usize::from(point.0.x >= center.x)
            | (usize::from(point.0.y >= center.y) << 1)
            | (usize::from(point.0.z >= center.z) << 2);
        buckets[index].push(point);
    }
    let children = buckets
        .into_iter()
        .filter(|bucket| !bucket.is_empty())
        .map(|bucket| build_fmm_node(bucket, depth + 1, maximum_depth))
        .collect();
    FmmNode {
        half_width: extent * 0.5,
        moment,
        center_of_mass,
        children,
        points: Vec::new(),
    }
}

fn multipole_acceleration(moment: FmmMoment, center_of_mass: DVec3, target: DVec3) -> DVec3 {
    let displacement = center_of_mass - target;
    let radius_squared = displacement.length_squared().max(1.0e-18);
    let inverse_radius = radius_squared.sqrt().recip();
    let inverse_radius_cubed = inverse_radius / radius_squared;
    let central = moment.second;
    let [x, y, z] = center_of_mass.to_array();
    let central = [
        central[0] - moment.mass * x * x,
        central[1] - moment.mass * x * y,
        central[2] - moment.mass * x * z,
        central[3] - moment.mass * y * y,
        central[4] - moment.mass * y * z,
        central[5] - moment.mass * z * z,
    ];
    let trace = central[0] + central[3] + central[5];
    let qd = DVec3::new(
        (3.0 * central[0] - trace) * displacement.x + 3.0 * central[1] * displacement.y
            + 3.0 * central[2] * displacement.z,
        3.0 * central[1] * displacement.x + (3.0 * central[3] - trace) * displacement.y
            + 3.0 * central[4] * displacement.z,
        3.0 * central[2] * displacement.x + 3.0 * central[4] * displacement.y
            + (3.0 * central[5] - trace) * displacement.z,
    );
    let scalar = displacement.dot(qd);
    moment.mass * displacement * inverse_radius_cubed
        - qd * (inverse_radius_cubed / radius_squared)
        + displacement * (2.5 * scalar * inverse_radius_cubed / radius_squared.powi(2))
}

fn evaluate_fmm(node: &FmmNode, target: DVec3, theta: f64) -> DVec3 {
    let distance = (node.center_of_mass - target).length().max(1.0e-9);
    let opening = (3.0_f64).sqrt() * node.half_width / distance;
    if node.children.is_empty() || opening < theta {
        if node.children.is_empty() {
            return node.points.iter().fold(DVec3::ZERO, |sum, &(position, mass)| {
                let displacement = position - target;
                let radius_squared = displacement.length_squared().max(1.0e-18);
                let inverse_radius = radius_squared.sqrt().recip();
                sum + mass * displacement * (inverse_radius / radius_squared)
            });
        }
        return multipole_acceleration(node.moment, node.center_of_mass, target);
    }
    node.children
        .iter()
        .fold(DVec3::ZERO, |sum, child| sum + evaluate_fmm(child, target, theta))
}

fn high_resolution_reference_tree(source: &RadialGravitySource) -> Option<FmmNode> {
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
    (!points.is_empty()).then(|| build_fmm_node(points, 0, REFERENCE_MAX_DEPTH))
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
    source: &RadialGravitySource,
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
        let grid = [coordinate(position.x), coordinate(position.y), coordinate(position.z)];
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
                build_fmm_node(points, 0, REFERENCE_MAX_DEPTH)
            })
            .collect(),
    )
}

fn evaluate_reference_basis(
    trees: &[FmmNode],
    samples: &[TrajectoryInversionKnot],
) -> Vec<Vec3> {
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

pub(crate) fn fmm_voxel_basis_sensitivities(
    basis: &VoxelBasisSources,
    samples: &[TrajectoryInversionKnot],
) -> Vec<Vec3> {
    let trees = basis
        .columns
        .iter()
        .map(|column| {
            build_fmm_node(
                column
                    .iter()
                    .map(|source| (source.position, source.volume))
                    .collect(),
                0,
                FMM_METHOD_MAX_DEPTH,
            )
        })
        .collect::<Vec<_>>();

    let mut result = Vec::with_capacity(samples.len() * basis.columns.len());
    for sample in samples {
        for tree in &trees {
            let body_position = sample.body_rotation.inverse() * sample.position;
            let acceleration_body =
                evaluate_fmm(tree, body_position.as_dvec3(), FMM_THETA) * G as f64;
            result.push(sample.body_rotation * acceleration_body.as_vec3());
        }
    }
    result
}
