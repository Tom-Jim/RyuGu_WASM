// Three-dimensional density inversion from sixteen trajectory states.
//
// The unknown is a positive density in every occupied Cartesian voxel, not a
// global multiplier.  Radial source cells provide a mass/volume-preserving
// tessellation of the actual Ryugu shape. Each method's sensitivity along the
// complete quintic Hermite trajectory is precomputed once; the optimizer then
// changes individual voxel densities subject to data, total-mass, smoothness,
// and weak prior terms. Kinematic acceleration comes from the interpolated
// trajectory itself; captured acceleration from a previously selected method
// and the reference density used for validation are never inverse inputs.

use crate::interface::components::*;
use crate::bevy_app::ui::TrajectoryInversionButton;
use bevy::math::DVec3;
use bevy::platform::time::Instant;
use bevy::prelude::*;

// Fifteen Hermite intervals provide 241 acceleration samples (723 scalar
// components). A 4³ grid leaves the 56 shape-intersecting cells; mass
// conservation and spatial regularization control the remaining gravity null
// space without removing three-dimensional voxel freedom.
const VOXEL_SIDE: usize = 4;
const EXPECTED_VOXEL_COUNT: usize = 56;
/// Quintic Hermite is evaluated throughout every knot interval. The sixteen
/// controls therefore produce 15 * 16 + 1 = 241 trajectory observations.
const TRAJECTORY_SAMPLES_PER_SEGMENT: usize = 16;
const QP_SOLVE_COUNT: u32 = 1;
const MASS_WEIGHT: f64 = 25.0;
const SMOOTHNESS_WEIGHT: f64 = 0.02;
const RADIAL_SYMMETRY_WEIGHT: f64 = 0.15;
const PRIOR_WEIGHT: f64 = 0.000_1;

#[derive(Clone, Copy, Default)]
struct VoxelAccumulator {
    volume: f64,
    mass: f64,
    volume_moment: DVec3,
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0; 4]))
}

fn build_density_voxels(
    source: &RadialGravitySource,
    method: ActiveGravityMethod,
) -> Option<(Vec<InvertedDensityVoxel>, f32)> {
    let radius = source
        .bytes
        .chunks_exact(32)
        .map(|record| read_f32(record, 20))
        .filter(|radius| radius.is_finite())
        .fold(0.0_f32, f32::max);
    if radius <= 0.0 {
        return None;
    }
    let voxel_size = 2.0 * radius / VOXEL_SIDE as f32;
    let mut bins = vec![VoxelAccumulator::default(); VOXEL_SIDE.pow(3)];
    for record in source.bytes.chunks_exact(32) {
        let direction = DVec3::new(
            read_f32(record, 0) as f64,
            read_f32(record, 4) as f64,
            read_f32(record, 8) as f64,
        )
        .normalize_or_zero();
        let solid_angle = read_f32(record, 12).max(0.0) as f64;
        let inner = read_f32(record, 16).max(0.0) as f64;
        let outer = read_f32(record, 20).max(inner as f32) as f64;
        let density = read_f32(record, 24).max(0.0) as f64;
        if direction == DVec3::ZERO || outer <= inner || solid_angle <= 0.0 || density <= 0.0 {
            continue;
        }
        let volume = solid_angle * (outer.powi(3) - inner.powi(3)) / 3.0;
        let centroid_radius = 0.75 * (outer.powi(4) - inner.powi(4))
            / (outer.powi(3) - inner.powi(3)).max(f64::MIN_POSITIVE);
        let center = direction * centroid_radius;
        let coordinate = |value: f64| {
            (((value + radius as f64) / voxel_size as f64).floor() as isize)
                .clamp(0, VOXEL_SIDE as isize - 1) as usize
        };
        let grid = [
            coordinate(center.x),
            coordinate(center.y),
            coordinate(center.z),
        ];
        let index = (grid[2] * VOXEL_SIDE + grid[1]) * VOXEL_SIDE + grid[0];
        bins[index].volume += volume;
        bins[index].mass += volume * density;
        bins[index].volume_moment += center * volume;
    }

    let total_volume = bins.iter().map(|bin| bin.volume).sum::<f64>();
    if total_volume <= f64::EPSILON {
        return None;
    }
    let uniform_density = (RYUGU_MASS as f64 / total_volume) as f32;
    let voxels = bins
        .into_iter()
        .enumerate()
        .filter_map(|(index, bin)| {
            if bin.volume <= f64::EPSILON || bin.mass <= 0.0 {
                return None;
            }
            let x = index % VOXEL_SIDE;
            let y = (index / VOXEL_SIDE) % VOXEL_SIDE;
            let z = index / (VOXEL_SIDE * VOXEL_SIDE);
            let reference_density = if method == ActiveGravityMethod::HomogeneousWerner {
                // Werner is a homogeneous closed-polyhedron model. Its
                // admissible density field has one scalar degree of freedom,
                // so the reference field is the same uniform mass/volume
                // density used by the forward Werner evaluator.
                uniform_density
            } else {
                (bin.mass / bin.volume) as f32
            };
            Some(InvertedDensityVoxel {
                center: (bin.volume_moment / bin.volume).as_vec3(),
                volume: bin.volume as f32,
                density: uniform_density,
                baseline_density: uniform_density,
                reference_density,
                grid: [x as u8, y as u8, z as u8],
            })
        })
        .collect::<Vec<_>>();

    // Critical separation: `density` and `baseline_density` are uniform and
    // are the only values visible to optimizer. `reference_density` is the
    // selected model's validation field and is read only after optimization.
    (!voxels.is_empty()).then_some((voxels, voxel_size))
}

fn build_neighbours(voxels: &[InvertedDensityVoxel]) -> Vec<(usize, usize)> {
    let mut neighbours = Vec::new();
    for (left, a) in voxels.iter().enumerate() {
        for (right, b) in voxels.iter().enumerate().skip(left + 1) {
            let distance = a.grid[0].abs_diff(b.grid[0])
                + a.grid[1].abs_diff(b.grid[1])
                + a.grid[2].abs_diff(b.grid[2]);
            if distance == 1 {
                neighbours.push((left, right));
            }
        }
    }
    neighbours
}

/// Differentiate the velocity controls with the derivative of the quadratic
/// through each non-uniform three-knot time stencil. This is second-order at
/// the endpoints too; a two-point endpoint rule creates spurious quintic
/// curvature and therefore false gravity observations.
pub(crate) fn quintic_knot_accelerations(knots: &[TrajectoryInversionKnot]) -> Option<Vec<Vec3>> {
    if knots.len() < 2 {
        return None;
    }
    if knots.len() == 2 {
        let dt = (knots[1].simulation_time_seconds - knots[0].simulation_time_seconds) as f32;
        if !dt.is_finite() || dt <= f32::EPSILON {
            return None;
        }
        let acceleration = (knots[1].velocity - knots[0].velocity) / dt;
        return acceleration.is_finite().then_some(vec![acceleration; 2]);
    }
    let mut result = Vec::with_capacity(knots.len());
    for index in 0..knots.len() {
        let (a, b, c, evaluation) = if index == 0 {
            (0, 1, 2, 0)
        } else if index + 1 == knots.len() {
            (index - 2, index - 1, index, 2)
        } else {
            (index - 1, index, index + 1, 1)
        };
        let h0 = knots[b].simulation_time_seconds - knots[a].simulation_time_seconds;
        let h1 = knots[c].simulation_time_seconds - knots[b].simulation_time_seconds;
        if !h0.is_finite() || !h1.is_finite() || h0 <= f64::EPSILON || h1 <= f64::EPSILON {
            return None;
        }
        let (wa, wb, wc) = match evaluation {
            0 => (
                -(2.0 * h0 + h1) / (h0 * (h0 + h1)),
                (h0 + h1) / (h0 * h1),
                -h0 / (h1 * (h0 + h1)),
            ),
            1 => (
                -h1 / (h0 * (h0 + h1)),
                (h1 - h0) / (h0 * h1),
                h0 / (h1 * (h0 + h1)),
            ),
            _ => (
                h1 / (h0 * (h0 + h1)),
                -(h0 + h1) / (h0 * h1),
                (h0 + 2.0 * h1) / (h1 * (h0 + h1)),
            ),
        };
        let acceleration = knots[a].velocity * wa as f32
            + knots[b].velocity * wb as f32
            + knots[c].velocity * wc as f32;
        if !acceleration.is_finite() {
            return None;
        }
        result.push(acceleration);
    }
    Some(result)
}

/// Position and inertial acceleration of one quintic Hermite segment.
pub(crate) fn quintic_segment_position_acceleration(
    start: TrajectoryInversionKnot,
    end: TrajectoryInversionKnot,
    start_acceleration: Vec3,
    end_acceleration: Vec3,
    u: f32,
) -> Option<(Vec3, Vec3)> {
    let duration = (end.simulation_time_seconds - start.simulation_time_seconds) as f32;
    if !duration.is_finite() || duration <= f32::EPSILON {
        return None;
    }
    let h2 = duration * duration;
    let delta = end.position - start.position;
    let c0 = start.position;
    let c1 = start.velocity * duration;
    let c2 = start_acceleration * (0.5 * h2);
    let c3 = delta * 10.0
        - start.velocity * (6.0 * duration)
        - end.velocity * (4.0 * duration)
        - start_acceleration * (1.5 * h2)
        + end_acceleration * (0.5 * h2);
    let c4 = delta * -15.0
        + start.velocity * (8.0 * duration)
        + end.velocity * (7.0 * duration)
        + start_acceleration * (1.5 * h2)
        - end_acceleration * h2;
    let c5 = delta * 6.0
        - (start.velocity + end.velocity) * (3.0 * duration)
        - (start_acceleration - end_acceleration) * (0.5 * h2);
    let u = u.clamp(0.0, 1.0);
    let position = ((((c5 * u + c4) * u + c3) * u + c2) * u + c1) * u + c0;
    let acceleration =
        (c2 * 2.0 + c3 * (6.0 * u) + c4 * (12.0 * u * u) + c5 * (20.0 * u * u * u)) / h2;
    (position.is_finite() && acceleration.is_finite()).then_some((position, acceleration))
}

/// Expands the immutable sixteen-knot capture into the exact sample array
/// shared by every forward method and inverse solve.
pub(crate) fn sample_frozen_trajectory(
    knots: &[TrajectoryInversionKnot],
) -> Option<Vec<TrajectoryInversionKnot>> {
    let accelerations = quintic_knot_accelerations(knots)?;
    let mut samples = Vec::with_capacity(
        (knots.len().saturating_sub(1)) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1,
    );
    for segment in 0..knots.len().saturating_sub(1) {
        let start = knots[segment];
        let end = knots[segment + 1];
        let duration = (end.simulation_time_seconds - start.simulation_time_seconds) as f32;
        let first_sample = usize::from(segment > 0);
        for sample in first_sample..=TRAJECTORY_SAMPLES_PER_SEGMENT {
            let u = sample as f32 / TRAJECTORY_SAMPLES_PER_SEGMENT as f32;
            let (position, acceleration) = quintic_segment_position_acceleration(
                start,
                end,
                accelerations[segment],
                accelerations[segment + 1],
                u,
            )?;
            samples.push(TrajectoryInversionKnot {
                position,
                velocity: start.velocity.lerp(end.velocity, u),
                simulation_time_seconds: start.simulation_time_seconds
                    + duration as f64 * u as f64,
                baseline_acceleration: acceleration,
                body_rotation: start.body_rotation.slerp(end.body_rotation, u),
            });
        }
    }
    Some(samples)
}

fn build_observations_and_sensitivities(
    knots: &[TrajectoryInversionKnot],
    voxels: &[InvertedDensityVoxel],
) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    if knots.len() < 2 {
        return None;
    }
    let sample_count = (knots.len() - 1) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1;
    let mut observations = Vec::with_capacity(sample_count);
    let mut sensitivities = Vec::with_capacity(sample_count * voxels.len());
    let sensitivity_at = |knot: TrajectoryInversionKnot, voxel: &InvertedDensityVoxel| {
        let displacement = knot.body_rotation * voxel.center - knot.position;
        displacement * (G * voxel.volume * displacement.length_squared().powf(-1.5))
    };
    let samples = sample_frozen_trajectory(knots)?;
    let observed = interpolate_captured_accelerations(knots, &samples)?;
    for (sample_knot, observation) in samples.into_iter().zip(observed) {
        for voxel in voxels {
            let sensitivity = sensitivity_at(sample_knot, voxel);
            sensitivities.push(sensitivity);
        }
        observations.push(observation);
    }
    (observations.len() == sample_count).then_some((observations, sensitivities))
}

fn interpolate_captured_accelerations(
    knots: &[TrajectoryInversionKnot],
    samples: &[TrajectoryInversionKnot],
) -> Option<Vec<Vec3>> {
    if knots.len() < 2 {
        return None;
    }
    samples
        .iter()
        .map(|sample| {
            let upper = knots
                .iter()
                .position(|knot| knot.simulation_time_seconds >= sample.simulation_time_seconds)
                .unwrap_or(knots.len() - 1);
            let lower = upper.saturating_sub(1);
            let a = knots[lower];
            let b = knots[upper];
            let span = (b.simulation_time_seconds - a.simulation_time_seconds)
                .max(f64::EPSILON);
            let factor = ((sample.simulation_time_seconds - a.simulation_time_seconds) / span)
                .clamp(0.0, 1.0) as f32;
            let acceleration = a.baseline_acceleration.lerp(b.baseline_acceleration, factor);
            acceleration.is_finite().then_some(acceleration)
        })
        .collect()
}

fn trajectory_data_error(job: &ConvexOptimizationJob) -> f64 {
    density_data_error(job, &job.current_densities)
}

fn density_data_error(job: &ConvexOptimizationJob, densities: &[f32]) -> f64 {
    let mut data_error = 0.0_f64;
    let n = job.voxels.len();
    for (sample, observed) in job.observed_accelerations.iter().enumerate() {
        let prediction = (0..n).fold(Vec3::ZERO, |sum, voxel| {
            sum + job.sensitivities[sample * n + voxel] * densities[voxel]
        });
        let scale = observed.length_squared().max(1.0e-12);
        data_error += ((prediction - *observed).length_squared() / scale) as f64;
    }
    data_error / job.observed_accelerations.len().max(1) as f64
}

fn objective(job: &ConvexOptimizationJob) -> f64 {
    objective_for_density_slice(job, &job.current_densities)
}

fn objective_for_density_slice(job: &ConvexOptimizationJob, densities: &[f32]) -> f64 {
    // Exterior gravity differences caused by an internal density rearrangement
    // are orders of magnitude smaller than the total field. Normalize the
    // uniform-start mismatch to one so regularization cannot erase the signal.
    let data_error = density_data_error(job, densities) / job.data_error_scale.max(1.0e-24);

    let mass = job
        .voxels
        .iter()
        .zip(densities)
        .map(|(voxel, density)| voxel.volume as f64 * *density as f64)
        .sum::<f64>();
    let mass_error = (mass / RYUGU_MASS as f64 - 1.0).powi(2);
    let mean_density = (mass
        / job
            .voxels
            .iter()
            .map(|voxel| voxel.volume as f64)
            .sum::<f64>())
    .max(f64::MIN_POSITIVE);
    let smoothness = if job.neighbours.is_empty() {
        0.0
    } else {
        job.neighbours
            .iter()
            .map(|&(a, b)| {
                ((densities[a] - densities[b]) as f64 / mean_density).powi(2)
            })
            .sum::<f64>()
            / job.neighbours.len() as f64
    };
    // The source tessellation is radial by construction. Penalize angular
    // differences between voxels at the same radius, but do not inject the
    // original logarithmic density values into the inverse. This removes
    // poorly observable lateral null modes while retaining radial freedom.
    let radial_pairs = job
        .voxels
        .iter()
        .enumerate()
        .flat_map(|(left, a)| {
            job.voxels
                .iter()
                .enumerate()
                .skip(left + 1)
                .filter_map(move |(right, b)| {
                    ((a.center.length() - b.center.length()).abs() <= job.voxel_size * 0.45)
                        .then_some((left, right))
                })
        })
        .collect::<Vec<_>>();
    let radial_symmetry = if radial_pairs.is_empty() {
        0.0
    } else {
        radial_pairs
            .iter()
            .map(|&(a, b)| {
                ((densities[a] - densities[b]) as f64 / mean_density).powi(2)
            })
            .sum::<f64>()
            / radial_pairs.len() as f64
    };
    let prior = densities
        .iter()
        .zip(&job.voxels)
        .map(|(density, voxel)| {
            ((*density / voxel.baseline_density.max(f32::MIN_POSITIVE) - 1.0) as f64).powi(2)
        })
        .sum::<f64>()
        / job.voxels.len().max(1) as f64;
    data_error
        + MASS_WEIGHT * mass_error
        + SMOOTHNESS_WEIGHT * smoothness
        + RADIAL_SYMMETRY_WEIGHT * radial_symmetry
        + PRIOR_WEIGHT * prior
}

fn density_model_deviation(voxels: &[InvertedDensityVoxel]) -> f32 {
    let (error_energy, reference_energy) = voxels.iter().fold(
        (0.0_f64, 0.0_f64),
        |(error_energy, reference_energy), voxel| {
            let volume = voxel.volume.max(0.0) as f64;
            let difference = (voxel.density - voxel.reference_density) as f64;
            let reference = voxel.reference_density as f64;
            (
                error_energy + volume * difference * difference,
                reference_energy + volume * reference * reference,
            )
        },
    );
    (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt() as f32
}

fn density_result_from_job(
    job: &ConvexOptimizationJob,
    densities: &[f32],
    objective: f64,
    inversion_time_ms: f64,
) -> DensityInversionResult {
    let mut voxels = job.voxels.clone();
    for (voxel, density) in voxels.iter_mut().zip(densities) {
        voxel.density = *density;
    }
    let total_volume = voxels.iter().map(|voxel| voxel.volume as f64).sum::<f64>();
    let inferred_mass = voxels
        .iter()
        .map(|voxel| voxel.density as f64 * voxel.volume as f64)
        .sum::<f64>();
    let reference_mass = voxels
        .iter()
        .map(|voxel| voxel.reference_density as f64 * voxel.volume as f64)
        .sum::<f64>();
    let model_deviation = density_model_deviation(&voxels);
    DensityInversionResult {
        method: job.method,
        capture_id: job.capture_id,
        source_hash: job.source_hash,
        capture_epoch: job.capture_epoch,
        problem_id: job.problem_id,
        initial_objective: job.initial_objective,
        data_error_scale: job.data_error_scale,
        density: (inferred_mass / total_volume.max(f64::MIN_POSITIVE)) as f32,
        density_scale: (inferred_mass / reference_mass.max(f64::MIN_POSITIVE)) as f32,
        objective,
        model_deviation,
        model_fit: (1.0 - model_deviation).clamp(0.0, 1.0),
        objective_improvement: ((job.initial_objective - objective)
            / job.initial_objective.max(f64::MIN_POSITIVE))
        .clamp(0.0, 1.0) as f32,
        inversion_time_ms,
        trajectory_samples: job.observed_accelerations.len(),
        iterations: job.iterations,
        voxel_size: job.voxel_size,
        voxels,
    }
}

pub fn start_density_inversion_system(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<Button>,
            With<TrajectoryInversionButton>,
        ),
    >,
    active_method: Res<ActiveGravityMethod>,
    radial_source: Option<Res<RadialGravitySource>>,
    mut eq106_sensitivity: ResMut<Eq106SensitivityMatrix>,
    mut show_section: ResMut<ShowSection>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    let pressed = interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed);
    if pressed {
        if matches!(
            *active_method,
            ActiveGravityMethod::RadialAnalytic | ActiveGravityMethod::HomogeneousWerner
        ) {
            inversion.error = Some(
                "Radial generates the frozen observations; Werner is forward-only. Neither participates in density inversion.".into(),
            );
            return;
        }
        if !inversion.ready || inversion.knots.len() != TRAJECTORY_INVERSION_SAMPLE_COUNT {
            return;
        }
        if inversion.capture_id.is_none() {
            inversion.error = Some("The frozen trajectory capture has no identity.".into());
            return;
        }
        inversion.optimizer = None;
        let capture_id = inversion.capture_id.expect("validated above");
        if inversion.batch_capture_id != Some(capture_id) {
            inversion.results = std::array::from_fn(|_| None);
            inversion.batch_capture_id = Some(capture_id);
        }
        inversion.results[active_method.performance_index()] = None;
        inversion.displayed_density = None;
        inversion.error = None;
    }
    if !pressed {
        return;
    }
    if inversion.optimizer.is_some() {
        return;
    }
    let started_at = Instant::now();
    let Some(source) = radial_source else {
        inversion.error = Some("The asteroid volume source is not ready.".into());
        return;
    };
    let Some(capture_id) = inversion.capture_id else {
        inversion.error = Some("The frozen trajectory capture has no identity.".into());
        return;
    };
    let method = *active_method;
    let source_hash = inversion.capture_source_hash;
    let problem_id = inversion_problem_id(capture_id, source_hash);
    let Some((voxels, voxel_size)) = build_density_voxels(&source, method) else {
        inversion.error = Some("The asteroid volume could not be voxelized.".into());
        return;
    };
    if voxels.len() != EXPECTED_VOXEL_COUNT {
        inversion.error = Some(format!(
            "The convex inverse requires a 56-voxel source, but voxelization produced {}.",
            voxels.len()
        ));
        return;
    }
    if method == ActiveGravityMethod::CurvedArcEq106
        && (eq106_sensitivity.capture_id != Some(capture_id)
            || eq106_sensitivity.voxel_count != voxels.len())
    {
        eq106_sensitivity.capture_id = Some(capture_id);
        eq106_sensitivity.voxel_count = voxels.len();
        eq106_sensitivity.sample_count = 0;
        eq106_sensitivity.columns.clear();
    }
    let Some((observed_accelerations, mut sensitivities)) =
        build_observations_and_sensitivities(&inversion.knots, &voxels)
    else {
        inversion.error =
            Some("The 16 states do not define valid acceleration observations.".into());
        return;
    };
    if method == ActiveGravityMethod::MmfftCompressed {
        let Some(samples) = sample_frozen_trajectory(&inversion.knots) else {
            inversion.error = Some("The frozen trajectory cannot be sampled for MMFFT.".into());
            return;
        };
        sensitivities = crate::gpu::mmfft::voxel_basis_sensitivities(&voxels, &samples);
    }
    let current_densities = voxels.iter().map(|voxel| voxel.density).collect::<Vec<_>>();
    let mut job = ConvexOptimizationJob {
        method,
        capture_id,
        source_hash,
        capture_epoch: inversion.capture_epoch,
        problem_id,
        neighbours: build_neighbours(&voxels),
        voxels,
        sensitivities,
        observed_accelerations,
        current_densities: current_densities.clone(),
        best_densities: current_densities,
        initial_objective: f64::INFINITY,
        data_error_scale: 1.0,
        iterations: QP_SOLVE_COUNT,
        voxel_size,
        started_at,
    };
    job.data_error_scale = trajectory_data_error(&job).max(1.0e-24);
    job.initial_objective = objective(&job);
    if !job.initial_objective.is_finite() {
        inversion.error = Some("The voxel sensitivity matrix is not finite.".into());
        return;
    }
    inversion.inverted = true;
    // A visible D-key prior section otherwise wins the render branch. Enter
    // inverse view explicitly; D can still be pressed again for comparison.
    show_section.0 = false;
    inversion.displayed_density = Some(density_result_from_job(
        &job,
        &job.best_densities,
        job.initial_objective,
        0.0,
    ));
    inversion.selected = None;
    inversion.edit_buffer.clear();
    inversion.error = None;
    inversion.optimizer = Some(job);
}

fn inversion_problem_id(capture_id: u64, source_hash: u64) -> u64 {
    // Identical frozen data and source geometry identify the same inverse
    // problem even though method-specific forward operators produce different
    // sensitivity matrices.
    0x9e37_79b9_7f4a_7c15_u64 ^ capture_id.rotate_left(17) ^ source_hash.rotate_right(11)
}
