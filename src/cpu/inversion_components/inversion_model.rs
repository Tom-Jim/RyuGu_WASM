// Three-dimensional density inversion from sixteen trajectory states.
//
// The unknown is a positive density in every occupied Cartesian voxel, not a
// global multiplier.  Radial source cells provide a mass/volume-preserving
// tessellation of the actual Ryugu shape. Their Newtonian sensitivity along
// the complete Quintic Hermite trajectory is precomputed once; annealing then
// changes individual voxel densities subject to data, total-mass, smoothness,
// and weak prior terms. Kinematic acceleration comes from the interpolated
// trajectory itself; captured acceleration from a previously selected method
// and the reference density used for validation are never inverse inputs.

use crate::interface::components::*;
use crate::bevy_app::ui::TrajectoryInversionButton;
use bevy::math::DVec3;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};

// Fifteen Hermite intervals provide 241 acceleration samples (723 scalar
// components). A 4³ grid leaves the 56 shape-intersecting cells; mass
// conservation and spatial regularization control the remaining gravity null
// space without removing three-dimensional voxel freedom.
const VOXEL_SIDE: usize = 4;
/// Quintic Hermite is evaluated throughout every knot interval. The sixteen
/// controls therefore produce 15 * 16 + 1 = 241 trajectory observations.
const TRAJECTORY_SAMPLES_PER_SEGMENT: usize = 16;
const ANNEALING_ITERATIONS: u32 = 80_000;
const ITERATIONS_PER_FRAME: u32 = 1_024;
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
    // are the only values visible to annealing. `reference_density` is the
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

fn build_observations_and_sensitivities(
    knots: &[TrajectoryInversionKnot],
    voxels: &[InvertedDensityVoxel],
    method: ActiveGravityMethod,
    use_captured_acceleration: bool,
) -> Option<(Vec<Vec3>, Vec<Vec3>, Vec<Vec3>)> {
    if knots.len() < 2 {
        return None;
    }
    let sample_count = (knots.len() - 1) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1;
    let mut observations = Vec::with_capacity(sample_count);
    let mut sensitivities = Vec::with_capacity(sample_count * voxels.len());
    let mut predictions = Vec::with_capacity(sample_count);
    // MMFFT clips its discrete Green kernel at half of the finest 128 m cell.
    // The other exterior evaluators use the unsoftened Newton kernel. Avoid
    // inventing method-dependent smoothing scales merely to separate scores.
    let softening = match method {
        ActiveGravityMethod::MmfftCompressed => 64.0_f32,
        _ => 0.0,
    };
    let softening_squared = softening.powi(2);

    let sensitivity_at = |knot: TrajectoryInversionKnot, voxel: &InvertedDensityVoxel| {
        let displacement = knot.body_rotation * voxel.center - knot.position;
        displacement
            * (G * voxel.volume * (displacement.length_squared() + softening_squared).powf(-1.5))
    };

    // Captured knots already carry the acceleration from the same gravity
    // snapshot as their position and velocity.  Re-differentiating the
    // asynchronously sampled velocities amplifies readback jitter and was
    // the main source of the recent inversion accuracy collapse.  Edited
    // knots have no trustworthy force sample, so retain the velocity-derived
    // fallback for that path.
    let knot_accelerations = if use_captured_acceleration
        && knots
            .iter()
            .all(|knot| knot.baseline_acceleration.is_finite())
    {
        knots
            .iter()
            .map(|knot| knot.baseline_acceleration)
            .collect::<Vec<_>>()
    } else {
        quintic_knot_accelerations(knots)?
    };

    for segment in 0..knots.len() - 1 {
        let start = knots[segment];
        let end = knots[segment + 1];
        let duration = (end.simulation_time_seconds - start.simulation_time_seconds) as f32;
        if !duration.is_finite() || duration <= f32::EPSILON {
            return None;
        }
        let first_sample = usize::from(segment > 0);
        for sample in first_sample..=TRAJECTORY_SAMPLES_PER_SEGMENT {
            let u = sample as f32 / TRAJECTORY_SAMPLES_PER_SEGMENT as f32;
            let (position, observation) = quintic_segment_position_acceleration(
                start,
                end,
                knot_accelerations[segment],
                knot_accelerations[segment + 1],
                u,
            )?;
            let sample_knot = TrajectoryInversionKnot {
                position,
                velocity: Vec3::ZERO,
                simulation_time_seconds: start.simulation_time_seconds + duration as f64 * u as f64,
                baseline_acceleration: Vec3::ZERO,
                body_rotation: start.body_rotation.slerp(end.body_rotation, u),
            };
            let mut prediction = Vec3::ZERO;
            for voxel in voxels {
                let sensitivity = sensitivity_at(sample_knot, voxel);
                prediction += sensitivity * voxel.density;
                sensitivities.push(sensitivity);
            }
            observations.push(observation);
            predictions.push(prediction);
        }
    }
    (observations.len() == sample_count).then_some((observations, sensitivities, predictions))
}

fn trajectory_data_error(job: &SimulatedAnnealingJob) -> f64 {
    let mut data_error = 0.0_f64;
    for (prediction, observed) in job
        .predicted_accelerations
        .iter()
        .zip(&job.observed_accelerations)
    {
        let scale = observed.length_squared().max(1.0e-12);
        data_error += ((prediction - *observed).length_squared() / scale) as f64;
    }
    data_error /= job.observed_accelerations.len().max(1) as f64;
    data_error
}

fn objective(job: &SimulatedAnnealingJob) -> f64 {
    // Exterior gravity differences caused by an internal density rearrangement
    // are orders of magnitude smaller than the total field. Normalize the
    // uniform-start mismatch to one so regularization cannot erase the signal.
    let data_error = trajectory_data_error(job) / job.data_error_scale.max(1.0e-24);

    let mass_error = (job.current_mass / RYUGU_MASS as f64 - 1.0).powi(2);
    let mean_density = (job.current_mass
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
                ((job.current_densities[a] - job.current_densities[b]) as f64 / mean_density)
                    .powi(2)
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
                ((job.current_densities[a] - job.current_densities[b]) as f64 / mean_density)
                    .powi(2)
            })
            .sum::<f64>()
            / radial_pairs.len() as f64
    };
    let prior = job
        .current_densities
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

fn next_random(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
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
    job: &SimulatedAnnealingJob,
    densities: &[f32],
    objective: f64,
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
        rng_seed: job.rng_seed,
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
    mut show_section: ResMut<ShowSection>,
    mut inversion: ResMut<TrajectoryInversionState>,
) {
    let pressed = interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed);
    if pressed {
        if !inversion.ready || inversion.knots.len() != TRAJECTORY_INVERSION_SAMPLE_COUNT {
            return;
        }
        if inversion.capture_id.is_none() {
            inversion.error = Some("The frozen trajectory capture has no identity.".into());
            return;
        }
        // One click starts exactly the method selected by G. The frozen
        // trajectory is reused only for that method; there is no hidden
        // five-method queue behind this button.
        inversion.annealing = None;
        inversion.pending_inversion_methods.clear();
        inversion.batch_capture_id = None;
        // Fill one method slot at a time.  Results from other completed
        // methods belong to their own capture and remain visible in the UI.
        inversion.results[active_method.performance_index()] = None;
        inversion.displayed_density = None;
        inversion.error = None;
    }
    if !pressed {
        return;
    }
    if inversion.annealing.is_some() {
        return;
    }
    let Some(source) = radial_source else {
        inversion.error = Some("The asteroid volume source is not ready.".into());
        return;
    };
    let Some(capture_id) = inversion.capture_id else {
        inversion.error = Some("The frozen trajectory capture has no identity.".into());
        inversion.pending_inversion_methods.clear();
        return;
    };
    let method = *active_method;
    let source_hash = inversion.capture_source_hash;
    let rng_seed = inversion_rng_seed(capture_id, source_hash);
    let Some((voxels, voxel_size)) = build_density_voxels(&source, method) else {
        inversion.error = Some("The asteroid volume could not be voxelized.".into());
        return;
    };
    let Some((observed_accelerations, sensitivities, predicted_accelerations)) =
        build_observations_and_sensitivities(
            &inversion.knots,
            &voxels,
            method,
            !inversion.knots_edited,
        )
    else {
        inversion.error =
            Some("The 16 states do not define valid acceleration observations.".into());
        return;
    };
    let current_densities = voxels.iter().map(|voxel| voxel.density).collect::<Vec<_>>();
    let current_mass = voxels
        .iter()
        .map(|voxel| voxel.density as f64 * voxel.volume as f64)
        .sum();
    let mut job = SimulatedAnnealingJob {
        method,
        capture_id,
        source_hash,
        capture_epoch: inversion.capture_epoch,
        rng_seed,
        neighbours: build_neighbours(&voxels),
        voxels,
        sensitivities,
        observed_accelerations,
        current_densities: current_densities.clone(),
        best_densities: current_densities,
        predicted_accelerations,
        current_mass,
        current_objective: f64::INFINITY,
        best_objective: f64::INFINITY,
        initial_objective: f64::INFINITY,
        data_error_scale: 1.0,
        iteration: 0,
        iterations: ANNEALING_ITERATIONS,
        rng: ChaCha8Rng::seed_from_u64(rng_seed),
        voxel_size,
    };
    job.data_error_scale = trajectory_data_error(&job).max(1.0e-24);
    job.current_objective = objective(&job);
    job.best_objective = job.current_objective;
    job.initial_objective = job.current_objective;
    if !job.current_objective.is_finite() {
        inversion.error = Some("The voxel sensitivity matrix is not finite.".into());
        return;
    }
    inversion.inverted = true;
    // A visible D-key prior section otherwise wins the render branch and makes
    // a completed inversion appear identical to the forward density.  Enter
    // inverse view explicitly; D can still be pressed again for comparison.
    show_section.0 = false;
    inversion.displayed_density = Some(density_result_from_job(
        &job,
        &job.best_densities,
        job.best_objective,
    ));
    inversion.selected = None;
    inversion.edit_buffer.clear();
    inversion.error = None;
    inversion.annealing = Some(job);
}

fn inversion_rng_seed(capture_id: u64, source_hash: u64) -> u64 {
    // Identical frozen data and source geometry must traverse exactly the same
    // proposal sequence. Method-specific forward kernels still make the
    // objectives and accepted density updates different.
    0x9e37_79b9_7f4a_7c15_u64 ^ capture_id.rotate_left(17) ^ source_hash.rotate_right(11)
}
