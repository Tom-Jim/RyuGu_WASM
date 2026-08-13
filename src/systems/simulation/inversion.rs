//! Three-dimensional density inversion from sixteen trajectory states.
//!
//! The unknown is a positive density in every occupied Cartesian voxel, not a
//! global multiplier.  Radial source cells provide a mass/volume-preserving
//! tessellation of the actual Ryugu shape. Their Newtonian sensitivity along
//! the complete Quintic Hermite trajectory is precomputed once; annealing then
//! changes individual voxel densities subject to data, total-mass, smoothness,
//! and weak prior terms. Kinematic acceleration comes from the interpolated
//! trajectory itself; captured acceleration from a previously selected method
//! and the reference density used for validation are never inverse inputs.

use crate::components::*;
use crate::systems::ui::TrajectoryInversionButton;
use bevy::math::DVec3;
use bevy::prelude::*;

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
            Some(InvertedDensityVoxel {
                center: (bin.volume_moment / bin.volume).as_vec3(),
                volume: bin.volume as f32,
                density: uniform_density,
                baseline_density: uniform_density,
                reference_density: if method == ActiveGravityMethod::HomogeneousWerner {
                    uniform_density
                } else {
                    (bin.mass / bin.volume) as f32
                },
                grid: [x as u8, y as u8, z as u8],
            })
        })
        .collect::<Vec<_>>();

    // Critical separation: `density` and `baseline_density` are uniform and
    // are the only values visible to annealing. `reference_density` is the
    // volume-averaged original law and is read only after optimization.
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
) -> Option<(Vec<Vec3>, Vec<Vec3>, Vec<Vec3>)> {
    if knots.len() < 2 {
        return None;
    }
    let sample_count = (knots.len() - 1) * TRAJECTORY_SAMPLES_PER_SEGMENT + 1;
    let mut observations = Vec::with_capacity(sample_count);
    let mut sensitivities = Vec::with_capacity(sample_count * voxels.len());
    let mut predictions = Vec::with_capacity(sample_count);
    // The FFT hierarchy clips its discrete Green kernel at half of the finest
    // 128 m cell. Other methods evaluate the exterior Newton kernel directly.
    // This is the only finite-resolution correction needed at these targets.
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

    // Accelerations at the controls are reconstructed only from their entered
    // velocities and time stamps. The original forward density and the stale
    // acceleration captured before a method switch never enter the inverse.
    let knot_accelerations = quintic_knot_accelerations(knots)?;

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
    let prior = job
        .current_densities
        .iter()
        .zip(&job.voxels)
        .map(|(density, voxel)| {
            ((*density / voxel.baseline_density.max(f32::MIN_POSITIVE) - 1.0) as f64).powi(2)
        })
        .sum::<f64>()
        / job.voxels.len().max(1) as f64;
    data_error + MASS_WEIGHT * mass_error + SMOOTHNESS_WEIGHT * smoothness + PRIOR_WEIGHT * prior
}

fn next_random(state: &mut u64) -> f64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    (value >> 11) as f64 / ((1_u64 << 53) - 1) as f64
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
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
        || !inversion.ready
        || inversion.knots.len() != TRAJECTORY_INVERSION_SAMPLE_COUNT
    {
        return;
    }
    let Some(source) = radial_source else {
        inversion.error = Some("The asteroid volume source is not ready.".into());
        return;
    };
    let method = *active_method;
    let Some((voxels, voxel_size)) = build_density_voxels(&source, method) else {
        inversion.error = Some("The asteroid volume could not be voxelized.".into());
        return;
    };
    let Some((observed_accelerations, sensitivities, predicted_accelerations)) =
        build_observations_and_sensitivities(&inversion.knots, &voxels, method)
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
        rng_state: 0x9e37_79b9_7f4a_7c15_u64 ^ method.performance_index() as u64,
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
    inversion.displayed_density = None;
    inversion.selected = None;
    inversion.edit_buffer.clear();
    inversion.error = None;
    inversion.annealing = Some(job);
}

pub fn simulated_annealing_system(mut inversion: ResMut<TrajectoryInversionState>) {
    let Some(mut job) = inversion.annealing.take() else {
        return;
    };
    let voxel_count = job.voxels.len();
    let total_volume = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mean_density = (job.current_mass as f32 / total_volume).max(f32::MIN_POSITIVE);
    let maximum_radius = job
        .voxels
        .iter()
        .map(|voxel| voxel.center.length())
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    let mean_normalized_radius = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume * voxel.center.length() / maximum_radius)
        .sum::<f32>()
        / total_volume;
    let mean_squared_radius = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume * (voxel.center.length() / maximum_radius).powi(2))
        .sum::<f32>()
        / total_volume;
    let (linear_variance, quadratic_covariance) =
        job.voxels
            .iter()
            .fold((0.0_f32, 0.0_f32), |(variance, covariance), voxel| {
                let q = voxel.center.length() / maximum_radius;
                let linear = q - mean_normalized_radius;
                let quadratic = q * q - mean_squared_radius;
                (
                    variance + voxel.volume * linear * linear,
                    covariance + voxel.volume * linear * quadratic,
                )
            });
    let quadratic_projection = quadratic_covariance / linear_variance.max(f32::MIN_POSITIVE);
    let mut proposal_deltas = vec![0.0_f32; voxel_count];
    for _ in 0..ITERATIONS_PER_FRAME {
        if job.iteration >= job.iterations || voxel_count == 0 {
            break;
        }
        let progress = job.iteration as f64 / job.iterations as f64;
        let temperature = 0.035 * (1.0 - progress).powi(2) + 2.0e-6;
        let log_width = 0.65 * (1.0 - progress) + 0.008;
        proposal_deltas.fill(0.0);

        let proposal_kind = next_random(&mut job.rng_state);
        if proposal_kind < 0.62 {
            // Low-frequency mass-conserving modes efficiently discover the
            // smooth radial component without restricting the final model to
            // radial symmetry.
            let use_curvature = next_random(&mut job.rng_state) < 0.38;
            let signed_width = (next_random(&mut job.rng_state) * 2.0 - 1.0) * log_width * 0.28;
            for (delta, voxel) in proposal_deltas.iter_mut().zip(&job.voxels) {
                let q = voxel.center.length() / maximum_radius;
                let linear = q - mean_normalized_radius;
                let radial_mode = if use_curvature {
                    q * q - mean_squared_radius - quadratic_projection * linear
                } else {
                    linear
                };
                *delta = mean_density * signed_width as f32 * radial_mode;
            }
        } else {
            // The remaining proposals retain the full three-dimensional voxel
            // freedom. Moving equal mass along a real adjacency edge preserves
            // total mass exactly and lets the dense interpolated trajectory
            // resolve supported lateral structure without arbitrary distant
            // swaps.
            let edge = ((next_random(&mut job.rng_state) * job.neighbours.len() as f64) as usize)
                .min(job.neighbours.len().saturating_sub(1));
            let (first, second) = if let Some(pair) = job.neighbours.get(edge) {
                *pair
            } else {
                (0, voxel_count.saturating_sub(1))
            };
            if first != second {
                let signed_width = (next_random(&mut job.rng_state) * 2.0 - 1.0) * log_width * 0.12;
                proposal_deltas[first] = mean_density * signed_width as f32;
                proposal_deltas[second] =
                    -proposal_deltas[first] * job.voxels[first].volume / job.voxels[second].volume;
            }
        }

        if proposal_deltas.iter().enumerate().any(|(index, delta)| {
            let proposed = job.current_densities[index] + *delta;
            let baseline = job.voxels[index].baseline_density.max(f32::MIN_POSITIVE);
            proposed < 0.02 * baseline || proposed > 8.0 * baseline
        }) {
            job.iteration += 1;
            continue;
        }

        for (density, delta) in job.current_densities.iter_mut().zip(&proposal_deltas) {
            *density += *delta;
        }
        for observation in 0..job.predicted_accelerations.len() {
            for (voxel, delta) in proposal_deltas.iter().enumerate() {
                job.predicted_accelerations[observation] +=
                    job.sensitivities[observation * voxel_count + voxel] * *delta;
            }
        }
        let proposal_objective = objective(&job);
        let delta = proposal_objective - job.current_objective;
        let accept = delta <= 0.0
            || next_random(&mut job.rng_state) < (-delta / temperature).exp().clamp(0.0, 1.0);
        if accept {
            job.current_objective = proposal_objective;
            if proposal_objective < job.best_objective {
                job.best_objective = proposal_objective;
                job.best_densities.clone_from(&job.current_densities);
            }
        } else {
            for (density, delta) in job.current_densities.iter_mut().zip(&proposal_deltas) {
                *density -= *delta;
            }
            for observation in 0..job.predicted_accelerations.len() {
                for (voxel, delta) in proposal_deltas.iter().enumerate() {
                    job.predicted_accelerations[observation] -=
                        job.sensitivities[observation * voxel_count + voxel] * *delta;
                }
            }
        }
        job.iteration += 1;
    }

    if job.iteration < job.iterations {
        inversion.annealing = Some(job);
        return;
    }
    for (voxel, density) in job.voxels.iter_mut().zip(&job.best_densities) {
        voxel.density = *density;
    }
    let total_volume = job
        .voxels
        .iter()
        .map(|voxel| voxel.volume as f64)
        .sum::<f64>();
    let inferred_mass = job
        .voxels
        .iter()
        .map(|voxel| voxel.density as f64 * voxel.volume as f64)
        .sum::<f64>();
    let baseline_mass = job
        .voxels
        .iter()
        .map(|voxel| voxel.reference_density as f64 * voxel.volume as f64)
        .sum::<f64>();
    let model_deviation = density_model_deviation(&job.voxels);
    let objective_improvement = ((job.initial_objective - job.best_objective)
        / job.initial_objective.max(f64::MIN_POSITIVE))
    .clamp(0.0, 1.0) as f32;
    let result = DensityInversionResult {
        method: job.method,
        density: (inferred_mass / total_volume.max(f64::MIN_POSITIVE)) as f32,
        density_scale: (inferred_mass / baseline_mass.max(f64::MIN_POSITIVE)) as f32,
        objective: job.best_objective,
        model_deviation,
        model_fit: (1.0 - model_deviation).clamp(0.0, 1.0),
        objective_improvement,
        trajectory_samples: job.observed_accelerations.len(),
        iterations: job.iterations,
        voxel_size: job.voxel_size,
        voxels: job.voxels,
    };
    let index = job.method.performance_index();
    inversion.results[index] = Some(result.clone());
    inversion.displayed_density = Some(result);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_record(direction: Vec3, density: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for value in [
            direction.x,
            direction.y,
            direction.z,
            1.0, // solid angle
            0.0, // inner radius
            100.0,
            density,
            0.0, // record padding
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn voxel_prior_does_not_copy_forward_density() {
        let mut bytes = source_record(Vec3::X, 100.0);
        bytes.extend(source_record(Vec3::NEG_X, 10_000.0));
        let source = RadialGravitySource { bytes, count: 2 };

        let (voxels, _) = build_density_voxels(&source, ActiveGravityMethod::HomogeneousWerner)
            .expect("valid voxel source");

        assert_eq!(voxels.len(), 2);
        assert_eq!(voxels[0].density, voxels[1].density);
        assert_eq!(voxels[0].baseline_density, voxels[1].baseline_density);
    }

    #[test]
    fn logarithmic_reference_is_separate_from_uniform_annealing_prior() {
        let mut bytes = source_record(Vec3::X, 100.0);
        let mut outer = source_record(Vec3::NEG_X, 10_000.0);
        outer[20..24].copy_from_slice(&200.0_f32.to_le_bytes());
        bytes.extend(outer);
        let source = RadialGravitySource { bytes, count: 2 };

        let (voxels, _) =
            build_density_voxels(&source, ActiveGravityMethod::RadialAnalytic).unwrap();

        let mut ordered = voxels.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.center.length().total_cmp(&b.center.length()));
        assert_eq!(ordered[0].density, ordered[1].density);
        assert_eq!(ordered[0].density, ordered[0].baseline_density);
        assert!(ordered[0].reference_density < ordered[1].reference_density);
        assert!(density_model_deviation(&voxels) > 0.0);
    }

    #[test]
    fn model_deviation_is_volume_weighted_relative_rmse() {
        let voxels = [
            InvertedDensityVoxel {
                center: Vec3::ZERO,
                volume: 1.0,
                density: 2.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [0, 0, 0],
            },
            InvertedDensityVoxel {
                center: Vec3::X,
                volume: 3.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [1, 0, 0],
            },
        ];
        assert!((density_model_deviation(&voxels) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn spatial_sensitivity_distinguishes_voxel_locations() {
        let start = TrajectoryInversionKnot {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            simulation_time_seconds: 0.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let end = TrajectoryInversionKnot {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            simulation_time_seconds: 1.0,
            baseline_acceleration: Vec3::ZERO,
            body_rotation: Quat::IDENTITY,
        };
        let voxels = [
            InvertedDensityVoxel {
                center: Vec3::new(1.0, 0.0, 0.0),
                volume: 1.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [0, 0, 0],
            },
            InvertedDensityVoxel {
                center: Vec3::new(-1.0, 0.0, 0.0),
                volume: 1.0,
                density: 1.0,
                baseline_density: 1.0,
                reference_density: 1.0,
                grid: [1, 0, 0],
            },
        ];
        let (_, sensitivities, _) = build_observations_and_sensitivities(
            &[start, end],
            &voxels,
            ActiveGravityMethod::RadialAnalytic,
        )
        .unwrap();
        assert_ne!(sensitivities[0], sensitivities[1]);
        assert!(sensitivities[0].length() > sensitivities[1].length());
    }

    #[test]
    fn quintic_track_is_densely_sampled_and_does_not_reuse_captured_acceleration() {
        let acceleration = Vec3::new(2.0, -1.0, 0.5);
        let initial_velocity = Vec3::new(1.0, 2.0, 3.0);
        let knots = [
            TrajectoryInversionKnot {
                position: Vec3::ZERO,
                velocity: initial_velocity,
                simulation_time_seconds: 0.0,
                baseline_acceleration: Vec3::splat(99.0),
                body_rotation: Quat::IDENTITY,
            },
            TrajectoryInversionKnot {
                position: initial_velocity + 0.5 * acceleration,
                velocity: initial_velocity + acceleration,
                simulation_time_seconds: 1.0,
                baseline_acceleration: Vec3::splat(-99.0),
                body_rotation: Quat::IDENTITY,
            },
        ];
        let voxels = [InvertedDensityVoxel {
            center: Vec3::ZERO,
            volume: 1.0,
            density: 1.0,
            baseline_density: 1.0,
            reference_density: 2.0,
            grid: [0, 0, 0],
        }];

        let (observations, sensitivities, predictions) = build_observations_and_sensitivities(
            &knots,
            &voxels,
            ActiveGravityMethod::RadialAnalytic,
        )
        .unwrap();

        assert_eq!(observations.len(), TRAJECTORY_SAMPLES_PER_SEGMENT + 1);
        assert_eq!(sensitivities.len(), observations.len());
        assert_eq!(predictions.len(), observations.len());
        assert!(observations.iter().all(|sample| {
            (*sample - acceleration).length() <= 2.0e-5
                && *sample != knots[0].baseline_acceleration
                && *sample != knots[1].baseline_acceleration
        }));
    }
}
