//! Reference-line Volterra/Picard orbit propagation.
//!
//! The implementation follows equations (27), (28), (40), and (42) in
//! `docs/mathtidy.md`: longitudinal speed, elapsed time, and transverse motion
//! are solved together on a monotone `h` grid.  The separable transverse kernel
//! is evaluated with two trapezoidal prefix integrals, so each Picard sweep is
//! linear in the number of trajectory nodes.

use bevy::math::{DVec2, DVec3};
use bevy::prelude::Resource;

#[derive(Clone, Copy, Debug)]
pub struct VolterraConfig {
    pub node_count: usize,
    pub maximum_picard_iterations: usize,
    pub maximum_endpoint_iterations: usize,
    pub damping: f64,
    pub relative_tolerance: f64,
    pub minimum_longitudinal_speed: f64,
    pub maximum_transverse_distance: f64,
}

impl Default for VolterraConfig {
    fn default() -> Self {
        Self {
            node_count: 33,
            maximum_picard_iterations: 10,
            maximum_endpoint_iterations: 6,
            damping: 0.75,
            relative_tolerance: 2.0e-8,
            minimum_longitudinal_speed: 1.0e-6,
            maximum_transverse_distance: f64::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VolterraSample {
    pub elapsed_seconds: f64,
    pub h: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub acceleration: DVec3,
}

impl VolterraSample {
    fn interpolate(self, other: Self, weight: f64) -> Self {
        Self {
            elapsed_seconds: self.elapsed_seconds
                + weight * (other.elapsed_seconds - self.elapsed_seconds),
            h: self.h + weight * (other.h - self.h),
            position: self.position.lerp(other.position, weight),
            velocity: self.velocity.lerp(other.velocity, weight),
            acceleration: self.acceleration.lerp(other.acceleration, weight),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VolterraSolveDiagnostics {
    pub picard_iterations: usize,
    pub endpoint_iterations: usize,
    pub relative_residual: f64,
    pub maximum_transverse_distance: f64,
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct VolterraPropagationStatus {
    pub accepted_segments: u64,
    pub rejected_segments: u64,
    pub latest: Option<VolterraSolveDiagnostics>,
}

#[derive(Debug)]
pub enum VolterraError<E> {
    InvalidInput,
    Force(E),
    NonMonotoneLongitudinalMotion,
    TaylorTubeExceeded,
    PicardDidNotConverge,
    EndpointDidNotConverge,
}

#[derive(Clone, Debug)]
pub struct VolterraSolution {
    pub samples: Vec<VolterraSample>,
    pub diagnostics: VolterraSolveDiagnostics,
}

impl VolterraSolution {
    pub fn sample_at(&self, elapsed_seconds: f64) -> Option<VolterraSample> {
        let first = *self.samples.first()?;
        let last = *self.samples.last()?;
        if elapsed_seconds <= first.elapsed_seconds {
            return Some(first);
        }
        if elapsed_seconds >= last.elapsed_seconds {
            return Some(last);
        }
        let upper = self
            .samples
            .partition_point(|sample| sample.elapsed_seconds < elapsed_seconds);
        let lower = self.samples[upper - 1];
        let upper = self.samples[upper];
        let span = (upper.elapsed_seconds - lower.elapsed_seconds).max(f64::MIN_POSITIVE);
        Some(lower.interpolate(
            upper,
            ((elapsed_seconds - lower.elapsed_seconds) / span).clamp(0.0, 1.0),
        ))
    }

    /// Linear-cursor interpolation for a nondecreasing sequence of query
    /// times.  Planning and fixed-step consumers walk the complete waveform in
    /// order, so repeating a binary search for every output point is wasteful.
    pub(crate) fn sample_at_ordered(
        &self,
        elapsed_seconds: f64,
        upper_cursor: &mut usize,
    ) -> Option<VolterraSample> {
        let first = *self.samples.first()?;
        let last = *self.samples.last()?;
        if elapsed_seconds <= first.elapsed_seconds {
            return Some(first);
        }
        if elapsed_seconds >= last.elapsed_seconds {
            *upper_cursor = self.samples.len().saturating_sub(1);
            return Some(last);
        }
        *upper_cursor = (*upper_cursor).clamp(1, self.samples.len() - 1);
        while *upper_cursor + 1 < self.samples.len()
            && self.samples[*upper_cursor].elapsed_seconds < elapsed_seconds
        {
            *upper_cursor += 1;
        }
        let lower = self.samples[*upper_cursor - 1];
        let upper = self.samples[*upper_cursor];
        let span = (upper.elapsed_seconds - lower.elapsed_seconds).max(f64::MIN_POSITIVE);
        Some(lower.interpolate(
            upper,
            ((elapsed_seconds - lower.elapsed_seconds) / span).clamp(0.0, 1.0),
        ))
    }
}

struct FixedExtentSolution {
    samples: Vec<VolterraSample>,
    picard_iterations: usize,
    relative_residual: f64,
    maximum_transverse_distance: f64,
}

/// One force-evaluation point for a complete Picard sweep.  Keeping positions
/// and times contiguous lets callers evaluate/interpolate the whole trajectory
/// in one pass instead of paying a closure call and lookup for every node.
#[derive(Clone, Copy, Debug, Default)]
pub struct VolterraForceInput {
    pub position: DVec3,
    pub elapsed_seconds: f64,
}

struct VolterraWorkspace {
    force_inputs: Vec<VolterraForceInput>,
    p: Vec<f64>,
    next_p: Vec<f64>,
    elapsed: Vec<f64>,
    next_elapsed: Vec<f64>,
    transverse: Vec<DVec2>,
    next_transverse: Vec<DVec2>,
    accelerations: Vec<DVec3>,
    transverse_force: Vec<DVec2>,
    transverse_prefix: Vec<DVec2>,
    next_transverse_prefix: Vec<DVec2>,
}

impl VolterraWorkspace {
    fn new(node_count: usize) -> Self {
        Self {
            force_inputs: vec![VolterraForceInput::default(); node_count],
            p: vec![0.0; node_count],
            next_p: vec![0.0; node_count],
            elapsed: vec![0.0; node_count],
            next_elapsed: vec![0.0; node_count],
            transverse: vec![DVec2::ZERO; node_count],
            next_transverse: vec![DVec2::ZERO; node_count],
            accelerations: vec![DVec3::ZERO; node_count],
            transverse_force: vec![DVec2::ZERO; node_count],
            transverse_prefix: vec![DVec2::ZERO; node_count],
            next_transverse_prefix: vec![DVec2::ZERO; node_count],
        }
    }
}

/// Propagates one dynamically self-consistent segment in a fixed inertial
/// reference-line frame. `acceleration_at` is evaluated on every Picard iterate,
/// closing the position -> field -> trajectory loop instead of freezing force
/// samples on the initial geometric curve.
pub fn propagate_reference_line<E, F>(
    initial_position: DVec3,
    initial_velocity: DVec3,
    reference_direction: DVec3,
    duration_seconds: f64,
    config: VolterraConfig,
    mut acceleration_at: F,
) -> Result<VolterraSolution, VolterraError<E>>
where
    F: FnMut(DVec3, f64) -> Result<DVec3, E>,
{
    propagate_reference_line_batched(
        initial_position,
        initial_velocity,
        reference_direction,
        duration_seconds,
        config,
        |inputs, accelerations| {
            for (input, acceleration) in inputs.iter().zip(accelerations) {
                *acceleration = acceleration_at(input.position, input.elapsed_seconds)?;
            }
            Ok(())
        },
    )
}

/// Batched variant of [`propagate_reference_line`].  `acceleration_batch` is
/// called once per Picard iteration and must fill every output element.  This
/// is the preferred path for interpolated fields and GPU/vectorized evaluators.
pub fn propagate_reference_line_batched<E, F>(
    initial_position: DVec3,
    initial_velocity: DVec3,
    reference_direction: DVec3,
    duration_seconds: f64,
    config: VolterraConfig,
    mut acceleration_batch: F,
) -> Result<VolterraSolution, VolterraError<E>>
where
    F: FnMut(&[VolterraForceInput], &mut [DVec3]) -> Result<(), E>,
{
    if !initial_position.is_finite()
        || !initial_velocity.is_finite()
        || !reference_direction.is_finite()
        || !duration_seconds.is_finite()
        || duration_seconds <= 0.0
        || config.node_count < 2
        || config.maximum_picard_iterations == 0
        || config.maximum_endpoint_iterations == 0
        || !config.damping.is_finite()
        || !(0.0..=1.0).contains(&config.damping)
        || config.damping == 0.0
        || !config.relative_tolerance.is_finite()
        || config.relative_tolerance <= 0.0
        || !config.minimum_longitudinal_speed.is_finite()
        || config.minimum_longitudinal_speed <= 0.0
    {
        return Err(VolterraError::InvalidInput);
    }

    let ez = reference_direction.normalize_or_zero();
    if ez == DVec3::ZERO {
        return Err(VolterraError::InvalidInput);
    }
    let helper = if ez.x.abs() <= ez.y.abs() && ez.x.abs() <= ez.z.abs() {
        DVec3::X
    } else if ez.y.abs() <= ez.z.abs() {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let ex = ez.cross(helper).normalize_or_zero();
    let ey = ez.cross(ex).normalize_or_zero();
    if ex == DVec3::ZERO || ey == DVec3::ZERO {
        return Err(VolterraError::InvalidInput);
    }

    let longitudinal_speed = initial_velocity.dot(ez);
    if !longitudinal_speed.is_finite() || longitudinal_speed <= config.minimum_longitudinal_speed {
        return Err(VolterraError::NonMonotoneLongitudinalMotion);
    }
    let transverse_velocity = DVec2::new(initial_velocity.dot(ex), initial_velocity.dot(ey));
    let mut h_extent = longitudinal_speed * duration_seconds;
    let endpoint_tolerance = (duration_seconds * config.relative_tolerance).max(1.0e-10);
    // Endpoint correction re-solves the same node layout.  Allocate its
    // working set once and only reinitialize values between solves.
    let mut workspace = VolterraWorkspace::new(config.node_count);
    for endpoint_iteration in 1..=config.maximum_endpoint_iterations {
        let solved = solve_fixed_extent(
            initial_position,
            ez,
            ex,
            ey,
            longitudinal_speed,
            transverse_velocity,
            h_extent,
            config,
            &mut workspace,
            &mut acceleration_batch,
        )?;
        let solved_duration = solved
            .samples
            .last()
            .map_or(0.0, |sample| sample.elapsed_seconds);
        if !solved_duration.is_finite() || solved_duration <= 0.0 {
            return Err(VolterraError::NonMonotoneLongitudinalMotion);
        }
        let endpoint_error = solved_duration - duration_seconds;
        if endpoint_error.abs() <= endpoint_tolerance {
            let mut samples = solved.samples;
            if let Some(last) = samples.last_mut() {
                last.elapsed_seconds = duration_seconds;
            }
            return Ok(VolterraSolution {
                samples,
                diagnostics: VolterraSolveDiagnostics {
                    picard_iterations: solved.picard_iterations,
                    endpoint_iterations: endpoint_iteration,
                    relative_residual: solved.relative_residual,
                    maximum_transverse_distance: solved.maximum_transverse_distance,
                },
            });
        }
        h_extent *= (duration_seconds / solved_duration).clamp(0.5, 2.0);
    }
    Err(VolterraError::EndpointDidNotConverge)
}

#[allow(clippy::too_many_arguments)]
fn solve_fixed_extent<E, F>(
    origin: DVec3,
    ez: DVec3,
    ex: DVec3,
    ey: DVec3,
    longitudinal_speed: f64,
    transverse_velocity: DVec2,
    h_extent: f64,
    config: VolterraConfig,
    workspace: &mut VolterraWorkspace,
    acceleration_batch: &mut F,
) -> Result<FixedExtentSolution, VolterraError<E>>
where
    F: FnMut(&[VolterraForceInput], &mut [DVec3]) -> Result<(), E>,
{
    if !h_extent.is_finite() || h_extent <= 0.0 {
        return Err(VolterraError::NonMonotoneLongitudinalMotion);
    }
    let step = h_extent / (config.node_count - 1) as f64;
    let p0 = longitudinal_speed * longitudinal_speed;
    for index in 0..config.node_count {
        let h = step * index as f64;
        workspace.p[index] = p0;
        workspace.elapsed[index] = h / longitudinal_speed;
        workspace.transverse[index] = transverse_velocity * workspace.elapsed[index];
        workspace.transverse_prefix[index] = DVec2::ZERO;
    }
    let mut relative_residual = f64::INFINITY;
    let mut completed_iterations = 0;

    for iteration in 1..=config.maximum_picard_iterations {
        for index in 0..config.node_count {
            let transverse = workspace.transverse[index];
            workspace.force_inputs[index] = VolterraForceInput {
                position: origin
                    + ez * (step * index as f64)
                    + ex * transverse.x
                    + ey * transverse.y,
                elapsed_seconds: workspace.elapsed[index],
            };
        }
        acceleration_batch(&workspace.force_inputs, &mut workspace.accelerations)
            .map_err(VolterraError::Force)?;

        workspace.next_p[0] = p0;
        let minimum_p = config.minimum_longitudinal_speed * config.minimum_longitudinal_speed;
        for index in 0..config.node_count {
            let acceleration = workspace.accelerations[index];
            if !acceleration.is_finite() {
                return Err(VolterraError::InvalidInput);
            }
            workspace.transverse_force[index] =
                DVec2::new(acceleration.dot(ex), acceleration.dot(ey));
            if index > 0 {
                let previous_longitudinal = workspace.accelerations[index - 1].dot(ez);
                let longitudinal = acceleration.dot(ez);
                // Eq. (27): 2 times the trapezoid's 1/2 factor cancels.
                workspace.next_p[index] =
                    workspace.next_p[index - 1] + step * (previous_longitudinal + longitudinal);
                if !workspace.next_p[index].is_finite() || workspace.next_p[index] <= minimum_p {
                    return Err(VolterraError::NonMonotoneLongitudinalMotion);
                }
            }
        }

        if !workspace.next_p[0].is_finite() || workspace.next_p[0] <= minimum_p {
            return Err(VolterraError::NonMonotoneLongitudinalMotion);
        }

        // Eqs. (28), (41), and (42) in one forward sweep.  C and D are
        // prefix integrals; no MxM Volterra kernel or temporary weighted array
        // is materialized.
        workspace.next_elapsed[0] = 0.0;
        workspace.next_transverse_prefix[0] = DVec2::ZERO;
        workspace.next_transverse[0] = workspace.transverse[0].lerp(DVec2::ZERO, config.damping);
        let mut previous_inverse_speed = workspace.next_p[0].sqrt().recip();
        let mut previous_integrand = workspace.transverse_force[0] * previous_inverse_speed;
        let mut weighted_prefix = DVec2::ZERO;
        for index in 1..config.node_count {
            let inverse_speed = workspace.next_p[index].sqrt().recip();
            let elapsed = workspace.next_elapsed[index - 1]
                + 0.5 * step * (previous_inverse_speed + inverse_speed);
            let integrand = workspace.transverse_force[index] * inverse_speed;
            let transverse_prefix = workspace.next_transverse_prefix[index - 1]
                + 0.5 * step * (previous_integrand + integrand);
            weighted_prefix += 0.5
                * step
                * (previous_integrand * workspace.next_elapsed[index - 1] + integrand * elapsed);
            let candidate =
                transverse_velocity * elapsed + transverse_prefix * elapsed - weighted_prefix;
            workspace.next_elapsed[index] = elapsed;
            workspace.next_transverse_prefix[index] = transverse_prefix;
            workspace.next_transverse[index] =
                workspace.transverse[index].lerp(candidate, config.damping);
            previous_inverse_speed = inverse_speed;
            previous_integrand = integrand;
        }

        let mut p_scale = p0;
        let mut time_scale = 1.0_f64;
        let mut transverse_scale = 1.0_f64;
        let mut p_delta = 0.0_f64;
        let mut time_delta = 0.0_f64;
        let mut transverse_delta = 0.0_f64;
        let mut maximum_transverse_distance = 0.0_f64;
        for index in 0..config.node_count {
            p_scale = p_scale.max(workspace.next_p[index].abs());
            time_scale = time_scale.max(workspace.next_elapsed[index].abs());
            let transverse_length = workspace.next_transverse[index].length();
            transverse_scale = transverse_scale.max(transverse_length);
            maximum_transverse_distance = maximum_transverse_distance.max(transverse_length);
            p_delta = p_delta.max((workspace.next_p[index] - workspace.p[index]).abs());
            time_delta =
                time_delta.max((workspace.next_elapsed[index] - workspace.elapsed[index]).abs());
            transverse_delta = transverse_delta
                .max((workspace.next_transverse[index] - workspace.transverse[index]).length());
        }
        relative_residual = (p_delta / p_scale)
            .max(time_delta / time_scale)
            .max(transverse_delta / transverse_scale);
        std::mem::swap(&mut workspace.p, &mut workspace.next_p);
        std::mem::swap(&mut workspace.elapsed, &mut workspace.next_elapsed);
        std::mem::swap(&mut workspace.transverse, &mut workspace.next_transverse);
        std::mem::swap(
            &mut workspace.transverse_prefix,
            &mut workspace.next_transverse_prefix,
        );
        completed_iterations = iteration;

        if maximum_transverse_distance > config.maximum_transverse_distance {
            return Err(VolterraError::TaylorTubeExceeded);
        }
        if relative_residual <= config.relative_tolerance {
            break;
        }
    }

    if relative_residual > config.relative_tolerance {
        return Err(VolterraError::PicardDidNotConverge);
    }
    let maximum_transverse_distance = workspace
        .transverse
        .iter()
        .map(|value| value.length())
        .fold(0.0, f64::max);
    let samples = (0..config.node_count)
        .map(|index| VolterraSample {
            elapsed_seconds: workspace.elapsed[index],
            h: step * index as f64,
            position: origin
                + ez * (step * index as f64)
                + ex * workspace.transverse[index].x
                + ey * workspace.transverse[index].y,
            velocity: ez * workspace.p[index].sqrt()
                + ex * (transverse_velocity.x + workspace.transverse_prefix[index].x)
                + ey * (transverse_velocity.y + workspace.transverse_prefix[index].y),
            acceleration: workspace.accelerations[index],
        })
        .collect();
    Ok(FixedExtentSolution {
        samples,
        picard_iterations: completed_iterations,
        relative_residual,
        maximum_transverse_distance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accurate_config() -> VolterraConfig {
        VolterraConfig {
            node_count: 129,
            maximum_picard_iterations: 20,
            maximum_endpoint_iterations: 10,
            damping: 1.0,
            relative_tolerance: 1.0e-10,
            minimum_longitudinal_speed: 1.0e-8,
            maximum_transverse_distance: f64::INFINITY,
        }
    }

    #[test]
    fn constant_acceleration_matches_the_analytic_trajectory() {
        let position = DVec3::new(3.0, -2.0, 1.0);
        let velocity = DVec3::new(4.0, 0.5, -0.25);
        let acceleration = DVec3::new(0.2, -0.03, 0.04);
        let duration = 2.5;
        let solution = propagate_reference_line(
            position,
            velocity,
            velocity,
            duration,
            accurate_config(),
            |_, _| Ok::<_, ()>(acceleration),
        )
        .unwrap();
        let endpoint = solution.samples.last().unwrap();
        let expected_position =
            position + velocity * duration + 0.5 * acceleration * duration.powi(2);
        let expected_velocity = velocity + acceleration * duration;
        assert!((endpoint.position - expected_position).length() < 2.0e-5);
        assert!((endpoint.velocity - expected_velocity).length() < 2.0e-5);
        assert!((endpoint.elapsed_seconds - duration).abs() < 1.0e-12);
    }

    #[test]
    fn position_dependent_force_is_closed_by_picard_iteration() {
        let stiffness = 2.5e-3;
        let duration = 4.0;
        let solution = propagate_reference_line(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::X * 2.0,
            DVec3::X,
            duration,
            accurate_config(),
            |position, _| Ok::<_, ()>(DVec3::new(0.0, -stiffness * position.y, 0.0)),
        )
        .unwrap();
        let endpoint = solution.samples.last().unwrap();
        let omega = stiffness.sqrt();
        assert!((endpoint.position.y - (omega * duration).cos()).abs() < 2.0e-5);
        assert!(solution.diagnostics.picard_iterations > 1);
    }

    #[test]
    fn longitudinal_turning_point_is_rejected() {
        let result = propagate_reference_line(
            DVec3::ZERO,
            DVec3::X,
            DVec3::X,
            2.0,
            accurate_config(),
            |_, _| Ok::<_, ()>(-DVec3::X),
        );
        assert!(matches!(
            result,
            Err(VolterraError::NonMonotoneLongitudinalMotion)
        ));
    }

    #[test]
    fn interpolation_returns_requested_time() {
        let solution = propagate_reference_line(
            DVec3::ZERO,
            DVec3::X,
            DVec3::X,
            3.0,
            accurate_config(),
            |_, _| Ok::<_, ()>(DVec3::ZERO),
        )
        .unwrap();
        let sample = solution.sample_at(1.25).unwrap();
        assert!((sample.elapsed_seconds - 1.25).abs() < 1.0e-12);
        assert!((sample.position - DVec3::X * 1.25).length() < 1.0e-12);
    }
}
