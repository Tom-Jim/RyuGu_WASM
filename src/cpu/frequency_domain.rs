//! Density preparation shared by the frequency-domain runtime and planners.

use crate::interface::components::*;
use bevy::math::DVec3;
use bevy::prelude::*;
use num_complex::Complex64;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

pub(crate) const EQ184_RADIAL_SHELLS: usize = 4;
pub(crate) const EQ184_DIRECTIONS_PER_SHELL: usize = 16;
pub(crate) const EQ184_QUADRATURE_COUNT: usize = EQ184_RADIAL_SHELLS * EQ184_DIRECTIONS_PER_SHELL;
pub(crate) const EQ184_BASE_LAPLACE_SIGMA: f64 = 1.0e-3;

/// Positive real Laplace samples used for the finite equation-(184) operator.
pub(crate) fn eq184_laplace_sigma(index: usize, count: usize) -> f64 {
    let normalized = if count > 1 {
        index.min(count - 1) as f64 / (count - 1) as f64
    } else {
        0.0
    };
    EQ184_BASE_LAPLACE_SIGMA * (1.0 + 7.0 * normalized)
}

/// One midpoint shell/Fibonacci-direction node for the R^3 integral in (184).
pub(crate) fn eq184_quadrature_node(index: usize, source_radius: f64) -> Option<(DVec3, f64)> {
    if index >= EQ184_QUADRATURE_COUNT || !source_radius.is_finite() || source_radius <= 0.0 {
        return None;
    }
    let angular = index % EQ184_DIRECTIONS_PER_SHELL;
    let base = angular % (EQ184_DIRECTIONS_PER_SHELL / 2);
    let sign = if angular >= EQ184_DIRECTIONS_PER_SHELL / 2 {
        -1.0
    } else {
        1.0
    };
    let z = 1.0 - 2.0 * (base as f64 + 0.5) / 8.0;
    let radius_xy = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.399_963_229_728_653 * base as f64;
    let direction = sign * DVec3::new(radius_xy * phi.cos(), radius_xy * phi.sin(), z);
    let maximum_wave_number = std::f64::consts::PI / (0.10 * source_radius.max(1.0));
    let radial_step = maximum_wave_number / EQ184_RADIAL_SHELLS as f64;
    let wave_number = (index / EQ184_DIRECTIONS_PER_SHELL) as f64 * radial_step + 0.5 * radial_step;
    let angular_weight = std::f64::consts::TAU * 2.0 / EQ184_DIRECTIONS_PER_SHELL as f64;
    Some((
        direction * wave_number,
        wave_number.powi(2) * radial_step * angular_weight,
    ))
}

/// Composite-trapezoid contribution to T_gamma(s,k), with absolute t as in
/// equation (143). Callers must provide a nondecreasing physical time axis.
pub(crate) fn eq184_time_weight(
    previous_time: f64,
    time: f64,
    next_time: f64,
    index: usize,
    count: usize,
    sigma: f64,
) -> Option<f64> {
    if count == 0
        || index >= count
        || !previous_time.is_finite()
        || !time.is_finite()
        || !next_time.is_finite()
        || !sigma.is_finite()
        || sigma <= 0.0
        || previous_time < 0.0
        || previous_time > time
        || time > next_time
    {
        return None;
    }
    let left_dt = time - previous_time;
    let right_dt = next_time - time;
    let trapezoid_weight = match (index, count) {
        (_, 1) => 1.0,
        (0, _) => 0.5 * right_dt,
        (i, n) if i + 1 == n => 0.5 * left_dt,
        _ => 0.5 * (left_dt + right_dt),
    };
    Some(trapezoid_weight * (-sigma * time).exp())
}

/// Complex phase factor completing the time weight in equation (143).
pub(crate) fn eq184_trajectory_term(
    k: DVec3,
    position: DVec3,
    previous_time: f64,
    time: f64,
    next_time: f64,
    index: usize,
    count: usize,
    sigma: f64,
) -> Option<Complex64> {
    if !k.is_finite() || !position.is_finite() {
        return None;
    }
    let weight = eq184_time_weight(previous_time, time, next_time, index, count, sigma)?;
    Some(Complex64::from_polar(weight, k.dot(position)))
}

/// Generate the fixed-length known trajectory used by equation (184).
///
/// This is the discrete Picard/fixed-point step of equation (185): starting
/// from the probe initial state, evaluate the body-frame Newton field of the
/// frozen density source and advance a fixed-duration leapfrog arc.  The
/// returned knots are the sole trajectory input for the frequency-domain,
/// MMFFT, and FMM forward/inverse paths; no radial-history samples are used.
pub fn generate_fixed_point_trajectory(
    source: &AggregatedGravitySource,
    initial_position: Vec3,
    initial_velocity: Vec3,
    duration_seconds: f64,
    knot_count: usize,
) -> Option<Vec<TrajectoryInversionKnot>> {
    if knot_count < 2
        || !initial_position.is_finite()
        || !initial_velocity.is_finite()
        || !duration_seconds.is_finite()
        || duration_seconds <= 0.0
        || source.sources.is_empty()
    {
        return None;
    }
    let dt = duration_seconds / (knot_count - 1) as f64;
    let acceleration = |position: Vec3| -> Vec3 {
        source.sources.iter().fold(Vec3::ZERO, |sum, point| {
            let delta = position - point.position.as_vec3();
            let r2 = delta.length_squared().max(1.0);
            sum - G * point.mass as f32 * delta / (r2 * r2.sqrt())
        })
    };
    let mut position = initial_position;
    let mut velocity = initial_velocity;
    let mut knots = Vec::with_capacity(knot_count);
    for index in 0..knot_count {
        let time = index as f64 * dt;
        let a = acceleration(position);
        knots.push(TrajectoryInversionKnot {
            position,
            velocity,
            simulation_time_seconds: time,
            baseline_acceleration: a,
            body_rotation: Quat::IDENTITY,
        });
        if index + 1 < knot_count {
            velocity += a * (0.5 * dt as f32);
            position += velocity * dt as f32;
            velocity += acceleration(position) * (0.5 * dt as f32);
        }
    }
    knots
        .iter()
        .all(|knot| knot.position.is_finite() && knot.velocity.is_finite())
        .then_some(knots)
}

/// Mass-preserving point residue used by the equation-184 density spectrum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyDomainPointSource {
    pub position: DVec3,
    pub mass: f64,
}

#[derive(Resource, Default)]
pub struct AggregatedGravitySource {
    pub sources: Vec<FrequencyDomainPointSource>,
    pub total_mass: f64,
    pub radius: f64,
    pub source_hash: u64,
}

/// Converts every radial quadrature cell into one mass-preserving residue
/// record used to evaluate the discrete density Fourier transform. Cells are
/// not rebinned: doing so damps high-k content before equation (184) sees it.
pub fn build_aggregated_gravity_source_system(
    mut commands: Commands,
    radial: Option<Res<RadialGravitySource>>,
    existing: Option<Res<AggregatedGravitySource>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(radial) = radial else { return };
    let record_count = radial.bytes.len() / 32;
    if record_count == 0 {
        return;
    }

    let mut sources = Vec::with_capacity(record_count);
    let mut total_mass = 0.0;
    let mut radius = 0.0_f64;
    for chunk in radial.bytes.as_chunks::<32>().0 {
        let direction = DVec3::new(
            read_f32_le(chunk, 0) as f64,
            read_f32_le(chunk, 4) as f64,
            read_f32_le(chunk, 8) as f64,
        )
        .try_normalize()
        .unwrap_or(DVec3::Z);
        let solid_angle = (read_f32_le(chunk, 12) as f64).max(0.0);
        let inner = (read_f32_le(chunk, 16) as f64).max(0.0);
        let outer = (read_f32_le(chunk, 20) as f64).max(inner);
        let density = (read_f32_le(chunk, 24) as f64).max(0.0);
        let shell_volume = solid_angle * (outer.powi(3) - inner.powi(3)) / 3.0;
        let mass = shell_volume * density;
        if !mass.is_finite() || mass <= 0.0 || outer <= inner {
            continue;
        }
        let radial_centroid = 0.75 * (outer.powi(4) - inner.powi(4))
            / (outer.powi(3) - inner.powi(3)).max(f64::MIN_POSITIVE);
        let position = direction * radial_centroid;
        if !position.is_finite() {
            continue;
        }
        sources.push(FrequencyDomainPointSource { position, mass });
        total_mass += mass;
        radius = radius.max(outer);
    }
    if sources.is_empty() || !total_mass.is_finite() || radius <= 0.0 {
        return;
    }
    commands.insert_resource(AggregatedGravitySource {
        sources,
        total_mass,
        radius,
        source_hash: hash_source_bytes(&radial.bytes),
    });
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0_u8; 4]))
}

fn hash_source_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes);
    hasher.finish()
}
