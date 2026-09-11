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
/// Bumps GPU node/sensitivity caches when the shared 64-node κ layout changes.
pub(crate) const EQ184_QUADRATURE_LAYOUT: u64 = 2;
pub(crate) const EQ184_BASE_LAPLACE_SIGMA: f64 = 1.0e-3;

/// Packed Eq.121 modes: `[kx, ky, kz, coeff_re, coeff_im]` per quadrature node.
pub(crate) const EQUATION121_MODE_STRIDE: usize = 5;
/// Trailer after the 64 Fourier records: `[cmx, cmy, cmz, GM, sentinel]`.
/// The sentinel (`2`) marks the analytic k→0 monopole of the same integral.
const EQUATION121_NEWTON_SENTINEL: f64 = 2.0;

const GAUSS_NODES: [f64; 4] = [
    -0.861_136_311_6,
    -0.339_981_043_6,
    0.339_981_043_6,
    0.861_136_311_6,
];
const GAUSS_WEIGHTS: [f64; 4] = [
    0.347_854_845_1,
    0.652_145_154_9,
    0.652_145_154_9,
    0.347_854_845_1,
];

/// Finite reciprocal-space realization of equation (121):
/// `g(q) = G/(2π)³ ∫ (4π i κ / κ²) ρ̂(κ) e^{i κ·q} d³κ`.
///
/// `coeff` already folds `G·w/(2π² κ²) ρ̂(κ)` so evaluation is
/// `g -= Im(coeff e^{i κ·q}) κ` and `U += Re(coeff e^{i κ·q})`.
pub fn build_equation121_modes(bytes: &[u8], source_radius: f64) -> Option<Vec<f64>> {
    if bytes.len() < 32 || !source_radius.is_finite() || source_radius <= 0.0 {
        return None;
    }
    let cells = bytes.as_chunks::<32>().0;
    if cells.is_empty() {
        return None;
    }
    let (mass, center) = quadrature_mass_centroid(cells)?;
    let mut packed = Vec::with_capacity((EQ184_QUADRATURE_COUNT + 1) * EQUATION121_MODE_STRIDE);
    for index in 0..EQ184_QUADRATURE_COUNT {
        let (k, weight) = eq184_quadrature_node(index, source_radius)?;
        let k_squared = k.length_squared();
        if !(k_squared > 0.0) {
            return None;
        }
        let mut rho = Complex64::new(0.0, 0.0);
        for chunk in cells {
            rho += volume_cell_spectrum(k, chunk);
        }
        // Discrete 64-node Riemann sums miss κ→0, which is exactly the 1/r²
        // monopole. Subtract the point-mass spectrum M e^{-iκ·cm} from ρ̂ so the
        // 64 nodes carry only the residual, then restore GM r̂/r² at evaluation.
        // That is still equation (121): IR analytic + UV quadrature of the same
        // integrand. Without it, live FD is a weak flyby hook while FMM orbits.
        rho -= Complex64::from_polar(mass, -k.dot(center));
        let coefficient =
            f64::from(G) * weight / (2.0 * std::f64::consts::PI.powi(2) * k_squared);
        let mode = rho * coefficient;
        packed.extend([k.x, k.y, k.z, mode.re, mode.im]);
    }
    packed.extend([
        center.x,
        center.y,
        center.z,
        f64::from(G) * mass,
        EQUATION121_NEWTON_SENTINEL,
    ]);
    Some(packed)
}

/// Evaluate the packed Eq.121 operator at a body-frame position.
#[cfg_attr(not(test), allow(dead_code))]
pub fn evaluate_equation121(modes: &[f64], position: DVec3) -> Option<(DVec3, f64)> {
    if modes.len() % EQUATION121_MODE_STRIDE != 0 || modes.is_empty() {
        return None;
    }
    let (fourier, newton) = split_equation121_modes(modes);
    let mut gravity = DVec3::ZERO;
    let mut potential = 0.0;
    for mode in fourier.as_chunks::<EQUATION121_MODE_STRIDE>().0 {
        let k = DVec3::new(mode[0], mode[1], mode[2]);
        let phase = k.dot(position);
        let (sin_phase, cos_phase) = phase.sin_cos();
        let re = mode[3] * cos_phase - mode[4] * sin_phase;
        let im = mode[3] * sin_phase + mode[4] * cos_phase;
        gravity -= im * k;
        potential += re;
    }
    if let Some((center, gravitational_parameter)) = newton {
        let offset = position - center;
        let distance_squared = offset.length_squared();
        if distance_squared > 0.0 {
            let inverse_distance = distance_squared.sqrt().recip();
            // Formula-faithful IR/UV split of Eq.(121): uncapped residual of
            // (ρ̂−M) plus the analytic k→0 monopole. Near-surface erf/erfc
            // (derivation appendix) is a separate long/short split, not a
            // residual magnitude clamp.
            let monopole = -gravitational_parameter * offset * inverse_distance.powi(3);
            gravity += monopole;
            potential += gravitational_parameter * inverse_distance;
        }
    }
    (gravity.is_finite() && potential.is_finite()).then_some((gravity, potential))
}

fn split_equation121_modes(modes: &[f64]) -> (&[f64], Option<(DVec3, f64)>) {
    let trailer_start = EQ184_QUADRATURE_COUNT * EQUATION121_MODE_STRIDE;
    if modes.len() != trailer_start + EQUATION121_MODE_STRIDE {
        return (modes, None);
    }
    let trailer = &modes[trailer_start..];
    if trailer[4] != EQUATION121_NEWTON_SENTINEL || !(trailer[3] > 0.0) {
        return (modes, None);
    }
    (
        &modes[..trailer_start],
        Some((DVec3::new(trailer[0], trailer[1], trailer[2]), trailer[3])),
    )
}

pub(crate) fn quadrature_mass_centroid(cells: &[[u8; 32]]) -> Option<(f64, DVec3)> {
    let mut mass = 0.0;
    let mut mass_position = DVec3::ZERO;
    for chunk in cells {
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
        let cell_mass = shell_volume * density;
        if !cell_mass.is_finite() || cell_mass <= 0.0 || outer <= inner {
            continue;
        }
        let radial_centroid = 0.75 * (outer.powi(4) - inner.powi(4))
            / (outer.powi(3) - inner.powi(3)).max(f64::MIN_POSITIVE);
        let position = direction * radial_centroid;
        if !position.is_finite() {
            continue;
        }
        mass += cell_mass;
        mass_position += cell_mass * position;
    }
    (mass > 0.0 && mass_position.is_finite()).then_some((mass, mass_position / mass))
}

fn volume_cell_spectrum(k: DVec3, chunk: &[u8; 32]) -> Complex64 {
    let direction = DVec3::new(
        read_f32_le(chunk, 0) as f64,
        read_f32_le(chunk, 4) as f64,
        read_f32_le(chunk, 8) as f64,
    );
    let solid_angle = (read_f32_le(chunk, 12) as f64).max(0.0);
    let inner = (read_f32_le(chunk, 16) as f64).max(0.0);
    let outer = (read_f32_le(chunk, 20) as f64).max(inner);
    let density = (read_f32_le(chunk, 24) as f64).max(0.0);
    let half_width = 0.5 * (outer - inner);
    let midpoint = 0.5 * (outer + inner);
    if half_width <= 0.0 || density <= 0.0 || solid_angle <= 0.0 || !direction.is_finite() {
        return Complex64::new(0.0, 0.0);
    }
    let mut result = Complex64::new(0.0, 0.0);
    for (node, weight) in GAUSS_NODES.into_iter().zip(GAUSS_WEIGHTS) {
        let radius = midpoint + half_width * node;
        let volume_weight = density * solid_angle * radius * radius * half_width * weight;
        let phase = -k.dot(direction * radius);
        result += Complex64::from_polar(volume_weight, phase);
    }
    result
}

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
    // Four Gauss–Legendre shells on ln κ keep the packed 64-node layout while
    // covering quadrupole-small IR residual (κR ~ 1/2π) through body-scale UV
    // (κR ~ 4π, λ = R/2). Linear [1/R, π/R] left the residual almost without
    // near-field content, so live Eq.121 orbits collapsed toward a hook.
    // Analytic IR monopole stays in the trailer; nodes never sample κ → 0.
    let radius = source_radius.max(1.0);
    let k_min = 1.0 / (2.0 * std::f64::consts::PI * radius);
    let k_max = 4.0 * std::f64::consts::PI / radius;
    let ln_min = k_min.ln();
    let ln_max = k_max.ln();
    let ln_mid = 0.5 * (ln_max + ln_min);
    let ln_half = 0.5 * (ln_max - ln_min);
    let shell = index / EQ184_DIRECTIONS_PER_SHELL;
    let ln_k = ln_mid + ln_half * GAUSS_NODES[shell];
    let wave_number = ln_k.exp();
    let angular_weight = std::f64::consts::TAU * 2.0 / EQ184_DIRECTIONS_PER_SHELL as f64;
    let volume_weight = wave_number.powi(3) * ln_half * GAUSS_WEIGHTS[shell] * angular_weight;
    Some((direction * wave_number, volume_weight))
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

/// Mass-preserving centroid residue retained for CPU comparison, planning, and
/// inversion basis construction. The runtime Eq.121/Eq.184 GPU paths upload
/// `DensityQuadratureSource` cells directly and do not use this as the asteroid
/// force model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyDomainPointSource {
    pub position: DVec3,
    pub mass: f64,
}

#[derive(Resource, Default)]
pub struct AggregatedGravitySource {
    pub sources: Vec<FrequencyDomainPointSource>,
    pub constant_sources: Vec<FrequencyDomainPointSource>,
    pub total_mass: f64,
    pub constant_total_mass: f64,
    pub radius: f64,
    pub source_hash: u64,
    pub constant_hash: u64,
}

/// Converts every shared density quadrature cell into one mass-preserving residue
/// record used to evaluate the discrete density Fourier transform. Cells are
/// not rebinned: doing so damps high-k content before equation (184) sees it.
pub fn build_aggregated_gravity_source_system(
    mut commands: Commands,
    quadrature: Option<Res<DensityQuadratureSource>>,
    existing: Option<Res<AggregatedGravitySource>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(quadrature) = quadrature else { return };
    let record_count = quadrature.bytes.len() / 32;
    if record_count == 0 {
        return;
    }

    let (sources, total_mass, radius) = parse_quadrature(&quadrature.bytes);
    let (constant_sources, constant_total_mass, constant_radius) =
        parse_quadrature(&quadrature.constant_bytes);
    if sources.is_empty() || !total_mass.is_finite() || radius <= 0.0 {
        return;
    }
    if constant_sources.is_empty() || !constant_total_mass.is_finite() || constant_radius <= 0.0 {
        return;
    }
    commands.insert_resource(AggregatedGravitySource {
        sources,
        constant_sources,
        total_mass,
        constant_total_mass,
        radius: radius.max(constant_radius),
        source_hash: hash_source_bytes(&quadrature.bytes),
        constant_hash: hash_source_bytes(&quadrature.constant_bytes),
    });
}

fn parse_quadrature(bytes: &[u8]) -> (Vec<FrequencyDomainPointSource>, f64, f64) {
    let mut sources = Vec::with_capacity(bytes.len() / 32);
    let mut total_mass = 0.0;
    let mut radius = 0.0_f64;
    for chunk in bytes.as_chunks::<32>().0 {
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
    (sources, total_mass, radius)
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0_u8; 4]))
}

fn hash_source_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell_bytes(direction: DVec3, solid_angle: f32, inner: f32, outer: f32, density: f32) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [
            direction.x as f32,
            direction.y as f32,
            direction.z as f32,
            solid_angle,
            inner,
            outer,
            density,
            0.0,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn fibonacci_sphere_bytes(radius: f32, density: f32, directions: usize) -> Vec<u8> {
        let solid_angle = (4.0 * std::f32::consts::PI) / directions as f32;
        let mut bytes = Vec::new();
        for index in 0..directions {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / directions as f64;
            let radius_xy = (1.0 - z * z).max(0.0).sqrt();
            let phi = 2.399_963_229_728_653 * index as f64;
            bytes.extend(shell_bytes(
                DVec3::new(radius_xy * phi.cos(), radius_xy * phi.sin(), z),
                solid_angle,
                0.0,
                radius,
                density,
            ));
        }
        bytes
    }

    #[test]
    fn equation121_modes_attract_toward_mass() {
        let bytes = shell_bytes(DVec3::Z, std::f32::consts::TAU * 2.0, 0.0, 100.0, 1.0e3);
        let modes = build_equation121_modes(&bytes, 100.0).expect("modes");
        let (gravity, potential) =
            evaluate_equation121(&modes, DVec3::new(0.0, 0.0, 500.0)).expect("field");
        assert!(potential > 0.0);
        // Outside a centered shell along +z, attraction should pull toward -z.
        assert!(gravity.z < 0.0);
        assert!(gravity.length() > 0.0);
    }

    #[test]
    fn equation121_recovers_near_newtonian_sphere() {
        // Uniform sphere: R=450 m, ρ chosen so M = RYUGU_MASS.
        let radius = 450.0_f32;
        let volume = 4.0 / 3.0 * std::f64::consts::PI * (radius as f64).powi(3);
        let density = (f64::from(RYUGU_MASS) / volume) as f32;
        let bytes = fibonacci_sphere_bytes(radius, density, 64);
        let modes = build_equation121_modes(&bytes, radius as f64).expect("modes");
        let mass = bytes.as_chunks::<32>().0.iter().map(|chunk| {
            let solid_angle = read_f32_le(chunk, 12) as f64;
            let inner = read_f32_le(chunk, 16) as f64;
            let outer = read_f32_le(chunk, 20) as f64;
            let density = read_f32_le(chunk, 24) as f64;
            solid_angle * (outer.powi(3) - inner.powi(3)) / 3.0 * density
        }).sum::<f64>();
        let gm = f64::from(G) * mass;
        for distance in [500.0, 620.0, 900.0] {
            let position = DVec3::new(-distance, 0.0, 0.0);
            let (gravity, _) = evaluate_equation121(&modes, position).expect("field");
            let newton = gm / (distance * distance);
            let ratio = gravity.length() as f64 / newton;
            assert!(
                gravity.x > 0.0,
                "sphere at -x must attract toward +x, got {gravity:?}"
            );
            assert!(
                (0.85..=1.15).contains(&ratio),
                "Eq.121 |g| / (GM/r²) = {ratio:.3} at r={distance} (newton={newton:.3e}, |g|={:.3e})",
                gravity.length()
            );
        }
        let ic = DVec3::new(-617.0, 0.0, -65.0);
        let (gravity, _) = evaluate_equation121(&modes, ic).expect("IC field");
        let newton = gm / ic.length_squared();
        let ratio = gravity.length() as f64 / newton;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "Eq.121 at the live IC must be near-Newtonian, got {ratio:.3}"
        );
        let fourier_only = &modes[..EQ184_QUADRATURE_COUNT * EQUATION121_MODE_STRIDE];
        let (residual, _) = evaluate_equation121(fourier_only, ic).expect("residual");
        assert!(
            residual.length() < 0.35 * newton,
            "64-node residual should be a correction, not the monopole; |g_res|/Newton={:.3}",
            residual.length() / newton
        );
    }

    #[test]
    fn eq184_quadrature_uses_log_gauss_body_scale_band() {
        let radius = 450.0;
        let mut wave_numbers = Vec::new();
        for index in 0..EQ184_QUADRATURE_COUNT {
            let (k, weight) = eq184_quadrature_node(index, radius).expect("node");
            assert!(weight > 0.0 && weight.is_finite());
            wave_numbers.push(k.length());
        }
        wave_numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k_min = *wave_numbers.first().expect("k");
        let k_max = *wave_numbers.last().expect("k");
        assert!(
            k_min < 1.0 / radius,
            "IR edge should sit below 1/R so ρ̂−M is quadrupole-small, got {}",
            k_min * radius
        );
        assert!(
            k_max * radius > std::f64::consts::PI,
            "UV edge should exceed the old π/R slab, got κR={}",
            k_max * radius
        );
        wave_numbers.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
        assert_eq!(wave_numbers.len(), EQ184_RADIAL_SHELLS);
    }
}
