// Numerically certified building blocks for Equation (106).
//
// The documents define the transformed straight-line kernel through
// `F(s; a, z') = integral_0^infinity exp(-s h) / R(h) dh`.  This module
// evaluates that definition directly in complex arithmetic, then derives
// `K_H` and `K_V` from the identities in Eqs. (63)--(69).  It is intentionally
// independent from the render pipeline so the CPU reference can certify a
// future WGSL implementation before that implementation is allowed to drive
// the trajectory.

use bevy::math::DVec3;

const GL8_NODES: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL8_WEIGHTS: [f64; 8] = [
    0.101_228_536_290_376_3,
    0.222_381_034_453_374_5,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

const DEFAULT_QUADRATURE_TOLERANCE: f64 = 1.0e-9;
const MAX_ADAPTIVE_DEPTH: u8 = 24;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    pub fn exp(self) -> Self {
        let amplitude = self.re.exp();
        Self::new(amplitude * self.im.cos(), amplitude * self.im.sin())
    }

    pub fn reciprocal(self) -> Option<Self> {
        let norm_squared = self.re.mul_add(self.re, self.im * self.im);
        (norm_squared.is_finite() && norm_squared > f64::MIN_POSITIVE)
            .then_some(Self::new(self.re / norm_squared, -self.im / norm_squared))
    }

    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl std::ops::AddAssign for Complex64 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl std::ops::Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re.mul_add(other.re, -self.im * other.im),
            self.re.mul_add(other.im, self.im * other.re),
        )
    }
}

impl std::ops::Mul<f64> for Complex64 {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self::new(self.re * scalar, self.im * scalar)
    }
}

impl std::ops::Div for Complex64 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let norm_squared = other.re.mul_add(other.re, other.im * other.im);
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Self::new(f64::NAN, f64::NAN);
        }
        Self::new(
            (self.re * other.re + self.im * other.im) / norm_squared,
            (self.im * other.re - self.re * other.im) / norm_squared,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComplexVec3 {
    pub x: Complex64,
    pub y: Complex64,
    pub z: Complex64,
}

impl ComplexVec3 {
    pub const ZERO: Self = Self {
        x: Complex64::ZERO,
        y: Complex64::ZERO,
        z: Complex64::ZERO,
    };

    pub fn real(self) -> DVec3 {
        DVec3::new(self.x.re, self.y.re, self.z.re)
    }

    pub fn imaginary_norm(self) -> f64 {
        DVec3::new(self.x.im, self.y.im, self.z.im).length()
    }

    pub fn scale(self, scalar: Complex64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::AddAssign for ComplexVec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eq106ReferenceLine {
    pub origin: DVec3,
    pub direction: DVec3,
}

impl Eq106ReferenceLine {
    pub fn new(origin: DVec3, direction: DVec3) -> Option<Self> {
        let direction = direction.try_normalize()?;
        Some(Self { origin, direction })
    }

    pub fn point_at(self, h: f64) -> DVec3 {
        self.origin + self.direction * h
    }

    pub fn decompose_source(self, source: DVec3) -> (f64, DVec3) {
        let relative = source - self.origin;
        let z_prime = relative.dot(self.direction);
        (z_prime, relative - z_prime * self.direction)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eq106PointSource {
    pub position: DVec3,
    pub mass: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eq106KernelSample {
    pub psi: Complex64,
    pub k_h: Complex64,
    pub k_v: Complex64,
    pub horizontal_distance: f64,
    pub source_height: f64,
    pub boundary_identity_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eq106FrequencyGrid {
    pub sigma: f64,
    pub omega_step: f64,
    pub half_count: u32,
}

impl Eq106FrequencyGrid {
    pub fn frequencies(self) -> impl Iterator<Item = Complex64> {
        let half = self.half_count as i32;
        (-half..=half).map(move |index| Complex64::new(self.sigma, index as f64 * self.omega_step))
    }

    pub fn validate(self) -> bool {
        self.sigma.is_finite()
            && self.sigma > 0.0
            && self.omega_step.is_finite()
            && self.omega_step > 0.0
            && self.half_count > 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eq106TransformSample {
    pub frequency: Complex64,
    pub acceleration: ComplexVec3,
    pub potential: Complex64,
    pub boundary_identity_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eq106Certificate {
    pub max_acceleration_relative_error: f64,
    pub max_potential_relative_error: f64,
    pub max_boundary_identity_error: f64,
    pub sample_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Eq106Error {
    InvalidFrequency,
    SourceOnReferenceLine,
    QuadratureFailed,
    SingularPadeSystem,
    PadePoleOnInterval,
    PadeNotConverged,
    CertificationFailed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadeAccelerationCertificate {
    pub acceleration: DVec3,
    pub relative_residual: f64,
    pub adjacent_order_residual: f64,
    pub order: usize,
}

/// Evaluates the scalar transform `Psi = F` and derives the two Equation (106)
/// scalar kernels without dropping the vertical boundary term.
pub fn eq106_kernel_sample(
    line: Eq106ReferenceLine,
    source: DVec3,
    frequency: Complex64,
    tolerance: f64,
) -> Result<Eq106KernelSample, Eq106Error> {
    eq106_kernel_sample_internal(line, source, frequency, tolerance, true)
}

fn eq106_kernel_sample_internal(
    line: Eq106ReferenceLine,
    source: DVec3,
    frequency: Complex64,
    tolerance: f64,
    verify_boundary: bool,
) -> Result<Eq106KernelSample, Eq106Error> {
    if !frequency.is_finite() || frequency.re <= 0.0 {
        return Err(Eq106Error::InvalidFrequency);
    }
    let (z_prime, transverse) = line.decompose_source(source);
    let a = transverse.length();
    if !a.is_finite() || a <= f64::MIN_POSITIVE {
        return Err(Eq106Error::SourceOnReferenceLine);
    }

    let scalar_integrand = |h: f64| {
        let separation = line.point_at(h) - source;
        let radius = separation.length();
        Complex64::new(-frequency.re * h, -frequency.im * h).exp() * radius.recip()
    };
    let horizontal_integrand = |h: f64| {
        let separation = line.point_at(h) - source;
        let radius_squared = separation.length_squared();
        let radius = radius_squared.sqrt();
        Complex64::new(-frequency.re * h, -frequency.im * h).exp()
            * (-a / (radius_squared * radius))
    };
    let vertical_integrand = |h: f64| {
        let separation = line.point_at(h) - source;
        let radius_squared = separation.length_squared();
        let radius = radius_squared.sqrt();
        Complex64::new(-frequency.re * h, -frequency.im * h).exp()
            * ((z_prime - h) / (radius_squared * radius))
    };

    let psi = integrate_on_half_line(scalar_integrand, tolerance)?;
    let horizontal_derivative = integrate_on_half_line(horizontal_integrand, tolerance)?;
    let boundary_distance = (a.mul_add(a, z_prime * z_prime)).sqrt();
    let vertical_from_identity = frequency * psi - Complex64::new(boundary_distance.recip(), 0.0);
    let boundary_identity_error = if verify_boundary {
        let vertical_from_integral = integrate_on_half_line(vertical_integrand, tolerance)?;
        (vertical_from_integral - vertical_from_identity).abs()
    } else {
        0.0
    };

    Ok(Eq106KernelSample {
        psi,
        k_h: horizontal_derivative * a,
        k_v: vertical_from_identity * a,
        horizontal_distance: a,
        source_height: z_prime,
        boundary_identity_error,
    })
}

/// Computes the full density-summed transformed acceleration and potential for
/// one shared complex frequency. A caller can reuse this result for every
/// observation height on the same reference line through Bromwich inversion.
pub fn eq106_transform(
    line: Eq106ReferenceLine,
    sources: &[Eq106PointSource],
    frequency: Complex64,
    gravitational_constant: f64,
    tolerance: f64,
) -> Result<Eq106TransformSample, Eq106Error> {
    let mut acceleration = ComplexVec3::ZERO;
    let mut potential = Complex64::ZERO;
    let mut boundary_identity_error = 0.0_f64;
    for source in sources {
        if !source.mass.is_finite() || source.mass == 0.0 {
            continue;
        }
        let kernel =
            eq106_kernel_sample_internal(line, source.position, frequency, tolerance, false)?;
        boundary_identity_error = boundary_identity_error.max(kernel.boundary_identity_error);
        let (_, transverse) = line.decompose_source(source.position);
        let horizontal_direction = transverse / kernel.horizontal_distance;
        // The reference line is the local cylindrical axis, so the observer's
        // transverse coordinate is zero and Eq. (106) contributes `-e_rho`.
        let horizontal =
            kernel.k_h * (-gravitational_constant * source.mass / kernel.horizontal_distance);
        let vertical =
            kernel.k_v * (gravitational_constant * source.mass / kernel.horizontal_distance);
        acceleration.x += horizontal * horizontal_direction.x + vertical * line.direction.x;
        acceleration.y += horizontal * horizontal_direction.y + vertical * line.direction.y;
        acceleration.z += horizontal * horizontal_direction.z + vertical * line.direction.z;
        potential += kernel.psi * (gravitational_constant * source.mass);
    }
    Ok(Eq106TransformSample {
        frequency,
        acceleration,
        potential,
        boundary_identity_error,
    })
}

pub fn inverse_bromwich_acceleration(
    samples: &[Eq106TransformSample],
    grid: Eq106FrequencyGrid,
    h: f64,
) -> Result<ComplexVec3, Eq106Error> {
    if !grid.validate() || !h.is_finite() {
        return Err(Eq106Error::InvalidFrequency);
    }
    let mut result = ComplexVec3::ZERO;
    for sample in samples {
        let phase = Complex64::new(grid.sigma * h, sample.frequency.im * h).exp();
        result += sample.acceleration.scale(phase);
    }
    let endpoint_factor = if h == 0.0 { 2.0 } else { 1.0 };
    Ok(result.scale(Complex64::new(
        endpoint_factor * grid.omega_step / (2.0 * std::f64::consts::PI),
        0.0,
    )))
}

pub fn inverse_bromwich_potential(
    samples: &[Eq106TransformSample],
    grid: Eq106FrequencyGrid,
    h: f64,
) -> Result<Complex64, Eq106Error> {
    if !grid.validate() || !h.is_finite() {
        return Err(Eq106Error::InvalidFrequency);
    }
    let mut result = Complex64::ZERO;
    for sample in samples {
        let phase = Complex64::new(grid.sigma * h, sample.frequency.im * h).exp();
        result += sample.potential * phase;
    }
    let endpoint_factor = if h == 0.0 { 2.0 } else { 1.0 };
    Ok(result * (endpoint_factor * grid.omega_step / (2.0 * std::f64::consts::PI)))
}

pub fn direct_point_field(
    observer: DVec3,
    sources: &[Eq106PointSource],
    gravitational_constant: f64,
) -> (DVec3, f64) {
    let mut acceleration = DVec3::ZERO;
    let mut potential = 0.0;
    for source in sources {
        if !source.mass.is_finite() || source.mass == 0.0 {
            continue;
        }
        let displacement = source.position - observer;
        let radius_squared = displacement.length_squared();
        if !radius_squared.is_finite() || radius_squared <= f64::MIN_POSITIVE {
            continue;
        }
        let radius = radius_squared.sqrt();
        let scale = gravitational_constant * source.mass;
        acceleration += displacement * (scale / (radius_squared * radius));
        potential += scale / radius;
    }
    (acceleration, potential)
}

pub fn direct_point_gradient(
    observer: DVec3,
    sources: &[Eq106PointSource],
    gravitational_constant: f64,
) -> [[f64; 3]; 3] {
    let mut gradient = [[0.0_f64; 3]; 3];
    for source in sources {
        let displacement = source.position - observer;
        let radius_squared = displacement.length_squared();
        if !source.mass.is_finite()
            || !radius_squared.is_finite()
            || radius_squared <= f64::MIN_POSITIVE
        {
            continue;
        }
        let radius = radius_squared.sqrt();
        let scale = gravitational_constant * source.mass / (radius_squared * radius);
        let factor = 3.0 / radius_squared;
        let components = [displacement.x, displacement.y, displacement.z];
        for row in 0..3 {
            for column in 0..3 {
                let identity = if row == column { 1.0 } else { 0.0 };
                gradient[row][column] +=
                    scale * (-identity + factor * components[row] * components[column]);
            }
        }
    }
    gradient
}

/// Builds the density-summed one-parameter Taylor jet along a displaced line.
pub fn directional_acceleration_taylor_coefficients(
    center: DVec3,
    displacement: DVec3,
    sources: &[Eq106PointSource],
    gravitational_constant: f64,
    maximum_order: usize,
) -> Result<[Vec<Complex64>; 3], Eq106Error> {
    let mut coefficients =
        std::array::from_fn(|_| vec![Complex64::ZERO; maximum_order.saturating_add(1)]);
    for source in sources {
        if !source.mass.is_finite() || source.mass == 0.0 {
            continue;
        }
        let r0 = source.position - center;
        let x0 = r0.length_squared();
        if !x0.is_finite() || x0 <= f64::MIN_POSITIVE {
            return Err(Eq106Error::CertificationFailed);
        }
        let x1 = -2.0 * r0.dot(displacement);
        let x2 = displacement.length_squared();
        let radial_power = quadratic_power_series(x0, x1, x2, -1.5, maximum_order)?;
        let scale = gravitational_constant * source.mass;
        let r0_components = [r0.x, r0.y, r0.z];
        let displacement_components = [displacement.x, displacement.y, displacement.z];
        for component in 0..3 {
            for order in 0..=maximum_order {
                let previous = order
                    .checked_sub(1)
                    .map_or(0.0, |index| radial_power[index]);
                let value = scale
                    * (r0_components[component] * radial_power[order]
                        - displacement_components[component] * previous);
                coefficients[component][order] += Complex64::new(value, 0.0);
            }
        }
    }
    coefficients
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(coefficients)
        .ok_or(Eq106Error::CertificationFailed)
}

pub fn evaluate_directional_taylor(
    coefficients: &[Vec<Complex64>; 3],
    order: usize,
) -> Result<DVec3, Eq106Error> {
    if coefficients.iter().any(|series| series.len() <= order) {
        return Err(Eq106Error::PadeNotConverged);
    }
    let values = std::array::from_fn::<_, 3, _>(|component| {
        coefficients[component][..=order]
            .iter()
            .fold(Complex64::ZERO, |sum, value| sum + *value)
    });
    let acceleration = DVec3::new(values[0].re, values[1].re, values[2].re);
    (values.iter().all(|value| value.is_finite()) && acceleration.is_finite())
        .then_some(acceleration)
        .ok_or(Eq106Error::CertificationFailed)
}

/// Constructs a certified diagonal [m/m] Pade continuation from the Taylor jet.
pub fn certify_directional_pade(
    center: DVec3,
    displacement: DVec3,
    sources: &[Eq106PointSource],
    gravitational_constant: f64,
    order: usize,
    tolerance: f64,
) -> Result<PadeAccelerationCertificate, Eq106Error> {
    if order < 2 || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(Eq106Error::PadeNotConverged);
    }
    let coefficients = directional_acceleration_taylor_coefficients(
        center,
        displacement,
        sources,
        gravitational_constant,
        2 * order,
    )?;
    let mut value = [0.0_f64; 3];
    let mut adjacent = [0.0_f64; 3];
    for component in 0..3 {
        let component_scale = coefficients[component]
            .iter()
            .map(|coefficient| coefficient.abs())
            .fold(0.0_f64, f64::max);
        if component_scale <= 1.0e-30 {
            continue;
        }
        let approximant = pade_from_taylor(&coefficients[component], order, order)?;
        let evaluated = approximant.evaluate(Complex64::ONE)?;
        let adjacent_approximant =
            pade_from_taylor(&coefficients[component], order - 1, order - 1)?;
        let adjacent_evaluated = adjacent_approximant.evaluate(Complex64::ONE)?;
        if evaluated.im.abs() > tolerance || adjacent_evaluated.im.abs() > tolerance {
            return Err(Eq106Error::PadeNotConverged);
        }
        value[component] = evaluated.re;
        adjacent[component] = adjacent_evaluated.re;
    }
    let acceleration = DVec3::from_array(value);
    let adjacent_acceleration = DVec3::from_array(adjacent);
    let (reference, _) = direct_point_field(center + displacement, sources, gravitational_constant);
    let scale = reference.length().max(1.0e-14);
    let relative_residual = (acceleration - reference).length() / scale;
    let adjacent_order_residual = (acceleration - adjacent_acceleration).length() / scale;
    if !acceleration.is_finite()
        || !relative_residual.is_finite()
        || !adjacent_order_residual.is_finite()
        || relative_residual > tolerance
        || adjacent_order_residual > tolerance * 4.0
    {
        return Err(Eq106Error::PadeNotConverged);
    }
    Ok(PadeAccelerationCertificate {
        acceleration,
        relative_residual,
        adjacent_order_residual,
        order,
    })
}

fn quadratic_power_series(
    x0: f64,
    x1: f64,
    x2: f64,
    exponent: f64,
    maximum_order: usize,
) -> Result<Vec<f64>, Eq106Error> {
    if !x0.is_finite() || x0 <= f64::MIN_POSITIVE {
        return Err(Eq106Error::CertificationFailed);
    }
    let polynomial = [x0, x1, x2];
    let mut result = vec![0.0_f64; maximum_order.saturating_add(1)];
    result[0] = x0.powf(exponent);
    for order in 1..=maximum_order {
        let mut numerator = 0.0;
        for degree in 1..=2.min(order) {
            numerator += ((exponent + 1.0) * degree as f64 - order as f64)
                * polynomial[degree]
                * result[order - degree];
        }
        result[order] = numerator / (order as f64 * x0);
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
        .ok_or(Eq106Error::CertificationFailed)
}

pub fn certify_eq106_line(
    line: Eq106ReferenceLine,
    sources: &[Eq106PointSource],
    grid: Eq106FrequencyGrid,
    gravitational_constant: f64,
    h_values: &[f64],
    tolerance: f64,
) -> Result<(Vec<Eq106TransformSample>, Eq106Certificate), Eq106Error> {
    if sources.is_empty() || h_values.is_empty() || !grid.validate() {
        return Err(Eq106Error::CertificationFailed);
    }
    // Runtime certification uses bounded tolerance; direct-field and boundary
    // probes remain the acceptance checks.
    let transform_tolerance = 1.0e-6;
    let samples = grid
        .frequencies()
        .map(|frequency| {
            eq106_transform(
                line,
                sources,
                frequency,
                gravitational_constant,
                transform_tolerance,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let spectral_potential_origin = inverse_bromwich_potential(&samples, grid, 0.0)?.re;
    let (_, direct_potential_origin) =
        direct_point_field(line.origin, sources, gravitational_constant);
    let mut max_acceleration_relative_error = 0.0_f64;
    let mut max_potential_relative_error = 0.0_f64;
    for &h in h_values {
        if !h.is_finite() || h < 0.0 {
            return Err(Eq106Error::CertificationFailed);
        }
        let spectral_acceleration = inverse_bromwich_acceleration(&samples, grid, h)?.real();
        let spectral_potential = inverse_bromwich_potential(&samples, grid, h)?.re;
        let (direct_acceleration, direct_potential) =
            direct_point_field(line.point_at(h), sources, gravitational_constant);
        let acceleration_scale = direct_acceleration.length().max(1.0e-14);
        // Keep tiny test masses from amplifying endpoint quadrature noise.
        let potential_scale = direct_potential
            .abs()
            .max(direct_potential_origin.abs())
            .max(1.0e-12);
        max_acceleration_relative_error = max_acceleration_relative_error
            .max((spectral_acceleration - direct_acceleration).length() / acceleration_scale);
        let spectral_delta = spectral_potential - spectral_potential_origin;
        let direct_delta = direct_potential - direct_potential_origin;
        max_potential_relative_error = max_potential_relative_error
            .max((spectral_delta - direct_delta).abs() / potential_scale);
    }
    let representative_frequencies = [
        Complex64::new(grid.sigma, -(grid.half_count as f64) * grid.omega_step),
        Complex64::new(grid.sigma, 0.0),
        Complex64::new(grid.sigma, grid.half_count as f64 * grid.omega_step),
    ];
    let mut max_boundary_identity_error = 0.0_f64;
    for source in sources.iter().take(4) {
        for frequency in representative_frequencies {
            let kernel = eq106_kernel_sample(line, source.position, frequency, 2.0e-8)?;
            max_boundary_identity_error =
                max_boundary_identity_error.max(kernel.boundary_identity_error);
        }
    }
    let certificate = Eq106Certificate {
        max_acceleration_relative_error,
        max_potential_relative_error,
        max_boundary_identity_error,
        sample_count: h_values.len(),
    };
    (max_acceleration_relative_error <= tolerance
        && max_potential_relative_error <= tolerance
        && max_boundary_identity_error <= 2.0e-6)
        .then_some((samples, certificate))
        .ok_or(Eq106Error::CertificationFailed)
}
