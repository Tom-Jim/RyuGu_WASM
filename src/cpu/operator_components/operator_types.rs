// Shared operator data for the Fourier-Chebyshev side of Eq. (106).
//
// The table approximates the universal toroidal harmonics
// `Q_{m-1/2}(chi)` on `x = log(chi - 1)`.  The CPU builds it with f64
// adaptive Gauss-Legendre quadrature and certifies interpolation at the
// segment midpoints.  The render-world WGSL evaluator consumes the exact
// serialized f32 coefficient buffer with a fixed Clenshaw loop.
//
// This is a truncated discrete operator: mode truncation, interval mapping,
// coefficient quantization and GPU f32 arithmetic are reported explicitly.

use crate::cpu::eq106_reference::{Complex64, Eq106Error};
use crate::interface::components::{
    ActiveGravityMethod, GravityRuntimeError, PlanningComparisonState,
};
use bevy::prelude::*;
#[cfg(feature = "regenerate-operators")]
use std::f64::consts::PI;

pub const TOROIDAL_MAX_MODE: usize = 16;
pub const TOROIDAL_MODE_COUNT: usize = TOROIDAL_MAX_MODE + 1;
pub const TOROIDAL_SEGMENT_COUNT: usize = 12;
pub const TOROIDAL_DEGREE: usize = 12;
pub const TOROIDAL_COEFFICIENT_COUNT: usize = TOROIDAL_DEGREE + 1;
#[cfg(feature = "regenerate-operators")]
pub const TOROIDAL_X_MIN: f64 = -10.0;
#[cfg(feature = "regenerate-operators")]
pub const TOROIDAL_X_MAX: f64 = 8.0;
#[cfg(feature = "regenerate-operators")]
pub const TOROIDAL_SEGMENT_STEP: f64 =
    (TOROIDAL_X_MAX - TOROIDAL_X_MIN) / TOROIDAL_SEGMENT_COUNT as f64;

/// The production frequency grid is conjugate symmetric, so the universal
/// Struve--Neumann part only stores k=0..64. Negative frequencies are obtained
/// by conjugation in WGSL.
pub const PSI_FREQUENCY_COUNT: usize = 65;
pub const PSI_SEGMENT_COUNT: usize = 16;
pub const PSI_DEGREE: usize = 8;
pub const PSI_COEFFICIENT_COUNT: usize = PSI_DEGREE + 1;
pub const PSI_LOG_A_MIN: f64 = -3.218_875_824_868_200_6; // ln(0.04 R)
pub const PSI_LOG_A_MAX: f64 = 1.791_759_469_228_055; // ln(6 R)
pub const PSI_LOG_A_STEP: f64 = (PSI_LOG_A_MAX - PSI_LOG_A_MIN) / PSI_SEGMENT_COUNT as f64;
const PSI_OMEGA_STEP: f64 = 0.002;

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
const GL16_NODES: [f64; 16] = [
    0.087_649_410_478_927_84,
    0.462_696_328_915_080_8,
    1.141_057_774_831_227,
    2.129_283_645_098_380_6,
    3.437_086_633_893_206_6,
    5.078_018_614_549_768,
    7.070_338_535_048_234,
    9.438_314_336_391_938,
    12.214_223_368_866_16,
    15.441_527_368_781_62,
    19.180_156_856_753_13,
    23.515_905_693_991_91,
    28.578_729_742_882_14,
    34.583_398_702_286_63,
    41.940_452_647_688_33,
    51.701_160_339_543_32,
];
const GL16_WEIGHTS: [f64; 16] = [
    2.061_517_149_578_01e-1,
    3.310_578_549_508_842e-1,
    2.657_957_776_442_141_5e-1,
    1.362_969_342_963_775_4e-1,
    4.732_892_869_412_522e-2,
    1.129_990_008_033_945_4e-2,
    1.849_070_943_526_31e-3,
    2.042_719_153_082_784_6e-4,
    1.484_458_687_398_129_9e-5,
    6.828_319_330_871_2e-7,
    1.881_024_841_079_67e-8,
    2.862_350_242_973_88e-10,
    2.127_079_033_224_1e-12,
    6.297_967_002_517_88e-15,
    5.050_473_700_035_51e-18,
    4.161_462_370_372_85e-22,
];
#[cfg(feature = "regenerate-operators")]
const MAX_ADAPTIVE_DEPTH: u8 = 20;
#[cfg(feature = "regenerate-operators")]
const QUADRATURE_TOLERANCE: f64 = 2.0e-11;

#[derive(Clone, Debug, PartialEq)]
pub struct ToroidalOperatorTensor {
    pub coefficients: Vec<f32>,
    pub max_midpoint_error: f64,
}

/// Piecewise-Chebyshev representation of
/// `L(x)=integral_0^infinity exp(-x t)/sqrt(1+t^2)dt = pi/2 (H_0(x)-Y_0(x))`
/// and its complex derivative. Together with the finite-eta recurrence in the
/// shader this is the certified `(x,eta)->(Psi,Psi_x)` map used by Eq. (70).
#[derive(Clone, Debug, PartialEq)]
pub struct PsiOperatorTable {
    /// Four interleaved f32 components per coefficient:
    /// `(Re L, Im L, Re L_x, Im L_x)`.
    pub coefficients: Vec<f32>,
    pub radius: f64,
    pub max_validation_error: f64,
    pub max_asymptotic_remainder: f64,
    pub max_axis_limit_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsiMapSample {
    pub psi: Complex64,
    pub psi_x: Complex64,
}

#[derive(Resource, Clone, Debug)]
pub struct Eq106OperatorTensorResource {
    pub tensor: ToroidalOperatorTensor,
}

#[derive(Resource)]
pub(crate) struct Eq106OperatorBuildAttempted;

/// Builds the universal operator once after the density source exists. A
/// failed certificate is surfaced through the existing runtime error overlay;
/// no alternate physics method is selected implicitly.
pub fn build_eq106_operator_tensor_system(
    mut commands: Commands,
    existing: Option<Res<Eq106OperatorTensorResource>>,
    attempted: Option<Res<Eq106OperatorBuildAttempted>>,
    source_data: Option<Res<crate::cpu::curved_arc::AggregatedGravitySource>>,
    active_method: Res<ActiveGravityMethod>,
    planning: Res<PlanningComparisonState>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if existing.is_some()
        || attempted.is_some()
        || source_data.is_none()
        || (*active_method != ActiveGravityMethod::CurvedArcEq106 && !planning.run_requested)
    {
        return;
    }
    // A failed immutable-table certificate is terminal for this app build.
    // Retrying the immutable operator assembly every frame would freeze the
    // browser and hide the actual error overlay.
    commands.insert_resource(Eq106OperatorBuildAttempted);
    match ToroidalOperatorTensor::embedded() {
        Ok(tensor) if tensor.validate(2.0e-4) => {
            commands.insert_resource(Eq106OperatorTensorResource { tensor });
        }
        Ok(tensor) => runtime_error.raise(format!(
            "Equation (106) toroidal operator certification failed ({:.3e}).",
            tensor.max_midpoint_error
        )),
        Err(error) => runtime_error.raise(format!(
            "Equation (106) operator tensor assembly failed: {error}"
        )),
    }
}

impl PsiOperatorTable {
    pub fn coefficient_count() -> usize {
        PSI_FREQUENCY_COUNT * PSI_SEGMENT_COUNT * PSI_COEFFICIENT_COUNT * 4
    }

    #[cfg(feature = "regenerate-operators")]
    pub fn build(radius: f64) -> Result<Self, String> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err("invalid radius for complex Psi operator".into());
        }
        let mut coefficients = vec![0.0_f32; Self::coefficient_count()];
        let nodes = PSI_COEFFICIENT_COUNT;
        let mut max_asymptotic_remainder = 0.0_f64;
        for frequency in 0..PSI_FREQUENCY_COUNT {
            for segment in 0..PSI_SEGMENT_COUNT {
                let x0 = PSI_LOG_A_MIN + segment as f64 * PSI_LOG_A_STEP;
                let x1 = x0 + PSI_LOG_A_STEP;
                let midpoint = 0.5 * (x0 + x1);
                let half_width = 0.5 * (x1 - x0);
                let mut values = vec![[Complex64::ZERO; 2]; nodes];
                for (node, value) in values.iter_mut().enumerate() {
                    let theta = PI * (node as f64 + 0.5) / nodes as f64;
                    let log_a = midpoint + half_width * theta.cos();
                    let normalized_a = log_a.exp();
                    let x = Complex64::new(
                        2.0 * normalized_a,
                        frequency as f64 * PSI_OMEGA_STEP * radius * normalized_a,
                    );
                    let (pair, remainder) = struve_neumann_pair(x)?;
                    *value = pair;
                    max_asymptotic_remainder = max_asymptotic_remainder.max(remainder);
                }
                for degree in 0..=PSI_DEGREE {
                    let mut coefficient = [Complex64::ZERO; 2];
                    for (node, value) in values.iter().enumerate() {
                        let theta = PI * (node as f64 + 0.5) / nodes as f64;
                        let weight = (degree as f64 * theta).cos();
                        coefficient[0] += value[0] * weight;
                        coefficient[1] += value[1] * weight;
                    }
                    let scale = if degree == 0 {
                        1.0 / nodes as f64
                    } else {
                        2.0 / nodes as f64
                    };
                    coefficient[0] = coefficient[0] * scale;
                    coefficient[1] = coefficient[1] * scale;
                    let base = psi_coefficient_index(frequency, segment, degree);
                    for (offset, component) in [
                        coefficient[0].re,
                        coefficient[0].im,
                        coefficient[1].re,
                        coefficient[1].im,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if !component.is_finite() {
                            return Err("non-finite complex Psi coefficient".into());
                        }
                        coefficients[base + offset] = component as f32;
                    }
                }
            }
        }

        // Quarter points exercise both halves of every patch rather than only
        // interpolating back at construction nodes.
        let mut max_validation_error = 0.0_f64;
        for frequency in 0..PSI_FREQUENCY_COUNT {
            for segment in 0..PSI_SEGMENT_COUNT {
                let x0 = PSI_LOG_A_MIN + segment as f64 * PSI_LOG_A_STEP;
                for fraction in [0.25_f64, 0.5, 0.75] {
                    let log_a = x0 + fraction * PSI_LOG_A_STEP;
                    let normalized_a = log_a.exp();
                    let x = Complex64::new(
                        2.0 * normalized_a,
                        frequency as f64 * PSI_OMEGA_STEP * radius * normalized_a,
                    );
                    let (exact, remainder) = struve_neumann_pair(x)?;
                    max_asymptotic_remainder = max_asymptotic_remainder.max(remainder);
                    let estimate =
                        evaluate_psi_coefficients(&coefficients, frequency, segment, log_a);
                    for component in 0..2 {
                        // Mixed relative/absolute certificate: derivatives
                        // crossing zero must not turn sub-nanometric f32
                        // quantization into a percent-level relative error.
                        let scale = exact[component].abs().max(1.0e-6);
                        max_validation_error = max_validation_error
                            .max((estimate[component] - exact[component]).abs() / scale);
                    }
                    for eta in [-0.9_f64, -0.5, 0.0, 0.5, 0.9] {
                        if x.abs() >= 16.0 || -x.re * eta >= 6.0 {
                            max_asymptotic_remainder =
                                max_asymptotic_remainder.max(psi_map_asymptotic(x, eta).1);
                        }
                        let approximate = psi_map_from_base(estimate, x, eta, false)?;
                        let exact_map = psi_map_from_base(exact, x, eta, true)?;
                        for (approximate, exact) in [
                            (approximate.psi, exact_map.psi),
                            (approximate.psi_x, exact_map.psi_x),
                        ] {
                            let error = (approximate - exact).abs() / exact.abs().max(1.0e-6);
                            if error > max_validation_error {
                                max_validation_error = error;
                            }
                        }
                    }
                }
            }
        }
        let mut max_axis_limit_error = 0.0_f64;
        for frequency in [0_usize, 1, 8, 32, 64] {
            for normalized_height in [0.1_f64, 0.5, 1.0, 4.0] {
                let w = Complex64::new(
                    2.0 * normalized_height,
                    frequency as f64 * PSI_OMEGA_STEP * radius * normalized_height,
                );
                let approximate = scaled_e1_gauss_laguerre(w);
                let exact = scaled_e1_reference(w)?;
                max_axis_limit_error =
                    max_axis_limit_error.max((approximate - exact).abs() / exact.abs().max(1.0e-8));
            }
        }
        Ok(Self {
            coefficients,
            radius,
            max_validation_error,
            max_asymptotic_remainder,
            max_axis_limit_error,
        })
    }

    /// Loads the build-time-certified table. No defining integrals are run on
    /// the browser main thread; startup only validates metadata and uploads
    /// the coefficient buffer.
    pub fn embedded(runtime_radius: f64) -> Result<Self, String> {
        const BYTES: &[u8] = include_bytes!("../../../assets/operators/eq106_psi_table.bin");
        if BYTES.len() != 40 + Self::coefficient_count() * 4 || &BYTES[..8] != b"EQ106PSI" {
            return Err("embedded complex Psi table has an invalid header or length".into());
        }
        let read_f64 = |offset: usize| {
            f64::from_le_bytes(BYTES[offset..offset + 8].try_into().unwrap_or([0; 8]))
        };
        let radius = read_f64(8);
        if !runtime_radius.is_finite()
            || runtime_radius <= 0.0
            || (runtime_radius - radius).abs() / radius > 2.0e-3
        {
            return Err(format!(
                "embedded complex Psi table radius {radius:.6} does not match runtime radius {runtime_radius:.6}"
            ));
        }
        let coefficients = BYTES[40..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let table = Self {
            coefficients,
            radius,
            max_validation_error: read_f64(16),
            max_asymptotic_remainder: read_f64(24),
            max_axis_limit_error: read_f64(32),
        };
        table
            .validate(3.0e-3)
            .then_some(table)
            .ok_or_else(|| "embedded complex Psi table certificate is invalid".into())
    }

    pub fn validate(&self, tolerance: f64) -> bool {
        self.coefficients.len() == Self::coefficient_count()
            && self.coefficients.iter().all(|value| value.is_finite())
            && self.radius.is_finite()
            && self.radius > 0.0
            && self.max_validation_error.is_finite()
            && self.max_validation_error <= tolerance
            && self.max_asymptotic_remainder.is_finite()
            && self.max_asymptotic_remainder <= tolerance
            && self.max_axis_limit_error.is_finite()
            && self.max_axis_limit_error <= tolerance
    }

    pub fn as_le_bytes(&self) -> Vec<u8> {
        bytemuck::cast_slice(&self.coefficients).to_vec()
    }

    pub fn evaluate(
        &self,
        signed_frequency: i32,
        normalized_a: f64,
        eta: f64,
    ) -> Result<PsiMapSample, String> {
        if signed_frequency.unsigned_abs() as usize >= PSI_FREQUENCY_COUNT
            || !normalized_a.is_finite()
            || !(PSI_LOG_A_MIN.exp()..=PSI_LOG_A_MAX.exp()).contains(&normalized_a)
            || !eta.is_finite()
        {
            return Err("complex Psi query is outside the certified table domain".into());
        }
        let log_a = normalized_a.ln();
        let segment = (((log_a - PSI_LOG_A_MIN) / PSI_LOG_A_STEP).floor() as usize)
            .min(PSI_SEGMENT_COUNT - 1);
        let mut base = evaluate_psi_coefficients(
            &self.coefficients,
            signed_frequency.unsigned_abs() as usize,
            segment,
            log_a,
        );
        if signed_frequency < 0 {
            base[0].im = -base[0].im;
            base[1].im = -base[1].im;
        }
        let x = Complex64::new(
            2.0 * normalized_a,
            signed_frequency as f64 * PSI_OMEGA_STEP * self.radius * normalized_a,
        );
        psi_map_from_base(base, x, eta, false)
    }

    /// Regular `a->0` limit for a source behind the half-line origin. The
    /// forward-axis case is genuinely singular for a point-source quadrature
    /// and is intentionally rejected by the GPU operator-domain gate.
    pub fn evaluate_axis_limit(
        &self,
        signed_frequency: i32,
        normalized_negative_height: f64,
    ) -> Result<Complex64, String> {
        if signed_frequency.unsigned_abs() as usize >= PSI_FREQUENCY_COUNT
            || !normalized_negative_height.is_finite()
            || normalized_negative_height <= 0.0
        {
            return Err("invalid regular axis limit".into());
        }
        let w = Complex64::new(
            2.0 * normalized_negative_height,
            signed_frequency as f64 * PSI_OMEGA_STEP * self.radius * normalized_negative_height,
        );
        Ok(scaled_e1_gauss_laguerre(w))
    }
}

fn psi_coefficient_index(frequency: usize, segment: usize, degree: usize) -> usize {
    ((frequency * PSI_SEGMENT_COUNT + segment) * PSI_COEFFICIENT_COUNT + degree) * 4
}

fn evaluate_psi_coefficients(
    coefficients: &[f32],
    frequency: usize,
    segment: usize,
    log_a: f64,
) -> [Complex64; 2] {
    let x0 = PSI_LOG_A_MIN + segment as f64 * PSI_LOG_A_STEP;
    let x1 = x0 + PSI_LOG_A_STEP;
    let t = ((2.0 * log_a - x0 - x1) / (x1 - x0)).clamp(-1.0, 1.0);
    let mut output = [Complex64::ZERO; 2];
    for (component, value) in output.iter_mut().enumerate() {
        let component_offset = 2 * component;
        let mut b1 = Complex64::ZERO;
        let mut b2 = Complex64::ZERO;
        for degree in (1..=PSI_DEGREE).rev() {
            let base = psi_coefficient_index(frequency, segment, degree);
            let coefficient = Complex64::new(
                coefficients[base + component_offset] as f64,
                coefficients[base + component_offset + 1] as f64,
            );
            let b = b1 * (2.0 * t) - b2 + coefficient;
            b2 = b1;
            b1 = b;
        }
        let base = psi_coefficient_index(frequency, segment, 0);
        let coefficient = Complex64::new(
            coefficients[base + component_offset] as f64,
            coefficients[base + component_offset + 1] as f64,
        );
        *value = b1 * t - b2 + coefficient;
    }
    output
}

fn psi_map_from_base(
    base: [Complex64; 2],
    x: Complex64,
    eta: f64,
    exact_correction: bool,
) -> Result<PsiMapSample, String> {
    if !exact_correction && (x.abs() >= 16.0 || -x.re * eta >= 6.0) {
        return Ok(psi_map_asymptotic(x, eta).0);
    }
    let correction = finite_eta_correction_pair(x, eta, exact_correction)?;
    let phase = (x * -eta).exp();
    Ok(PsiMapSample {
        psi: phase * base[0] + correction[0],
        psi_x: phase * (base[1] - base[0] * eta) + correction[1],
    })
}

/// Returns the already-scaled finite incomplete term and its x derivative.
fn finite_eta_correction_pair(
    x: Complex64,
    eta: f64,
    exact: bool,
) -> Result<[Complex64; 2], String> {
    if !exact && eta.abs() <= 0.72 && x.abs() * eta.abs() <= 4.0 {
        let mut c_minus_two = Complex64::ZERO;
        let mut c_minus_one = Complex64::ZERO;
        let mut c = Complex64::ONE;
        let mut dc_minus_two = Complex64::ZERO;
        let mut dc_minus_one = Complex64::ZERO;
        let mut dc = Complex64::ZERO;
        let mut eta_power = eta;
        let mut j = Complex64::ZERO;
        let mut j_x = Complex64::ZERO;
        for order in 0..=36_usize {
            let denominator = (order + 1) as f64;
            j += c * (eta_power / denominator);
            j_x += dc * (eta_power / denominator);
            let next = (x * c + x * c_minus_two - c_minus_one * order as f64) * denominator.recip();
            let next_derivative = (c + x * dc + c_minus_two + x * dc_minus_two
                - dc_minus_one * order as f64)
                * denominator.recip();
            c_minus_two = c_minus_one;
            c_minus_one = c;
            c = next;
            dc_minus_two = dc_minus_one;
            dc_minus_one = dc;
            dc = next_derivative;
            eta_power *= eta;
        }
        let phase = (x * -eta).exp();
        return Ok([phase * j, phase * (j_x - j * eta)]);
    }

    let integrand = |v: f64| {
        let phase = (x * v).exp();
        let base = phase * (1.0 + v * v).sqrt().recip();
        [base, base * v]
    };
    let pair = if exact {
        adaptive_gl8_pair(&integrand, 0.0, eta, 2.0e-11, 0)
            .map_err(|_| "finite-eta reference quadrature did not converge".to_owned())?
    } else {
        let segment_count = ((x.abs() * eta.abs() / 2.0).ceil() as usize).clamp(1, 12);
        let mut sum = [Complex64::ZERO; 2];
        for segment in 0..segment_count {
            let start = eta * segment as f64 / segment_count as f64;
            let end = eta * (segment + 1) as f64 / segment_count as f64;
            let value = gl8_pair(&integrand, start, end);
            sum[0] += value[0];
            sum[1] += value[1];
        }
        sum
    };
    let phase = (x * -eta).exp();
    Ok([phase * pair[0], phase * (pair[1] - pair[0] * eta)])
}

fn psi_map_asymptotic(x: Complex64, eta: f64) -> (PsiMapSample, f64) {
    let polynomial = [1.0 + eta * eta, -2.0 * eta, 1.0];
    let inverse_x = x.reciprocal().unwrap_or(Complex64::ZERO);
    let mut coefficients = [0.0_f64; 33];
    coefficients[0] = polynomial[0].sqrt().recip();
    let exponent = -0.5_f64;
    let mut factorial = 1.0_f64;
    let mut inverse_power = inverse_x;
    let mut psi = Complex64::ZERO;
    let mut psi_x = Complex64::ZERO;
    let mut previous_term = f64::INFINITY;
    let mut remainder = f64::INFINITY;
    for order in 0..coefficients.len() {
        if order > 0 {
            let mut numerator = 0.0;
            for degree in 1..=2.min(order) {
                numerator += ((exponent + 1.0) * degree as f64 - order as f64)
                    * polynomial[degree]
                    * coefficients[order - degree];
            }
            coefficients[order] = numerator / (order as f64 * polynomial[0]);
            factorial *= order as f64;
        }
        let term = inverse_power * (factorial * coefficients[order]);
        let magnitude = term.abs();
        if magnitude > 0.0 {
            if magnitude >= previous_term {
                break;
            }
            previous_term = magnitude;
        }
        psi += term;
        psi_x += -(term * inverse_x) * (order + 1) as f64;
        remainder = magnitude / psi.abs().max(1.0e-12);
        inverse_power = inverse_power * inverse_x;
    }
    (PsiMapSample { psi, psi_x }, remainder)
}

fn scaled_e1_gauss_laguerre(w: Complex64) -> Complex64 {
    if w.abs() < 4.0 {
        let logarithm = Complex64::new(w.abs().ln(), w.im.atan2(w.re));
        let mut e1 = -logarithm - Complex64::new(0.577_215_664_901_532_9, 0.0);
        let mut power = Complex64::ONE;
        let mut factorial = 1.0_f64;
        for order in 1..=36_usize {
            power = power * -w;
            factorial *= order as f64;
            e1 = e1 - power * (1.0 / (order as f64 * factorial));
        }
        return w.exp() * e1;
    }
    GL16_NODES
        .into_iter()
        .zip(GL16_WEIGHTS)
        .fold(Complex64::ZERO, |sum, (node, weight)| {
            sum + (w + Complex64::new(node, 0.0))
                .reciprocal()
                .unwrap_or(Complex64::ZERO)
                * weight
        })
}

#[cfg(feature = "regenerate-operators")]
fn scaled_e1_reference(w: Complex64) -> Result<Complex64, String> {
    let mapped = |u: f64| {
        let denominator = 1.0 - u;
        let t = u / denominator;
        let value = Complex64::new(-t, 0.0).exp() / (w + Complex64::new(t, 0.0))
            * denominator.recip().powi(2);
        [value, Complex64::ZERO]
    };
    adaptive_gl8_pair(&mapped, 0.0, 1.0, 2.0e-11, 0)
        .map(|pair| pair[0])
        .map_err(|_| "axis-limit reference quadrature did not converge".into())
}

#[cfg(feature = "regenerate-operators")]
fn struve_neumann_pair(x: Complex64) -> Result<([Complex64; 2], f64), String> {
    if !x.is_finite() || x.re <= 0.0 {
        return Err("complex Psi table requires Re(x)>0".into());
    }
    if x.abs() >= 24.0 {
        return Ok(struve_neumann_asymptotic(x));
    }
    integrate_struve_neumann_pair(x, 4.0e-9)
        .map(|pair| (pair, 0.0))
        .map_err(|_| "complex Struve--Neumann quadrature did not converge".into())
}

#[cfg(feature = "regenerate-operators")]
fn struve_neumann_asymptotic(x: Complex64) -> ([Complex64; 2], f64) {
    let inverse = x.reciprocal().unwrap_or(Complex64::ZERO);
    let inverse_squared = inverse * inverse;
    let mut term = inverse;
    let mut sum = term;
    let mut derivative = -(term * inverse);
    let mut best = (sum, derivative, f64::INFINITY);
    for order in 0..32_usize {
        let odd = (2 * order + 1) as f64;
        let next = term * inverse_squared * (-(odd * odd));
        if next.abs() >= term.abs() {
            break;
        }
        term = next;
        sum += term;
        derivative += -(term * inverse) * (2 * order + 3) as f64;
        best = (sum, derivative, term.abs());
    }
    let scale = best.0.abs().max(1.0e-12);
    ([best.0, best.1], best.2 / scale)
}
