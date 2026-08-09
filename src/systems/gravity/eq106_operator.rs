//! Shared operator data for the Fourier-Chebyshev side of Eq. (106).
//!
//! The table approximates the universal toroidal harmonics
//! `Q_{m-1/2}(chi)` on `x = log(chi - 1)`.  The CPU builds it with f64
//! adaptive Gauss-Legendre quadrature and certifies interpolation at the
//! segment midpoints.  The render-world WGSL evaluator consumes the exact
//! serialized f32 coefficient buffer with a fixed Clenshaw loop.
//!
//! This is a truncated discrete operator: mode truncation, interval mapping,
//! coefficient quantization and GPU f32 arithmetic are reported explicitly.

use crate::components::GravityRuntimeError;
use bevy::prelude::*;
use std::f64::consts::PI;

pub const TOROIDAL_MAX_MODE: usize = 16;
pub const TOROIDAL_MODE_COUNT: usize = TOROIDAL_MAX_MODE + 1;
pub const TOROIDAL_SEGMENT_COUNT: usize = 12;
pub const TOROIDAL_DEGREE: usize = 12;
pub const TOROIDAL_COEFFICIENT_COUNT: usize = TOROIDAL_DEGREE + 1;
pub const TOROIDAL_X_MIN: f64 = -10.0;
pub const TOROIDAL_X_MAX: f64 = 8.0;
pub const TOROIDAL_SEGMENT_STEP: f64 =
    (TOROIDAL_X_MAX - TOROIDAL_X_MIN) / TOROIDAL_SEGMENT_COUNT as f64;

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
const MAX_ADAPTIVE_DEPTH: u8 = 20;
const QUADRATURE_TOLERANCE: f64 = 2.0e-11;

#[derive(Clone, Debug, PartialEq)]
pub struct ToroidalOperatorTensor {
    pub coefficients: Vec<f32>,
    pub max_midpoint_error: f64,
}

#[derive(Resource, Clone, Debug)]
pub struct Eq106OperatorTensorResource {
    pub tensor: ToroidalOperatorTensor,
}

/// Builds the universal operator once after the density source exists. A
/// failed certificate is surfaced through the existing runtime error overlay;
/// no alternate physics method is selected implicitly.
pub fn build_eq106_operator_tensor_system(
    mut commands: Commands,
    existing: Option<Res<Eq106OperatorTensorResource>>,
    source_data: Option<Res<crate::systems::curved_arc::Eq106SourceData>>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if existing.is_some() || source_data.is_none() {
        return;
    }
    match ToroidalOperatorTensor::build() {
        Ok(tensor) if tensor.validate(2.0e-4) => {
            commands.insert_resource(Eq106OperatorTensorResource { tensor });
        }
        Ok(tensor) => runtime_error.raise(format!(
            "Equation (106) operator tensor certification failed (relative midpoint error {:.3e}).",
            tensor.max_midpoint_error
        )),
        Err(error) => runtime_error.raise(format!(
            "Equation (106) operator tensor assembly failed: {error}"
        )),
    }
}

impl ToroidalOperatorTensor {
    pub fn coefficient_count() -> usize {
        TOROIDAL_MODE_COUNT * TOROIDAL_SEGMENT_COUNT * TOROIDAL_COEFFICIENT_COUNT
    }

    pub fn build() -> Result<Self, String> {
        let mut coefficients = vec![0.0_f32; Self::coefficient_count()];
        let nodes = TOROIDAL_DEGREE + 1;
        for segment in 0..TOROIDAL_SEGMENT_COUNT {
            let x0 = TOROIDAL_X_MIN + segment as f64 * TOROIDAL_SEGMENT_STEP;
            let x1 = x0 + TOROIDAL_SEGMENT_STEP;
            let midpoint = 0.5 * (x0 + x1);
            let half_width = 0.5 * (x1 - x0);
            let mut values = vec![[0.0_f64; TOROIDAL_MODE_COUNT]; TOROIDAL_DEGREE + 1];
            for (node, value) in values.iter_mut().enumerate().take(nodes) {
                let theta = PI * (node as f64 + 0.5) / nodes as f64;
                let t = theta.cos();
                let x = midpoint + half_width * t;
                let chi = 1.0 + x.exp();
                *value = toroidal_q_modes(chi)?;
            }
            for mode in 0..TOROIDAL_MODE_COUNT {
                for degree in 0..=TOROIDAL_DEGREE {
                    let value = values.iter().enumerate().take(nodes).fold(
                        0.0,
                        |sum, (node, node_values)| {
                            let theta = PI * (node as f64 + 0.5) / nodes as f64;
                            sum + node_values[mode] * (degree as f64 * theta).cos()
                        },
                    );
                    let mut coefficient = 2.0 * value / nodes as f64;
                    if degree == 0 {
                        coefficient *= 0.5;
                    }
                    if !coefficient.is_finite() {
                        return Err("non-finite Chebyshev coefficient".into());
                    }
                    let index = coefficient_index(mode, segment, degree);
                    coefficients[index] = coefficient as f32;
                }
            }
        }

        let mut max_midpoint_error = 0.0_f64;
        for segment in 0..TOROIDAL_SEGMENT_COUNT {
            let x0 = TOROIDAL_X_MIN + segment as f64 * TOROIDAL_SEGMENT_STEP;
            let x1 = x0 + TOROIDAL_SEGMENT_STEP;
            let x = 0.5 * (x0 + x1);
            let chi = 1.0 + x.exp();
            let exact = toroidal_q_modes(chi)?;
            for (mode, exact_value) in exact.iter().enumerate() {
                let estimate = evaluate(&coefficients, mode, x);
                let scale = exact_value.abs().max(1.0e-12);
                max_midpoint_error = max_midpoint_error.max((estimate - exact_value).abs() / scale);
            }
        }
        Ok(Self {
            coefficients,
            max_midpoint_error,
        })
    }

    pub fn validate(&self, tolerance: f64) -> bool {
        self.coefficients.len() == Self::coefficient_count()
            && self.coefficients.iter().all(|value| value.is_finite())
            && self.max_midpoint_error.is_finite()
            && self.max_midpoint_error <= tolerance
    }

    pub fn as_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.coefficients.len() * 4);
        for value in &self.coefficients {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

pub fn coefficient_index(mode: usize, segment: usize, degree: usize) -> usize {
    (mode * TOROIDAL_SEGMENT_COUNT + segment) * TOROIDAL_COEFFICIENT_COUNT + degree
}

fn evaluate(coefficients: &[f32], mode: usize, x: f64) -> f64 {
    let segment = (((x - TOROIDAL_X_MIN) / TOROIDAL_SEGMENT_STEP).floor() as usize)
        .min(TOROIDAL_SEGMENT_COUNT - 1);
    let x0 = TOROIDAL_X_MIN + segment as f64 * TOROIDAL_SEGMENT_STEP;
    let x1 = x0 + TOROIDAL_SEGMENT_STEP;
    let t = ((2.0 * x - x0 - x1) / (x1 - x0)).clamp(-1.0, 1.0);
    let base = coefficient_index(mode, segment, 0);
    let mut b_k1 = 0.0;
    let mut b_k2 = 0.0;
    for degree in (1..=TOROIDAL_DEGREE).rev() {
        let b_k = 2.0 * t * b_k1 - b_k2 + coefficients[base + degree] as f64;
        b_k2 = b_k1;
        b_k1 = b_k;
    }
    t * b_k1 - b_k2 + coefficients[base] as f64
}

fn toroidal_q_modes(chi: f64) -> Result<[f64; TOROIDAL_MODE_COUNT], String> {
    if !chi.is_finite() || chi <= 1.0 {
        return Err("toroidal harmonic requires chi > 1".into());
    }
    let mut result = [0.0_f64; TOROIDAL_MODE_COUNT];
    let integral = integrate_modes(0.0, PI, chi, 0)?;
    // The quadrature integrand already contains
    // `1 / sqrt(2 * (chi - cos(theta)))`, so its cosine integral is
    // exactly the Q_{m-1/2} coefficient used by Eq. (79).
    result.copy_from_slice(&integral);
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
        .ok_or_else(|| "non-finite toroidal harmonic".into())
}

fn integrate_modes(
    lower: f64,
    upper: f64,
    chi: f64,
    depth: u8,
) -> Result<[f64; TOROIDAL_MODE_COUNT], String> {
    let whole = gauss_legendre_modes(lower, upper, chi);
    let midpoint = 0.5 * (lower + upper);
    let left = gauss_legendre_modes(lower, midpoint, chi);
    let right = gauss_legendre_modes(midpoint, upper, chi);
    let mut split = [0.0_f64; TOROIDAL_MODE_COUNT];
    let mut error = 0.0_f64;
    for mode in 0..TOROIDAL_MODE_COUNT {
        split[mode] = left[mode] + right[mode];
        error = error.max((split[mode] - whole[mode]).abs());
    }
    if error <= QUADRATURE_TOLERANCE || depth >= MAX_ADAPTIVE_DEPTH {
        return Ok(split);
    }
    let left = integrate_modes(lower, midpoint, chi, depth + 1)?;
    let right = integrate_modes(midpoint, upper, chi, depth + 1)?;
    let mut result = [0.0_f64; TOROIDAL_MODE_COUNT];
    for mode in 0..TOROIDAL_MODE_COUNT {
        result[mode] = left[mode] + right[mode];
    }
    Ok(result)
}

fn gauss_legendre_modes(lower: f64, upper: f64, chi: f64) -> [f64; TOROIDAL_MODE_COUNT] {
    let midpoint = 0.5 * (lower + upper);
    let half_width = 0.5 * (upper - lower);
    let mut result = [0.0_f64; TOROIDAL_MODE_COUNT];
    for (node, weight) in GL8_NODES.into_iter().zip(GL8_WEIGHTS) {
        let theta = midpoint + half_width * node;
        let base = (2.0 * (chi - theta.cos()))
            .max(f64::MIN_POSITIVE)
            .sqrt()
            .recip();
        let angle = theta;
        let mut cosine = 1.0;
        let mut sine = 0.0;
        let (sin_theta, cos_theta) = angle.sin_cos();
        for (mode, value) in result.iter_mut().enumerate() {
            if mode > 0 {
                let next_cosine = cosine * cos_theta - sine * sin_theta;
                sine = sine * cos_theta + cosine * sin_theta;
                cosine = next_cosine;
            }
            *value += weight * base * cosine * half_width;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_is_finite_and_midpoint_certified() {
        let tensor = ToroidalOperatorTensor::build().expect("tensor build");
        assert!(
            tensor.validate(2.0e-4),
            "error={}",
            tensor.max_midpoint_error
        );
    }

    #[test]
    fn truncated_fourier_operator_reconstructs_kernel_away_from_near_field() {
        let tensor = ToroidalOperatorTensor::build().expect("tensor build");
        let chi = 2.0_f64;
        let x = (chi - 1.0).ln();
        let mut sum = evaluate(&tensor.coefficients, 0, x);
        let angle = 0.73_f64;
        let cos_angle = angle.cos();
        let mut cosine_previous = 1.0;
        let mut cosine = cos_angle;
        for mode in 1..TOROIDAL_MODE_COUNT {
            sum += 2.0 * evaluate(&tensor.coefficients, mode, x) * cosine;
            let next = 2.0 * cos_angle * cosine - cosine_previous;
            cosine_previous = cosine;
            cosine = next;
        }
        let reconstructed = sum / PI;
        let exact = (2.0 * (chi - angle.cos())).sqrt().recip();
        assert!(
            (reconstructed - exact).abs() < 2.0e-4,
            "reconstructed={reconstructed:.12e}, exact={exact:.12e}, error={:.3e}",
            (reconstructed - exact).abs()
        );
    }
}
