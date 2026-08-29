/// One-dimensional rational continuation built from actual Taylor
/// coefficients. It never accepts an approximant with a denominator pole on
/// the requested interval and requires an independently supplied residual
/// check before a trajectory planner may certify it.
#[derive(Clone, Debug, PartialEq)]
pub struct PadeApproximant {
    pub numerator: Vec<Complex64>,
    pub denominator: Vec<Complex64>,
}

impl PadeApproximant {
    pub fn evaluate(&self, value: Complex64) -> Result<Complex64, Eq106Error> {
        let numerator = polynomial_value(&self.numerator, value);
        let denominator = polynomial_value(&self.denominator, value);
        denominator
            .reciprocal()
            .map(|inverse| numerator * inverse)
            .ok_or(Eq106Error::PadePoleOnInterval)
    }

    pub fn has_pole_on_unit_interval(&self) -> bool {
        const STEPS: usize = 256;
        let mut previous = polynomial_value(&self.denominator, Complex64::ZERO).abs();
        if previous <= 1.0e-10 {
            return true;
        }
        for index in 1..=STEPS {
            let t = index as f64 / STEPS as f64;
            let current = polynomial_value(&self.denominator, Complex64::new(t, 0.0)).abs();
            if current <= 1.0e-10 || current > previous * 1.0e8 {
                return true;
            }
            previous = current;
        }
        false
    }
}

pub fn pade_from_taylor(
    coefficients: &[Complex64],
    numerator_order: usize,
    denominator_order: usize,
) -> Result<PadeApproximant, Eq106Error> {
    if coefficients.len() < numerator_order + denominator_order + 1 {
        return Err(Eq106Error::PadeNotConverged);
    }
    if denominator_order == 0 {
        return Ok(PadeApproximant {
            numerator: coefficients[..=numerator_order].to_vec(),
            denominator: vec![Complex64::ONE],
        });
    }

    let mut matrix = vec![vec![Complex64::ZERO; denominator_order + 1]; denominator_order];
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        let n = numerator_order + 1 + row;
        for (column, value) in matrix_row.iter_mut().take(denominator_order).enumerate() {
            *value = coefficients[n - column - 1];
        }
        matrix_row[denominator_order] = -coefficients[n];
    }
    let q_tail = solve_complex_linear_system(&mut matrix)?;
    let mut denominator = Vec::with_capacity(denominator_order + 1);
    denominator.push(Complex64::ONE);
    denominator.extend(q_tail);

    let mut numerator = vec![Complex64::ZERO; numerator_order + 1];
    for index in 0..=numerator_order {
        for (q_index, q) in denominator.iter().enumerate().take(index + 1) {
            numerator[index] += coefficients[index - q_index] * *q;
        }
    }
    let approximant = PadeApproximant {
        numerator,
        denominator,
    };
    (!approximant.has_pole_on_unit_interval())
        .then_some(approximant)
        .ok_or(Eq106Error::PadePoleOnInterval)
}

fn integrate_on_half_line(
    integrand: impl Fn(f64) -> Complex64,
    tolerance: f64,
) -> Result<Complex64, Eq106Error> {
    let mapped = |u: f64| {
        let denominator = 1.0 - u;
        let h = u / denominator;
        integrand(h) * denominator.recip().powi(2)
    };
    adaptive_gl8(
        &mapped,
        0.0,
        1.0,
        tolerance.max(DEFAULT_QUADRATURE_TOLERANCE),
        0,
    )
}

fn adaptive_gl8(
    integrand: &impl Fn(f64) -> Complex64,
    start: f64,
    end: f64,
    tolerance: f64,
    depth: u8,
) -> Result<Complex64, Eq106Error> {
    let coarse = gl8(integrand, start, end);
    let middle = 0.5 * (start + end);
    let refined = gl8(integrand, start, middle) + gl8(integrand, middle, end);
    let error = (refined - coarse).abs();
    if error <= tolerance * (1.0 + refined.abs()) {
        return Ok(refined);
    }
    if depth >= MAX_ADAPTIVE_DEPTH {
        return Err(Eq106Error::QuadratureFailed);
    }
    Ok(
        adaptive_gl8(integrand, start, middle, tolerance * 0.5, depth + 1)?
            + adaptive_gl8(integrand, middle, end, tolerance * 0.5, depth + 1)?,
    )
}

fn gl8(integrand: &impl Fn(f64) -> Complex64, start: f64, end: f64) -> Complex64 {
    let midpoint = 0.5 * (start + end);
    let half_width = 0.5 * (end - start);
    GL8_NODES
        .iter()
        .zip(GL8_WEIGHTS)
        .fold(Complex64::ZERO, |sum, (node, weight)| {
            sum + integrand(midpoint + half_width * node) * weight
        })
        * half_width
}

fn polynomial_value(coefficients: &[Complex64], value: Complex64) -> Complex64 {
    coefficients
        .iter()
        .rev()
        .fold(Complex64::ZERO, |sum, coefficient| {
            sum * value + *coefficient
        })
}

fn solve_complex_linear_system(
    matrix: &mut [Vec<Complex64>],
) -> Result<Vec<Complex64>, Eq106Error> {
    let size = matrix.len();
    for pivot_column in 0..size {
        let pivot_row = (pivot_column..size)
            .max_by(|&left, &right| {
                matrix[left][pivot_column]
                    .abs()
                    .total_cmp(&matrix[right][pivot_column].abs())
            })
            .ok_or(Eq106Error::SingularPadeSystem)?;
        if matrix[pivot_row][pivot_column].abs() <= 1.0e-14 {
            return Err(Eq106Error::SingularPadeSystem);
        }
        matrix.swap(pivot_column, pivot_row);
        let inverse = matrix[pivot_column][pivot_column]
            .reciprocal()
            .ok_or(Eq106Error::SingularPadeSystem)?;
        for value in matrix[pivot_column][pivot_column..=size].iter_mut() {
            *value = *value * inverse;
        }
        for row in 0..size {
            if row == pivot_column {
                continue;
            }
            let factor = matrix[row][pivot_column];
            let pivot_slice = matrix[pivot_column][pivot_column..=size].to_vec();
            for (value, pivot_value) in matrix[row][pivot_column..=size].iter_mut().zip(pivot_slice)
            {
                *value = *value - factor * pivot_value;
            }
        }
    }
    Ok((0..size).map(|row| matrix[row][size]).collect())
}
