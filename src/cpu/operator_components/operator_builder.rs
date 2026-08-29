#[cfg(feature = "regenerate-operators")]
fn integrate_struve_neumann_pair(
    x: Complex64,
    tolerance: f64,
) -> Result<[Complex64; 2], Eq106Error> {
    let mapped = |u: f64| {
        let denominator = 1.0 - u;
        let t = u / denominator;
        let phase = Complex64::new(-x.re * t, -x.im * t).exp();
        let jacobian = denominator.recip().powi(2);
        let base = phase * (jacobian / (1.0 + t * t).sqrt());
        [base, base * -t]
    };
    adaptive_gl8_pair(&mapped, 0.0, 1.0, tolerance, 0)
}

fn adaptive_gl8_pair(
    integrand: &impl Fn(f64) -> [Complex64; 2],
    start: f64,
    end: f64,
    tolerance: f64,
    depth: u8,
) -> Result<[Complex64; 2], Eq106Error> {
    let coarse = gl8_pair(integrand, start, end);
    let middle = 0.5 * (start + end);
    let left = gl8_pair(integrand, start, middle);
    let right = gl8_pair(integrand, middle, end);
    let refined = [left[0] + right[0], left[1] + right[1]];
    let error = (refined[0] - coarse[0])
        .abs()
        .max((refined[1] - coarse[1]).abs());
    let scale = 1.0 + refined[0].abs().max(refined[1].abs());
    if error <= tolerance * scale {
        return Ok(refined);
    }
    if depth >= 22 {
        return Err(Eq106Error::QuadratureFailed);
    }
    let left = adaptive_gl8_pair(integrand, start, middle, tolerance * 0.5, depth + 1)?;
    let right = adaptive_gl8_pair(integrand, middle, end, tolerance * 0.5, depth + 1)?;
    Ok([left[0] + right[0], left[1] + right[1]])
}

fn gl8_pair(integrand: &impl Fn(f64) -> [Complex64; 2], start: f64, end: f64) -> [Complex64; 2] {
    let midpoint = 0.5 * (start + end);
    let half_width = 0.5 * (end - start);
    let mut result = [Complex64::ZERO; 2];
    for (node, weight) in GL8_NODES.into_iter().zip(GL8_WEIGHTS) {
        let value = integrand(midpoint + half_width * node);
        result[0] += value[0] * (weight * half_width);
        result[1] += value[1] * (weight * half_width);
    }
    result
}

impl ToroidalOperatorTensor {
    pub fn coefficient_count() -> usize {
        TOROIDAL_MODE_COUNT * TOROIDAL_SEGMENT_COUNT * TOROIDAL_COEFFICIENT_COUNT
    }

    #[cfg(feature = "regenerate-operators")]
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

    pub fn embedded() -> Result<Self, String> {
        const BYTES: &[u8] = include_bytes!("../../../assets/operators/eq106_toroidal_table.bin");
        if BYTES.len() != 16 + Self::coefficient_count() * 4 || &BYTES[..8] != b"EQ106TOR" {
            return Err("embedded toroidal table has an invalid header or length".into());
        }
        let max_midpoint_error = f64::from_le_bytes(BYTES[8..16].try_into().unwrap_or([0; 8]));
        let coefficients = BYTES[16..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let table = Self {
            coefficients,
            max_midpoint_error,
        };
        table
            .validate(2.0e-4)
            .then_some(table)
            .ok_or_else(|| "embedded toroidal table certificate is invalid".into())
    }

    pub fn validate(&self, tolerance: f64) -> bool {
        self.coefficients.len() == Self::coefficient_count()
            && self.coefficients.iter().all(|value| value.is_finite())
            && self.max_midpoint_error.is_finite()
            && self.max_midpoint_error <= tolerance
    }

    pub fn as_le_bytes(&self) -> Vec<u8> {
        bytemuck::cast_slice(&self.coefficients).to_vec()
    }
}

#[cfg(feature = "regenerate-operators")]
pub fn coefficient_index(mode: usize, segment: usize, degree: usize) -> usize {
    (mode * TOROIDAL_SEGMENT_COUNT + segment) * TOROIDAL_COEFFICIENT_COUNT + degree
}

#[cfg(feature = "regenerate-operators")]
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

#[cfg(feature = "regenerate-operators")]
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

#[cfg(feature = "regenerate-operators")]
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

#[cfg(feature = "regenerate-operators")]
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
