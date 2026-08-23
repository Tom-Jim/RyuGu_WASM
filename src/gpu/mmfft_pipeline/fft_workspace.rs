struct Fft3dWorkspace {
    n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    line: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl Fft3dWorkspace {
    fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        Self {
            n,
            forward,
            inverse,
            line: vec![Complex64::default(); n],
            scratch: vec![Complex64::default(); scratch_len],
        }
    }

    fn transform(&mut self, values: &mut [Complex64], inverse: bool) {
        debug_assert_eq!(values.len(), self.n.pow(3));
        let transform = Arc::clone(if inverse { &self.inverse } else { &self.forward });
        for line in values.chunks_exact_mut(self.n) {
            process_fft_line_with_scratch(line, transform.as_ref(), inverse, &mut self.scratch);
        }
        for z in 0..self.n {
            for x in 0..self.n {
                for y in 0..self.n {
                    self.line[y] = values[grid_index(x, y, z, self.n)];
                }
                process_fft_line_with_scratch(
                    &mut self.line,
                    transform.as_ref(),
                    inverse,
                    &mut self.scratch,
                );
                for y in 0..self.n {
                    values[grid_index(x, y, z, self.n)] = self.line[y];
                }
            }
        }
        for y in 0..self.n {
            for x in 0..self.n {
                for z in 0..self.n {
                    self.line[z] = values[grid_index(x, y, z, self.n)];
                }
                process_fft_line_with_scratch(
                    &mut self.line,
                    transform.as_ref(),
                    inverse,
                    &mut self.scratch,
                );
                for z in 0..self.n {
                    values[grid_index(x, y, z, self.n)] = self.line[z];
                }
            }
        }
    }
}

struct MmfftLevelWorkspace {
    n: usize,
    p: usize,
    half_extent: f64,
    spacing: f64,
    fft: Fft3dWorkspace,
    kernel_spectrum: Vec<Complex64>,
    density: Vec<Complex64>,
    field: Vec<[f32; 4]>,
}

impl MmfftLevelWorkspace {
    fn new(n: usize, half_extent: f64) -> Self {
        let p = 2 * n;
        let spacing = 2.0 * half_extent / n as f64;
        let mut fft = Fft3dWorkspace::new(p);
        let mut kernel_spectrum = vec![Complex64::default(); p.pow(3)];
        for z in 0..p {
            for y in 0..p {
                for x in 0..p {
                    let signed = |index: usize| {
                        if index < n {
                            index as isize
                        } else {
                            index as isize - p as isize
                        }
                    };
                    let displacement =
                        DVec3::new(signed(x) as f64, signed(y) as f64, signed(z) as f64)
                            * spacing;
                    kernel_spectrum[grid_index(x, y, z, p)].re =
                        displacement.length().max(0.5 * spacing).recip();
                }
            }
        }
        fft.transform(&mut kernel_spectrum, false);
        Self {
            n,
            p,
            half_extent,
            spacing,
            fft,
            kernel_spectrum,
            density: vec![Complex64::default(); p.pow(3)],
            field: vec![[0.0; 4]; n.pow(3)],
        }
    }

    fn build(&mut self, records: &[(DVec3, f64)]) -> &[[f32; 4]] {
        self.density.fill(Complex64::default());
        deposit_density(
            &mut self.density,
            records,
            self.half_extent,
            self.spacing,
            self.n,
            self.p,
        );
        self.fft.transform(&mut self.density, false);
        for (value, kernel) in self.density.iter_mut().zip(&self.kernel_spectrum) {
            *value *= *kernel;
        }
        self.fft.transform(&mut self.density, true);
        for z in 0..self.n {
            for y in 0..self.n {
                for x in 0..self.n {
                    self.field[grid_index(x, y, z, self.n)][3] = (G as f64
                        * self.density[grid_index(x, y, z, self.p)].re)
                        as f32;
                }
            }
        }
        &self.field
    }
}

fn deposit_density(
    density: &mut [Complex64],
    records: &[(DVec3, f64)],
    half_extent: f64,
    spacing: f64,
    n: usize,
    p: usize,
) {
    for &(position, mass) in records {
        let grid = (position + DVec3::splat(half_extent)) / spacing - DVec3::splat(0.5);
        let base = grid.floor();
        let fraction = grid - base;
        let weights = [
            [1.0 - fraction.x, fraction.x],
            [1.0 - fraction.y, fraction.y],
            [1.0 - fraction.z, fraction.z],
        ];
        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    let cell = [
                        base.x as isize + dx,
                        base.y as isize + dy,
                        base.z as isize + dz,
                    ];
                    if cell.iter().any(|value| *value < 0 || *value >= n as isize) {
                        continue;
                    }
                    density[grid_index(cell[0] as usize, cell[1] as usize, cell[2] as usize, p)]
                        .re += mass * weights[0][dx as usize] * weights[1][dy as usize]
                        * weights[2][dz as usize];
                }
            }
        }
    }
}

fn process_fft_line_with_scratch(
    values: &mut [Complex64],
    transform: &dyn Fft<f64>,
    inverse: bool,
    scratch: &mut [Complex64],
) {
    transform.process_with_scratch(values, scratch);
    if inverse {
        let scale = (values.len() as f64).recip();
        for value in values {
            *value *= scale;
        }
    }
}

fn grid_index(x: usize, y: usize, z: usize, side: usize) -> usize {
    (z * side + y) * side + x
}

#[cfg(test)]
fn fft_1d(values: &mut [Complex64], inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let transform = if inverse {
        planner.plan_fft_inverse(values.len())
    } else {
        planner.plan_fft_forward(values.len())
    };
    let mut scratch = vec![Complex64::default(); transform.get_inplace_scratch_len()];
    process_fft_line_with_scratch(values, transform.as_ref(), inverse, &mut scratch);
}

#[cfg(test)]
fn build_level(records: &[(DVec3, f64)], half_extent: f64, n: usize) -> Vec<[f32; 4]> {
    MmfftLevelWorkspace::new(n, half_extent)
        .build(records)
        .to_vec()
}
