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

    /// One potential grid per unit-density voxel, in f64 until the final RHS
    /// combination. Deposit all source volumes once; run the existing rustfft
    /// convolution once per column. Storage is 56*n^3, not 56*(2n)^3 complex
    /// spectra and not eight heap records per source.
    #[cfg(test)]
    fn unit_density_potentials(&mut self, records: &[PlanningBasisRecord]) -> Option<Vec<f64>> {
        let nodes = self.n.checked_pow(3)?;
        let mut basis = Vec::new();
        basis.try_reserve_exact(nodes.checked_mul(56)?).ok()?;
        basis.resize(nodes * 56, 0.0_f64);
        for record in records {
            let voxel = record.voxel_index as usize;
            let volume = f64::from(record.position_volume[3]);
            if voxel >= 56 || !volume.is_finite() || volume < 0.0 { return None; }
            let position = DVec3::new(f64::from(record.position_volume[0]),
                f64::from(record.position_volume[1]), f64::from(record.position_volume[2]));
            if !position.is_finite() { return None; }
            visit_deposition_weights(position, self.half_extent, self.spacing, self.n,
                |x, y, z, weight| {
                    basis[voxel * nodes + grid_index(x, y, z, self.n)] += volume * weight;
                });
        }
        for column in basis.chunks_exact_mut(nodes) {
            if column.iter().all(|value| *value == 0.0) { continue; }
            self.density.fill(Complex64::default());
            for z in 0..self.n {
                for y in 0..self.n {
                    for x in 0..self.n {
                        self.density[grid_index(x, y, z, self.p)].re = column[grid_index(x, y, z, self.n)];
                    }
                }
            }
            self.fft.transform(&mut self.density, false);
            for (value, kernel) in self.density.iter_mut().zip(&self.kernel_spectrum) { *value *= *kernel; }
            self.fft.transform(&mut self.density, true);
            for z in 0..self.n {
                for y in 0..self.n {
                    for x in 0..self.n {
                        let potential = f64::from(G) * self.density[grid_index(x, y, z, self.p)].re;
                        if !potential.is_finite() { return None; }
                        column[grid_index(x, y, z, self.n)] = potential;
                    }
                }
            }
        }
        Some(basis)
    }

    #[cfg(test)]
    fn build_from_basis(
        &mut self,
        records: &[PlanningBasisRecord],
        densities: &[f32],
    ) -> Option<&[[f32; 4]]> {
        if densities.len() != 56 {
            return None;
        }
        self.density.fill(Complex64::default());
        for record in records {
            let density = f64::from(*densities.get(record.voxel_index as usize)?);
            let mass = f64::from(record.position_volume[3]) * density;
            if !mass.is_finite() || mass <= 0.0 {
                return None;
            }
            let position = DVec3::new(
                f64::from(record.position_volume[0]),
                f64::from(record.position_volume[1]),
                f64::from(record.position_volume[2]),
            );
            deposit_particle(
                &mut self.density,
                position,
                mass,
                self.half_extent,
                self.spacing,
                self.n,
                self.p,
            );
        }
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
        Some(&self.field)
    }
}

fn deposit_particle(
    density: &mut [Complex64],
    position: DVec3,
    mass: f64,
    half_extent: f64,
    spacing: f64,
    n: usize,
    p: usize,
) {
    visit_deposition_weights(position, half_extent, spacing, n, |x, y, z, weight| {
        density[grid_index(x, y, z, p)].re += mass * weight;
    });
}

fn visit_deposition_weights(
    position: DVec3, half_extent: f64, spacing: f64, n: usize,
    mut visit: impl FnMut(usize, usize, usize, f64),
) {
    let grid = (position + DVec3::splat(half_extent)) / spacing - DVec3::splat(0.5);
    let base = grid.floor();
    let fraction = grid - base;
    let weights = [[1.0 - fraction.x, fraction.x], [1.0 - fraction.y, fraction.y],
        [1.0 - fraction.z, fraction.z]];
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let cell = [base.x as isize + dx, base.y as isize + dy, base.z as isize + dz];
                if cell.iter().any(|value| *value < 0 || *value >= n as isize) { continue; }
                visit(cell[0] as usize, cell[1] as usize, cell[2] as usize,
                    weights[0][dx as usize] * weights[1][dy as usize] * weights[2][dz as usize]);
            }
        }
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
        deposit_particle(density, position, mass, half_extent, spacing, n, p);
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

/// Appends one potential grid as two IEEE-754 binary16 values per `u32`.
/// MMFFT interpolation only consumes this scalar; its field and Jacobian are
/// derivatives of the same tricubic potential. A per-level scale preserves
/// the binary16 mantissa across the different nested-grid ranges.
fn append_compressed_potential_level(bytes: &mut Vec<u8>, field: &[[f32; 4]]) -> f32 {
    let scale = field
        .iter()
        .map(|sample| sample[3].abs())
        .filter(|value| value.is_finite())
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    let mut samples = field.iter();
    while let Some(first) = samples.next() {
        let low = half::f16::from_f32((first[3] / scale).clamp(-1.0, 1.0)).to_bits();
        let high = samples.next().map_or(0, |second| {
            half::f16::from_f32((second[3] / scale).clamp(-1.0, 1.0)).to_bits()
        });
        bytes.extend_from_slice(&(u32::from(low) | (u32::from(high) << 16)).to_le_bytes());
    }
    scale
}
