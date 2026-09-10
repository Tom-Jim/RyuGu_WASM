fn build_surface_sources(
    density_mode: DensityMode,
    topology: &AsteroidTopologyGpuData,
    aggregated: Option<&AggregatedGravitySource>,
    scale: f32,
) -> Vec<PointMass> {
    if let Some(source) = aggregated {
        let selected = match density_mode {
            DensityMode::Variable => &source.sources,
            DensityMode::Constant => &source.constant_sources,
        };
        if !selected.is_empty() {
            return selected
                .iter()
                .filter_map(|source| {
                    (source.position.is_finite() && source.mass.is_finite() && source.mass > 0.0)
                        .then_some(PointMass {
                            position: source.position,
                            mass: source.mass,
                        })
                })
                .collect();
        }
    }
    build_constant_sources(topology, scale)
}

fn build_constant_sources(topology: &AsteroidTopologyGpuData, scale: f32) -> Vec<PointMass> {
    let faces = topology.triangles.as_chunks::<3>().0;
    if faces.is_empty() {
        return Vec::new();
    }
    let stride = faces.len().div_ceil(CONSTANT_SOURCE_LIMIT).max(1);
    let mut raw = Vec::with_capacity(faces.len().div_ceil(stride));
    let mut selected_volume = 0.0_f64;
    for (face_index, face) in faces.iter().enumerate() {
        if !face_index.is_multiple_of(stride) {
            continue;
        }
        let Some(p0) = topology.positions.get(face[0] as usize).copied() else {
            continue;
        };
        let Some(p1) = topology.positions.get(face[1] as usize).copied() else {
            continue;
        };
        let Some(p2) = topology.positions.get(face[2] as usize).copied() else {
            continue;
        };
        let p0 = p0 * scale;
        let p1 = p1 * scale;
        let p2 = p2 * scale;
        let volume = (p0.dot(p1.cross(p2)).abs() / 6.0) as f64;
        let position = ((p0 + p1 + p2) / 4.0).as_dvec3();
        if volume.is_finite() && volume > 0.0 && position.is_finite() {
            selected_volume += volume;
            raw.push((position, volume));
        }
    }
    if selected_volume <= f64::EPSILON {
        return Vec::new();
    }
    raw.into_iter()
        .map(|(position, volume)| PointMass {
            position,
            mass: RYUGU_MASS as f64 * volume / selected_volume,
        })
        .collect()
}

fn build_evaluator(
    method: ActiveGravityMethod,
    sources: Vec<PointMass>,
    topology: &AsteroidTopologyGpuData,
    scale: f32,
    _density_mode: DensityMode,
) -> SurfaceEvaluator {
    match method {
        ActiveGravityMethod::MmfftCompressed
        | ActiveGravityMethod::Fmm
        | ActiveGravityMethod::HomogeneousWerner
        | ActiveGravityMethod::RadialAnalytic => SurfaceEvaluator::Cpp(method),
        /* Legacy evaluator dispatch retained for reference.
        ActiveGravityMethod::MmfftCompressed => build_fft_evaluator(sources, FFT_GRID_SIZE),
        ActiveGravityMethod::Fmm => SurfaceEvaluator::Fmm(FmmSurfaceEvaluator {
            tree: CpuFmmTree::new(sources),
        }),
        ActiveGravityMethod::HomogeneousWerner if density_mode == DensityMode::Constant => {
            build_werner_evaluator(topology, scale)
                .map(SurfaceEvaluator::Werner)
                .unwrap_or_else(|| SurfaceEvaluator::Point(PointMassEvaluator { sources }))
        }
        ActiveGravityMethod::HomogeneousWerner => {
            SurfaceEvaluator::Point(PointMassEvaluator { sources })
        }
        ActiveGravityMethod::RadialAnalytic => {
            let radial_sources = build_radial_surface_sources(topology, scale)
                .filter(|sources| !sources.is_empty())
                .unwrap_or(sources);
            SurfaceEvaluator::Point(PointMassEvaluator {
                sources: radial_sources,
            })
        }
        */
        // Surface fields use the Eq.106 inverse-pole spatial operator. Eq.184
        // remains exclusively a trajectory transform, never a coarse FFT alias.
        ActiveGravityMethod::FrequencyDomain => {
            let radius = topology
                .positions
                .iter()
                .map(|p| f64::from(p.length() * scale))
                .fold(1.0_f64, f64::max);
            let modes = (0..EQ184_QUADRATURE_COUNT)
                .filter_map(|index| {
                    let (k, weight) = eq184_quadrature_node(index, radius)?;
                    let density = sources
                        .iter()
                        .fold(Complex64::new(0.0, 0.0), |sum, source| {
                            sum + Complex64::from_polar(source.mass, -k.dot(source.position))
                        });
                    let coefficient = f64::from(G) * weight
                        / (2.0 * std::f64::consts::PI.powi(2) * k.length_squared());
                    Some((k, density * coefficient))
                })
                .collect();
            SurfaceEvaluator::Equation106(modes)
        }
    }
}

fn evaluate_patch(evaluator: &SurfaceEvaluator, patch: SurfaceFieldPatch) -> SurfaceFieldSample {
    let value = evaluator.field_at(patch.body_position);
    let jacobian = if matches!(evaluator, SurfaceEvaluator::Equation106(_)) {
        value.jacobian
    } else {
        finite_difference_jacobian(evaluator, patch.body_position)
    };
    let effective = value.gravity + centrifugal_acceleration(patch.body_position);
    let outward_force = (-effective).normalize_or_zero();
    let alignment = outward_force.dot(patch.normal).clamp(-1.0, 1.0);
    SurfaceFieldSample {
        position: patch.body_position,
        normal: patch.normal,
        gravity: value.gravity,
        effective_gravity: effective,
        gravity_magnitude: value.gravity.length(),
        effective_gravity_magnitude: effective.length(),
        gradient_magnitude: frobenius_norm(jacobian),
        slope_degrees: alignment.acos().to_degrees(),
    }
}

fn centrifugal_acceleration(position: Vec3) -> Vec3 {
    let omega =
        RYUGU_SPIN_AXIS.normalize_or_zero() * (std::f32::consts::TAU / RYUGU_ROTATION_PERIOD_SECS);
    -omega.cross(omega.cross(position))
}

fn finite_difference_jacobian(evaluator: &SurfaceEvaluator, position: Vec3) -> DMat3 {
    let h = evaluator.derivative_step();
    let mut columns = [DVec3::ZERO; 3];
    for (axis, column) in columns.iter_mut().enumerate() {
        let direction = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };
        let plus = evaluator
            .field_at(position + direction * h)
            .gravity
            .as_dvec3();
        let minus = evaluator
            .field_at(position - direction * h)
            .gravity
            .as_dvec3();
        *column = (plus - minus) / (2.0 * h as f64);
    }
    DMat3::from_cols(columns[0], columns[1], columns[2])
}

impl SurfaceEvaluator {
    fn field_at(&self, position: Vec3) -> FieldValue {
        match self {
            Self::Cpp(method) => match crate::cpp_backend::evaluate(*method, position) {
                Ok((gravity, potential)) => FieldValue {
                    gravity,
                    potential,
                    jacobian: DMat3::ZERO,
                },
                Err(_) => FieldValue {
                    gravity: Vec3::splat(f32::NAN),
                    potential: f32::NAN,
                    jacobian: DMat3::ZERO,
                },
            },
            Self::Equation106(modes) => {
                let mut field = FieldValue::default();
                let mut gravity = DVec3::ZERO;
                let mut potential = 0.0;
                for (k, density) in modes {
                    let value = *density * Complex64::from_polar(1.0, k.dot(position.as_dvec3()));
                    gravity -= value.im * *k;
                    potential += value.re;
                    field.jacobian -= DMat3::from_cols(*k * k.x, *k * k.y, *k * k.z) * value.re;
                }
                field.gravity = gravity.as_vec3();
                field.potential = potential as f32;
                field
            }
        }
    }

    fn derivative_step(&self) -> f32 {
        match self {
            Self::Cpp(ActiveGravityMethod::MmfftCompressed) => 64.0,
            Self::Cpp(_) => 0.5,
            Self::Equation106(_) => 0.5,
        }
    }
}

fn normalize_range(value: f32, range: (f32, f32)) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let span = range.1 - range.0;
    if !span.is_finite() || span.abs() <= f32::EPSILON {
        0.5
    } else {
        ((value - range.0) / span).clamp(0.0, 1.0)
    }
}

fn finite_range(range: (f32, f32)) -> (f32, f32) {
    if range.0.is_finite() && range.1.is_finite() {
        range
    } else {
        (0.0, 0.0)
    }
}

fn frobenius_norm(matrix: DMat3) -> f32 {
    (matrix.x_axis.length_squared()
        + matrix.y_axis.length_squared()
        + matrix.z_axis.length_squared())
    .sqrt() as f32
}

fn scientific_color(t: f32, metric: SurfaceFieldMetric) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let (low, middle, high) = match metric {
        SurfaceFieldMetric::Gravity => (
            Vec3::new(0.05, 0.12, 0.62),
            Vec3::new(0.04, 0.78, 0.92),
            Vec3::new(1.0, 0.82, 0.10),
        ),
        SurfaceFieldMetric::Gradient => (
            Vec3::new(0.15, 0.05, 0.52),
            Vec3::new(0.64, 0.22, 0.84),
            Vec3::new(1.0, 0.62, 0.08),
        ),
        SurfaceFieldMetric::Slope | SurfaceFieldMetric::Error => (
            Vec3::new(0.02, 0.26, 0.55),
            Vec3::new(0.05, 0.82, 0.72),
            Vec3::new(1.0, 0.45, 0.08),
        ),
    };
    let rgb = if t < 0.5 {
        low.lerp(middle, t * 2.0)
    } else {
        middle.lerp(high, (t - 0.5) * 2.0)
    };
    [rgb.x, rgb.y, rgb.z, 0.92]
}

fn diverging_error_color(value: f32) -> [f32; 4] {
    let value = value.clamp(-1.0, 1.0);
    let neutral = Vec3::new(0.96, 0.96, 0.92);
    let rgb = if value < 0.0 {
        Vec3::new(0.08, 0.28, 0.92).lerp(neutral, value + 1.0)
    } else {
        neutral.lerp(Vec3::new(0.92, 0.12, 0.08), value)
    };
    [rgb.x, rgb.y, rgb.z, 0.94]
}
