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
        // Surface fields use the same discrete Eq.121 IR/UV split as live
        // Worker propagation. Eq.184 remains exclusively a trajectory Laplace
        // observation, never a coarse FFT alias of the force.
        ActiveGravityMethod::FrequencyDomain => {
            let radius = topology
                .positions
                .iter()
                .map(|p| f64::from(p.length() * scale))
                .fold(1.0_f64, f64::max);
            let mass: f64 = sources.iter().map(|source| source.mass).sum();
            let center = if mass > 0.0 {
                sources.iter().fold(DVec3::ZERO, |sum, source| {
                    sum + source.position * source.mass
                }) / mass
            } else {
                DVec3::ZERO
            };
            let modes = (0..EQ184_QUADRATURE_COUNT)
                .filter_map(|index| {
                    let (k, weight) = eq184_quadrature_node(index, radius)?;
                    let density = sources
                        .iter()
                        .fold(Complex64::new(0.0, 0.0), |sum, source| {
                            sum + Complex64::from_polar(source.mass, -k.dot(source.position))
                        })
                        - Complex64::from_polar(mass, -k.dot(center));
                    let coefficient = f64::from(G) * weight
                        / (2.0 * std::f64::consts::PI.powi(2) * k.length_squared());
                    Some((k, density * coefficient))
                })
                .collect();
            SurfaceEvaluator::Equation121 {
                modes,
                center,
                gravitational_parameter: f64::from(G) * mass,
            }
        }
    }
}

/// Eq.121 surface sample (residual Fourier + analytic IR monopole).
fn evaluate_patch_locally(
    modes: &[(DVec3, Complex64)],
    center: DVec3,
    gravitational_parameter: f64,
    patch: SurfaceFieldPatch,
) -> SurfaceFieldSample {
    let (gravity, jacobian) =
        equation121_field(modes, center, gravitational_parameter, patch.body_position);
    surface_sample(patch, gravity, jacobian)
}

fn surface_sample(patch: SurfaceFieldPatch, gravity: Vec3, jacobian: DMat3) -> SurfaceFieldSample {
    let effective = gravity + centrifugal_acceleration(patch.body_position);
    let outward_force = (-effective).normalize_or_zero();
    let alignment = outward_force.dot(patch.normal).clamp(-1.0, 1.0);
    SurfaceFieldSample {
        position: patch.body_position,
        normal: patch.normal,
        gravity,
        effective_gravity: effective,
        gravity_magnitude: gravity.length(),
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

/// Number of Worker evaluations per surface patch: the patch centre plus the
/// `±h` stencil along each axis for the central-difference Jacobian.
const SURFACE_STENCIL: usize = 7;

/// Body-frame targets for one chunk of patches, laid out
/// `[centre, +x, -x, +y, -y, +z, -z]` per patch.
fn surface_stencil_targets(patches: &[SurfaceFieldPatch], h: f32) -> Vec<DVec3> {
    let mut targets = Vec::with_capacity(patches.len() * SURFACE_STENCIL);
    for patch in patches {
        let position = patch.body_position;
        targets.push(position.as_dvec3());
        for direction in [Vec3::X, Vec3::Y, Vec3::Z] {
            targets.push((position + direction * h).as_dvec3());
            targets.push((position - direction * h).as_dvec3());
        }
    }
    targets
}

/// Turns the Worker's `[gx, gy, gz, potential]` records for one stencil chunk
/// into surface samples with central-difference Jacobians.
fn decode_surface_chunk(
    patches: &[SurfaceFieldPatch],
    h: f32,
    values: &[f64],
) -> Result<Vec<SurfaceFieldSample>, String> {
    let records = crate::cpp_backend::decode_field_batch(values, patches.len() * SURFACE_STENCIL)?;
    let gravity = |record: &[f64; 4]| Vec3::new(record[0] as f32, record[1] as f32, record[2] as f32);
    Ok(patches
        .iter()
        .zip(records.as_chunks::<SURFACE_STENCIL>().0)
        .map(|(patch, stencil)| {
            let columns = [0, 1, 2].map(|axis| {
                let plus = gravity(&stencil[1 + 2 * axis]).as_dvec3();
                let minus = gravity(&stencil[2 + 2 * axis]).as_dvec3();
                (plus - minus) / (2.0 * h as f64)
            });
            surface_sample(
                *patch,
                gravity(&stencil[0]),
                DMat3::from_cols(columns[0], columns[1], columns[2]),
            )
        })
        .collect())
}

fn equation121_field(
    modes: &[(DVec3, Complex64)],
    center: DVec3,
    gravitational_parameter: f64,
    position: Vec3,
) -> (Vec3, DMat3) {
    let mut jacobian = DMat3::ZERO;
    let mut gravity = DVec3::ZERO;
    let query = position.as_dvec3();
    for (k, density) in modes {
        let value = *density * Complex64::from_polar(1.0, k.dot(query));
        gravity -= value.im * *k;
        jacobian -= DMat3::from_cols(*k * k.x, *k * k.y, *k * k.z) * value.re;
    }
    let offset = query - center;
    let distance_squared = offset.length_squared();
    if distance_squared > 0.0 && gravitational_parameter > 0.0 {
        let inverse_distance = distance_squared.sqrt().recip();
        let inv3 = inverse_distance.powi(3);
        let inv5 = inverse_distance.powi(5);
        gravity += -gravitational_parameter * offset * inv3;
        // ∂/∂q (−GM r̂/r²) = −GM (I/r³ − 3 r⊗r / r⁵)
        let monopole_jacobian = DMat3::from_cols(
            DVec3::new(
                -gravitational_parameter * (inv3 - 3.0 * offset.x * offset.x * inv5),
                gravitational_parameter * 3.0 * offset.x * offset.y * inv5,
                gravitational_parameter * 3.0 * offset.x * offset.z * inv5,
            ),
            DVec3::new(
                gravitational_parameter * 3.0 * offset.y * offset.x * inv5,
                -gravitational_parameter * (inv3 - 3.0 * offset.y * offset.y * inv5),
                gravitational_parameter * 3.0 * offset.y * offset.z * inv5,
            ),
            DVec3::new(
                gravitational_parameter * 3.0 * offset.z * offset.x * inv5,
                gravitational_parameter * 3.0 * offset.z * offset.y * inv5,
                -gravitational_parameter * (inv3 - 3.0 * offset.z * offset.z * inv5),
            ),
        );
        jacobian += monopole_jacobian;
    }
    (gravity.as_vec3(), jacobian)
}

/// Central-difference step of the pointwise Worker evaluators.
fn derivative_step(method: ActiveGravityMethod) -> f32 {
    match method {
        ActiveGravityMethod::MmfftCompressed => 64.0,
        _ => 0.5,
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
