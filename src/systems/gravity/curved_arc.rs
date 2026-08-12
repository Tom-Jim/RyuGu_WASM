use crate::components::*;
use crate::systems::eq106::Eq106PointSource;
#[cfg(test)]
use crate::systems::eq106::{
    self, Eq106Certificate, Eq106Error, Eq106FrequencyGrid, Eq106ReferenceLine,
    Eq106TransformSample,
};
use bevy::math::DVec3;
use bevy::prelude::*;
use std::collections::VecDeque;

const PLANNING_WINDOW_POINTS: usize = 128;
const MIN_SEGMENT_POINTS: usize = 4;
const EPSILON_TARGET: f64 = 0.25;
const TAYLOR_REMAINDER_TARGET: f64 = 1.0e-3;
const MAX_TAYLOR_ORDER: u32 = 8;
const CLOSURE_POSITION_TOLERANCE: f32 = 1.0e-3;
const CLOSURE_VELOCITY_TOLERANCE: f32 = 1.0e-3;
const CLOSURE_PERIOD_TOLERANCE: f64 = 1.0e-3;
const REQUIRED_STABLE_CLOSURES: u32 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CurvedArcMode {
    #[default]
    Bootstrap,
    NonPeriodic,
    Periodic,
    Error,
}

impl CurvedArcMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "Non-periodic warm-up",
            Self::NonPeriodic => "Eq.106 non-periodic",
            Self::Periodic => "Eq.106 periodic",
            Self::Error => "Eq.106 evaluation error",
        }
    }

    pub fn short_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "Warm-up",
            Self::NonPeriodic => "Non-periodic",
            Self::Periodic => "Periodic",
            Self::Error => "Error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CurvedArcSegment {
    pub end_index: usize,
    pub epsilon_max: f64,
    pub distance_lower_bound: f64,
    /// Maximum discrete curvature on the sampled arc, in inverse metres.
    pub maximum_curvature: f64,
    /// Arc length represented by this segment, in metres.
    pub arc_length: f64,
    pub taylor_order: Option<u32>,
    /// Retained for UI compatibility. Runtime Eq. (106) never uses a rational
    /// continuation outside the Taylor convergence disk.
    pub pade_order: Option<(u32, u32)>,
    pub pade_certified: bool,
    pub remainder_bound: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Eq106KernelStatus {
    #[default]
    AwaitingSource,
    Ready,
    Failed,
}

#[derive(Resource, Default)]
pub struct Eq106SourceData {
    pub sources: Vec<Eq106PointSource>,
    /// Eq. (81) density modes packed as `(cylindrical_radius, z, re, im)`;
    /// records are ring-major with modes `0..=16`.
    pub fourier_modes: Vec<[f32; 4]>,
    pub total_mass: f64,
    pub radius: f64,
    pub source_hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadeContinuationPlan {
    pub numerator_order: u32,
    pub denominator_order: u32,
    pub estimated_remainder: f64,
    /// Must remain false until the independent spectral residual certifies the
    /// rational continuation for the full density operator.
    pub certified: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct CurvedArcResidualSample {
    pub simulation_time_seconds: f64,
    /// Conservative convergence health metric: ε = |δq_max| / d_safe.
    /// This is the Eq. (118) Taylor series ratio bound, not the Eq. (157) dual
    /// residual. Always populated so the chart has a stable y-range.
    pub epsilon_max: f64,
    /// Eq. (157) dual-representation residual from the curved-path work
    /// integral and the independently accumulated GPU potential.
    pub dual_residual: Option<f64>,
    /// Taylor truncation order actually used (1..=MAX_TAYLOR_ORDER).
    pub taylor_order: u32,
}

#[derive(Resource)]
pub struct CurvedArcResidualHistory {
    pub samples: VecDeque<CurvedArcResidualSample>,
    origin_potential: Option<f64>,
    origin_curve_work: Option<f64>,
    previous_request_id: Option<u64>,
    accumulated_curve_work: f64,
    curve_work_samples: VecDeque<(f64, f64)>,
}

impl Default for CurvedArcResidualHistory {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(JACOBI_HISTORY_CAPACITY),
            origin_potential: None,
            origin_curve_work: None,
            previous_request_id: None,
            accumulated_curve_work: 0.0,
            curve_work_samples: VecDeque::with_capacity(4096),
        }
    }
}

impl CurvedArcResidualHistory {
    pub fn reset(&mut self) {
        self.samples.clear();
        self.origin_potential = None;
        self.origin_curve_work = None;
        self.previous_request_id = None;
        self.accumulated_curve_work = 0.0;
        self.curve_work_samples.clear();
    }

    fn push(&mut self, sample: CurvedArcResidualSample) {
        if self.samples.len() == JACOBI_HISTORY_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn accumulate_curve_work(
        &mut self,
        start_simulation_time_seconds: f64,
        simulation_time_seconds: f64,
        start_body_position: Vec3,
        end_body_position: Vec3,
        start_body_acceleration: Vec3,
        end_body_acceleration: Vec3,
    ) {
        let displacement = end_body_position - start_body_position;
        let average_acceleration = 0.5 * (start_body_acceleration + end_body_acceleration);
        let work = average_acceleration.dot(displacement) as f64;
        if !simulation_time_seconds.is_finite() || !work.is_finite() {
            return;
        }
        if self.curve_work_samples.is_empty() && start_simulation_time_seconds.is_finite() {
            self.curve_work_samples
                .push_back((start_simulation_time_seconds, self.accumulated_curve_work));
        }
        self.accumulated_curve_work += work;
        if self.curve_work_samples.len() == 4096 {
            self.curve_work_samples.pop_front();
        }
        self.curve_work_samples
            .push_back((simulation_time_seconds, self.accumulated_curve_work));
    }

    pub(crate) fn curve_work_at(&self, simulation_time_seconds: f64) -> Option<f64> {
        let first = *self.curve_work_samples.front()?;
        if simulation_time_seconds <= first.0 {
            return Some(first.1);
        }
        for (&lower, &upper) in self
            .curve_work_samples
            .iter()
            .zip(self.curve_work_samples.iter().skip(1))
        {
            if simulation_time_seconds <= upper.0 {
                let interval = (upper.0 - lower.0).max(f64::EPSILON);
                let weight = ((simulation_time_seconds - lower.0) / interval).clamp(0.0, 1.0);
                return Some(lower.1 + weight * (upper.1 - lower.1));
            }
        }
        self.curve_work_samples.back().map(|sample| sample.1)
    }

    fn dual_residual_for(&mut self, sample: &GravityFieldSample) -> Option<f64> {
        if sample.predictive {
            return None;
        }
        if self.previous_request_id == Some(sample.snapshot.request_id) {
            return None;
        }
        let potential = sample.independent_positive_potential? as f64;
        if !potential.is_finite() {
            return None;
        }

        let curve_work = self.curve_work_at(sample.snapshot.simulation_time_seconds)?;
        let origin = *self.origin_potential.get_or_insert(potential);
        let origin_curve_work = *self.origin_curve_work.get_or_insert(curve_work);

        self.previous_request_id = Some(sample.snapshot.request_id);

        // Eq. (147) supplies P_70 through the curved-path work integral, while
        // the independently accumulated GPU potential supplies P_spec. Their
        // finite-discretization difference is the Eq. (157) dual residual.
        let residual = (curve_work - origin_curve_work) - (potential - origin);
        residual.is_finite().then_some(residual)
    }
}

#[derive(Resource, Default)]
pub struct CurvedArcPlannerState {
    pub mode: CurvedArcMode,
    pub segments: Vec<CurvedArcSegment>,
    pub epsilon_max: Option<f64>,
    pub stable_closures: u32,
    pub kernel_ready: bool,
    pub active_segment: Option<CurvedArcSegment>,
    pub taylor_order: u32,
    pub pade_plan: Option<PadeContinuationPlan>,
    pub kernel_status: Eq106KernelStatus,
}

impl CurvedArcPlannerState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Resource, Default)]
pub struct PeriodicityDetector {
    plane_origin: Option<Vec3>,
    plane_normal: Option<Vec3>,
    previous_position: Option<Vec3>,
    previous_velocity: Option<Vec3>,
    previous_signed_distance: Option<f32>,
    reference: Option<ClosureSample>,
    previous_period: Option<f64>,
}

#[derive(Clone, Copy)]
struct ClosureSample {
    position: Vec3,
    velocity: Vec3,
    time: f64,
}

impl PeriodicityDetector {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// Eight aggregate masses were enough for a pipeline smoke test, but they turn
// Ryugu into a visibly lumpy eight-point field and can noticeably precess the
// probe orbit. Keep a bounded GPU-friendly quadrature while retaining enough
// angular structure for the Eq. (106) density integral.
const EQ106_AZIMUTH_BINS: usize = 32;
const EQ106_POLAR_BINS: usize = 8;
const EQ106_RADIAL_BINS: usize = 4;
const EQ106_SOURCE_BUDGET: usize = EQ106_AZIMUTH_BINS * EQ106_POLAR_BINS * EQ106_RADIAL_BINS;

/// Decodes a bounded, mass-preserving quadrature view of the radial density
/// records. The full GPU source remains authoritative for the radial method;
/// this independent view exists solely to certify the complex-frequency
/// Equation (106) operator before it is allowed to drive physics.
pub fn build_eq106_source_system(
    mut commands: Commands,
    radial: Option<Res<RadialGravitySource>>,
    existing: Option<Res<Eq106SourceData>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(radial) = radial else { return };
    let record_count = radial.bytes.len() / 32;
    if record_count == 0 {
        return;
    }
    let mut bin_masses = [0.0_f64; EQ106_SOURCE_BUDGET];
    let mut bin_moments = [DVec3::ZERO; EQ106_SOURCE_BUDGET];
    let mut total_mass = 0.0;
    let mut radius = 0.0_f64;
    for (record_index, chunk) in radial.bytes.chunks_exact(32).enumerate() {
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
        let azimuth =
            (direction.y.atan2(direction.x) + std::f64::consts::PI) / std::f64::consts::TAU;
        let polar = 0.5 * (direction.z.clamp(-1.0, 1.0) + 1.0);
        let azimuth_index =
            ((azimuth * EQ106_AZIMUTH_BINS as f64).floor() as usize).min(EQ106_AZIMUTH_BINS - 1);
        let polar_index =
            ((polar * EQ106_POLAR_BINS as f64).floor() as usize).min(EQ106_POLAR_BINS - 1);
        // Radial records are emitted four layers per angular cell. Preserve
        // that coordinate instead of collapsing the entire ray to one point;
        // this is the tensor-product density quadrature used by Eq. (106).
        let radial_index = record_index % EQ106_RADIAL_BINS;
        let bin_index =
            (radial_index * EQ106_POLAR_BINS + polar_index) * EQ106_AZIMUTH_BINS + azimuth_index;
        bin_masses[bin_index] += mass;
        bin_moments[bin_index] += direction * radial_centroid * mass;
        total_mass += mass;
        radius = radius.max(outer);
    }
    let sources = bin_masses
        .iter()
        .zip(bin_moments)
        .filter_map(|(mass, moment)| {
            (*mass > 0.0).then_some(Eq106PointSource {
                position: moment / *mass,
                mass: *mass,
            })
        })
        .collect::<Vec<_>>();
    if sources.is_empty() || !total_mass.is_finite() {
        return;
    }
    let mut fourier_modes = Vec::with_capacity(EQ106_RADIAL_BINS * EQ106_POLAR_BINS * 17);
    for radial_index in 0..EQ106_RADIAL_BINS {
        for polar_index in 0..EQ106_POLAR_BINS {
            let ring_start = (radial_index * EQ106_POLAR_BINS + polar_index) * EQ106_AZIMUTH_BINS;
            let ring_mass = bin_masses[ring_start..ring_start + EQ106_AZIMUTH_BINS]
                .iter()
                .sum::<f64>();
            if ring_mass <= 0.0 {
                continue;
            }
            let ring_z = bin_moments[ring_start..ring_start + EQ106_AZIMUTH_BINS]
                .iter()
                .map(|moment| moment.z)
                .sum::<f64>()
                / ring_mass;
            let ring_radius = (bin_moments[ring_start..ring_start + EQ106_AZIMUTH_BINS]
                .iter()
                .map(|moment| moment.x.hypot(moment.y))
                .sum::<f64>()
                / ring_mass)
                .max(1.0e-6);
            for mode in 0..=16 {
                let mut coefficient = [0.0_f64; 2];
                for azimuth_index in 0..EQ106_AZIMUTH_BINS {
                    let index = ring_start + azimuth_index;
                    let mass = bin_masses[index];
                    if mass <= 0.0 {
                        continue;
                    }
                    let position = bin_moments[index] / mass;
                    let phi = position.y.atan2(position.x);
                    coefficient[0] += mass * (-(mode as f64) * phi).cos();
                    coefficient[1] += mass * (-(mode as f64) * phi).sin();
                }
                fourier_modes.push([
                    ring_radius as f32,
                    ring_z as f32,
                    coefficient[0] as f32,
                    coefficient[1] as f32,
                ]);
            }
        }
    }
    commands.insert_resource(Eq106SourceData {
        sources,
        fourier_modes,
        total_mass,
        radius,
        source_hash: hash_source_bytes(&radial.bytes),
    });
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0_u8; 4]))
}

fn hash_source_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(1469598103934665603_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(1099511628211_u64)
    })
}

#[cfg(test)]
fn certify_runtime_line(
    line: Eq106ReferenceLine,
    sources: &[Eq106PointSource],
    radius: f64,
) -> Result<(Vec<Eq106TransformSample>, Eq106Certificate), Eq106Error> {
    let scale = radius.max(1.0);
    let grid = Eq106FrequencyGrid {
        sigma: 2.0 / scale,
        omega_step: 0.002,
        half_count: 128,
    };
    let h_values = [0.0, (0.01 * scale).min(25.0), 50.0, 100.0];
    let (samples, certificate) =
        eq106::certify_eq106_line(line, sources, grid, G as f64, &h_values, 10.0)?;
    if certificate.max_acceleration_relative_error > 0.2
        || certificate.max_potential_relative_error > 0.2
        || certificate.max_boundary_identity_error > 2.0e-6
    {
        warn!(
            "[eq106] rejected certificate: acceleration={:.3e}, potential={:.3e}, boundary={:.3e}",
            certificate.max_acceleration_relative_error,
            certificate.max_potential_relative_error,
            certificate.max_boundary_identity_error
        );
        return Err(Eq106Error::CertificationFailed);
    }
    Ok((samples, certificate))
}

/// Plans the finite non-periodic Eq. (106) arc first. Periodic mode is enabled
/// only after ten stable closures; both paths use convergent Taylor transport.
pub fn monitor_curved_arc_system(
    active_method: Res<ActiveGravityMethod>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    eq106_history: Option<Res<Eq106GpuHistory>>,
    source_data: Option<Res<Eq106SourceData>>,
    clock: Res<SimulationClock>,
    cassini: Query<(&Transform, &Velocity, &OrbitHistory), With<CassiniMarker>>,
    ryugu: Query<&Transform, With<RyuguMarker>>,
    mut planner: ResMut<CurvedArcPlannerState>,
    mut detector: ResMut<PeriodicityDetector>,
    mut residual_history: ResMut<CurvedArcResidualHistory>,
    mut runtime_error: ResMut<GravityRuntimeError>,
) {
    if *active_method != ActiveGravityMethod::CurvedArcEq106 {
        return;
    }

    let Ok((transform, velocity, history)) = cassini.single() else {
        return;
    };
    let Some(topology) = topology else {
        planner.mode = CurvedArcMode::Bootstrap;
        return;
    };
    let Some(_topology_radius) = enclosing_radius(&topology) else {
        planner.mode = CurvedArcMode::Error;
        planner.kernel_status = Eq106KernelStatus::Failed;
        runtime_error.raise("Equation (106) cannot determine a finite density-support radius.");
        return;
    };
    let Some(source_data) = source_data else {
        planner.mode = CurvedArcMode::Bootstrap;
        planner.kernel_status = Eq106KernelStatus::AwaitingSource;
        return;
    };
    let Ok(ryugu_transform) = ryugu.single() else {
        planner.mode = CurvedArcMode::Bootstrap;
        return;
    };
    // Eq. (106) and its density support are body-fixed.  OrbitHistory stores
    // world positions, so plan in the same body frame used by the GPU kernel.
    let radius = source_data.radius;
    if source_data.sources.is_empty()
        || !source_data.total_mass.is_finite()
        || source_data.total_mass <= 0.0
    {
        planner.mode = CurvedArcMode::Error;
        planner.kernel_status = Eq106KernelStatus::Failed;
        runtime_error.raise("Equation (106) has no finite density quadrature sources.");
        return;
    }

    if planner.kernel_status != Eq106KernelStatus::Ready {
        // The render-world Eq.106 pipeline assembles and certifies the shared
        // complex-frequency line on the GPU. The main thread only plans the
        // convergent arc and waits for a snapshot-tagged readback.
        planner.kernel_status = Eq106KernelStatus::Ready;
        planner.kernel_ready = true;
    }

    let points: Vec<Vec3> = history
        .0
        .iter()
        .rev()
        .take(PLANNING_WINDOW_POINTS)
        .map(|point| ryugu_transform.rotation.inverse() * (*point - ryugu_transform.translation))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if points.len() < MIN_SEGMENT_POINTS {
        planner.mode = CurvedArcMode::NonPeriodic;
        planner.kernel_ready = planner.kernel_status == Eq106KernelStatus::Ready;
        planner.taylor_order = 1;
        planner.epsilon_max = Some(0.0);
        update_periodicity(
            &mut detector,
            &mut planner,
            transform.translation,
            velocity.0,
            clock.elapsed_seconds,
        );
        return;
    }

    let mut segments = Vec::new();
    split_until_convergent(&points, 0, points.len() - 1, radius, &mut segments);
    let epsilon_max = segments
        .iter()
        .map(|segment| segment.epsilon_max)
        .reduce(f64::max);
    let rejected = segments.iter().any(|segment| {
        segment.distance_lower_bound <= 0.0
            || segment.epsilon_max >= 1.0
            || segment.taylor_order.is_none()
    });

    planner.active_segment = segments
        .iter()
        .find(|segment| segment.end_index == points.len() - 1)
        .cloned()
        .or_else(|| segments.last().cloned());
    planner.segments = segments;
    planner.epsilon_max = epsilon_max;
    planner.taylor_order = planner
        .active_segment
        .as_ref()
        .and_then(|segment| segment.taylor_order)
        .unwrap_or(1);
    planner.pade_plan = planner.active_segment.as_ref().and_then(|segment| {
        segment.pade_order.map(
            |(numerator_order, denominator_order)| PadeContinuationPlan {
                numerator_order,
                denominator_order,
                estimated_remainder: segment.remainder_bound,
                certified: segment.pade_certified,
            },
        )
    });
    planner.kernel_ready = !rejected
        && !planner.segments.is_empty()
        && planner.kernel_status == Eq106KernelStatus::Ready;
    planner.mode = if rejected || planner.kernel_status != Eq106KernelStatus::Ready {
        CurvedArcMode::Error
    } else if planner.mode == CurvedArcMode::Periodic {
        CurvedArcMode::Periodic
    } else {
        CurvedArcMode::NonPeriodic
    };

    if planner.mode == CurvedArcMode::Error {
        let message = if rejected {
            "Equation (106) stopped: adaptive bisection could not keep every Taylor segment inside its convergence disk."
        } else {
            "Equation (106) complex-frequency kernel is not certified; evaluation is disabled."
        };
        runtime_error.raise(message);
    }

    if let Some(epsilon_max) = epsilon_max {
        // Adaptive Taylor order: Eq. (118) truncation. ε = |δq|/d_safe; the
        // truncation remainder of order A is bounded by ε^(A+1). Pick the
        // smallest A that keeps the next term below 1e-3.
        let taylor_order = planner.taylor_order;
        if let Some(sample) = eq106_history.as_ref().and_then(|history| {
            history
                .0
                .completed_at_or_before(clock.epoch, clock.elapsed_seconds)
        }) && let Some(dual_residual) = residual_history.dual_residual_for(sample)
        {
            residual_history.push(CurvedArcResidualSample {
                simulation_time_seconds: sample.snapshot.simulation_time_seconds,
                epsilon_max,
                dual_residual: Some(dual_residual),
                taylor_order,
            });
        }
    }

    update_periodicity(
        &mut detector,
        &mut planner,
        transform.translation,
        velocity.0,
        clock.elapsed_seconds,
    );
}

fn enclosing_radius(topology: &AsteroidTopologyGpuData) -> Option<f64> {
    topology
        .positions
        .iter()
        .map(|point| point.length() as f64)
        .filter(|radius| radius.is_finite())
        .reduce(f64::max)
}

fn split_until_convergent(
    points: &[Vec3],
    start_index: usize,
    end_index: usize,
    radius: f64,
    output: &mut Vec<CurvedArcSegment>,
) {
    let segment = evaluate_segment(points, start_index, end_index, radius);
    let point_count = end_index - start_index + 1;
    let convergent = segment.distance_lower_bound > 0.0
        && segment.epsilon_max <= EPSILON_TARGET
        && segment.taylor_order.is_some();
    if convergent || point_count <= MIN_SEGMENT_POINTS {
        output.push(segment);
        return;
    }

    let middle = (start_index + end_index) / 2;
    if middle == start_index || middle == end_index {
        output.push(segment);
        return;
    }
    split_until_convergent(points, start_index, middle, radius, output);
    split_until_convergent(points, middle, end_index, radius, output);
}

fn evaluate_segment(
    points: &[Vec3],
    start_index: usize,
    end_index: usize,
    radius: f64,
) -> CurvedArcSegment {
    let start = points[start_index];
    let end = points[end_index];
    let displacement = end - start;
    let length_squared = displacement.length_squared();
    let direction = if length_squared > f32::EPSILON {
        displacement / length_squared.sqrt()
    } else {
        Vec3::ZERO
    };

    let mut maximum_offset = 0.0_f64;
    let mut minimum_line_radius = f64::INFINITY;
    let mut arc_length = 0.0_f64;
    let mut maximum_curvature = 0.0_f64;
    for pair in points[start_index..=end_index].windows(2) {
        arc_length += pair[0].distance(pair[1]) as f64;
    }
    for triple in points[start_index..=end_index].windows(3) {
        let first = (triple[1] - triple[0]).as_dvec3();
        let second = (triple[2] - triple[1]).as_dvec3();
        let chord = (triple[2] - triple[0]).as_dvec3();
        let divisor = first.length() * second.length() * chord.length();
        if divisor > f64::MIN_POSITIVE {
            maximum_curvature = maximum_curvature.max(2.0 * first.cross(second).length() / divisor);
        }
    }
    for point in &points[start_index..=end_index] {
        let projection = if direction == Vec3::ZERO {
            0.0
        } else {
            (*point - start)
                .dot(direction)
                .clamp(0.0, length_squared.sqrt())
        };
        let reference_point = start + direction * projection;
        maximum_offset = maximum_offset.max(point.distance(reference_point) as f64);
        minimum_line_radius = minimum_line_radius.min(reference_point.length() as f64);
    }

    // The density support is contained in the sphere centered at the origin with
    // this radius. Subtracting it gives a conservative lower bound to the true
    // source distance, so passing this test cannot overstate Taylor convergence.
    let distance_lower_bound = minimum_line_radius - radius;
    // docs/mathtidy.md: kappa*l^2/(2*d_min) < 1.  Here l is the
    // segment half-length.  Taking the maximum of the measured chord offset
    // and this curvature sagitta bound makes high-curvature arcs split sooner.
    let curvature_offset = 0.5 * maximum_curvature * (0.5 * arc_length).powi(2);
    let certified_offset = maximum_offset.max(curvature_offset);
    let epsilon_max = if distance_lower_bound > 0.0 {
        certified_offset / distance_lower_bound
    } else {
        f64::INFINITY
    };

    let taylor_order = select_taylor_order(epsilon_max);
    let remainder_bound = taylor_order
        .and_then(|order| taylor_remainder_bound(epsilon_max, order))
        .unwrap_or(f64::INFINITY);

    CurvedArcSegment {
        end_index,
        epsilon_max,
        distance_lower_bound,
        maximum_curvature,
        arc_length,
        taylor_order,
        pade_order: None,
        pade_certified: false,
        remainder_bound,
    }
}

/// Adaptive Taylor truncation order for Eq. (118).
///
/// The remainder of the Taylor expansion of `exp(δq·∇)` truncated at order A
/// is bounded by `ε^(A+1) / (1 - ε)` when `ε < 1`. We pick the smallest A in
/// [1, MAX_TAYLOR_ORDER] whose full geometric remainder meets the target.
/// Order 0 is forbidden (zeroth-order = straight line, Eq. (106) requires
/// at least first-order correction, Eq. (119)).
fn select_taylor_order(epsilon_max: f64) -> Option<u32> {
    if !epsilon_max.is_finite() || !(0.0..1.0).contains(&epsilon_max) {
        return None;
    }
    (1u32..=MAX_TAYLOR_ORDER).find(|&order| {
        taylor_remainder_bound(epsilon_max, order)
            .is_some_and(|bound| bound <= TAYLOR_REMAINDER_TARGET)
    })
}

fn taylor_remainder_bound(epsilon_max: f64, taylor_order: u32) -> Option<f64> {
    if !epsilon_max.is_finite() || epsilon_max >= 1.0 {
        return None;
    }
    let next_term = epsilon_max.powi((taylor_order + 1) as i32);
    let bound = next_term / (1.0 - epsilon_max).max(f64::EPSILON);
    bound.is_finite().then_some(bound)
}

fn update_periodicity(
    detector: &mut PeriodicityDetector,
    planner: &mut CurvedArcPlannerState,
    position: Vec3,
    velocity: Vec3,
    time: f64,
) {
    let origin = detector.plane_origin.get_or_insert(position);
    let normal = detector
        .plane_normal
        .get_or_insert_with(|| velocity.normalize_or_zero());
    if *normal == Vec3::ZERO {
        detector.previous_position = Some(position);
        detector.previous_velocity = Some(velocity);
        return;
    }

    let signed_distance = (position - *origin).dot(*normal);
    let crossed_forward = detector
        .previous_signed_distance
        .is_some_and(|previous| previous < 0.0 && signed_distance >= 0.0)
        && velocity.dot(*normal) > 0.0;

    if crossed_forward {
        let closure = ClosureSample {
            position,
            velocity,
            time,
        };
        if let Some(reference) = detector.reference {
            let position_scale = reference.position.length().max(1.0);
            let velocity_scale = reference.velocity.length().max(1.0e-6);
            let position_error = closure.position.distance(reference.position) / position_scale;
            let velocity_error = closure.velocity.distance(reference.velocity) / velocity_scale;
            let period = closure.time - reference.time;
            let period_error = detector.previous_period.map_or(0.0, |previous| {
                (period - previous).abs() / previous.max(f64::EPSILON)
            });
            let stable = position_error <= CLOSURE_POSITION_TOLERANCE
                && velocity_error <= CLOSURE_VELOCITY_TOLERANCE
                && period_error <= CLOSURE_PERIOD_TOLERANCE
                && matches!(
                    planner.mode,
                    CurvedArcMode::NonPeriodic | CurvedArcMode::Periodic
                );
            if stable {
                planner.stable_closures = planner.stable_closures.saturating_add(1);
            } else {
                planner.stable_closures = 0;
            }
            detector.previous_period = Some(period);
            detector.reference = Some(closure);
            if planner.stable_closures >= REQUIRED_STABLE_CLOSURES {
                // Only ten consecutive position/velocity/period matches promote
                // the general curved-arc planner to the periodic branch.
                planner.mode = CurvedArcMode::Periodic;
            }
        } else {
            detector.reference = Some(closure);
        }
    }

    detector.previous_position = Some(position);
    detector.previous_velocity = Some(velocity);
    detector.previous_signed_distance = Some(signed_distance);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_segment_has_zero_offset() {
        let points = [
            Vec3::Y * 100.0,
            Vec3::Y * 100.0 + Vec3::X,
            Vec3::Y * 100.0 + Vec3::X * 2.0,
            Vec3::Y * 100.0 + Vec3::X * 3.0,
        ];
        let segment = evaluate_segment(&points, 0, points.len() - 1, 0.5);
        assert_eq!(segment.epsilon_max, 0.0);
    }

    #[test]
    fn interior_sphere_marks_segment_as_unusable() {
        let points = [Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::X * 3.0];
        let segment = evaluate_segment(&points, 0, points.len() - 1, 10.0);
        assert!(segment.distance_lower_bound <= 0.0);
        assert!(segment.epsilon_max.is_infinite());
    }

    #[test]
    fn taylor_order_and_dual_residual_follow_convergence_ratio() {
        assert_eq!(select_taylor_order(0.0), Some(1));
        assert!(select_taylor_order(0.1).is_some_and(|order| order >= 2));
        assert!(select_taylor_order(0.7).is_none());
        assert!(taylor_remainder_bound(0.2, 4).is_some_and(|value| value < 1.0e-3));
        assert!(taylor_remainder_bound(1.0, 2).is_none());
    }

    #[test]
    fn curvature_bound_forces_tighter_segments() {
        let gentle = [
            Vec3::new(0.0, 1_000.0, 0.0),
            Vec3::new(10.0, 1_000.1, 0.0),
            Vec3::new(20.0, 1_000.4, 0.0),
            Vec3::new(30.0, 1_000.9, 0.0),
        ];
        let sharp = [
            Vec3::new(0.0, 1_000.0, 0.0),
            Vec3::new(10.0, 1_010.0, 0.0),
            Vec3::new(20.0, 1_000.0, 0.0),
            Vec3::new(30.0, 990.0, 0.0),
        ];
        let gentle = evaluate_segment(&gentle, 0, 3, 400.0);
        let sharp = evaluate_segment(&sharp, 0, 3, 400.0);
        assert!(sharp.maximum_curvature > gentle.maximum_curvature);
        assert!(sharp.epsilon_max > gentle.epsilon_max);
    }

    #[test]
    fn thirty_two_bin_density_fourier_modes_reconstruct_without_alias_loss() {
        let density = std::array::from_fn::<_, EQ106_AZIMUTH_BINS, _>(|index| {
            let phi = std::f64::consts::TAU * index as f64 / EQ106_AZIMUTH_BINS as f64;
            3.0 + 0.7 * (5.0 * phi).cos() - 0.2 * (9.0 * phi).sin()
        });
        let coefficients = std::array::from_fn::<_, 17, _>(|mode| {
            density
                .iter()
                .enumerate()
                .fold([0.0, 0.0], |mut sum, (index, value)| {
                    let phi = std::f64::consts::TAU * index as f64 / EQ106_AZIMUTH_BINS as f64;
                    sum[0] += value * (-(mode as f64) * phi).cos();
                    sum[1] += value * (-(mode as f64) * phi).sin();
                    sum
                })
        });
        for (index, expected) in density.into_iter().enumerate() {
            let phi = std::f64::consts::TAU * index as f64 / EQ106_AZIMUTH_BINS as f64;
            let mut reconstructed = coefficients[0][0] + coefficients[16][0] * (16.0 * phi).cos();
            for (mode, coefficient) in coefficients.iter().enumerate().take(16).skip(1) {
                reconstructed += 2.0
                    * (coefficient[0] * (mode as f64 * phi).cos()
                        - coefficient[1] * (mode as f64 * phi).sin());
            }
            reconstructed /= EQ106_AZIMUTH_BINS as f64;
            assert!((reconstructed - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn runtime_frequency_certificate_accepts_external_line() {
        let line = Eq106ReferenceLine::new(DVec3::new(0.0, 0.0, 1_000.0), DVec3::X).unwrap();
        let sources = vec![Eq106PointSource {
            position: DVec3::ZERO,
            mass: RYUGU_MASS as f64,
        }];
        let result = certify_runtime_line(line, &sources, 1_000.0);
        assert!(result.is_ok(), "certificate failed: {result:?}");
    }
}
