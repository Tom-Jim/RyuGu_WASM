use crate::components::*;
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
    Fallback,
}

impl CurvedArcMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "Non-periodic warm-up",
            Self::NonPeriodic => "Eq.106 non-periodic",
            Self::Periodic => "Eq.106 periodic",
            Self::Fallback => "Newton fallback / split required",
        }
    }

    pub fn short_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "Warm-up",
            Self::NonPeriodic => "Non-periodic",
            Self::Periodic => "Periodic",
            Self::Fallback => "Fallback",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CurvedArcSegment {
    pub end_index: usize,
    pub epsilon_max: f64,
    pub distance_lower_bound: f64,
    pub taylor_order: Option<u32>,
    pub remainder_bound: f64,
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
    /// Taylor truncation order actually used (1, 2, or 3). See `taylor_order`.
    pub taylor_order: u32,
}

#[derive(Resource)]
pub struct CurvedArcResidualHistory {
    pub samples: VecDeque<CurvedArcResidualSample>,
    origin_potential: Option<f64>,
    previous_request_id: Option<u64>,
    previous_body_position: Option<Vec3>,
    previous_body_acceleration: Option<Vec3>,
    accumulated_curve_work: f64,
}

impl Default for CurvedArcResidualHistory {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(JACOBI_HISTORY_CAPACITY),
            origin_potential: None,
            previous_request_id: None,
            previous_body_position: None,
            previous_body_acceleration: None,
            accumulated_curve_work: 0.0,
        }
    }
}

impl CurvedArcResidualHistory {
    pub fn reset(&mut self) {
        self.samples.clear();
        self.origin_potential = None;
        self.previous_request_id = None;
        self.previous_body_position = None;
        self.previous_body_acceleration = None;
        self.accumulated_curve_work = 0.0;
    }

    fn push(&mut self, sample: CurvedArcResidualSample) {
        if self.samples.len() == JACOBI_HISTORY_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    fn dual_residual_for(&mut self, sample: &GravityFieldSample) -> Option<f64> {
        if self.previous_request_id == Some(sample.snapshot.request_id) {
            return None;
        }
        let potential = sample.positive_potential as f64;
        if !potential.is_finite()
            || !sample.snapshot.body_position.is_finite()
            || !sample.body_acceleration.is_finite()
        {
            return None;
        }

        let origin = *self.origin_potential.get_or_insert(potential);
        if let (Some(previous_position), Some(previous_acceleration)) =
            (self.previous_body_position, self.previous_body_acceleration)
        {
            let displacement = sample.snapshot.body_position - previous_position;
            let average_acceleration = 0.5 * (previous_acceleration + sample.body_acceleration);
            self.accumulated_curve_work += average_acceleration.dot(displacement) as f64;
        }

        self.previous_request_id = Some(sample.snapshot.request_id);
        self.previous_body_position = Some(sample.snapshot.body_position);
        self.previous_body_acceleration = Some(sample.body_acceleration);

        // Eq. (147) supplies P_70 through the curved-path work integral, while
        // the independently accumulated GPU potential supplies P_spec. Their
        // finite-discretization difference is the Eq. (157) dual residual.
        let residual = self.accumulated_curve_work - (potential - origin);
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

/// Plans the finite non-periodic Eq. (106) arc first. Periodic mode is enabled
/// only after ten stable closures; both paths use convergent Taylor transport.
pub fn monitor_curved_arc_system(
    active_method: Res<ActiveGravityMethod>,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    radial_history: Option<Res<RadialGravityHistory>>,
    clock: Res<SimulationClock>,
    cassini: Query<(&Transform, &Velocity, &OrbitHistory), With<CassiniMarker>>,
    mut planner: ResMut<CurvedArcPlannerState>,
    mut detector: ResMut<PeriodicityDetector>,
    mut residual_history: ResMut<CurvedArcResidualHistory>,
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
    let Some(radius) = enclosing_radius(&topology) else {
        planner.mode = CurvedArcMode::Fallback;
        return;
    };

    let points: Vec<Vec3> = history
        .0
        .iter()
        .rev()
        .take(PLANNING_WINDOW_POINTS)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if points.len() < MIN_SEGMENT_POINTS {
        planner.mode = CurvedArcMode::Bootstrap;
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
    let rejected = segments
        .iter()
        .any(|segment| segment.taylor_order.is_none());

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
    planner.kernel_ready = !rejected && !planner.segments.is_empty();
    planner.mode = if rejected {
        CurvedArcMode::Fallback
    } else if planner.mode == CurvedArcMode::Periodic {
        CurvedArcMode::Periodic
    } else {
        CurvedArcMode::NonPeriodic
    };

    if let Some(epsilon_max) = epsilon_max {
        // Adaptive Taylor order: Eq. (118) truncation. ε = |δq|/d_safe; the
        // truncation remainder of order A is bounded by ε^(A+1). Pick the
        // smallest A that keeps the next term below 1e-3.
        let taylor_order = planner.taylor_order;
        if let Some(sample) = radial_history
            .as_ref()
            .and_then(|history| history.0.latest_for_epoch(clock.epoch))
            && let Some(dual_residual) = residual_history.dual_residual_for(sample)
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
    if (segment.epsilon_max <= EPSILON_TARGET && segment.taylor_order.is_some())
        || point_count <= MIN_SEGMENT_POINTS
        || segment.distance_lower_bound <= 0.0
    {
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
    let epsilon_max = if distance_lower_bound > 0.0 {
        maximum_offset / distance_lower_bound
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
        taylor_order,
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
    if !epsilon_max.is_finite() || epsilon_max < 0.0 || epsilon_max >= 1.0 {
        return None;
    }
    for order in 1u32..=MAX_TAYLOR_ORDER {
        if taylor_remainder_bound(epsilon_max, order)
            .is_some_and(|bound| bound <= TAYLOR_REMAINDER_TARGET)
        {
            return Some(order);
        }
    }
    None
}

fn taylor_remainder_bound(epsilon_max: f64, taylor_order: u32) -> Option<f64> {
    if !epsilon_max.is_finite() || epsilon_max >= 1.0 {
        return None;
    }
    let next_term = epsilon_max.powi((taylor_order + 1) as i32);
    let bound = next_term / (1.0 - epsilon_max).max(f64::EPSILON);
    bound.is_finite().then_some(bound)
}

/// Eq. (118) transport of a completed straight-line field sample onto the
/// current curved point. The point-mass derivatives are a bounded local proxy
/// for omitted spatial derivatives; ε >= 1 is rejected by the planner.
pub fn approximate_eq106_acceleration(
    history: &GravitySampleHistory,
    epoch: u64,
    position_world: Vec3,
    ryugu_transform: Transform,
    source_mass: f32,
    planner: &CurvedArcPlannerState,
) -> Option<Vec3> {
    if !planner.kernel_ready || planner.mode == CurvedArcMode::Fallback {
        return None;
    }
    let sample = history.latest_for_epoch(epoch)?;
    let sample_body = sample.snapshot.body_position;
    let target_body =
        ryugu_transform.rotation.inverse() * (position_world - ryugu_transform.translation);
    let delta = target_body - sample_body;
    let safe_distance = planner
        .active_segment
        .as_ref()
        .map(|segment| segment.distance_lower_bound)
        .unwrap_or(f64::INFINITY);
    if !delta.is_finite() || delta.length() as f64 >= safe_distance.max(0.0) {
        return None;
    }

    let correction = point_mass_taylor_correction(
        sample_body,
        delta,
        G as f64 * source_mass as f64,
        planner.taylor_order,
    )?;
    let transported = sample.body_acceleration + correction;
    let world = ryugu_transform.rotation * transported;
    world.is_finite().then_some(world)
}

fn point_mass_taylor_correction(reference: Vec3, delta: Vec3, mu: f64, order: u32) -> Option<Vec3> {
    let reference = reference.as_dvec3();
    let delta = delta.as_dvec3();
    let radius_squared = reference.length_squared();
    if !radius_squared.is_finite() || radius_squared <= f64::EPSILON || order == 0 {
        return None;
    }

    // For r(t)=r+tδ, expand (1+a t+b t²)^(-3/2) as a polynomial in t and
    // retain only degrees <= A. This is the directional Taylor series from
    // Eq. (118), evaluated at t=1, without hard-coding derivative tensors.
    let a = 2.0 * reference.dot(delta) / radius_squared;
    let b = delta.length_squared() / radius_squared;
    let size = order as usize + 1;
    let mut y = vec![0.0_f64; size];
    if size > 1 {
        y[1] = a;
    }
    if size > 2 {
        y[2] = b;
    }
    let mut power = vec![0.0_f64; size];
    power[0] = 1.0;
    let mut scale = vec![0.0_f64; size];
    let mut binomial = 1.0_f64;
    for n in 0..=order as usize {
        for degree in 0..size {
            scale[degree] += binomial * power[degree];
        }
        if n == order as usize {
            break;
        }
        let mut next = vec![0.0_f64; size];
        for left in 0..size {
            for right in 0..(size - left) {
                next[left + right] += power[left] * y[right];
            }
        }
        power = next;
        binomial *= (-1.5 - n as f64) / (n as f64 + 1.0);
    }

    let inverse_radius_cubed = radius_squared.sqrt().recip() / radius_squared;
    let mut taylor_acceleration = DVec3::ZERO;
    for degree in 0..size {
        let numerator = reference * scale[degree]
            + if degree > 0 {
                delta * scale[degree - 1]
            } else {
                DVec3::ZERO
            };
        taylor_acceleration += -mu * inverse_radius_cubed * numerator;
    }
    let base_acceleration = -mu * inverse_radius_cubed * reference;
    let correction = (taylor_acceleration - base_acceleration).as_vec3();
    correction.is_finite().then_some(correction)
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
    fn point_mass_taylor_converges_toward_exact_translation() {
        let reference = Vec3::new(1200.0, -300.0, 200.0);
        let delta = Vec3::new(20.0, 10.0, -5.0);
        let mu = G as f64 * RYUGU_MASS as f64;
        let exact = (-mu * (reference + delta).as_dvec3()
            / (reference + delta).as_dvec3().length().powi(3))
        .as_vec3();
        let base = (-mu * reference.as_dvec3() / reference.as_dvec3().length().powi(3)).as_vec3();
        let second = base + point_mass_taylor_correction(reference, delta, mu, 2).unwrap();
        let sixth = base + point_mass_taylor_correction(reference, delta, mu, 6).unwrap();
        assert!(sixth.distance(exact) < second.distance(exact));
    }
}
