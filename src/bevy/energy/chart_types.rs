use crate::interface::components::*;
use crate::interface::select_history;
use crate::cpu::curved_arc::{CurvedArcPlannerState, CurvedArcResidualHistory};
use bevy::math::Rot2;
use bevy::prelude::*;

const CHART_WIDTH: f32 = 350.0;
const CHART_HEIGHT: f32 = 170.0;
const FIXED_UPDATES_PER_SECOND: f64 = 60.0;
const JACOBI_BASE_WINDOW_SECONDS: f64 =
    (JACOBI_HISTORY_CAPACITY - 1) as f64 * TIME_SCALE as f64 / FIXED_UPDATES_PER_SECOND;
const CHART_LINE_COLOR: Color = Color::srgb(0.2, 0.9, 0.45);
const RESIDUAL_LINE_COLOR: Color = Color::srgb(1.0, 0.55, 0.2);

#[derive(Component)]
pub(crate) struct JacobiChartSegment(usize);

#[derive(Component)]
pub(crate) struct JacobiLatestPoint;

#[derive(Component)]
pub(crate) struct JacobiChartTitle;

#[derive(Component)]
pub(crate) struct JacobiChartAxisLabel;

#[derive(Component, Clone, Copy)]
pub(crate) enum JacobiChartLabel {
    Current,
    RelativeDrift,
    Minimum,
    Maximum,
    TimeStart,
    TimeEnd,
}

#[derive(Component)]
pub(crate) struct Eq106ResidualChartRoot;

#[derive(Component)]
pub(crate) struct Eq106ResidualChartSegment(usize);

#[derive(Component)]
pub(crate) struct Eq106ResidualLatestPoint;

#[derive(Component, Clone, Copy)]
pub(crate) enum Eq106ResidualChartLabel {
    Current,
    Status,
    Minimum,
    Maximum,
    TimeStart,
    TimeEnd,
}

pub fn rotating_frame_jacobi_constant(
    body_position: Vec3,
    inertial_velocity_body_frame: Vec3,
    positive_gravitational_potential: f32,
    angular_velocity_body_frame: Vec3,
) -> Option<f64> {
    if !body_position.is_finite()
        || !inertial_velocity_body_frame.is_finite()
        || !positive_gravitational_potential.is_finite()
        || positive_gravitational_potential <= 0.0
        || !angular_velocity_body_frame.is_finite()
    {
        return None;
    }

    let frame_velocity =
        inertial_velocity_body_frame - angular_velocity_body_frame.cross(body_position);
    let centrifugal_speed = angular_velocity_body_frame.cross(body_position);
    let jacobi = 2.0 * positive_gravitational_potential as f64
        + centrifugal_speed.length_squared() as f64
        - frame_velocity.length_squared() as f64;
    jacobi.is_finite().then_some(jacobi)
}

/// Potential paired with the conservative local field used by the CPU Eq.106
/// substeps. For `g(x) = g0 + H (x-x0)`, this construction guarantees
/// `grad(U_loc) = g` when `H` is symmetric.
fn eq106_local_positive_potential(sample: &GravityFieldSample, body_position: Vec3) -> Option<f32> {
    let jacobian = sample.body_acceleration_jacobian?;
    let displacement = body_position - sample.snapshot.body_position;
    if !body_position.is_finite() || !displacement.is_finite() || !jacobian.is_finite() {
        return None;
    }
    let hessian = (jacobian + jacobian.transpose()) * 0.5;
    let potential = sample.positive_potential
        + sample.body_acceleration.dot(displacement)
        + 0.5 * displacement.dot(hessian * displacement);
    (potential.is_finite() && potential > 0.0).then_some(potential)
}

fn eq106_interpolated_positive_potential(
    history: &GravitySampleHistory,
    epoch: u64,
    simulation_time_seconds: f64,
    body_position: Vec3,
) -> Option<f32> {
    let (lower, upper) = history.bracketing(epoch, simulation_time_seconds)?;
    let lower_potential = eq106_local_positive_potential(lower, body_position)?;
    if std::ptr::eq(lower, upper) {
        return Some(lower_potential);
    }
    let upper_potential = eq106_local_positive_potential(upper, body_position)?;
    let interval = upper.snapshot.simulation_time_seconds - lower.snapshot.simulation_time_seconds;
    if interval <= f64::EPSILON {
        return Some(lower_potential);
    }
    let weight = ((simulation_time_seconds - lower.snapshot.simulation_time_seconds) / interval)
        .clamp(0.0, 1.0) as f32;
    lower_potential
        .lerp(upper_potential, weight)
        .is_finite()
        .then_some(lower_potential.lerp(upper_potential, weight))
}

fn eq106_coordinates_at(
    diagnostics: Eq106SampleDiagnostics,
    body_position: Vec3,
) -> Eq106SampleDiagnostics {
    let tangent = diagnostics.line_direction.normalize_or_zero();
    let helper = if tangent.z.abs() > 0.8 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let normal = helper.cross(tangent).normalize_or_zero();
    let binormal = tangent.cross(normal).normalize_or_zero();
    let relative = body_position - diagnostics.line_origin;
    Eq106SampleDiagnostics {
        h: relative.dot(tangent),
        u: relative.dot(normal),
        v: relative.dot(binormal),
        ..diagnostics
    }
}
