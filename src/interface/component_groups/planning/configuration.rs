pub const PROBE_R0: Vec3 = Vec3::new(-616.535, 0.0, -65.459);
pub const PROBE_SPEED_FACTOR: f32 = 1.07;
pub const PROBE_ORBIT_NORMAL: Vec3 = Vec3::new(0.037_806, -0.933_691, -0.356_079);

pub const NEAR_SYNC_SEGMENT_MAX_SECONDS: f32 = 300.0;

pub const PLANNING_GRAVITY_ERROR_LIMIT: f32 = 2.0e-2;
pub const PLANNING_GRADIENT_ERROR_LIMIT: f32 = 2.5e-1;
pub const PLANNING_PERICENTER_ERROR_LIMIT_METERS: f32 = 1.0;

/// Reporting policy only: changing it never changes numerical outputs or
/// reference samples. Screening is explicitly unsuitable for strict claims.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanningAccuracyProfile {
    #[default]
    Strict,
    Screening,
}

#[derive(Clone, Copy)]
pub struct PlanningAccuracyLimits {
    pub gravity: f32,
    pub gradient: f32,
    pub gravity_p99: f32,
    pub gradient_p99: f32,
    pub gravity_max: f32,
    pub gradient_max: f32,
    pub pericenter_m: f32,
}

impl PlanningAccuracyProfile {
    pub fn key(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Screening => "screening",
        }
    }

    pub fn limits(self) -> PlanningAccuracyLimits {
        match self {
            Self::Strict => PlanningAccuracyLimits {
                // Equation (184) is evaluated with a finite reciprocal-space
                // quadrature.  Keep strict gates meaningful but account for
                // the declared spectral truncation and interpolation error.
                gravity: PLANNING_GRAVITY_ERROR_LIMIT,
                gradient: PLANNING_GRADIENT_ERROR_LIMIT,
                gravity_p99: 2.5 * PLANNING_GRAVITY_ERROR_LIMIT,
                gradient_p99: 2.0 * PLANNING_GRADIENT_ERROR_LIMIT,
                gravity_max: 5.0 * PLANNING_GRAVITY_ERROR_LIMIT,
                gradient_max: 4.0 * PLANNING_GRADIENT_ERROR_LIMIT,
                pericenter_m: PLANNING_PERICENTER_ERROR_LIMIT_METERS,
            },
            Self::Screening => PlanningAccuracyLimits {
                gravity: 0.02,
                gradient: 0.25,
                gravity_p99: 0.05,
                gradient_p99: 0.50,
                gravity_max: 0.10,
                gradient_max: 1.0,
                pericenter_m: 10.0,
            },
        }
    }
}

pub fn planning_accuracy_failure_labels(mask: u32) -> Vec<&'static str> {
    [
        "GPU/workload verification",
        "gravity RMS",
        "gradient RMS",
        "gravity p99/max",
        "gradient p99/max",
        "pericenter drift",
        "candidate coverage/score",
        "invalid timing",
        "missing/common reference samples",
        "external validation rejection",
    ]
    .into_iter()
    .enumerate()
    .filter_map(|(bit, reason)| (mask & (1 << bit) != 0).then_some(reason))
    .collect()
}
pub const PLANNING_SOURCE_COUNTS: [u32; 9] = [
    32_000, 64_000, 128_000, 256_000, 512_000, 1_024_000, 2_048_000, 4_096_000, 8_192_000,
];
pub const PLANNING_SOURCE_REPEATS: u32 = 7;
pub const PLANNING_DENSITY_MODEL_COUNTS: [u32; 7] = [1, 4, 16, 64, 256, 512, 1024];
pub const PLANNING_TARGET_COUNTS: [u32; 5] = [8, 64, 241, 1024, 8192];
pub const RYUGU_COLLISION_RADIUS_METERS: f32 = 464.765;
pub const PROBE_COLLISION_RADIUS_METERS: f32 = 3.35;

pub fn probe_initial_velocity_for_normal(
    position: Vec3,
    speed_factor: f32,
    orbit_normal: Vec3,
) -> Vec3 {
    let radius = position.length();
    if !radius.is_finite() || radius <= f32::EPSILON {
        return Vec3::ZERO;
    }
    let radial = position / radius;
    let tangent = orbit_normal
        .normalize_or_zero()
        .cross(radial)
        .normalize_or_zero();
    let speed = speed_factor.clamp(0.0, 2.0) * (G * RYUGU_MASS / radius).sqrt();
    tangent * speed
}

pub fn probe_initial_velocity(position: Vec3, speed_factor: f32) -> Vec3 {
    probe_initial_velocity_for_normal(position, speed_factor, PROBE_ORBIT_NORMAL)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProbeOrbitPreset {
    #[default]
    CurrentBenchmark,
    Custom,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ProbeInitialConditions {
    pub position: Vec3,
    pub speed_factor: f32,
    pub orbit_normal: Vec3,
    pub preset: ProbeOrbitPreset,
}

impl Default for ProbeInitialConditions {
    fn default() -> Self {
        Self {
            position: PROBE_R0,
            speed_factor: PROBE_SPEED_FACTOR,
            orbit_normal: PROBE_ORBIT_NORMAL,
            preset: ProbeOrbitPreset::CurrentBenchmark,
        }
    }
}

impl ProbeInitialConditions {
    pub fn velocity(self) -> Vec3 {
        if self.preset == ProbeOrbitPreset::CurrentBenchmark
            && self.orbit_normal == PROBE_ORBIT_NORMAL
        {
            probe_initial_velocity(self.position, self.speed_factor)
        } else {
            probe_initial_velocity_for_normal(self.position, self.speed_factor, self.orbit_normal)
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanningWorkloadProfile {
    #[default]
    First,
    /// Large, interruptible workload for exercising the planning pipelines
    /// without taking ownership of the browser frame loop.
    InteractiveStress,
    /// Fixed geometry crossover over source count, density RHS count and
    /// target count, with independent repeated timings in every sweep cell.
    SourceCrossover,
}

impl PlanningWorkloadProfile {
    pub fn is_compute_benchmark(self) -> bool {
        matches!(self, Self::First | Self::SourceCrossover)
    }

    pub fn dimensions(self) -> (u32, u32, u32) {
        match self {
            Self::First => (PLANNING_FIRST_CANDIDATE_COUNT, 4, 241),
            Self::InteractiveStress => (PLANNING_CANDIDATE_COUNT, 32, 512),
            // Initial sweep cell. PlanningComparisonState supplies the current
            // K_rho x N_t dimensions for every subsequent cell.
            Self::SourceCrossover => (
                1,
                PLANNING_DENSITY_MODEL_COUNTS[0],
                PLANNING_TARGET_COUNTS[0],
            ),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::First => "First 32x4x241",
            Self::InteractiveStress => "Stress 2048x32x512",
            Self::SourceCrossover => "Quadrature-source/density/target crossover",
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComparisonMetric {
    #[default]
    DensityFit,
    InversionTime,
    GravityRelativeError,
    GradientRelativeError,
    PericenterError,
    MinimumAltitude,
    ModelDiscrimination,
    PlanningObjective,
    SegmentCount,
    SpeedupVsGpuFmm,
    ColdStartAmortization,
}

impl ComparisonMetric {
    pub fn is_inversion(self) -> bool {
        matches!(self, Self::DensityFit | Self::InversionTime)
    }
}
