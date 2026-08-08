use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
pub const G: f32 = 6.6743e-11;
pub const RYUGU_MASS: f32 = 4.5e11;
pub const CASSINI_MASS: f32 = 2500.0;
pub const GRAVITY_EPSILON: f32 = 1.0;
pub const TIME_SCALE: f32 = 500.0;
pub const ORBIT_HISTORY_LEN: usize = 27500;
pub const JACOBI_HISTORY_CAPACITY: usize = 256;
pub const GRAVITY_SAMPLE_HISTORY_CAPACITY: usize = 8;
pub const PHYSICS_SUBSTEPS: usize = 12;
pub const MIN_SIMULATION_ACCELERATION: u32 = 1;
pub const MAX_SIMULATION_ACCELERATION: u32 = 8;
pub const VISIBILITY_THRESHOLD: f32 = 250.0;
pub const NORMAL_ARROW_LENGTH: f32 = 35.0;

pub const RYUGU_ROTATION_PERIOD_SECS: f32 = 7.63 * 3600.0;
pub const RYUGU_SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);

pub const DENSITY_EPSILON: f32 = 10.0;
pub const SECTION_CLIP_RADIUS: f32 = 450.0;
pub const PROBE_R0: Vec3 = Vec3::new(-1000.0, 1200.0, 100.0);
pub const PROBE_SPEED_FACTOR: f32 = 1.053;

pub fn probe_initial_velocity(position: Vec3, speed_factor: f32) -> Vec3 {
    let radius = position.length();
    if !radius.is_finite() || radius <= f32::EPSILON {
        return Vec3::ZERO;
    }

    let radial = position / radius;
    let reference_axis = if radial.dot(Vec3::Y).abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = radial.cross(reference_axis).normalize_or_zero();
    let speed = speed_factor.clamp(0.0, 2.0) * (G * RYUGU_MASS / radius).sqrt();
    tangent * speed
}

pub static PROBE_V_INIT: LazyLock<Vec3> =
    LazyLock::new(|| probe_initial_velocity(PROBE_R0, PROBE_SPEED_FACTOR));

#[derive(Resource, Clone, Copy, PartialEq)]
pub struct ProbeInitialConditions {
    pub position: Vec3,
    pub speed_factor: f32,
}

impl Default for ProbeInitialConditions {
    fn default() -> Self {
        Self {
            position: PROBE_R0,
            speed_factor: PROBE_SPEED_FACTOR,
        }
    }
}

impl ProbeInitialConditions {
    pub fn velocity(self) -> Vec3 {
        if self.position == PROBE_R0 && self.speed_factor == PROBE_SPEED_FACTOR {
            *PROBE_V_INIT
        } else {
            probe_initial_velocity(self.position, self.speed_factor)
        }
    }
}
#[derive(Component)]
pub struct TargetSize(pub f32);
#[derive(Component)]
pub struct ScaleNormalized;
#[derive(Component)]
pub struct TopologyBuilt;
#[derive(Component)]
pub struct RyuguMarker;
#[derive(Component)]
pub struct CassiniMarker;
#[derive(Component)]
pub struct UiTextMarker;
#[derive(Component)]
pub struct FpsTextMarker;
#[derive(Component)]
pub struct Mass(pub f32);
#[derive(Component)]
pub struct Velocity(pub Vec3);
#[derive(Component)]
pub struct OrbitHistory(pub VecDeque<Vec3>);

#[derive(Clone, Copy, Debug)]
pub struct JacobiSample {
    pub simulation_time_seconds: f64,
    pub jacobi_constant: f64,
}

#[derive(Resource)]
pub struct JacobiHistory {
    pub samples: VecDeque<JacobiSample>,
    pub elapsed_simulation_seconds: f64,
    pub origin_simulation_seconds: Option<f64>,
    pub last_request_id: Option<u64>,
}

impl Default for JacobiHistory {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(JACOBI_HISTORY_CAPACITY),
            elapsed_simulation_seconds: 0.0,
            origin_simulation_seconds: None,
            last_request_id: None,
        }
    }
}

impl JacobiHistory {
    pub fn reset(&mut self) {
        self.samples.clear();
        self.elapsed_simulation_seconds = 0.0;
        self.origin_simulation_seconds = None;
        self.last_request_id = None;
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SimulationClock {
    pub request_id: u64,
    pub epoch: u64,
    pub elapsed_seconds: f64,
}

impl SimulationClock {
    pub fn advance(&mut self, seconds: f64) {
        self.elapsed_seconds += seconds;
        self.request_id = self.request_id.wrapping_add(1);
    }

    pub fn reset_state(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.request_id = self.request_id.wrapping_add(1);
        self.elapsed_seconds = 0.0;
    }
}

/// Number of fully integrated stable physics frames advanced before presenting
/// the next visual state. This changes throughput, never the integration step.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimulationAcceleration(pub u32);

impl Default for SimulationAcceleration {
    fn default() -> Self {
        Self(MIN_SIMULATION_ACCELERATION)
    }
}

impl SimulationAcceleration {
    pub fn stable_steps(self) -> u32 {
        self.0
            .clamp(MIN_SIMULATION_ACCELERATION, MAX_SIMULATION_ACCELERATION)
    }
}

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub enum CameraMode {
    #[default]
    Overview,
    FollowCassini,
}

#[derive(Resource, Default)]
pub struct ShowNormals(pub bool);

#[derive(Resource, Default)]
pub struct ShowSection(pub bool);

/// Solved density constant: C = RYUGU_MASS / ∫(1/(||r||+ε))dV
/// Kernel is 1/(||r||+ε), NOT 1/(r³+ε³). Used for section-view colormap.
#[derive(Resource)]
pub struct DensityC(pub f32);

impl Default for DensityC {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Constant density used by the homogeneous Werner polyhedron model.
#[derive(Resource, Default)]
pub struct WernerDensity(pub f32);

#[derive(Resource)]
pub struct AsteroidTopologyGpuData {
    pub mesh_entity: Option<Entity>,
    pub node_count: u32,
    pub positions: Vec<Vec3>,
    pub triangles: Vec<u32>,
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
}

#[derive(Resource)]
pub struct AsteroidNormalsGpuData(pub Vec<Vec3>);

#[derive(Resource, Clone)]
pub struct NormalsReadbackChannel(pub Arc<Mutex<Option<Vec<[f32; 4]>>>>);

impl Default for NormalsReadbackChannel {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

/// Radial-analytic discretization. Each 32-byte record stores one angular cell
/// and one radial layer as `[direction.xyz, solid_angle]` followed by
/// `[r_inner, r_outer, density, padding]`.
#[derive(Resource)]
pub struct RadialGravitySource {
    pub bytes: Vec<u8>,
    pub count: u32,
}

/// Latest GPU-computed gravity acceleration for Cassini (Ryugu body frame).
#[derive(Resource, Default)]
pub struct GravityAcceleration(pub Vec3);

/// Latest positive gravitational potential U returned by the radial GPU model.
#[derive(Resource, Default)]
pub struct GravityPotential(pub Option<f32>);

/// Main-world state captured when a render-world gravity dispatch is submitted.
/// The returned acceleration and potential are only valid for this snapshot.
#[derive(Clone, Debug)]
pub struct GravityRequestSnapshot {
    pub request_id: u64,
    pub epoch: u64,
    pub simulation_time_seconds: f64,
    pub body_position: Vec3,
    pub ryugu_transform: Transform,
    pub probe_position: Vec3,
    pub probe_velocity: Vec3,
}

#[derive(Clone, Debug)]
pub struct GravityReadbackPacket {
    pub partial_sums: Vec<[f32; 4]>,
    pub snapshot: GravityRequestSnapshot,
}

#[derive(Clone, Debug)]
pub struct GravityFieldSample {
    pub snapshot: GravityRequestSnapshot,
    pub body_acceleration: Vec3,
    pub positive_potential: f32,
}

#[derive(Default)]
pub struct GravitySampleHistory {
    pub samples: VecDeque<GravityFieldSample>,
}

impl GravitySampleHistory {
    pub fn push(&mut self, sample: GravityFieldSample) {
        if self
            .samples
            .back()
            .is_some_and(|latest| latest.snapshot.request_id == sample.snapshot.request_id)
        {
            self.samples.pop_back();
        }
        if self.samples.len() == GRAVITY_SAMPLE_HISTORY_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn latest_for_epoch(&self, epoch: u64) -> Option<&GravityFieldSample> {
        self.samples
            .iter()
            .rev()
            .find(|sample| sample.snapshot.epoch == epoch)
    }
}

#[derive(Resource, Default)]
pub struct RadialGravityHistory(pub GravitySampleHistory);

#[derive(Resource, Default)]
pub struct WernerGravityHistory(pub GravitySampleHistory);

/// Blend weight for smooth GPU gravity warm-up transition.
/// 0.0 = 100% Newtonian fallback, 1.0 = 100% GPU-computed.
/// Increments by 1/GRAVITY_BLEND_FRAMES per frame once the first valid GPU result
/// arrives, reaching 1.0 after GRAVITY_BLEND_FRAMES frames.
#[derive(Resource, Default)]
pub struct GravityBlendFactor(pub f32);

pub const GRAVITY_BLEND_FRAMES: f32 = 60.0;

/// Shared channel: render world writes workgroup sums, main world reduces them.
/// `in_flight` prevents mapping the same staging buffer twice.
#[derive(Resource, Clone)]
pub struct GravityReadbackChannel {
    pub data: Arc<Mutex<Option<GravityReadbackPacket>>>,
    pub in_flight: Arc<AtomicBool>,
}

impl Default for GravityReadbackChannel {
    fn default() -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub enum ActiveGravityMethod {
    #[default]
    RadialAnalytic,
    HomogeneousWerner,
    /// Eq. (106) adaptive curved-arc mode; it starts non-periodic and promotes
    /// itself to periodic only after the planner sees stable orbit closures.
    CurvedArcEq106,
}

impl ActiveGravityMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RadialAnalytic => "GPU Radial Analytic",
            Self::HomogeneousWerner => "GPU Werner Polyhedron",
            Self::CurvedArcEq106 => "Eq.106 Adaptive Curved-Arc",
        }
    }
}

#[cfg(test)]
mod probe_initial_condition_tests {
    use super::*;

    #[test]
    fn zero_position_has_safe_zero_velocity() {
        assert_eq!(probe_initial_velocity(Vec3::ZERO, 1.0), Vec3::ZERO);
    }

    #[test]
    fn position_parallel_to_y_axis_has_finite_tangent_velocity() {
        let velocity = probe_initial_velocity(Vec3::Y * 1000.0, 1.0);
        assert!(velocity.is_finite());
        assert!(velocity.length() > 0.0);
        assert!(velocity.dot(Vec3::Y).abs() < 1.0e-6);
    }

    #[test]
    fn simulation_acceleration_is_bounded() {
        assert_eq!(SimulationAcceleration(0).stable_steps(), 1);
        assert_eq!(SimulationAcceleration(4).stable_steps(), 4);
        assert_eq!(SimulationAcceleration(99).stable_steps(), 8);
    }
}
