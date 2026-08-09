use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
pub const G: f32 = 6.6743e-11;
pub const RYUGU_MASS: f32 = 4.5e11;
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

/// Shared inverse-radial density law used by the radial-analytic and
/// Equation (106) modes: rho(r) = C / (r + epsilon).
pub fn inverse_radial_density(radius: f32, density_c: f32) -> f32 {
    density_c / (radius.max(0.0) + DENSITY_EPSILON)
}

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
pub struct VramTextMarker;
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
    pub last_sample_method: Option<ActiveGravityMethod>,
}

impl Default for JacobiHistory {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(JACOBI_HISTORY_CAPACITY),
            elapsed_simulation_seconds: 0.0,
            origin_simulation_seconds: None,
            last_request_id: None,
            last_sample_method: None,
        }
    }
}

impl JacobiHistory {
    pub fn reset(&mut self) {
        self.samples.clear();
        self.elapsed_simulation_seconds = 0.0;
        self.origin_simulation_seconds = None;
        self.last_request_id = None;
        self.last_sample_method = None;
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

/// Browser-visible GPU memory accounting. WebGPU does not expose portable
/// driver VRAM counters, so these values are the exact sizes of the buffers
/// allocated by each project pipeline, reported as an auditable estimate.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemoryEstimate {
    pub bytes: [u64; 5],
}

impl GpuMemoryEstimate {
    pub fn total_bytes(self) -> u64 {
        self.bytes.iter().sum()
    }
}

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

/// Snapshot-aligned samples produced by the GPU Equation (106) pipeline.
#[derive(Resource, Default)]
pub struct Eq106GpuHistory(pub GravitySampleHistory);

#[derive(Resource, Clone)]
pub struct Eq106GpuReadbackChannel {
    pub data: Arc<Mutex<Option<GravityReadbackPacket>>>,
    pub in_flight: Arc<AtomicBool>,
}

#[derive(Resource, Default)]
pub struct FmmGravityHistory(pub GravitySampleHistory);

#[derive(Resource)]
pub struct FmmSource {
    pub bytes: Vec<u8>,
    pub node_count: u32,
    pub maximum_level: u32,
}

#[derive(Resource, Clone)]
pub struct FmmReadbackChannel {
    pub data: Arc<Mutex<Option<GravityReadbackPacket>>>,
    pub in_flight: Arc<AtomicBool>,
}

impl Default for FmmReadbackChannel {
    fn default() -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Default for Eq106GpuReadbackChannel {
    fn default() -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Runtime configuration for the MMFFT + GPU-memory-compression implementation.
///
/// The first integration keeps this resource independent from the existing
/// radial/Werner layouts so a future implementation can change packing,
/// quantization, and tiled dispatch without changing ECS call sites.
#[allow(dead_code)]
#[derive(Resource, Clone, Debug)]
pub struct MmfftCompressedConfig {
    /// Number of source records represented by one compressed tile.
    pub tile_size: u32,
    /// Bytes per source record after compression (future format contract).
    pub compressed_record_bytes: u32,
    /// Whether the render pipeline should decode records in shared memory.
    pub decode_in_workgroup: bool,
}

impl Default for MmfftCompressedConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            compressed_record_bytes: 8,
            decode_in_workgroup: true,
        }
    }
}

/// Snapshot-aligned history populated by the dedicated compressed readback
/// channel. Its layout matches the shared gravity sample contract so physics
/// and Jacobi diagnostics can consume it without special-case math.
#[allow(dead_code)]
#[derive(Resource, Default)]
pub struct MmfftCompressedHistory(pub GravitySampleHistory);

#[derive(Resource)]
pub struct MmfftCompressedSource {
    pub bytes: Vec<u8>,
    pub count: u32,
    pub tile_size: u32,
    pub solid_angle_scale: f32,
    pub radius_scale: f32,
    pub density_scale: f32,
}

#[derive(Resource, Clone)]
pub struct MmfftReadbackChannel {
    pub data: Arc<Mutex<Option<GravityReadbackPacket>>>,
    pub in_flight: Arc<AtomicBool>,
}

impl Default for MmfftReadbackChannel {
    fn default() -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Blend weight retained for performance/Jacobi warm-up bookkeeping.
/// Physics never substitutes an alternate force model while this value ramps.
#[derive(Resource, Default)]
pub struct GravityBlendFactor(pub f32);

/// A non-recoverable numerical or pipeline failure. The UI renders this as a
/// blocking modal so an invalid force sample can never silently advance the
/// trajectory with a different physical model.
#[derive(Resource, Clone, Debug, Default)]
pub struct GravityRuntimeError {
    pub message: Option<String>,
}

impl GravityRuntimeError {
    pub fn raise(&mut self, message: impl Into<String>) {
        if self.message.is_none() {
            self.message = Some(message.into());
        }
    }

    pub fn clear(&mut self) {
        self.message = None;
    }

    pub fn is_active(&self) -> bool {
        self.message.is_some()
    }
}

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

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum ActiveGravityMethod {
    #[default]
    RadialAnalytic,
    HomogeneousWerner,
    /// Eq. (106) adaptive curved-arc mode; it starts non-periodic and promotes
    /// itself to periodic only after the planner sees stable orbit closures.
    CurvedArcEq106,
    /// Fourth method: tiled MMFFT source evaluation with compressed GPU
    /// records and a dedicated snapshot-tagged readback channel.
    MmfftCompressed,
    /// Fifth method: fixed-depth GPU fast multipole evaluation.
    Fmm,
}

pub const PERFORMANCE_PHASE_FRAMES: u32 = 120;
pub const PERFORMANCE_HISTORY_CAPACITY: usize = 180;
pub const PERFORMANCE_TEST_DURATION_HOURS: f64 = 100.0;

/// Quarter-turn rotation applied to the complete browser display frame.
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub struct DisplayRotation(pub u8);

impl DisplayRotation {
    pub fn advance(&mut self) -> u8 {
        self.0 = (self.0 + 1) % 4;
        self.0
    }
}

#[derive(Resource, Debug)]
pub struct PerformanceComparisonState {
    pub active: bool,
    pub measuring: bool,
    pub phase: usize,
    pub phase_frames: u32,
    pub phase_elapsed_seconds: f64,
    pub frames_per_second: [f64; 5],
    pub pending_method: Option<ActiveGravityMethod>,
    pub return_method: ActiveGravityMethod,
    pub fps_history: [VecDeque<f32>; 5],
    pub jacobi_history: [VecDeque<f64>; 6],
    pub jacobi_last_request_ids: [Option<u64>; 5],
    pub jacobi_initial_constant: Option<f64>,
    pub jacobi_seeded: [bool; 5],
    /// Algorithms included in the performance rotation. The five entries map
    /// to Radial, Werner, Eq.106, MMFFT, and FMM respectively.
    pub enabled_methods: [bool; 5],
    /// One benchmark pass visits each enabled method once. Keeping this
    /// separate from `enabled_methods` prevents phase wrap-around from
    /// concatenating independently reset trajectories into one curve.
    pub completed_methods: [bool; 5],
}

impl Default for PerformanceComparisonState {
    fn default() -> Self {
        Self {
            active: false,
            measuring: false,
            phase: 0,
            phase_frames: 0,
            phase_elapsed_seconds: 0.0,
            frames_per_second: [0.0; 5],
            pending_method: None,
            return_method: ActiveGravityMethod::RadialAnalytic,
            fps_history: std::array::from_fn(|_| {
                VecDeque::with_capacity(PERFORMANCE_HISTORY_CAPACITY)
            }),
            jacobi_history: std::array::from_fn(|_| {
                VecDeque::with_capacity(PERFORMANCE_HISTORY_CAPACITY)
            }),
            jacobi_last_request_ids: [None; 5],
            jacobi_initial_constant: None,
            jacobi_seeded: [false; 5],
            enabled_methods: [true; 5],
            completed_methods: [false; 5],
        }
    }
}

impl PerformanceComparisonState {
    #[allow(dead_code)]
    pub fn start(&mut self, return_method: ActiveGravityMethod) {
        self.start_with_baseline(return_method, None);
    }

    pub fn start_with_baseline(
        &mut self,
        return_method: ActiveGravityMethod,
        jacobi_initial_constant: Option<f64>,
    ) {
        self.active = true;
        self.measuring = self.enabled_methods.iter().any(|enabled| *enabled);
        self.phase = 0;
        self.phase_frames = 0;
        self.phase_elapsed_seconds = 0.0;
        self.frames_per_second = [0.0; 5];
        self.pending_method = self
            .first_uncompleted_enabled_method()
            .map(|(_, method)| method);
        self.completed_methods = [false; 5];
        self.jacobi_last_request_ids = [None; 5];
        self.jacobi_initial_constant = jacobi_initial_constant.filter(|value| value.is_finite());
        self.jacobi_seeded = [false; 5];
        self.return_method = return_method;
        for history in &mut self.fps_history {
            history.clear();
        }
        for history in &mut self.jacobi_history {
            history.clear();
        }
    }

    pub fn stop(&mut self) {
        self.active = false;
        self.measuring = false;
        self.pending_method = Some(self.return_method);
    }

    #[allow(dead_code)]
    pub fn restart(&mut self) {
        self.start(self.return_method);
    }

    pub fn restart_with_baseline(&mut self, jacobi_initial_constant: Option<f64>) {
        self.start_with_baseline(self.return_method, jacobi_initial_constant);
    }

    pub fn first_enabled_method(&self) -> Option<(usize, ActiveGravityMethod)> {
        (0..self.enabled_methods.len()).find_map(|index| {
            self.enabled_methods[index]
                .then(|| (index, ActiveGravityMethod::from_performance_index(index)))
        })
    }

    pub fn first_uncompleted_enabled_method(&self) -> Option<(usize, ActiveGravityMethod)> {
        (0..self.enabled_methods.len()).find_map(|index| {
            (self.enabled_methods[index] && !self.completed_methods[index])
                .then(|| (index, ActiveGravityMethod::from_performance_index(index)))
        })
    }

    /// Legacy cyclic selector retained for future continuous benchmark modes.
    #[allow(dead_code)]
    pub fn next_enabled_method(
        &self,
        current_index: usize,
    ) -> Option<(usize, ActiveGravityMethod)> {
        if self.enabled_methods.iter().all(|enabled| !*enabled) {
            return None;
        }
        (1..=self.enabled_methods.len()).find_map(|offset| {
            let index = (current_index + offset) % self.enabled_methods.len();
            self.enabled_methods[index]
                .then(|| (index, ActiveGravityMethod::from_performance_index(index)))
        })
    }

    pub fn next_uncompleted_enabled_method(
        &self,
        current_index: usize,
    ) -> Option<(usize, ActiveGravityMethod)> {
        ((current_index + 1)..self.enabled_methods.len()).find_map(|index| {
            (self.enabled_methods[index] && !self.completed_methods[index])
                .then(|| (index, ActiveGravityMethod::from_performance_index(index)))
        })
    }
}

impl ActiveGravityMethod {
    pub fn performance_index(self) -> usize {
        match self {
            Self::RadialAnalytic => 0,
            Self::HomogeneousWerner => 1,
            Self::CurvedArcEq106 => 2,
            Self::MmfftCompressed => 3,
            Self::Fmm => 4,
        }
    }

    pub fn from_performance_index(index: usize) -> Self {
        match index {
            0 => Self::RadialAnalytic,
            1 => Self::HomogeneousWerner,
            2 => Self::CurvedArcEq106,
            3 => Self::MmfftCompressed,
            4 => Self::Fmm,
            // Keep malformed UI state deterministic instead of silently
            // selecting a different algorithm.
            _ => Self::RadialAnalytic,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RadialAnalytic => "GPU Radial Analytic",
            Self::HomogeneousWerner => "GPU Werner Polyhedron",
            Self::CurvedArcEq106 => "Eq.106 Adaptive Curved-Arc",
            Self::MmfftCompressed => "GPU MMFFT + VRAM Compression",
            Self::Fmm => "GPU Fast Multipole Method",
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

    #[test]
    fn display_rotation_advances_in_quarter_turns() {
        let mut rotation = DisplayRotation::default();
        assert_eq!(rotation.advance(), 1);
        assert_eq!(rotation.advance(), 2);
        assert_eq!(rotation.advance(), 3);
        assert_eq!(rotation.advance(), 0);
    }

    #[test]
    fn performance_selection_defaults_to_all_methods() {
        let state = PerformanceComparisonState::default();
        assert_eq!(state.enabled_methods, [true; 5]);
        assert_eq!(
            state.first_enabled_method().map(|(index, _)| index),
            Some(0)
        );
    }

    #[test]
    fn performance_indices_cover_all_five_methods_without_aliasing() {
        assert_eq!(
            (0..5)
                .map(ActiveGravityMethod::from_performance_index)
                .collect::<Vec<_>>(),
            vec![
                ActiveGravityMethod::RadialAnalytic,
                ActiveGravityMethod::HomogeneousWerner,
                ActiveGravityMethod::CurvedArcEq106,
                ActiveGravityMethod::MmfftCompressed,
                ActiveGravityMethod::Fmm,
            ]
        );
        assert_eq!(
            ActiveGravityMethod::from_performance_index(99),
            ActiveGravityMethod::RadialAnalytic
        );
    }

    #[test]
    fn performance_rotation_skips_disabled_methods() {
        let state = PerformanceComparisonState {
            enabled_methods: [true, false, false, true, true],
            ..Default::default()
        };
        assert_eq!(
            state.next_enabled_method(0).map(|(index, _)| index),
            Some(3)
        );
        assert_eq!(
            state.next_enabled_method(3).map(|(index, _)| index),
            Some(4)
        );
    }

    #[test]
    fn performance_rotation_handles_all_methods_disabled() {
        let mut state = PerformanceComparisonState {
            enabled_methods: [false; 5],
            ..Default::default()
        };
        state.start(ActiveGravityMethod::RadialAnalytic);
        assert!(!state.measuring);
        assert!(state.pending_method.is_none());
        assert!(state.next_enabled_method(0).is_none());
    }

    #[test]
    fn benchmark_pass_does_not_wrap_to_a_completed_method() {
        let state = PerformanceComparisonState {
            enabled_methods: [true, true, false, true, true],
            completed_methods: [true, false, false, false, false],
            ..Default::default()
        };
        assert_eq!(
            state
                .next_uncompleted_enabled_method(0)
                .map(|(index, _)| index),
            Some(1)
        );
        assert_eq!(
            PerformanceComparisonState {
                completed_methods: [true, true, false, true, true],
                ..state
            }
            .next_uncompleted_enabled_method(1),
            None
        );
    }

    #[test]
    fn repeat_benchmark_preserves_original_return_method() {
        let mut state = PerformanceComparisonState::default();
        state.start(ActiveGravityMethod::RadialAnalytic);
        state.phase = 3;
        state.frames_per_second = [60.0; 5];
        state.restart();

        assert_eq!(state.return_method, ActiveGravityMethod::RadialAnalytic);
        assert_eq!(
            state.pending_method,
            Some(ActiveGravityMethod::RadialAnalytic)
        );
        assert_eq!(state.frames_per_second, [0.0; 5]);
        assert_eq!(state.completed_methods, [false; 5]);
    }
}
