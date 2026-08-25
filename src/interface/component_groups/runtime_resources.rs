
#[derive(Resource, Default)]
pub struct FmmGravityHistory(pub GravitySampleHistory);

#[derive(Clone, Copy, Debug)]
pub struct VoxelBasisSource {
    pub position: bevy::math::DVec3,
    /// Unit-density source volume in cubic metres.
    pub volume: f64,
}

#[derive(Clone, Debug, Default)]
pub struct VoxelBasisSources {
    /// One distributed source column for every inversion voxel.
    pub columns: Vec<Vec<VoxelBasisSource>>,
    pub hash: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InversionTimingBreakdown {
    /// Common high-resolution truth construction, excluded from method time.
    pub truth_prepare_ms: f64,
    pub matrix_build_ms: f64,
    pub matrix_cache_hit: bool,
    pub convex_solve_ms: f64,
    pub verification_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct DensitySensitivityCache {
    pub capture_id: Option<u64>,
    pub source_hash: u64,
    pub basis_hash: u64,
    pub sample_count: usize,
    pub values: Vec<Vec3>,
}

#[derive(Resource, Default)]
pub struct DensitySensitivityCaches(pub [DensitySensitivityCache; 5]);

#[derive(Resource)]
pub struct FmmSource {
    pub bytes: Vec<u8>,
    /// Leaf particles packed as `(x, y, z, mass)` for the exact P2P near field.
    pub particle_bytes: Vec<u8>,
    pub node_count: u32,
    pub particle_count: u32,
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
            pipeline_error: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
            submitted_at: Arc::new(Mutex::new(None)),
            rebuild_requested: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Eq106TimingSample {
    pub spectrum_build_ms: Option<f64>,
    pub target_evaluation_ms: Option<f64>,
    pub gpu_readback_copy_ms: Option<f64>,
    pub cpu_readback_wait_ms: f64,
    pub target_count: u32,
    pub spectral_element_count: u32,
    pub dispatch_count: u32,
    pub spectrum_rebuild_count: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Eq106InversionTiming {
    pub source_preparation_ms: f64,
    pub spectrum_build_ms: Option<f64>,
    pub target_evaluation_ms: Option<f64>,
    pub gpu_readback_ms: f64,
    pub design_matrix_assembly_ms: f64,
    pub convex_solve_ms: f64,
    pub verification_ms: f64,
    pub total_ms: f64,
    pub dispatch_count: u32,
    pub spectrum_rebuild_count: u32,
    pub matrix_cache_hit: bool,
}

#[derive(Resource, Default)]
pub struct Eq106PerformanceMetrics {
    pub latest: Option<Eq106TimingSample>,
    pub full_inversion_iteration_ms: Option<f64>,
    pub inversion: Option<Eq106InversionTiming>,
}

/// Snapshot-aligned history populated by the dedicated compressed readback
/// channel. Its layout matches the shared gravity sample contract so physics
/// and Jacobi diagnostics can consume it without special-case math.
#[derive(Resource, Default)]
pub struct MmfftCompressedHistory(pub GravitySampleHistory);

#[derive(Resource)]
pub struct MmfftCompressedSource {
    pub bytes: Vec<u8>,
    /// Number of Cartesian samples on each side of one physical grid.
    pub grid_sizes: [u32; 2],
    /// Number of nested grids, ordered finest to coarsest.
    pub level_count: u32,
    /// Half-widths of the nested physical grids (metres).
    pub half_extents: [f32; 2],
    pub total_mass: f32,
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
    /// Fourth method: CPU-preprocessed FFT grids with compressed GPU
    /// interpolation records and a snapshot-tagged readback channel.
    MmfftCompressed,
    /// Fifth method: fixed-depth order-two GPU octree treecode evaluation.
    Fmm,
}

pub const PERFORMANCE_PHASE_SIMULATION_SECONDS: f64 = BENCHMARK_DURATION_SECONDS;
pub const PERFORMANCE_HISTORY_CAPACITY: usize = 180;

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
    /// Performance runs are deliberately normalized to 1x; restore the
    /// user's throughput setting when leaving the comparison view.
    pub return_simulation_acceleration: u32,
    pub fps_history: [VecDeque<f32>; 5],
    /// Jacobi histories map to Radial, Werner, Eq.106, MMFFT, and FMM.
    /// Each series contains only samples emitted by that algorithm.
    pub jacobi_history: [VecDeque<JacobiSample>; 5],
    pub jacobi_last_request_ids: [Option<u64>; 5],
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
            return_simulation_acceleration: MIN_SIMULATION_ACCELERATION,
            fps_history: std::array::from_fn(|_| {
                VecDeque::with_capacity(PERFORMANCE_HISTORY_CAPACITY)
            }),
            jacobi_history: std::array::from_fn(|_| {
                VecDeque::with_capacity(PERFORMANCE_HISTORY_CAPACITY)
            }),
            jacobi_last_request_ids: [None; 5],
            enabled_methods: [true; 5],
            completed_methods: [false; 5],
        }
    }
}

impl PerformanceComparisonState {
    pub fn start(&mut self, return_method: ActiveGravityMethod) {
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

    pub fn restart(&mut self) {
        self.start(self.return_method);
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
            Self::MmfftCompressed => "CPU FFT Grid + GPU Interpolation",
            Self::Fmm => "GPU Order-2 Octree Treecode",
        }
    }

    pub fn planning_label(self) -> &'static str {
        match self {
            Self::CurvedArcEq106 => "Eq.106 full forward",
            Self::MmfftCompressed => "CPU FFT grid + GPU interpolation",
            Self::Fmm => "GPU order-2 octree treecode",
            _ => self.as_str(),
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
    fn benchmark_initial_velocity_is_derived_from_the_speed_factor() {
        let velocity = probe_initial_velocity(PROBE_R0, PROBE_SPEED_FACTOR);
        let requested = Vec3::new(0.023_216, 0.084_329, -0.218_658);

        assert!((PROBE_R0.length() - 620.0).abs() < 1.0e-3);
        assert!((velocity.length() - 0.235_503).abs() < 2.0e-4);
        assert!((velocity - requested).length() < 2.0e-4);
        assert!(velocity.dot(PROBE_R0).abs() < 1.0e-4);
        assert!(velocity.dot(PROBE_ORBIT_NORMAL).abs() < 1.0e-4);
    }

    #[test]
    fn formal_benchmark_uses_ten_millisecond_authoritative_steps() {
        let fixed_update_seconds = 1.0 / 60.0;
        let substep = fixed_update_seconds * TIME_SCALE as f64 / PHYSICS_SUBSTEPS as f64;
        let sample_count = (BENCHMARK_DURATION_SECONDS / substep).round() as usize + 1;

        assert!((substep - BENCHMARK_SAMPLE_INTERVAL_SECONDS).abs() < 1.0e-12);
        assert_eq!(sample_count, 90_167);
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
    fn performance_rotation_handles_all_methods_disabled() {
        let mut state = PerformanceComparisonState {
            enabled_methods: [false; 5],
            ..Default::default()
        };
        state.start(ActiveGravityMethod::RadialAnalytic);
        assert!(!state.measuring);
        assert!(state.pending_method.is_none());
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
