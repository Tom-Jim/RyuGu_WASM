
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
impl FmmReadbackChannel {
    pub fn reset_after_device_loss(&self) {
        if let Ok(mut data) = self.data.try_lock() { data.take(); }
        self.in_flight.store(false, Ordering::Release);
    }
}


impl Default for FrequencyDomainGpuReadbackChannel {
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
pub struct FrequencyDomainTimingSample {
    pub spectrum_build_ms: Option<f64>,
    pub target_evaluation_ms: Option<f64>,
    pub cpu_readback_wait_ms: f64,
    pub target_count: u32,
    pub dispatch_count: u32,
    pub spectrum_rebuild_count: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrequencyDomainInversionTiming {
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
}

#[derive(Resource, Default)]
pub struct FrequencyDomainPerformanceMetrics {
    pub latest: Option<FrequencyDomainTimingSample>,
    pub full_inversion_iteration_ms: Option<f64>,
    pub inversion: Option<FrequencyDomainInversionTiming>,
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
    /// Per-level scale used by the packed binary16 potential samples.
    pub grid_scales: [f32; 2],
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
impl MmfftReadbackChannel {
    pub fn reset_after_device_loss(&self) {
        if let Ok(mut data) = self.data.try_lock() { data.take(); }
        self.in_flight.store(false, Ordering::Release);
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
impl GravityReadbackChannel {
    pub fn reset_after_device_loss(&self) {
        if let Ok(mut data) = self.data.try_lock() { data.take(); }
        self.in_flight.store(false, Ordering::Release);
    }
}

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum ActiveGravityMethod {
    #[default]
    RadialAnalytic,
    HomogeneousWerner,
    /// Known-trajectory reciprocal-space gravity evaluation.
    FrequencyDomain,
    /// Fourth method: CPU-preprocessed FFT grids with scale-normalized packed
    /// binary16 GPU potential samples and a snapshot-tagged readback channel.
    MmfftCompressed,
    /// Fifth method: order-two source/target-cell FMM with exact P2P near field.
    Fmm,
}

pub const PERFORMANCE_PHASE_SIMULATION_SECONDS: f64 = BENCHMARK_DURATION_SECONDS;
pub const PERFORMANCE_HISTORY_CAPACITY: usize = 180;

/// Quarter-turn rotation applied to the complete browser display frame.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub struct DisplayRotation(pub u8);

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl DisplayRotation {
    pub fn advance(&mut self) -> u8 {
        self.0 = (self.0 + 1) % 4;
        self.0
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
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
    /// Method-native stability histories. Pointwise methods publish Jacobi
    /// constants; frequency-domain publishes a repeated transform norm.
    pub diagnostic_history: [VecDeque<PerformanceDiagnosticSample>; 5],
    pub diagnostic_last_ids: [Option<u64>; 5],
    /// Algorithms included in the performance rotation. The five entries map
    /// to Radial, Werner, Frequency-domain algorithm, MMFFT, and FMM respectively.
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
            diagnostic_history: std::array::from_fn(|_| {
                VecDeque::with_capacity(PERFORMANCE_HISTORY_CAPACITY)
            }),
            diagnostic_last_ids: [None; 5],
            enabled_methods: [true; 5],
            completed_methods: [false; 5],
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
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
        self.diagnostic_last_ids = [None; 5];
        self.return_method = return_method;
        for history in &mut self.fps_history {
            history.clear();
        }
        for history in &mut self.diagnostic_history {
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
            Self::FrequencyDomain => 2,
            Self::MmfftCompressed => 3,
            Self::Fmm => 4,
        }
    }

    pub fn from_performance_index(index: usize) -> Self {
        match index {
            0 => Self::RadialAnalytic,
            1 => Self::HomogeneousWerner,
            2 => Self::FrequencyDomain,
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
            Self::FrequencyDomain => "GPU Frequency-domain Algorithm",
            Self::MmfftCompressed => "Packed MMFFT + GPU Interpolation",
            Self::Fmm => "GPU Order-2 Target-Cell FMM",
        }
    }

    pub fn planning_label(self) -> &'static str {
        match self {
            Self::FrequencyDomain => "GPU Frequency-domain Algorithm reciprocal evaluation",
            Self::MmfftCompressed => "GPU FFT 56-basis convolution + quintic interpolation",
            Self::Fmm => "GPU order-2 FMM + 56 density bases",
            _ => self.as_str(),
        }
    }
}
