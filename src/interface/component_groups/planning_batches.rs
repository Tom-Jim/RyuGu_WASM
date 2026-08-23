pub const PLANNING_CANDIDATE_COUNT: u32 = 2_048;
pub const PLANNING_TRAJECTORY_TUBE_RADIUS_METERS: f32 = 15.0;
pub const PLANNING_GPU_TILE_INITIAL_CANDIDATES: u32 = 8;
pub const PLANNING_GPU_TILE_MIN_CANDIDATES: u32 = 8;
pub const PLANNING_GPU_TILE_MAX_CANDIDATES: u32 = 32;
pub const PLANNING_BUILD_CANDIDATES_PER_FRAME: u32 = 8;
pub const PLANNING_MIN_INTERACTIVE_FPS: f64 = 30.0;
pub const PLANNING_TARGET_REQUEST_MS: f64 = 120.0;
pub const PLANNING_MAX_REQUEST_MS: f64 = 240.0;
pub const PLANNING_GPU_UPLOAD_BYTES_PER_FRAME: usize = 1024 * 1024;
pub const PLANNING_REFERENCE_STRIDE: u32 = 32;
pub const PLANNING_REFERENCE_CANDIDATE_STRIDE: u32 = 128;
pub const PLANNING_REFERENCE_MODEL_STRIDE: u32 = 4;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct PlanningCandidateState {
    /// Body-fixed position and sample time.
    pub position_time: [f32; 4],
    /// Body-fixed velocity and transverse distance from the reference arc.
    pub velocity_distance: [f32; 4],
    /// Body-to-world rotation quaternion `(x, y, z, w)`.
    pub body_rotation: [f32; 4],
    /// Candidate index, sample index, Eq.106 segment index, reserved.
    pub identity: [u32; 4],
}

impl PlanningCandidateState {
    pub fn body_position(self) -> Vec3 {
        Vec3::from_array(self.position_time[..3].try_into().expect("three position values"))
    }

    pub fn body_velocity(self) -> Vec3 {
        Vec3::from_array(
            self.velocity_distance[..3]
                .try_into()
                .expect("three velocity values"),
        )
    }

}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct PlanningBasisRecord {
    pub position_volume: [f32; 4],
    pub voxel_index: u32,
    pub _padding: [u32; 3],
}

#[derive(Resource, Clone, Debug, Default)]
pub struct PlanningCandidateBatch {
    pub batch_id: u64,
    pub capture_id: u64,
    pub capture_epoch: u64,
    pub source_hash: u64,
    pub candidate_count: u32,
    pub density_model_count: u32,
    pub samples_per_candidate: u32,
    pub body_radius: f32,
    pub states: Arc<[PlanningCandidateState]>,
    pub gpu_position_bytes: Arc<[u8]>,
    /// Row-major `K x 56` density matrix. Every row has identical total mass.
    pub density_models: Arc<[f32]>,
    pub basis_records: Arc<[PlanningBasisRecord]>,
    pub reference_arc_hash: u64,
    pub candidate_hash: u64,
    pub density_model_hash: u64,
    pub sample_hash: u64,
    pub basis_hash: u64,
}

impl PlanningCandidateBatch {
    pub fn state_count(&self) -> usize {
        self.candidate_count as usize * self.samples_per_candidate as usize
    }

    pub fn workload_identity(&self) -> PlanningWorkloadIdentity {
        PlanningWorkloadIdentity {
            reference_capture_id: self.capture_id,
            reference_capture_epoch: self.capture_epoch,
            source_hash: self.source_hash,
            basis_hash: self.basis_hash,
            reference_arc_hash: self.reference_arc_hash,
            candidate_hash: self.candidate_hash,
            density_model_hash: self.density_model_hash,
            sample_hash: self.sample_hash,
            tolerance_hash: 0x1060_1570_000f_2048,
            candidate_count: self.candidate_count,
            density_model_count: self.density_model_count,
            samples_per_candidate: self.samples_per_candidate,
            outputs: PlanningWorkloadIdentity::REQUIRED_OUTPUTS,
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct PlanningGpuRequest {
    pub request_id: u64,
    pub batch_id: u64,
    pub method: Option<ActiveGravityMethod>,
    pub density_model: u32,
    pub candidate_start: u32,
    pub candidate_count: u32,
    pub warm_repetition: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlanningGpuTiming {
    pub method_preprocess_ms: f64,
    pub command_submission_ms: f64,
    /// Wall time from queue submission through GPU execution, copy, and map.
    /// WebGPU timestamp queries are intentionally not used on browser Metal.
    pub gpu_completion_map_ms: f64,
    pub dispatch_count: u32,
    pub forward_kernel_evaluations: u64,
    pub spectral_element_count: u32,
}

#[derive(Clone, Debug)]
pub struct PlanningGpuPacket {
    pub request: PlanningGpuRequest,
    /// Local target indices for the compact deterministic verification subset.
    pub state_indices: Vec<u32>,
    /// Four rows per target: acceleration/potential and three Jacobian columns.
    pub rows: Vec<[f32; 4]>,
    /// One GPU-reduced row per candidate: field separation, baseline energy,
    /// minimum altitude, and Jacobian energy.
    pub candidate_metrics: Vec<[f32; 4]>,
    pub readback_valid: bool,
    pub timing: PlanningGpuTiming,
    pub backend: PlanningExecutionBackend,
}

#[derive(Resource, Default)]
pub struct PlanningGpuResult(pub Option<PlanningGpuPacket>);

#[derive(Resource, Clone)]
pub struct PlanningGpuReadbackChannel {
    pub data: Arc<Mutex<Option<PlanningGpuPacket>>>,
    pub in_flight: Arc<AtomicBool>,
}

impl Default for PlanningGpuReadbackChannel {
    fn default() -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct PlanningMethodPayload {
    pub request_id: u64,
    pub method: Option<ActiveGravityMethod>,
    pub density_model: u32,
    pub primary: Arc<[u8]>,
    pub secondary: Arc<[u8]>,
    pub item_count: u32,
    pub secondary_count: u32,
    pub maximum_level: u32,
    pub grid_sizes: [u32; 2],
    pub half_extents: [f32; 2],
    pub total_mass: f32,
    pub preparation_ms: f64,
}
