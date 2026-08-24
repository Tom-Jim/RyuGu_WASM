pub const PLANNING_CANDIDATE_COUNT: u32 = 2_048;
pub const PLANNING_FIRST_CANDIDATE_COUNT: u32 = 32;
pub const PLANNING_TRAJECTORY_TUBE_RADIUS_METERS: f32 = 15.0;
pub const PLANNING_GPU_TILE_INITIAL_CANDIDATES: u32 = 8;
pub const PLANNING_GPU_TILE_MIN_CANDIDATES: u32 = 8;
pub const PLANNING_GPU_TILE_MAX_CANDIDATES: u32 = 16;
// Stress uses the same candidate-tile range for all three GPU backends.  A
// method-specific tile width changes request counts and hides dispatch/readback
// overhead inside what is meant to be a shared-workload robustness run.
pub const PLANNING_GENERIC_TILE_INITIAL_CANDIDATES: u32 = 8;
pub const PLANNING_GENERIC_TILE_MIN_CANDIDATES: u32 = 8;
pub const PLANNING_GENERIC_TILE_MAX_CANDIDATES: u32 = 16;
pub const PLANNING_BUILD_CANDIDATES_PER_FRAME: u32 = 8;
pub const PLANNING_MIN_INTERACTIVE_FPS: f64 = 57.0;
pub const PLANNING_TARGET_REQUEST_MS: f64 = 18.0;
pub const PLANNING_MAX_REQUEST_MS: f64 = 34.0;
pub const PLANNING_MAX_RECENT_FRAME_MS: f64 = 18.5;
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
    pub source_count: u32,
    pub candidate_count: u32,
    pub density_model_count: u32,
    pub samples_per_candidate: u32,
    pub body_radius: f32,
    /// Exact enclosing radius of the spatially refined quadrature sources.
    /// Eq.106 convergence bounds must use this value, not the radius of the
    /// pre-refinement 1024-source aggregate.
    pub eq106_source_radius: f32,
    /// Frozen centre arc in body-fixed coordinates. Eq.106 builds one
    /// canonical spectrum from this arc and shares it across every candidate
    /// trajectory in the certified tube.
    pub reference_states: Arc<[PlanningCandidateState]>,
    pub states: Arc<[PlanningCandidateState]>,
    pub gpu_position_bytes: Arc<[u8]>,
    /// Row-major `K x 56` density matrix. Every row has identical total mass.
    pub density_models: Arc<[f32]>,
    /// Auditable f64 mass reconstructed from every randomized density row.
    pub density_model_masses: Arc<[f64]>,
    pub density_seed: u64,
    pub target_mass: f64,
    pub basis_records: Arc<[PlanningBasisRecord]>,
    /// Eq.106 geometry-only source records `(x,y,z,volume)`. Unlike the
    /// per-method payload this buffer is invariant across density models.
    pub eq106_volume_source_bytes: Arc<[u8]>,
    /// Contiguous `(start,count)` ranges for the 56 density voxels.
    pub eq106_voxel_source_ranges: Arc<[[u32; 2]]>,
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

    pub fn density_mass_is_conserved(&self) -> bool {
        self.density_seed != 0
            && self.target_mass.is_finite()
            && self.target_mass > 0.0
            && self.density_model_masses.len() == self.density_model_count as usize
            && self.density_model_masses.iter().all(|mass| {
                mass.is_finite()
                    && ((mass - self.target_mass) / self.target_mass).abs() <= 2.0e-7
            })
    }

    pub fn workload_identity(&self) -> PlanningWorkloadIdentity {
        PlanningWorkloadIdentity {
            reference_capture_id: self.capture_id,
            reference_capture_epoch: self.capture_epoch,
            source_hash: self.source_hash,
            source_count: self.source_count,
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
    /// Eq.106 only: enable independent certificate and five-step self-FD.
    /// The raw path still computes the complete field and 3x3 Jacobian.
    pub eq106_certified: bool,
    /// First and Interactive Stress both use the fairness-oriented fixed
    /// schedule; the latter remains interactive through progress rendering.
    pub compute_benchmark: bool,
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
    /// Eq.106 analytic transverse Jacobian versus a same-field central finite
    /// difference. Other backends report zero because this diagnostic is
    /// specific to the cached Taylor reconstruction.
    pub gradient_self_fd_relative_error: f32,
}

#[derive(Clone, Debug)]
pub struct PlanningGpuPacket {
    pub request: PlanningGpuRequest,
    /// Local target indices for the compact deterministic verification subset.
    pub state_indices: Vec<u32>,
    /// Four rows per target: acceleration/potential and three Jacobian columns.
    pub rows: Vec<[f32; 4]>,
    /// Unmodified method output before certificate failures are converted to
    /// common full-penalty rows.
    pub raw_rows: Vec<[f32; 4]>,
    /// Taylor, imaginary, spectral, transverse, self-FD, and non-finite
    /// rejection counts. Non-Eq.106 methods leave these at zero.
    pub rejection_counts: [u64; 6],
    /// Validation rows charged the full penalty after candidate-level gating.
    pub rejected_sample_count: u64,
    /// Maximum self-FD mismatch at 0.25, 0.5, 1, 2, and 4 metres.
    pub self_fd_step_maxima: [f32; 5],
    /// First certificate rejection as density, global candidate, sample,
    /// Taylor segment, and reason index. Self-FD warnings do not reject.
    pub first_rejection: Option<[u32; 5]>,
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
    /// Program-lifetime setup excluded symmetrically from repeated-workload
    /// totals (for example FFT plans/static Newton kernel or Eq operator table).
    pub one_time_preparation_ms: f64,
    pub preparation_ms: f64,
}
