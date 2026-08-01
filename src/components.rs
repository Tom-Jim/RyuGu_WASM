use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
pub const G: f32 = 6.6743e-11;
pub const RYUGU_MASS: f32 = 4.5e11;
pub const CASSINI_MASS: f32 = 2500.0;
pub const GRAVITY_EPSILON: f32 = 1.0;
pub const TIME_SCALE: f32 = 20000.0;
pub const ORBIT_HISTORY_LEN: usize = 27500;
pub const VISIBILITY_THRESHOLD: f32 = 250.0;
pub const NORMAL_ARROW_LENGTH: f32 = 35.0;

pub const RYUGU_ROTATION_PERIOD_SECS: f32 = 7.63 * 3600.0;
pub const RYUGU_SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);

pub const DENSITY_EPSILON: f32 = 10.0;
pub const SECTION_CLIP_RADIUS: f32 = 450.0;
pub const PROBE_R0: Vec3 = Vec3::new(-1000.0, 200.0, 100.0);
pub static PROBE_V_INIT: LazyLock<Vec3> = LazyLock::new(|| {
    let r0 = PROBE_R0;
    let speed = 1.253 * (G * RYUGU_MASS / r0.length()).sqrt();
    r0.normalize().cross(Vec3::Y).normalize() * speed
});
#[derive(Component)]
pub struct TargetSize(pub f32);
#[derive(Component)]
pub struct ScaleNormalized;
#[derive(Component)]
pub struct TopologyBuilt;
#[derive(Component)]
pub struct RyuguMarker;
#[derive(Component)]
pub struct CassiniMarker; //GPU acceleration
#[derive(Component)]
pub struct CassiniWernerMarker; //Werner 1996 GPU acceleration
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

/// Packed voxel data built once from the asteroid mesh, shared to render world.
#[derive(Resource)]
pub struct GravVoxelSource {
    pub bytes: Vec<u8>,
    pub count: u32,
}

/// Latest GPU-computed gravity acceleration for Cassini (world space).
#[derive(Resource, Default)]
pub struct GravityAcceleration(pub Vec3);

/// Blend weight for smooth GPU gravity warm-up transition.
/// 0.0 = 100% Newtonian fallback, 1.0 = 100% GPU-computed.
/// Increments by 1/GRAVITY_BLEND_FRAMES per frame once the first valid GPU result
/// arrives, reaching 1.0 after GRAVITY_BLEND_FRAMES frames.
#[derive(Resource, Default)]
pub struct GravityBlendFactor(pub f32);

pub const GRAVITY_BLEND_FRAMES: f32 = 60.0;

/// Shared channel: render world writes partial sums, main world reads + reduces.
#[derive(Resource, Clone)]
pub struct GravityReadbackChannel(pub Arc<Mutex<Option<Vec<[f32; 4]>>>>);

impl Default for GravityReadbackChannel {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub enum ActiveGravityMethod {
    #[default]
    VoxelStehfest, // Cyan trajectory: GPU voxel NILT (default)
    DecomposedWerner, // Red trajectory: decomposed-Werner polyhedron kernel
}

impl ActiveGravityMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VoxelStehfest => "GPU Voxel (Stehfest)",
            Self::DecomposedWerner => "Decomposed Werner (1/R)",
        }
    }
}
