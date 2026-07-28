use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub const G: f32 = 6.6743e-11;
pub const RYUGU_MASS: f32 = 4.5e11;
pub const CASSINI_MASS: f32 = 2500.0;
pub const GRAVITY_EPSILON: f32 = 1.0;
pub const TIME_SCALE: f32 = 6000.0;
pub const ORBIT_HISTORY_LEN: usize = 3500;
pub const VISIBILITY_THRESHOLD: f32 = 250.0;
pub const NORMAL_ARROW_LENGTH: f32 = 35.0;

pub const RYUGU_ROTATION_PERIOD_SECS: f32 = 7.63 * 3600.0;
pub const RYUGU_SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);

pub const DENSITY_EPSILON: f32 = 10.0;
pub const SECTION_CLIP_RADIUS: f32 = 450.0;

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

/// Solved density constant: C = RYUGU_MASS / ∫(1/(r^3+ε^3))dV
#[derive(Resource)]
pub struct DensityC(pub f32);

impl Default for DensityC {
    fn default() -> Self { Self(1.0) }
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

/// Shared channel: render world writes partial sums, main world reads + reduces.
#[derive(Resource, Clone)]
pub struct GravityReadbackChannel(pub Arc<Mutex<Option<Vec<[f32; 4]>>>>);

impl Default for GravityReadbackChannel {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

