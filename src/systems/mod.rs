//! Runtime systems grouped by responsibility.
//!
//! The public module names below intentionally remain stable because the ECS
//! schedule and tests refer to them. `#[path]` keeps those API names while
//! placing the implementations in functional folders for maintainability.

#[path = "gpu/normals.rs"]
pub mod compute_pipeline;
#[path = "gravity/curved_arc.rs"]
pub mod curved_arc;
#[path = "simulation/energy.rs"]
pub mod energy;
#[path = "gravity/eq106_reference.rs"]
pub mod eq106;
#[path = "gravity/eq106_gpu.rs"]
pub mod eq106_gpu_pipeline;
#[path = "gravity/eq106_operator.rs"]
pub mod eq106_operator;
#[path = "gravity/fmm.rs"]
pub mod fmm_pipeline;
#[path = "gravity/radial.rs"]
pub mod gravity_pipeline;
#[path = "simulation/inversion.rs"]
pub mod inversion;
#[path = "gravity/mmfft.rs"]
pub mod mmfft_pipeline;
#[path = "simulation/physics.rs"]
pub mod physics;
#[path = "presentation/render.rs"]
pub mod render;
#[path = "model/scale.rs"]
pub mod scale;
#[path = "presentation/ui.rs"]
pub mod ui;
#[path = "gravity/werner.rs"]
pub mod werner_pipeline;
