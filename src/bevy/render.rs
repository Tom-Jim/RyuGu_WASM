include!("rendering/scene_setup.rs");
include!("rendering/render_systems.rs");
#[cfg(target_arch = "wasm32")]
include!("rendering/mobile_material.rs");
