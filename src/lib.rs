mod components;
mod topology;
mod welding;
mod systems;

use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::render::{
    RenderPlugin,
    settings::{Backends, RenderCreation, WgpuSettings},
};
use bevy::render::render_resource::WgpuLimits;
use bevy_obj::ObjPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use components::{CameraMode, DensityC, GravityAcceleration, ShowNormals, ShowSection};
use systems::{
    compute_normals::compute_asteroid_normals_system,
    compute_pipeline::NormalsComputePlugin,
    gravity_pipeline::{GravityComputePlugin, build_gravity_voxels_system},
    physics::{physics_system, ryugu_rotation_system},
    render::{
        camera_follow_system, camera_switch_system, render_gizmos_system,
        render_section_system, section_alpha_system, setup_scene, setup_ui,
    },
    scale::{build_topology_system, normalize_model_scale_system},
    ui::{fps_update_system, normal_toggle_system, section_toggle_system, setup_fps_ui, update_hint_on_mode_change},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.01, 0.01, 0.03)))
        .init_resource::<CameraMode>()
        .init_resource::<ShowNormals>()
        .init_resource::<ShowSection>()
        .init_resource::<DensityC>()
        .init_resource::<GravityAcceleration>()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        canvas: Some("#bevy".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        backends: Some(Backends::BROWSER_WEBGPU),
                        limits: WgpuLimits {
                            max_storage_buffers_per_shader_stage: 8,
                            max_compute_workgroups_per_dimension: 65535,
                            ..WgpuLimits::default()
                        },
                        ..default()
                    })),
                    ..default()
                }),
        )
        .add_plugins(ObjPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(NormalsComputePlugin)
        .add_plugins(GravityComputePlugin)
        .add_systems(Startup, (setup_scene, setup_ui, setup_fps_ui))
        .add_systems(
            Update,
            (
                normalize_model_scale_system,
                build_topology_system,
                build_gravity_voxels_system,
                compute_asteroid_normals_system,
                physics_system,
                ryugu_rotation_system,
                camera_switch_system,
                normal_toggle_system,
                section_toggle_system,
                update_hint_on_mode_change,
                camera_follow_system,
                fps_update_system,
                section_alpha_system,
                render_gizmos_system,
                render_section_system,
            )
                .chain(),
        )
        .run();
}
