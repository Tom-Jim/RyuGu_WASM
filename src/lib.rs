mod components;
mod systems;
mod topology;
mod welding;

use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::render::render_resource::WgpuLimits;
use bevy::render::{
    RenderPlugin,
    settings::{Backends, RenderCreation, WgpuSettings},
};
use bevy_obj::ObjPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use components::{
    CameraMode, DensityC, GravityAcceleration, GravityBlendFactor, ShowNormals, ShowSection,
};
use systems::{
    compute_pipeline::NormalsComputePlugin,
    gravity_pipeline::GravityComputePlugin,
    physics::{physics_system, ryugu_rotation_system},
    render::{
        camera_follow_system, camera_switch_system, render_gizmos_system, render_section_system,
        section_alpha_system, setup_scene, setup_ui,
    },
    scale::{build_topology_system, normalize_model_scale_system},
    ui::{
        fps_update_system, normal_toggle_system, section_toggle_system, setup_fps_ui,
        update_hint_on_mode_change,
    },
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// 1. sniff navigator.gpu
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = navigator, js_name = gpu)]
    static GPU: JsValue;
}

// 2. DOM windows insert
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = r#"
    export function show_webgpu_warning() {
        const div = document.createElement("div");
        div.style.cssText = "position:absolute;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.9);color:white;display:flex;flex-direction:column;justify-content:center;align-items:center;z-index:99999;font-family:sans-serif;text-align:center;padding:20px;box-sizing:border-box;";
        div.innerHTML = `
            <h1 style="color:#ff4444;margin-bottom:20px;font-size:2.5rem;">⚠️ WebGPU Not Supported</h1>
            <p style="font-size:1.2rem;max-width:700px;line-height:1.6;">
                This simulation features a custom <b>GPU-accelerated Gravity architecture</b>.
            </p>
            <p style="font-size:1.1rem;max-width:700px;line-height:1.6;color:#ccc;margin-top:15px;">
                Your current browser does not support WebGPU. The simulation has fallen back to a standard Newtonian CPU calculation,
                <b style="color:#ffaa00;">which bypasses the core parallel compute optimization of this research.</b>
            </p>
            <p style="font-size:0.95rem;color:#888;margin-top:30px;">
                <i>* Please open this page in a modern browser (e.g., Chrome 113+, Edge) or enable WebGPU flags in Safari to evaluate the actual GPU rendering performance.</i>
            </p>
            <button onclick="this.parentElement.style.display='none'" style="margin-top:40px;padding:12px 24px;font-size:1rem;cursor:pointer;background:#444;color:white;border:1px solid #666;border-radius:5px;transition:0.2s;">
                Acknowledge & View CPU Fallback
            </button>
        `;
        document.body.appendChild(div);
    }
"#)]
extern "C" {
    fn show_webgpu_warning();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    let has_webgpu = {
        #[cfg(target_arch = "wasm32")]
        {
            !GPU.is_undefined()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            true
        }
    };

    // warning if no WebGPU
    #[cfg(target_arch = "wasm32")]
    if !has_webgpu {
        show_webgpu_warning();
    }

    let (backends, limits) = if has_webgpu {
        (
            Backends::BROWSER_WEBGPU,
            WgpuLimits {
                max_storage_buffers_per_shader_stage: 8,
                max_compute_workgroups_per_dimension: 65535,
                ..WgpuLimits::default()
            },
        )
    } else {
        (Backends::GL, WgpuLimits::downlevel_webgl2_defaults())
    };

    let mut app = App::new();

    app.insert_resource(ClearColor(Color::srgb(0.01, 0.01, 0.03)))
        .init_resource::<CameraMode>()
        .init_resource::<ShowNormals>()
        .init_resource::<ShowSection>()
        .init_resource::<DensityC>()
        .init_resource::<GravityAcceleration>()
        .init_resource::<GravityBlendFactor>()
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
                        backends: Some(backends),
                        limits,
                        ..default()
                    })),
                    ..default()
                }),
        )
        .add_plugins(ObjPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default());

    if has_webgpu {
        app.add_plugins(NormalsComputePlugin);
        app.add_plugins(GravityComputePlugin);
    }

    app.add_systems(Startup, (setup_scene, setup_ui, setup_fps_ui))
        .add_systems(
            Update,
            (
                normalize_model_scale_system,
                build_topology_system,
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
