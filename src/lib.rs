#![allow(clippy::too_many_arguments, clippy::type_complexity)]

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
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
#[cfg(not(target_arch = "wasm32"))]
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};
use bevy_obj::ObjPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use components::{
    ActiveGravityMethod, CameraMode, DensityC, GravityAcceleration, GravityBlendFactor,
    GravityPotential, JacobiHistory, ProbeInitialConditions, ShowNormals, ShowSection,
    SimulationAcceleration, SimulationClock,
};
use std::time::Duration;
use systems::{
    compute_pipeline::NormalsComputePlugin,
    energy::{record_probe_jacobi_system, setup_jacobi_chart, update_jacobi_chart_system},
    gravity_pipeline::GravityComputePlugin,
    physics::{physics_system, ryugu_rotation_system},
    render::{
        camera_follow_system, camera_switch_system, render_gizmos_system, render_section_system,
        section_alpha_system, setup_scene, setup_ui,
    },
    scale::{build_topology_system, normalize_model_scale_system},
    ui::{
        fps_update_system, method_toggle_system, normal_toggle_system, probe_slider_system,
        probe_slider_visual_system, section_toggle_system, setup_fps_ui, setup_probe_controls,
        setup_simulation_acceleration_control, simulation_acceleration_slider_system,
        simulation_acceleration_slider_visual_system, update_hint_on_mode_change,
    },
    werner_pipeline::WernerComputePlugin,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Browser capability check and fallback warning.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = r#"
    export function has_webgpu() {
        return typeof navigator !== "undefined" && navigator.gpu !== undefined;
    }

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
    #[wasm_bindgen(js_name = has_webgpu)]
    fn browser_has_webgpu() -> bool;

    fn show_webgpu_warning();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    let has_webgpu = {
        #[cfg(target_arch = "wasm32")]
        {
            browser_has_webgpu()
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
        .init_resource::<ActiveGravityMethod>()
        .init_resource::<GravityAcceleration>()
        .init_resource::<GravityPotential>()
        .init_resource::<GravityBlendFactor>()
        .init_resource::<JacobiHistory>()
        .init_resource::<SimulationClock>()
        .init_resource::<SimulationAcceleration>()
        .init_resource::<ProbeInitialConditions>()
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Reactive {
                wait: Duration::from_secs_f64(1.0 / 60.0),
                react_to_device_events: false,
                react_to_user_events: false,
                react_to_window_events: false,
            },
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 30.0)),
        })
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
                        present_mode: PresentMode::AutoVsync,
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

    // Native builds use a precise sleep/spin limiter. Browsers already own the
    // animation clock, so the Winit 60 Hz reactive mode above is the cap there.
    #[cfg(not(target_arch = "wasm32"))]
    {
        app.add_plugins(FramepacePlugin);
        app.insert_resource(FramepaceSettings {
            limiter: Limiter::from_framerate(60.0),
        });
    }

    if has_webgpu {
        app.add_plugins(NormalsComputePlugin);
        app.add_plugins(GravityComputePlugin);
        app.add_plugins(WernerComputePlugin);
    }

    app.add_systems(
        Startup,
        (
            setup_scene,
            setup_ui,
            setup_fps_ui,
            setup_probe_controls,
            setup_simulation_acceleration_control,
            setup_jacobi_chart,
        ),
    )
    .add_systems(
        Update,
        (
            normalize_model_scale_system,
            build_topology_system,
            camera_switch_system,
            normal_toggle_system,
            section_toggle_system,
            method_toggle_system,
            probe_slider_system,
            probe_slider_visual_system,
            simulation_acceleration_slider_system,
            simulation_acceleration_slider_visual_system,
            update_hint_on_mode_change,
            camera_follow_system,
            fps_update_system,
            update_jacobi_chart_system,
            section_alpha_system,
            render_gizmos_system,
            render_section_system,
        )
            .chain(),
    )
    .add_systems(
        FixedUpdate,
        (
            physics_system,
            ryugu_rotation_system,
            record_probe_jacobi_system,
        )
            .chain(),
    )
    .run();
}

#[cfg(test)]
mod shader_tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    fn validate_wgsl(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL parsing failed");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("WGSL validation failed");
    }

    #[test]
    fn radial_gravity_shader_is_valid() {
        validate_wgsl(include_str!("../assets/shaders/gravity.wgsl"));
    }

    #[test]
    fn werner_shader_is_valid() {
        validate_wgsl(include_str!("../assets/shaders/werner_gravity.wgsl"));
    }
}
