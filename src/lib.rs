#![allow(clippy::too_many_arguments, clippy::type_complexity)]

#[path = "bevy/mod.rs"]
mod bevy_app;
mod cpu;
mod gpu;
mod interface;
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
use bevy_app::{
    energy::{
        record_probe_jacobi_system, setup_eq106_residual_chart, setup_jacobi_chart,
        update_eq106_residual_chart_system, update_jacobi_chart_system,
    },
    render::{
        camera_follow_system, camera_switch_system, capture_trajectory_inversion_system,
        render_gizmos_system, render_section_system, section_alpha_system, setup_scene, setup_ui,
    },
    scale::{build_topology_system, normalize_model_scale_system},
    ui::{
        clear_runtime_error_on_probe_change, density_inversion_timing_ui_system, fps_update_system,
        method_toggle_system, normal_toggle_system, performance_button_system,
        performance_comparison_system, performance_method_checkbox_system,
        planning_batch_evaluator_system, planning_comparison_control_system,
        probe_collision_system, probe_crash_overlay_system, probe_orbit_preset_style_system,
        probe_orbit_preset_system, probe_slider_system, probe_slider_visual_system,
        reset_after_probe_crash_scene_system, reset_after_probe_crash_state_system,
        reset_inversion_on_method_change, runtime_error_overlay_system, runtime_error_reset_system,
        section_toggle_system, setup_density_inversion_timing_panel, setup_fps_ui,
        setup_performance_chart_segments, setup_performance_controls, setup_probe_controls,
        setup_probe_crash_overlay, setup_runtime_error_overlay,
        setup_simulation_acceleration_control, setup_trajectory_inversion_controls,
        simulation_acceleration_slider_system, simulation_acceleration_slider_visual_system,
        trajectory_inversion_input_system, trajectory_inversion_ui_system,
        update_gpu_memory_estimate_system, update_hint_on_mode_change,
        update_planning_results_from_inversion_system, update_ui_scale_system,
    },
};
#[cfg(not(target_arch = "wasm32"))]
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};
use bevy_panorbit_camera::PanOrbitCameraPlugin;
pub use cpu::eq106_operator::{PsiOperatorTable, ToroidalOperatorTensor};
pub use cpu::eq106_reference as eq106;
use cpu::{
    curved_arc::{
        CurvedArcPlannerState, CurvedArcResidualHistory, build_aggregated_gravity_source_system,
        monitor_curved_arc_system,
    },
    eq106_operator::build_eq106_operator_tensor_system,
    inversion::{convex_optimization_system, start_density_inversion_system},
    physics::{physics_system, ryugu_rotation_system},
};
use gpu::{
    eq106::Eq106GpuComputePlugin,
    fmm::FmmComputePlugin,
    mmfft::MmfftCompressedComputePlugin,
    normals::NormalsComputePlugin,
    radial::{GravityComputePlugin, build_radial_gravity_source_system},
    werner::WernerComputePlugin,
};
use interface::components::{
    ActiveGravityMethod, CameraMode, DensityC, DensitySensitivityCaches, DisplayRotation,
    GpuMemoryEstimate, GravityAcceleration, GravityBenchmarkTrajectory, GravityBlendFactor,
    GravityPotential, GravityRuntimeError, JacobiHistory, PerformanceComparisonState,
    PlanningComparisonState, ProbeCrashResetRequest, ProbeCrashState, ProbeInitialConditions,
    ShowNormals, ShowSection, SimulationAcceleration, SimulationClock, TrajectoryInversionState,
};
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Browser capability check. WebGPU is mandatory because no alternate force
// model is allowed to run when the GPU evaluator is unavailable.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = r#"
    export function has_webgpu() {
        return typeof navigator !== "undefined" && navigator.gpu !== undefined;
    }

    export function show_webgpu_error() {
        const div = document.createElement("div");
        div.style.cssText = "position:absolute;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.9);color:white;display:flex;flex-direction:column;justify-content:center;align-items:center;z-index:99999;font-family:sans-serif;text-align:center;padding:20px;box-sizing:border-box;";
        div.innerHTML = `
            <h1 style="color:#ff4444;margin-bottom:20px;font-size:2.5rem;">⚠️ WebGPU Not Supported</h1>
            <p style="font-size:1.2rem;max-width:700px;line-height:1.6;">
                This simulation features a custom <b>GPU-accelerated Gravity architecture</b>.
            </p>
            <p style="font-size:1.1rem;max-width:700px;line-height:1.6;color:#ccc;margin-top:15px;">
                No valid GPU gravity evaluator is available. The simulation is stopped because it never substitutes a different physical model.
            </p>
            <p style="font-size:0.95rem;color:#888;margin-top:30px;">
                <i>* Please open this page in a modern browser (e.g., Chrome 113+, Edge) or enable WebGPU flags in Safari to evaluate the actual GPU rendering performance.</i>
            </p>
        `;
        document.body.appendChild(div);
    }

    export function set_display_rotation(quarter_turn) {
        window.setRyuguDisplayRotation?.(quarter_turn);
    }
"#)]
extern "C" {
    #[wasm_bindgen(js_name = has_webgpu)]
    fn browser_has_webgpu() -> bool;

    fn show_webgpu_error();

    #[wasm_bindgen(js_name = set_display_rotation)]
    fn browser_set_display_rotation(quarter_turn: u8);
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn set_display_rotation(quarter_turn: u8) {
    browser_set_display_rotation(quarter_turn);
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn set_display_rotation(_quarter_turn: u8) {}

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

    // WebGPU is a hard requirement. Do not create a GL app that could hide a
    // missing gravity evaluator behind an alternate execution path.
    #[cfg(target_arch = "wasm32")]
    if !has_webgpu {
        show_webgpu_error();
        return;
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
        unreachable!("WebGPU absence is handled before app construction");
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
        .init_resource::<GpuMemoryEstimate>()
        .init_resource::<GravityBlendFactor>()
        .init_resource::<GravityRuntimeError>()
        .init_resource::<JacobiHistory>()
        .init_resource::<SimulationClock>()
        .init_resource::<TrajectoryInversionState>()
        .init_resource::<DensitySensitivityCaches>()
        .init_resource::<GravityBenchmarkTrajectory>()
        .init_resource::<SimulationAcceleration>()
        .init_resource::<ProbeInitialConditions>()
        .init_resource::<PlanningComparisonState>()
        .init_resource::<ProbeCrashState>()
        .init_resource::<ProbeCrashResetRequest>()
        .init_resource::<DisplayRotation>()
        .init_resource::<PerformanceComparisonState>()
        .init_resource::<CurvedArcPlannerState>()
        .init_resource::<CurvedArcResidualHistory>()
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
        app.add_plugins(Eq106GpuComputePlugin);
        // MMFFT+compression is the fourth GPU integration slot. Its packed
        // source buffer and tiled reduction are built once and evaluated in
        // the render-world compute pass.
        app.add_plugins(MmfftCompressedComputePlugin);
        app.add_plugins(FmmComputePlugin);
    }

    app.add_systems(
        Startup,
        (
            setup_scene,
            setup_ui,
            setup_fps_ui,
            setup_runtime_error_overlay,
            setup_probe_crash_overlay,
            setup_probe_controls,
            setup_simulation_acceleration_control,
            setup_jacobi_chart,
            setup_eq106_residual_chart,
        ),
    )
    .add_systems(
        Startup,
        (
            setup_performance_controls,
            setup_trajectory_inversion_controls,
            setup_density_inversion_timing_panel,
            setup_performance_chart_segments,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            normalize_model_scale_system,
            build_topology_system,
            camera_switch_system,
            normal_toggle_system,
        )
            .chain(),
    )
    .add_systems(
        Update,
        build_aggregated_gravity_source_system.after(build_radial_gravity_source_system),
    )
    .add_systems(
        Update,
        build_eq106_operator_tensor_system.after(build_aggregated_gravity_source_system),
    )
    .add_systems(Update, section_toggle_system)
    .add_systems(
        Update,
        (
            performance_button_system,
            performance_method_checkbox_system,
            method_toggle_system,
            reset_inversion_on_method_change,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            capture_trajectory_inversion_system,
            trajectory_inversion_input_system,
            start_density_inversion_system,
            convex_optimization_system,
            trajectory_inversion_ui_system,
            planning_comparison_control_system,
            update_planning_results_from_inversion_system,
            planning_batch_evaluator_system,
            density_inversion_timing_ui_system,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            probe_orbit_preset_system,
            probe_slider_system,
            runtime_error_reset_system,
            clear_runtime_error_on_probe_change,
            probe_slider_visual_system,
            probe_orbit_preset_style_system,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            probe_crash_overlay_system,
            reset_after_probe_crash_scene_system,
            reset_after_probe_crash_state_system,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            update_ui_scale_system,
            runtime_error_overlay_system,
            simulation_acceleration_slider_system,
            simulation_acceleration_slider_visual_system,
            update_hint_on_mode_change,
            performance_comparison_system,
            camera_follow_system,
            update_gpu_memory_estimate_system,
            fps_update_system,
            update_jacobi_chart_system,
            update_eq106_residual_chart_system,
            section_alpha_system,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (render_gizmos_system, render_section_system).chain(),
    )
    .add_systems(
        FixedUpdate,
        (
            monitor_curved_arc_system,
            physics_system,
            probe_collision_system,
            ryugu_rotation_system,
            record_probe_jacobi_system,
        )
            .chain(),
    )
    .run();
}

pub use cpu::benchmark::benchmark_gravity_algorithms;
