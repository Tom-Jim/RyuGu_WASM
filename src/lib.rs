#![allow(clippy::too_many_arguments, clippy::type_complexity)]

#[path = "bevy/mod.rs"]
mod bevy_app;
mod cpu;
mod gpu;
#[cfg(target_arch = "wasm32")]
mod html;
mod interface;
mod wgsl;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::log::{Level, LogPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::WgpuLimits;
use bevy::render::{
    RenderPlugin,
    settings::{Backends, InstanceFlags, RenderCreation, WgpuSettings},
};
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_app::{
    backend::{
        apply_probe_input_system, clear_gpu_histories_on_method_change,
        clear_inversion_request_on_probe_change, clear_runtime_error_on_probe_change,
        method_selection_system, performance_comparison_system, planning_batch_evaluator_system,
        probe_collision_system, reset_after_probe_crash_scene_system,
        reset_after_probe_crash_state_system, reset_inversion_on_method_change,
        update_gpu_memory_estimate_system, update_planning_results_from_inversion_system,
    },
    energy::record_probe_jacobi_system,
    render::{
        camera_follow_system, capture_trajectory_inversion_system, render_gizmos_system,
        render_section_system, section_alpha_system, setup_scene,
    },
    scale::{build_topology_system, normalize_model_scale_system},
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
    volterra::VolterraPropagationStatus,
};
use gpu::{
    eq106::Eq106GpuComputePlugin,
    fmm::FmmComputePlugin,
    mmfft::MmfftCompressedComputePlugin,
    normals::NormalsComputePlugin,
    planning::PlanningGpuComputePlugin,
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
use wgsl::WgslPlugin;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum UiComputeOrdering {
    BrowserActions,
    MethodReset,
    InversionAndPlanning,
}

#[cfg(target_arch = "wasm32")]
use html::{browser_ui_action_system, browser_ui_publish_system};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Browser capability check. WebGPU is mandatory because no alternate force
// model is allowed to run when the GPU evaluator is unavailable.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = r#"
    const ryuguFrameIntervals = new Float64Array(60);
    let ryuguFrameIntervalCount = 0;
    let ryuguFrameIntervalCursor = 0;
    let ryuguLastAnimationFrame = 0;

    function record_ryugu_animation_frame(timestamp) {
        if (ryuguLastAnimationFrame > 0) {
            ryuguFrameIntervals[ryuguFrameIntervalCursor] = timestamp - ryuguLastAnimationFrame;
            ryuguFrameIntervalCursor = (ryuguFrameIntervalCursor + 1) % ryuguFrameIntervals.length;
            ryuguFrameIntervalCount = Math.min(ryuguFrameIntervalCount + 1, ryuguFrameIntervals.length);
        }
        ryuguLastAnimationFrame = timestamp;
        requestAnimationFrame(record_ryugu_animation_frame);
    }

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(record_ryugu_animation_frame);
    }

    export function browser_actual_fps() {
        if (ryuguFrameIntervalCount === 0) return 0;
        let elapsed = 0;
        for (let index = 0; index < ryuguFrameIntervalCount; index += 1) {
            elapsed += ryuguFrameIntervals[index];
        }
        return elapsed > 0 ? 1000 * ryuguFrameIntervalCount / elapsed : 0;
    }

    export function browser_recent_frame_ms() {
        if (ryuguFrameIntervalCount === 0) return 0;
        const sampleCount = Math.min(8, ryuguFrameIntervalCount);
        let longest = 0;
        for (let offset = 1; offset <= sampleCount; offset += 1) {
            const index = (ryuguFrameIntervalCursor - offset + ryuguFrameIntervals.length)
                % ryuguFrameIntervals.length;
            longest = Math.max(longest, ryuguFrameIntervals[index]);
        }
        return longest;
    }

    export function has_webgpu() {
        return typeof navigator !== "undefined" && navigator.gpu !== undefined;
    }

    export function is_mobile_browser() {
        if (typeof navigator === "undefined") return false;
        if (navigator.userAgentData?.mobile === true) return true;
        return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent ?? "");
    }

    export function show_webgpu_error() {
        const div = document.createElement("div");
        div.style.cssText = "position:absolute;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.9);color:white;display:flex;flex-direction:column;justify-content:center;align-items:center;z-index:99999;font-family:sans-serif;text-align:center;padding:20px;box-sizing:border-box;";
        div.innerHTML = `
            <h1 style="color:#ff4444;margin-bottom:20px;font-size:2.5rem;">WebGPU Not Supported</h1>
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
    #[wasm_bindgen(js_name = browser_actual_fps)]
    fn browser_actual_fps_js() -> f64;

    #[wasm_bindgen(js_name = browser_recent_frame_ms)]
    fn browser_recent_frame_ms_js() -> f64;

    #[wasm_bindgen(js_name = has_webgpu)]
    fn browser_has_webgpu() -> bool;

    #[wasm_bindgen(js_name = is_mobile_browser)]
    fn browser_is_mobile_js() -> bool;

    fn show_webgpu_error();

    #[wasm_bindgen(js_name = set_display_rotation)]
    fn browser_set_display_rotation(quarter_turn: u8);
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn browser_frame_rate() -> Option<f64> {
    let fps = browser_actual_fps_js();
    (fps.is_finite() && fps > 0.0).then_some(fps)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn browser_recent_frame_ms() -> Option<f64> {
    let milliseconds = browser_recent_frame_ms_js();
    (milliseconds.is_finite() && milliseconds > 0.0).then_some(milliseconds)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn browser_frame_rate() -> Option<f64> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn browser_recent_frame_ms() -> Option<f64> {
    None
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn set_display_rotation(quarter_turn: u8) {
    browser_set_display_rotation(quarter_turn);
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn browser_is_mobile() -> bool {
    browser_is_mobile_js()
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
    let is_mobile_browser = {
        #[cfg(target_arch = "wasm32")]
        {
            browser_is_mobile()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
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
        .init_resource::<VolterraPropagationStatus>()
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .insert_resource(WinitSettings {
            // In browsers, Continuous is driven by requestAnimationFrame.
            // Adding a 16.7 ms reactive wait on top of VSync can miss every
            // other presentation and produce an artificial ~30 FPS ceiling.
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 30.0)),
        })
        .add_plugins(
            DefaultPlugins
                .build()
                // The application has no audio. Disabling the plugin avoids
                // creating a browser AudioContext before a user gesture.
                .disable::<bevy::audio::AudioPlugin>()
                .set(LogPlugin {
                    // Browser consoles should contain actionable failures,
                    // not Bevy startup/debug telemetry. Numerical diagnostics
                    // remain visible through the in-app status panels.
                    level: Level::WARN,
                    filter: "warn,wgpu=error,naga=error,bevy_render=error,bevy_shader=error,ryugu_wasm=warn".into(),
                    ..default()
                })
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        canvas: Some("#bevy".into()),
                        fit_canvas_to_parent: true,
                        // Keep browser selection/gesture handling out of the
                        // canvas so PanOrbit receives left-drag and wheel
                        // input whenever the pointer is not over a UI control.
                        prevent_default_event_handling: true,
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        backends: Some(backends),
                        limits,
                        instance_flags: if is_mobile_browser {
                            InstanceFlags::empty()
                        } else {
                            InstanceFlags::debugging()
                        },
                        ..default()
                    })),
                    ..default()
                }),
        )
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(WgslPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default());

    // Native builds use a precise sleep/spin limiter. Browsers use continuous
    // requestAnimationFrame scheduling with VSync above.
    #[cfg(not(target_arch = "wasm32"))]
    {
        app.add_plugins(FramepacePlugin);
        app.insert_resource(FramepaceSettings {
            limiter: Limiter::from_framerate(60.0),
        });
    }

    if has_webgpu {
        app.add_plugins(PlanningGpuComputePlugin);
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

    app.add_systems(Startup, setup_scene);

    #[cfg(target_arch = "wasm32")]
    if is_mobile_browser {
        app.add_plugins(MaterialPlugin::<bevy_app::render::MobileUnlitMaterial>::default());
        // GLTF scenes are instantiated by Bevy between Update and PostUpdate.
        // Convert their materials in PostUpdate so no StandardMaterial reaches
        // the render extraction stage, even for the first visible frame.
        app.add_systems(
            PostUpdate,
            bevy_app::render::configure_mobile_materials_system,
        );
        app.add_systems(
            Update,
            bevy_app::render::mobile_section_alpha_system
                .after(bevy_app::render::section_alpha_system),
        );
    }

    #[cfg(target_arch = "wasm32")]
    app.add_systems(
        Update,
        browser_ui_action_system.in_set(UiComputeOrdering::BrowserActions),
    )
    .add_systems(
        Update,
        browser_ui_publish_system.after(UiComputeOrdering::InversionAndPlanning),
    );

    app.configure_sets(
        Update,
        (
            UiComputeOrdering::BrowserActions.before(UiComputeOrdering::MethodReset),
            UiComputeOrdering::MethodReset.before(UiComputeOrdering::InversionAndPlanning),
        ),
    )
    .add_systems(
        Update,
        (normalize_model_scale_system, build_topology_system).chain(),
    )
    .add_systems(
        Update,
        build_aggregated_gravity_source_system.after(build_radial_gravity_source_system),
    )
    .add_systems(
        Update,
        build_eq106_operator_tensor_system.after(build_aggregated_gravity_source_system),
    )
    .add_systems(
        Update,
        (
            method_selection_system,
            clear_gpu_histories_on_method_change,
            reset_inversion_on_method_change,
        )
            .chain()
            .in_set(UiComputeOrdering::MethodReset),
    )
    .add_systems(
        Update,
        (
            capture_trajectory_inversion_system,
            start_density_inversion_system,
            convex_optimization_system,
            update_planning_results_from_inversion_system,
            planning_batch_evaluator_system,
        )
            .chain()
            .in_set(UiComputeOrdering::InversionAndPlanning),
    )
    .add_systems(Update, apply_probe_input_system)
    .add_systems(Update, clear_inversion_request_on_probe_change)
    .add_systems(Update, clear_runtime_error_on_probe_change)
    .add_systems(
        Update,
        (
            reset_after_probe_crash_scene_system,
            reset_after_probe_crash_state_system,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            performance_comparison_system,
            camera_follow_system,
            update_gpu_memory_estimate_system,
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
    );

    app.run();
}

pub use cpu::benchmark::benchmark_gravity_algorithms;
