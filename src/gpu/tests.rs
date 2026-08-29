//! GPU shader validation tests.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use bevy::tasks::block_on;
#[cfg(not(target_arch = "wasm32"))]
use std::borrow::Cow;

fn preprocess_eq106(source: &str, definition: &str) -> String {
    let mut enabled = true;
    let mut output = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("#ifdef ") {
            enabled = name.trim() == definition;
            continue;
        }
        if trimmed == "#endif" {
            enabled = true;
            continue;
        }
        if enabled {
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}

fn validate_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source).expect("WGSL parsing failed");
    // Browser WebGPU exposes the baseline capability set plus the standard
    // pack/unpack-f16 builtins used by the compressed FFT. `all()` masked
    // accidental use of unrelated native-only shader features.
    Validator::new(
        ValidationFlags::all(),
        Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    )
    .validate(&module)
    .expect("WGSL validation failed");
}

#[test]
fn radial_gravity_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/gravity.wgsl"));
}

#[test]
fn werner_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/werner_gravity.wgsl"));
}

#[test]
fn mmfft_compressed_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/mmfft_compressed.wgsl"));
}

#[test]
fn eq106_complex_shader_is_valid() {
    let source = include_str!("../wgsl/eq106_complex.wgsl");
    assert!(!source.contains("if global_id.x == 0u"));
    assert!(!source.contains("mix(p1, p2, fraction)"));
    assert!(source.contains("SOURCE_REDUCTION_CHUNK: u32 = 2u"));
    assert!(
        source.contains("chunk < MAX_TAYLOR_COEFFICIENT_COUNT; chunk += SOURCE_REDUCTION_CHUNK")
    );
    assert!(source.matches("@builtin(local_invocation_index)").count() >= 3);
    for definition in ["EQ106_SOURCE", "EQ106_SPECTRUM", "EQ106_EVALUATOR"] {
        validate_wgsl(&preprocess_eq106(source, definition));
    }
}

#[test]
fn eq106_complex_shader_matches_bevy_shader_composition() {
    let source = include_str!("../wgsl/eq106_complex.wgsl");
    for definition in ["EQ106_SOURCE", "EQ106_SPECTRUM", "EQ106_EVALUATOR"] {
        let mut shader_defs = HashMap::new();
        shader_defs.insert(definition.to_owned(), ShaderDefValue::Bool(true));
        let mut composer =
            Composer::default().with_capabilities(Capabilities::SHADER_FLOAT16_IN_FLOAT32);
        let module = composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: "eq106_complex.wgsl",
                shader_defs,
                ..Default::default()
            })
            .unwrap_or_else(|error| panic!("{definition} composition failed: {error:?}"));
        Validator::new(
            ValidationFlags::all(),
            Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{definition} validation failed: {error:?}"));
        let module_info = Validator::new(
            ValidationFlags::all(),
            Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .unwrap();
        let generated = naga::back::wgsl::write_string(
            &module,
            &module_info,
            naga::back::wgsl::WriterFlags::empty(),
        )
        .unwrap_or_else(|error| panic!("{definition} WGSL generation failed: {error:?}"));
        validate_wgsl(&generated);
        std::fs::write(format!("/private/tmp/eq106_{definition}.wgsl"), generated).unwrap();
    }
}

#[test]
fn eq106_type2_nufft_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/eq106_nufft.wgsl"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn eq106_compute_pipelines_compile_on_backend() {
    block_on(async {
        let mut instance_descriptor = wgpu29::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = wgpu29::Backends::all();
        let instance = wgpu29::Instance::new(instance_descriptor);
        let Ok(adapter) = instance
            .request_adapter(&wgpu29::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("skipping native shader validation: no GPU adapter is available");
            return;
        };
        let (device, _queue) = adapter
            .request_device(&wgpu29::DeviceDescriptor {
                label: Some("Eq.106 shader validation device"),
                required_limits: wgpu29::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("failed to create shader validation device");

        let source = include_str!("../wgsl/eq106_complex.wgsl");
        validate_compute_module(
            &device,
            "eq106_source",
            &preprocess_eq106(source, "EQ106_SOURCE"),
            &["assemble_line_samples", "assemble_voxel_line_samples"],
        )
        .await;
        validate_compute_module(
            &device,
            "eq106_spectrum",
            &preprocess_eq106(source, "EQ106_SPECTRUM"),
            &[
                "assemble_spectrum",
                "assemble_voxel_spectrum",
                "combine_voxel_spectrum",
            ],
        )
        .await;
        validate_compute_module(
            &device,
            "eq106_evaluator",
            &preprocess_eq106(source, "EQ106_EVALUATOR"),
            &["evaluate_field"],
        )
        .await;
        validate_compute_module(
            &device,
            "eq106_nufft",
            include_str!("../wgsl/eq106_nufft.wgsl"),
            &["build_type2_nufft_grid"],
        )
        .await;
    });
}

#[cfg(not(target_arch = "wasm32"))]
async fn validate_compute_module(
    device: &wgpu29::Device,
    label: &str,
    source: &str,
    entry_points: &[&str],
) {
    let scope = device.push_error_scope(wgpu29::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu29::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu29::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let compilation_info = module.get_compilation_info().await;
    let module_error = scope.pop().await;
    assert!(
        module_error.is_none(),
        "{label} shader module creation failed: {module_error:#?}; compilation info: {compilation_info:#?}"
    );
    assert!(
        compilation_info
            .messages
            .iter()
            .all(|message| message.message_type != wgpu29::CompilationMessageType::Error),
        "{label} shader compilation failed: {compilation_info:#?}"
    );

    for &entry_point in entry_points {
        let scope = device.push_error_scope(wgpu29::ErrorFilter::Validation);
        let _pipeline = device.create_compute_pipeline(&wgpu29::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: None,
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_error = scope.pop().await;
        assert!(
            pipeline_error.is_none(),
            "{label}::{entry_point} pipeline creation failed: {pipeline_error:#?}"
        );
    }
}

#[test]
fn fmm_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/fmm_gravity.wgsl"));
}

#[test]
fn planning_fmm_shader_is_valid() {
    let source = include_str!("../wgsl/planning_fmm.wgsl");
    validate_wgsl(source);
    assert!(!source.contains("if local_index >= params.local_count"));
}

#[test]
fn planning_mmfft_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/planning_mmfft.wgsl"));
}

#[test]
fn planning_reduction_shader_is_valid() {
    validate_wgsl(include_str!("../wgsl/planning_metrics.wgsl"));
}
