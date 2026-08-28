//! GPU shader validation tests.

use naga::valid::{Capabilities, ValidationFlags, Validator};

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
    validate_wgsl(source);
    assert!(!source.contains("if global_id.x == 0u"));
    assert!(source.matches("@builtin(local_invocation_index)").count() >= 3);
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
