//! GPU shader validation tests.

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source).expect("WGSL parsing failed");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("WGSL validation failed");
}

#[test]
fn radial_gravity_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/gravity.wgsl"));
}

#[test]
fn werner_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/werner_gravity.wgsl"));
}

#[test]
fn mmfft_compressed_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/mmfft_compressed.wgsl"));
}

#[test]
fn eq106_complex_shader_is_valid() {
    let source = include_str!("../../assets/shaders/eq106_complex.wgsl");
    validate_wgsl(source);
    assert!(!source.contains("if global_id.x == 0u"));
    assert!(source.matches("@builtin(local_invocation_index)").count() >= 3);
}

#[test]
fn fmm_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/fmm_gravity.wgsl"));
}

#[test]
fn planning_fmm_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/planning_fmm.wgsl"));
}

#[test]
fn planning_mmfft_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/planning_mmfft.wgsl"));
}

#[test]
fn planning_reduction_shader_is_valid() {
    validate_wgsl(include_str!("../../assets/shaders/planning_metrics.wgsl"));
}
