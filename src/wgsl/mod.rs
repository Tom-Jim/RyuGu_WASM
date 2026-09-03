//! Compile-time embedded WebGPU shaders.
//!
//! Pipeline resources are created in Bevy's render world, which owns its own
//! resource set. Registering WGSL through Bevy's embedded asset source keeps
//! the files inside the WASM binary while allowing both worlds to resolve the
//! same stable handles through their existing `AssetServer`.

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::prelude::*;
use bevy::shader::Shader;

#[derive(Clone, Copy)]
pub(crate) enum EmbeddedShader {
    Gravity,
    Werner,
    FrequencyDomain,
    Mmfft,
    Fmm,
    Normals,
    PlanningFmm,
    PlanningMmfft,
    PlanningFftBasis,
    PlanningFmmBasis,
    PlanningMetrics,
}

pub(crate) fn load(server: &AssetServer, shader: EmbeddedShader) -> Handle<Shader> {
    match shader {
        EmbeddedShader::Gravity => load_embedded_asset!(server, "gravity.wgsl"),
        EmbeddedShader::Werner => load_embedded_asset!(server, "werner_gravity.wgsl"),
        EmbeddedShader::FrequencyDomain => load_embedded_asset!(server, "frequency_domain.wgsl"),
        EmbeddedShader::Mmfft => load_embedded_asset!(server, "mmfft_compressed.wgsl"),
        EmbeddedShader::Fmm => load_embedded_asset!(server, "fmm_gravity.wgsl"),
        EmbeddedShader::Normals => load_embedded_asset!(server, "normals.wgsl"),
        EmbeddedShader::PlanningFmm => load_embedded_asset!(server, "planning_fmm.wgsl"),
        EmbeddedShader::PlanningMmfft => load_embedded_asset!(server, "planning_mmfft.wgsl"),
        EmbeddedShader::PlanningFmmBasis => load_embedded_asset!(server, "planning_fmm_basis.wgsl"),
        EmbeddedShader::PlanningFftBasis => load_embedded_asset!(server, "planning_fft_basis.wgsl"),
        EmbeddedShader::PlanningMetrics => {
            load_embedded_asset!(server, "planning_metrics.wgsl")
        }
    }
}

pub(crate) struct WgslPlugin;

impl Plugin for WgslPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "gravity.wgsl");
        embedded_asset!(app, "werner_gravity.wgsl");
        embedded_asset!(app, "frequency_domain.wgsl");
        embedded_asset!(app, "mmfft_compressed.wgsl");
        embedded_asset!(app, "fmm_gravity.wgsl");
        embedded_asset!(app, "normals.wgsl");
        embedded_asset!(app, "planning_fmm.wgsl");
        embedded_asset!(app, "planning_mmfft.wgsl");
        embedded_asset!(app, "planning_fft_basis.wgsl");
        embedded_asset!(app, "planning_fmm_basis.wgsl");
        embedded_asset!(app, "planning_metrics.wgsl");
    }
}

// Parse and validate the actual production modules with wgpu's own Naga
// version. This catches helper-function errors before they cascade into
// misleading "entry point does not exist" messages for every GPU pipeline.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod shader_validation_tests {
    use wgpu29::naga;

    #[test]
    fn planning_modules_validate_with_the_runtime_shader_frontend() {
        let modules: [(&str, &str, &[&str], u32, u32); 4] = [
            (
                "planning_fft_basis",
                include_str!("planning_fft_basis.wgsl"),
                &[
                    "deposit",
                    "seed_kernel",
                    "load_column",
                    "transform",
                    "convolve",
                    "store_column",
                    "combine",
                ],
                64,
                8,
            ),
            (
                "planning_fmm_basis",
                include_str!("planning_fmm_basis.wgsl"),
                &["p2m", "m2m", "response_basis"],
                48,
                9,
            ),
            (
                "planning_fmm",
                include_str!("planning_fmm.wgsl"),
                &["main"],
                16,
                4,
            ),
            (
                "planning_mmfft",
                include_str!("planning_mmfft.wgsl"),
                &["main"],
                80,
                4,
            ),
        ];
        for (name, source, entry_names, uniform_bytes, binding_count) in modules {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
            assert_eq!(module.entry_points.len(), entry_names.len(), "{name}");
            for entry in entry_names {
                assert!(
                    module.entry_points.iter().any(|point| {
                        point.name == *entry && point.stage == naga::ShaderStage::Compute
                    }),
                    "{name}: missing compute entry {entry}"
                );
            }
            let mut bindings = Vec::new();
            for (_, variable) in module.global_variables.iter() {
                let Some(binding) = &variable.binding else {
                    continue;
                };
                assert_eq!(binding.group, 0, "{name}");
                bindings.push(binding.binding);
                if binding.binding == 0 {
                    assert_eq!(variable.space, naga::AddressSpace::Uniform, "{name}");
                    let naga::TypeInner::Struct { span, .. } = &module.types[variable.ty].inner
                    else {
                        panic!("{name}: expected uniform struct");
                    };
                    assert_eq!(*span, uniform_bytes, "{name}: host/WGSL uniform size");
                }
            }
            bindings.sort_unstable();
            assert_eq!(bindings, (0..binding_count).collect::<Vec<_>>(), "{name}");
        }
    }

    #[test]
    fn frequency_domain_variants_validate_with_the_runtime_shader_frontend() {
        let original = include_str!("frequency_domain.wgsl");
        assert!(original.contains(&format!(
            "const WAVE_VECTOR_COUNT: u32 = {}u;",
            crate::cpu::frequency_domain::EQ184_QUADRATURE_COUNT,
        )));
        for (definition, entry_names) in [
            (
                "FREQUENCY_DOMAIN_SOURCE",
                &[
                    "assemble_density_spectrum",
                    "assemble_voxel_density_spectrum",
                ][..],
            ),
            (
                "FREQUENCY_DOMAIN_SPECTRUM",
                &[
                    "publish_density_spectrum",
                    "publish_voxel_density_spectrum",
                    "combine_density_spectrum",
                ][..],
            ),
            (
                "FREQUENCY_DOMAIN_EVALUATOR",
                &["evaluate_trajectory_field"][..],
            ),
        ] {
            let source = select_shader_definition(original, definition);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{definition}: {}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{definition}: {}", error.emit_to_string(&source)));
            assert_eq!(module.entry_points.len(), entry_names.len(), "{definition}");
            for entry in entry_names {
                assert!(
                    module.entry_points.iter().any(|point| {
                        point.name == *entry && point.stage == naga::ShaderStage::Compute
                    }),
                    "{definition}: missing compute entry {entry}",
                );
            }
        }
    }

    fn select_shader_definition(source: &str, enabled: &str) -> String {
        let mut include = true;
        let mut selected = String::with_capacity(source.len());
        for line in source.lines() {
            if let Some(definition) = line.trim().strip_prefix("#ifdef ") {
                include = definition == enabled;
            } else if line.trim() == "#endif" {
                include = true;
            } else if include {
                selected.push_str(line);
                selected.push('\n');
            }
        }
        selected
    }
}
