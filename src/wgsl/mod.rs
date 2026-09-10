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
    Equation106,
    FrequencyDomain,
    Normals,
    PlanningMetrics,
}

pub(crate) fn load(server: &AssetServer, shader: EmbeddedShader) -> Handle<Shader> {
    match shader {
        EmbeddedShader::Equation106 => load_embedded_asset!(server, "equation106.wgsl"),
        EmbeddedShader::FrequencyDomain => load_embedded_asset!(server, "frequency_domain.wgsl"),
        EmbeddedShader::Normals => load_embedded_asset!(server, "normals.wgsl"),
        EmbeddedShader::PlanningMetrics => {
            load_embedded_asset!(server, "planning_metrics.wgsl")
        }
    }
}

pub(crate) struct WgslPlugin;

impl Plugin for WgslPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "equation106.wgsl");
        embedded_asset!(app, "frequency_domain.wgsl");
        embedded_asset!(app, "normals.wgsl");
        embedded_asset!(app, "planning_metrics.wgsl");
        embedded_asset!(app, "mobile_unlit.wgsl");
    }
}

// Parse and validate the actual production modules with wgpu's own Naga
// version. This catches helper-function errors before they cascade into
// misleading "entry point does not exist" messages for every GPU pipeline.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod shader_validation_tests {
    use wgpu29::naga;

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
            let params = module
                .types
                .iter()
                .find_map(|(_, ty)| {
                    (ty.name.as_deref() == Some("FrequencyDomainParams")).then_some(ty)
                })
                .unwrap_or_else(|| panic!("{definition}: missing FrequencyDomainParams"));
            let naga::TypeInner::Struct { span, .. } = &params.inner else {
                panic!("{definition}: FrequencyDomainParams is not a struct");
            };
            assert_eq!(*span, 52, "{definition}: host/WGSL parameter stride");
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
