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
    Eq106,
    Eq106Nufft,
    Mmfft,
    Fmm,
    Normals,
    PlanningFmm,
    PlanningMmfft,
    PlanningMetrics,
}

pub(crate) fn load(server: &AssetServer, shader: EmbeddedShader) -> Handle<Shader> {
    match shader {
        EmbeddedShader::Gravity => load_embedded_asset!(server, "gravity.wgsl"),
        EmbeddedShader::Werner => load_embedded_asset!(server, "werner_gravity.wgsl"),
        EmbeddedShader::Eq106 => load_embedded_asset!(server, "eq106_complex.wgsl"),
        EmbeddedShader::Eq106Nufft => load_embedded_asset!(server, "eq106_nufft.wgsl"),
        EmbeddedShader::Mmfft => load_embedded_asset!(server, "mmfft_compressed.wgsl"),
        EmbeddedShader::Fmm => load_embedded_asset!(server, "fmm_gravity.wgsl"),
        EmbeddedShader::Normals => load_embedded_asset!(server, "normals.wgsl"),
        EmbeddedShader::PlanningFmm => load_embedded_asset!(server, "planning_fmm.wgsl"),
        EmbeddedShader::PlanningMmfft => load_embedded_asset!(server, "planning_mmfft.wgsl"),
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
        embedded_asset!(app, "eq106_complex.wgsl");
        embedded_asset!(app, "eq106_nufft.wgsl");
        embedded_asset!(app, "mmfft_compressed.wgsl");
        embedded_asset!(app, "fmm_gravity.wgsl");
        embedded_asset!(app, "normals.wgsl");
        embedded_asset!(app, "planning_fmm.wgsl");
        embedded_asset!(app, "planning_mmfft.wgsl");
        embedded_asset!(app, "planning_metrics.wgsl");
    }
}
