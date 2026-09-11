//! Shared history selection for Bevy-facing diagnostics and CPU integration.

use crate::interface::components::{
    ActiveGravityMethod, FmmGravityHistory, GravitySampleHistory, MmfftCompressedHistory,
    RadialGravityHistory, WernerGravityHistory,
};

/// Select the physical snapshot history used by pointwise runtime systems.
/// Frequency-domain live dynamics evaluate Eq.121 inside the Worker; GPU
/// Eq.121 stamps remain diagnostics. Eq.184 aggregate observations cannot
/// drive pointwise dynamics through this contract.
pub fn select_history<'a>(
    method: ActiveGravityMethod,
    radial: Option<&'a RadialGravityHistory>,
    werner: Option<&'a WernerGravityHistory>,
    mmfft: Option<&'a MmfftCompressedHistory>,
    fmm: Option<&'a FmmGravityHistory>,
    equation106: Option<&'a crate::gpu::equation106::Equation106History>,
) -> Option<&'a GravitySampleHistory> {
    match method {
        ActiveGravityMethod::RadialAnalytic => radial.map(|history| &history.0),
        ActiveGravityMethod::HomogeneousWerner => werner.map(|history| &history.0),
        ActiveGravityMethod::FrequencyDomain => equation106.map(|history| &history.0),
        ActiveGravityMethod::MmfftCompressed => mmfft.map(|history| &history.0),
        ActiveGravityMethod::Fmm => fmm.map(|history| &history.0),
    }
}
