//! Shared history selection for Bevy-facing diagnostics and CPU integration.

use crate::interface::components::{
    ActiveGravityMethod, FmmGravityHistory, GravitySampleHistory, MmfftCompressedHistory,
    RadialGravityHistory, WernerGravityHistory,
};

/// Select the physical snapshot history used by pointwise runtime systems.
/// Equation (184) deliberately has no entry: its result is an aggregate
/// transform over a known trajectory, not a pointwise force history.
pub fn select_history<'a>(
    method: ActiveGravityMethod,
    radial: Option<&'a RadialGravityHistory>,
    werner: Option<&'a WernerGravityHistory>,
    mmfft: Option<&'a MmfftCompressedHistory>,
    fmm: Option<&'a FmmGravityHistory>,
) -> Option<&'a GravitySampleHistory> {
    match method {
        ActiveGravityMethod::RadialAnalytic => radial.map(|history| &history.0),
        ActiveGravityMethod::HomogeneousWerner => werner.map(|history| &history.0),
        ActiveGravityMethod::FrequencyDomain => None,
        ActiveGravityMethod::MmfftCompressed => mmfft.map(|history| &history.0),
        ActiveGravityMethod::Fmm => fmm.map(|history| &history.0),
    }
}
