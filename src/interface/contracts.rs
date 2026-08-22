//! Shared history selection for Bevy-facing diagnostics and CPU integration.

use crate::interface::components::{
    ActiveGravityMethod, Eq106GpuHistory, FmmGravityHistory, GravitySampleHistory,
    MmfftCompressedHistory, RadialGravityHistory, WernerGravityHistory,
};

/// Select the snapshot history owned by the active backend. Bevy systems use
/// this adapter instead of repeating backend-selection matches.
pub fn select_history<'a>(
    method: ActiveGravityMethod,
    radial: Option<&'a RadialGravityHistory>,
    werner: Option<&'a WernerGravityHistory>,
    eq106: Option<&'a Eq106GpuHistory>,
    mmfft: Option<&'a MmfftCompressedHistory>,
    fmm: Option<&'a FmmGravityHistory>,
) -> Option<&'a GravitySampleHistory> {
    match method {
        ActiveGravityMethod::RadialAnalytic => radial.map(|history| &history.0),
        ActiveGravityMethod::HomogeneousWerner => werner.map(|history| &history.0),
        ActiveGravityMethod::CurvedArcEq106 => eq106.map(|history| &history.0),
        ActiveGravityMethod::MmfftCompressed => mmfft.map(|history| &history.0),
        ActiveGravityMethod::Fmm => fmm.map(|history| &history.0),
    }
}
