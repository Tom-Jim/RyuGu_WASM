//! GPU backends and render-world compute adapters.

pub(crate) mod eq106;
pub(crate) mod fmm;
pub(crate) mod mmfft;
pub(crate) mod normals;
pub(crate) mod planning;
pub(crate) mod planning_reduction;
pub(crate) mod radial;
pub(crate) mod werner;

#[cfg(test)]
mod tests;
