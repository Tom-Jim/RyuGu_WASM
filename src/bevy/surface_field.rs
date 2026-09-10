//! Surface gravity, effective-gravity slope, and method-comparison products.
//!
//! The real-time GPU paths remain responsible for the probe trajectory. This
//! module owns an explicit, reproducible surface product: common surface
//! patches are evaluated with method-specific CPU reference operators, then
//! uploaded as vertex colors on a thin overlay mesh. That keeps the display
//! useful for validation without making a pointwise claim for equation (184).

include!("surface/state.rs");
include!("surface/geometry.rs");
include!("surface/compute.rs");
include!("surface/render.rs");
include!("surface/evaluation.rs");
