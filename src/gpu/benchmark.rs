//! GPU benchmark metadata shared by render-world timestamp instrumentation.
//!
//! The Eq.106 pipeline records four separately measurable stages: spectrum
//! construction, target evaluation, GPU readback copy, and CPU readback wait.
//! Keeping the stage count here makes the benchmark contract explicit without
//! coupling the benchmark harness to Bevy scheduling.

/// Number of timestamp stages reserved per dispatched Eq.106 target element.
pub(crate) const TIMESTAMP_STAGE_COUNT: u32 = 4;
