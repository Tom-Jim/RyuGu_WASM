include!("eq106_pipeline/batch_types.rs");
include!("eq106_pipeline/input_extraction.rs");
include!("eq106_pipeline/sensitivity_dispatch.rs");
include!("eq106_pipeline/dispatch.rs");
include!("eq106_pipeline/layout.rs");
include!("eq106_pipeline/planning_payload.rs");
include!("eq106_pipeline/planning_dispatch.rs");

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "eq106_pipeline/nufft_gpu_tests.rs"]
mod nufft_gpu_tests;
