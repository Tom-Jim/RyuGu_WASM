//! Backend-independent deterministic benchmark entry points.

/// Deterministic WASM-side microbenchmark used by the host benchmark harness.
/// The browser performance panel measures complete Bevy/WebGPU paths; this
/// function measures the corresponding scalar kernels without DOM imports.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub fn benchmark_gravity_algorithms(iterations: u32) -> f64 {
    let iterations = iterations.max(1);
    let mut checksum = 0.0_f64;
    for index in 0..iterations {
        let radius = 120.0 + (index % 4096) as f64 * 0.125;
        let logarithmic_density =
            (1.0 + radius / crate::interface::components::DENSITY_EPSILON as f64).ln();
        let radial = logarithmic_density * radius * radius;
        let edge_log = ((radius + 900.0 + 42.0) / (radius + 900.0 - 42.0)).ln();
        let werner = edge_log * (radius + 1.0).recip();
        let displacement = 0.05 * (index as f64 * 0.017).sin();
        let ratio = displacement / radius;
        let taylor = ratio + 0.5 * ratio * ratio + 0.375 * ratio * ratio * ratio;
        checksum += radial + werner + taylor;
    }
    std::hint::black_box(checksum)
}
