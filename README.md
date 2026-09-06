# Ryugu Dynamics Laboratory

[![CI/CD](https://github.com/Tom-jim/RyuGu_WASM/actions/workflows/deploy.yml/badge.svg)](https://github.com/Tom-jim/RyuGu_WASM/actions/workflows/deploy.yml)
[![Live demo](https://img.shields.io/badge/Live_demo-WebGPU-success)](https://tom-jim.github.io/RyuGu_WASM/)
[![Bevy](https://img.shields.io/badge/Bevy-0.19.1-purple)](https://bevy.org/)
[![Rust](https://img.shields.io/badge/Rust-2024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Ryugu Dynamics Laboratory is an in-browser laboratory for gravity, orbit propagation, surface-field inspection, and density inversion around the near-Earth asteroid Ryugu. Rust and Bevy own the physical state and scene, WebGPU/WGSL runs the parallel numerical kernels, and the HTML interface presents the live orbit, diagnostics, density views, and performance comparisons.

The public mathematical and algorithmic description is maintained in [mathpub.md](mathpub.md). It is the single public reference for the numerical model; implementation notes and working derivations remain outside the GitHub README.

## Workbench

- **Left rail:** probe initial conditions, camera controls, normals and density-section display, surface fields, patch inspection, and trajectory-knot editing.
- **Center:** the Bevy 3D scene, live orbit, and diagnostics for the selected method.
- **Right rail:** density inversion, trajectory-design comparisons, accuracy limits, run state, and background-execution settings.
- **Top bar:** FMM, FFT, Werner, Radial, and Frequency-domain selection, performance comparison, gesture mode, and layout reset.

Panels can be dragged, resized, and scrolled independently. **Reset UI view** restores panel positions and the overall UI scale. Narrow screens switch to a vertical layout instead of shrinking desktop text below a readable size.

### Mobile layout and gestures

The default gesture mode is **Bevy camera**: one finger rotates the camera, two fingers pan or pinch-zoom it, and camera-wheel zoom leaves HTML control dimensions unchanged.

Switching to **whole UI** makes pinch or wheel gestures scale the complete workbench and dragging pan the complete view. The switch waits for the current touch sequence to finish so the camera does not receive a partial gesture. **Rotate 90°** remains available, and panel dragging accounts for rotation and UI scale.

Mobile devices use simplified materials, disable scene MSAA, and cap the canvas pixel ratio. Desktop user agents on iPad are still detected as touch devices. Forms respect safe areas and use large touch targets; trajectory number fields avoid triggering iOS auto-zoom.

## Density and surface fields

Choose a solver, then click **Calculate field** to display effective gravity, gravity gradient, or effective slope. The default source profile is the mass-preserving logarithmic radial distribution for Radial, FFT, FMM, and Frequency-domain methods. Werner uses its homogeneous closed-polyhedron density. The **Section** button displays the active method's density field without changing the source model or starting a surface-field calculation.

Surface inspection uses a shared patch index for position, normal, field, and error. Comparison signs are reported as comparison minus baseline. Werner remains a homogeneous-polyhedron reference; heterogeneous modes use their own volume representation and are not silently substituted with the Werner result.

## Frequency-domain workflow

Frequency-domain propagation first advances the spacecraft with the continuous Ryugu density field and its independent field history. The captured trajectory is then consumed by the whole-trajectory transform and density-inversion stages. A fixed prerecorded track is never played back, and the frequency-domain path does not borrow acceleration history from Radial, FFT, FMM, or Werner.

The public mathematical description, notation, discretization, GPU reductions, inversion model, validation strategy, and numerical limits are in [mathpub.md](mathpub.md).

## Performance comparison

<img src="https://github.com/user-attachments/assets/84ac0b30-d669-4a40-99a5-31ef39b3f8c0" width="100%" alt="algorithm comparison" />

The figure is retained from the original performance record and was not remeasured after the latest implementation changes. It is a historical comparison, not a performance guarantee for every device or workload.

- The performance page retains FPS and method-diagnostic curves.
- First, Stress, and Quadrature retain controls for density-model count, target count, randomized order, repetitions, source sizes, medians, ranges, and accuracy thresholds.
- GPU time comes only from GPU timestamps; CPU encoding and readback waits are not presented as GPU kernel time. Samples without timestamps remain missing.
- Whole-trajectory observations and point-field references are checked in their own output spaces; their errors are not interchangeable.

## Background execution and keep-alive

The existing three-layer mechanism is retained: foreground native RAF, background Worker scheduling, and a local-server WebSocket clock with a macOS sleep-prevention process. The background path keeps one pending tick and never creates a second WASM engine. Page restoration, network reconnection, and device-loss handling remain enabled.

- **Continue scheduling when this tab is hidden** controls background scheduling; the local-server connection also maintains the sleep-prevention lease.
- **Keep the screen awake while visible** requests the browser Screen Wake Lock for long foreground mobile jobs. It is requested again when the page becomes visible and does not replace the Worker or local keep-alive path.
- Screenshots, completed results, retryable exports, and recovery after device loss remain handled by the existing capture, background-maintenance, and server paths.
- Local screenshots are stored in `benchmark-captures/`; the static-hosted page retains its download path.

Browser freeze/discard, forced system sleep, and lid closure cannot be overridden by a web page. Mobile background execution and the static-hosted page also cannot use the server Mac's local sleep-prevention process. The status area reports the capabilities that are actually available.

## Code map

| Path | Responsibility |
| --- | --- |
| `src/cpu/density.rs` | Shared density geometry and mass-preserving source representation |
| `src/cpu/physics.rs` | Orbit integration, time consistency, and observation-arc capture |
| `src/gpu/frequency_domain_pipeline/` | Whole-trajectory transform, sensitivities, planning, and readback |
| `src/gpu/fmm_pipeline/` | Tree construction and multipole evaluation |
| `src/gpu/mmfft_pipeline/` | Grid, spectrum, and interpolated field evaluation |
| `src/gpu/radial.rs` | Radial analytic field evaluation and reductions |
| `src/cpu/inversion_components/` | Reference operators, density basis, and constrained optimization |
| `src/bevy/surface_field.rs` | Surface field, slope, and comparison display |
| `src/html/navigation.js` | Camera or whole-UI gesture switching, scaling, and panel dragging |
| `src/html/background*.js` | Background scheduling, keep-alive, and mobile Wake Lock |
| `src/html/ui.js` / `app.js` | Controls, state, and charts |
| `src/wgsl/` | Unified WGSL shaders loaded by the Rust GPU pipelines |

## Build and checks

```sh
bun install
bun run build
bun run serve
```

Deployment is defined in [.github/workflows/deploy.yml](.github/workflows/deploy.yml). The generated WebAssembly package is loaded by `src/html/index.html`; deployment also copies gesture controls, the background Worker, and the required shader assets.

## Numerical limits

The current source geometry uses a Ryugu-suitable star-shaped angular-cell and shell approximation. Finite wave-number and radial quadrature, f32 arithmetic, FFT grid quantization, FMM order, near-field treatment, and asynchronous extrapolation each introduce separate errors and require separate convergence checks.

A single external orbit cannot uniquely recover an arbitrary three-dimensional density without priors, gradient measurements, or additional trajectories. Identifiability is therefore discussed only for a specified finite parameterization, feasible constraints, and a regularized optimization problem.

## License

MIT; see [LICENSE](LICENSE).
