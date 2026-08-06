# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```sh
# JS dependencies (none at runtime; just primes Bun's module cache)
bun install

# Dev: builds WASM (debug, opt-level 1) then starts server at http://localhost:3000
bun run dev

# Production build only (release: opt-level "z", LTO) -> pkg/
bun run build

# Production-mode preview (build + serve)
bun run preview

# Start dev server only (requires pkg/ to already exist)
bun run serve

# Syntax check only (host build, no WASM)
cargo build

# Lint / format (native host, for IDE feedback only — not the WASM target)
cargo clippy
cargo fmt
```

> `cargo run` / `cargo build` without `--target wasm32-unknown-unknown` are only useful for syntax checking. The runnable artifact is the WASM package produced by `wasm-pack --target web`.

**Before running:** `lut_struve.bin` must exist in the project root. If missing, the simulation panics immediately at `build_gravity_voxels_system`. Regenerate with:

```sh
python scripts/gen_lut.py
```

## Architecture

Bevy 0.19.0 asteroid simulator of real asteroid Ryugu, compiled to WASM via `wasm-pack`. Entry point is `pub fn main()` in `src/lib.rs`, annotated `#[wasm_bindgen(start)]`, mounted on the `#bevy` canvas in `index.html`.

**Source layout**

| Path | Role |
|---|---|
| `src/lib.rs` | App setup: plugin stack, resource init, system scheduling |
| `src/components.rs` | All ECS components, resources, and physics constants |
| `src/topology.rs` | Builds a CSR adjacency list from the welded mesh |
| `src/welding.rs` | Vertex deduplication (quantized HashMap) → `WeldedMesh` |
| `src/systems/scale.rs` | `normalize_model_scale_system` + `build_topology_system` (solves `DensityC`) |
| `src/systems/compute_pipeline.rs` | `NormalsComputePlugin` — one-shot render-world compute pass for surface normals |
| `src/systems/gravity_pipeline.rs` | `GravityComputePlugin` — per-frame GPU gravity dispatch + readback |
| `src/systems/physics.rs` | Semi-implicit Euler orbit integration + `ryugu_rotation_system` |
| `src/systems/render.rs` | Scene setup, camera follow/switch, gizmos, section plane |
| `src/systems/ui.rs` | FPS display, keyboard toggles (normals / section view / gravity method) |
| `src/systems/werner_pipeline.rs` | `WernerComputePlugin` — per-frame GPU Werner-decomposition gravity dispatch + readback |
| `assets/shaders/normals.wgsl` | WGSL compute shader for surface normal averaging (CSR neighbor ring) |
| `assets/shaders/gravity.wgsl` | WGSL compute shader: Gaver-Stehfest NILT loop + Struve-Neumann LUT |
| `assets/shaders/werner_gravity.wgsl` | WGSL compute shader: Werner-series decomposition kernel |
| `scripts/gen_lut.py` | Precomputes `S₀(z)`, `S₁(z)` via SciPy → `lut_struve.bin` (32 KB, 4096×2 f32) |

**Plugin stack (lib.rs)**

- `DefaultPlugins` — `AssetPlugin { meta_check: Never }`, `WindowPlugin` (canvas `#bevy`, fits parent), `RenderPlugin` forces `BROWSER_WEBGPU` backend and raises `max_storage_buffers_per_shader_stage = 8` / `max_compute_workgroups_per_dimension = 65535`
- `ObjPlugin` (`bevy_obj`) — registered for Wavefront `.obj` mesh support
- `PanOrbitCameraPlugin` (`bevy_panorbit_camera`) — mouse-orbit camera
- `FrameTimeDiagnosticsPlugin` — FPS counter
- `NormalsComputePlugin` — custom render-world compute pass (WGSL)
- `GravityComputePlugin` — custom render-world compute pass (WGSL)
- `WernerComputePlugin` — custom render-world compute pass (WGSL, Werner-decomposition gravity)

All three compute plugins are only registered when `navigator.gpu` is available; on a non-WebGPU browser the app falls back to a Newtonian CPU gravity path and shows an inline warning overlay.

**One-shot initialization via marker components**

`normalize_model_scale_system` and `build_topology_system` run every frame but are guarded by `Without<ScaleNormalized>` / `Without<TopologyBuilt>` queries. They insert these markers on completion so the work only runs once. The same pattern applies to `build_gravity_voxels_system`. Do not remove these marker inserts.

**System ordering**

All Update systems in `lib.rs` are registered with `.chain()`, enforcing strict sequential ordering every frame:
`normalize_model_scale` → `build_topology` → `build_gravity_voxels` → `physics` → `ryugu_rotation` → camera/UI/render systems.

**GPU compute / readback pattern**

Compute plugins live in the Bevy render world and communicate results back to the main world via two channels:
- `Arc<Mutex<Option<Vec<[f32;4]>>>>` for `GravityComputePlugin` (`GravityReadbackChannel`) and `WernerComputePlugin` (`WernerReadbackChannel`) — fixed-size GPU-output readbacks.
- `WernerAcceleration(pub Vec3)` resource and `GravityAcceleration(Vec3)` resource carry the latest per-probe acceleration onto the main world.

Staging buffers are mapped async; each main-world system drains its channel each frame if data is ready.

**Gravity-method selection**

`ActiveGravityMethod` (`components.rs`) toggles between `VoxelStehfest` (default, `gravity.wgsl`) and `DecomposedWerner` (`werner_gravity.wgsl`). Pressing **G** in `method_toggle_system` switches the active shader used by `physics_system`; if a method's GPU readback hasn't landed yet, the integrator falls back to a Newtonian point-mass acceleration.

**Physics constants (components.rs)**

Real-world values: `RYUGU_MASS = 4.5e11 kg`, `RYUGU_ROTATION_PERIOD_SECS = 7.63 * 3600`, `RYUGU_SPIN_AXIS = Vec3(-0.043, -0.914, 0.405)`, `TIME_SCALE = 20000` (simulation speedup).

`DensityC` is solved once in `build_topology_system` as `RYUGU_MASS / ∫(1/(‖r‖ + ε))dV` over the mesh using 4-point Gaussian quadrature on signed tetrahedra. The kernel is `1/(‖r‖ + ε)` with `ε = DENSITY_EPSILON = 10.0`. The same kernel is used in `build_gravity_voxels_system` to assign per-voxel masses, keeping them consistent.

**Orbit integration**

`physics_system` is symplectic Euler / Euler-Cromer: `v += a·dt` then `x += v·dt`. Phase-space volume is preserved, but unlike Verlet the step is only first-order accurate — so the cap and blend below matter for stability. GPU readback is delayed by one frame, so the same chosen acceleration is used for both updates within a single frame. The chosen GPU acceleration is clamped to `MAX_ACC = 1.5e-3`, then blended toward the GPU value via `GravityBlendFactor` (ramps 0 → 1 over `GRAVITY_BLEND_FRAMES = 60` frames from the first valid GPU result). A Newtonian point-mass anchor is used while the GPU result is non-finite, zero, or not yet ready.

**Gravity shader kernel**

`gravity.wgsl` implements Gaver-Stehfest NILT with M=3 (6 terms); the 12-entry coefficient table in `gravity_pipeline.rs` pads the trailing 6 slots with zeros. The LUT argument simplifies to `as_val = h * s_k = k * ln2` (h cancels), so the LUT is sampled at the 6 nonzero Stehfest points regardless of geometry. The per-voxel contribution is `-(G * mass * (S0+S1) / h²) * unit_dir`, summed across Stehfest terms, then scaled by `ln2/h`. The 80-byte uniform buffer layout is: `[probe_xyz, G, voxel_count, M=3, pad, pad, V[0..12] as 3×vec4<f32>]`.

**Keyboard controls (runtime)**

| Key | Action |
|---|---|
| `S` | Switch camera mode (Overview ↔ Follow Cassini) |
| `F` | Toggle surface normals display |
| `D` | Toggle section plane view |
| `G` | Toggle gravity method (VoxelStehfest ↔ DecomposedWerner) |

**Deployment**

`.github/workflows/deploy.yml` builds with `wasm-pack --release`, assembles `site/` from `index.html + assets/ + pkg/`, and deploys to GitHub Pages. It copies `ryugu_wasm.{js,wasm}` to `Ryugu_wasm.{js,wasm}` to handle Linux case-sensitivity vs macOS case-insensitive FS.

## Dev server

`server.ts` — plain Bun HTTP server on port 3000. Serves from project root. Sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on every response (required for `SharedArrayBuffer` / Bevy WASM threads).

## Bevy 0.19.0 API Notes

- `Aabb` import: `bevy::camera::primitives::Aabb`
- `gizmos.sphere(position, radius, color)` — 3 args, no `Isometry`
- `gizmos.circle(position, radius, color)` — 3 args
- Use `WorldAssetRoot(asset_server.load(...))` to spawn GLTF/OBJ scenes
- WASM canvas: `canvas: Some("#bevy".into())` in `WindowPlugin`
- `index.html` does a `navigator.gpu` + `requestAdapter()` check before importing the WASM bundle; shows a fallback `#no-webgpu` panel if WebGPU is unavailable. Update that panel (not the canvas) when changing browser-support messaging.
- Avoid `std::fs` (panics in WASM) and bare `std::time::Instant`; use `bevy::time::Time`
- `AssetMetaCheck::Never` is required to suppress missing `.meta` file 404s in WASM
- WebGPU backend: `Backends::BROWSER_WEBGPU` (from `bevy::render::settings`)

## WASM dependency gotcha

Any crate that uses random number generation must enable the `wasm_js` feature for `getrandom`:
```toml
getrandom = { version = "0.3", features = ["wasm_js"] }
```
Without this, the WASM build panics at runtime when entropy is requested.
