# Ryugu WASM

A WebGPU-accelerated 3D visualization of asteroid (162173) Ryugu with real-time gravitational dynamics, built with Rust and Bevy Engine, compiled to WebAssembly.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-WebGPU_Simulation-success?style=for-the-badge)](https://tom-jim.github.io/RyuGu_WASM/)
![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Bevy](https://img.shields.io/badge/Bevy-0.19.0-purple.svg)
![Rust](https://img.shields.io/badge/Rust-Edition_2024-orange.svg)

<!-- 1. 顶部 Hero GIF 动图演示 -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/f6fe44af-676f-4436-87c1-21b7f67a8e3a" width="100%" alt="Ryugu WebGPU Real-time Simulation"/>
  <br>
  <i>Real-time 60 FPS gravitational dynamics & orbital trajectory simulation running directly in browser via WebGPU & WASM.</i>
</p>

## 📸 Visual Highlights

<!-- 2. 4 张静态截图 (2×2 网格，屏占比与清晰度最佳) -->
| Orbital Trajectory Overview | Precession Demonstration |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/cdca307b-7b0b-4eca-a28b-96cc822f8069" width="100%"/> | <img src="https://github.com/user-attachments/assets/3c152cf3-071a-4807-8fdd-6c981ec3f4d4" width="100%"/> |
| **Detector display** | **Normal vector display** |
| <img src="https://github.com/user-attachments/assets/977a59b3-a0b1-44b4-8426-2237dbcac1fe" width="100%"/> | <img src="https://github.com/user-attachments/assets/d914c5f5-9a68-44ee-8ed6-0b75625bd312" width="100%"/> |
## Features

### Physics Simulation
- **GPU-Accelerated Gravity**: Massively parallel gravitational force evaluation via WebGPU compute shaders
- **Non-Uniform Density Model**: Radial `1/r` density distribution with mass-conserving normalization
- **Orbital Mechanics**: Real-time Cassini probe trajectory integration around Ryugu
- **Asteroid Rotation**: Physically accurate 7.63-hour spin period with real spin-axis orientation

### Visualization
- **Real-Time 3D Rendering**: GLTF model rendering with WebGPU backend
- **Interactive Camera**: Pan-orbit with overview / follow-probe dual modes
- **Orbit Trail**: 3500-point trajectory history
- **GPU-Computed Surface Normals**: Topology-aware parallel normal computation
- **Cross-Section View**: Live density-field slice with rainbow-thermal colormap

### Scientific Parameters
| Parameter | Value |
|---|---|
| Ryugu mass | 4.5 × 10¹¹ kg |
| Ryugu diameter | ~900 m |
| Cassini probe length | 6.7 m |
| Rotation period | 7.63 hours |
| Spin axis (XYZ) | (−0.043, −0.914, 0.405) |
| G | 6.6743 × 10⁻¹¹ m³ kg⁻¹ s⁻² |
| Time acceleration | 6000× real-time |

## Architecture

### Data Flow
```
GLTF Mesh
   │
   ├─► Vertex Welding  ──► Half-Edge Topology
   │                              │
   │                    4-pt Gaussian Quadrature
   │                    ∫(1/(r+ε)) dV  →  DensityC
   │
   └─► Scale Normalization
              │
        GPU Voxel Buffer  (x, y, z, mass) × N  — built once
              │
        Compute Shader   ─── dispatched every frame
              │
        Readback Channel
              │
        Physics Integrator  ──► Cassini Transform
```

### Systems
| File | Responsibility |
|---|---|
| `scale.rs` | AABB-based model normalization; 4-pt Gauss quadrature for density constant |
| `topology.rs` | Half-edge mesh construction from welded vertices |
| `welding.rs` | Vertex deduplication with configurable spatial tolerance |
| `gravity_pipeline.rs` | Voxel buffer build + GPU compute dispatch + readback |
| `compute_pipeline.rs` | GPU normal computation (NormalsComputePlugin) |
| `physics.rs` | Orbital integration + Ryugu rotation |
| `render.rs` | Scene setup, gizmos, cross-section renderer, camera |
| `ui.rs` | FPS counter, key-hint text, input toggles |

### Shaders
- `assets/shaders/gravity.wgsl` — 64-thread workgroup parallel gravity reduction
- `assets/shaders/compute_normals.wgsl` — topology-indexed normal averaging

## Commands

```sh
# Install JS dependencies
bun install

# Dev: build WASM (debug) then start server at http://localhost:3000
bun run dev

# Production WASM build (release, output → pkg/)
bun run build

# Start server only (requires pkg/ to already exist)
bun run serve

# Syntax / lint check (native host)
cargo clippy
cargo fmt
```

> `cargo run` / `cargo build` without a WASM target are useful only for syntax checks.
> The actual runnable artifact is the WASM package produced by `wasm-pack`.

## Browser Requirements

WebGPU is required. Supported environments:

| Browser | Minimum Version |
|---|---|
| Chrome / Chromium | 113+ |
| Edge | 113+ |
| Safari | 18+ |
| Firefox | Not yet supported (WebGPU behind flag) |

The dev server (`server.ts`) sets `Cross-Origin-Embedder-Policy: require-corp` and
`Cross-Origin-Opener-Policy: same-origin` headers — required for `SharedArrayBuffer` / WASM threads.

## Project Structure

```
Ryugu_wasm/
├── assets/
│   ├── models/
│   │   ├── ryugu.glb          # Asteroid mesh  (~900 m diameter)
│   │   └── cassini.gltf       # Probe mesh     (~6.7 m)
│   └── shaders/
│       ├── gravity.wgsl
│       └── compute_normals.wgsl
├── src/
│   ├── lib.rs                 # App entry point (wasm_bindgen start)
│   ├── components.rs          # ECS components, constants, shared resources
│   ├── topology.rs            # Half-edge mesh
│   ├── welding.rs             # Vertex deduplication
│   └── systems/
│       ├── mod.rs
│       ├── compute_normals.rs
│       ├── compute_pipeline.rs
│       ├── gravity_pipeline.rs
│       ├── physics.rs
│       ├── render.rs
│       ├── scale.rs
│       └── ui.rs
├── index.html
├── server.ts                  # Bun HTTP dev server
├── package.json
└── Cargo.toml
```

## Physical Model

### Density Field

Internal mass distribution follows a radial power law:

$$\rho(r) = \frac{C}{r + \varepsilon}$$

The normalization constant *C* is solved from mass conservation:

$$C = \frac{M_{\text{Ryugu}}}{\displaystyle\int_V \frac{dV}{r + \varepsilon}}$$

Integration uses **4-point Gaussian quadrature** over the signed tetrahedral decomposition of the mesh.  
Each surface triangle forms a tetrahedron with the origin; volumes can be negative (back-faces cancel self-intersections automatically).

### GPU Gravity Kernel

Each frame, the compute shader evaluates:

$$\mathbf{g} = G \sum_{i=1}^{N} \frac{m_i\,(\mathbf{r}_i - \mathbf{r}_{\text{probe}})}{\lVert \mathbf{r}_i - \mathbf{r}_{\text{probe}} \rVert^3}$$

Work is split into 64-thread workgroups; each writes a partial sum to the output buffer.
The CPU reads back the partial sums and reduces them to a single `Vec3`.

### Mass Conservation

Voxel masses are normalized so that:

$$\sum_{i} m_i = M_{\text{Ryugu}}$$

Each vertex receives mass proportional to its `1/r` weight:

```
mass_i = (weight_i / Σ weight_j) × M_Ryugu
```

## Interactive Controls

| Key | Action |
|---|---|
| `S` | Toggle camera: Overview ↔ Follow Cassini |
| `F` | Toggle surface normal arrows |
| `D` | Toggle cross-section density view |
| Mouse drag | Rotate (pan-orbit) |
| Scroll wheel | Zoom |

### Cross-Section View (D key)

The section plane is perpendicular to the camera viewing direction and passes through Ryugu's center.  
Sampling grid: 31 × 31 points, clipped to the asteroid boundary by projecting against surface vertex positions.  
Color mapping — linear `1/r` density normalized to [0, 1]:

| Color | Region |
|---|---|
| Red | Center (highest density) |
| Yellow → Cyan | Mid-radius |
| Blue | Surface edge (lowest density) |

The asteroid mesh turns semi-transparent (α = 0.25) while section view is active.

## Bevy 0.19 API Notes

- `Aabb` import: `bevy::camera::primitives::Aabb`
- `gizmos.sphere(position, radius, color)` — 3 args, no `Isometry`
- `gizmos.circle(position, radius, color)` — 3 args
- Use `WorldAssetRoot(asset_server.load(...))` to spawn GLTF scenes
- WASM canvas: `canvas: Some("#bevy".into())` in `WindowPlugin`
- `AssetMetaCheck::Never` suppresses missing `.meta` file errors in WASM
- Avoid `std::fs` and `std::time::Instant`; use `bevy::time::Time`

## Build Profiles

```toml
[profile.dev]
opt-level = 1          # Faster incremental builds

[profile.dev.package."*"]
opt-level = 3          # Dependencies still optimized in dev

[profile.release]
opt-level = "z"        # Minimize binary size
lto = true             # Link-time optimization
codegen-units = 1      # Maximum optimization
```

## Mathematical Background

The gravity kernel derivation — including Laplace-domain operator decomposition,
complex contour integration, modified Bessel function spectral modes, and boundary term
cancellation proofs 

## References

- Watanabe S. et al. (2019). Hayabusa2 arrives at the carbonaceous asteroid 162173 Ryugu.
  *Science* 364, 268–272.
- JAXA Hayabusa2 Mission: https://www.hayabusa2.jaxa.jp/en/
- Bevy Engine: https://bevyengine.org/
- W3C WebGPU Specification: https://www.w3.org/TR/webgpu/

## License

Copyright (C) 2026 Ryugu WASM Contributors

This program is free software: you can redistribute it and/or modify it under the terms of the
**GNU General Public License v3.0** as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**;
without even the implied warranty of **MERCHANTABILITY** or **FITNESS FOR A PARTICULAR PURPOSE**.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.
