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

## ✨ Features

### High-Performance Computational Physics

* **Laplace-Domain Gravity Kernel**: A mathematical breakthrough eliminating the need for artificial softening ($\epsilon$) in near-boundary N-body simulations.
* **GPU-Accelerated NILT**: Highly parallelized Numerical Inverse Laplace Transform (Gaver-Stehfest) executed entirely within WebGPU compute shaders.
* **Non-Keplerian Secular Perturbation**: Real-time simulation of apsidal precession and nodal regression caused by Ryugu's equatorial ridge and inhomogeneous density.
* **Asteroid Rotation**: Physically accurate 7.63-hour spin period with real spin-axis orientation.



### Scientific Visualization

* **Real-Time 3D Rendering**: GLTF model rendering with a WebGPU backend.


* **GPU-Computed Surface Normals**: Topology-aware parallel normal computation.


* **Cross-Section View**: Live density-field slice with a rainbow-thermal colormap mapped to internal physics.



### Scientific Parameters

| Parameter | Value |
| --- | --- |
| Ryugu mass | 4.5 × 10¹¹ kg

 |
| Ryugu diameter | ~900 m

 |
| Cassini probe length | 6.7 m

 |
| Rotation period | 7.63 hours

 |
| Spin axis (XYZ) | (−0.043, −0.914, 0.405)

 |
| G | 6.6743 × 10⁻¹¹ m³ kg⁻¹ s⁻²

 |
| Time acceleration | 60000× real-time |

## 📐 Mathematical Foundation

Traditional numerical integration of potential fields faces a severe challenge: the $1/R^3$ singularity near the boundary. Conventional methods (like Boundary Element Methods or discrete Vortex Methods) rely on artificial cut-off radii or geometric approximations, leading to unstable dynamics in grazing orbits.

This simulator utilizes a novel **Spatial-to-Laplace Domain Mapping**:

1. **Analytical Kernel Formulation**: The spatial integrals are mathematically mapped into the complex frequency domain (s-domain). By evaluating the contour integrals, the spatial singularities are analytically annihilated and transformed into harmonic oscillations and spectral decay modes.
2. **Struve-Neumann Spectral Modes**: The resulting analytical closed-form solution leverages a combination of Struve and Neumann (Bessel of the second kind) functions, which naturally absorb the near-field divergence without requiring an artificial $\epsilon$.
3. **Gaver-Stehfest Inversion**: To return to the spatial domain, a robust Numerical Inverse Laplace Transform (NILT) is employed, ensuring absolute energy conservation and highly accurate non-Keplerian perturbation tracking for the spacecraft.

*(Note: The full algebraic proofs of the boundary-term annihilation and contour integration are part of ongoing research and are abstracted within the compute shader pipeline).*

## ⚡ GPU Compute Architecture

Executing complex special functions (Struve-Neumann) and NILT in a real-time rendering loop traditionally causes massive CPU bottlenecking and catastrophic floating-point cancellation. This project achieves a **60 FPS** performance via a highly optimized hardware-algorithm co-design:

* **LUT-Based Dimensionality Reduction**: The complex spectral modes are pre-computed in Python and packed into an interleaved 1D Lookup Table (LUT). The WGSL Compute Shader performs $O(1)$ hardware-accelerated linear interpolation (`mix`), bypassing expensive runtime series expansions.
* **In-Shader Parallel NILT**: The Gaver-Stehfest inversion loop is pushed down to the individual thread level. Real-time evaluations are dynamically accumulated using strict memory-aligned structs to prevent WebGPU out-of-bounds crashes.
* **Tree-Reduction Synchronization**: The global force summation over the non-uniform asteroid mesh is completed using workgroup-shared memory and bitwise parallel stride reduction, drastically minimizing VRAM latency.

## 🛠️ Build & Run Commands

```sh
# Install JS dependencies[cite: 12]
bun install

# Dev: build WASM (debug) then start server at http://localhost:3000[cite: 12]
bun run dev

# Production WASM build (release, output → pkg/)[cite: 12]
bun run build

# Start server only (requires pkg/ to already exist)[cite: 12]
bun run serve

# Syntax / lint check (native host)[cite: 12]
cargo clippy
cargo fmt

```

> `cargo run` / `cargo build` without a WASM target are useful only for syntax checks.
> The actual runnable artifact is the WASM package produced by `wasm-pack`.
> 
> 

## 🌐 Browser Requirements

WebGPU is required. Supported environments:

| Browser | Minimum Version |
| --- | --- |
| Chrome / Chromium | 113+

 |
| Edge | 113+

 |
| Safari | 18+

 |
| Firefox | Not yet supported (WebGPU behind flag)

 |

The dev server (`server.ts`) sets `Cross-Origin-Embedder-Policy: require-corp` and
`Cross-Origin-Opener-Policy: same-origin` headers — required for `SharedArrayBuffer` / WASM threads.

## 🎮 Interactive Controls

| Key | Action |
| --- | --- |
| `S` | Toggle camera: Overview ↔ Follow Cassini

 |
| `F` | Toggle surface normal arrows

 |
| `D` | Toggle cross-section density view

 |
| Mouse drag | Rotate (pan-orbit)

 |
| Scroll wheel | Zoom

 |

## 📚 References

* Watanabe S. et al. (2019). Hayabusa2 arrives at the carbonaceous asteroid 162173 Ryugu. *Science* 364, 268–272.


* JAXA Hayabusa2 Mission: [https://www.hayabusa2.jaxa.jp/en/](https://www.hayabusa2.jaxa.jp/en/)

* Bevy Engine: [https://bevyengine.org/](https://bevyengine.org/)

* W3C WebGPU Specification: [https://www.w3.org/TR/webgpu/](https://www.w3.org/TR/webgpu/)


## 📜 License

Copyright (C) 2026 Ryugu WASM Contributors

This program is free software: you can redistribute it and/or modify it under the terms of the
**GNU General Public License v3.0** as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.