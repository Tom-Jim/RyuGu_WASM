# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# JS dependencies
bun install

# Dev: builds WASM (debug) then starts server at http://localhost:3000
bun run dev

# Production WASM build (release profile, output → pkg/)
bun run build

# Start server only (requires pkg/ to already exist)
bun run serve

# Lint / format (native host, for IDE feedback only)
cargo clippy
cargo fmt
```

> `cargo run` and `cargo build` without a target are only useful for syntax checking; the actual runnable artifact is the WASM package built by `wasm-pack`.

## Architecture

Single-file Bevy app (`src/lib.rs`) compiled to WASM via `wasm-pack`. The entry point is `main()` annotated with `#[wasm_bindgen(start)]`, mounted on the `#bevy` canvas in `index.html`.

**Scene**: Asteroid Ryugu (900 m diameter) + Cassini probe (6.7 m) loaded as GLTF scenes. Both use `WorldAssetRoot` + `TargetSize` for deferred auto-scaling once Bevy computes their `Aabb`.

**Systems**:
- `normalize_model_scale_system` — runs every frame until `ScaleNormalized` is inserted; walks the entity tree to find the max AABB extent and applies a uniform scale factor so the model matches its real-world `TargetSize` in meters.
- `probe_tracker_system` — renders a pulsing gizmo crosshair on the Cassini probe when the camera is more than 250 m away.

**Dev server** (`server.ts`): plain Bun HTTP server with COEP/COOP headers required for `SharedArrayBuffer` / WASM threads.

## Bevy 0.19.0 API Notes

- `Aabb` import path: `bevy::camera::primitives::Aabb`.
- `gizmos.sphere(position, radius, color)` — 3 args, no `Isometry`.
- `gizmos.circle(position, radius, color)` — 3 args.
- Use `WorldAssetRoot(asset_server.load(...))` to spawn GLTF scenes.
- WASM canvas: `canvas: Some("#bevy".into())` in `WindowPlugin`.
- Avoid `std::fs` and bare `std::time::Instant`; use `bevy::time::Time` instead.
- `AssetMetaCheck::Never` is required to suppress missing `.meta` file errors in WASM.
