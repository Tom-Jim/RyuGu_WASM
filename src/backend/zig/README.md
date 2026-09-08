# C++ Numerical WASM Backend

Zig 0.16 compiles Basilisk serial task scheduling, polyhedral gravity, Boost quadrature, ExaFMM-t,
FLUPS and FFTW into a single `wasm32-wasi` module. The browser-compatible
JavaScript host is `src/backend/host/cpp_backend.mjs`; it does not need a server-side
runtime or Node's WASI implementation.

Run from the project root:

```sh
node src/tools/fetch_cpp.mjs
node src/tools/build_cpp_wasm.mjs
```

The build writes `pkg/ryugu_backend.wasm`, `pkg/backend.mjs`, and dependency
license notices. Numerical tests run only with the explicit `--test` flag. Dependencies
are Git checkouts in `C++`, pinned in `C++/sources.lock.json`. Browser patches
are versioned under `C++/patches` and applied by the fetch script.

`C++/fftw3` is FFTW's official development checkout. `C++/fftw3-release` is
Debian's upstream release-source mirror, including generated FFTW codelets.
CMake invokes Zig to compile FFTW and FLUPS; no native FFTW/MPI binaries are
linked into the module.

ExaFMM uses its upstream scalar near-field branch and full FMM translations.
Eigen supplies real BLAS/SVD calls. Matrix disk caching is disabled. FLUPS
uses its all-to-all implementation with MPI-SERIAL in one process. OpenMP,
diagnostic filesystem exports and native backtraces are disabled.

Probe gravity, planning, inversion sensitivities, and surface-field dispatch
call this module through `src/cpp_backend.rs` and `src/cpp_planning.rs`.
The Pages build packages this C++ module, independent Rust backend WASM,
and Bevy frontend WASM. `scheduler.cpp` uses upstream SimModel, SysProcess,
and SysModelTask with the versioned browser-serial patch. Its task calls
the Rust backend and publishes SCStatesMsgPayload through Basilisk messaging.
The exported scheduler API is `ryugu_scheduler_reset`,
`ryugu_scheduler_advance`, and `ryugu_scheduler_time`, using nanoseconds.
Native threads, Python bindings, and complete spacecraft/Vizard behavior
are not provided. The old `ryugu_basilisk_step` is not exported or called.

Direct/radial/Werner/FMM return positive potential. FLUPS currently returns
negative potential and `-grad(phi)` acceleration; the application adapter
converts its potential to positive U. See the root README for the remaining
architecture, performance, and validation limits.
