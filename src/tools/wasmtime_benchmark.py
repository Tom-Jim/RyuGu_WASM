"""Benchmark the generated Ryugu WASM numeric kernel with Wasmtime.

The browser UI measures complete Bevy/WebGPU frame throughput. This script is
the reproducible headless companion: it loads the generated ``*_bg.wasm``,
calls the exported deterministic kernel benchmark, and reports compilation,
instantiation, and repeated-call throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import wasmtime


def _wasm_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = sorted(Path("pkg").glob("*_bg.wasm"))
    if not candidates:
        raise FileNotFoundError("No generated pkg/*_bg.wasm found; run `uv run wasm-pack build ...` first.")
    return candidates[0]


def _compile(engine: wasmtime.Engine, wasm: bytes) -> wasmtime.Module:
    return wasmtime.Module(engine, wasm)


def benchmark(path: Path, calls: int, iterations: int) -> dict[str, object]:
    engine = wasmtime.Engine()
    wasm = path.read_bytes()

    compile_samples: list[float] = []
    module = None
    for _ in range(calls):
        start = time.perf_counter()
        module = _compile(engine, wasm)
        compile_samples.append(time.perf_counter() - start)
    assert module is not None

    store = wasmtime.Store(engine)
    linker = wasmtime.Linker(engine)
    instance = None
    instantiate_error = None
    start = time.perf_counter()
    try:
        # The generated Bevy module contains browser imports. Instantiation is
        # attempted explicitly so the report distinguishes compilation from a
        # browser-only import graph; the numeric export remains benchmarkable
        # whenever the generated artifact exposes it without DOM imports.
        instance = linker.instantiate(store, module)
    except Exception as exc:  # pragma: no cover - depends on generated imports
        instantiate_error = str(exc)
    instantiate_seconds = time.perf_counter() - start

    call_samples: list[float] = []
    checksum = None
    if instance is not None:
        export = instance.exports(store).get("benchmark_gravity_algorithms")
        if export is not None:
            for _ in range(calls):
                start = time.perf_counter()
                checksum = export(store, iterations)
                call_samples.append(time.perf_counter() - start)

    result: dict[str, object] = {
        "wasm": str(path),
        "bytes": len(wasm),
        "calls": calls,
        "iterations_per_call": iterations,
        "compile_mean_ms": statistics.mean(compile_samples) * 1e3,
        "compile_p95_ms": sorted(compile_samples)[max(0, int(0.95 * len(compile_samples)) - 1)] * 1e3,
        "instantiate_ms": instantiate_seconds * 1e3,
        "numeric_call_mean_ms": statistics.mean(call_samples) * 1e3 if call_samples else None,
        "numeric_checksum": checksum,
        "instantiation_error": instantiate_error,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm", help="Path to generated *_bg.wasm")
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    result = benchmark(_wasm_path(args.wasm), max(1, args.calls), max(1, args.iterations))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"WASM: {result['wasm']} ({result['bytes']:,} bytes)")
        print(f"compile mean/p95: {result['compile_mean_ms']:.3f}/{result['compile_p95_ms']:.3f} ms")
        print(f"instantiate: {result['instantiate_ms']:.3f} ms")
        print(f"numeric export mean: {result['numeric_call_mean_ms']!s} ms")
        if result["instantiation_error"]:
            print("note: browser imports prevented headless instantiation")


if __name__ == "__main__":
    main()
