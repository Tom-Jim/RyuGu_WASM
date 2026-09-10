// Compile and inspect modules without instantiating or executing application code.
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join, dirname, resolve } from "node:path";

const root = fileURLToPath(new URL("../../", import.meta.url));
const host = readFileSync(join(root, "src/backend/host/cpp_backend.mjs"), "utf8");
for (const [file, required] of [
  ["pkg/ryugu_backend.wasm", ["memory", "__wasm_call_ctors", "ryugu_scheduler_reset", "ryugu_scheduler_advance", "ryugu_scheduler_time", "ryugu_direct_sum_eval", "ryugu_radial_boost_eval", "ryugu_werner_eval", "ryugu_werner_reset_cache", "ryugu_exafmm_eval", "ryugu_flups_free_space_eval"]],
  ["pkg/backend/ryugu_backend_bg.wasm", ["memory", "configure", "evaluate", "evaluate_sources", "prepare_candidate_sources", "clear_candidate_sources", "advance_frame", "tick", "solve_density", "propagate_candidate", "propagate_candidates", "protocol_version"]],
  ["pkg/ryugu_wasm_bg.wasm", ["memory", "deliver_backend_advance_result", "deliver_backend_evaluate_sources_result", "deliver_backend_candidates_result", "deliver_backend_density_result", "deliver_backend_evaluate_result", "deliver_backend_configure_result"]],
]) {
  const module = await WebAssembly.compile(readFileSync(join(root, file)));
  const exports = new Set(WebAssembly.Module.exports(module).map(entry => entry.name));
  for (const name of required) assert(exports.has(name), `${file}: missing export ${name}`);
  if (file === "pkg/ryugu_backend.wasm") {
    for (const entry of WebAssembly.Module.imports(module)) {
      assert(["env", "wasi_snapshot_preview1"].includes(entry.module), `Unsupported import module ${entry.module}`);
      assert(host.includes(`${entry.name}(`), `Missing host import ${entry.name}`);
    }
  }
  console.log(`${file}: valid WASM and required exports`);
}
for (const file of ["pkg/ryugu_wasm.js", "pkg/backend/ryugu_backend.js"]) {
  const source = readFileSync(join(root, file), "utf8");
  for (const match of source.matchAll(/from\s+['"](\.\/[^'"]+)['"]/g)) {
    assert(existsSync(resolve(root, dirname(file), match[1])), `Missing module ${match[1]}`);
  }
}
assert(existsSync(join(root, "pkg/backend.mjs")), "Missing C++ host");
for (const file of ["src/html/backend-client.js", "src/html/backend-worker.js"]) {
  assert(existsSync(join(root, file)), `Missing numerical Worker source ${file}`);
}
console.log("Static module checks passed; no WASM instantiated or executed.");
