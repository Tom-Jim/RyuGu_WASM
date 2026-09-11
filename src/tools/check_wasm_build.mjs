// Compile and inspect modules without instantiating or executing application code.
import { dirname, join, resolve } from "path";

const root = join(import.meta.dir, "../..");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function readText(path) {
  return await Bun.file(path).text();
}

async function exists(path) {
  return await Bun.file(path).exists();
}

const host = await readText(join(root, "src/backend/host/cpp_backend.mjs"));

// Source-level contract first: every request kind the frontend glue posts must
// be executed by the Worker and routed back by the page, every delivery export
// the page calls must exist in the frontend crate, and the Worker must be the
// only place that instantiates the numerical modules.
for (const file of ["src/html/backend-client.js", "src/html/backend-worker.js"]) {
  assert(await exists(join(root, file)), `Missing numerical Worker source ${file}`);
}
const glue = await readText(join(root, "src/cpp_backend.rs"));
const worker = await readText(join(root, "src/html/backend-worker.js"));
const page = await readText(join(root, "src/html/index.html"));
const requestedKinds = new Set(
  [...glue.matchAll(/(?:post_backend_request|\.request)\(\s*'([a-z_]+)'/g)].map(match => match[1]),
);
assert(requestedKinds.size > 0, "No numerical request kinds found in src/cpp_backend.rs");
const workerKinds = new Set([...worker.matchAll(/case '([a-z_]+)':/g)].map(match => match[1]));
const pageKinds = new Set([...page.matchAll(/case '([a-z_]+)':/g)].map(match => match[1]));
for (const kind of requestedKinds) {
  assert(workerKinds.has(kind), `backend-worker.js does not execute request kind ${kind}`);
  assert(pageKinds.has(kind), `index.html does not route Worker result kind ${kind}`);
}
for (const kind of workerKinds) {
  assert(requestedKinds.has(kind), `backend-worker.js handles unused request kind ${kind}`);
}
const deliveryExports = new Set([...glue.matchAll(/\b(deliver_backend_[a-z_]+_result)\b/g)].map(match => match[1]));
const pageDeliveries = new Set([...page.matchAll(/wasm\.(deliver_backend_[a-z_]+_result)\(/g)].map(match => match[1]));
for (const name of pageDeliveries) {
  assert(deliveryExports.has(name), `index.html calls unknown frontend export ${name}`);
}
for (const name of deliveryExports) {
  assert(pageDeliveries.has(name), `index.html never delivers into ${name}`);
}
for (const module of ["pkg/backend.mjs", "pkg/backend/ryugu_backend.js", "pkg/ryugu_backend.wasm"]) {
  assert(!page.includes(module), `index.html must not load ${module} on the main thread`);
}
console.log(`Numerical Worker contract consistent (${requestedKinds.size} request kinds, ${deliveryExports.size} deliveries).`);

// Built artifacts: valid WASM with the exports the contract above relies on.
for (const [file, required] of [
  ["pkg/ryugu_backend.wasm", ["memory", "__wasm_call_ctors", "ryugu_scheduler_reset", "ryugu_scheduler_advance", "ryugu_scheduler_time", "ryugu_direct_sum_eval", "ryugu_radial_boost_eval", "ryugu_werner_eval", "ryugu_werner_reset_cache", "ryugu_exafmm_eval", "ryugu_flups_free_space_eval"]],
  ["pkg/backend/ryugu_backend_bg.wasm", ["memory", "configure", "evaluate", "evaluate_sources", "prepare_candidate_sources", "clear_candidate_sources", "set_frequency_domain_modes", "advance_frame", "tick", "solve_density", "propagate_candidate", "propagate_candidates", "protocol_version"]],
  ["pkg/ryugu_wasm_bg.wasm", ["memory", ...deliveryExports]],
]) {
  const module = await WebAssembly.compile(await Bun.file(join(root, file)).bytes());
  const exports = new Set(WebAssembly.Module.exports(module).map(entry => entry.name));
  for (const name of required) assert(exports.has(name), `${file}: missing export ${name} (rebuild with \`bun run build\`)`);
  if (file === "pkg/ryugu_backend.wasm") {
    for (const entry of WebAssembly.Module.imports(module)) {
      assert(["env", "wasi_snapshot_preview1"].includes(entry.module), `Unsupported import module ${entry.module}`);
      assert(host.includes(`${entry.name}(`), `Missing host import ${entry.name}`);
    }
  }
  console.log(`${file}: valid WASM and required exports`);
}
for (const file of ["pkg/ryugu_wasm.js", "pkg/backend/ryugu_backend.js"]) {
  const source = await readText(join(root, file));
  for (const match of source.matchAll(/from\s+['"](\.\/[^'"]+)['"]/g)) {
    assert(await exists(resolve(root, dirname(file), match[1])), `Missing module ${match[1]}`);
  }
}
assert(await exists(join(root, "pkg/backend.mjs")), "Missing C++ host");
console.log("Static module checks passed; no WASM instantiated or executed.");
