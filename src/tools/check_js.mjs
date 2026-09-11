import { mkdirSync, rmSync } from "fs";
import { join } from "path";
import { root } from "./build_helpers.mjs";

const browser = [
  "src/html/navigation.js",
  "src/html/ui.js",
  "src/html/capture.js",
  "src/html/visit.js",
  "src/html/app.js",
  "src/html/background.js",
  "src/html/background-worker.js",
  "src/html/backend-client.js",
  "src/html/backend-worker.js",
  "src/backend/host/cpp_backend.mjs",
];
const tooling = [
  "src/tools/build_helpers.mjs",
  "src/tools/build_cpp_wasm.mjs",
  "src/tools/build_rust_backend.mjs",
  "src/tools/build_wasm.mjs",
  "src/tools/run.mjs",
  "src/tools/check_js.mjs",
  "src/tools/check_wasm_build.mjs",
  "src/tools/fetch_cpp.mjs",
  "src/tools/normalize_frontend_wasm.mjs",
  "src/tools/test_cpp_wasm.mjs",
  "src/server.ts",
];
const extras = Bun.argv.slice(2);

function parse(file, target) {
  const outfile = join(dir, "check.js");
  const result = Bun.spawnSync({
    cmd: ["bun", "build", file, `--target=${target}`, `--outfile=${outfile}`],
    cwd: root,
    stdout: "ignore",
    stderr: "inherit",
  });
  if (result.exitCode !== 0) {
    throw new Error(`Bun could not parse ${file}`);
  }
}

const dir = join(Bun.env.TMPDIR || "/tmp", `ryugu-jscheck-${crypto.randomUUID()}`);
mkdirSync(dir, { recursive: true });
try {
  for (const file of browser) parse(file, "browser");
  for (const file of tooling) parse(file, "bun");
  for (const file of extras) parse(file, "browser");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
console.log("PASS: Bun parsed the JavaScript and TypeScript tooling sources");
