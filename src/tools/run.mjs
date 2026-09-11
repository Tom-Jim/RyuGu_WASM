import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { root } from "./build_helpers.mjs";

const args = process.argv.slice(2);
const profile = args.includes("--dev") ? "--dev" : "--release";
const forwarded = args.filter(arg => arg !== "--dev" && arg !== "--release");

function run(command, argv) {
  const result = spawnSync(command, argv, { cwd: root, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run(process.execPath, [join(root, "src/tools/build_wasm.mjs"), profile, ...forwarded]);
run("bun", [join(root, "src/server.ts")]);
