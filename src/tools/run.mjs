import { join } from "path";
import { root } from "./build_helpers.mjs";

const args = Bun.argv.slice(2);
const profile = args.includes("--dev") ? "--dev" : "--release";
const forwarded = args.filter(arg => arg !== "--dev" && arg !== "--release");

function run(command, argv) {
  const result = Bun.spawnSync({
    cmd: [command, ...argv],
    cwd: root,
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  if (result.exitCode !== 0) process.exit(result.exitCode ?? 1);
}

run("bun", [join(root, "src/tools/build_wasm.mjs"), profile, ...forwarded]);
run("bun", [join(root, "src/server.ts")]);
