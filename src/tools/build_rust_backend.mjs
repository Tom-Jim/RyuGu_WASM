import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const cwd = fileURLToPath(new URL("../backend/rust/", import.meta.url));
const result = spawnSync("wasm-pack", ["build", "--locked", "--release", "--target", "web",
  "--out-dir", "../../../pkg/backend", "--out-name", "ryugu_backend"], {
  cwd, stdio: "inherit", env: { ...process.env, RUSTC_WRAPPER: "" },
});
if (result.status !== 0) throw new Error(`Rust backend build failed: ${result.status}`);
