import { existsSync, unlinkSync } from "fs";
import { join } from "path";
import { isUpToDate, root, run } from "./build_helpers.mjs";

const crate = join(root, "src/backend/rust");
const output = join(root, "pkg/backend/ryugu_backend_bg.wasm");
if (isUpToDate(output, [join(crate, "src"), join(crate, "Cargo.toml"), join(crate, "Cargo.lock")])) {
  console.log("Skipping Rust backend WASM (up to date)");
} else {
  const manifest = join(root, "pkg/backend/package.json");
  if (existsSync(manifest)) unlinkSync(manifest);
  run("wasm-pack", [
    "build", "--locked", "--release", "--target", "web",
    "--out-dir", "../../../pkg/backend", "--out-name", "ryugu_backend",
  ], crate, { RUSTC_WRAPPER: "" });
}
