import { existsSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { isUpToDate, root, run } from "./build_helpers.mjs";

const args = Bun.argv.slice(2);
const profile = args.includes("--dev") ? "dev" : "release";
const extra = [];
if (args.includes("--force")) extra.push("--force");
if (args.includes("--fetch") || Bun.env.RYUGU_FETCH_CPP === "1") extra.push("--fetch");

run("bun", ["src/tools/build_cpp_wasm.mjs", ...extra]);
run("bun", ["src/tools/build_rust_backend.mjs", ...extra.filter(flag => flag !== "--fetch")]);

const css = join(root, "src/html/tailwind.generated.css");
if (!isUpToDate(css, [join(root, "src/html/tailwind.css"), join(root, "src/html/index.html"), join(root, "src/html/app.js")])) {
  run("bun", ["run", "styles"]);
} else {
  console.log("Skipping Tailwind stylesheet (up to date)");
}

const frontend = join(root, "pkg/ryugu_wasm_bg.wasm");
const stamp = join(root, "pkg/.wasm-profile");
const previous = existsSync(stamp) ? readFileSync(stamp, "utf8").trim() : "";
const frontendInputs = [
  join(root, "Cargo.toml"),
  join(root, "Cargo.lock"),
  join(root, "src"),
  join(root, "src/html/rust"),
];
const skipPaths = new Set([
  join(root, "src/backend/zig"),
  join(root, "src/backend/rust"),
  join(root, "src/backend/host"),
  join(root, "src/tools"),
  join(root, "src/html"),
]);
const profileChanged = previous !== "" && previous !== profile;
if (profileChanged || !isUpToDate(frontend, frontendInputs, {
  exts: [".rs", ".wgsl", ".toml", ".lock"],
  skipPaths,
})) {
  run("wasm-pack", [
    "build", "--locked", `--${profile}`, "--target", "web", "--out-dir", "pkg", "--out-name", "ryugu_wasm",
  ], root, { RUSTC_WRAPPER: "" });
  run("bun", ["src/tools/normalize_frontend_wasm.mjs"]);
  writeFileSync(stamp, profile);
} else {
  if (!previous) writeFileSync(stamp, profile);
  console.log(`Skipping frontend WASM (${profile} up to date)`);
}
