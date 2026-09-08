import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, readdirSync, readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const root = fileURLToPath(new URL("../../", import.meta.url));
function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${command} failed: ${result.status}`);
}
const sources = JSON.parse(readFileSync(join(root, "C++/sources.lock.json"), "utf8"));
if (process.argv.includes("--fetch") || Object.keys(sources).some(name => !existsSync(join(root, "C++", name)))) {
  run(process.execPath, ["src/tools/fetch_cpp.mjs"]);
}
const toolchain = `-DCMAKE_TOOLCHAIN_FILE=${join(root, "src/backend/zig/wasm-toolchain.cmake")}`;
run("cmake", ["-S", "C++/fftw3-release", "-B", "src/backend/zig/deps/fftw", toolchain,
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DBUILD_SHARED_LIBS=OFF", "-DBUILD_TESTS=OFF", "-DDISABLE_FORTRAN=ON"]);
run("cmake", ["--build", "src/backend/zig/deps/fftw", "-j", "4"]);
run("cmake", ["-S", "src/backend/zig/flups", "-B", "src/backend/zig/deps/flups", toolchain]);
run("cmake", ["--build", "src/backend/zig/deps/flups", "-j", "4"]);
run("zig", ["build", "-Dtarget=wasm32-wasi", "-Doptimize=ReleaseFast",
  "-Dexafmm-root=../../../C++/exafmm-t", "-Dflups-root=../../../C++/flups"], join(root, "src/backend/zig"));
if (process.argv.includes("--test")) {
  run(process.execPath, ["src/tools/test_cpp_wasm.mjs", "--fmm", "--flups"]);
}
mkdirSync(join(root, "pkg"), { recursive: true });
copyFileSync(join(root, "src/backend/zig/zig-out/bin/ryugu_backend.wasm"), join(root, "pkg/ryugu_backend.wasm"));
copyFileSync(join(root, "src/backend/host/cpp_backend.mjs"), join(root, "pkg/backend.mjs"));
mkdirSync(join(root, "pkg/licenses"), { recursive: true });
for (const dependency of ["basilisk", "boost", "eigen", "exafmm-t", "flups", "fftw3", "fftw3-release", "mpi-serial", "h3lpr"]) {
  for (const file of readdirSync(join(root, "C++", dependency))) {
    if (/^(LICENSE|COPYING)([._-].*)?$/.test(file)) {
      copyFileSync(join(root, "C++", dependency, file), join(root, "pkg/licenses", `${dependency}-${file}`));
    }
  }
}
