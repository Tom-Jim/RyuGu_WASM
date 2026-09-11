import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "fs";
import { dirname, join } from "path";
import {
  cmakeCacheUsable, cmakeConfigure, force, isUpToDate, jobs, rmWritable, root, run,
} from "./build_helpers.mjs";

const sources = JSON.parse(readFileSync(join(root, "C++/sources.lock.json"), "utf8"));
if (Bun.argv.includes("--fetch") || Bun.env.RYUGU_FETCH_CPP === "1"
  || Object.keys(sources).some(name => !existsSync(join(root, "C++", name)))) {
  run("bun", ["src/tools/fetch_cpp.mjs"]);
}

const toolchainFile = join(root, "src/backend/zig/wasm-toolchain.cmake");
const cmakeRoot = join(root, "target/cpp-wasm");
const fftwBin = join(cmakeRoot, "fftw");
const flupsBin = join(cmakeRoot, "flups");
const fftwLib = join(fftwBin, "libfftw3.a");
const flupsLib = join(flupsBin, "libflups.a");
const zigDir = join(root, "src/backend/zig");
const zigWasm = join(zigDir, "zig-out/bin/ryugu_backend.wasm");
const pkgWasm = join(root, "pkg/ryugu_backend.wasm");
const host = join(root, "src/backend/host/cpp_backend.mjs");
const cmakeArgs = [`-DCMAKE_TOOLCHAIN_FILE=${toolchainFile}`];
const fftwInputs = [
  join(root, "C++/fftw3-release/CMakeLists.txt"),
  toolchainFile,
  join(root, "C++/sources.lock.json"),
];
const flupsInputs = [
  join(root, "src/backend/zig/flups/CMakeLists.txt"),
  join(root, "src/backend/zig/browser"),
  toolchainFile,
  join(root, "C++/sources.lock.json"),
];

function reuseArchive(oldLib, newLib, oldBin) {
  mkdirSync(dirname(newLib), { recursive: true });
  if (!existsSync(newLib) && existsSync(oldLib)) {
    copyFileSync(oldLib, newLib);
    console.log(`Reusing ${oldLib}`);
  }
  rmWritable(join(oldBin, "CMakeFiles", "pkgRedirects"));
  rmWritable(join(oldBin, "CMakeFiles"));
}

reuseArchive(join(root, "src/backend/zig/deps/fftw/libfftw3.a"), fftwLib, join(root, "src/backend/zig/deps/fftw"));
reuseArchive(join(root, "src/backend/zig/deps/flups/libflups.a"), flupsLib, join(root, "src/backend/zig/deps/flups"));

function buildArchive(src, bin, lib, inputs, extraArgs) {
  if (isUpToDate(lib, inputs)) {
    console.log(`Skipping cmake in ${bin} (archive up to date)`);
    return;
  }
  if (force || !cmakeCacheUsable(bin, toolchainFile, inputs)) {
    cmakeConfigure(src, bin, extraArgs);
  }
  run("cmake", ["--build", bin, "-j", jobs]);
}

buildArchive("C++/fftw3-release", fftwBin, fftwLib, fftwInputs, [
  ...cmakeArgs,
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.10",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DBUILD_SHARED_LIBS=OFF",
  "-DBUILD_TESTS=OFF",
  "-DDISABLE_FORTRAN=ON",
]);
buildArchive("src/backend/zig/flups", flupsBin, flupsLib, flupsInputs, [
  ...cmakeArgs,
  "-DCMAKE_BUILD_TYPE=Release",
]);

const zigInputs = [
  join(zigDir, "build.zig"),
  join(zigDir, "wasm-toolchain.cmake"),
  join(zigDir, "wasm_exports.cpp"),
  join(zigDir, "scheduler.cpp"),
  join(zigDir, "basilisk_bridge.cpp"),
  join(zigDir, "basilisk_bridge.h"),
  join(zigDir, "browser"),
  join(zigDir, "flups"),
  join(root, "C++/sources.lock.json"),
  fftwLib,
  flupsLib,
];
const needsZig = !isUpToDate(pkgWasm, zigInputs) || !existsSync(zigWasm);
if (needsZig) {
  run("zig", [
    "build", "-Dtarget=wasm32-wasi", "-Doptimize=ReleaseFast",
    "-Dexafmm-root=../../../C++/exafmm-t", "-Dflups-root=../../../C++/flups",
    `-Dfftw-archive=${fftwLib}`, `-Dflups-archive=${flupsLib}`,
  ], zigDir);
} else {
  console.log("Skipping zig wasm link (up to date)");
}
if (Bun.argv.includes("--test")) {
  run("bun", ["src/tools/test_cpp_wasm.mjs", "--fmm", "--flups"]);
}
mkdirSync(join(root, "pkg"), { recursive: true });
if (existsSync(zigWasm) && (!existsSync(pkgWasm) || statSync(zigWasm).mtimeMs >= statSync(pkgWasm).mtimeMs)) {
  copyFileSync(zigWasm, pkgWasm);
}
copyFileSync(host, join(root, "pkg/backend.mjs"));
mkdirSync(join(root, "pkg/licenses"), { recursive: true });
for (const dependency of ["basilisk", "boost", "eigen", "exafmm-t", "flups", "fftw3", "fftw3-release", "mpi-serial", "h3lpr"]) {
  const directory = join(root, "C++", dependency);
  if (!existsSync(directory)) continue;
  for (const file of readdirSync(directory)) {
    if (/^(LICENSE|COPYING)([._-].*)?$/.test(file)) {
      copyFileSync(join(directory, file), join(root, "pkg/licenses", `${dependency}-${file}`));
    }
  }
}
