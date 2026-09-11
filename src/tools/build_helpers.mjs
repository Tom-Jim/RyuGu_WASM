import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from "fs";
import { join } from "path";

export const root = join(import.meta.dir, "../..");
export const jobs = String(navigator.hardwareConcurrency || 4);
export const force = Bun.argv.includes("--force");

export function run(command, args, cwd = root, extraEnv) {
  const result = Bun.spawnSync({
    cmd: [command, ...args],
    cwd,
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
    env: extraEnv ? { ...process.env, ...extraEnv } : undefined,
  });
  if (result.exitCode !== 0) throw new Error(`${command} failed: ${result.exitCode}`);
}

export function rmWritable(path) {
  if (!existsSync(path)) return;
  Bun.spawnSync({
    cmd: ["chmod", "-R", "u+w", path],
    stdout: "ignore",
    stderr: "ignore",
  });
  rmSync(path, { recursive: true, force: true });
}

export function latestMtime(paths, options = {}) {
  const skipNames = options.skipNames ?? new Set([
    ".git", "node_modules", "target", ".zig-cache", "zig-out", "deps", "pkg",
  ]);
  const skipPaths = options.skipPaths ?? new Set();
  const exts = options.exts;
  let max = 0;
  const stack = [...paths];
  while (stack.length) {
    const path = stack.pop();
    if (!path || skipPaths.has(path) || !existsSync(path)) continue;
    const stat = statSync(path);
    if (stat.isDirectory()) {
      for (const name of readdirSync(path)) {
        if (skipNames.has(name)) continue;
        stack.push(join(path, name));
      }
      continue;
    }
    if (exts && !exts.some(ext => path.endsWith(ext))) continue;
    if (stat.mtimeMs > max) max = stat.mtimeMs;
  }
  return max;
}

export function isUpToDate(output, inputs, options) {
  if (force || !existsSync(output)) return false;
  return statSync(output).mtimeMs >= latestMtime(inputs, options);
}

export function cmakeConfigure(src, bin, extraArgs) {
  mkdirSync(bin, { recursive: true });
  const args = ["-S", src, "-B", bin, ...extraArgs];
  try {
    run("cmake", args);
  } catch {
    console.warn(`cmake configure failed; clearing pkgRedirects under ${bin}`);
    rmWritable(join(bin, "CMakeFiles", "pkgRedirects"));
    try {
      run("cmake", args);
    } catch {
      rmWritable(join(bin, "CMakeFiles"));
      run("cmake", args);
    }
  }
}

export function cmakeCacheUsable(bin, toolchainFile, inputs) {
  const cache = join(bin, "CMakeCache.txt");
  if (!existsSync(cache)) return false;
  const text = readFileSync(cache, "utf8");
  if (!text.includes(toolchainFile)) return false;
  return statSync(cache).mtimeMs >= latestMtime(inputs);
}
