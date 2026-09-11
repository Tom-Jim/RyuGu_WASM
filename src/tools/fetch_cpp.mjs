import { existsSync } from "fs";
import { join } from "path";

const root = join(import.meta.dir, "../../C++");
const sources = await Bun.file(join(root, "sources.lock.json")).json();

function git(args) {
  const result = Bun.spawnSync({
    cmd: ["git", ...args],
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode !== 0) throw new Error(result.stderr.toString() || "git failed");
  return result.stdout.toString().trim();
}

for (const [name, source] of Object.entries(sources)) {
  const path = join(root, name);
  if (!existsSync(path)) {
    const args = ["clone", "--depth", "1"];
    if (source.sparse) args.push("--filter=blob:none", "--sparse");
    if (source.branch) args.push("--branch", source.branch);
    git([...args, source.url, path]);
    if (git(["-C", path, "rev-parse", "HEAD"]) !== source.commit) {
      git(["-C", path, "fetch", "--depth", "1", "origin", source.commit]);
      git(["-C", path, "checkout", "--detach", source.commit]);
    }
    if (source.sparse) git(["-C", path, "sparse-checkout", "set", ...source.sparse]);
  }
  if (git(["-C", path, "rev-parse", "HEAD"]) !== source.commit) {
    throw new Error(`${name}: checkout differs from lock; refusing to overwrite`);
  }
  if (source.submodules) git(["-C", path, "submodule", "update", "--init", "--depth", "1", ...source.submodules]);
  if (name === "exafmm-t" || name === "flups" || name === "basilisk") {
    const patch = join(root, "patches", name === "exafmm-t" ? "exafmm-browser.patch" : `${name}-browser.patch`);
    const applied = Bun.spawnSync({
      cmd: ["git", "-C", path, "apply", "--reverse", "--check", patch],
      stdout: "ignore",
      stderr: "ignore",
    });
    if (applied.exitCode !== 0) {
      git(["-C", path, "apply", "--check", patch]);
      git(["-C", path, "apply", patch]);
    }
  }
  console.log(`${name}: ${source.commit}`);
}
