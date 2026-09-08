import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

const root = fileURLToPath(new URL("../../C++/", import.meta.url));
const sources = JSON.parse(readFileSync(join(root, "sources.lock.json"), "utf8"));
function git(args) {
  const result = spawnSync("git", args, { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || "git failed");
  return result.stdout.trim();
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
    const applied = spawnSync("git", ["-C", path, "apply", "--reverse", "--check", patch]);
    if (applied.status !== 0) {
      git(["-C", path, "apply", "--check", patch]);
      git(["-C", path, "apply", patch]);
    }
  }
  console.log(`${name}: ${source.commit}`);
}
