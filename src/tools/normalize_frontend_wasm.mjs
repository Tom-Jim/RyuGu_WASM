import { existsSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const root = fileURLToPath(new URL("../../", import.meta.url));
const packageDirectory = join(root, "pkg");

function renameCaseInsensitive(actual, expected) {
  const source = join(packageDirectory, actual);
  const destination = join(packageDirectory, expected);
  if (actual === expected) return;
  const temporary = `${destination}.normalizing`;
  renameSync(source, temporary);
  renameSync(temporary, destination);
}

for (const [actual, expected] of [
  ["Ryugu_wasm.js", "ryugu_wasm.js"],
  ["Ryugu_wasm.d.ts", "ryugu_wasm.d.ts"],
  ["Ryugu_wasm_bg.wasm", "ryugu_wasm_bg.wasm"],
  ["Ryugu_wasm_bg.wasm.d.ts", "ryugu_wasm_bg.wasm.d.ts"],
]) {
  if (existsSync(join(packageDirectory, actual))) renameCaseInsensitive(actual, expected);
}

const frontend = join(packageDirectory, "ryugu_wasm.js");
if (!existsSync(frontend)) throw new Error("wasm-pack did not produce pkg/ryugu_wasm.js");
const source = readFileSync(frontend, "utf8");
const normalized = source
  .replaceAll("Ryugu_wasm_bg.wasm", "ryugu_wasm_bg.wasm")
  .replaceAll("Ryugu_wasm.d.ts", "ryugu_wasm.d.ts");
if (normalized !== source) writeFileSync(frontend, normalized);

// wasm-pack 0.13 re-parses an existing output package manifest on the next
// invocation and rejects its own object-form repository field. The browser
// only consumes the JS, WASM, declarations, and snippets, so omit this
// generated publish manifest from the reusable local output directory.
const packageManifest = join(packageDirectory, "package.json");
if (existsSync(packageManifest)) unlinkSync(packageManifest);
