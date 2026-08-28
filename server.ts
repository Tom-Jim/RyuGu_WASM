import { join } from "path";

const PORT = Number.parseInt(Bun.env.PORT ?? "3000", 10);
const ROOT = import.meta.dir;

const MIME: Record<string, string> = {
  ".html": "text/html",
  ".js":   "application/javascript",
  ".wasm": "application/wasm",
  ".obj":  "text/plain",
  ".mtl":  "text/plain",
  ".png":  "image/png",
  ".jpg":  "image/jpeg",
  ".css":  "text/css",
  ".wgsl": "text/plain; charset=utf-8",
};

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    let pathname = url.pathname === "/" ? "/src/html/index.html" : url.pathname;
    const filePath = join(ROOT, pathname);

    const file = Bun.file(filePath);
    if (!(await file.exists())) {
      return new Response("Not Found", { status: 404 });
    }

    const ext = pathname.slice(pathname.lastIndexOf("."));
    const contentType = MIME[ext] ?? "application/octet-stream";

    return new Response(file, {
      headers: {
        "Content-Type": contentType,
        // Required for SharedArrayBuffer / WASM threads (if needed)
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cache-Control": "no-store",
      },
    });
  },
});

console.log(`Dev server running at http://localhost:${PORT}`);
