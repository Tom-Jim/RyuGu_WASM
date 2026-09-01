import { join } from "path";
import { mkdir, rename, rm } from "node:fs/promises";

const PORT = Number.parseInt(Bun.env.PORT ?? "3000", 10);
const ROOT = import.meta.dir;

// A real page connection owns idle-sleep prevention on this local Mac. Screen
// sleep is deliberately allowed (-i/-s, not -d; -s applies on AC power). No browser security flags or
// permanent OS settings are changed, and this does not override forced sleep.
type PowerData = { kind: "power"; tick?: ReturnType<typeof setTimeout> };
type PowerSocket = Bun.ServerWebSocket<PowerData>;
const powerClients = new Set<PowerSocket>();
function startCaffeinate() {
  return Bun.spawn(["/usr/bin/caffeinate", "-i", "-s", "-w", String(process.pid)], {
    stdin: "ignore", stdout: "ignore", stderr: "ignore",
  });
}
let powerProcess: ReturnType<typeof startCaffeinate> | null = null;
let powerDetail = "Local idle-sleep prevention is inactive.";
let powerRetry: ReturnType<typeof setTimeout> | null = null;

function reportPower() {
  const message = JSON.stringify({
    type: "power-status", active: powerProcess !== null, detail: powerDetail,
  });
  for (const client of powerClients) client.send(message);
}
function releasePower() {
  if (powerRetry !== null) clearTimeout(powerRetry);
  powerRetry = null;
  const previous = powerProcess;
  powerProcess = null;
  previous?.kill();
  powerDetail = "Local idle-sleep prevention released.";
}
function acquirePower() {
  if (powerProcess) return;
  if (process.platform !== "darwin" || Bun.env.RYUGU_KEEP_AWAKE === "0") {
    powerDetail = "Local system sleep prevention is unavailable/disabled; browser scheduling is best effort.";
    return;
  }
  try {
    const child = startCaffeinate();
    powerProcess = child;
    powerDetail = "Server Mac idle sleep inhibited; display may turn off. Browser freeze and forced sleep are not prevented.";
    void child.exited.then((code) => {
      if (powerProcess !== child) return;
      powerProcess = null;
      powerDetail = `Local sleep prevention exited (${code}); system sleep is no longer inhibited.`;
      reportPower();
      retryPower();
    });
  } catch (error) {
    powerDetail = `Local sleep prevention could not start: ${String(error)}`;
    retryPower();
  }
}

function retryPower() {
  if (powerRetry !== null || powerClients.size === 0) return;
  powerRetry = setTimeout(() => {
    powerRetry = null;
    if (powerClients.size) { acquirePower(); reportPower(); }
  }, 5000);
}

// Same-origin loopback writes only. No arbitrary path, shell command, or
// browser download permission is accepted by the unattended export service.
const EXPORT_ROOT = join(ROOT, "benchmark-captures");
const MAX_EXPORT_BYTES = 16 * 1024 * 1024;
async function saveBenchmarkExport(req: Request, url: URL) {
  if (req.method !== "PUT" || req.headers.get("Origin") !== url.origin
    || !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) {
    return new Response("Forbidden", { status: 403 });
  }
  const match = /^\/__ryugu\/exports\/([a-zA-Z0-9_-]{1,96})\/(results\.json|0[1-9]-(?:32|64|128|256|512|1024|2048|4096|8192)K\.png)$/.exec(url.pathname);
  if (!match) return new Response("Invalid export name", { status: 400 });
  // Read with a hard bound even when a client omits Content-Length.
  const reader = req.body?.getReader();
  if (!reader) return new Response("Empty export", { status: 400 });
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_EXPORT_BYTES) {
        await reader.cancel();
        return new Response("Export too large", { status: 413 });
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const body = Buffer.concat(chunks);
  const [, run, name] = match;
  if (name.endsWith(".png")) {
    if (!body.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
      return new Response("Invalid PNG", { status: 400 });
    }
  } else {
    try { JSON.parse(body.toString("utf8")); }
    catch { return new Response("Invalid JSON", { status: 400 }); }
  }
  const directory = join(EXPORT_ROOT, run);
  await mkdir(directory, { recursive: true });
  const temporary = join(directory, `.${name}-${crypto.randomUUID()}.tmp`);
  try {
    await Bun.write(temporary, body);
    await rename(temporary, join(directory, name));
  } finally { await rm(temporary, { force: true }); }
  return Response.json({ saved: `benchmark-captures/${run}/${name}` });
}

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

const server = Bun.serve<PowerData>({
  port: PORT,
  // USB `adb reverse` forwards the phone's localhost to this loopback socket.
  // Keeping it loopback-only avoids exposing the development tree on Wi-Fi.
  hostname: "127.0.0.1",
  maxRequestBodySize: MAX_EXPORT_BYTES,
  async fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname.startsWith("/__ryugu/exports/")) {
      try { return await saveBenchmarkExport(req, url); }
      catch { return new Response("Export could not be saved; retry later", { status: 503 }); }
    }
    if (url.pathname === "/__ryugu/power") {
      // Reject cross-origin scripts/DNS rebinding; the power endpoint is not
      // available to arbitrary sites or remote clients through this server.
      if (!["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)
        || req.headers.get("Origin") !== url.origin || req.method !== "GET") {
        return new Response("Forbidden", { status: 403 });
      }
      if (powerClients.size >= 32) return new Response("Too many sessions", { status: 429 });
      if (server.upgrade(req, { data: { kind: "power" } })) return;
      return new Response("WebSocket upgrade required", { status: 400 });
    }
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    const relativePath = pathname === "/index.html" ? "src/html/index.html" : pathname.slice(1);
    const filePath = join(ROOT, relativePath);

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
  websocket: {
    // Protocol ping/pong detects broken connections without a page timer.
    // Closing the final tab (or disabling its toggle) releases the assertion.
    idleTimeout: 120,
    sendPings: true,
    maxPayloadLength: 1024,
    open(socket) {
      powerClients.add(socket);
      acquirePower();
      reportPower();
    },
    message(socket, message) {
      if (message === "status") reportPower();
      else {
        let packet: { type?: string; id?: number };
        try {
          packet = JSON.parse(String(message));
          if (!packet || typeof packet !== "object") throw new Error("Invalid packet");
        }
        catch { socket.close(1008, "Invalid scheduler message"); return; }
        if (packet.type === "cancel-tick") {
          clearTimeout(socket.data.tick);
          socket.data.tick = undefined;
        } else if (packet.type === "tick" && Number.isSafeInteger(packet.id)) {
          // One demand-driven tick, not a push loop: a slow/frozen page cannot
          // accumulate frames. Server timers are independent of tab throttling.
          clearTimeout(socket.data.tick);
          const id = packet.id;
          socket.data.tick = setTimeout(() => {
            socket.data.tick = undefined;
            if (powerClients.has(socket)) socket.send(JSON.stringify({ type: "tick", id }));
          }, 4);
        } else socket.close(1008, "Unexpected power-control message");
      }
    },
    close(socket) {
      clearTimeout(socket.data.tick);
      powerClients.delete(socket);
      if (powerClients.size === 0) releasePower();
    },
  },
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => { releasePower(); server.stop(true); process.exit(0); });
}
process.once("exit", releasePower);

console.log(`Dev server running at http://localhost:${PORT}`);
