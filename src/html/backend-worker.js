// Dedicated numerical worker. It owns the page's only Rust backend and
// C++/Zig WASM instances; the main thread only sends immutable request payloads
// and receives copied result buffers. Request kinds mirror the frontend's
// `cpp_backend::request_*` functions one-to-one; pacing and stale-result
// rejection live in the Bevy channels, never here. The few kinds that loop
// over backend calls (`surface_field`, `reference_sources`, `source_sets`) are
// pure batching glue: each backend call is exactly what one synchronous call
// used to be, and the answers are concatenated in call order.
let backendReady;
const LIVE_KINDS = new Set(['advance', 'evaluate', 'frequency_domain_modes', 'configure']);
const liveQueue = [];
const batchQueue = [];
let pumping = false;
let batchStreak = 0;
const BATCH_WEIGHT = 4;
const TARGET_SLICE = 512;
const CONTINUE = { continue: true };

function versionedUrl(path, cacheSuffix) {
    const url = new URL(path, import.meta.url);
    if (cacheSuffix) url.search = cacheSuffix.startsWith('?') ? cacheSuffix : `?${cacheSuffix}`;
    return url.href;
}

async function initialize(cacheSuffix) {
    const hostUrl = versionedUrl('../../pkg/backend.mjs', cacheSuffix);
    const cppUrl = versionedUrl('../../pkg/ryugu_backend.wasm', cacheSuffix);
    const rustUrl = versionedUrl('../../pkg/backend/ryugu_backend.js', cacheSuffix);
    const rustWasmUrl = versionedUrl('../../pkg/backend/ryugu_backend_bg.wasm', cacheSuffix);

    const [{ instantiateBackend, createFieldBackend }, cppResponse] = await Promise.all([
        import(hostUrl),
        fetch(cppUrl),
    ]);
    if (!cppResponse.ok) {
        throw new Error(`C++ backend download failed: ${cppResponse.status}`);
    }
    const numericalModule = await instantiateBackend(await cppResponse.arrayBuffer());
    globalThis.ryuguCpp = createFieldBackend(numericalModule);
    globalThis.ryuguScheduler = numericalModule.api;

    const rustBackend = await import(rustUrl);
    await rustBackend.default({ module_or_path: rustWasmUrl });
    if (rustBackend.protocol_version() !== 1) {
        throw new Error('Unsupported backend protocol');
    }
    globalThis.ryuguRustBackend = rustBackend;
}

function detachedResult(value) {
    if (!ArrayBuffer.isView(value)) return value;
    return new value.constructor(value);
}

function isInvertKind(kind) {
    return kind === 'source_sets' || kind === 'solve_density';
}

function hasInvertWork() {
    return batchQueue.some((item) => isInvertKind(item.kind));
}

// Drop visual-only field batches so Invert owns the Worker immediately.
function dropVisualFieldWork() {
    for (let i = batchQueue.length - 1; i >= 0; i -= 1) {
        const kind = batchQueue[i].kind;
        if (kind === 'gravity_field' || kind === 'surface_field') {
            batchQueue.splice(i, 1);
        }
    }
}

async function yieldForLiveOrbit() {
    await new Promise((resolve) => setTimeout(resolve, 0));
    // Abort mid-tile Section glyphs when Invert arrives; otherwise a long FMM
    // gravity_field holds `pumping` and solve_density never starts (UI stuck on
    // "Convex density inversion running…" with FPS still high).
    if (hasInvertWork()) {
        throw new Error('Field evaluation cancelled for density inversion');
    }
    // Pointwise `surface_field` / `gravity_field` tiles call this. Prefer live
    // `advance` so FMM glyph batches do not hitch the orbit. Source-chunk work
    // uses CONTINUE + the weighted pump instead, so a 64-step advance does not
    // nest inside an `evaluate_sources` tile.
    const index = liveQueue.findIndex((item) => (
        item.kind === 'advance'
        || item.kind === 'evaluate'
        || item.kind === 'frequency_domain_modes'
    ));
    if (index < 0) return;
    const item = liveQueue.splice(index, 1)[0];
    await handleRequest(item);
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (hasInvertWork()) {
        throw new Error('Field evaluation cancelled for density inversion');
    }
}

function evaluateSourcesFull(backend, method, xyz, masses, targets) {
    return backend.evaluate_sources(method, xyz, masses, targets);
}

function continueEvaluateSources(backend, message) {
    const payload = message.payload;
    const targetCount = (payload.targets.length / 3) | 0;
    if (targetCount === 0) return new Float64Array(0);
    const start = message._targetStart || 0;
    if (start === 0 && targetCount <= TARGET_SLICE) {
        return detachedResult(evaluateSourcesFull(
            backend,
            payload.method,
            payload.xyz,
            payload.masses,
            payload.targets,
        ));
    }
    const end = Math.min(targetCount, start + TARGET_SLICE);
    const piece = evaluateSourcesFull(
        backend,
        payload.method,
        payload.xyz,
        payload.masses,
        payload.targets.subarray(3 * start, 3 * end),
    );
    if (!message._out) message._out = new Float64Array(targetCount * 4);
    message._out.set(piece, start * 4);
    if (end < targetCount) {
        message._targetStart = end;
        return CONTINUE;
    }
    const out = message._out;
    message._out = null;
    message._targetStart = 0;
    return out;
}

function continueReferenceSources(backend, message) {
    const payload = message.payload;
    const chunk = Math.max(1, payload.chunk | 0);
    const sourceCount = payload.masses.length;
    const chunkCount = Math.ceil(sourceCount / chunk) || 0;
    const targetValues = (payload.targets.length / 3) * 4;
    if (chunkCount === 0) return new Float64Array(0);
    const index = message._chunkIndex || 0;
    if (!message._out) message._out = new Float64Array(chunkCount * targetValues);
    const start = index * chunk;
    const end = Math.min(sourceCount, start + chunk);
    // Full source chunk, one tree/grid. Do not 512-slice sources.
    const block = evaluateSourcesFull(
        backend,
        payload.method || 'direct',
        payload.xyz.subarray(3 * start, 3 * end),
        payload.masses.subarray(start, end),
        payload.targets,
    );
    message._out.set(block, index * targetValues);
    if (index + 1 < chunkCount) {
        message._chunkIndex = index + 1;
        return CONTINUE;
    }
    const out = message._out;
    message._out = null;
    message._chunkIndex = 0;
    return out;
}

async function runRequest(message) {
    const backend = globalThis.ryuguRustBackend;
    const { kind, payload } = message;
    switch (kind) {
        case 'configure':
            backend.configure(payload.cells, payload.vertices, payload.facets, payload.mass);
            return null;
        case 'evaluate':
            return detachedResult(backend.evaluate(payload.method, payload.x, payload.y, payload.z));
        case 'surface_field':
        case 'gravity_field': {
            // Pointwise field of the configured geometry at every target.
            // Gravity glyphs under FMM/FFT are especially costly — yield often
            // and drain live advance so orbit stays smooth while Section is on.
            const targets = payload.targets;
            const count = targets.length / 3;
            const out = new Float64Array(count * 4);
            const yieldEvery = kind === 'gravity_field' ? 8 : 16;
            for (let i = 0; i < count; i += 1) {
                if (i % yieldEvery === 0) await yieldForLiveOrbit();
                out.set(
                    backend.evaluate(payload.method, targets[3 * i], targets[3 * i + 1], targets[3 * i + 2]),
                    4 * i,
                );
            }
            return out;
        }
        case 'evaluate_sources':
        case 'comparison_sources':
            return continueEvaluateSources(backend, message);
        case 'reference_sources':
            return continueReferenceSources(backend, message);
        case 'source_sets': {
            // Invert reference + voxel columns. Chunk with CONTINUE so the Worker
            // event loop can breathe (UI / cancel / solve_density). Do NOT drain
            // live advance here: under 64× that starved Preparing forever.
            // Frequency-domain invert skips this Worker path (CPU Eq.184).
            const offsets = payload.setOffsets;
            const setCount = offsets.length - 1;
            const targetValues = (payload.targets.length / 3) * 4;
            if (setCount <= 0) return new Float64Array(0);
            const index = message._setIndex || 0;
            if (!message._out) message._out = new Float64Array(setCount * targetValues);
            const start = offsets[index];
            const end = offsets[index + 1];
            const block = backend.evaluate_sources(
                payload.method,
                payload.xyz.subarray(3 * start, 3 * end),
                payload.masses.subarray(start, end),
                payload.targets,
            );
            message._out.set(block, index * targetValues);
            if (index + 1 < setCount) {
                message._setIndex = index + 1;
                return CONTINUE;
            }
            const out = message._out;
            message._out = null;
            message._setIndex = 0;
            return out;
        }
        case 'prepare_candidate_sources':
            backend.prepare_candidate_sources(payload.xyz, payload.masses);
            return null;
        case 'clear_candidate_sources':
            backend.clear_candidate_sources();
            return null;
        case 'frequency_domain_modes':
            backend.set_frequency_domain_modes(payload.modes);
            return null;
        case 'advance':
            return detachedResult(backend.advance_frame(
                payload.epoch,
                payload.method,
                payload.initial,
                payload.step,
                payload.steps,
                payload.history,
            ));
        case 'solve_density':
            return detachedResult(backend.solve_density(payload.data));
        case 'propagate_candidates':
            return detachedResult(backend.propagate_candidates(payload.data));
        default:
            throw new Error(`Unknown numerical request: ${kind}`);
    }
}

function asU64(value) {
    if (typeof value === 'bigint') return value;
    if (typeof value === 'number' && Number.isFinite(value)) return BigInt(Math.trunc(value));
    if (typeof value === 'string' && value !== '') return BigInt(value);
    return 0n;
}

async function handleRequest(message) {
    const requestId = asU64(message.requestId);
    const epoch = asU64(message.epoch);
    try {
        await backendReady;
        const value = await runRequest(message);
        if (value === CONTINUE) {
            batchQueue.unshift(message);
            return;
        }
        const transfer = ArrayBuffer.isView(value) ? [value.buffer] : [];
        postMessage({
            type: 'result',
            kind: message.kind,
            requestId,
            epoch,
            ok: true,
            value,
        }, transfer);
    } catch (error) {
        postMessage({
            type: 'result',
            kind: message.kind,
            requestId,
            epoch,
            ok: false,
            error: error instanceof Error ? error.message : String(error),
        });
    }
}

self.onmessage = ({ data }) => {
    if (data?.type === 'initialize') {
        if (backendReady) return;
        backendReady = initialize(data.cacheSuffix || '')
            .then(() => postMessage({ type: 'ready' }))
            .catch((error) => {
                postMessage({
                    type: 'failed',
                    error: error instanceof Error ? error.message : String(error),
                });
                throw error;
            });
        return;
    }
    if (data?.type !== 'request') return;
    // Configure stays first. Weighted pump: K batch : 1 live while batch
    // work is queued so First/Stress/Invert make progress without freezing orbit.
    if (isInvertKind(data.kind)) {
        dropVisualFieldWork();
    }
    if (LIVE_KINDS.has(data.kind)) liveQueue.push(data);
    else batchQueue.push(data);
    pump();
};

function takeQueuedRequest() {
    const configure = liveQueue.findIndex((item) => item.kind === 'configure');
    if (configure >= 0) {
        batchStreak = 0;
        return liveQueue.splice(configure, 1)[0];
    }
    // Prefer live advance over visual-only gravity glyphs.
    const advance = liveQueue.findIndex((item) => item.kind === 'advance');
    if (advance >= 0 && batchQueue.some((item) => item.kind === 'gravity_field')) {
        batchStreak = 0;
        return liveQueue.splice(advance, 1)[0];
    }
    // Prefer density QP / invert source_sets over any leftover visual batch.
    const invertWork = batchQueue.findIndex((item) => isInvertKind(item.kind));
    if (invertWork > 0) {
        const [item] = batchQueue.splice(invertWork, 1);
        batchQueue.unshift(item);
    }
    if (liveQueue.length && batchQueue.length) {
        if (batchStreak < BATCH_WEIGHT) {
            batchStreak += 1;
            return batchQueue.shift();
        }
        batchStreak = 0;
        return liveQueue.shift();
    }
    batchStreak = 0;
    return liveQueue.shift() || batchQueue.shift();
}

async function pump() {
    if (pumping) return;
    pumping = true;
    try {
        while (liveQueue.length || batchQueue.length) {
            const data = takeQueuedRequest();
            if (!data) break;
            await handleRequest(data);
            // Yield after every request (including CONTINUE chunks) so the page
            // stays responsive during long invert source_sets / solve_density.
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
    } finally {
        pumping = false;
    }
}
