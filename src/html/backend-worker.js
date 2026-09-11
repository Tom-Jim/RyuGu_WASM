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

async function yieldForLiveOrbit() {
    await new Promise((resolve) => setTimeout(resolve, 0));
    // Nested yields must not run a 64-step advance inside a timed tile.
    // Jacobi `evaluate` and Eq.121 modes may still drain; live `advance`
    // returns to the weighted pump instead.
    const index = liveQueue.findIndex((item) => (
        item.kind === 'evaluate' || item.kind === 'frequency_domain_modes'
    ));
    if (index < 0) return;
    const item = liveQueue.splice(index, 1)[0];
    await handleRequest(item);
    await new Promise((resolve) => setTimeout(resolve, 0));
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
        case 'surface_field': {
            // Pointwise field of the configured geometry at every target.
            const targets = payload.targets;
            const count = targets.length / 3;
            const out = new Float64Array(count * 4);
            for (let i = 0; i < count; i += 1) {
                if (i % 16 === 0) await yieldForLiveOrbit();
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
            // Invert reference + voxel columns. Do not yield to live advance
            // between sets: that drain never finished under 64×, so FMM/FFT
            // invert sat on "Preparing inversion observations…" forever.
            // Frequency-domain invert skips this Worker path entirely.
            const offsets = payload.setOffsets;
            const setCount = offsets.length - 1;
            const targetValues = (payload.targets.length / 3) * 4;
            const out = new Float64Array(setCount * targetValues);
            for (let index = 0; index < setCount; index += 1) {
                const start = offsets[index];
                const end = offsets[index + 1];
                const block = backend.evaluate_sources(
                    payload.method,
                    payload.xyz.subarray(3 * start, 3 * end),
                    payload.masses.subarray(start, end),
                    payload.targets,
                );
                out.set(block, index * targetValues);
            }
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
        }
    } finally {
        pumping = false;
    }
}
