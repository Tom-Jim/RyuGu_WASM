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
// First/Stress `evaluate_sources` used to run as one 65k-source call that
// occupied the only Worker until it finished, freezing live orbit + Jacobi.
// Slice and yield so `advance` can run between chunks.
const LIVE_YIELD_SOURCES = 512;

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
    while (liveQueue.length) {
        const data = liveQueue.shift();
        await handleRequest(data);
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
}

function addInto(accumulator, piece) {
    for (let index = 0; index < accumulator.length; index += 1) {
        accumulator[index] += piece[index];
    }
}

async function evaluateSourcesYielding(backend, method, xyz, masses, targets) {
    const sourceCount = masses.length;
    const accumulator = new Float64Array((targets.length / 3) * 4);
    if (sourceCount === 0) return accumulator;
    for (let start = 0; start < sourceCount; start += LIVE_YIELD_SOURCES) {
        await yieldForLiveOrbit();
        const end = Math.min(sourceCount, start + LIVE_YIELD_SOURCES);
        const piece = backend.evaluate_sources(
            method,
            xyz.subarray(3 * start, 3 * end),
            masses.subarray(start, end),
            targets,
        );
        addInto(accumulator, piece);
    }
    return accumulator;
}

async function runRequest(kind, payload) {
    const backend = globalThis.ryuguRustBackend;
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
            return await evaluateSourcesYielding(
                backend,
                payload.method,
                payload.xyz,
                payload.masses,
                payload.targets,
            );
        case 'reference_sources': {
            // One field block per consecutive source chunk, in chunk order, so
            // the frontend can accumulate them exactly like its former loop.
            // Internally the chunk is sliced so live `advance` is not blocked.
            const chunk = Math.max(1, payload.chunk | 0);
            const sourceCount = payload.masses.length;
            const chunkCount = Math.ceil(sourceCount / chunk);
            const targetValues = (payload.targets.length / 3) * 4;
            const out = new Float64Array(chunkCount * targetValues);
            for (let index = 0; index < chunkCount; index += 1) {
                const start = index * chunk;
                const end = Math.min(sourceCount, start + chunk);
                const block = await evaluateSourcesYielding(
                    backend,
                    payload.method || 'direct',
                    payload.xyz.subarray(3 * start, 3 * end),
                    payload.masses.subarray(start, end),
                    payload.targets,
                );
                out.set(block, index * targetValues);
            }
            return out;
        }
        case 'source_sets': {
            // Independent source sets against one common target list; the
            // answer is laid out [set][target][4].
            const offsets = payload.setOffsets;
            const setCount = offsets.length - 1;
            const targetValues = (payload.targets.length / 3) * 4;
            const out = new Float64Array(setCount * targetValues);
            for (let index = 0; index < setCount; index += 1) {
                const start = offsets[index];
                const end = offsets[index + 1];
                const block = await evaluateSourcesYielding(
                    backend,
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
            // Clarabel can monopolize the Worker for seconds; drain live orbit first.
            await yieldForLiveOrbit();
            return detachedResult(backend.solve_density(payload.data));
        case 'propagate_candidates':
            // First/Stress slices are still synchronous WASM, but draining the
            // live queue between slices keeps advance from starving entirely.
            await yieldForLiveOrbit();
            return detachedResult(backend.propagate_candidates(payload.data));
        default:
            throw new Error(`Unknown numerical request: ${kind}`);
    }
}

async function handleRequest(message) {
    try {
        await backendReady;
        const value = await runRequest(message.kind, message.payload);
        const transfer = ArrayBuffer.isView(value) ? [value.buffer] : [];
        postMessage({
            type: 'result',
            kind: message.kind,
            requestId: message.requestId,
            epoch: message.epoch,
            ok: true,
            value,
        }, transfer);
    } catch (error) {
        postMessage({
            type: 'result',
            kind: message.kind,
            requestId: message.requestId,
            epoch: message.epoch,
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
    // Live orbit/Jacobi/configure stay ahead of First/Stress batches so a
    // planning job cannot freeze the probe, trail, or bottom chart.
    if (LIVE_KINDS.has(data.kind)) liveQueue.push(data);
    else batchQueue.push(data);
    pump();
};

async function pump() {
    if (pumping) return;
    pumping = true;
    try {
        while (liveQueue.length || batchQueue.length) {
            const data = liveQueue.shift() || batchQueue.shift();
            await handleRequest(data);
        }
    } finally {
        pumping = false;
    }
}
