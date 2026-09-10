// Dedicated numerical worker. It owns the independent Rust backend and the
// C++/Zig WASM instance; the main thread only sends immutable request payloads
// and receives copied result buffers.
let backendReady;
let requestQueue = Promise.resolve();

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

function runRequest(kind, payload) {
    const backend = globalThis.ryuguRustBackend;
    switch (kind) {
        case 'configure':
            backend.configure(payload.cells, payload.vertices, payload.facets, payload.mass);
            return null;
        case 'evaluate':
            return detachedResult(backend.evaluate(payload.method, payload.x, payload.y, payload.z));
        case 'evaluate_sources':
            return detachedResult(backend.evaluate_sources(
                payload.method,
                payload.xyz,
                payload.masses,
                payload.targets,
            ));
        case 'prepare_candidate_sources':
            backend.prepare_candidate_sources(payload.xyz, payload.masses);
            return null;
        case 'clear_candidate_sources':
            backend.clear_candidate_sources();
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

async function handleRequest(message) {
    try {
        await backendReady;
        const value = runRequest(message.kind, message.payload);
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
    // Preserve configure/evaluate ordering and the stateful integrator contract.
    requestQueue = requestQueue.then(
        () => handleRequest(data),
        () => handleRequest(data),
    );
};
