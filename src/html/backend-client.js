/**
 * Create the main-thread half of the numerical Worker bridge. Delivery is
 * push-based: the Worker posts one completion, then `deliverResult` immediately
 * forwards it into the frontend WASM channel selected by the caller.
 */
export function createNumericalBackendClient({
    cacheSuffix = '',
    deliverResult,
    onReady = () => {},
    onFailed = () => {},
}) {
    if (typeof deliverResult !== 'function') {
        throw new TypeError('A numerical result delivery callback is required');
    }
    const workerUrl = new URL('./backend-worker.js', import.meta.url);
    if (cacheSuffix) {
        workerUrl.search = cacheSuffix.startsWith('?') ? cacheSuffix : `?${cacheSuffix}`;
    }
    const worker = new Worker(workerUrl, {
        type: 'module',
        name: 'ryugu-numerical-backend',
    });
    let ready = false;
    let failed = false;

    worker.onmessage = ({ data }) => {
        if (data?.type === 'ready') {
            ready = true;
            onReady();
            return;
        }
        if (data?.type === 'failed') {
            failed = true;
            onFailed(data.error || 'Numerical Worker initialization failed');
            return;
        }
        if (data?.type === 'result') deliverResult(data);
    };
    worker.onerror = (event) => {
        failed = true;
        onFailed(event.message || 'Numerical Worker failed');
    };
    worker.postMessage({ type: 'initialize', cacheSuffix });

    return Object.freeze({
        isReady: () => ready && !failed,
        request(kind, requestId, epoch, payload) {
            if (!ready || failed) return false;
            worker.postMessage({ type: 'request', kind, requestId, epoch, payload });
            return true;
        },
        terminate() {
            ready = false;
            worker.terminate();
        },
    });
}
