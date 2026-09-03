/*
 * Page-open telemetry is deliberately inert outside the canonical production
 * URL. GitHub Pages is static, so /api/visit must be supplied by a separately
 * deployed privacy-reviewed endpoint before events can be persisted.
 */
(() => {
  const canonicalUrl = 'https://tom-jim.github.io/RyuGu_WASM/';
  const endpoint = '/RyuGu_WASM/api/visit';
  if (location.href !== canonicalUrl || !navigator.sendBeacon) return;

  const sessionKey = 'ryugu-page-open-recorded';
  try {
    if (sessionStorage.getItem(sessionKey) === '1') return;
    sessionStorage.setItem(sessionKey, '1');

    let visitorId = localStorage.getItem('ryugu-anonymous-visitor-id');
    if (!visitorId) {
      visitorId = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      localStorage.setItem('ryugu-anonymous-visitor-id', visitorId);
    }

    const payload = JSON.stringify({
      visitorId,
      openedAt: new Date().toISOString(),
      path: location.pathname,
    });
    navigator.sendBeacon(endpoint, new Blob([payload], { type: 'application/json' }));
  } catch {
    // Storage restrictions and an unavailable endpoint must never affect startup.
  }
})();
