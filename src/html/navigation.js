// Gesture ownership is explicit. Camera mode lets Bevy/winit receive native
// pointer AND touch events; workbench mode captures them before either engine.
(() => {
  const frame = document.getElementById('viewport-frame');
  const canvas = document.getElementById('bevy');
  const toggle = document.getElementById('gesture-mode');
  const reset = document.getElementById('display-reset');
  if (!frame || !canvas || !toggle) return;
  let wholeUi = false;
  let pendingMode = null;
  let zoom = 1;
  let offset = { x: 0, y: 0 };
  const pointers = new Map();
  let pinch = null;
  let mousePan = null;
  const stop = (event) => { if (event.cancelable) event.preventDefault(); event.stopPropagation(); };
  const center = () => ({ x: document.documentElement.clientWidth / 2, y: innerHeight / 2 });
  const apply = () => {
    frame.style.setProperty('--user-zoom', String(zoom));
    frame.style.setProperty('--view-offset-x', `${offset.x}px`);
    frame.style.setProperty('--view-offset-y', `${offset.y}px`);
  };
  const setMode = (next) => {
    wholeUi = next;
    frame.dataset.gestureMode = next ? 'workbench' : 'camera';
    toggle.setAttribute('aria-pressed', String(next));
    toggle.textContent = next ? 'Gestures: whole UI' : 'Gestures: Bevy camera';
    toggle.title = next ? 'Pinch or wheel over the scene to scale the entire workbench. Drag to pan.'
      : 'Pinch or wheel to zoom the Bevy camera. Drag to orbit; two fingers pan.';
    pinch = null;
    mousePan = null;
  };
  const zoomAt = (oldAnchor, anchor, requestedZoom) => {
    const nextZoom = Math.max(0.5, Math.min(3, requestedZoom));
    const ratio = nextZoom / zoom;
    const origin = center();
    offset = { x: anchor.x - origin.x - ratio * (oldAnchor.x - origin.x - offset.x),
      y: anchor.y - origin.y - ratio * (oldAnchor.y - origin.y - offset.y) };
    zoom = nextZoom;
    apply();
  };
  const pinchState = () => {
    const [a, b] = [...pointers.values()];
    return { anchor: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)) };
  };
  document.addEventListener('pointerdown', (event) => {
    if (event.target !== canvas) return;
    if (event.pointerType === 'touch') {
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (wholeUi) {
        canvas.setPointerCapture?.(event.pointerId);
        if (pointers.size === 2) pinch = pinchState();
        stop(event);
      }
    } else if (wholeUi && (event.button === 0 || event.button === 2)) {
      mousePan = { id: event.pointerId, x: event.clientX, y: event.clientY };
      canvas.setPointerCapture?.(event.pointerId);
      stop(event);
    }
  }, { capture: true, passive: false });
  document.addEventListener('pointermove', (event) => {
    if (!wholeUi) return;
    if (mousePan?.id === event.pointerId) {
      offset.x += event.clientX - mousePan.x;
      offset.y += event.clientY - mousePan.y;
      mousePan = { id: event.pointerId, x: event.clientX, y: event.clientY };
      apply(); stop(event); return;
    }
    const previous = pointers.get(event.pointerId);
    if (!previous) return;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2) {
      const next = pinchState();
      if (pinch) zoomAt(pinch.anchor, next.anchor, zoom * next.distance / pinch.distance);
      pinch = next;
    } else if (pointers.size === 1) {
      offset.x += event.clientX - previous.x;
      offset.y += event.clientY - previous.y;
      apply();
    }
    stop(event);
  }, { capture: true, passive: false });
  const end = (event) => {
    const owned = pointers.delete(event.pointerId) || mousePan?.id === event.pointerId;
    if (mousePan?.id === event.pointerId) mousePan = null;
    if (pointers.size < 2) pinch = null;
    if (owned && wholeUi) stop(event);
    if (!pointers.size && !mousePan && pendingMode !== null) {
      const next = pendingMode;
      pendingMode = null;
      // Let winit see the matching native touchend before changing ownership.
      setTimeout(() => setMode(next), 0);
    }
  };
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    document.addEventListener(type, end, { capture: true, passive: false });
  }
  // winit also listens to TouchEvent on some mobile backends. Blocking only
  // PointerEvent made one pinch move BOTH the camera and the HTML viewport.
  for (const type of ['touchstart', 'touchmove', 'touchend', 'touchcancel']) {
    document.addEventListener(type, (event) => {
      if (wholeUi && event.target === canvas) stop(event);
    }, { capture: true, passive: false });
  }
  document.addEventListener('wheel', (event) => {
    if (!wholeUi || event.target !== canvas) return;
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
    const anchor = { x: event.clientX, y: event.clientY };
    zoomAt(anchor, anchor, zoom * Math.exp(-delta * .0015));
    stop(event);
  }, { capture: true, passive: false });
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  toggle.addEventListener('click', () => {
    const next = !(pendingMode ?? wholeUi);
    if (pointers.size || mousePan) pendingMode = next;
    else setMode(next);
  });
  reset?.addEventListener('click', () => {
    zoom = 1; offset = { x: 0, y: 0 }; apply();
    document.querySelectorAll('[data-float-panel]').forEach((panel) => {
      panel.style.transform = ''; panel.style.width = ''; panel.style.height = '';
      panel._ryuguOffset = { x: 0, y: 0 };
    });
  });
  window.addEventListener('blur', () => {
    pointers.clear(); pinch = null; mousePan = null;
    if (pendingMode !== null) { setMode(pendingMode); pendingMode = null; }
  });
  // Drag offsets live in layout pixels; invert display rotation and zoom.
  document.addEventListener('pointerdown', (event) => {
    const handle = event.target.closest?.('.drag-handle');
    const panel = handle?.closest?.('[data-float-panel]');
    if (!panel || event.button !== 0) return;
    const origin = panel._ryuguOffset ?? { x: 0, y: 0 };
    const start = { x: event.clientX, y: event.clientY };
    const angle = -Number(frame.dataset.quarterTurn ?? 0) * Math.PI / 2;
    const scale = zoom * (Number.parseFloat(getComputedStyle(frame).getPropertyValue('--display-scale')) || 1);
    panel.classList.add('is-dragging');
    handle.setPointerCapture?.(event.pointerId);
    const move = (next) => {
      if (next.pointerId !== event.pointerId) return;
      const dx = (next.clientX - start.x) / scale, dy = (next.clientY - start.y) / scale;
      const x = origin.x + dx * Math.cos(angle) - dy * Math.sin(angle);
      const y = origin.y + dx * Math.sin(angle) + dy * Math.cos(angle);
      panel._ryuguOffset = { x, y };
      panel.style.transform = `translate(${x}px, ${y}px)`;
    };
    const finish = (next) => {
      if (next.pointerId !== event.pointerId) return;
      panel.classList.remove('is-dragging');
      handle.removeEventListener('pointermove', move);
      for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) handle.removeEventListener(type, finish);
    };
    handle.addEventListener('pointermove', move);
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) handle.addEventListener(type, finish);
    stop(event);
  });
  setMode(false); apply();
})();
