// Scheduling/power-control worker, not a second WASM engine. Never copy the
// simulation or GPU banks into this worker. One requested tick is outstanding.
let timer = null;
let task = null;
let tickId = null;
let socket = null;
let powerUrl = null;
let reconnect = null;
let heartbeat = null;
let stopped = false;

function cancelTick() {
  if (timer !== null) clearTimeout(timer);
  timer = null;
  task?.abort();
  task = null;
  tickId = null;
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'cancel-tick' }));
}
function deliver(id) {
  if (id !== tickId || stopped) return;
  cancelTick();
  postMessage({ type: 'tick', id });
}
function scheduleTick(id) {
  cancelTick();
  tickId = id;
  const connected = socket?.readyState === WebSocket.OPEN;
  if (connected) socket.send(JSON.stringify({ type: 'tick', id }));
  // Loopback server timers escape background tab timer throttling. One
  // worker task races the server as a watchdog, never as a second update loop.
  const delay = connected ? 1000 : 4;
  if (self.scheduler?.postTask) {
    task = new AbortController();
    self.scheduler.postTask(() => deliver(id), { delay, priority: 'user-blocking', signal: task.signal })
      .catch((error) => {
        if (error.name !== 'AbortError' && tickId === id) timer = setTimeout(() => deliver(id), 4);
      });
  } else timer = setTimeout(() => deliver(id), delay);
}
function powerStatus(active, detail) {
  postMessage({ type: 'power-status', active, detail });
}
function connectPower() {
  clearTimeout(reconnect);
  reconnect = null;
  if (!powerUrl || stopped || socket) return;
  let current;
  try { current = new WebSocket(powerUrl); }
  catch {
    powerStatus(false, 'Local power service unavailable; retrying.');
    reconnect = setTimeout(connectPower, 5000);
    return;
  }
  socket = current;
  current.onopen = () => {
    if (socket !== current) return;
    if (tickId !== null) scheduleTick(tickId);
    // Keep the lease alive even if protocol pongs do not reset idleTimeout.
    // Neither frame scheduling nor power prevention depends on this timer.
    heartbeat = setInterval(() => {
      if (current.readyState === WebSocket.OPEN) current.send('status');
    }, 20_000);
  };
  current.onmessage = ({ data }) => {
    if (socket !== current) return;
    try {
      const status = JSON.parse(data);
      if (status.type === 'power-status') {
        powerStatus(status.active === true, `${status.detail} Local server frame scheduling connected.`);
        postMessage({ type: 'maintenance' });
      } else if (status.type === 'tick') deliver(status.id);
    } catch { /* Only the local power/scheduler protocol is consumed. */ }
  };
  current.onerror = () => powerStatus(false, 'Local power service unavailable; system sleep is not prevented.');
  current.onclose = () => {
    if (socket !== current) return;
    socket = null;
    clearInterval(heartbeat);
    heartbeat = null;
    powerStatus(false, 'Local power connection closed; retrying. System sleep is not prevented.');
    if (tickId !== null) scheduleTick(tickId);
    if (!stopped) reconnect = setTimeout(connectPower, 5000);
  };
}

self.onmessage = ({ data }) => {
  if (data.type === 'tick') {
    scheduleTick(data.id);
  } else if (data.type === 'cancel-tick') {
    cancelTick();
  } else if (data.type === 'reconnect') {
    // Do not wait for stale TCP connections to time out after thaw/network loss.
    const previous = socket;
    socket = null;
    clearInterval(heartbeat);
    heartbeat = null;
    previous?.close();
    connectPower();
  } else if (data.type === 'power') {
    // No requests to an unrelated host or an exposed network power endpoint.
    const url = data.url ? new URL(data.url) : null;
    if (url && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
      && ['ws:', 'wss:'].includes(url.protocol) && url.pathname === '/__ryugu/power') {
      powerUrl = url.href;
      connectPower();
    } else {
      powerStatus(false, 'Hosted page: no local sleep prevention. Browser/OS suspension remains possible.');
    }
  } else if (data.type === 'stop') {
    stopped = true;
    cancelTick();
    clearTimeout(reconnect);
    clearInterval(heartbeat);
    socket?.close();
    self.close();
  }
};
