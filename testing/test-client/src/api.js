// api wrappers

// API base URL - uses proxy by default, can be overridden with VITE_TEST_BACKEND_URL
// Native: uses Vite proxy (/api/* → http://localhost:4000/api/*)
// Docker: uses nginx proxy (/api/* → http://test-server:4000/api/*)
// Direct: set VITE_TEST_BACKEND_URL=http://localhost:4000 to bypass proxy
const API_BASE = import.meta.env.VITE_TEST_BACKEND_URL || '';

export function fetchShuttles() {
    return fetch(`${API_BASE}/api/shuttles`);
}

export function fetchEvents() {
    return fetch(`${API_BASE}/api/events/today`);
}

export function deleteEvents(keepShuttles) {
    return fetch(`${API_BASE}/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
}

export function fetchRoutes() {
    return fetch(`${API_BASE}/api/routes`);
}

// signal is optional
export function addShuttle(signal) {
    return fetch(`${API_BASE}/api/shuttles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: signal
    });
}

// expects stateObj { state, route, more data if necessary... }
export function setNextState(shuttleId, stateObj, signal) {
    return fetch(`${API_BASE}/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(stateObj),
        signal: signal
    });
}
