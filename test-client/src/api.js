// api wrappers

const API_BASE_URL = import.meta.env.VITE_TEST_BACKEND_URL || '';

export function fetchShuttles() {
    return fetch(`${API_BASE_URL}/api/shuttles`);
}

export function fetchEvents() {
    return fetch(`${API_BASE_URL}/api/events/today`);
}

export function deleteEvents(keepShuttles) {
    return fetch(`${API_BASE_URL}/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
}

export function fetchRoutes() {
    return fetch(`${API_BASE_URL}/api/routes`);
}

// signal is optional
export function addShuttle(signal) {
    return fetch(`${API_BASE_URL}/api/shuttles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: signal
    });
}

// expects stateObj { state, route, more data if necessary... }
export function setNextState(shuttleId, stateObj, signal) {
    return fetch(`${API_BASE_URL}/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(stateObj),
        signal: signal
    });
}
