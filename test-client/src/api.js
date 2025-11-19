// API call wrapper functions

export function fetchShuttles() {
    return fetch("/api/shuttles");
}

export function fetchEvents() {
    return fetch("/api/events/today");
}

export function deleteEvents(keepShuttles) {
    return fetch(`/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
}

export function addShuttle(signal) {
    return fetch("/api/shuttles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: signal
    });
}

// data and signal are optional
export function setNextState(shuttleId, state, data, signal) {
    return fetch(`/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({state, data}),
        signal: signal
    });
}
