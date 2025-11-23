// api wrappers

export function fetchShuttles() {
    return fetch("/api/shuttles");
}

export function fetchEvents() {
    return fetch("/api/events/today");
}

export function deleteEvents(keepShuttles) {
    return fetch(`/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
}

// signal is optional
export function addShuttle(signal) {
    return fetch("/api/shuttles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: signal
    });
}

// route and signal are optional
// refactor if more data is ever passed through this endpoint
export function setNextState(shuttleId, state, route, signal) {
    return fetch(`/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({state, route}),
        signal: signal
    });
}
