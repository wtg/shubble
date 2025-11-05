// API call wrapper functions

export function fetchShuttles() {
    return fetch("/api/shuttles");
}

export function addShuttle() {
    return fetch("/api/shuttles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
    });
}

export function setNextState(state) {
    return fetch(`/api/shuttles/${selectedId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: state }),
    });
}

export function fetchEvents() {
    fetch("/api/events/today");
}

export function deleteEvents(keepShuttles) {
    fetch(`/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
}
