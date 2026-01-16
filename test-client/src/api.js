// api wrappers
import { config} from './config.js';

export function fetchShuttles() {
    return fetch(`${config.apiBaseUrl}/api/shuttles`);
}

export function fetchEvents() {
    return fetch(`${config.apiBaseUrl}/api/events/today`);
}

export function deleteEvents(keepShuttles) {
    return fetch(`${config.apiBaseUrl}/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
}

export function fetchRoutes() {
    return fetch(`${config.apiBaseUrl}/api/routes`);
}

// signal is optional
export function addShuttle(signal) {
    return fetch(`${config.apiBaseUrl}/api/shuttles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: signal
    });
}

// expects stateObj { state, route, more data if necessary... }
export function setNextState(shuttleId, stateObj, signal) {
    return fetch(`${config.apiBaseUrl}/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(stateObj),
        signal: signal
    });
}
