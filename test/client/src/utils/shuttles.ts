// Shuttle state management and API calls
import config from '../api/config.ts';
import type { ShuttlesState, ShuttleAction } from '../types.ts';

// API calls
export async function fetchShuttlesFromApi(): Promise<ShuttlesState> {
    const res = await fetch(`${config.apiBaseUrl}/api/shuttles`);
    return res.json() as Promise<ShuttlesState>;
}

export async function fetchRoutes(): Promise<string[]> {
    const res = await fetch(`${config.apiBaseUrl}/api/routes`);
    return res.json() as Promise<string[]>;
}

export async function addShuttleToApi(signal?: AbortSignal): Promise<Response> {
    return fetch(`${config.apiBaseUrl}/api/shuttles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal
    });
}

export interface QueueActionInput {
    action: ShuttleAction;
    route?: string;
    duration?: number;
}

export async function addToQueueApi(
    shuttleId: string,
    actions: QueueActionInput[],
    signal?: AbortSignal
): Promise<Response> {
    return fetch(`${config.apiBaseUrl}/api/shuttles/${shuttleId}/queue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actions }),
        signal
    });
}

