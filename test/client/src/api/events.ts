// Event-related API calls and state management
import config from './config.ts';
import type { EventCounts } from '../types.ts';

export async function fetchEventCounts(): Promise<EventCounts> {
    const res = await fetch(`${config.apiBaseUrl}/api/events/today`);
    return res.json() as Promise<EventCounts>;
}

export async function deleteEvents(keepShuttles: boolean): Promise<void> {
    await fetch(`${config.apiBaseUrl}/api/events/today?keepShuttles=${keepShuttles}`, {
        method: "DELETE"
    });
}
