// Core types for the test client

export const ACTIONS = {
    ENTERING: "entering",
    LOOPING: "looping",
    ON_BREAK: "on_break",
    EXITING: "exiting"
} as const;

export type ShuttleAction = typeof ACTIONS[keyof typeof ACTIONS];

export interface QueuedAction {
    id: string;
    action: ShuttleAction;
    route?: string;
    duration?: number;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
}

export interface Shuttle {
    id: string;
    state: ShuttleAction | null;
    next_state: ShuttleAction | null;
    queue: QueuedAction[];
}

export interface ShuttlesState {
    [shuttleId: string]: Shuttle;
}

export interface EventCounts {
    locationCount: number;
    geofenceCount: number;
}

export interface TestEvent {
    type: ShuttleAction;
    route?: string;
    duration?: number;
}

export interface TestShuttle {
    id: string;
    events: TestEvent[];
}

export interface TestData {
    shuttles: TestShuttle[];
}

export class TestCancel extends Error {
    constructor() {
        super('Test cancelled');
    }
}

export function warnUser(e: unknown): void {
    console.error(e);
}

export async function assertOk(res: Response): Promise<Response> {
    if (!res.ok) {
        let detail: string | undefined;
        try {
            detail = await res.text();
        } catch (err) {
            console.warn("Failed to read response body:", err);
        }
        throw new Error(`HTTP ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
    }
    return res;
}
