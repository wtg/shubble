// Test file loading - parses test JSON files and queues actions
import type { TestData, ShuttleAction } from '../types.ts';
import { ACTIONS } from '../types.ts';
import { addShuttleToApi, addToQueueApi } from './shuttles.ts';

export interface ValidationError {
    shuttleIndex?: number;
    eventIndex?: number;
    message: string;
}

export interface ValidationResult {
    valid: boolean;
    errors: ValidationError[];
    data: TestData | null;
}

export interface ParseResult {
    success: boolean;
    error?: string;
    data?: TestData;
}

/* export interface for confirming export is successfull?*/

/**
 * Parse a test file from text content.
 */
export function parseTestFile(text: string): ParseResult {
    try {
        const data = JSON.parse(text) as TestData;
        return { success: true, data };
    } catch (err) {
        return {
            success: false,
            error: err instanceof Error ? err.message : 'Invalid JSON'
        };
    }
}

/**
 * Validate test data structure and return detailed results.
 */
export function validateTestData(data: TestData): ValidationResult {
    const errors: ValidationError[] = [];

    if (!data.shuttles || !Array.isArray(data.shuttles)) {
        return {
            valid: false,
            errors: [{ message: 'Missing or invalid "shuttles" array' }],
            data: null
        };
    }

    if (data.shuttles.length === 0) {
        return {
            valid: false,
            errors: [{ message: 'No shuttles defined in file' }],
            data: null
        };
    }

    const validActions = new Set<ShuttleAction>(Object.values(ACTIONS));

    for (let shuttleIndex = 0; shuttleIndex < data.shuttles.length; shuttleIndex++) {
        const shuttle = data.shuttles[shuttleIndex];

        if (!shuttle.events || !Array.isArray(shuttle.events)) {
            errors.push({
                shuttleIndex,
                message: 'Missing or invalid "events" array'
            });
            continue;
        }

        if (shuttle.events.length === 0) {
            errors.push({
                shuttleIndex,
                message: 'No events defined'
            });
        }

        for (let eventIndex = 0; eventIndex < shuttle.events.length; eventIndex++) {
            const evt = shuttle.events[eventIndex];

            if (!evt.type) {
                errors.push({
                    shuttleIndex,
                    eventIndex,
                    message: 'Event missing "type" field'
                });
                continue;
            }

            if (!validActions.has(evt.type)) {
                errors.push({
                    shuttleIndex,
                    eventIndex,
                    message: `Unknown action type "${evt.type}"`
                });
            }

            if (evt.type === ACTIONS.LOOPING && !evt.route) {
                errors.push({
                    shuttleIndex,
                    eventIndex,
                    message: 'Looping action requires "route" field'
                });
            }

            if (evt.type === ACTIONS.ON_BREAK && typeof evt.duration !== 'number') {
                errors.push({
                    shuttleIndex,
                    eventIndex,
                    message: 'On break action requires numeric "duration" field'
                });
            }
        }
    }

    return {
        valid: errors.length === 0,
        errors,
        data: errors.length === 0 ? data : null
    };
}

/**
 * Load a test file and queue all actions for each shuttle.
 * Creates new shuttles as needed.
 */
export async function loadTestFile(file: File): Promise<void> {
    const text = await file.text();
    const parseResult = parseTestFile(text);

    if (!parseResult.success || !parseResult.data) {
        throw new Error(parseResult.error || 'Failed to parse file');
    }

    const validation = validateTestData(parseResult.data);
    if (!validation.valid) {
        throw new Error(validation.errors[0]?.message || 'Validation failed');
    }

    console.log('Loading test file:', file.name);

    for (const testShuttle of parseResult.data.shuttles) {
        // Create a new shuttle
        const res = await addShuttleToApi();
        if (!res.ok) {
            throw new Error(`Failed to add shuttle: ${res.statusText}`);
        }

        const shuttle = await res.json();
        const shuttleId = shuttle.id;

        // Queue all actions for this shuttle
        const actions = testShuttle.events.map(evt => ({
            action: evt.type,
            route: evt.route,
            duration: evt.duration
        }));

        await addToQueueApi(shuttleId, actions);
        console.log(`Queued ${actions.length} actions for shuttle ${shuttleId}`);
    }

    console.log('Test file loaded successfully');
}

/**
 * Import validated test data directly.
 */
export async function importTestData(data: TestData): Promise<void> {
    for (const testShuttle of data.shuttles) {
        const res = await addShuttleToApi();
        if (!res.ok) {
            throw new Error(`Failed to add shuttle: ${res.statusText}`);
        }

        const shuttle = await res.json();
        const shuttleId = shuttle.id;

        const actions = testShuttle.events.map(evt => ({
            action: evt.type,
            route: evt.route,
            duration: evt.duration
        }));

        await addToQueueApi(shuttleId, actions);
    }
}

/*
export async function buildTestFile(){
//accepts a list of QueuedActions
// see test/client/types.ts
// and builds a json test file from the actions
}

*/