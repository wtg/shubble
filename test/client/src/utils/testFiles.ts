// Test file loading - parses test JSON files and queues actions
import type { TestData, ShuttleAction } from '../types.ts';
import { ACTIONS } from '../types.ts';
import { addShuttleToApi, addToQueueApi } from './shuttles.ts';

/**
 * Load a test file and queue all actions for each shuttle.
 * Creates new shuttles as needed.
 */
export async function loadTestFile(file: File): Promise<void> {
    const text = await file.text();
    const testData = JSON.parse(text) as TestData;

    validateTestData(testData);

    console.log('Loading test file:', file.name);

    for (const testShuttle of testData.shuttles) {
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

function validateTestData(data: TestData): void {
    if (!data.shuttles || !Array.isArray(data.shuttles)) {
        throw new Error('Invalid test data: missing shuttles array');
    }

    const validActions = new Set<ShuttleAction>(Object.values(ACTIONS));

    for (const shuttle of data.shuttles) {
        if (typeof shuttle.id !== 'string') {
            throw new Error('Invalid test data: shuttle missing id');
        }

        if (!shuttle.events || !Array.isArray(shuttle.events)) {
            throw new Error(`Invalid test data: shuttle ${shuttle.id} missing events array`);
        }

        for (const evt of shuttle.events) {
            if (!validActions.has(evt.type)) {
                throw new Error(`Invalid test data: unknown action type "${evt.type}"`);
            }

            if (evt.type === ACTIONS.LOOPING && !evt.route) {
                throw new Error(`Invalid test data: looping action requires route`);
            }

            if (evt.type === ACTIONS.ON_BREAK && typeof evt.duration !== 'number') {
                throw new Error(`Invalid test data: ${evt.type} action requires duration`);
            }
        }
    }
}
