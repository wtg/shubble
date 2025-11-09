import * as api from "./api.js";
import { STATES, warnUser, assertOk } from "./utils.js";

let isRunning = false;
let getShuttles = null;
const eventChains = new Map();

/*
Normal run: all chains resolve, promise.all resolves
User stop: queued tasks check isRunning and resolve silently, promise.all resolves
Any shuttle error: test fails. error task throws, promise.all throws, further errors ignored
*/
export async function startTest(testData) {
    if (isRunning) {
        warnUser("Test already running");
        return;
    }
    isRunning = true;
    console.log("Started automated test");

    try {
        buildEventChains(testData);
    } catch (err) {
        warnUser(err);
        isRunning = false;
        eventChains.clear();
        return;
    }
    console.log("Built shuttle event chains", eventChains);

    try {
        // concurrently execute all shuttle event chains
        await Promise.all([...eventChains.values()]);
        console.log("Test finished successfully");
    } catch (err) {
        warnUser(err);
        console.log("Test failed");
    } finally {
        isRunning = false;
        eventChains.clear();
    }
}

function buildEventChains(testData) {
    for (const shuttle of testData.shuttles) {
        // queue addShuttle first, before the test case events
        enqueue(shuttle.id, addShuttle);
        let lastEvt = null;
        for (const evt of shuttle.events) {
            if (!Object.values(STATES).includes(evt.type)) {
                throw new Error(`Unexpected event type ${evt.type}`);
            }
            // capture the current state of lastEvt for the anonymous function's closure
            const prev = lastEvt;
            enqueue(shuttle.id, () => executeEvent(shuttle.id, evt, prev));
            lastEvt = evt;
        }
    }
}

function enqueue(shuttleId, task) {
    if (!isRunning) {
        throw new Error("enqueue() called while test is not running");
    }
    const current = eventChains.get(shuttleId) ?? Promise.resolve();
    const next = current.then(async () => {
        if (isRunning) {
            try {
                await task();
            } catch (err) {
                isRunning = false;
                throw err;
            }
        }
    });
    eventChains.set(shuttleId, next);
}

async function executeEvent(id, evt, lastEvt) {
    // if the shuttle was just added, verify it's there
    if (lastEvt === null) {
        await retryUntil(() => findShuttle(id) !== undefined);
    }

    // request a state change
    await setNextState(id, evt.type);
    console.log(`shuttle ${id} set next_state to`, evt)

    // verify that the event started (assumes that the last event finished and cannot block)
    await retryUntil(() => {
        const shuttle = findShuttle(id);
        return shuttle.state === evt.type ||
              (shuttle.state === STATES.WAITING && evt.type === STATES.ON_BREAK);
    });
    console.log(`shuttle ${id} started event`, evt);

    // wait until the event finishes before returning and resolving the event's promise
    if (evt.type === STATES.WAITING || evt.type === STATES.ON_BREAK) {
        await new Promise(resolve => setTimeout(resolve, evt.duration * 1000));
    } else {
        // 1 sec interval, 10 min timeout. no event should take longer than 10 minutes
        await waitUntil(
            () => findShuttle(id).state === STATES.WAITING,
        1000, 600000);
    }
    console.log(`shuttle ${id} completed event`, evt);
}

// check that fn returns true within a limited number of retries
async function retryUntil(fn, tries = 4, interval = 1000) {
    for (let i = 0; i < tries; i++) {
        await new Promise(resolve => setTimeout(resolve, interval));
        if (fn()) return;
    }
    throw new Error(`retry() failed on ${fn}`);
}

async function waitUntil(fn, interval, timeout) {
    const start = Date.now();
    while (isRunning) {
        await new Promise(resolve => setTimeout(resolve, interval))
        if (Date.now() - start > timeout) {
            throw new Error("Timeout waiting for condition");
        }
        if (fn()) return;
    }
}

function findShuttle(id) {
    return getShuttles().find(s => s.id === id);
}

// in-execution promises will cancel their tasks and resolve silently
export function stopTest() {
    isRunning = false;
}

export function setGetShuttles(func) {
    getShuttles = func;
}

// api call wrappers
async function addShuttle() {
    if (isRunning) {
        return api.addShuttle().then(assertOk);
    }
}

async function setNextState(shuttleId, state) {
    if (isRunning) {
        return api.setNextState(shuttleId, state).then(assertOk);
    }
}
