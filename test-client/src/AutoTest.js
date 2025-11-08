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
        console.log("All event calls have been executed");
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
    await verifyCurrentState(id, lastEvt);
    if (lastEvt && (lastEvt.type === STATES.WAITING || lastEvt.type === STATES.ON_BREAK)) {
        await new Promise(resolve => setTimeout(resolve, lastEvt.duration * 1000));
    }
    await setNextState(id, evt.type);
    console.log("set next state for", id, evt);
}

// This function waits until shuttle.next_state is free
// If, at that time, shuttle.state is mismatched, throw an error
// This ensures correctness (last state must equal last test case event)
async function verifyCurrentState(id, lastEvt) {
    let checkCount = 0, checkLimit = 3;
    const checkLast = true;

    while (isRunning) {
        await new Promise(resolve => setTimeout(resolve, 1000));

        const shuttle = await (lastEvt === null ? findShuttle(id, 3) : findShuttle(id));

        if (shuttle.next_state === STATES.WAITING) {
            if (checkLast && lastEvt !== null) {
                if (shuttle.state !== lastEvt.type
                    && !(shuttle.state === STATES.WAITING && lastEvt.type === STATES.ON_BREAK)) {
                    if (checkCount < checkLimit) {
                        checkCount++;
                        continue;
                    }
                    throw new Error(`Shuttle ${id} expected state ${lastEvt.type}, got ${shuttle.state}`);
                }
            }
            return;
        }
    }
}

// allow multiple tries to find shuttles by polling
async function findShuttle(id, tries = 1) {
    for (let i = 0; i < tries; i++) {
        const shuttle = getShuttles().find(s => s.id === id);
        if (shuttle !== undefined) return shuttle;
        if (i < tries - 1) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    throw new Error(`Couldn't find shuttle with id ${id}`);
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
