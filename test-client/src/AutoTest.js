let isRunning = false;
let getShuttles = null;
const eventChains = new Map();
const NEXT_STATES = ["waiting", "entering", "looping", "on_break", "exiting"];

/*
Normal run: all chains resolve, promise.all resolves
User stop: queued tasks check isRunning and resolve silently, promise.all resolves
Any shuttle error: entire test fails. error task throws, promise.all throws, further errors ignored
*/
export async function executeTest(testData) {
    if (isRunning) {
        warnUser("Test already running");
        return;
    }

    console.log("Starting automated test");
    isRunning = true;
    try {
        // build all shuttle event chains
        for (const shuttle of testData.shuttles) {
            // queue addShuttle first, before the test case events
            enqueue(shuttle.id, addShuttle);
            let lastEvtType = null;
            for (const evt of shuttle.events) {
                if (!NEXT_STATES.includes(evt.type)) {
                    throw new Error(`Unexpected event type ${evt.type}`);
                }
                const prev = lastEvtType;
                enqueue(shuttle.id, () => executeEvent(shuttle.id, evt, prev));
                lastEvtType = evt.type;
            }
        }
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

async function executeEvent(id, evt, lastEvtType) {
    await verifyCurrentState(id, lastEvtType);
    await setNextState(id, evt.type);
    console.log("set next state for", id, evt);
    if (evt.type === NEXT_STATES[0] || evt.type === NEXT_STATES[3]) {
        await new Promise(resolve => setTimeout(resolve, evt.duration * 1000));
    }
}

// This function waits until shuttle.next_state is free
// If, at that time, shuttle.state is mismatched, throw an error
// This ensures correctness (last state must equal last test case event)
async function verifyCurrentState(id, lastEvtType) {
    let checkCount = 0, checkLimit = 3;
    const checkLast = true;

    while (isRunning) {
        await new Promise(resolve => setTimeout(resolve, 1000));

        const shuttle = await (lastEvtType === null ? findShuttle(id, 3) : findShuttle(id));

        if (shuttle.next_state === NEXT_STATES[0]) {
            if (checkLast && lastEvtType !== null) {
                if ((shuttle.state !== lastEvtType
                    && !(shuttle.state === NEXT_STATES[0] && lastEvtType === NEXT_STATES[3]))) {
                    if (checkCount < checkLimit) {
                        checkCount++;
                        continue;
                    }
                    throw new Error(`Shuttle ${id} expected state ${lastEvtType}, got ${shuttle.state}`);
                }
            }
            return;
        }
    }
}

// allow multiple tries to find shuttles, useful for processing the first event
async function findShuttle(id, tries = 1) {
    for (let i = 0; i < tries; i++) {
        const shuttle = getShuttles().find(s => s.id === id);
        if (shuttle !== undefined) return shuttle;
        if (i < tries - 1) await new Promise(resolve => setTimeout(resolve, 1000));
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

// TODO: replace console errors with UI alerts in App.jsx
function warnUser(e) {
    console.error(e);
}

// api call wrappers
function addShuttle() {
    if (!isRunning) return;
    return fetch("/api/shuttles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
    }).then(assertOk);
}

function setNextState(shuttleId, state) {
    if (!isRunning) return;
    return fetch(`/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: state }),
    }).then(assertOk);
}

/*
function clearData() {
    if (!isRunning) return;
    return fetch(`/api/events/today?keepShuttles=false`, {
        method: "DELETE",
    }).then(assertOk);
}
*/

// helper to reveal http errors
async function assertOk(res) {
    if (!res.ok) {
        let detail;
        try {
            detail = await res.text();
        } catch (err) {
            console.warn("Failed to read response body:", err);
        }
        throw new Error(`HTTP ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`)
    }
    return res;
}
