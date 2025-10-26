let isRunning = false;
let getShuttles = null;
const eventChains = new Map();
const NEXT_STATES = ["waiting", "entering", "looping", "on_break", "exiting"];

/*
Normal run: all chains resolve, promise.all resolves
User stop: queued tasks check isRunning and resolve early, promise.all resolves
Any shuttle error: whole run stops. that chain rejects, promise.all rejects, all further errors are ignored
*/
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
                warnUser(err);
                throw err;
            }
        }
    });
    eventChains.set(shuttleId, next);
}

export async function executeTest(testData) {
    if (isRunning) {
        warnUser("Automated test already running");
        return;
    }

    console.log("Starting automated test");
    isRunning = true;
    try {
        // queue all shuttle events
        for (const shuttle of testData.shuttles) {
            // add shuttle first, then add its events
            enqueue(shuttle.id, addShuttle);
            for (const evt of shuttle.events) {
                if (!NEXT_STATES.includes(evt.type)) {
                    throw new Error(`Unexpected event type ${evt.type}`);
                }
                enqueue(shuttle.id, () => executeEvent(shuttle.id, evt));
            }
        }
    } catch (err) {
        warnUser(err);
        isRunning = false;
        eventChains.clear();
        return;
    }

    console.log("Loaded shuttle event chains", eventChains);
    try {
        // concurrently execute all shuttle event chains
        await Promise.all([...eventChains.values()]);
        console.log("Automated test finished successfully");
    } finally {
        isRunning = false;
        eventChains.clear();
    }
}

async function executeEvent(id, evt) {
    await validateState(id);
    console.log("validiated", id);
    await setState(id, evt.type);
    console.log("set next state", id, evt);
}

// current state is finished when the next state is free (waiting)
async function validateState(id) {
    while (isRunning) {
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log(getShuttles());
        const shuttle = getShuttles().find(s => s.id === id);
        if (shuttle.next_state === NEXT_STATES[0]) {
            return;
        }
    }
}

// TODO: use event driven updates instead of polling with getShuttles
export function setGetShuttles(func) {
    getShuttles = func;
}

// TODO: fix stopTest to also stop in-execution promises
export function stopTest() {
    isRunning = false;
}

// TODO: replace console errors with UI alerts in App.jsx
function warnUser(e) {
    console.error(e);
}

// api call wrappers
function addShuttle() {
    return fetch("/api/shuttles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
    }).then(assertOk);
}

function setState(shuttleId, state) {
    return fetch(`/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: state }),
    }).then(assertOk);
}

function clearData() {
    return fetch(`/api/events/today?keepShuttles=false`, {
        method: "DELETE",
    }).then(assertOk);
}

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
