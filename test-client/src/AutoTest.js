let isRunning = false;
let controller = null;

// TODO: replace console errors with UI alerts in App.jsx
function warnUser(e) {
    console.error(e);
}

// expects a parsed json object with events array as testData
export async function executeTest(testData) {
    if (isRunning) {
        warnUser("Automated test already running");
        return;
    }
    if (!testData || !Array.isArray(testData.events)) {
        warnUser("Missing test data events array");
        return;
    }

    isRunning = true;
    for (const evt of testData.events) {
        if (!isRunning) return;

        controller = new AbortController();
        try {
            await executeEvent(evt, controller.signal);
        } catch (err) {
            if (err.name !== "AbortError") {
                warnUser(err);
            }
            break;
        } finally {
            controller = null;
        }
    }
    isRunning = false;
}

async function executeEvent(evt, signal) {
    if (evt.type === "AddShuttle") {
        await addShuttles(evt.count, signal);
    } else if (evt.type === "SetState") {
        await setState(evt.shuttleId, evt.state, signal);
    } else if (evt.type === "ClearData") {
        await clearData(signal);
    } else {
        throw new Error(`Unexpected event type ${evt.type} in test case`);
    }
}

export function stopCurrentTest() {
    isRunning = false;
    if (controller) {
        controller.abort();
    }
}

// api call wrappers
async function addShuttles(count, signal) {
    if (!Number.isInteger(count) || count < 0 || count > 15) {
        throw new Error("Invalid shuttle count");
    }
    const tasks = Array.from({length: count}, () => 
        fetch("/api/shuttles", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: signal
        }).then(assertOk)
    );
    await Promise.all(tasks);
}

function setState(shuttleId, state, signal) {
    return fetch(`/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: state }),
        signal: signal
    }).then(assertOk);
}

function clearData(signal) {
    return fetch(`/api/events/today?keepShuttles=false`, {
        method: "DELETE",
        signal: signal
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
