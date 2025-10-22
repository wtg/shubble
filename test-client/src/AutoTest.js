let isRunning = false;
let controller = null;

// expects a parsed json object as testData
export async function executeTest(testData) {
    if (isRunning) return;
    isRunning = true;

    for (const evt of testData.events) {
        if (!isRunning) return;

        controller = new AbortController();
        try {
            if (evt.type === "AddShuttle") {
                await addShuttles(evt.count);
            } else if (evt.type === "SetState") {
                await setState(evt.shuttleId, evt.state);
            } else if (evt.type === "ClearData") {
                await clearData();
            } else {
                throw new Error(`Unexpected event type (${evt.type}) in test case`);
            }
        } catch (err) {
            console.error(err);
        } finally {
            controller = null;
        }
    }

    isRunning = false;
}

async function addShuttles(count) {
    if (count > 15) throw new Error("Too many shuttles");
    for (let i = 0; i < count; i++) {
        await fetch("/api/shuttles", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });
    }
}

function setState(shuttleId, state) {
    return fetch(`/api/shuttles/${shuttleId}/set-next-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: state }),
    });
}

function clearData() {
    return fetch(`/api/events/today?keepShuttles=false`, {method: "DELETE"});
}

export function stopCurrentTest() {
    isRunning = false;
    if (controller) {
        controller.abort();
    }
}
