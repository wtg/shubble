import * as api from "./api.js";
import { STATES, TestCancel, warnUser, assertOk } from "./utils.js";

export default function Tester(getShuttlesFn) {
    let getShuttles = getShuttlesFn;
    let controller = null;
    const eventChains = new Map();

    const isRunning = () => controller && !controller.signal.aborted;
    const findShuttle = (id) => getShuttles().find(s => s.id === id);

    /*
    Normal run: all chains resolve, promise.all resolves, test succeeds
    User stop: first task to see the abort throws, promise.all throws, test stops nicely
    Any event error: first task to throw throws, promise.all throws, further errors ignored
    buildEventChains error: gate prevents earlier promises from running, and the map is wiped safely
    */
    async function startTest(testData) {
        if (isRunning()) {
            warnUser("Test already running");
            return;
        }

        controller = new AbortController();
        let openGate;
        const gate = new Promise(resolve => { openGate = resolve; });
        console.log("Started automated test");

        try {
            buildEventChains(testData, gate);
        } catch (err) {
            warnUser(err);
            console.log("Error building event chains");
            eventChains.clear();
            controller = null;
            return;
        }
        console.log("Built shuttle event chains", eventChains);

        try {
            // concurrently execute all shuttle event chains
            openGate();
            await Promise.all([...eventChains.values()]);
            console.log("Test finished successfully");
        } catch (err) {
            // only the first task that throws will be caught, promise.all ignores the rest
            if (err instanceof TestCancel) {
                // if user stopped, controller already aborted. other tasks will check and throw
                console.log("Test stopped successfully");
            } else {
                // else, an error happened. abort now so that other tasks will check and throw
                controller.abort();
                warnUser(err);
                console.log("Test failed");
            }
        } finally {
            eventChains.clear();
            controller = null;
        }
    }

    function buildEventChains(testData, gate) {
        const VALID_STATES = new Set(Object.values(STATES));
        for (const shuttle of testData.shuttles) {
            // queue addShuttle first, before the test case events
            enqueue(shuttle.id, makeApiCall, [api.addShuttle], gate);
            let lastEvt = null;
            for (const evt of shuttle.events) {
                if (!VALID_STATES.has(evt.type)) {
                    throw new Error(`Unexpected event type ${evt.type}`);
                }
                // capture the current state of lastEvt for proper closure
                const prev = lastEvt;
                enqueue(shuttle.id, executeEvent, [shuttle.id, evt, prev]);
                lastEvt = evt;
            }
        }
    }

    function enqueue(shuttleId, task, params, gate) {
        const current = eventChains.get(shuttleId) ?? gate;
        const next = current.then(() => task(...params));
        eventChains.set(shuttleId, next);
    }

    // at the boundary of every async operation, this function implicitly checks for
    // controller.aborted. if aborted, the awaits will throw
    async function executeEvent(id, evt, lastEvt) {
        throwIfAborted();

        // if the shuttle was just added, verify it's there
        if (lastEvt === null) {
            await waitUntil(() => findShuttle(id) !== undefined, 1000, 10000);
        }

        // request a state change
        await makeApiCall(api.setNextState, id, evt.type);
        console.log(`shuttle ${id} set next_state to`, evt);

        // verify that the event started (assumes that the last event finished and cannot block)
        await waitUntil(() => {
            const shuttle = findShuttle(id);
            return shuttle.state === evt.type ||
                (shuttle.state === STATES.WAITING && evt.type === STATES.ON_BREAK);
        }, 1000, 10000);
        console.log(`shuttle ${id} started event`, evt);

        // wait until the event finishes before returning and resolving the event's promise
        if (evt.type === STATES.WAITING || evt.type === STATES.ON_BREAK) {
            await sleep(evt.duration * 1000);
        } else {
            // 1 sec interval, 10 min timeout, no event should take longer than 10 minutes
            await waitUntil(
                () => findShuttle(id).state === STATES.WAITING,
            1000, 600000);
        }
        console.log(`shuttle ${id} completed event`, evt);
    }

    // check that fn returns true within some timeout
    async function waitUntil(fn, interval, timeout) {
        const start = Date.now();
        while (true) {
            await sleep(interval);
            // probably misses the last check due to processing overhead, but not a big deal
            if (Date.now() - start > timeout) {
                throw new Error(`timeout waiting for condition ${fn}`);
            }
            if (fn()) return;
        }
    }

    // abortable sleep function
    function sleep(delay) {
        return new Promise((resolve, reject) => {
            let timer;

            const onAbort = () => {
                cleanup();
                reject(new TestCancel());
            };

            const cleanup = () => {
                clearTimeout(timer);
                controller.signal.removeEventListener("abort", onAbort);
            };

            // attach the abort listener first
            controller.signal.addEventListener("abort", onAbort);
            // controller could've aborted before the listener is attached
            if (controller.signal.aborted) {
                onAbort();
                return;
            }
            // only then start the timeout
            timer = setTimeout(() => {
                cleanup();
                resolve();
            }, delay);
        });
    }

    function stopTest() {
        if (controller) {
            controller.abort();
        }
    }

    function throwIfAborted() {
        if (!isRunning()) {
            throw new TestCancel();
        }
    }

    // wrapper for abortable api calls
    async function makeApiCall(apiFn, ...params) {
        throwIfAborted();
        try {
            const res = await apiFn(...params, controller.signal);
            return assertOk(res);
        } catch (err) {
            if (err.name === "AbortError") {
                // if the call aborted, convert to TestCancel so it's caught properly
                throw new TestCancel();
            }
            throw err;
        }
    }

    return { startTest, stopTest };
}
