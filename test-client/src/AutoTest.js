import * as api from "./api.js";
import { STATES, TestCancel, warnUser, assertOk } from "./utils.js";

export default function Tester() {
    let getShuttles = null;
    let controller = null;
    const eventChains = new Map();

    /*
    Normal run: all chains resolve, promise.all resolves, test succeeds
    User stop: first task to see the abort throws, promise.all throws, test stops nicely
    Any event error: first task to throw throws, promise.all throws, further errors ignored
    buildEventChains error: gate prevents earlier promises from running, and the map is wiped safely
    */
    async function startTest(testData) {
        if (controller && !controller.signal.aborted) {
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
            if (err instanceof TestCancel) {
                console.log("Test stopped successfully");
            } else {
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
            enqueue(shuttle.id, addShuttle, [], gate);
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

    async function executeEvent(id, evt, lastEvt) {
        throwIfAborted();

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
            await sleep(evt.duration * 1000);
        } else {
            // 1 sec interval, 10 min timeout. no event should take longer than 10 minutes
            await waitUntil(
                () => findShuttle(id).state === STATES.WAITING,
            1000, 600000);
        }
        console.log(`shuttle ${id} completed event`, evt);
    }

    // check that fn returns true within some number of retries
    async function retryUntil(fn, tries = 5, interval = 1000) {
        for (let i = 0; i < tries; i++) {
            await sleep(interval);
            if (fn()) return;
        }
        throw new Error(`retry() failed on ${fn}`);
    }

    // check that fn returns true within some timeout
    async function waitUntil(fn, interval, timeout) {
        const start = Date.now();
        while (true) {
            await sleep(interval);
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

    function throwIfAborted() {
        if (controller.signal.aborted) {
            throw new TestCancel();
        }
    }

    function stopTest() {
        if (controller) {
            controller.abort();
        }
    }

    function findShuttle(id) {
        return getShuttles().find(s => s.id === id);
    }

    function setGetShuttles(func) {
        getShuttles = func;
    }

    // api call wrappers
    function addShuttle() {
        throwIfAborted();
        return api.addShuttle().then(assertOk);
    }

    function setNextState(shuttleId, state) {
        throwIfAborted();
        return api.setNextState(shuttleId, state).then(assertOk);
    }

    return { startTest, stopTest, setGetShuttles };
}
