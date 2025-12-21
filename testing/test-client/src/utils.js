// utils.js

export const STATES = Object.freeze({
    WAITING: "waiting",
    ENTERING: "entering",
    LOOPING: "looping",
    ON_BREAK: "on_break",
    EXITING: "exiting"
});

export class TestCancel extends Error {
    constructor() {
        super();
    }
}

// TODO: replace console errors with UI alerts in App.jsx
export function warnUser(e) {
    console.error(e);
}

// helper to reveal http errors
export async function assertOk(res) {
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
