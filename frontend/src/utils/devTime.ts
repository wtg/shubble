/**
 * DEV ONLY: Centralized time override for testing.
 * Shifts all frontend time to simulate a target hour.
 * Set DEV_ENABLED = false or remove this file for production.
 */

const DEV_ENABLED = false; // import.meta.env.DEV;
const TARGET_HOUR = 14; // 2:00 PM
const TARGET_MINUTE = 0;

// Compute offset once at module load
const _offset = DEV_ENABLED
  ? (() => {
      const target = new Date();
      target.setHours(TARGET_HOUR, TARGET_MINUTE, 0, 0);
      return target.getTime() - Date.now();
    })()
  : 0;

/** Returns a Date shifted by the dev offset (no-op in production). */
export function devNow(): Date {
  return new Date(Date.now() + _offset);
}

/** Returns Date.now() shifted by the dev offset (no-op in production). */
export function devNowMs(): number {
  return Date.now() + _offset;
}
