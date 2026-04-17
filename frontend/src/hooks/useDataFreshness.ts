import { useEffect, useReducer } from 'react';

/** Freshness categories for live data. */
export type FreshnessState = 'fresh' | 'aging' | 'stale' | 'none';

/**
 * Age thresholds (seconds from last update):
 *   < FRESH_UNTIL_SEC  → fresh   (pulsing)
 *   < STALE_AFTER_SEC  → aging   (still trustworthy, no pulse)
 *   >= STALE_AFTER_SEC → stale   (warning color + icon)
 *
 * 30s fresh matches the worker cycle (~5s) × 6 — within 6 worker cycles
 * the data is "current". 120s stale matches TRUST-04 requirement.
 */
const FRESH_UNTIL_SEC = 30;
const STALE_AFTER_SEC = 120;

/**
 * Freshness state machine over a last-update timestamp.
 *
 * State is DERIVED at render time from `timestamp + Date.now()`. The
 * effect only schedules tiny force-update ticks at exact threshold
 * boundaries (30s and 120s) to cause a re-render when the derived state
 * would change. Between transitions, zero work. On visibility change,
 * we resync from wall-clock so a long background pause doesn't leave
 * the displayed state stale.
 *
 * @param timestamp Last known data timestamp (null while loading).
 * @returns One of 'fresh' | 'aging' | 'stale' | 'none'.
 */
export function useDataFreshness(timestamp: Date | null): FreshnessState {
  // A monotonic counter that increments on each scheduled transition.
  // Incrementing forces a re-render without holding any derived data in
  // state — that derived data is computed below from `timestamp` and
  // the current wall-clock, which is exactly what we want to stay in
  // sync with. This pattern avoids the "setState directly in effect"
  // lint because no setState is called synchronously.
  const [, forceTick] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    if (!timestamp) return;

    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      const ageSec = (Date.now() - timestamp.getTime()) / 1000;
      if (ageSec < FRESH_UNTIL_SEC) {
        // Fire exactly when fresh → aging.
        timeoutId = setTimeout(() => {
          forceTick();
          schedule();
        }, (FRESH_UNTIL_SEC - ageSec) * 1000);
      } else if (ageSec < STALE_AFTER_SEC) {
        // Fire exactly when aging → stale.
        timeoutId = setTimeout(() => {
          forceTick();
          // Stale is terminal — no further scheduling needed.
        }, (STALE_AFTER_SEC - ageSec) * 1000);
      }
      // else: already stale, terminal.
    };
    schedule();

    const onVisibilityChange = () => {
      if (!document.hidden) {
        // Tab returned. setTimeout may have been throttled; reschedule
        // from wall-clock so the next transition fires at the right time,
        // and force an immediate re-render to reflect the current age.
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        forceTick();
        schedule();
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [timestamp]);

  return computeState(timestamp);
}

function computeState(timestamp: Date | null): FreshnessState {
  if (!timestamp) return 'none';
  const ageSec = (Date.now() - timestamp.getTime()) / 1000;
  if (ageSec < FRESH_UNTIL_SEC) return 'fresh';
  if (ageSec < STALE_AFTER_SEC) return 'aging';
  return 'stale';
}
