import { useEffect, useState } from 'react';

/**
 * RAF-driven "now" hook that re-renders subscribers only when the current
 * time crosses a bucket boundary (minute or second). Used by the schedule
 * countdown so per-frame ticks don't cause per-frame React re-renders.
 *
 * Why requestAnimationFrame instead of setInterval:
 *   - setInterval drifts when backgrounded (browsers throttle to ~1min),
 *     so the countdown would show stale values after a tab switch.
 *   - RAF pauses entirely while backgrounded. On return, we listen for
 *     `visibilitychange` and force an immediate resync from wall-clock
 *     time — no drift, no jumps.
 *
 * Why bucket comparison instead of raw ms:
 *   - RAF fires ~60 times/second. If we setState with Date.now() each
 *     frame, React re-renders 60x/sec. Bucketed (minute floor) state
 *     only changes at minute boundaries, so React re-renders ~once/min
 *     despite polling at 60Hz.
 *
 * @param granularity 'minute' for countdown displays, 'second' for fine
 *                    state transitions (e.g. data-freshness state edges).
 * @returns The epoch-ms at the start of the current bucket. Stable across
 *          the bucket window, changes on crossing.
 */
export function useCurrentTime(granularity: 'minute' | 'second' = 'minute'): number {
  const divisor = granularity === 'minute' ? 60_000 : 1_000;

  const [bucket, setBucket] = useState(() => Math.floor(Date.now() / divisor));

  useEffect(() => {
    let rafId: number;

    const tick = () => {
      const current = Math.floor(Date.now() / divisor);
      // setState with identity-equal value is a no-op for re-renders,
      // so this is effectively "only re-render when the bucket changes".
      setBucket(prev => (prev !== current ? current : prev));
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    const onVisibilityChange = () => {
      // When the tab returns to foreground, force an immediate resync.
      // RAF was paused while hidden, so `bucket` may be minutes stale.
      if (!document.hidden) {
        setBucket(Math.floor(Date.now() / divisor));
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [divisor]);

  return bucket * divisor;
}
