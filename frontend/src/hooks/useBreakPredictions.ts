import { useState, useEffect } from 'react';
import config from '../utils/config';

/**
 * A single upcoming-break prediction as returned by /api/predictions.
 * Matches backend.fastapi.break_detection.predict_upcoming_breaks entries.
 */
export interface BreakPrediction {
  run: string;                 // e.g. "West-1 (223/229)"
  predicted_start: string;     // ISO, campus-local
  predicted_end: string;       // ISO, campus-local
  confidence: number;          // 0..1
  lead_min: number;            // minutes from now to predicted_start
  source: string;              // scheduled-active | discovered | bimodal-mode | bimodal-mode-driver | *-driver | ...
  sigma_min: number;           // spread (minutes)
  db_verified: boolean | null; // null = DB check skipped
  driver_id: number | null;    // non-null when driver override applied
}

export interface ReactiveObserved {
  vehicle_id: string;
  observed_at: string;         // ISO
  lead_min: number;            // always 0 — happening now
  source: 'reactive-observed';
}

export interface BreakPredictionsResponse {
  generated_at: string;
  lookahead_min: number;
  db_slots_count: number | null;
  active_drivers_matched: number;
  n_predictions: number;
  predictions: BreakPrediction[];
  n_reactive_observed: number;
  reactive_observed: ReactiveObserved[];
}

interface UseBreakPredictionsOpts {
  lookaheadMin?: number;
  pollIntervalMs?: number;  // default 60s
}

/**
 * Polls /api/predictions and exposes the current set of upcoming breaks.
 * Returns `{ data, loading, error }`. Refreshes every `pollIntervalMs`.
 */
export function useBreakPredictions(
  opts: UseBreakPredictionsOpts = {},
) {
  const { lookaheadMin = 180, pollIntervalMs = 60_000 } = opts;
  const [data, setData] = useState<BreakPredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctl = new AbortController();
    let timer: number | undefined;

    const fetchOnce = async () => {
      try {
        const url = `${config.apiBaseUrl}/api/predictions?lookahead_min=${lookaheadMin}`;
        const res = await fetch(url, { signal: ctl.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = (await res.json()) as BreakPredictionsResponse;
        setData(body);
        setError(null);
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          setError((e as Error).message);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchOnce();
    timer = window.setInterval(fetchOnce, pollIntervalMs);
    return () => {
      ctl.abort();
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [lookaheadMin, pollIntervalMs]);

  return { data, loading, error };
}
