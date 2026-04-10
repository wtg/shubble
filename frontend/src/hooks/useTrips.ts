import { useState, useEffect } from 'react';
import config from '../utils/config';

/** Per-stop entry within a trip */
export interface TripStopETA {
  eta: string | null; // ISO datetime, null if passed or no prediction
  last_arrival: string | null;
  passed: boolean;
}

/** A single trip — one shuttle's run through a route */
export interface Trip {
  trip_id: string;
  route: string;
  departure_time: string; // ISO datetime
  actual_departure: string | null;
  scheduled: boolean;
  vehicle_id: string | null;
  status: 'scheduled' | 'active' | 'unassigned' | 'completed';
  stop_etas: Record<string, TripStopETA>;
}

/**
 * Formatted ETA strings keyed by stop key.
 * Used by the map stop-marker tooltip. Derived client-side from /api/trips.
 */
export type StopETAs = Record<string, string>;

/** Per-stop detail map for "Last arrived: HH:MM" tooltips. */
export type StopETADetails = Record<string, {
  eta: string;
  etaISO: string;
  route: string;
  vehicleId: string;
  lastArrival?: string;
}>;

const TIME_FORMAT: Intl.DateTimeFormatOptions = { hour: 'numeric', minute: '2-digit' };

/**
 * Derive per-stop "next shuttle" and "last arrived" views from the trips
 * array. Replaces the deleted `/api/etas` endpoint's role in feeding map
 * stop-pin tooltips — one source of truth is /api/trips.
 *
 * For each stop:
 *  - `stopETAs[stop]` = formatted HH:MM of the earliest upcoming ETA across
 *    all active/scheduled trips.
 *  - `stopETADetails[stop].lastArrival` = formatted HH:MM of the most
 *    recent last_arrival across all trips.
 */
export function deriveStopEtasFromTrips(trips: Trip[]): {
  stopETAs: StopETAs;
  stopETADetails: StopETADetails;
} {
  const stopETAs: StopETAs = {};
  const stopETADetails: StopETADetails = {};
  const earliestEtaMs: Record<string, number> = {};
  const latestLaMs: Record<string, number> = {};
  const now = Date.now();

  for (const trip of trips) {
    // Only active and scheduled trips contribute upcoming ETAs. Completed
    // trips contribute last_arrival history.
    const contributesEta = trip.status === 'active' || trip.status === 'scheduled';
    for (const [stop, info] of Object.entries(trip.stop_etas)) {
      if (contributesEta && info.eta) {
        const etaMs = new Date(info.eta).getTime();
        if (etaMs > now) {
          const prev = earliestEtaMs[stop];
          if (prev === undefined || etaMs < prev) {
            earliestEtaMs[stop] = etaMs;
            const formatted = new Date(etaMs).toLocaleTimeString(undefined, TIME_FORMAT);
            stopETAs[stop] = formatted;
            stopETADetails[stop] = {
              ...(stopETADetails[stop] ?? {}),
              eta: formatted,
              etaISO: info.eta,
              route: trip.route,
              vehicleId: trip.vehicle_id ?? '',
            };
          }
        }
      }
      if (info.last_arrival) {
        const laMs = new Date(info.last_arrival).getTime();
        const prev = latestLaMs[stop];
        if (prev === undefined || laMs > prev) {
          latestLaMs[stop] = laMs;
          const formatted = new Date(laMs).toLocaleTimeString(undefined, TIME_FORMAT);
          const existing = stopETADetails[stop] ?? {
            eta: '', etaISO: '', route: trip.route, vehicleId: trip.vehicle_id ?? '',
          };
          stopETADetails[stop] = { ...existing, lastArrival: formatted };
        }
      }
    }
  }
  return { stopETAs, stopETADetails };
}

/**
 * Fetches per-trip ETAs from the backend.
 *
 * Transport: server-sent events (SSE) via `/api/trips/stream` as the
 * primary push channel, with `/api/trips` polling as a slow fallback.
 * The backend publishes to the `shubble:trips_updated` Redis channel
 * at the end of every worker cycle, so SSE delivery is ~immediate.
 *
 * Failure mode: if the SSE connection permanently closes (endpoint
 * missing, proxy blocking, etc.), the hook switches to polling at
 * `pollInterval`. Transient reconnects (handled by the browser's
 * built-in EventSource retry) do not trigger the fallback.
 */
export function useTrips(enabled = true, pollInterval = 5000) {
  const [trips, setTrips] = useState<Trip[]>([]);

  useEffect(() => {
    if (!enabled) return;

    const controller = new AbortController();
    let cancelled = false;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let es: EventSource | null = null;

    const applyTrips = (data: unknown) => {
      if (Array.isArray(data) && !cancelled) setTrips(data as Trip[]);
    };

    const fetchTrips = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/trips`, {
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!response.ok || cancelled) return;
        const data = (await response.json()) as unknown;
        applyTrips(data);
      } catch (error) {
        if ((error as Error).name === 'AbortError') return;
        // Silent: network issues are common and the next tick will retry.
      }
    };

    const startPolling = () => {
      if (pollTimer !== null || cancelled) return;
      void fetchTrips();
      pollTimer = setInterval(() => { void fetchTrips(); }, pollInterval);
    };

    const stopPolling = () => {
      if (pollTimer !== null) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };

    // Kick off an initial fetch so the UI isn't blank while the SSE
    // connection is still opening. Subsequent updates come via SSE.
    void fetchTrips();

    // Try SSE push first. If EventSource isn't available (very old
    // browser), fall straight back to polling.
    if (typeof EventSource !== 'undefined') {
      try {
        es = new EventSource(`${config.apiBaseUrl}/api/trips/stream`);
        es.onopen = () => { stopPolling(); };
        es.onmessage = (ev: MessageEvent<string>) => {
          if (cancelled) return;
          try {
            const data = JSON.parse(ev.data) as unknown;
            applyTrips(data);
          } catch {
            // Malformed event — ignore; next message will overwrite.
          }
        };
        es.onerror = () => {
          // readyState CLOSED = permanent failure (e.g., 404 on /stream).
          // CONNECTING = browser is retrying; leave it alone.
          if (es?.readyState === EventSource.CLOSED) {
            startPolling();
          }
        };
      } catch {
        startPolling();
      }
    } else {
      startPolling();
    }

    return () => {
      cancelled = true;
      controller.abort();
      stopPolling();
      if (es) {
        es.close();
        es = null;
      }
    };
  }, [enabled, pollInterval]);

  return trips;
}
