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
 * Fetches per-trip ETAs from the backend. Each trip is independently
 * tracked so concurrent shuttles on the same route don't fight over
 * displayed data.
 */
export function useTrips(enabled = true, pollInterval = 15000) {
  const [trips, setTrips] = useState<Trip[]>([]);

  useEffect(() => {
    if (!enabled) return;

    const controller = new AbortController();
    let cancelled = false;

    const fetchTrips = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/trips`, {
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!response.ok || cancelled) return;
        const data = (await response.json()) as Trip[];
        if (Array.isArray(data) && !cancelled) setTrips(data);
      } catch (error) {
        // Ignore AbortError (component unmount) and network failures
        // during polling — the next interval tick will retry.
        if ((error as Error).name === 'AbortError') return;
        // Silent: network issues are common and handled by retry.
      }
    };

    fetchTrips();
    const interval = setInterval(fetchTrips, pollInterval);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(interval);
    };
  }, [enabled, pollInterval]);

  return trips;
}
