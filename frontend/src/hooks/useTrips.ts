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
 * Fetches per-trip ETAs from the backend. Each trip is independently
 * tracked so concurrent shuttles on the same route don't fight over
 * displayed data.
 */
export function useTrips(enabled = true, pollInterval = 15000) {
  const [trips, setTrips] = useState<Trip[]>([]);

  useEffect(() => {
    if (!enabled) return;

    const fetchTrips = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/trips`, {
          cache: 'no-store',
        });
        if (!response.ok) return;
        const data = (await response.json()) as Trip[];
        if (Array.isArray(data)) setTrips(data);
      } catch (error) {
        console.error('Failed to fetch trips:', error);
      }
    };

    fetchTrips();
    const interval = setInterval(fetchTrips, pollInterval);
    return () => clearInterval(interval);
  }, [enabled, pollInterval]);

  return trips;
}

/**
 * Formats a trip's departure time to a local HH:MM AM/PM string for
 * matching against the aggregated_schedule.json entries.
 */
export function formatDepartureTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

/**
 * Groups trips by route for easy lookup.
 */
export function tripsByRoute(trips: Trip[]): Record<string, Trip[]> {
  const out: Record<string, Trip[]> = {};
  for (const t of trips) {
    if (!out[t.route]) out[t.route] = [];
    out[t.route].push(t);
  }
  for (const r of Object.keys(out)) {
    out[r].sort((a, b) => a.departure_time.localeCompare(b.departure_time));
  }
  return out;
}
