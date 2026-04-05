import { useState, useEffect } from 'react';
import config from '../utils/config';
import { devNow } from '../utils/devTime';
import type { StopETAMap } from '../types/vehicleLocation';

// Formatted ETA strings keyed by stop key
export type StopETAs = Record<string, string>;

// Full ETA data keyed by stop key (includes route, vehicle_id, last_arrival)
export type StopETADetails = Record<string, {
  eta: string;           // formatted time string
  etaISO: string;        // original ISO string
  route: string;
  vehicleId: string;
  lastArrival?: string;  // formatted time string
}>;

const TIME_FORMAT: Intl.DateTimeFormatOptions = {
  hour: 'numeric',
  minute: '2-digit',
};

/**
 * Shared hook for fetching per-stop ETAs from the backend.
 * Used by both Schedule and MapKitCanvas to avoid duplicate API calls.
 *
 * @param enabled - Whether to poll (false disables polling, e.g. for static ETAs)
 * @param pollInterval - Polling interval in ms (default 30000)
 */
export function useStopETAs(enabled = true, pollInterval = 30000) {
  const [stopETAs, setStopETAs] = useState<StopETAs>({});
  const [stopETADetails, setStopETADetails] = useState<StopETADetails>({});

  useEffect(() => {
    if (!enabled) return;

    const fetchETAs = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/etas`, { cache: 'no-store' });
        if (!response.ok) return;

        const data = (await response.json()) as StopETAMap;
        const now = devNow();

        const newStopETAs: StopETAs = {};
        const newDetails: StopETADetails = {};

        for (const [stopKey, stopInfo] of Object.entries(data)) {
          let lastArrivalFormatted: string | undefined;
          if (stopInfo.last_arrival) {
            lastArrivalFormatted = new Date(stopInfo.last_arrival).toLocaleTimeString(undefined, TIME_FORMAT);
          }

          // Entry has a future ETA
          if (stopInfo.eta) {
            const etaDate = new Date(stopInfo.eta);
            if (etaDate <= now) {
              // Past ETA — still include if there's a last_arrival
              if (lastArrivalFormatted) {
                newDetails[stopKey] = {
                  eta: '',
                  etaISO: '',
                  route: stopInfo.route || '',
                  vehicleId: stopInfo.vehicle_id || '',
                  lastArrival: lastArrivalFormatted,
                };
              }
              continue;
            }

            const formattedTime = etaDate.toLocaleTimeString(undefined, TIME_FORMAT);
            newStopETAs[stopKey] = formattedTime;

            newDetails[stopKey] = {
              eta: formattedTime,
              etaISO: stopInfo.eta,
              route: stopInfo.route,
              vehicleId: stopInfo.vehicle_id,
              lastArrival: lastArrivalFormatted,
            };
          } else if (lastArrivalFormatted) {
            // No future ETA but has last_arrival (stop already passed)
            newDetails[stopKey] = {
              eta: '',
              etaISO: '',
              route: stopInfo.route || '',
              vehicleId: stopInfo.vehicle_id || '',
              lastArrival: lastArrivalFormatted,
            };
          }
        }

        setStopETAs(newStopETAs);
        setStopETADetails(newDetails);
      } catch (error) {
        console.error('Failed to fetch ETAs:', error);
      }
    };

    fetchETAs();
    const interval = setInterval(fetchETAs, pollInterval);
    return () => clearInterval(interval);
  }, [enabled, pollInterval]);

  return { stopETAs, stopETADetails };
}
