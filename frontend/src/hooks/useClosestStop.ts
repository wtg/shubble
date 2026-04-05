import { useState, useEffect } from 'react';
import { buildAllStops, findClosestStop } from '../types/ClosestStop';
import type { Stop } from '../types/schedule';
import type { ShuttleRouteData } from '../types/route';
import type { AggregatedScheduleType } from '../types/schedule';
import rawRouteData from '../shared/routes.json';
import rawAggregatedSchedule from '../shared/aggregated_schedule.json';

const routeData = rawRouteData as unknown as ShuttleRouteData;
const aggregatedSchedule = rawAggregatedSchedule as unknown as AggregatedScheduleType;

/**
 * Hook that detects the user's closest shuttle stop via geolocation.
 * Requests permission once, then returns the nearest stop on the given route.
 *
 * @param routeFilter - Only consider stops on this route (null = all routes)
 */
export function useClosestStop(routeFilter?: string | null) {
  const [closestStop, setClosestStop] = useState<Stop | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);

  useEffect(() => {
    if (!navigator.geolocation) return;

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const userPoint = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        const dayIndex = new Date().getDay();
        const allStops = buildAllStops(routeData, aggregatedSchedule, dayIndex);
        const filtered = routeFilter
          ? allStops.filter(s => s.route === routeFilter)
          : allStops;
        const closest = findClosestStop(userPoint, filtered);
        if (closest) setClosestStop(closest);
      },
      () => setPermissionDenied(true),
      { enableHighAccuracy: false, timeout: 5000 }
    );
  }, [routeFilter]);

  return { closestStop, permissionDenied };
}
