import { useState, useEffect, useMemo } from 'react';

import LiveLocationMapKit from './components/LiveLocationMapKit';
import Schedule from '../schedule/Schedule';
import "./styles/LiveLocation.css";
import routeData from '../shared/routes.json';
import type { ShuttleRouteData } from '../types/route';
import aggregatedSchedule from '../shared/aggregated_schedule.json';
import type { AggregatedScheduleType } from '../types/schedule';
import config from '../utils/config';
import { useTrips, deriveStopEtasFromTrips } from '../hooks/useTrips';

// PERF: pure function of the static JSON imports — compute once at module
// load, not on every render. The aggregatedSchedule is baked into the bundle
// at build time, so nothing about this ever changes at runtime.
const _rawAggregatedSchedule = aggregatedSchedule as unknown as AggregatedScheduleType;
const _filteredRouteData = Object.fromEntries(
  Object.entries(routeData).filter(
    ([routeName]) => _rawAggregatedSchedule.some(daySchedule => routeName in daySchedule)
  )
) as unknown as ShuttleRouteData;

export default function LiveLocation() {
  const filteredRouteData = _filteredRouteData;
  const [selectedRoute, setSelectedRoute] = useState<string | null>(
    () => localStorage.getItem('shubble-route')
  );
  useEffect(() => {
    if (selectedRoute) localStorage.setItem('shubble-route', selectedRoute);
  }, [selectedRoute]);

  // Single source of truth: /api/trips. The map stop-marker tooltips and
  // the Schedule timeline both consume derived views of this data. The
  // older /api/etas endpoint was deleted in favor of this consolidation.
  const trips = useTrips(!config.staticETAs);
  const { stopETAs, stopETADetails } = useMemo(
    () => deriveStopEtasFromTrips(trips),
    [trips]
  );

  return (
    <div className="live-location-div">
      <LiveLocationMapKit
        routeData={filteredRouteData}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        stopETAs={stopETAs}
        stopETADetails={stopETADetails}
      />
      <div className="schedule-table">
        <Schedule
          selectedRoute={selectedRoute}
          setSelectedRoute={setSelectedRoute}
          stopETAs={stopETAs}
          stopETADetails={stopETADetails}
          trips={trips}
        />
      </div>
    </div>
  );
}
