import { useState, useEffect } from 'react';

import LiveLocationMapKit from './components/LiveLocationMapKit';
import Schedule from '../schedule/Schedule';
import "./styles/LiveLocation.css";
import routeData from '../shared/routes.json';
import type { ShuttleRouteData } from '../types/route';
import aggregatedSchedule from '../shared/aggregated_schedule.json';
import type { AggregatedScheduleType } from '../types/schedule';
import config from '../utils/config';
import { useStopETAs } from '../hooks/useStopETAs';

export default function LiveLocation() {
  // Filter routeData to only include routes present in aggregatedSchedule
  // TODO: figure out how to make this type correct...
  const rawAggregatedSchedule = aggregatedSchedule as unknown as AggregatedScheduleType;
  const filteredRouteData = Object.fromEntries(
      Object.entries(routeData).filter(([routeName]) => rawAggregatedSchedule.some(daySchedule => routeName in daySchedule))
    ) as unknown as ShuttleRouteData;
  const [selectedRoute, setSelectedRoute] = useState<string | null>(
    () => localStorage.getItem('shubble-route')
  );
  useEffect(() => {
    if (selectedRoute) localStorage.setItem('shubble-route', selectedRoute);
  }, [selectedRoute]);

  // Shared ETA hook — feeds both the map and schedule to avoid duplicate API calls
  const { stopETAs, stopETADetails } = useStopETAs(!config.staticETAs);

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
        />
      </div>
    </div>
  );
}
