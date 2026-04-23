import { useState, useCallback, useEffect, useRef } from 'react';

import LiveLocationMapKit from './components/LiveLocationMapKit';
import Schedule from '../schedule/Schedule';
import "./styles/LiveLocation.css";
import routeData from '../shared/routes.json';
import type { ShuttleRouteData } from '../types/route';
import aggregatedSchedule from '../shared/aggregated_schedule.json';
import type { AggregatedScheduleType } from '../types/schedule';
import { getInactiveStops } from '../utils/stopSchedule';

export default function LiveLocation() {
  // Filter routeData to only include routes present in aggregatedSchedule
  // TODO: figure out how to make this type correct...
  const rawAggregatedSchedule = aggregatedSchedule as unknown as AggregatedScheduleType;
  const filteredRouteData = Object.fromEntries(
      Object.entries(routeData).filter(([routeName]) => rawAggregatedSchedule.some(daySchedule => routeName in daySchedule))
    ) as unknown as ShuttleRouteData;
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

  // Track schedule panel scroll position for potential time-based interactions
  const [scheduleScrollY, setScheduleScrollY] = useState(0);
  const scheduleRef = useRef<HTMLDivElement>(null);

  const handleScheduleScroll = useCallback(() => {
    if (scheduleRef.current) {
      setScheduleScrollY(scheduleRef.current.scrollTop);
    }
  }, []);

  // Recompute inactive stops every minute so the map reflects the current time
  const [inactiveStops, setInactiveStops] = useState<Set<string>>(
    () => getInactiveStops(filteredRouteData, new Date())
  );

  useEffect(() => {
    const tick = () => setInactiveStops(getInactiveStops(filteredRouteData, new Date()));
    // Refresh at the top of every minute
    const msUntilNextMinute = (60 - new Date().getSeconds()) * 1000;
    const timeout = setTimeout(() => {
      tick();
      const interval = setInterval(tick, 60_000);
      return () => clearInterval(interval);
    }, msUntilNextMinute);
    return () => clearTimeout(timeout);
  // filteredRouteData is stable (derived from imported JSON), no need to list it as dep
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="live-location-div">
      <LiveLocationMapKit
        routeData={filteredRouteData}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        inactiveStops={inactiveStops}
        scheduleScrollY={scheduleScrollY}
      />
      <div
        className="schedule-table"
        ref={scheduleRef}
        onScroll={handleScheduleScroll}
      >
        <Schedule
          selectedRoute={selectedRoute}
          setSelectedRoute={setSelectedRoute}
        />
      </div>
    </div>
  );
}
