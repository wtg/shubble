import {
  useState,
  useEffect,
} from 'react';

import MapKitMap from './components/MapKitMap';
import Schedule from '../schedule/Schedule';
import "./styles/LiveLocation.css";
import routeData from '../shared/routes.json';
import type { ShuttleRouteData } from '../types/route';
import aggregatedSchedule from '../shared/aggregated_schedule.json';

export default function LiveLocation() {
  const [filteredRouteData, setFilteredRouteData] = useState<ShuttleRouteData | null>(null);

  //New selection state for the schedule
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

  // Filter routeData to only include routes present in aggregatedSchedule
  useEffect(() => {
    // TODO: figure out how to make this type correct...
    setFilteredRouteData(
      Object.fromEntries(
        Object.entries(routeData).filter(([routeName]) => aggregatedSchedule.some(daySchedule => routeName in daySchedule))
      ) as unknown as ShuttleRouteData
    );
  }, []);

  return (
    <div className="live-location-div">
      <MapKitMap
        routeData={filteredRouteData}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
      />
      <div className="schedule-table">
        <Schedule
          selectedRoute={selectedRoute}
          setSelectedRoute={setSelectedRoute}
        />
      </div>
    </div>
  );
}
