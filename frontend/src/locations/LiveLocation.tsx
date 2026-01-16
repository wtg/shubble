import {
  useState,
  useEffect,
} from 'react';

import LiveLocationMapKit from './components/LiveLocationMapKit';
import Schedule from '../schedule/Schedule';
import "./styles/LiveLocation.css";
import routeData from '../shared/routes.json';
import type { ShuttleRouteData } from '../types/route';
import type { VehicleETAs } from '../types/vehicleLocation';
import aggregatedSchedule from '../shared/aggregated_schedule.json';

export default function LiveLocation() {
  const [filteredRouteData, setFilteredRouteData] = useState<ShuttleRouteData | null>(null);

  //New selection state for the schedule
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

  // Stop times state (historical, predicted, and future)
  const [stopTimes, setStopTimes] = useState<VehicleETAs>({});

  // Current stops state: maps stop_name to array of vehicle IDs at that stop
  const [vehiclesAtStops, setVehiclesAtStops] = useState<Record<string, string[]>>({});

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
      <LiveLocationMapKit
        routeData={filteredRouteData}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        onStopTimesUpdate={setStopTimes}
        onVehiclesAtStopsUpdate={setVehiclesAtStops}
      />
      <div className="schedule-table">
        <Schedule
          selectedRoute={selectedRoute}
          setSelectedRoute={setSelectedRoute}
          stopTimes={stopTimes}
          vehiclesAtStops={vehiclesAtStops}
        />
      </div>
    </div>
  );
}
