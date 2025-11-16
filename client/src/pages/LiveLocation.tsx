import {
  useState,
  useEffect,
} from 'react';

import MapKitMap from '../components/MapKitMap';
import Schedule from '../components/Schedule';
import "../styles/LiveLocation.css";
import routeData from '../data/routes.json';
import type { VehicleInformationMap } from '../ts/types/vehicleLocation';
import type { ShuttleRouteData } from '../ts/types/route';
import aggregatedSchedule from '../data/aggregated_schedule.json';

export default function LiveLocation() {

  const [location, setLocation] = useState<VehicleInformationMap | null>(null);
  const [filteredRouteData, setFilteredRouteData] = useState<ShuttleRouteData | null>(null);

  //New selection state for the schedule
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

  const now = new Date();
  const [selectedDay, setSelectedDay] = useState(now.getDay());

  // Filter routeData to only include routes present in aggregatedSchedule
  useEffect(() => {
    // TODO: figure out how to make this type correct...
    setFilteredRouteData(
      Object.fromEntries(
       Object.entries(routeData).filter(([routeName]) => routeName in aggregatedSchedule[selectedDay])
      ) as unknown as ShuttleRouteData
    );
  }, [selectedDay]);

  // Fetch location data on component mount and set up polling
  useEffect(() => {
    const pollLocation = async () => {
      try {
        const response = await fetch('/api/locations');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setLocation(data);
      } catch (error) {
        console.error('Error fetching location:', error);
      }
    }

    pollLocation();

    // refresh location every 5 seconds
    const refreshLocation = setInterval(pollLocation, 5000);

    return () => {
      clearInterval(refreshLocation);
    }

  }, []);

  return (
    <div className="live-location-div">
      <MapKitMap
        routeData={filteredRouteData}
        vehicles={location}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
      />
      <div className="schedule-table">
        <Schedule
          selectedRoute={selectedRoute}
          setSelectedRoute={setSelectedRoute}
          selectedDay={selectedDay}
          setSelectedDay={setSelectedDay}
        />
      </div>
    </div>
  );
}
