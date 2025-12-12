import {
  useState,
  useEffect,
} from 'react';

import MapKitMap from '../components/MapKitMap';
import Schedule from '../components/Schedule';
import "../styles/LiveLocation.css";
import routeData from '../data/routes.json';
import type { VehicleInformationMap } from '../ts/types/vehicleLocation';
import DataAgeIndicator from '../components/DataAgeIndicator';
import type { ShuttleRouteData } from '../ts/types/route';
import aggregatedSchedule from '../data/aggregated_schedule.json';

export default function LiveLocation() {

  const [location, setLocation] = useState<VehicleInformationMap | null>(null);
  const [filteredRouteData, setFilteredRouteData] = useState<ShuttleRouteData | null>(null);
  const [dataAge, setDataAge] = useState<number | null>(null);

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

  // Fetch location data on component mount and set up polling synced with server
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let isMounted = true;

    const pollLocation = async () => {
      try {
        const response = await fetch('/api/locations');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data: VehicleInformationMap = await response.json();
        setLocation(data);

        // Read data age from response header
        const dataAgeHeader = response.headers.get('X-Data-Age-Seconds');
        setDataAge(dataAgeHeader ? parseFloat(dataAgeHeader) : null);

        // Calculate optimal delay for next fetch to sync with server's 5-second update cycle
        // If data is N seconds old, next update should arrive in (5 - (N % 5)) seconds
        // Add a small buffer (200ms) to ensure the server has updated
        const dataAge = dataAgeHeader ? parseFloat(dataAgeHeader) : 0;
        const nextUpdateIn = Math.max(1000, (5 - (dataAge % 5)) * 1000 + 200);

        if (isMounted) {
          timeoutId = setTimeout(pollLocation, nextUpdateIn);
        }
      } catch (error) {
        console.error('Error fetching location:', error);
        // On error, retry after 5 seconds
        if (isMounted) {
          timeoutId = setTimeout(pollLocation, 5000);
        }
      }
    }

    pollLocation();

    return () => {
      isMounted = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    }

  }, []);

  return (
    <div className="live-location-div">
      <DataAgeIndicator dataAge={dataAge} />
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
        />
      </div>
    </div>
  );
}
