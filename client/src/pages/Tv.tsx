import {
  useState,
  useEffect,
} from 'react';

import MapKitMap from '../components/MapKitMap';
import "../styles/LiveLocation.css";
import routeData from '../data/routes.json';
import type { VehicleInformationMap } from '../ts/types/vehicleLocation';
import type { ShuttleRouteData } from '../ts/types/route';
import aggregatedSchedule from '../data/aggregated_schedule.json';

export default function Tv() {

  const [location, setLocation] = useState<VehicleInformationMap | null>(null);
  const [filteredRouteData, setFilteredRouteData] = useState<ShuttleRouteData | null>(null);

  // Filter routeData to only include routes present in aggregatedSchedule
  useEffect(() => {
    setFilteredRouteData(
      Object.fromEntries(
        Object.entries(routeData).filter(([routeName]) => aggregatedSchedule.some(daySchedule => routeName in daySchedule))
      ) as unknown as ShuttleRouteData
    );
  }, []);

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
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <MapKitMap
        routeData={filteredRouteData}
        vehicles={location}
        isFullscreen={true}
      />
    </div>
  );
}
