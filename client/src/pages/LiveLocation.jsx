import React, {
    useState,
    useEffect,
} from 'react';

import MapKitMap from '../components/MapKitMap';
import Schedule from '../components/Schedule';
import "../styles/LiveLocation.css";
import routeData from '../data/routes.json';
import { aggregatedSchedule } from '../data/parseSchedule';

export default function LiveLocation() {

    const [location, setLocation] = useState(null);
    const [filteredRouteData, setFilteredRouteData] = useState({});

    // Filter routeData to only include routes present in aggregatedSchedule
    useEffect(() => {
        setFilteredRouteData(
            Object.fromEntries(
                Object.entries(routeData).filter(([routeName]) => aggregatedSchedule.some(daySchedule => routeName in daySchedule))
            )
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
	<div className = "live-location-div">
	    <MapKitMap routeData={ filteredRouteData } vehicles={ location } />
	    <div className = "schedule-table">
		<Schedule />
	    </div>
	</div>
    );
}
