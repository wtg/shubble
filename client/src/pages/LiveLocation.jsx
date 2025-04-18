import React, {
    useState,
    useEffect,
} from 'react';

export default function LiveLocation() {

    /*
    // https://developers.samsara.com/reference/getvehiclestatsfeed
    response body:
    {
        data: [
            {
                gps: {
                    headingDegrees,
                    latitude,
                    longitude,
                    speedMilesPerHour,
                    time,
                }
            }
        ]
    }
    */

    const [location, setLocation] = useState(null);

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
        <div>
            <h1>Live Location</h1>
            {location && location.length > 0 ? (
                <div>
                    <h2>Current Location:</h2>
                    <p>Latitude: {location.data[0].gps.latitude}</p>
                    <p>Longitude: {location.data[0].gps.longitude}</p>
                    <p>Speed: {location.data[0].gps.speedMilesPerHour} mph</p>
                    <p>Heading: {location.data[0].gps.headingDegrees} degrees</p>
                </div>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    )
}
