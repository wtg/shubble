import React, {
    useState,
    useEffect,
} from 'react';
import MapKitMap from '../components/MapKitMap';

export default function LiveLocation() {

    const [location, setLocation] = useState(null);

    useEffect(() => {
        const pollLocation = async () => {
            try {
                const response = await fetch('/api/locations');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                console.log(data);
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
        <MapKitMap vehicles={ location }/>
    )
}
