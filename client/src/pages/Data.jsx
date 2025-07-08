import React, {
    useState,
    useEffect,
} from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';

export default function Data() {

    const [location, setLocation] = useState(null);

    const [isLoading, setIsLoading] = useState(false);

    const fetchLocation = async () => {
	setIsLoading(true);
	try {
            const response = await fetch('/api/today');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            console.log(data);
            setLocation(data);
        } catch (error) {
            console.error('Error fetching location:', error);
        }
	finally {
	    setIsLoading(false);
	}
    }
    
    useEffect(() => {
        const pollLocation = async () => {
            fetchLocation();
            pollLocation();
	}
    }, []);

    const handleReload = async () => {
	fetchLocation();
    }

    function formatTimestamp(tStamp) {
	if (tStamp === null) {
	    return "timestamp was set to null";
	}
	let hours = parseInt(tStamp.substring(11, 13));
	let minutes = tStamp.substring(14, 16);
	let seconds = tStamp.substring(17, 19);
	let formattedTimestamp =  ":" + minutes + ":" + seconds;
	if (hours > 12) {
	    hours -= 12;
	    formattedTimestamp = formattedTimestamp + "PM";
	}
	else {
	    formattedTimestamp = formattedTimestamp + "AM";
	}
	formattedTimestamp = hours + formattedTimestamp;
	    return formattedTimestamp;
    }

    const [selectedShuttleID, setSelectedShuttleID] = useState(null);

    const handleShuttleChange = (event) => {
	setSelectedShuttleID(event.target.value === '' ? null : event.target.value);
    }

    return (
	<>
	    <div className = "header">
		<div className = "flex-header-reload">
		    <h1>Shubble Data</h1>
		    <button onClick={handleReload} className = "reload-button">
			<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M2 12a9 9 0 0 0 9 9c2.39 0 4.68-.94 6.4-2.6l-1.5-1.5A6.7 6.7 0 0 1 11 19c-6.24 0-9.36-7.54-4.95-11.95S18 5.77 18 12h-3l4 4h.1l3.9-4h-3a9 9 0 0 0-18 0"/></svg>
		    </button>
		</div>
		<p>Here you can view location history by shuttle.</p>
	    </div>

	    <div className = "table-map-sidebyside">
		<div className = "left-screen">
		{location ? (
		<div>
		    <p className = "dropdown-p-style">
			Shuttle: <select value={selectedShuttleID || ''} onChange={handleShuttleChange} className = "dropdown-style">
				     <option value="">Select a shuttle</option>
			    {Object.keys(location).map(selectedShuttleID => (
				<option key={selectedShuttleID} value={selectedShuttleID}>
				    {selectedShuttleID}
				</option>
			    ))}
			</select>
		    </p>

		    {selectedShuttleID ? (

			<div className = "location-table-overflow-scroll">
			    <table>
				<thead>
				    <tr>
					<th>
					    Timestamp
					</th>
					<th>
					    Latitude, Longitude
					</th>
					<th>
					    Speed
					</th>
				    </tr>
				</thead>
				<tbody>
				    {location[selectedShuttleID].reverse().map((shuttleLocation, index) => (
					<tr key={index}>
					    <td>
						{formatTimestamp(shuttleLocation.timestamp)}
					    </td>
					    <td>
						{shuttleLocation.latitude.toFixed(3) + ", " + shuttleLocation.longitude.toFixed(3)}
					    </td>
					    <td>
						{shuttleLocation.speed_mph + " mph"}
					    </td>
					</tr>
				    ))}
				    
				</tbody>
			    </table>
			</div>
		    ) : (
			<p>No shuttle selected</p>
		    )}
		</div>
		) : (
		    <p>No locations found</p>
		)}
		</div>
		<MapKitMap vehicles={ location } />
	    </div>
	</>
    );
}
