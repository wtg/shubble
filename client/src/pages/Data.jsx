import React, {
    useState,
    useEffect,
} from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';

export default function Data() {

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

    }, []);

    function formatTimestamp(shuttleLocation) {
	if ("timestamp" in shuttleLocation) {
	    let tStamp = shuttleLocation.timestamp;
	    let hours = parseInt(tStamp.substring(11, 13));
	    let minutes = tStamp.substring(14, 16);
	    let seconds = tStamp.substring(17, 19);

	    if (hours > 12) {
		hours -= 12;
		return hours + ":" + minutes + ":" + seconds + "PM";
	    }
	    return hours + ":" + minutes + ":" + seconds + "AM";
	}
	return "No timestamp given"
    }

    const [shuttleID, setShuttleID] = useState(null);

    const handleShuttleChange = (event) => {
	setShuttleID(event.target.value === '' ? null : event.target.value);
    }
    
    return (
	<>
	    <div className = "header">
		<h1>Shubble Data</h1>
		<p>Here you can view location history by shuttle.</p>
	    </div>

	    <div className = "table-map-sidebyside">
		<div className = "left-screen">
		{location ? (
		<div>
		    <p className = "dropdown-p-style">
			Shuttle: <select value={shuttleID || ''} onChange={handleShuttleChange} className = "dropdown-style">
				     <option value="">Select a shuttle</option>
			    {Object.keys(location).map(shuttleID => (
				<option key={shuttleID} value={shuttleID}>
				    {shuttleID}
				</option>
			    ))}
			</select>
		    </p>

		    {shuttleID ? (

		    <table className = "data-table">
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
			    <tr>
				<td>{formatTimestamp(location[shuttleID])}</td>
				<td>{location[shuttleID].lat}, {location[shuttleID].lng}</td>
				<td>{location[shuttleID].speed} mph</td>
			    </tr>
			</tbody>
		    </table>
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
