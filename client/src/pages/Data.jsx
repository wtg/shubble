import { useState, useEffect} from 'react';
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

    // made up shuttle data
    const shuttleList = {
	"shuttleID1": {
	    lat: 0,
            lng: 0,
            timestamp: "2020-07-03T02:11:51Z",
            speed: 0,
            heading: "N",
	    address: "zero"
	},
	"shuttleID2": {
            lat: 1,
            lng: 1,
            timestamp: "2020-07-03T12:11:51Z",
            speed: 1,
            heading: "E",
	    address: "one"
	},
	"shuttleID3": {
            lat: 2,
            lng: 2,
            timestamp: "2020-07-03T22:11:51Z",
            speed: 2,
            heading: "S",
	    address: "two"
	}
    };


    function formatTimestamp(tStamp) {
	let hours = parseInt(tStamp.substring(11, 13));
	let minutes = parseInt(tStamp.substring(14, 16));
	let seconds = parseInt(tStamp.substring(17, 19));

	if (hours > 12) {
	    hours -= 12;
	    return hours + ":" + minutes + ":" + seconds + "PM";
	}
	return hours + ":" + minutes + ":" + seconds + "AM";
    }

    const [shuttleID, setShuttleID] = useState("shuttleID1");
    const handleShuttleChange = (event) => {
	setShuttleID(event.target.value);
    }
    
    return (
	<>
	    <div className = "header">
		<h1>Shubble Data</h1>
		<p>Here you can view location history by shuttle.</p>
	    </div>

	    <div className = "table-map-sidebyside">
		<div>
		    <p className = "dropdown-p-style">
			Shuttle: <select value={shuttleID} onChange={handleShuttleChange} className = "dropdown-style">
			    {Object.keys(location).map(shuttleID => (
				<option key={shuttleID} value={shuttleID}>
				    {shuttleID}
				</option>
			    ))}
			</select>
		    </p>

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
				<td>{formatTimestamp(location[shuttleID].timestamp)}</td>
				<td>{location[shuttleID].lat}, {location[shuttleID].lng}</td>
				<td>{location[shuttleID].speed} mph</td>
			    </tr>
			</tbody>
		    </table>
		</div>
		<MapKitMap vehicles={ location } />
	    </div>
	</>
    );
}
