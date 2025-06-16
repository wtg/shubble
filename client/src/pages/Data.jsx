import { useState } from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';

export default function Data() {

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
		    <p>
			Shuttle:
			<select value={shuttleID} onChange={handleShuttleChange}>
			    {Object.keys(shuttleList).map(shuttleID => (
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
				<td>{((shuttleList[shuttleID].timestamp.substring(11, 12) === "0") ? shuttleList[shuttleID].timestamp.substring(12, 19) : shuttleList[shuttleID].timestamp.substring(11, 19))}</td>
				<td>{shuttleList[shuttleID].lat}, {shuttleList[shuttleID].lng}</td>
				<td>{shuttleList[shuttleID].speed} mph</td>
			    </tr>
			</tbody>
		    </table>
		</div>
		<MapKitMap vehicles={ location } />
	    </div>
	</>
    );
}
