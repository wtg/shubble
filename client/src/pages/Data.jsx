import { useState } from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';

export default function Data() {

    // made up shuttle data
    const shuttleList = [
	{id: 0,
         lat: 0,
         lng: 0,
         timestamp: "Timestamp1",
         speed: 0,
         heading: "N",
	 address: "zero"},
	{id: 1,
         lat: 1,
         lng: 1,
         timestamp: "Timestamp2",
         speed: 1,
         heading: "E",
	 address: "one"},
	{id: 2,
         lat: 2,
         lng: 2,
         timestamp: "Timestamp3",
         speed: 2,
         heading: "S",
	 address: "two"}			  
    ];

    const [shuttle, setShuttle] = useState(shuttleList[0]);
    const handleShuttleChange = (event) => {
	setShuttle(shuttleList[parseInt(event.target.value)]);
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
			<select value={shuttle.id} onChange={handleShuttleChange}>
			    {shuttleList.map(shuttle => (
				<option key={shuttle.id} value={shuttle.id}>
				    {shuttle.address}
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
				    Location
				</th>
				<th>
				    Speed
				</th>
			    </tr>
			</thead>
			<tbody>
			    <tr>
				<td>{shuttle.timestamp}</td>
				<td>{shuttle.lat}, {shuttle.lng}</td>
				<td>{shuttle.speed} mph</td>
			    </tr>
			</tbody>
		    </table>
		</div>
		<MapKitMap vehicles={ location } />
	    </div>
	</>
    );
}
