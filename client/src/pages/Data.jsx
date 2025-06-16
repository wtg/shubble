import { useState } from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';

export default function Data() {

    // made up shuttle data
    const shuttleList = [
	{id: 0,
         lat: 0,
         lng: 0,
         timestamp: 0,
         speed: 0,
         heading: "N",
	 address: "zero"},
	{id: 1,
         lat: 1,
         lng: 1,
         timestamp: 1,
         speed: 1,
         heading: "N",
	 address: "one"},
	{id: 2,
         lat: 2,
         lng: 2,
         timestamp: 2,
         speed: 2,
         heading: "N",
	 address: "two"}			  
    ];

    const [shuttle, setShuttle] = useState(shuttleList[0]);
    const handleShuttleChange = (e) => {
	setShuttle(parseInt(event.target.value));
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
		    <select value={shuttle} onChange={handleShuttleChange}>
			{
			    shuttleList.map((shuttle, index) =>
				<option key={index} vlaue={index}>
				    {shuttle.id}
				</option>
			    )
			}	
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
			{shuttleList.map((shuttle, index) => (
			    <tr key={index}>
				<td>{shuttle.timestamp}</td>
				<td>{shuttle.lat}, {shuttle.lng}</td>
				<td>{shuttle.speed} mph</td>
			    </tr>
			))}
		    </tbody>
		</table>
		</div>
		<MapKitMap vehicles={ location } />
	    </div>
	</>
    );
}
