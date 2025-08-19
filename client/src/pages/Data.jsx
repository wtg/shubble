import React, {
    useState,
    useEffect,
} from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';
import DataBoard from '../components/DataBoard';
import ShuttleRow from '../components/ShuttleRow';

export default function Data() {

    const [shuttleData, setShuttleData] = useState(null);

    const [selectedShuttleID, setSelectedShuttleID] = useState(null);
    
    const fetchShuttleData = async () => {
	try {
            const response = await fetch('/api/today');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            setShuttleData(data);
        } catch (error) {
            console.error('Error fetching shuttleData:', error);
        }
    }

    useEffect(() => {
        fetchShuttleData();
    }, []);

    useEffect(() => {
	if (shuttleData != null) {
	    if (!(selectedShuttleID in shuttleData)) {
		setSelectedShuttleID(Object.keys(shuttleData)[0]);
	    }
	}
    }, [shuttleData]);

    const handleShuttleChange = (event) => {
	setSelectedShuttleID(event.target.value);
    }

    function formatTimestamp(tStamp) {
	if (tStamp === null) {
	    return "timestamp was set to null";
	}
	var hours = tStamp.getHours();
	var minutes = tStamp.getMinutes();
	var amOrPm = " AM";
	if (hours > 12) {
	    hours -= 12;
	    amOrPm = " PM";
	}
	if (minutes < 10) {
	    minutes = "0" + minutes;
	}
	return hours + ":" + minutes + amOrPm;
    }

    function addTimeUnits(quantity, unit) {
	if (quantity == 1) {
	    return quantity + " " + unit;
	}
	return quantity + " " + unit + "s";
    }

    function formatTime(minutes) {
	var days = addTimeUnits(Math.floor(minutes/(60 * 24)), "day");
	var hours = addTimeUnits(Math.floor((minutes % (60 * 24))/(60)), "hour");
	var mins = addTimeUnits((minutes % 60), "minute");
	return (minutes > 60 * 24 ? days + ", " : "") + (minutes > 60 ? hours + ", " : "") + mins;
    }

    function formatEntryExit(entry, exit) {
	if (entry == null) {
	    return "Shuttle never entered GeoFence";
	}
	var exitStr = "NOW";
	if (exit != null) {
	    exitStr = new Date(exit).toLocaleTimeString();
	}
	return new Date(entry).toLocaleTimeString() + "-" + exitStr;
    }

    function formatLoopsBreaks(loopBreakList) {
	if (!loopBreakList) {
	    return "No data given"
	}
	var formattedList = [new Array(loopBreakList.length), new Array(loopBreakList.length)];
	var totalTime = 0;
	loopBreakList.forEach((loopOrBreak, l) => {
	    const dStart = new Date(loopOrBreak.start);
	    if (loopOrBreak.end == null) {
		formattedList[0][l] = "IN PROGRESS";
		formattedList[1][l] = formatTimestamp(dStart) + " - NOW";
		const now = new Date();
		totalTime += Math.round((now - dStart)/(1000 * 60)); // convert milliseconds to minutes
	    }
	    else {
		const dEnd = new Date(loopOrBreak.end);
		var durationMinutes = Math.round((dEnd-dStart)/(1000 * 60)); // convert milliseconds to minutes
		totalTime += durationMinutes; 
		formattedList[0][l] = formatTime(durationMinutes);
		formattedList[1][l] = formatTimestamp(dStart) + " - " + formatTimestamp(dEnd);
	    }
	})
	return [formattedList, formatTime(totalTime)];
    }
    
    return (
	<>
	    <div className="page-container">
		<div className="sidebar">
		    <table className="sidebar-table">
			<thead>
			    <tr>
				<th colSpan={3}>{new Date().toLocaleDateString('en-US', {
				    weekday: 'short',
				    month: 'long',
				    day: 'numeric'
				})}</th>
			    </tr>
			</thead>
			{shuttleData ? (
			    <tbody>
				{Object.keys(shuttleData).map(vehicleId => (
				    <tr key={vehicleId}>
					<ShuttleRow
					    shuttleId={vehicleId}
					    isActive={false}
					    isAm={false}
					/>
				    </tr>
				))}
			    </tbody>
			) : (
			    <p>No shuttle data given</p>
			)}
		    </table>
		</div>

		{shuttleData ? (
		    <div>
			{shuttleData[selectedShuttleID] ? (
			    <div className="main-content">
				<DataBoard
				    title="Summary"
				    dataToDisplay={[[formatEntryExit(shuttleData[selectedShuttleID].entry, shuttleData[selectedShuttleID].exit), shuttleData[selectedShuttleID]["loops"].length + " loops", formatLoopsBreaks(shuttleData[selectedShuttleID]["breaks"])[1] + " of break time"]]}
				/>
				<DataBoard
				    title="Loops"
				    dataToDisplay={formatLoopsBreaks(shuttleData[selectedShuttleID]["loops"])[0]}
				/>
				<DataBoard
				    title="Breaks"
				    dataToDisplay={formatLoopsBreaks(shuttleData[selectedShuttleID]["breaks"])[0]}
				/>
				<DataBoard
				    title="Historical Locations"
				    dataToDisplay={[["..."]]}
				/>
				<div className="map-container">
				    <MapKitMap vehicles={ shuttleData } />
				</div>
			    </div>
			) : (
			    <p>No shuttle selected</p>
			)}
		    </div>
		) : (
		    <p>No shuttle data given</p>
		)}
	    </div>
	</>
    );
}
