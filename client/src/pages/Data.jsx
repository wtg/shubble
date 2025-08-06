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
	var timeStampDate = new Date(tStamp);
	return timeStampDate.toLocaleTimeString();
    }

    function formatEntryExit(entry, exit) {
	if (entry === null) {
	    return "Shuttle never entered GeoFence";
	}
	var exitStr = "NOW";
	if (exit != null) {
	    exitStr = new Date(exit).toLocaleString();
	}
	return new Date(entry).toLocaleTimeString() + "-" + exitStr;
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
				<tr>
				    <ShuttleRow
					shuttleId="038471299"
					isActive={true}
					isAm={false}
				    />
				</tr>
				<tr>
				    <ShuttleRow
					shuttleId="038471300"
					isActive={false}
					isAm={true}
				    />
				</tr>
			    </tbody>
			) : (
			    <p>No shuttle data given</p>
			)}
		    </table>
		</div>

		{shuttleData ? (
		    <div className="main-content">
			<DataBoard
			    title="Summary"
			    children={selectedShuttleID}
			    numColums={1}
			/>
			<DataBoard
			    title="Loops"
			    children="..."
			    numColums={2}
			/>
			<DataBoard
			    title="Breaks"
			    children="..."
			    numColums={2}
			/>
			<DataBoard
			    title="Historical Locations"
			    children="..."
			    numColums={1}
			/>
			<div className="map-container">
			    <MapKitMap vehicles={ shuttleData } />
			</div>
		    </div>
		) : (
		    <p>No shuttle data given</p>
		)}
	    </div>
	</>
    );
}
