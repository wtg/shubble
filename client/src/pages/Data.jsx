import React, {
    useState,
    useEffect,
} from 'react';
import "../styles/Data.css"
import MapKitMap from '../components/MapKitMap';
import DataBoard from '../components/DataBoard';

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
            console.log(data);
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
			This is the sidebar
		</div>
		<div className="main-content">
		    <DataBoard
			title="Summary"
			children="..."
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
	    </div>
	</>
    );
}
