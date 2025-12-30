import {
  useState,
  useEffect,
} from 'react';
import "./styles/Dashboard.css"
import DataBoard from './components/DataBoard';
import ShuttleRow from './components/ShuttleRow';
import type { VehicleInformationMap } from '../types/vehicleLocation';
import config from '../utils/config';

export default function Data() {

  const [shuttleData, setShuttleData] = useState<VehicleInformationMap | null>(null);
  const [selectedShuttleID, setSelectedShuttleID] = useState<string | null>(null);

  const fetchShuttleData = async () => {
    try {
      const response = await fetch(`${config.apiBaseUrl}/api/today`);
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
      if (selectedShuttleID === null || !(selectedShuttleID in shuttleData)) {
        setSelectedShuttleID(Object.keys(shuttleData)[0]);
      }
    }
  }, [shuttleData]);

  return (
    <>
      <div className="page-container">
        <div className="sidebar">
          <table className="sidebar-table">
            <thead>
              <tr>
                <th colSpan={3}>
                  {new Date().toLocaleDateString('en-US', {
                    weekday: 'short',
                    month: 'long',
                    day: 'numeric'
                  })}
                </th>
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
            {selectedShuttleID && shuttleData[selectedShuttleID] ? (
              <div className="main-content">
                <DataBoard
                  title="Summary"
                  datatable={[["11:05 AM - NOW"], ["13 loops"], ["23 minutes of break time"]]}
                />
                <DataBoard
                  title="Loops"
                  datatable={[["12 minutes", "11:07-11:19"], ["11 minutes", "11:23-11:34"]]}
                />
                <DataBoard
                  title="Breaks"
                  datatable={[["17 minutes", "12:32-12:49"]]}
                />
                {/* <DataBoard
                  title="Historical Locations"
                  dataToDisplay={["..."]}
                /> */}
                <div className="map-container">
                  {/* <MapKitMap vehicles={shuttleData} /> */}
                </div>
              </div>
            ) : (
              <p>Invalid shuttle selected</p>
            )}
          </div>
        ) : (
          <p>No shuttle data given</p>
        )}
      </div>
    </>
  );
}
