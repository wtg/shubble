import { useEffect, useState, useCallback, useRef } from "react";
import * as api from "./api.js";
import { STATES } from "./utils.js";
import { startTest, stopTest, setGetShuttles } from "./AutoTest.js";
import "./App.css";

function App() {
  const [shuttles, setShuttles] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [locationCount, setLocationCount] = useState(0);
  const [geofenceCount, setGeofenceCount] = useState(0);
  const [keepShuttles, setKeepShuttles] = useState(false);
  const shuttlesRef = useRef([]);
  const getShuttles = useCallback(() => shuttlesRef.current, []);

  // call api to get shuttles, then update the frontend's representation
  const updateShuttles = async () => {
    const res = await api.fetchShuttles();
    const data = await res.json();
    setShuttles(data);
  };

  const updateEvents = async () => {
    const res = await api.fetchEvents();
    const data = await res.json();
    setLocationCount(data.locationCount);
    setGeofenceCount(data.geofenceCount);
  };

  const clearEvents = async () => {
    await api.deleteEvents(keepShuttles);
    if (!keepShuttles) {
      setSelectedId(null);
    }
  };

  const uploadTest = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    await api.deleteEvents(false);
    setSelectedId(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      let test;
      try {
        test = JSON.parse(e.target.result);
      } catch (err) {
        console.error(err);
        return;
      }
      startTest(test);
    }
    reader.readAsText(file);
  };

  // pass getShuttles as a stable function
  useEffect(() => { shuttlesRef.current = shuttles; }, [shuttles]);
  useEffect(() => { setGetShuttles(getShuttles); }, []);

  useEffect(() => {
    updateEvents();
    const interval = setInterval(updateEvents, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    updateShuttles();
    const interval = setInterval(updateShuttles, 100);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (shuttles.length > 0 && !selectedId) {
      setSelectedId(shuttles[0].id);
    }
  }, [shuttles]);

  const selected = shuttles.find((s) => s.id === selectedId);

  return (
    <div className="app">
      <h1>Shuttle Manager</h1>
      <button onClick={api.addShuttle}>Add Shuttle</button>

      <div className="tabs">
        {shuttles.map((shuttle) => (
          <button
            key={shuttle.id}
            className={shuttle.id === selectedId ? "active" : ""}
            onClick={() => setSelectedId(shuttle.id)}
          >
            Shuttle {shuttle.id}
          </button>
        ))}
      </div>

      {selected ? (
        <div className="shuttle-details">
          <h2>Shuttle {selected.id} Details</h2>
          <table>
            <tbody>
              <tr>
                <th>ID</th>
                <td>{selected.id}</td>
              </tr>
              <tr>
                <th>Current State</th>
                <td>{selected.state}</td>
              </tr>
              {/* Add any other shuttle info you want here */}
            </tbody>
          </table>

          <h3>Set Next State</h3>
          <div className="state-buttons">
            {Object.values(STATES).map((state) => (
              <button
                key={state}
                disabled={selected.next_state === state}
                onClick={(() => {
                  if (selected) api.setNextState(state);
                })}
              >
                {state}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <p>No shuttle selected.</p>
      )}

      <div className="events-container">
        <div className="events-today">
          {locationCount} previous shuttle locations and {geofenceCount} previous entry/exit events
        </div>
        <button onClick={clearEvents}>Clear Events</button>
        <label>
          <input
            type="checkbox"
            checked={keepShuttles}
            onChange={(e) => setKeepShuttles(e.target.checked)}
          />
          Keep Shuttles
        </label>
      </div>

      <div className="test-container">
        <p><strong>JSON Test Case Executor</strong></p>
        <input type="file" accept=".json" onChange={uploadTest}></input>
        <button onClick={stopTest}>
          Stop Test
        </button>
      </div>
    </div>
  );
}

export default App;
