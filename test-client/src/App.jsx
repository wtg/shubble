import { useEffect, useState, useCallback, useRef } from "react";
import { executeTest, stopTest, setGetShuttles } from "./AutoTest.js";
import "./App.css";

const NEXT_STATES = ["waiting", "entering", "looping", "on_break", "exiting"];

function App() {
  const [shuttles, setShuttles] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [locationCount, setLocationCount] = useState(0);
  const [geofenceCount, setGeofenceCount] = useState(0);
  const [keepShuttles, setKeepShuttles] = useState(false);

  const fetchShuttles = async () => {
    const res = await fetch("/api/shuttles");
    const data = await res.json();
    setShuttles(data);
    //console.log("Fetched shuttles:", data);
  };

  const addShuttle = async () => {
    await fetch("/api/shuttles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    await fetchShuttles();
  };

  const setNextState = async (nextState) => {
    if (!selectedId) return;
    await fetch(`/api/shuttles/${selectedId}/set-next-state`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: nextState }),
    });
    await fetchShuttles();
  };

  const fetchEvents = async () => {
    const res = await fetch("/api/events/today");
    const data = await res.json();
    setLocationCount(data.locationCount);
    setGeofenceCount(data.geofenceCount);
  };

  const clearEvents = async () => {
    await fetch(`/api/events/today?keepShuttles=${keepShuttles}`, {method: "DELETE"});
    if (!keepShuttles) {
      setSelectedId(null);
    }
    console.log("Cleared events for today");
  };

  const uploadTest = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    await fetch("/api/events/today?keepShuttles=false", {method: "DELETE"});
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
      executeTest(test);
    }
    reader.readAsText(file);
  };

  // this passes getShuttles as a stable function to the automatic testing module
  const shuttlesRef = useRef([]);
  const getShuttles = useCallback(() => shuttlesRef.current, []);
  useEffect(() => { shuttlesRef.current = shuttles; }, [shuttles]);
  useEffect(() => { setGetShuttles(getShuttles); }, []);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchShuttles();
    const interval = setInterval(fetchShuttles, 100);
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
      <button onClick={addShuttle}>Add Shuttle</button>

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
            {NEXT_STATES.map((state) => (
              <button
                key={state}
                disabled={selected.next_state === state}
                onClick={() => setNextState(state)}
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
