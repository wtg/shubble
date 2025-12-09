import { useEffect, useState, useCallback, useRef } from "react";
import * as api from "./api.js";
import { STATES } from "./utils.js";
import Tester from "./AutoTest.js";
import "./App.css";

function App() {
  const [shuttles, setShuttles] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [locationCount, setLocationCount] = useState(0);
  const [geofenceCount, setGeofenceCount] = useState(0);
  const [keepShuttles, setKeepShuttles] = useState(false);
  const [routes, setRoutes] = useState([]);
  const [pendingRouteIds, setPendingRouteIds] = useState(new Set());
  const shuttlesRef = useRef([]);
  const getShuttles = useCallback(() => shuttlesRef.current, []);
  const testerRef = useRef(null);

  if (!testerRef.current) {
    testerRef.current = Tester(getShuttles);
  }
  const tester = testerRef.current;

  // call api, then update the frontend's representation
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

  const getRoutes = async () => {
    const res = await api.fetchRoutes();
    const data = await res.json();
    setRoutes(data);
  };

  const uploadTest = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    await api.deleteEvents(false);
    setSelectedId(null);

    const text = await file.text();
    const json = JSON.parse(text);
    await tester.startTest(json);
    // reset the chosen file after the test finishes
    event.target.value = "";
  };

  // give react a copy (new reference) so it detects the change
  const addPending = (id) => {
    setPendingRouteIds(prev => new Set([...prev, id]));
  };

  const removePending = (id) => {
    setPendingRouteIds(prev => new Set([...prev].filter(x => x !== id)));
  };

  const selected = shuttles.find((s) => s.id === selectedId);

  useEffect(() => { getRoutes(); }, []);

  useEffect(() => { shuttlesRef.current = shuttles; }, [shuttles]);

  useEffect(() => {
    if (shuttles.length > 0 && !selectedId) {
      setSelectedId(shuttles[0].id);
    }
  }, [shuttles]);

  useEffect(() => {
    updateShuttles();
    const interval = setInterval(updateShuttles, 100);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    updateEvents();
    const interval = setInterval(updateEvents, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      <h1>Shuttle Manager</h1>
      <button onClick={() => api.addShuttle()}>Add Shuttle</button>

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
            {Object.values(STATES).map(state => (
              <button
                key={state}
                disabled={selected.next_state === state}
                onClick={() => {
                  if (state === STATES.LOOPING) {
                    addPending(selected.id);
                  } else {
                    removePending(selected.id);
                    api.setNextState(selected.id, {state, route: undefined});
                  }
                }}
              >
                {state}
              </button>
            ))}

            {pendingRouteIds.has(selected.id) && (
              <div className="select-route">
                <label>
                  Route:
                  <select
                    value=""
                    onChange={e => {
                      api.setNextState(selected.id, {state: STATES.LOOPING, route: e.target.value});
                      removePending(selected.id);
                    }}
                  >
                    <option value="" disabled>
                      Select route
                    </option>
                    {routes.map(route => (
                      <option key={route} value={route}>
                        {route}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
          </div>
        </div>
      ) : (
        <p>No shuttle selected.</p>
      )}

      <div className="events-container">
        <div className="events-today">
          {locationCount} previous shuttle locations and {geofenceCount} previous entry/exit events
        </div>
        <button onClick={() => clearEvents()}>Clear Events</button>
        <label>
          <input
            type="checkbox"
            checked={keepShuttles}
            onChange={(e) => setKeepShuttles(e.target.checked)}
          />
          Keep Shuttles
        </label>
      </div>

      <h3>JSON Test Case Executor</h3>
      <div className="test-container">
        <input type="file" accept=".json" onChange={(e) => uploadTest(e)}/>
        <button onClick={() => tester.stopTest()}>Stop Test</button>
      </div>
    </div>
  );
}

export default App;
