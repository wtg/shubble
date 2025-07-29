import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [shuttles, setShuttles] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  const fetchShuttles = async () => {
    const res = await fetch("/api/shuttles");
    const data = await res.json();
    setShuttles(data);
    if (data.length && !selectedId) setSelectedId(data[0].id);
  };

  const addShuttle = async () => {
    const res = await fetch("/api/shuttles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ loop: selectedLoop })
    });
    await fetchShuttles();
  };

  const trigger = async (action) => {
    await fetch(`/api/shuttles/${selectedId}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action })
    });
    await fetchShuttles();
  };

  useEffect(() => {
    fetchShuttles();
    const interval = setInterval(fetchShuttles, 5000);
    return () => clearInterval(interval);
  }, []);

  const selected = shuttles.find(s => s.id === selectedId);

  return (
    <div className="app">
      <div className="sidebar">
        <h2>Shuttles</h2>
        <select value={selectedLoop} onChange={(e) => setSelectedLoop(e.target.value)}>
          <option value="north">North Loop</option>
          <option value="west">West Loop</option>
        </select>
        <button onClick={addShuttle}>Add Shuttle</button>
        <ul>
          {shuttles.map(shuttle => (
            <li
              key={shuttle.id}
              className={shuttle.id === selectedId ? "selected" : ""}
              onClick={() => setSelectedId(shuttle.id)}
            >
              Shuttle {shuttle.id.slice(-4)} ({shuttle.loop})
            </li>
          ))}
        </ul>
      </div>
      <div className="details">
        {selected ? (
          <>
            <h2>Shuttle {selected.id}</h2>
            <p><strong>Loop:</strong> {selected.loop}</p>
            <p><strong>State:</strong> {selected.state}</p>
            <p><strong>Speed:</strong> {selected.speed}</p>
            <p><strong>Location:</strong> {selected.location.toFixed(2)}</p>
            <div className="buttons">
              <button onClick={() => trigger("entering")}>Trigger Entry</button>
              <button onClick={() => trigger("on_break")}>Trigger Break</button>
              <button onClick={() => trigger("exiting")}>Trigger Exit</button>
            </div>
          </>
        ) : (
          <p>No shuttle selected</p>
        )}
      </div>
    </div>
  );
}

export default App;
