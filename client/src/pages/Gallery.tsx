import React, { useState } from 'react';
import rawRouteData from '../data/routes.json';
import '../styles/Gallery.css';

const routeData: any = rawRouteData;

const AVAILABLE_ROUTES = ['NORTH', 'WEST'];

export default function Gallery() {
  const [route, setRoute] = useState<string>('NORTH');
  const [selectedStop, setSelectedStop] = useState<string | null>(null);
  const routeColor = (routeData as any)[route] && (routeData as any)[route].COLOR || '#888';

  const stopsForRoute = (r: string) => {
    const rdata = routeData[r];
    if (!rdata) return [];
    return rdata.STOPS && rdata.STOPS.length ? rdata.STOPS : (rdata.POLYLINE_STOPS || []);
  };

  const displayName = (r: string, stopKey: string) => {
    try {
      const name = routeData[r] && routeData[r][stopKey] && routeData[r][stopKey].NAME;
      return name || stopKey;
    } catch (e) {
      return stopKey;
    }
  };

  const imageUrlForStop = (stopKey: string) => {
    // Use the single placeholder `stop.jpg` for all stops if a specific file is not present.
    return `/stops/${stopKey}.jpg`;
  };

  const coordsForStop = (r: string, stopKey: string) => {
    try {
      const coords = routeData[r] && routeData[r][stopKey] && routeData[r][stopKey].COORDINATES;
      if (!coords || !Array.isArray(coords)) return null;
      const [lat, lon] = coords;
      return `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    } catch (e) {
      return null;
    }
  };

  const stops = stopsForRoute(route);

  return (
    <div className="gallery-page">
      <h1>Gallery</h1>

      <div className="route-selector">
        {AVAILABLE_ROUTES.map(r => (
          <button
            key={r}
            className={`route-btn ${r === route ? 'active' : ''}`}
            onClick={() => { setRoute(r); setSelectedStop(null); }}
          >
            {r}
          </button>
        ))}
      </div>

      <div className="gallery-container">
        <div className="stop-list">
          <h2>{route} Stops</h2>
          <ul>
            {stops.map((s: string) => (
              <li
                key={s}
                className={`stop-card ${s === selectedStop ? 'selected' : ''}`}
                style={s === selectedStop ? { borderLeftColor: routeColor } : undefined}
              >
                <button onClick={() => setSelectedStop(s)}>
                  <img
                    className="stop-thumb"
                    src={imageUrlForStop(s)}
                    alt={displayName(route, s)}
                    onError={(e) => { (e.target as HTMLImageElement).src = '/stop.jpg'; }}
                  />

                  <div className="stop-info">
                    <div className="stop-name">{displayName(route, s)}</div>
                    <div className="stop-coords">{coordsForStop(route, s) || ''}</div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="stop-preview">
          {selectedStop ? (
            <div>
              <h3>{displayName(route, selectedStop)}</h3>
              <img
                src={imageUrlForStop(selectedStop)}
                alt={displayName(route, selectedStop)}
                onError={(e) => { (e.target as HTMLImageElement).src = '/stop.jpg'; }}
              />
            </div>
          ) : (
            <div className="placeholder">Select a stop to view its photo</div>
          )}
        </div>
      </div>
    </div>
  );
}
