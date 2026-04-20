import { useState, useMemo, useEffect, useCallback } from 'react';
import rawRouteData from '../shared/routes.json';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';
import './styles/Gallery.css';

type StopInfo = {
  key: string;
  name: string;
  coordinates: [number, number];
  routes: string[];
  image: string | null;
};

// Map stop keys to image filenames in /gallery/
// Add entries here as photos are added to frontend/public/gallery/
const STOP_IMAGES: Record<string, string> = {
  // Example: 'STUDENT_UNION': 'student_union.jpg',
};

function buildStopList(routeData: ShuttleRouteData): StopInfo[] {
  const stopMap = new Map<string, StopInfo>();

  for (const [routeName, direction] of Object.entries(routeData)) {
    for (const stopKey of direction.STOPS) {
      const stopData = direction[stopKey] as ShuttleStopData | undefined;
      if (!stopData || !stopData.NAME) continue;

      if (stopKey.endsWith('_RETURN')) continue;

      const existing = stopMap.get(stopKey);
      if (existing) {
        if (!existing.routes.includes(routeName)) {
          existing.routes.push(routeName);
        }
      } else {
        stopMap.set(stopKey, {
          key: stopKey,
          name: stopData.NAME,
          coordinates: stopData.COORDINATES,
          routes: [routeName],
          image: STOP_IMAGES[stopKey] || null,
        });
      }
    }
  }

  return Array.from(stopMap.values());
}

const ROUTE_COLORS: Record<string, string> = {};

function getRouteColor(routeData: ShuttleRouteData, routeName: string): string {
  if (!ROUTE_COLORS[routeName]) {
    const data = routeData[routeName as keyof ShuttleRouteData];
    ROUTE_COLORS[routeName] = data?.COLOR || '#578FCA';
  }
  return ROUTE_COLORS[routeName];
}

function formatRouteName(name: string): string {
  return name.charAt(0) + name.slice(1).toLowerCase();
}

export default function Gallery() {
  const routeData = rawRouteData as unknown as ShuttleRouteData;
  const stops = useMemo(() => buildStopList(routeData), [routeData]);
  const [filter, setFilter] = useState<string>('ALL');
  const [currentIndex, setCurrentIndex] = useState(0);

  const routeNames = useMemo(() => {
    const names = new Set<string>();
    for (const stop of stops) {
      for (const r of stop.routes) names.add(r);
    }
    return Array.from(names);
  }, [stops]);

  const filteredStops = useMemo(() => {
    if (filter === 'ALL') return stops;
    return stops.filter((s) => s.routes.includes(filter));
  }, [stops, filter]);

  // Reset index when filter changes
  useEffect(() => {
    setCurrentIndex(0);
  }, [filter]);

  const goTo = useCallback((index: number) => {
    setCurrentIndex(index);
  }, []);

  const goPrev = useCallback(() => {
    setCurrentIndex((i) => (i === 0 ? filteredStops.length - 1 : i - 1));
  }, [filteredStops.length]);

  const goNext = useCallback(() => {
    setCurrentIndex((i) => (i === filteredStops.length - 1 ? 0 : i + 1));
  }, [filteredStops.length]);

  // Keyboard navigation
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') goPrev();
      else if (e.key === 'ArrowRight') goNext();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [goPrev, goNext]);

  const currentStop = filteredStops[currentIndex];
  if (!currentStop) return null;

  return (
    <div className="gallery">
      <section className="gallery-header">
        <h1>Shuttle Stop Gallery</h1>
        <p>
          See what each shuttle stop looks like so you know exactly where to wait.
        </p>
      </section>

      <section className="gallery-filters">
        <button
          className={`filter-btn ${filter === 'ALL' ? 'active' : ''}`}
          onClick={() => setFilter('ALL')}
        >
          All Stops
        </button>
        {routeNames.map((name) => (
          <button
            key={name}
            className={`filter-btn ${filter === name ? 'active' : ''}`}
            style={
              filter === name
                ? { backgroundColor: getRouteColor(routeData, name), borderColor: getRouteColor(routeData, name) }
                : { borderColor: getRouteColor(routeData, name), color: getRouteColor(routeData, name) }
            }
            onClick={() => setFilter(name)}
          >
            {formatRouteName(name)}
          </button>
        ))}
      </section>

      <section className="slideshow">
        {/* Main slide */}
        <div className="slide-container">
          <button className="slide-arrow slide-arrow-left" onClick={goPrev} aria-label="Previous stop">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>

          <div className="slide-content" key={currentStop.key}>
            <div className="slide-image">
              {currentStop.image ? (
                <img
                  src={`/gallery/${currentStop.image}`}
                  alt={`${currentStop.name} shuttle stop`}
                />
              ) : (
                <div className="placeholder-image">
                  <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#999" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                  <span>Photo coming soon</span>
                </div>
              )}
            </div>

            <div className="slide-info">
              <h2 className="slide-title">{currentStop.name}</h2>
              <div className="slide-routes">
                {currentStop.routes.map((r) => (
                  <span
                    key={r}
                    className="route-badge"
                    style={{ backgroundColor: getRouteColor(routeData, r) }}
                  >
                    {formatRouteName(r)} Route
                  </span>
                ))}
              </div>
              <p className="slide-counter">
                {currentIndex + 1} of {filteredStops.length}
              </p>
            </div>
          </div>

          <button className="slide-arrow slide-arrow-right" onClick={goNext} aria-label="Next stop">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        </div>

        {/* Dot indicators */}
        <div className="slide-dots">
          {filteredStops.map((stop, i) => (
            <button
              key={stop.key}
              className={`slide-dot ${i === currentIndex ? 'active' : ''}`}
              onClick={() => goTo(i)}
              aria-label={`Go to ${stop.name}`}
            />
          ))}
        </div>

        {/* Thumbnail strip */}
        <div className="thumbnail-strip">
          {filteredStops.map((stop, i) => (
            <button
              key={stop.key}
              className={`thumbnail ${i === currentIndex ? 'active' : ''}`}
              onClick={() => goTo(i)}
            >
              <div className="thumbnail-inner">
                {stop.image ? (
                  <img
                    src={`/gallery/${stop.image}`}
                    alt={stop.name}
                    loading="lazy"
                  />
                ) : (
                  <div className="thumbnail-placeholder">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#bbb" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                  </div>
                )}
              </div>
              <span className="thumbnail-label">{stop.name}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
