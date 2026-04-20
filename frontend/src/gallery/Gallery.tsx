import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
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
  const [slideDirection, setSlideDirection] = useState<'next' | 'prev'>('next');
  const thumbnailStripRef = useRef<HTMLDivElement>(null);

  const changeFilter = useCallback((newFilter: string) => {
    setFilter(newFilter);
    setCurrentIndex(0);
  }, []);

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

  // Scroll active thumbnail into view
  useEffect(() => {
    if (!thumbnailStripRef.current) return;
    const active = thumbnailStripRef.current.querySelector('.thumbnail.active') as HTMLElement | null;
    if (active) {
      active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [currentIndex]);

  const goTo = useCallback((index: number) => {
    setSlideDirection(index > currentIndex ? 'next' : 'prev');
    setCurrentIndex(index);
  }, [currentIndex]);

  const goPrev = useCallback(() => {
    setSlideDirection('prev');
    setCurrentIndex((i) => (i === 0 ? filteredStops.length - 1 : i - 1));
  }, [filteredStops.length]);

  const goNext = useCallback(() => {
    setSlideDirection('next');
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

  const progress = ((currentIndex + 1) / filteredStops.length) * 100;

  return (
    <div className="gallery">
      {/* Hero header */}
      <section className="gallery-hero">
        <div className="gallery-hero-bg" />
        <div className="gallery-hero-content">
          <span className="gallery-hero-label">Shuttle Stops</span>
          <h1>Stop Gallery</h1>
          <p>
            Know exactly where to wait. Browse every shuttle stop on campus.
          </p>
        </div>
      </section>

      {/* Route filter pills */}
      <section className="gallery-filters">
        <button
          className={`filter-pill ${filter === 'ALL' ? 'active' : ''}`}
          onClick={() => changeFilter('ALL')}
        >
          All Stops
        </button>
        {routeNames.map((name) => {
          const color = getRouteColor(routeData, name);
          const isActive = filter === name;
          return (
            <button
              key={name}
              className={`filter-pill ${isActive ? 'active' : ''}`}
              style={
                isActive
                  ? { backgroundColor: color, borderColor: color, color: '#fff' }
                  : { borderColor: color, color: color }
              }
              onClick={() => changeFilter(name)}
            >
              <span
                className="filter-pill-dot"
                style={{ backgroundColor: color }}
              />
              {formatRouteName(name)}
            </button>
          );
        })}
      </section>

      {/* Progress bar */}
      <div className="gallery-progress-track">
        <div
          className="gallery-progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Slideshow */}
      <section className="slideshow">
        <div className="slide-stage">
          {/* Prev arrow */}
          <button className="slide-nav slide-nav--prev" onClick={goPrev} aria-label="Previous stop">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>

          {/* Slide card */}
          <div
            className={`slide-card slide-card--${slideDirection}`}
            key={currentStop.key}
          >
            <div className="slide-card-image">
              {currentStop.image ? (
                <img
                  src={`/gallery/${currentStop.image}`}
                  alt={`${currentStop.name} shuttle stop`}
                />
              ) : (
                <div className="slide-card-placeholder">
                  <div className="placeholder-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10" />
                      <polyline points="21 15 16 10 5 21" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <line x1="2" y1="22" x2="22" y2="22" />
                    </svg>
                  </div>
                  <span className="placeholder-text">Photo coming soon</span>
                </div>
              )}

              {/* Overlay info */}
              <div className="slide-card-overlay">
                <div className="slide-card-overlay-inner">
                  <span className="slide-card-counter">{currentIndex + 1} / {filteredStops.length}</span>
                  <h2 className="slide-card-title">{currentStop.name}</h2>
                  <div className="slide-card-badges">
                    {currentStop.routes.map((r) => (
                      <span
                        key={r}
                        className="route-badge"
                        style={{ backgroundColor: getRouteColor(routeData, r) }}
                      >
                        {formatRouteName(r)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Next arrow */}
          <button className="slide-nav slide-nav--next" onClick={goNext} aria-label="Next stop">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        </div>

        {/* Thumbnail strip */}
        <div className="thumbnail-strip" ref={thumbnailStripRef}>
          {filteredStops.map((stop, i) => (
            <button
              key={stop.key}
              className={`thumbnail ${i === currentIndex ? 'active' : ''}`}
              onClick={() => goTo(i)}
              aria-label={`Go to ${stop.name}`}
            >
              <div className="thumbnail-img">
                {stop.image ? (
                  <img src={`/gallery/${stop.image}`} alt={stop.name} loading="lazy" />
                ) : (
                  <div className="thumbnail-img-empty">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                  </div>
                )}
              </div>
              <span className="thumbnail-name">{stop.name}</span>
            </button>
          ))}
        </div>

        {/* Keyboard hint */}
        <p className="gallery-hint">
          Use <kbd>&#8592;</kbd> <kbd>&#8594;</kbd> arrow keys to navigate
        </p>
      </section>
    </div>
  );
}
