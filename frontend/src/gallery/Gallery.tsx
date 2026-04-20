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

const SWIPE_THRESHOLD = 50;

export default function Gallery() {
  const routeData = rawRouteData as unknown as ShuttleRouteData;
  const stops = useMemo(() => buildStopList(routeData), [routeData]);
  const [filter, setFilter] = useState<string>('ALL');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [slideDirection, setSlideDirection] = useState<'next' | 'prev'>('next');
  const [dragOffset, setDragOffset] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const stopNavRef = useRef<HTMLDivElement>(null);
  const dragStartX = useRef(0);

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

  // Scroll active stop-name button into view
  useEffect(() => {
    if (!stopNavRef.current) return;
    const active = stopNavRef.current.querySelector('.stop-name.active') as HTMLElement | null;
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

  // Touch / mouse drag handlers
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    setIsDragging(true);
    dragStartX.current = e.clientX;
    setDragOffset(0);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return;
    setDragOffset(e.clientX - dragStartX.current);
  }, [isDragging]);

  const handlePointerUp = useCallback(() => {
    if (!isDragging) return;
    setIsDragging(false);
    if (dragOffset < -SWIPE_THRESHOLD) {
      goNext();
    } else if (dragOffset > SWIPE_THRESHOLD) {
      goPrev();
    }
    setDragOffset(0);
  }, [isDragging, dragOffset, goNext, goPrev]);

  const currentStop = filteredStops[currentIndex];
  if (!currentStop) return null;

  const prevIndex = currentIndex === 0 ? filteredStops.length - 1 : currentIndex - 1;
  const nextIndex = currentIndex === filteredStops.length - 1 ? 0 : currentIndex + 1;
  const prevStop = filteredStops[prevIndex];
  const nextStop = filteredStops[nextIndex];

  function renderSlideContent(stop: StopInfo) {
    if (stop.image) {
      return <img src={`/gallery/${stop.image}`} alt={`${stop.name} shuttle stop`} draggable={false} />;
    }
    return (
      <div className="slide-placeholder">
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10" />
          <polyline points="21 15 16 10 5 21" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <line x1="2" y1="22" x2="22" y2="22" />
        </svg>
        <span>Photo coming soon</span>
      </div>
    );
  }

  return (
    <div className="gallery">
      {/* Header */}
      <section className="gallery-hero">
        <div className="gallery-hero-bg" />
        <div className="gallery-hero-content">
          <span className="gallery-hero-label">Shuttle Stops</span>
          <h1>Stop Gallery</h1>
          <p>Know exactly where to wait. Browse every shuttle stop on campus.</p>
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
                  : { borderColor: color, color }
              }
              onClick={() => changeFilter(name)}
            >
              <span className="filter-pill-dot" style={{ backgroundColor: color }} />
              {formatRouteName(name)}
            </button>
          );
        })}
      </section>

      {/* Slideshow */}
      <section className="slideshow">
        <div
          className="carousel"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          style={{ touchAction: 'pan-y' }}
        >
          {/* Previous slide (peek) */}
          {filteredStops.length > 1 && (
            <button className="carousel-side carousel-side--prev" onClick={goPrev} aria-label={`Previous: ${prevStop.name}`}>
              {renderSlideContent(prevStop)}
            </button>
          )}

          {/* Current slide (main) */}
          <div
            className={`carousel-main ${isDragging ? '' : `carousel-main--${slideDirection}`}`}
            key={currentStop.key}
            style={isDragging ? { transform: `translateX(${dragOffset}px)`, transition: 'none' } : undefined}
          >
            {renderSlideContent(currentStop)}

            {/* Route badges */}
            <div className="slide-badges">
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

          {/* Next slide (peek) */}
          {filteredStops.length > 1 && (
            <button className="carousel-side carousel-side--next" onClick={goNext} aria-label={`Next: ${nextStop.name}`}>
              {renderSlideContent(nextStop)}
            </button>
          )}
        </div>

        {/* Stop name navigation bar */}
        <div className="stop-nav" ref={stopNavRef}>
          {filteredStops.map((stop, i) => (
            <button
              key={stop.key}
              className={`stop-name ${i === currentIndex ? 'active' : ''}`}
              onClick={() => goTo(i)}
            >
              {stop.name}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
