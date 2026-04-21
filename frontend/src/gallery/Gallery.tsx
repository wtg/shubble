import { useState, useMemo, useCallback, useRef } from 'react';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Pagination, Keyboard } from 'swiper/modules';
import type { Swiper as SwiperType } from 'swiper';
import rawRouteData from '../shared/routes.json';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';

import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/pagination';
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

  const hiddenRoutes = ['ENTRY1', 'EXIT1', 'EXIT2'];

  for (const [routeName, direction] of Object.entries(routeData)) {
    if (hiddenRoutes.includes(routeName)) continue;
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
  const [activeIndex, setActiveIndex] = useState(0);
  const swiperRef = useRef<SwiperType | null>(null);

  const changeFilter = useCallback((newFilter: string) => {
    setFilter(newFilter);
    setActiveIndex(0);
    if (swiperRef.current) {
      swiperRef.current.slideTo(0, 0);
    }
  }, []);

  const HIDDEN_ROUTES = ['ENTRY1', 'EXIT1', 'EXIT2'];

  const routeNames = useMemo(() => {
    const names = new Set<string>();
    for (const stop of stops) {
      for (const r of stop.routes) {
        if (!HIDDEN_ROUTES.includes(r)) names.add(r);
      }
    }
    return Array.from(names);
  }, [stops]);

  const filteredStops = useMemo(() => {
    if (filter === 'ALL') return stops;
    return stops.filter((s) => s.routes.includes(filter));
  }, [stops, filter]);

  if (filteredStops.length === 0) return null;

  return (
    <div className="gallery">
      <h1 className="gallery-title">Shuttle Stop Gallery</h1>

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

      {/* Swiper carousel */}
      <section className="slideshow">
        <div className="carousel-wrapper">
        <Swiper
          onSwiper={(swiper) => { swiperRef.current = swiper; }}
          onSlideChange={(swiper) => setActiveIndex(swiper.realIndex)}
          grabCursor
          slidesPerView={1}
          speed={500}
          loop
          keyboard={{ enabled: true }}
          navigation={{
            nextEl: '.gallery-arrow--next',
            prevEl: '.gallery-arrow--prev',
          }}
          pagination={{ clickable: true, el: '.gallery-pagination' }}
          modules={[Navigation, Pagination, Keyboard]}
          className="gallery-swiper"
        >
          {filteredStops.map((stop) => (
            <SwiperSlide key={stop.key} className="gallery-slide">
              {stop.image ? (
                <img
                  src={`/gallery/${stop.image}`}
                  alt={`${stop.name} shuttle stop`}
                  draggable={false}
                />
              ) : (
                <div className="slide-placeholder">
                  <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10" />
                    <polyline points="21 15 16 10 5 21" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <line x1="2" y1="22" x2="22" y2="22" />
                  </svg>
                  <span>Photo coming soon</span>
                </div>
              )}

              {/* Route badges */}
              <div className="slide-badges">
                {stop.routes.map((r) => (
                  <span
                    key={r}
                    className="route-badge"
                    style={{ backgroundColor: getRouteColor(routeData, r) }}
                  >
                    {formatRouteName(r)}
                  </span>
                ))}
              </div>
            </SwiperSlide>
          ))}
        </Swiper>

        {/* Navigation arrows */}
        <button className="gallery-arrow gallery-arrow--prev" aria-label="Previous">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <button className="gallery-arrow gallery-arrow--next" aria-label="Next">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </button>

        </div>

        <div className="gallery-pagination" />

        {/* Stop name navigation bar */}
        <div className="stop-nav">
          {filteredStops.map((stop, i) => (
            <button
              key={stop.key}
              className={`stop-name ${i === activeIndex ? 'active' : ''}`}
              onClick={() => {
                setActiveIndex(i);
                swiperRef.current?.slideToLoop(i, 600);
              }}
            >
              {stop.name}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
