import {
  BrowserRouter as Router,
  Routes,
  Route,
} from 'react-router';
import './App.css';
import LiveLocation from './locations/LiveLocation';
import Schedule from './schedule/Schedule';
import About from './about/About';
import rawRouteData from './shared/routes.json';
import { useState, useEffect, memo, lazy, Suspense } from "react";

// /map (fullscreen map) and /data (analytics dashboard) are infrequent
// visits but the heaviest modules in the bundle (~150-200 KB combined).
// Lazy-loading splits them into on-demand chunks so first paint doesn't
// pull them in. /generate-static-routes shares LiveLocationMapKit, so it
// gets the same lazy behavior for free.
const Data = lazy(() => import('./dashboard/Dashboard'));
const LiveLocationMapKit = lazy(() => import('./locations/components/LiveLocationMapKit'));
import type { ShuttleRouteData } from './types/route';
import Navigation from './components/Navigation';
import ErrorBoundary from './components/ErrorBoundary';
import config from "./utils/config";
import { devNow } from './utils/devTime';
import NotFound from './components/NotFound';
import ApplePrivacyPolicy from './privacy/ApplePrivacyPolicy';
import AppleAppSupport from './support/AppleAppSupport';

/** Isolated dev clock — 1-second ticks don't cascade re-renders to App children */
const DevClock = memo(function DevClock() {
  const [time, setTime] = useState(devNow().toLocaleTimeString());
  useEffect(() => {
    const id = setInterval(() => setTime(devNow().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div style={{ position: 'fixed', bottom: 8, left: 8, background: 'rgba(0,0,0,0.7)', color: '#0f0', padding: '4px 8px', borderRadius: 4, fontSize: 12, fontFamily: 'monospace', zIndex: 9999 }}>
      DEV {time}
    </div>
  );
});

function App() {
  const [selectedRoute, setSelectedRoute] = useState<string | null>(
    () => localStorage.getItem('shubble-route')
  );
  useEffect(() => {
    if (selectedRoute) localStorage.setItem('shubble-route', selectedRoute);
  }, [selectedRoute]);
  const GIT_REV = (import.meta.env.GIT_REV || 'unknown') as string;
  const routeData = rawRouteData as unknown as ShuttleRouteData;

  // if config.isStaging, add a meta tag to tell search engines not to index
  useEffect(() => {
    if (config.isStaging) {
      const meta = document.createElement('meta');
      meta.name = 'robots';
      meta.content = 'noindex, nofollow';
      document.getElementsByTagName('head')[0].appendChild(meta);
    } else {
      // add in Unami analytics script
      const script = document.createElement("script");
      script.src = "https://cloud.umami.is/script.js";
      script.dataset.websiteId = "d3082520-e157-498d-b9b3-67f83f4b8847";
      script.defer = true;

      document.head.appendChild(script);
    }
  }, []);

  // Shuttle setup is now manual — use POST /api/shuttles/schedule from the
  // terminal when you want to restart the test simulation. Automatic reset
  // on every page load was disruptive during iterative development.

  return (
    <ErrorBoundary>
      {import.meta.env.DEV && <DevClock />}
      <Router>
        <Routes>
          {/* with header and footer */}
          <Route element={<Navigation GIT_REV={GIT_REV} />} >
            <Route index element={<LiveLocation />} />
            <Route path='/schedule' element={<Schedule selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} />} />
            <Route path='/about' element={<About />} />
            <Route path='/data' element={
              <Suspense fallback={<div className="route-loading">Loading…</div>}>
                <Data />
              </Suspense>
            } />
            <Route path='/generate-static-routes' element={
              <Suspense fallback={<div className="route-loading">Loading…</div>}>
                <LiveLocationMapKit routeData={routeData} displayVehicles={true} generateRoutes={true} />
              </Suspense>
            } />
            <Route path='*' element={<NotFound />} />
          </Route>

          {/* without header and footer */}
          <Route>
            <Route path='/map' element={
              <Suspense fallback={<div className="route-loading">Loading map…</div>}>
                <LiveLocationMapKit routeData={routeData} generateRoutes={false} selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} isFullscreen={true} shuttleIconSize={35} />
              </Suspense>
            } />
            <Route path='/apple-privacy-policy' element={<ApplePrivacyPolicy />} />
            <Route path='/apple-app-support' element={<AppleAppSupport />} />
          </Route>
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
