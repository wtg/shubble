import {
  BrowserRouter as Router,
  Routes,
  Route,
} from 'react-router';
import './App.css';
import LiveLocation from './locations/LiveLocation';
import Schedule from './schedule/Schedule';
import About from './about/About';
import Data from './dashboard/Dashboard';
import LiveLocationMapKit from './locations/components/LiveLocationMapKit';
import rawRouteData from './shared/routes.json';
import { useState, useEffect, memo } from "react";
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

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    // Reset test server shuttles on page load so simulation starts fresh
    fetch('http://localhost:4000/api/shuttles/schedule', { method: 'POST' }).catch(() => {});
  }, []);

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
            <Route path='/data' element={<Data />} />
            <Route path='/generate-static-routes' element={<LiveLocationMapKit routeData={routeData} displayVehicles={true} generateRoutes={true} />} />
            <Route path='*' element={<NotFound />} />
          </Route>

          {/* without header and footer */}
          <Route>
            <Route path='/map' element={<LiveLocationMapKit routeData={routeData} generateRoutes={false} selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} isFullscreen={true} shuttleIconSize={35} />} />
            <Route path='/apple-privacy-policy' element={<ApplePrivacyPolicy />} />
            <Route path='/apple-app-support' element={<AppleAppSupport />} />
          </Route>
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
