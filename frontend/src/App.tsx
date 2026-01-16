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
import { useState, useEffect } from "react";
import type { ShuttleRouteData } from './types/route';
import Navigation from './components/Navigation';
import ErrorBoundary from './components/ErrorBoundary';
import { config } from "./utils/config";

function App() {
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);
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

  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          {/* with header and footer */}
          <Route element={<Navigation GIT_REV={GIT_REV} />} >
            <Route index element={<LiveLocation />} />
            <Route path='/schedule' element={<Schedule selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} />} />
            <Route path='/about' element={<About />} />
            <Route path='/data' element={<Data />} />
            <Route path='/generate-static-routes' element={<LiveLocationMapKit routeData={routeData} displayVehicles={true} generateRoutes={true} />} />
          </Route>

          {/* without header and footer */}
          <Route>
            <Route path='/map' element={<LiveLocationMapKit routeData={routeData} generateRoutes={false} selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} isFullscreen={true} />} />
          </Route>
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
