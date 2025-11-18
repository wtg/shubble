import {
  BrowserRouter as Router,
  Routes,
  Route,
} from 'react-router';
import './App.css';
import LiveLocation from './pages/LiveLocation';
import Schedule from './components/Schedule';
import About from './pages/About';
import Data from './pages/Data';
import MapKitMap from './components/MapKitMap';
import rawRouteData from './data/routes.json';
import { useState, useEffect, use } from "react";
import WarningBanner from './components/WarningBanner';
import type { ShuttleRouteData } from './ts/types/route';
import Navigation from './components/Navigation';
import config from "./ts/config";

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
    }
  }, []);

  return (
    <Router>
      <Routes>
        {/* with header and footer */}
        <Route element={<Navigation GIT_REV={GIT_REV} />} >
          <Route index element={<LiveLocation />} />
          <Route path='/schedule' element={<Schedule selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} />} />
          <Route path='/about' element={<About />} />
          <Route path='/data' element={<Data />} />
          <Route path='/generate-static-routes' element={<MapKitMap routeData={routeData} vehicles={null} generateRoutes={true} />} />
        </Route>

        {/* without header and footer */}
        <Route>
          <Route path='/map' element={<MapKitMap routeData={routeData} vehicles={null} generateRoutes={false} selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} isFullscreen={true} />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
