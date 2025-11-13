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
import { useState } from "react";
import type { ShuttleRouteData } from './ts/types/route';
import Navigation from './components/Navigation';

function App() {
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);
  const GIT_REV = (import.meta.env.GIT_REV || 'unknown') as string;
  const routeData = rawRouteData as unknown as ShuttleRouteData;
  return (
    <Router>
      <Routes>
        <Route element={<Navigation GIT_REV={GIT_REV} />} >
          <Route index element={<LiveLocation />} />
          <Route path='/schedule' element={<Schedule selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} />} />
          <Route path='/about' element={<About />} />
          <Route path='/data' element={<Data />} />
          <Route path='/generate-static-routes' element={<MapKitMap routeData={routeData} vehicles={null} generateRoutes={true} />} />
        </Route>
        <Route>
          <Route path='/map' element={<MapKitMap routeData={routeData} vehicles={null} generateRoutes={false} selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} isFullscreen={true} />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
