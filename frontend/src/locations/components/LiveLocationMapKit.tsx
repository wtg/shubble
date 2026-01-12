import { useEffect, useState, useMemo } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import '../styles/MapKitMap.css';
import ShuttleIcon from "./ShuttleIcon";
import config from "../../utils/config";

import type { ShuttleRouteData } from "../../types/route";
import type { VehicleInformationMap, VehicleETAs } from "../../types/vehicleLocation";
import type { Coordinate } from "../../utils/mapUtils";

import MapKitCanvas from "../../mapkit/MapKitCanvas";
import MapKitAnimation from "../../mapkit/MapKitAnimation";
import type { AnimatedAnnotation } from "../../mapkit/MapKitAnimation";
import MapKitOverlays from "../../mapkit/MapKitOverlays";

type LiveLocationMapKitProps = {
  routeData: ShuttleRouteData | null;
  displayVehicles?: boolean;
  generateRoutes?: boolean;
  selectedRoute?: string | null;
  setSelectedRoute?: (route: string | null) => void;
  isFullscreen?: boolean;
  showTrueLocation?: boolean;
  onStopTimesUpdate?: (stopTimes: VehicleETAs) => void;
  onVehiclesAtStopsUpdate?: (vehiclesAtStops: Record<string, string[]>) => void;
};

export default function LiveLocationMapKit({
  routeData,
  displayVehicles = true,
  generateRoutes = false,
  selectedRoute,
  setSelectedRoute,
  isFullscreen = false,
  showTrueLocation = true,
  onStopTimesUpdate,
  onVehiclesAtStopsUpdate
}: LiveLocationMapKitProps) {
  const [map, setMap] = useState<(mapkit.Map | null)>(null);
  const [vehicles, setVehicles] = useState<VehicleInformationMap | null>(null);
  const [vehicleAnnotations, setVehicleAnnotations] = useState<Record<string, mapkit.Annotation>>({});

  // Fetch location data on component mount and set up polling
  useEffect(() => {
    if (!displayVehicles) return;

    const pollLocation = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/locations`);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setVehicles(data);
      } catch (error) {
        console.error('Error fetching location:', error);
      }
    }

    pollLocation();

    // refresh location every 5 seconds
    const refreshLocation = setInterval(pollLocation, 5000);

    return () => {
      clearInterval(refreshLocation);
    }

  }, []);

  // Extract and pass stop times and vehicles at stops to parent component
  useEffect(() => {
    if (!vehicles) return;

    // Extract stop times
    if (onStopTimesUpdate) {
      const stopTimes: VehicleETAs = {};
      Object.entries(vehicles).forEach(([vehicleId, vehicleData]) => {
        if (vehicleData.eta?.etas) {
          stopTimes[vehicleId] = vehicleData.eta.etas;
        }
      });
      onStopTimesUpdate(stopTimes);
    }

    // Extract vehicles at stops
    if (onVehiclesAtStopsUpdate) {
      const vehiclesAtStops: Record<string, string[]> = {};
      Object.entries(vehicles).forEach(([vehicleId, vehicleData]) => {
        if (vehicleData.is_at_stop && vehicleData.current_stop) {
          const stopName = vehicleData.current_stop;
          if (!vehiclesAtStops[stopName]) {
            vehiclesAtStops[stopName] = [];
          }
          vehiclesAtStops[stopName].push(vehicleId);
        }
      });
      onVehiclesAtStopsUpdate(vehiclesAtStops);
    }
  }, [vehicles, onStopTimesUpdate, onVehiclesAtStopsUpdate]);

  // Memoize flattened routes to avoid recalculating on every render
  const flattenedRoutes = useMemo(() => {
    if (!routeData) return {};
    const flattened: Record<string, Coordinate[]> = {};

    for (const [routeKey, data] of Object.entries(routeData)) {
      if (data.ROUTES) {
        // Flatten all route segments into one continuous polyline
        const points: Coordinate[] = [];
        data.ROUTES.forEach(segment => {
          segment.forEach(pt => {
            points.push({ latitude: pt[0], longitude: pt[1] });
          });
        });
        flattened[routeKey] = points;
      }
    }
    return flattened;
  }, [routeData]);

  // Compute vehicle annotation props (list)
  const vehicleAnnotationProps = useMemo(() => {
    // We need map to be initialized to create mapkit.Coordinate
    if (!vehicles || !routeData || !map) return [];

    const list: AnimatedAnnotation[] = [];

    Object.keys(vehicles).forEach((key) => {
      const vehicle = vehicles[key];

      // Build SVG dynamically using ShuttleIcon component
      const routeColor = (() => {
        if (!vehicle.route_name) {
          return "#444444";
        }
        const routeKey = vehicle.route_name as keyof typeof routeData;
        const info = routeData[routeKey] as { COLOR?: string };
        return info.COLOR ?? "#444444";
      })();

      // Render ShuttleIcon JSX to a static SVG string
      const svgString = renderToStaticMarkup(<ShuttleIcon color={routeColor} size={25} />);
      const svgShuttle = `data:image/svg+xml;base64,${btoa(svgString)}`;

      // Use predicted speed if available, otherwise fall back to reported speed
      // If showTrueLocation is true, set speed to 0 to disable animation
      const displaySpeed = showTrueLocation ? 0 : (
        vehicle.predicted_location?.speed_kmh
          ? vehicle.predicted_location.speed_kmh * 0.621371  // Convert km/h to mph
          : vehicle.speed_mph
      );

      // Get route polyline
      let routePolyline: Coordinate[] | undefined;
      if (vehicle.route_name && flattenedRoutes[vehicle.route_name]) {
          routePolyline = flattenedRoutes[vehicle.route_name];
      }

      list.push({
        id: key,
        coordinate: new window.mapkit.Coordinate(vehicle.latitude, vehicle.longitude),
        title: vehicle.name,
        subtitle: `${displaySpeed.toFixed(1)} mph`,
        url: { 1: svgShuttle },
        size: { width: 25, height: 25 },
        anchorOffset: new DOMPoint(0, -13),
        
        // AnimatedAnnotation specific
        heading: vehicle.heading_degrees,
        speedMph: vehicle.speed_mph,
        predictedSpeedKmh: vehicle.predicted_location?.speed_kmh,
        timestamp: new Date(vehicle.timestamp).getTime(),
        routePolyline: routePolyline
      });
    });

    return list;
  }, [vehicles, routeData, showTrueLocation, flattenedRoutes, map]);

  return (
    <>
      <MapKitCanvas
        routeData={routeData}
        generateRoutes={generateRoutes}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        isFullscreen={isFullscreen}
        onMapReady={setMap}
      />
      <MapKitOverlays
        map={map}
        overlays={vehicleAnnotationProps}
        onAnnotationsReady={setVehicleAnnotations}
      />
      <MapKitAnimation
        annotations={vehicleAnnotationProps}
        vehicleAnnotations={vehicleAnnotations}
        showTrueLocation={showTrueLocation}
      />
    </>
  );
}