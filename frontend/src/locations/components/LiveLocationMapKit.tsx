import { useEffect, useState, useMemo } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import '../styles/MapKitMap.css';
import ShuttleIcon from "./ShuttleIcon";
import config from "../../utils/config";

import type { ShuttleRouteData } from "../../types/route";
import type { VehicleLocationMap, VehicleVelocityMap, VehicleCombinedMap } from "../../types/vehicleLocation";
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
};

export default function LiveLocationMapKit({
  routeData,
  displayVehicles = true,
  generateRoutes = false,
  selectedRoute,
  setSelectedRoute,
  isFullscreen = false,
  showTrueLocation = true
}: LiveLocationMapKitProps) {
  const [map, setMap] = useState<(mapkit.Map | null)>(null);
  const [vehicles, setVehicles] = useState<VehicleCombinedMap | null>(null);
  const [vehicleAnnotations, setVehicleAnnotations] = useState<Record<string, mapkit.Annotation>>({});

  // Fetch location and velocity data on component mount and set up polling
  useEffect(() => {
    if (!displayVehicles) return;

    const pollLocation = async () => {
      try {
        // Fetch locations and velocities in parallel
        const [locationsResponse, velocitiesResponse] = await Promise.all([
          fetch(`${config.apiBaseUrl}/api/locations`, { cache: 'no-store' }),
          fetch(`${config.apiBaseUrl}/api/velocities`, { cache: 'no-store' })
        ]);

        if (!locationsResponse.ok) {
          throw new Error('Failed to fetch locations');
        }

        const locationsData: VehicleLocationMap = await locationsResponse.json();

        // Velocities are optional - don't fail if they're unavailable
        let velocitiesData: VehicleVelocityMap = {};
        if (velocitiesResponse.ok) {
          velocitiesData = await velocitiesResponse.json();
        }

        // Merge location and velocity data
        const combined: VehicleCombinedMap = {};
        for (const [vehicleId, location] of Object.entries(locationsData)) {
          const velocity = velocitiesData[vehicleId];
          combined[vehicleId] = {
            ...location,
            route_name: velocity?.route_name ?? null,
            polyline_index: velocity?.polyline_index ?? null,
            predicted_location: velocity && velocity.speed_kmh !== null && velocity.timestamp !== null ? {
              speed_kmh: velocity.speed_kmh,
              timestamp: velocity.timestamp
            } : undefined,
            is_at_stop: velocity?.is_at_stop,
            current_stop: velocity?.current_stop,
          };
        }

        setVehicles(combined);
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
