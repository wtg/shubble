import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import MapKitOverlays, {
  syncShuttleAnnotations,
} from "../../mapkit/MapKitOverlays";
import type {
  RenderedShuttleAnnotationStore,
  ShuttleAnnotationRecord,
} from "../../mapkit/MapKitOverlays";

type LiveLocationMapKitProps = {
  routeData: ShuttleRouteData | null;
  displayVehicles?: boolean;
  generateRoutes?: boolean;
  selectedRoute?: string | null;
  setSelectedRoute?: (route: string | null) => void;
  isFullscreen?: boolean;
  showTrueLocation?: boolean;
  shuttleIconSize?: number;
  /** Set of "ROUTE:STOP_KEY" identifiers for stops that are currently inactive */
  inactiveStops?: Set<string>;
  /** Current scrollTop of the schedule panel (for time-based map interactions) */
  scheduleScrollY?: number;
};

type BuildVehicleAnnotationOptions = {
  flattenedRoutes: Record<string, Coordinate[]>;
  mapkitReady: boolean;
  routeData: ShuttleRouteData | null;
  shuttleIconSize: number;
  showTrueLocation: boolean;
  vehicles: VehicleCombinedMap | null;
};

function buildVehicleAnnotationProps({
  flattenedRoutes,
  mapkitReady,
  routeData,
  shuttleIconSize,
  showTrueLocation,
  vehicles,
}: BuildVehicleAnnotationOptions): AnimatedAnnotation[] {
  if (!mapkitReady || !vehicles || !routeData || typeof window === 'undefined' || !window.mapkit) {
    return [];
  }

  const list: AnimatedAnnotation[] = [];

  Object.keys(vehicles).forEach((key) => {
    const vehicle = vehicles[key];

    const routeColor = (() => {
      if (!vehicle.route_name) {
        return "#444444";
      }
      const routeKey = vehicle.route_name as keyof typeof routeData;
      const info = routeData[routeKey] as { COLOR?: string };
      return info.COLOR ?? "#444444";
    })();

    const svgString = renderToStaticMarkup(<ShuttleIcon color={routeColor} size={shuttleIconSize} />);
    const svgShuttle = `data:image/svg+xml;base64,${btoa(svgString)}`;

    const displaySpeed = showTrueLocation ? 0 : (
      vehicle.predicted_location?.speed_kmh
        ? vehicle.predicted_location.speed_kmh * 0.621371
        : vehicle.speed_mph
    );

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
      size: { width: shuttleIconSize, height: shuttleIconSize },
      anchorOffset: new DOMPoint(0, -13),

      heading: vehicle.heading_degrees,
      speedMph: vehicle.speed_mph,
      predictedSpeedKmh: vehicle.predicted_location?.speed_kmh,
      timestamp: new Date(vehicle.timestamp).getTime(),
      routePolyline,
    });
  });

  return list;
}

export default function LiveLocationMapKit({
  routeData,
  displayVehicles = true,
  generateRoutes = false,
  selectedRoute,
  setSelectedRoute,
  isFullscreen = false,
  showTrueLocation = true,
  shuttleIconSize = 25,
  inactiveStops,
}: LiveLocationMapKitProps) {
  const [map, setMap] = useState<mapkit.Map | null>(null);
  const [isMapKitReady, setIsMapKitReady] = useState(false);
  const [vehicles, setVehicles] = useState<VehicleCombinedMap | null>(null);
  const [vehicleAnnotations, setVehicleAnnotations] = useState<ShuttleAnnotationRecord>({});
  const renderedVehicleAnnotationsByKey = useRef<RenderedShuttleAnnotationStore>(new Map());

  useEffect(() => {
    if (!displayVehicles) return;

    let abortController: AbortController | null = null;

    const pollLocation = async () => {
      if (abortController) {
        abortController.abort();
      }
      abortController = new AbortController();
      const { signal } = abortController;

      try {
        const [locationsResponse, velocitiesResponse] = await Promise.all([
          fetch(`${config.apiBaseUrl}/api/locations`, { cache: 'no-store', signal }),
          fetch(`${config.apiBaseUrl}/api/velocities`, { cache: 'no-store', signal })
        ]);

        if (!locationsResponse.ok) {
          throw new Error('Failed to fetch locations');
        }

        const locationsData: VehicleLocationMap = await locationsResponse.json() as VehicleLocationMap;

        let velocitiesData: VehicleVelocityMap = {};
        if (velocitiesResponse.ok) {
          velocitiesData = await velocitiesResponse.json() as VehicleVelocityMap;
        }

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
        if (error instanceof DOMException && error.name === 'AbortError') return;
        console.error('Error fetching location:', error);
      }
    };

    pollLocation();

    const refreshLocation = setInterval(pollLocation, 5000);

    return () => {
      clearInterval(refreshLocation);
      if (abortController) abortController.abort();
    };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const flattenedRoutes = useMemo(() => {
    if (!routeData) return {};
    const flattened: Record<string, Coordinate[]> = {};

    for (const [routeKey, data] of Object.entries(routeData)) {
      if (data.ROUTES) {
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

  const vehicleAnnotationProps = useMemo(() => {
    return buildVehicleAnnotationProps({
      flattenedRoutes,
      mapkitReady: isMapKitReady,
      routeData,
      shuttleIconSize,
      showTrueLocation,
      vehicles,
    });
  }, [flattenedRoutes, isMapKitReady, routeData, shuttleIconSize, showTrueLocation, vehicles]);

  const handleMapReady = useCallback((readyMap: mapkit.Map) => {
    setMap(readyMap);
    setIsMapKitReady(true);

    syncShuttleAnnotations({
      map: readyMap,
      overlays: buildVehicleAnnotationProps({
        flattenedRoutes,
        mapkitReady: true,
        routeData,
        shuttleIconSize,
        showTrueLocation,
        vehicles,
      }),
      renderedAnnotationsByKey: renderedVehicleAnnotationsByKey.current,
      onAnnotationsReady: setVehicleAnnotations,
    });
  }, [flattenedRoutes, routeData, shuttleIconSize, showTrueLocation, vehicles]);

  return (
    <>
      <MapKitCanvas
        routeData={routeData}
        generateRoutes={generateRoutes}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        isFullscreen={isFullscreen}
        onMapReady={handleMapReady}
        inactiveStops={inactiveStops}
      />
      <MapKitOverlays
        map={map}
        overlays={vehicleAnnotationProps}
        onAnnotationsReady={setVehicleAnnotations}
        renderedAnnotationsByKey={renderedVehicleAnnotationsByKey}
      />
      <MapKitAnimation
        annotations={vehicleAnnotationProps}
        vehicleAnnotations={vehicleAnnotations}
        showTrueLocation={showTrueLocation}
      />
    </>
  );
}
