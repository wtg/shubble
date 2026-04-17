import { useEffect, useState, useMemo } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import '../styles/MapKitMap.css';
import ShuttleIcon from "./ShuttleIcon";
import config from "../../utils/config";

// PERF: SVG shuttle icons only depend on (color, size), and both rarely
// change. Cache the rendered data-URL so we don't pay the
// renderToStaticMarkup + btoa cost per vehicle per 5s poll tick.
const _shuttleIconCache = new Map<string, string>();
function getShuttleIconUrl(color: string, size: number): string {
  const key = `${color}|${size}`;
  const cached = _shuttleIconCache.get(key);
  if (cached) return cached;
  const svg = renderToStaticMarkup(<ShuttleIcon color={color} size={size} />);
  const url = `data:image/svg+xml;base64,${btoa(svg)}`;
  _shuttleIconCache.set(key, url);
  return url;
}

import type { ShuttleRouteData } from "../../types/route";
import type { VehicleLocationMap, VehicleVelocityMap, VehicleCombinedMap } from "../../types/vehicleLocation";
import type { Coordinate } from "../../utils/mapUtils";
import type { StopETAs, StopETADetails } from "../../hooks/useTrips";

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
  shuttleIconSize?: number;
  stopETAs?: StopETAs;
  stopETADetails?: StopETADetails;
};

export default function LiveLocationMapKit({
  routeData,
  displayVehicles = true,
  generateRoutes = false,
  selectedRoute,
  setSelectedRoute,
  isFullscreen = false,
  showTrueLocation = true,
  shuttleIconSize = 25,
  stopETAs,
  stopETADetails,
}: LiveLocationMapKitProps) {
  const [map, setMap] = useState<(mapkit.Map | null)>(null);
  const [vehicles, setVehicles] = useState<VehicleCombinedMap | null>(null);
  const [vehicleAnnotations, setVehicleAnnotations] = useState<Record<string, mapkit.Annotation>>({});

  // Fetch location and velocity data on component mount.
  //
  // Transport: SSE push via `/api/locations/stream` is the primary
  // trigger. Every time the worker commits new GPS rows it publishes
  // `shubble:locations_updated` on Redis; the SSE handler forwards that
  // to this client. On each trigger we re-fetch /api/locations AND
  // /api/velocities (velocities has no stream yet — they're updated at
  // the same worker cycle, so re-fetching together keeps them in sync).
  //
  // Fallback: if the stream endpoint is unreachable or the browser has
  // no EventSource, fall back to setInterval polling at 3s.
  useEffect(() => {
    if (!displayVehicles) return;

    let abortController: AbortController | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let es: EventSource | null = null;
    let cancelled = false;

    const pollLocation = async () => {
      if (cancelled) return;
      // Cancel any in-flight request before starting a new one
      if (abortController) {
        abortController.abort();
      }
      abortController = new AbortController();
      const { signal } = abortController;

      try {
        // Fetch locations and velocities in parallel
        const [locationsResponse, velocitiesResponse] = await Promise.all([
          fetch(`${config.apiBaseUrl}/api/locations`, { cache: 'no-store', signal }),
          fetch(`${config.apiBaseUrl}/api/velocities`, { cache: 'no-store', signal })
        ]);

        if (!locationsResponse.ok) {
          throw new Error('Failed to fetch locations');
        }

        const locationsData: VehicleLocationMap = await locationsResponse.json() as VehicleLocationMap;

        // Velocities are optional - don't fail if they're unavailable
        let velocitiesData: VehicleVelocityMap = {};
        if (velocitiesResponse.ok) {
          velocitiesData = await velocitiesResponse.json() as VehicleVelocityMap;
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

        if (!cancelled) setVehicles(combined);
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        console.error('Error fetching location:', error);
      }
    };

    const startPolling = () => {
      if (pollTimer !== null || cancelled) return;
      pollTimer = setInterval(() => { void pollLocation(); }, 3000);
    };

    const stopPolling = () => {
      if (pollTimer !== null) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };

    // Kick off an initial fetch so the map isn't blank while SSE opens.
    void pollLocation();

    // Try SSE push first; fall back to polling on permanent failure.
    if (typeof EventSource !== 'undefined') {
      try {
        es = new EventSource(`${config.apiBaseUrl}/api/locations/stream`);
        es.onopen = () => { stopPolling(); };
        es.onmessage = () => {
          // Trigger-only: SSE payload is informational. Always go
          // through pollLocation() so locations + velocities stay in
          // sync via a single code path.
          void pollLocation();
        };
        es.onerror = () => {
          if (es?.readyState === EventSource.CLOSED) {
            startPolling();
          }
        };
      } catch {
        startPolling();
      }
    } else {
      startPolling();
    }

    return () => {
      cancelled = true;
      stopPolling();
      if (es) {
        es.close();
        es = null;
      }
      if (abortController) abortController.abort();
    };

  // eslint-disable-next-line react-hooks/exhaustive-deps
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

      // Cached SVG data URL — renders once per unique (color, size).
      const svgShuttle = getShuttleIconUrl(routeColor, shuttleIconSize);

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
        size: { width: shuttleIconSize, height: shuttleIconSize },
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
  }, [vehicles, routeData, map, shuttleIconSize, showTrueLocation, flattenedRoutes]);

  return (
    <>
      <MapKitCanvas
        routeData={routeData}
        generateRoutes={generateRoutes}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        isFullscreen={isFullscreen}
        onMapReady={setMap}
        stopETAs={stopETAs}
        stopETADetails={stopETADetails}
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
