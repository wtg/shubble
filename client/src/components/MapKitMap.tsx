import { useEffect, useRef, useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import '../styles/MapKitMap.css';
import ShuttleIcon from "./ShuttleIcon";

import type { ShuttleRouteData, ShuttleStopData } from "../ts/types/route";
import '../styles/MapKitMap.css';
import type { VehicleInformationMap } from "../ts/types/vehicleLocation";
import type { Route } from "../ts/types/schedule";

import { log } from "../ts/logger";
import {
  type Coordinate,
  findNearestPointOnPolyline,
  moveAlongPolyline,
  calculateDistanceAlongPolyline,
  calculateBearing,
  getAngleDifference,
  easeInOutQuad
} from "../ts/mapUtils";

async function generateRoutePolylines(updatedRouteData: ShuttleRouteData) {
  // Use MapKit Directions API to generate polylines for each route segment
  const directions = new mapkit.Directions();

  for (const [routeName, routeInfo] of Object.entries(updatedRouteData)) {
    const polyStops = routeInfo.POLYLINE_STOPS || [];
    const realStops = routeInfo.STOPS || [];

    // Initialize ROUTES with empty arrays for each real stop segment
    routeInfo.ROUTES = Array(realStops.length - 1).fill(null).map(() => []);

    // Index of the current real stop segment we are populating
    // polyStops may include intermediate points between real stops
    let currentRealIndex = 0;

    for (let i = 0; i < polyStops.length - 1; i++) {
      // Get origin and destination stops
      const originStop = polyStops[i];
      const destStop = polyStops[i + 1];
      const originCoords = (routeInfo[originStop] as ShuttleStopData)?.COORDINATES;
      const destCoords = (routeInfo[destStop] as ShuttleStopData)?.COORDINATES;
      if (!originCoords || !destCoords) continue;

      // Fetch segment polyline
      const segment = await new Promise((resolve) => {
        directions.route(
          {
            origin: new mapkit.Coordinate(originCoords[0], originCoords[1]),
            destination: new mapkit.Coordinate(destCoords[0], destCoords[1]),
          },
          (error, data) => {
            if (error) {
              console.error(`Directions error for ${routeName} segment ${originStop}→${destStop}:`, error);
              resolve([]);
              return;
            }
            try {
              const coords = data.routes[0].polyline.points.map(pt => [pt.latitude, pt.longitude]);
              resolve(coords as [number, number][]);
            } catch (e) {
              console.error(`Unexpected response parsing for ${routeName} segment ${originStop}→${destStop}:`, e);
              resolve([]);
            }
          }
        );
      }) as [number, number][];

      // Add to the current real stop segment
      if (segment.length > 0) {
        if (routeInfo.ROUTES[currentRealIndex].length === 0) {
          // first segment for this route piece
          routeInfo.ROUTES[currentRealIndex].push(...segment);
        } else {
          // append, avoiding duplicate join point
          routeInfo.ROUTES[currentRealIndex].push(...segment.slice(1));
        }
      }

      // If the destStop is the next real stop, move to the next ROUTES index
      if (destStop === realStops[currentRealIndex + 1]) {
        currentRealIndex++;
      }
    }
  }

  // Trigger download
  function downloadJSON(data: ShuttleRouteData, filename = 'routeData.json') {
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  }

  downloadJSON(updatedRouteData);
  return updatedRouteData;
}

type MapKitMapProps = {
  routeData: ShuttleRouteData | null;
  vehicles: VehicleInformationMap | null;
  generateRoutes?: boolean;
  selectedRoute?: string | null;
  setSelectedRoute?: (route: string | null) => void;
  isFullscreen?: boolean;
};

// @ts-expect-error selectedRoutes is never used
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export default function MapKitMap({ routeData, vehicles, generateRoutes = false, selectedRoute, setSelectedRoute, isFullscreen = false }: MapKitMapProps) {
  const mapRef = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const token = import.meta.env.VITE_MAPKIT_KEY;
  const [map, setMap] = useState<(mapkit.Map | null)>(null);

  const vehicleOverlays = useRef<Record<string, mapkit.ShuttleAnnotation>>({});

  // Animation state
  const flattenedRoutesRef = useRef<Record<string, { points: Coordinate[], stopIndices: number[] }>>({});
  const vehicleAnimationStates = useRef<Record<string, {
    lastUpdateTime: number; // local time when we received the server update
    polylineIndex: number;
    currentPoint: Coordinate;
    targetDistance: number; // total distance to travel in this prediction window (meters)
    distanceTraveled: number; // distance already traveled in this window (meters)
    lastServerTime: number;
  }>>({});
  const animationFrameId = useRef<number | null>(null);


  const circleWidth = 15;
  const selectedMarkerRef = useRef<mapkit.MarkerAnnotation | null>(null);
  const overlays: mapkit.Overlay[] = [];

  // source: https://developer.apple.com/documentation/mapkitjs/loading-the-latest-version-of-mapkit-js
  const setupMapKitJs = async () => {
    if (!window.mapkit || window.mapkit.loadedLibraries.length === 0) {
      await new Promise(resolve => { window.initMapKit = () => resolve(null); });
      delete window.initMapKit;
    }
  };

  useEffect(() => {
    // initialize mapkit
    const mapkitScript = async () => {
      // load the MapKit JS library
      await setupMapKitJs();
      mapkit.init({
        authorizationCallback: (done) => {
          done(token);
        },
      });
      setMapLoaded(true);
    };
    mapkitScript();
  }, []);

  // create the map
  useEffect(() => {
    if (mapLoaded) {

      // center on RPI
      const center = new mapkit.Coordinate(42.730216, -73.675690);
      const span = new mapkit.CoordinateSpan(0.02, 0.005);
      const region = new mapkit.CoordinateRegion(center, span);

      const mapOptions = {
        center: center,
        region: region,
        isScrollEnabled: true,
        isZoomEnabled: true,
        showsZoomControl: true,
        isRotationEnabled: false,
        showsPointsOfInterest: false,
        showsUserLocation: true,
      };

      // create the map
      const thisMap = new mapkit.Map(mapRef.current!, mapOptions);
      // set zoom and boundary limits
      thisMap.setCameraZoomRangeAnimated(
        new mapkit.CameraZoomRange(200, 3000),
        false,
      );
      thisMap.setCameraBoundaryAnimated(
        new mapkit.CoordinateRegion(
          center,
          new mapkit.CoordinateSpan(0.02, 0.025)
        ),
        false,
      );
      thisMap.setCameraDistanceAnimated(2500);
      // Helper function to create and add stop marker
      const createStopMarker = (overlay: mapkit.CircleOverlay) => {
        if (selectedMarkerRef.current) {
          thisMap.removeAnnotation(selectedMarkerRef.current);
          selectedMarkerRef.current = null;
        }
        const marker = new mapkit.MarkerAnnotation(overlay.coordinate, {
          title: overlay.stopName,
          glyphImage: { 1: "map-marker.png" },
        });
        thisMap.addAnnotation(marker);
        selectedMarkerRef.current = marker;
        return marker;
      };

      thisMap.addEventListener("select", (e) => {
        if (!e.overlay) return;
        if (!(e.overlay instanceof mapkit.CircleOverlay)) return;

        // Only change schedule selection on desktop-sized screens
        const isDesktop = window.matchMedia('(min-width: 800px)').matches;

        if (e.overlay.stopKey) {
          // Create marker for both mobile and desktop
          createStopMarker(e.overlay);

          if (isDesktop) {
            // Desktop: handle schedule change
            const routeKey = e.overlay.routeKey;
            if (setSelectedRoute && routeKey) setSelectedRoute(routeKey);
          }
        }
      });
      thisMap.addEventListener("deselect", () => {
        // remove any selected stop/marker annotation on when deselected
        if (selectedMarkerRef.current) {
          thisMap.removeAnnotation(selectedMarkerRef.current);
          selectedMarkerRef.current = null;
        }
      });

      thisMap.addEventListener("region-change-start", () => {
        (thisMap.element as HTMLElement).style.cursor = "grab";
      });

      thisMap.addEventListener("region-change-end", () => {
        (thisMap.element as HTMLElement).style.cursor = "default";
      });

      // Working hover detection
      let currentHover: mapkit.CircleOverlay | null = null;
      thisMap.element.addEventListener('mousemove', (e) => {
        const rect = thisMap.element.getBoundingClientRect();
        const x = (e as MouseEvent).clientX - rect.left;
        const y = (e as MouseEvent).clientY - rect.top;

        let foundOverlay: mapkit.CircleOverlay | null = null;

        // Check overlays for mouse position
        for (const overlay of thisMap.overlays) {
          if (!(overlay instanceof mapkit.CircleOverlay)) continue;
          if (overlay.stopKey) {
            // Calculate overlay screen position
            const mapRect = thisMap.element.getBoundingClientRect();
            const centerLat = overlay.coordinate.latitude;
            const centerLng = overlay.coordinate.longitude;

            // Check if mouse is within overlay radius
            const region = thisMap.region;
            if (region) {
              const centerX = mapRect.width * (centerLng - region.center.longitude + region.span.longitudeDelta / 2) / region.span.longitudeDelta;
              const centerY = mapRect.height * (region.center.latitude - centerLat + region.span.latitudeDelta / 2) / region.span.latitudeDelta;

              const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
              if (distance < circleWidth) { // Within hover radius
                foundOverlay = overlay;
              }
            }
          }
        }

        if (foundOverlay !== currentHover) {
          // Clear previous hover style
          if (currentHover) {
            currentHover.style = new mapkit.Style({
              strokeColor: '#000000',
              fillColor: '#FFFFFF',
              fillOpacity: 0.1,
              lineWidth: 2,
            });
          }

          // Apply hover style
          if (foundOverlay) {
            foundOverlay.style = new mapkit.Style({
              strokeColor: '#6699ff',
              fillColor: '#a1c3ff',
              fillOpacity: 0.3,
              lineWidth: 2.5,
            });
            (thisMap.element as HTMLElement).style.cursor = "pointer";
          } else {
            (thisMap.element as HTMLElement).style.cursor = "default";
          }

          currentHover = foundOverlay;
        }
      });

      // Store reference to cleanup function
      thisMap._hoverCleanup = () => {
        // thisMap.element.removeEventListener('mousemove', _);
      };

      setMap(thisMap);
    }

    // Cleanup on component unmount
    return () => {
      if (map && map._hoverCleanup) {
        map._hoverCleanup();
      }
    };
  }, [mapLoaded]);

  // add fixed details to the map
  // includes routes and stops
  useEffect(() => {
    if (!map || !routeData) return;


    // display stop overlays
    for (const [route, thisRouteData] of Object.entries(routeData)) {
      for (const stopKey of thisRouteData.STOPS) {
        const stopData = thisRouteData[stopKey] as ShuttleStopData;
        const stopCoordinate = new mapkit.Coordinate(...(stopData.COORDINATES));
        // add stop overlay (circle)
        const stopOverlay = new mapkit.CircleOverlay(
          stopCoordinate,
          circleWidth,
          {
            style: new mapkit.Style(
              {
                strokeColor: '#000000',
                fillColor: '#FFFFFF', // White fill by default
                fillOpacity: 0.1,
                lineWidth: 2,
              }
            )
          }
        );
        // attach exact identifiers so the select handler can update selection precisely
        stopOverlay.routeKey = route;
        stopOverlay.stopKey = stopKey;
        stopOverlay.stopName = stopData.NAME;
        // cast circle overlay to generic overlay for adding to map
        overlays.push(stopOverlay as mapkit.Overlay);
      }
    }

    function displayRouteOverlays(routeData: ShuttleRouteData) {
      // display route overlays
      for (const [_route, thisRouteData] of Object.entries(routeData)) {
        // for route (WEST, NORTH)
        const routePolylines = thisRouteData.ROUTES?.map(
          // for segment (STOP1 -> STOP2, STOP2 -> STOP3, ...)
          (route) => {
            const coords = route.map(([lat, lon]) => new mapkit.Coordinate(lat, lon));
            if (coords.length === 0) return null;
            const polyline = new mapkit.PolylineOverlay(coords, {
              // for coordinate ([lat, lon], ...)
              style: new mapkit.Style({
                strokeColor: thisRouteData.COLOR,
                lineWidth: 2
              })
            });
            return polyline;
          }
        ).filter(p => p !== null);
        overlays.push(...routePolylines as mapkit.Overlay[]);
      }
    }

    if (generateRoutes) {
      // generate polylines for routes
      const routeDataCopy = JSON.parse(JSON.stringify(routeData)); // deep copy to avoid mutating original
      generateRoutePolylines(routeDataCopy).then((updatedRouteData) => {
        displayRouteOverlays(updatedRouteData);
        map.addOverlays(overlays);
      });
    } else {
      // use pre-generated polylines
      displayRouteOverlays(routeData);
      map.addOverlays(overlays);
    }

  }, [map, routeData]);

  // Flatten routes for animation usage whenever routeData changes
  useEffect(() => {
    if (!routeData) return;
    const flattened: Record<string, { points: Coordinate[], stopIndices: number[] }> = {};

    for (const [routeKey, data] of Object.entries(routeData)) {
      if (data.ROUTES) {
        // ROUTES is array of segments (coordinate arrays). Flatten them into one long line.
        // Each point is [lat, lon]
        const points: Coordinate[] = [];
        const stopIndices: number[] = [];

        // The first stop is at index 0
        stopIndices.push(0);

        data.ROUTES.forEach(segment => {
          segment.forEach(pt => {
            points.push({ latitude: pt[0], longitude: pt[1] });
          });
          // The end of this segment corresponds to the next stop location
          // (or extremely close to it).
          // The current length of 'points' is the index of the start of the *next* segment?
          // Actually, points.push appends. So points.length - 1 is the last point.
          // If the segments are continuous, the last point of Segment A might be the first point of Segment B
          // depending on how they were fetched.
          // Looking at generateRoutePolylines: `segment.slice(1)` is pushed for subsequent segments.
          // So points are unique.

          stopIndices.push(points.length - 1);
        });

        flattened[routeKey] = { points, stopIndices };
      }
    }
    flattenedRoutesRef.current = flattened;
  }, [routeData]);

  // display vehicles on map
  useEffect(() => {
    if (!map || !vehicles) return;

    Object.keys(vehicles).forEach((key) => {
      const vehicle = vehicles[key];
      const coordinate = new window.mapkit.Coordinate(vehicle.latitude, vehicle.longitude);

      const existingAnnotation = vehicleOverlays.current[key];

      // Build SVG dynamically using ShuttleIcon component
      const routeColor = (() => {
        if (!routeData || !vehicle.route_name || vehicle.route_name === "UNCLEAR") {
          return "#444444";
        }
        const routeKey = vehicle.route_name as keyof typeof routeData;
        const info = routeData[routeKey] as { COLOR?: string };
        return info.COLOR ?? "#444444";

      })();

      // Render ShuttleIcon JSX to a static SVG string
      const svgString = renderToStaticMarkup(<ShuttleIcon color={routeColor} size={25} />);
      const svgShuttle = `data:image/svg+xml;base64,${btoa(svgString)}`;

      // --- Update or create annotation ---
      if (existingAnnotation) {
        // existing vehicle — update position and subtitle
        // Only update coordinate directly if we don't have an animation state (to avoid flicker)
        if (!vehicleAnimationStates.current[key]) {
          existingAnnotation.coordinate = coordinate;
        }
        existingAnnotation.subtitle = `${vehicle.speed_mph.toFixed(1)} mph`;

        // Handle route status updates
        // If shuttle does not have a route null 
        if (vehicle.route_name === null) {
          // shuttle off-route (exiting)
          if (existingAnnotation.lockedRoute) {
            existingAnnotation.lockedRoute = null;
            existingAnnotation.url = { 1: svgShuttle };
          }
        } else if (vehicle.route_name !== "UNCLEAR" && vehicle.route_name !== existingAnnotation.lockedRoute) {
          existingAnnotation.lockedRoute = vehicle.route_name;
          existingAnnotation.url = { 1: svgShuttle };
        }
      } else {
        const annotationOptions = {
          title: vehicle.name,
          subtitle: `${vehicle.speed_mph.toFixed(1)} mph`,
          url: { 1: svgShuttle },
          size: { width: 25, height: 25 },
          anchorOffset: new DOMPoint(0, -13),
        };

        // create shuttle object
        const annotation = new window.mapkit.ImageAnnotation(coordinate, annotationOptions) as mapkit.ShuttleAnnotation;


        // lock route if known
        if (vehicle.route_name !== "UNCLEAR" && vehicle.route_name !== null) {
          annotation.lockedRoute = vehicle.route_name;
        }

        // add shuttle to map
        map.addAnnotation(annotation);
        vehicleOverlays.current[key] = annotation;
      }
    });

    // --- Update Animation State for new/updated vehicles ---
    const now = Date.now();
    Object.keys(vehicles).forEach((key) => {
      const vehicle = vehicles[key];
      // If we don't have a route for this vehicle, we can't animate along a path nicely. 
      // We'll just rely on the API updates or maybe simple linear extrapolation later?
      // For now, let's only set up animation if we have a valid route.
      if (!vehicle.route_name || !flattenedRoutesRef.current[vehicle.route_name]) return;

      const routeData = flattenedRoutesRef.current[vehicle.route_name];
      const routePolyline = routeData.points;
      const vehicleCoord = { latitude: vehicle.latitude, longitude: vehicle.longitude };

      const serverTime = new Date(vehicle.timestamp).getTime();

      // Check if we already have state
      let animState = vehicleAnimationStates.current[key];

      // If the server data hasn't changed (cached response), ignore this update
      // and let the client-side prediction continue running.
      if (animState && animState.lastServerTime === serverTime) {
        return;
      }

      const snapToPolyline = () => {
        const { index, point } = findNearestPointOnPolyline(vehicleCoord, routePolyline);
        vehicleAnimationStates.current[key] = {
          lastUpdateTime: now,
          polylineIndex: index,
          currentPoint: point,
          targetDistance: 0,
          distanceTraveled: 0,
          lastServerTime: serverTime
        };
      };

      if (!animState) {
        snapToPolyline();
      } else {
        // =======================================================================
        // PREDICTION SMOOTHING ALGORITHM
        // =======================================================================
        // Problem: Server updates arrive every ~5 seconds, causing the shuttle
        // to "jump" to its new position (rubberbanding).
        //
        // Solution: Instead of jumping, we calculate the speed needed for the
        // shuttle to smoothly travel from its current visual position to where
        // it *should* be when the next update arrives.
        //
        // Formula: speed = distance / time
        // Where:
        //   - distance = gap between current visual position and predicted target
        //   - time = 5 seconds (the update interval)
        // =======================================================================

        const PREDICTION_WINDOW_SECONDS = 5;

        // Step 1: Find where the server says the shuttle is right now
        const { index: serverIndex, point: serverPoint } = findNearestPointOnPolyline(vehicleCoord, routePolyline);

        // Step 2: Calculate where the shuttle will be in 5 seconds
        // Convert speed from mph to meters/second (1 mph = 0.44704 m/s)
        const speedMetersPerSecond = vehicle.speed_mph * 0.44704;
        const projectedDistanceMeters = speedMetersPerSecond * PREDICTION_WINDOW_SECONDS;

        // Move along the route polyline by that distance to find the target point
        const { index: targetIndex, point: targetPoint } = moveAlongPolyline(
          routePolyline,
          serverIndex,
          serverPoint,
          projectedDistanceMeters
        );

        // Step 3: Verify the shuttle is moving in the correct direction
        // Compare the vehicle's GPS heading to the route segment bearing.
        // If they differ by more than 90°, the shuttle may be going the wrong way.
        let isMovingCorrectDirection = true;
        if (routePolyline.length > serverIndex + 1 && vehicle.speed_mph > 1) {
          const segmentStart = routePolyline[serverIndex];
          const segmentEnd = routePolyline[serverIndex + 1];
          const segmentBearing = calculateBearing(segmentStart, segmentEnd);
          const headingDifference = getAngleDifference(segmentBearing, vehicle.heading_degrees);

          if (headingDifference > 90) {
            isMovingCorrectDirection = false;
          }
        }

        // Step 4: Calculate distance from current visual position to target
        const distanceToTarget = calculateDistanceAlongPolyline(
          routePolyline,
          animState.polylineIndex,
          animState.currentPoint,
          targetIndex,
          targetPoint
        );

        // Step 5: Calculate the total distance to travel with easing
        let targetDistanceMeters = distanceToTarget;

        // If moving wrong direction, stop the animation
        if (!isMovingCorrectDirection) {
          targetDistanceMeters = 0;
        }

        // Step 6: Update animation state.
        // If the gap is extremely large (>250m), something is wrong (route change, bad data).
        // In that case, snap immediately to the server position instead of animating.
        const MAX_REASONABLE_GAP_METERS = 250;
        if (distanceToTarget > MAX_REASONABLE_GAP_METERS) {
          snapToPolyline();
        } else {
          // Reset the animation progress - we're starting a new prediction window
          vehicleAnimationStates.current[key] = {
            lastUpdateTime: now,
            polylineIndex: animState.polylineIndex,
            currentPoint: animState.currentPoint,
            targetDistance: targetDistanceMeters,
            distanceTraveled: 0,
            lastServerTime: serverTime
          };
        }
      }
    });

    // --- Remove stale vehicles ---
    const currentVehicleKeys = new Set(Object.keys(vehicles));
    Object.keys(vehicleOverlays.current).forEach((key) => {
      if (!currentVehicleKeys.has(key)) {
        map.removeAnnotation(vehicleOverlays.current[key]);
        delete vehicleOverlays.current[key];
      }
    });
  }, [map, vehicles, routeData]);


  // --- Animation Loop ---
  useEffect(() => {
    // We use setTimeout/setInterval or requestAnimationFrame. The user "Considered" setTimeout.
    // We will use requestAnimationFrame for smoothness, but structure it to calculate delta
    // similar to how one might with setTimeout.

    let lastFrameTime = Date.now();

    const animate = () => {
      const now = Date.now();
      const dt = now - lastFrameTime; // ms
      lastFrameTime = now;

      // Avoid huge jumps if tab was backgrounded
      if (dt > 1000) {
        animationFrameId.current = requestAnimationFrame(animate);
        return;
      }

      Object.keys(vehicleAnimationStates.current).forEach(key => {
        const animState = vehicleAnimationStates.current[key];
        const vehicle = vehicles?.[key];
        const annotation = vehicleOverlays.current[key];

        if (!vehicle || !annotation || !animState) return;
        if (!vehicle.route_name || !flattenedRoutesRef.current[vehicle.route_name]) return;

        const routePolyline = flattenedRoutesRef.current[vehicle.route_name].points;

        // =======================================================================
        // EASED ANIMATION
        // =======================================================================
        // Instead of constant speed, we use an ease-in-out curve.
        // This makes the shuttle accelerate at the start and decelerate at the end
        // of each prediction window, creating smoother, more natural motion.
        // =======================================================================

        const PREDICTION_WINDOW_MS = 5000;
        const timeElapsed = now - animState.lastUpdateTime;

        // Calculate progress through the prediction window (0.0 to 1.0)
        const linearProgress = Math.min(timeElapsed / PREDICTION_WINDOW_MS, 1.0);

        // Apply easing function for smooth acceleration/deceleration
        const easedProgress = easeInOutQuad(linearProgress);

        // Calculate how far along the target distance we should be
        const targetPosition = animState.targetDistance * easedProgress;

        // Calculate how much to move this frame
        const distanceToMove = targetPosition - animState.distanceTraveled;

        if (distanceToMove <= 0) return;

        // Move along polyline
        const { index, point } = moveAlongPolyline(
          routePolyline,
          animState.polylineIndex,
          animState.currentPoint,
          distanceToMove
        );

        // Update state
        animState.polylineIndex = index;
        animState.currentPoint = point;
        animState.distanceTraveled = targetPosition;

        // Update MapKit annotation
        annotation.coordinate = new mapkit.Coordinate(point.latitude, point.longitude);
      });

      animationFrameId.current = requestAnimationFrame(animate);
    };

    animationFrameId.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    };
  }, [vehicles]); // Restart loop if vehicles change? Not strictly necessary if refs are used, but ensures we have latest `vehicles` closure if needed. Actually with refs we don't need to dependency on vehicles often if we read from ref, but here we read `vehicles` prop.



  return (
    <div
      className={isFullscreen ? 'map-fullscreen' : 'map'}
      ref={mapRef}
    >
    </div>
  );
};
