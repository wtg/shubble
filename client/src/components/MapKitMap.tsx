import { useEffect, useRef, useState } from "react";
import type { ShuttleRouteData, ShuttleStopData } from "../ts/types/route";
import '../styles/MapKitMap.css';
import type { VehicleInformationMap } from "../ts/types/vehicleLocation";
import type { Route } from "../ts/types/schedule";

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
  selectedStop?: string;
  setSelectedStop?: (stop: string) => void;
};

// @ts-expect-error selectedRoutes is never used
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export default function MapKitMap({ routeData, vehicles, generateRoutes = false, selectedRoute, setSelectedRoute, selectedStop, setSelectedStop }: MapKitMapProps) {

  const mapRef = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const token = import.meta.env.VITE_MAPKIT_KEY;
  const [map, setMap] = useState<(mapkit.Map | null)>(null);
  const vehicleOverlays = useRef<Record<string, mapkit.MarkerAnnotation>>({});
  const circleWidth = 15;
  const selectedMarkerRef = useRef<mapkit.MarkerAnnotation | null>(null);
  const overlays: mapkit.Overlay[] = [];


  // source: https://developer.apple.com/documentation/mapkitjs/loading-the-latest-version-of-mapkit-js
  const setupMapKitJs = async () => {
    if (!mapkit) {
      await new Promise(resolve => { window.initMapKit = resolve });
      delete window.initMapKit;
    }
  };

  useEffect(() => {
    // initialize mapkit
    const mapkitScript = async () => {
      // load the MapKit JS library
      await setupMapKitJs();
      mapkit.init({
        authorizationCallback: (done: (token: string) => void) => {
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
            const stopKey = e.overlay.stopKey;
            if (setSelectedRoute && routeKey) setSelectedRoute(routeKey);
            if (setSelectedStop && stopKey) setSelectedStop(stopKey);
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
              const centerX = mapRect.width * (centerLng - region.center.longitude + region.span.longitudeDelta/2) / region.span.longitudeDelta;
              const centerY = mapRect.height * (region.center.latitude - centerLat + region.span.latitudeDelta/2) / region.span.latitudeDelta;

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
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
        overlays.push(...routePolylines);
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

  // display vehicles on map
  useEffect(() => {
    if (!map || !vehicles) return;

    Object.keys(vehicles).map((key) => {
      const vehicle = vehicles[key];
      const coordinate = new mapkit.Coordinate(vehicle.latitude, vehicle.longitude);
      if (key in vehicleOverlays.current) {
        // old vehicle: update coordinate
        console.log(`Updating vehicle ${key} to ${vehicle.latitude}, ${vehicle.longitude}`);
        vehicleOverlays.current[key].coordinate = coordinate;
        vehicleOverlays.current[key].subtitle = `${vehicle.speed_mph} mph`;
      } else {
        // new vehicle: add to map
        console.log(`Adding vehicle ${key} to ${vehicle.latitude}, ${vehicle.longitude}`);
        const annotation = new mapkit.MarkerAnnotation(coordinate, {
          title: vehicle.name,
          subtitle: `${vehicle.speed_mph} mph`,
          color: vehicle.route_name && vehicle.route_name !== "UNCLEAR" && routeData ? routeData[vehicle.route_name as Route].COLOR : '#444444',
          glyphImage: { 1: 'shubble20.png' },
          selectedGlyphImage: { 1: 'shubble20.png', 2: 'shubble40.png' },
        });
        map.addAnnotation(annotation);
        vehicleOverlays.current[key] = annotation;
      }

      if (vehicle.route_name !== "UNCLEAR") {
        vehicleOverlays.current[key].color = vehicle.route_name && routeData ? routeData[vehicle.route_name as Route].COLOR : '#444444';
      }
    });

    const currentVehicleKeys = new Set(Object.keys(vehicles));

    // Remove vehicles no longer in response
    Object.keys(vehicleOverlays.current).forEach((key) => {
      if (!currentVehicleKeys.has(key)) {
        console.log(`Removing vehicle ${key}`);
        map.removeAnnotation(vehicleOverlays.current[key]);
        delete vehicleOverlays.current[key];
      }
    });

  }, [map, vehicles]);

  return (
    <div
      className='map'
      ref={mapRef}
    >
    </div>
  );
};
