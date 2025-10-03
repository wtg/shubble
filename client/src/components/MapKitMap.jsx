import { useEffect, useRef, useState } from "react";
import '../styles/MapKitMap.css';

async function generateRoutePolylines(updatedRouteData) {
  // Use MapKit Directions API to generate polylines for each route segment
  const directions = new window.mapkit.Directions();

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
      const originCoords = routeInfo[originStop]?.COORDINATES;
      const destCoords = routeInfo[destStop]?.COORDINATES;
      if (!originCoords || !destCoords) continue;

      // Fetch segment polyline
      const segment = await new Promise((resolve) => {
        directions.route(
          {
            origin: new window.mapkit.Coordinate(originCoords[0], originCoords[1]),
            destination: new window.mapkit.Coordinate(destCoords[0], destCoords[1]),
          },
          (error, data) => {
            if (error) {
              console.error(`Directions error for ${routeName} segment ${originStop}→${destStop}:`, error);
              resolve([]);
              return;
            }
            try {
              const coords = data.routes[0].polyline.points.map(pt => [pt.latitude, pt.longitude]);
              resolve(coords);
            } catch (e) {
              console.error(`Unexpected response parsing for ${routeName} segment ${originStop}→${destStop}:`, e);
              resolve([]);
            }
          }
        );
      });

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
  function downloadJSON(data, filename = 'routeData.json') {
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

export default function MapKitMap({ routeData, vehicles, generateRoutes = false, selectedRoute, setSelectedRoute, selectedStop, setSelectedStop }) {

  const mapRef = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const token = import.meta.env.VITE_MAPKIT_KEY;
  const [map, setMap] = useState(null);
  const vehicleOverlays = useRef({});
  const circleWidth = 15;
  const selectedMarkerRef = useRef(null);


  // source: https://developer.apple.com/documentation/mapkitjs/loading-the-latest-version-of-mapkit-js
  const setupMapKitJs = async () => {
    if (!window.mapkit || window.mapkit.loadedLibraries.length === 0) {
      await new Promise(resolve => { window.initMapKit = resolve });
      delete window.initMapKit;
    }
  };

  useEffect(() => {
    // initialize mapkit
    const mapkitScript = async () => {
      // load the MapKit JS library
      await setupMapKitJs();
      window.mapkit.init({
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
      const center = new window.mapkit.Coordinate(42.730216326401114, -73.67568961656735);
      const span = new window.mapkit.CoordinateSpan(0.02, 0.005);
      const region = new window.mapkit.CoordinateRegion(center, span);

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
      const thisMap = new window.mapkit.Map(mapRef.current, mapOptions);
      // set zoom and boundary limits
      thisMap.setCameraZoomRangeAnimated(
        new window.mapkit.CameraZoomRange(200, 3000),
        false,
      );
      thisMap.setCameraBoundaryAnimated(
        new window.mapkit.CoordinateRegion(
          center,
          new window.mapkit.CoordinateSpan(0.02, 0.025)
        ),
        false,
      );
      thisMap.setCameraDistanceAnimated(2500);
      // Helper function to create and add stop marker
      const createStopMarker = (overlay) => {
        if (selectedMarkerRef.current) {
          thisMap.removeAnnotation(selectedMarkerRef.current);
          selectedMarkerRef.current = null;
        }
        const marker = new window.mapkit.MarkerAnnotation(overlay.coordinate, {
          title: overlay.stopName,
          glyphImage: { 1: "map-marker.png" },
        });
        thisMap.addAnnotation(marker);
        selectedMarkerRef.current = marker;
        return marker;
      };

      thisMap.addEventListener("select", (e) => {
        if (!e.overlay) return;

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

      // Detect hover on stop overlays
      let currentHoveredOverlay = null;

      thisMap.addEventListener("region-change-start", () => {
        thisMap.element.style.cursor = "grab";
      });

      thisMap.addEventListener("region-change-end", () => {
        thisMap.element.style.cursor = "default";
      });

      // Working hover detection
      let currentHover = null;

      thisMap.element.addEventListener('mousemove', (e) => {
        const rect = thisMap.element.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        let foundOverlay = null;

        // Check overlays for mouse position
        thisMap.overlays.forEach(overlay => {
          if (overlay.stopKey) {
            // Calculate overlay screen position
            const mapRect = thisMap.element.getBoundingClientRect();
            const centerLat = overlay.coordinate.latitude;
            const centerLng = overlay.coordinate.longitude;

            // Check if mouse is within overlay radius
            const region = thisMap.region;
            if (region) {
              const pixelPerDegree = mapRect.width / region.span.longitudeDelta;
              const centerX = mapRect.width * (centerLng - region.center.longitude + region.span.longitudeDelta/2) / region.span.longitudeDelta;
              const centerY = mapRect.height * (region.center.latitude - centerLat + region.span.latitudeDelta/2) / region.span.latitudeDelta;

              const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
              if (distance < circleWidth) { // Within hover radius
                foundOverlay = overlay;
              }
            }
          }
        });

        if (foundOverlay !== currentHover) {
          // Clear previous hover style
          if (currentHover) {
            currentHover.style = new window.mapkit.Style({
              strokeColor: '#000000',
              fillColor: '#FFFFFF',
              fillOpacity: 0.1,
              lineWidth: 2,
            });
          }

          // Apply hover style
          if (foundOverlay) {
            foundOverlay.style = new window.mapkit.Style({
              strokeColor: '#6699ff',
              fillColor: '#a1c3ff',
              fillOpacity: 0.3,
              lineWidth: 2.5,
            });
            thisMap.element.style.cursor = "pointer";
          } else {
            thisMap.element.style.cursor = "default";
          }

          currentHover = foundOverlay;
        }
      });

      // Store reference to cleanup function
      thisMap._hoverCleanup = () => {
        thisMap.element.removeEventListener('mousemove', handleMouseMove);
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

    var overlays = [];


    // display stop overlays
    for (const [route, thisRouteData] of Object.entries(routeData)) {
      for (const stopKey of thisRouteData.STOPS) {
        const stopCoordinate = new window.mapkit.Coordinate(...thisRouteData[stopKey].COORDINATES);
        // add stop overlay (circle)
        const stopOverlay = new window.mapkit.CircleOverlay(
          stopCoordinate,
          circleWidth,
          {
            style: new window.mapkit.Style(
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
        stopOverlay.stopName = thisRouteData[stopKey].NAME;
        overlays.push(stopOverlay);


      }
    }

    function displayRouteOverlays(routeData) {
      // display route overlays
      for (const [route, thisRouteData] of Object.entries(routeData)) {
        // for route (WEST, NORTH)
        const routePolylines = thisRouteData.ROUTES?.map(
          // for segment (STOP1 -> STOP2, STOP2 -> STOP3, ...)
          (route) => {
            const coords = route.map(([lat, lon]) => new window.mapkit.Coordinate(lat, lon));
            if (coords.length === 0) return null;
            const polyline = new window.mapkit.PolylineOverlay(coords, {
              // for coordinate ([lat, lon], ...)
              style: new window.mapkit.Style({
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
      const coordinate = new window.mapkit.Coordinate(vehicle.latitude, vehicle.longitude);
      if (key in vehicleOverlays.current) {
        // old vehicle: update coordinate
        console.log(`Updating vehicle ${key} to ${vehicle.latitude}, ${vehicle.longitude}`);
        vehicleOverlays.current[key].coordinate = coordinate;
        vehicleOverlays.current[key].subtitle = `${vehicle.speed_mph} mph`;
        if (vehicle.route_name !== "UNCLEAR") {
          vehicleOverlays.current[key].color = vehicle.route_name ? routeData[vehicle.route_name].COLOR : '#444444';
        }
      } else {
        // new vehicle: add to map
        console.log(`Adding vehicle ${key} to ${vehicle.latitude}, ${vehicle.longitude}`);
        const annotation = new window.mapkit.MarkerAnnotation(coordinate, {
          title: vehicle.name,
          subtitle: `${vehicle.speed_mph} mph`,
          color: vehicle.route_name && vehicle.route_name !== "UNCLEAR" ? routeData[vehicle.route_name].COLOR : '#444444',
          glyphImage: { 1: 'shubble20.png' },
          selectedGlyphImage: { 1: 'shubble20.png', 2: 'shubble40.png' },
        });
        map.addAnnotation(annotation);
        vehicleOverlays.current[key] = annotation;
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
