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

export default function MapKitMap({ routeData, vehicles, generateRoutes=false }) {

    const mapRef = useRef(null);
    const [mapLoaded, setMapLoaded] = useState(false);
    const token = import.meta.env.VITE_MAPKIT_KEY;
    const [map, setMap] = useState(null);
    const vehicleOverlays = useRef({});
    const circleWidth = 15;
    const selectedRoute = useRef(null);

    // source: https://developer.apple.com/documentation/mapkitjs/loading-the-latest-version-of-mapkit-js
    const setupMapKitJs = async() => {
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
            thisMap.addEventListener("select", (e) => {
                if(e.overlay && e.overlay.stopName) {
                    console.log(`Selected overlay: ${e.overlay.stopName}`);
                    if (selectedRoute.current) {
                        map.removeAnnotation(selectedRoute.current);
                        selectedRoute.current = null;
                    }

                    // create temp marker for callout
                    selectedRoute.current = new window.mapkit.MarkerAnnotation(e.overlay.coordinate, {
                        title: e.overlay.stopName,
                        glyphText: "",
                        color: "transparent",
                    });

                    map.addAnnotation(selectedRoute.current);
                    map.selectAnnotation(selectedRoute.current, true);
                }
            });
            setMap(thisMap);
        }
    }, [mapLoaded]);

    // add fixed details to the map
    // includes routes and stops
    useEffect(() => {
        if (!map || !routeData) return;

        var overlays = [];

        // display stop overlays
        for (const [route, thisRouteData] of Object.entries(routeData)) {
            for (const stopName of thisRouteData.STOPS) {
                const stopCoordinate = new window.mapkit.Coordinate(...thisRouteData[stopName].COORDINATES);
                // add stop overlay (circle)
                const stopOverlay = new window.mapkit.CircleOverlay(
                    stopCoordinate,
                    circleWidth,
                    {
                        style: new window.mapkit.Style(
                            {
                                strokeColor: '#000000',
                                lineWidth: 2,
                            }
                        )
                    }
                );
                stopOverlay.stopName = stopName;
                overlays.push(stopOverlay);
            }
        }

        function displayRouteOverlays(routeData) {
            // display route overlays
            for (const [route, thisRouteData] of Object.entries(routeData)) {
                const routeCoordinates = thisRouteData.ROUTES?.map(
                    (route) => route.map(([lat, lon]) => new window.mapkit.Coordinate(lat, lon))
                ).flat();
                if (!routeCoordinates || routeCoordinates.length === 0) continue;
                // add route overlay (polyline)
                const routePolyline = new mapkit.PolylineOverlay(routeCoordinates, {
                    style: new mapkit.Style({
                        strokeColor: thisRouteData.COLOR,
                        lineWidth: 2
                        })
                });
                overlays.push(routePolyline);
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
                    title: vehicle.vehicle_name,
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
