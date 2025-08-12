import { useEffect, useRef, useState } from "react";
import '../styles/MapKitMap.css';
import routeData from '../data/routes.json';

async function generateRoutePolylines(updatedRouteData) {
    const directions = new window.mapkit.Directions();
    const promises = [];

    for (const [routeName, routeInfo] of Object.entries(updatedRouteData)) {
        const stops = routeInfo.STOPS || [];
        // Make sure ROUTES exists and has the same length as STOPS (number of segments)
        routeInfo.ROUTES = new Array(stops.length).fill(null);

        for (let i = 0; i < stops.length; i++) {
            const originStop = stops[i];
            const destStop = stops[(i + 1) % stops.length]; // wrap to first stop for loop

            const originCoords = routeInfo[originStop]?.COORDINATES;
            const destCoords = routeInfo[destStop]?.COORDINATES;

            if (!originCoords || !destCoords) {
                // missing stops: immediately push a resolved "empty" result so indexes stay consistent
                promises.push(Promise.resolve({ routeName, index: i, coords: [] }));
                continue;
            }

            const request = {
                origin: new window.mapkit.Coordinate(originCoords[0], originCoords[1]),
                destination: new window.mapkit.Coordinate(destCoords[0], destCoords[1]),
            };

            // Wrap the callback API in a promise that resolves with routeName + index + coords
            const p = new Promise((resolve) => {
                directions.route(request, (error, data) => {
                    if (error) {
                        console.error(`Directions error for ${routeName} segment ${i} (${originStop}→${destStop}):`, error);
                        // Resolve with empty coords array to avoid rejecting Promise.all
                        resolve({ routeName, index: i, coords: [] });
                        return;
                    }

                    try {
                        const route = data.routes[0];
                        const coords = route.polyline.points.map(pt => [pt.latitude, pt.longitude]);
                        resolve({ routeName, index: i, coords });
                    } catch (e) {
                        console.error(`Unexpected response parsing for ${routeName} segment ${i}:`, e);
                        resolve({ routeName, index: i, coords: [] });
                    }
                });
            });

            promises.push(p);
        }
    }

    // Wait for every segment promise. Order of results does NOT matter because we use index.
    const results = await Promise.all(promises);

    // Place the segment coords into the correct route.ROUTES[index]
    results.forEach((res) => {
        if (!res) return;
        const { routeName, index, coords } = res;
        // If route was removed in the meantime (unlikely), guard
        if (updatedRouteData[routeName]) {
            updatedRouteData[routeName].ROUTES[index] = coords;
        }
    });

    // Clean up any nulls (shouldn't be many, but safe)
    for (const routeInfo of Object.values(updatedRouteData)) {
        if (!Array.isArray(routeInfo.ROUTES)) routeInfo.ROUTES = [];
        for (let i = 0; i < routeInfo.ROUTES.length; i++) {
            if (!Array.isArray(routeInfo.ROUTES[i])) routeInfo.ROUTES[i] = [];
        }
    }

    // Trigger download of updatedRouteData as JSON
    function downloadJSON(data, filename = 'routeData.json') {
        const jsonStr = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();

        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }

    downloadJSON(updatedRouteData);

    return updatedRouteData;
}

export default function MapKitMap({ vehicles, generateRoutes=false }) {

    const mapRef = useRef(null);
    const [mapLoaded, setMapLoaded] = useState(false);
    const token = import.meta.env.VITE_MAPKIT_KEY;
    const [map, setMap] = useState(null);
    const vehicleOverlays = useRef({});

    // source: https://developer.apple.com/documentation/mapkitjs/loading-the-latest-version-of-mapkit-js
    const setupMapKitJs = async() => {
        if (!window.mapkit || window.mapkit.loadedLibraries.length === 0) {
            await new Promise(resolve => { window.initMapKit = resolve });
            delete window.initMapKit;
        }
    };

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

    // create the map
    useEffect(() => {
        if (mapLoaded) {

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

            const thisMap = new window.mapkit.Map(mapRef.current, mapOptions);
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
            setMap(thisMap);

        }
    }, [mapLoaded]);

    // add fixed details to the map
    // includes routes and stops
    useEffect(() => {
        if (!map) return;

        var overlays = [];

        // display stop overlays
        for (const [route, thisRouteData] of Object.entries(routeData)) {
            for (const stopName of thisRouteData.STOPS) {
                const stopCoordinate = new window.mapkit.Coordinate(...thisRouteData[stopName].COORDINATES);
                const stopOverlay = new window.mapkit.CircleOverlay(
                    stopCoordinate,
                    15,
                    {
                        style: new window.mapkit.Style(
                            {
                                strokeColor: '#000000',
                                lineWidth: 2,
                            }
                        )
                    }
                );
                overlays.push(stopOverlay);
            }
        }

        function displayRouteOverlays(routeData) {
            // display route overlays
            for (const [route, thisRouteData] of Object.entries(routeData)) {
                const routeCoordinates = thisRouteData.ROUTES.map(
                    (route) => route.map(([lat, lon]) => new window.mapkit.Coordinate(lat, lon))
                ).flat();
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
                console.log("Generated route polylines:", updatedRouteData);
                displayRouteOverlays(updatedRouteData);
            });
        } else {
            // use pre-generated polylines
            displayRouteOverlays(routeData);
        }

        map.addOverlays(overlays);

    }, [map]);

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
            } else {
                // new vehicle: add to map
                console.log(`Adding vehicle ${key} to ${vehicle.latitude}, ${vehicle.longitude}`);
                const annotation = new window.mapkit.MarkerAnnotation(coordinate, {
                    title: vehicle.vehicle_name,
                    subtitle: `${vehicle.speed_mph} mph`,
                    color: '#444444',
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
