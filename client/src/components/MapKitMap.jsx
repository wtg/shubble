import { useEffect, useRef, useState } from "react";
import '../styles/MapKitMap.css';
import routeData from '../data/routes.json';

export default function MapKitMap({ vehicles }) {

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
