import { useEffect, useRef, useState } from "react";
import '../styles/MapKitMap.css';

export default function MapKitMap({ vehicles }) {

    const mapRef = useRef(null);
    const [mapLoaded, setMapLoaded] = useState(false);
    const [token, setToken] = useState(null);
    const [map, setMap] = useState(null);
    const vehicleOverlays = useRef({});

    // source: https://developer.apple.com/documentation/mapkitjs/loading-the-latest-version-of-mapkit-js
    const setupMapKitJs = async() => {
        if (!window.mapkit || window.mapkit.loadedLibraries.length === 0) {
            await new Promise(resolve => { window.initMapKit = resolve });
            delete window.initMapKit;
        }
    };


    // fetch the MapKit token from the server
    useEffect(() => {
        fetch('/api/mapkit')
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then((data) => {
                setToken(data);
            })
            .catch((error) => {
                console.error('Error fetching MapKit token:', error);
            });
    }, []);

    // initialize mapkit
    useEffect(() => {
        if (!token) return;
        const mapkitScript = async () => {
            // load the MapKit JS library
            await setupMapKitJs();
            window.mapkit.init({
                authorizationCallback: (done) => {
                    done(token);
                },
            });
            setMapLoaded(true);
        }
        mapkitScript();
    }, [token]);

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

        // coordinates of stops

        // north
        // 42.730676958536144, -73.67674616623393
        // 42.737043669212134, -73.67036818086305
        // 42.735455332919045, -73.6636579612421
        // 42.73453830902714, -73.6634349282215
        // 42.7327033365768, -73.66522556880754
        // 42.731383488711145, -73.66654627298861
        // 42.73175755884203, -73.66967270972104

        // west
        // 42.730318398121575, -73.67656636425313
        // 42.72799822908236, -73.67809671921837
        // 42.72293385583282, -73.67960937432454
        // 42.72766469852938, -73.68716188006067
        // 42.73160170173599, -73.68627833913843
        // 42.731468785216094, -73.68128223685743
        // 42.731023124913804, -73.67909065365457

        // circle overlays for the stops
        const unionCircle = new window.mapkit.CircleOverlay(
            new window.mapkit.Coordinate(42.730676958536144, -73.67674616623393),
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

        const northCircles = [
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.737043669212134, -73.67036818086305),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.735455332919045, -73.6636579612421),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.73453830902714, -73.6634349282215),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.7327033365768, -73.66522556880754),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.731383488711145, -73.66654627298861),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.73175755884203, -73.66967270972104),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
        ];

        const westCircles = [
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.72799822908236, -73.67809671921837),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.72293385583282, -73.67960937432454),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.72766469852938, -73.68716188006067),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.73160170173599, -73.68627833913843),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.731468785216094, -73.68128223685743),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
            new window.mapkit.CircleOverlay(
                new window.mapkit.Coordinate(42.731023124913804, -73.67909065365457),
                15,
                {
                    style: new window.mapkit.Style(
                        {
                            strokeColor: '#000000',
                            lineWidth: 2,
                        }
                    )
                }
            ),
        ];


        // directions for the loops
        const directions = new window.mapkit.Directions();

        const northDirectionRequests = [
            {
                origin: new window.mapkit.Coordinate(42.730676958536144, -73.67674616623393),
                destination: new window.mapkit.Coordinate(42.737043669212134, -73.67036818086305),
            },
            {
                origin: new window.mapkit.Coordinate(42.737043669212134, -73.67036818086305),
                destination: new window.mapkit.Coordinate(42.737902319942435, -73.66773188495978),
            },
            {
                origin: new window.mapkit.Coordinate(42.737902319942435, -73.66773188495978),
                destination: new window.mapkit.Coordinate(42.735455332919045, -73.6636579612421),
            },
            {
                origin: new window.mapkit.Coordinate(42.735455332919045, -73.6636579612421),
                destination: new window.mapkit.Coordinate(42.73453830902714, -73.6634349282215),
            },
            {
                origin: new window.mapkit.Coordinate(42.73453830902714, -73.6634349282215),
                destination: new window.mapkit.Coordinate(42.7327033365768, -73.66522556880754),
            },
            {
                origin: new window.mapkit.Coordinate(42.7327033365768, -73.66522556880754),
                destination: new window.mapkit.Coordinate(42.73080472933945, -73.6673502020617),
            },
            {
                origin: new window.mapkit.Coordinate(42.73080472933945, -73.6673502020617),
                destination: new window.mapkit.Coordinate(42.73175755884203, -73.66967270972104),
            },
            {
                origin: new window.mapkit.Coordinate(42.73175755884203, -73.66967270972104),
                destination: new window.mapkit.Coordinate(42.73305698515937, -73.67655203737692),
            },
            {
                origin: new window.mapkit.Coordinate(42.73305698515937, -73.67655203737692),
                destination: new window.mapkit.Coordinate(42.730676958536144, -73.67674616623393),
            }
        ];

        const westDirectionRequests = [
            {
                origin: new window.mapkit.Coordinate(42.730676958536144, -73.67674616623393),
                destination: new window.mapkit.Coordinate(42.72799822908236, -73.67809671921837),
            },
            {
                origin: new window.mapkit.Coordinate(42.72799822908236, -73.67809671921837),
                destination: new window.mapkit.Coordinate(42.72293385583282, -73.67960937432454),
            },
            {
                origin: new window.mapkit.Coordinate(42.72293385583282, -73.67960937432454),
                destination: new window.mapkit.Coordinate(42.72779537209815, -73.68848176378644),
            },
            {
                origin: new window.mapkit.Coordinate(42.72779537209815, -73.68848176378644),
                destination: new window.mapkit.Coordinate(42.72766469852938, -73.68716188006067),
            },
            {
                origin: new window.mapkit.Coordinate(42.72766469852938, -73.68716188006067),
                destination: new window.mapkit.Coordinate(42.73160170173599, -73.68627833913843),
            },
            {
                origin: new window.mapkit.Coordinate(42.73160170173599, -73.68627833913843),
                destination: new window.mapkit.Coordinate(42.731468785216094, -73.68128223685743),
            },
            {
                origin: new window.mapkit.Coordinate(42.731468785216094, -73.68128223685743),
                destination: new window.mapkit.Coordinate(42.731023124913804, -73.67909065365457),
            },
            {
                origin: new window.mapkit.Coordinate(42.731023124913804, -73.67909065365457),
                destination: new window.mapkit.Coordinate(42.730676958536144, -73.67674616623393),
            }
        ];

        // add directions to map

        northDirectionRequests.forEach((request) => {
            directions.route(request, (error, data) => {
                if (error) {
                    console.error('Error calculating directions:', error);
                    return;
                }
                const route = data.routes[0];
                const routeOverlay = route.polyline;
                routeOverlay.points.map((point) => {
                    console.log(point);
                });
                routeOverlay.style = new window.mapkit.Style({
                    strokeColor: '#FF0000',
                    lineWidth: 2,
                });
                map.addOverlay(routeOverlay);
            });
        });

        westDirectionRequests.forEach((request) => {
            directions.route(request, (error, data) => {
                if (error) {
                    console.error('Error calculating directions:', error);
                    return;
                }
                const route = data.routes[0];
                const routeOverlay = route.polyline;
                routeOverlay.points.map((point) => {
                    console.log(point);
                });
                routeOverlay.style = new window.mapkit.Style({
                    strokeColor: '#0000FF',
                    lineWidth: 2,
                });
                map.addOverlay(routeOverlay);
            });
        });

        // add stops to map

        const overlays = [unionCircle, ...northCircles, ...westCircles];
        map.addOverlays(overlays);

    }, [map]);

    // display vehicles on map
    useEffect(() => {
        if (!map || !vehicles) return;

        Object.keys(vehicles).map((key) => {
            const vehicle = vehicles[key];
            const coordinate = new window.mapkit.Coordinate(vehicle.lat, vehicle.lng);
            if (key in vehicleOverlays) {
                // old vehicle: update coordinate
                console.log(`Updating vehicle ${key} to ${vehicle.lat}, ${vehicle.lng}`);
                vehicleOverlays[key].coordinate = coordinate;
            } else {
                // new vehicle: add to map
                console.log(`Adding vehicle ${key} to ${vehicle.lat}, ${vehicle.lng}`);
                const annotation = new window.mapkit.MarkerAnnotation(coordinate, {
                    title: `Vehicle ID: ${key}`,
                    subtitle: `Speed: ${vehicle.speed} mph`,
                    color: '#444444',
                });
                map.addAnnotation(annotation);
                vehicleOverlays[key] = annotation;
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
