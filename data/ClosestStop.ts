// Basic lat/lon point
export interface Point {
    lat: number;
    lon: number;
  }
  
  // A stop in stops array
  export interface Stop {
    id: string;
    name: string;
    lat: number;
    lon: number;
    route: string;
  }
  
  // The shape of one stop entry inside a route 
  export interface RouteStopInfo {
    COORDINATES: [number, number];
    OFFSET: number;
    NAME: string;
    [key: string]: unknown;
  }
  
  // The shape of each route block in routes.json
  export interface RouteData {
    COLOR: string;
    STOPS: string[];
    POLYLINE_STOPS: string[];
    [stopCode: string]: unknown | RouteStopInfo | string[] | string;
  }
  
  // All routes object imported from routes.json
  export type RoutesJson = Record<string, RouteData>;
  
  // A stop with its distance from the user in km
  export type ClosestStop = Stop & { distanceKm: number };
  
  // distance formula on a sphere
  export function haversine(pointA: Point, pointB: Point): number {
    const toRad = (deg: number) => (deg * Math.PI) / 180;
    const lat1 = toRad(pointA.lat), lon1 = toRad(pointA.lon);
    const lat2 = toRad(pointB.lat), lon2 = toRad(pointB.lon);
  
    const { sin, cos, sqrt, atan2 } = Math;
    const R = 6371; // radius of earth (km)
    const dLat = lat2 - lat1;
    const dLon = lon2 - lon1;
  
    const a =
      sin(dLat / 2) ** 2 +
      cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2;
    const c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c; 
  }
  
  // creates an array of stops for a single route
  export function extractStopsFromRoute(
    routeName: string,
    routeData: RouteData
  ): Stop[] {
    const stopCodes = routeData.STOPS;
    const stops: Stop[] = [];
  
    for (const code of stopCodes) {
      const info = routeData[code] as RouteStopInfo | undefined;
      if (!info || !('COORDINATES' in info)) continue;
  
      const [lat, lon] = info.COORDINATES;
      stops.push({
        id: code,
        name: info.NAME,
        lat,
        lon,
        route: routeName,
      });
    }
    return stops;
  }
  
  // creates an array of all valid stops (only NORTH + WEST)
  export function buildAllStops(data: RoutesJson): Stop[] {
    const allStops: Stop[] = [];
    for (const [routeName, routeData] of Object.entries(data)) {
      if (routeName !== 'NORTH' && routeName !== 'WEST') continue;
      allStops.push(...extractStopsFromRoute(routeName, routeData));
    }
  
    return allStops;
  }
  
  // go through valid stops and compare distance, returning closest stop
  export function findClosestStop(
    userPoint: Point,
    stops: Stop[]
  ): ClosestStop | null {
    let best: ClosestStop | null = null;
    let bestDist = Infinity;
    
    for (const stop of stops) {
      const distKm = haversine(userPoint, stop);
      if (distKm < bestDist) {
        bestDist = distKm;
        best = { ...stop, distanceKm: distKm };
      }
    }
    return best;
  }
  