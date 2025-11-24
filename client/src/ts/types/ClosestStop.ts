import type {ShuttleStopData, RouteDirectionData, ShuttleRouteData } from "./route"; 
import type {Route, ShuttleScheduleData} from "./schedule"; 
import {aggregateSchedule} from "./parseSchedule.js";


// a simple point
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

// A stop with its distance from the user + a distance in km 
export type ClosestStop = Stop & { distanceKm: number };
type AggregatedDay = Record<string, unknown>;

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
  
// use a route’s raw JSON data and turns it into an array of stop objects
export function extractStopsFromRoute( 
routeName: Route, routeData: RouteDirectionData 
): Stop[] {
  const stops: Stop[] = [];

  for (const code of routeData.STOPS) {
    const info = routeData[code] as ShuttleStopData | undefined;
    if (!info) continue;

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

  
// creates an array of all valid stops 
export function buildAllStops(
    routesData: ShuttleRouteData,
    scheduleData: ShuttleScheduleData,
    dayIndex: number
  ): Stop[] {
    const aggregated = aggregateSchedule(scheduleData);
  
    // Routes that actually have trips today
    const todaySchedule = aggregated[dayIndex] as AggregatedDay;
    const realRoutes = Object.keys(todaySchedule) as Route[];
  
    const stops: Stop[] = [];
  
    for (const routeName of realRoutes) {
      const routeData = routesData[routeName];
      if (!routeData) continue; // schedule references a route we don't have geometry for
      stops.push(...extractStopsFromRoute(routeName, routeData));
    }
  
    return stops;
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
  