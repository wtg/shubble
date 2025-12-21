import type { Route } from "./schedule";

export type ShuttleStopData = {
    COORDINATES: [number, number];
    OFFSET: number;
    NAME: string;
}

export type RoutePolylines = [number, number][][];

export type RouteDirectionData = {
    COLOR: string;
    STOPS: string[];
    POLYLINE_STOPS: string[];
    ROUTES: RoutePolylines;

    // This is *technically* the values of STOPS, but we are unable to get this in TS
    [key: string]: string | string[] | RoutePolylines | ShuttleStopData;
}

// route.json data type
export type ShuttleRouteData = Record<Route, RouteDirectionData>