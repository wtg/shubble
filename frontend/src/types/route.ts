import type { Route } from "./schedule";

export type StopSchedule = {
    /** Days of week this stop operates, using JS Date.getDay() day names */
    days: Array<'MONDAY' | 'TUESDAY' | 'WEDNESDAY' | 'THURSDAY' | 'FRIDAY' | 'SATURDAY' | 'SUNDAY'>;
    /** Operating hours in 24-hour "HH:MM" format */
    hours: {
        start: string;
        end: string;
    };
};

export type ShuttleStopData = {
    COORDINATES: [number, number];
    OFFSET: number;
    NAME: string;
    /** Optional operating schedule. If absent, stop operates whenever its route does. */
    SCHEDULE?: StopSchedule;
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