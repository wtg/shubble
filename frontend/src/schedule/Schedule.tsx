import { useState, useEffect, useCallback, useMemo } from 'react';
import './styles/Schedule.css';
import rawRouteData from '../shared/routes.json';
import rawAggregatedSchedule from '../shared/aggregated_schedule.json';
import config from '../utils/config';
import type { AggregatedDaySchedule, AggregatedScheduleType, Route } from '../types/schedule';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';
import {
  useTrips,
  deriveStopEtasFromTrips,
  type Trip,
  type StopETAs,
  type StopETADetails,
} from '../hooks/useTrips';
import { useClosestStop } from '../hooks/useClosestStop';

const aggregatedSchedule: AggregatedScheduleType = rawAggregatedSchedule as unknown as AggregatedScheduleType;
const routeData = rawRouteData as unknown as ShuttleRouteData;

import { devNow, devNowMs } from '../utils/devTime';

const TIME_FORMAT: Intl.DateTimeFormatOptions = { hour: 'numeric', minute: '2-digit' };

const POINTER_CURSOR_STYLE = { cursor: 'pointer' as const };

// --- Departure deviation labels (260415-3ec) -----------------------------
// Small inline label next to each timeline row's departure time that tells
// students whether the trip departed on/near its scheduled slot (with a
// signed delta) or off-schedule (anchored to the nearest real slot so the
// row doesn't look disconnected from the published plan).
//
// Returned strings:
//   scheduled-matched, on-time (|delta| <= 1 min):    null (no label)
//   scheduled-matched, late:    "(H:MM, +N min late)"
//   scheduled-matched, early:   "(H:MM, N min early)"
//   off-schedule, <=30 min from nearest slot (early): "↑ N min early to H:MM slot"
//   off-schedule, <=30 min from nearest slot (late):  "↓ N min late from H:MM slot"
//   off-schedule, >30 min from any slot:              "Unscheduled"
export type DepartureLabelKind =
  | 'matched-late'
  | 'matched-early'
  | 'unscheduled-early'
  | 'unscheduled-late'
  | 'unscheduled-far';

export type DepartureLabel = {
  text: string;
  kind: DepartureLabelKind;
};

// Module-level helper — pure function, no closures / hooks. Accepts every
// dependency as an argument so it can live above the component with no
// re-creation cost per render.
//
// `_selectedDay` is accepted for parity with the task brief's signature;
// `timeToDate` is already parameterized on the current day (its cache key
// is `selectedDay` — see Schedule.tsx timeToDateCache), so it's unused here.
// The underscore prefix satisfies eslint's argsIgnorePattern.
export function getDepartureLabel(
  trip: Trip,
  routeTimes: string[],
  _selectedDay: number,
  timeToDate: (t: string) => Date,
): DepartureLabel | null {
  const fmt = (d: Date) => d.toLocaleTimeString(undefined, TIME_FORMAT);

  // --- Case 1: scheduled-matched trip ---
  if (trip.scheduled) {
    // actual_departure can be null (e.g. a scheduled row with no vehicle
    // yet). Nothing to compare — no label.
    if (!trip.actual_departure) return null;

    const scheduledMs = new Date(trip.departure_time).getTime();
    const actualMs = new Date(trip.actual_departure).getTime();
    const deltaMin = Math.round((actualMs - scheduledMs) / 60_000);

    // Dead zone: |delta| <= 1 min → no label (on-time)
    if (Math.abs(deltaMin) <= 1) return null;

    const actualStr = fmt(new Date(actualMs));
    if (deltaMin > 0) {
      return {
        text: `(${actualStr}, +${deltaMin} min late)`,
        kind: 'matched-late',
      };
    }
    return {
      text: `(${actualStr}, ${Math.abs(deltaMin)} min early)`,
      kind: 'matched-early',
    };
  }

  // --- Case 2: off-schedule (injected) trip ---
  // For injected trips, trip.departure_time === trip.actual_departure per
  // backend contract (trips.py). Fall back to departure_time if actual is
  // null for any reason so the helper stays functional.
  const actualMs = trip.actual_departure
    ? new Date(trip.actual_departure).getTime()
    : new Date(trip.departure_time).getTime();

  if (routeTimes.length === 0) {
    return { text: 'Unscheduled', kind: 'unscheduled-far' };
  }

  // Find the nearest scheduled slot by absolute ms-distance.
  let nearestSlotStr = routeTimes[0];
  let nearestDeltaMs = Math.abs(actualMs - timeToDate(routeTimes[0]).getTime());
  let nearestSignedDeltaMs = actualMs - timeToDate(routeTimes[0]).getTime();
  for (let i = 1; i < routeTimes.length; i++) {
    const slotMs = timeToDate(routeTimes[i]).getTime();
    const absD = Math.abs(actualMs - slotMs);
    if (absD < nearestDeltaMs) {
      nearestDeltaMs = absD;
      nearestSignedDeltaMs = actualMs - slotMs;
      nearestSlotStr = routeTimes[i];
    }
  }

  const nearestMin = Math.round(nearestDeltaMs / 60_000);

  // >30 min from every slot → plain "Unscheduled" (no anchor)
  if (nearestMin > 30) {
    return { text: 'Unscheduled', kind: 'unscheduled-far' };
  }

  // Degenerate: actual falls exactly on a slot. Shouldn't happen in
  // practice (the row would have matched instead of being injected) but
  // guard anyway so we never emit a "0 min" pill.
  if (nearestSignedDeltaMs === 0) {
    return { text: 'Unscheduled', kind: 'unscheduled-far' };
  }

  // Normalize the slot label via timeToDate→fmt so its spacing matches
  // `actualStr` formatting (e.g. "5:30 PM" vs "5:30PM").
  const slotStr = fmt(timeToDate(nearestSlotStr));

  if (nearestSignedDeltaMs < 0) {
    return {
      text: `↑ ${nearestMin} min early to ${slotStr} slot`,
      kind: 'unscheduled-early',
    };
  }
  return {
    text: `↓ ${nearestMin} min late from ${slotStr} slot`,
    kind: 'unscheduled-late',
  };
}

type ScheduleProps = {
  selectedRoute: string | null;
  setSelectedRoute: (route: string | null) => void;
  /** Optional external trips array — if provided, Schedule won't double-poll. */
  trips?: Trip[];
  /** Optional externally-derived per-stop views. Recomputed internally if omitted. */
  stopETAs?: StopETAs;
  stopETADetails?: StopETADetails;
};

export default function Schedule({
  selectedRoute,
  setSelectedRoute,
  trips: externalTrips,
  stopETAs: externalStopETAs,
  stopETADetails: externalDetails,
}: ScheduleProps) {
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }

  const now = devNow();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [schedule, setSchedule] = useState<AggregatedDaySchedule>(aggregatedSchedule[selectedDay]);
  const [showFullSchedule, setShowFullSchedule] = useState(false);
  const [expandedLoops, setExpandedLoops] = useState<Set<string>>(new Set());
  // Selected stop is persisted per-route so switching routes doesn't
  // destroy the user's previously-picked stop on the other route.
  const [selectedStop, setSelectedStop] = useState<string | null>(
    () => localStorage.getItem('shubble-stop')
  );
  const [tick, setTick] = useState(0); // for countdown re-renders

  // Trips are the single source of truth. If the parent already polled
  // and passed them in (the common case via LiveLocation.tsx), reuse
  // them; otherwise set up our own poll.
  const hasExternalTrips = externalTrips !== undefined;
  const internalTrips = useTrips(!config.staticETAs && !hasExternalTrips);
  const trips: Trip[] = hasExternalTrips ? externalTrips! : internalTrips;

  // Derive the per-stop ETAs/details for legacy fallback branches that still
  // reference `liveETAs` / `liveETADetails`. Prefer externally-derived data
  // to avoid duplicate work when the parent already computed it.
  const derivedView = useMemo(
    () => (externalStopETAs && externalDetails)
      ? { stopETAs: externalStopETAs, stopETADetails: externalDetails }
      : deriveStopEtasFromTrips(trips),
    [trips, externalStopETAs, externalDetails]
  );
  const liveETAs = derivedView.stopETAs;
  const liveETADetails = derivedView.stopETADetails;

  const safeSelectedRoute = selectedRoute || routeNames[0];

  // Hoisted from below so the defensive filter can consult route.STOPS.
  const route = routeData[safeSelectedRoute as keyof typeof routeData];

  // Trips for the selected route only (row model built below, after `times`).
  // Defensive: a trip whose `route` field disagrees with its `stop_etas` keys
  // would render as, e.g., "NORTH stops on WEST page". The invariant holds on
  // the backend (trips.py builds stop_etas FROM route.STOPS), but if a stale
  // trip is cached in React state from an earlier moment -- or an SSE payload
  // is dropped mid-cycle -- drop it here so the user never sees cross-route
  // rows. Bug report: 2026-04-14 "5:20 PM #001 Student Union" on WEST expanded
  // to NORTH stops; vehicle 001 is NORTH-only, but a stale trip survived in
  // state after a route swap.
  const validStopKeys = new Set<string>(route?.STOPS ?? []);
  const routeTrips: Trip[] = trips.filter(t =>
    t.route === safeSelectedRoute &&
    Object.keys(t.stop_etas).every(s => validStopKeys.has(s))
  );

  // Auto-detect closest stop via geolocation
  const { closestStop } = useClosestStop(safeSelectedRoute);

  // Initialize selectedStop from geolocation if not already set
  useEffect(() => {
    if (!selectedStop && closestStop) {
      setSelectedStop(closestStop.id);
      localStorage.setItem('shubble-stop', closestStop.id);
    }
  }, [closestStop, selectedStop]);

  // When the user switches routes, restore their previous stop preference
  // on the new route (if any), or fall back to geolocation/first non-SU.
  // This avoids losing the user's carefully-picked stop when they briefly
  // peek at another route.
  useEffect(() => {
    const currentRoute = routeData[safeSelectedRoute as keyof typeof routeData];
    if (!currentRoute?.STOPS) return;
    if (selectedStop && currentRoute.STOPS.includes(selectedStop)) return; // already valid

    // 1. Restore per-route memory
    const remembered = localStorage.getItem(`shubble-stop:${safeSelectedRoute}`);
    if (remembered && currentRoute.STOPS.includes(remembered)) {
      setSelectedStop(remembered);
      localStorage.setItem('shubble-stop', remembered);
      return;
    }

    // 2. Geo-based closest on this route
    if (closestStop?.id && currentRoute.STOPS.includes(closestStop.id)) {
      setSelectedStop(closestStop.id);
      localStorage.setItem('shubble-stop', closestStop.id);
      localStorage.setItem(`shubble-stop:${safeSelectedRoute}`, closestStop.id);
      return;
    }

    // 3. Fall back to the second stop — the first is Student Union, which
    //    shuttles pass through continuously so its ETA is rarely useful.
    const fallback = currentRoute.STOPS.length > 1 ? currentRoute.STOPS[1] : currentRoute.STOPS[0];
    setSelectedStop(fallback);
    localStorage.setItem('shubble-stop', fallback);
    localStorage.setItem(`shubble-stop:${safeSelectedRoute}`, fallback);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeSelectedRoute]);

  // Persist stop selection. Store per-route so switching routes
  // doesn't destroy the user's other-route preferences.
  const handleStopSelect = useCallback((stopId: string) => {
    setSelectedStop(stopId);
    localStorage.setItem('shubble-stop', stopId);
    // Per-route memory: which stop did the user pick on this route?
    localStorage.setItem(`shubble-stop:${safeSelectedRoute}`, stopId);
  }, [safeSelectedRoute]);

  // Re-render countdown every 10 seconds
  useEffect(() => {
    const interval = setInterval(() => setTick(t => t + 1), 10_000);
    return () => clearInterval(interval);
  }, []);

  // Update schedule and routeNames when selectedDay changes
  useEffect(() => {
    setSchedule(aggregatedSchedule[selectedDay]);
    setRouteNames(Object.keys(aggregatedSchedule[selectedDay]));
    const firstRoute = Object.keys(aggregatedSchedule[selectedDay])[0];
    if (!selectedRoute || !(selectedRoute in aggregatedSchedule[selectedDay])) {
      setSelectedRoute(firstRoute);
    }
    setShowFullSchedule(false);
    setExpandedLoops(new Set());
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDay]);

  const handleDayChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDay(parseInt(e.target.value));
  };

  // P6: memoize timeToDate results across the render. Schedule strings repeat
  // across helpers (findCurrentLoopIndex, row building, fallback loops) and
  // the parse work is cheap but non-zero. Cache is invalidated on day change
  // since the "today at HH:MM" baseline depends on the current wall-clock day.
  const timeToDateCache = useMemo(() => new Map<string, Date>(), [selectedDay]);  // eslint-disable-line react-hooks/exhaustive-deps
  const timeToDate = useCallback((timeStr: string): Date => {
    const cached = timeToDateCache.get(timeStr);
    if (cached) return cached;
    const trimmed = timeStr.trim();
    const [time, modifier] = trimmed.split(" ");
    let [hours, minutes] = time.split(":").map(Number);
    if (modifier.toUpperCase() === "PM" && hours !== 12) {
      hours += 12;
    } else if (modifier.toUpperCase() === "AM" && hours === 12) {
      hours = 0;
    }
    const dateObj = new Date();
    dateObj.setHours(hours);
    dateObj.setMinutes(minutes);
    dateObj.setSeconds(0);
    dateObj.setMilliseconds(0);
    // Schedule times that wrap past midnight (e.g. "12:00 AM") represent
    // the END of the current service day — i.e. tomorrow at 00:00 — so
    // roll them forward. Matches backend handling in trips.py.
    if (trimmed === "12:00 AM") {
      dateObj.setDate(dateObj.getDate() + 1);
    }
    timeToDateCache.set(timeStr, dateObj);
    return dateObj;
  }, [timeToDateCache]);

  // Find the current loop index (the most recent loop that started)
  const findCurrentLoopIndex = (times: string[]): number => {
    if (selectedDay !== now.getDay()) return -1;

    let currentIdx = -1;
    for (let i = 0; i < times.length; i++) {
      const loopTime = timeToDate(times[i]);
      if (loopTime <= now) {
        currentIdx = i;
      } else {
        break;
      }
    }
    return currentIdx;
  };

  // Scroll to current time on route/day change, AND once more after
  // trips data first loads (the initial render has trips=[], so the
  // "current loop" rows don't exist in the DOM yet). We also re-trigger
  // when trips transitions from empty to non-empty, but NOT on every
  // update — hasTrips is a boolean flag, so the effect only re-fires
  // when it flips, not on regular polling updates.
  const hasTrips = trips.length > 0;
  useEffect(() => {
    const timelineContainer = document.querySelector('.timeline-container') as HTMLElement;
    if (!timelineContainer) return;
    if (selectedDay !== now.getDay()) return;

    // Priority 1: the first current-loop row that is NOT a completed
    // (DONE-badged) trip. Completed trips are still marked current-loop
    // for 2 minutes after their shuttle finishes, which lets the user
    // see "loop just finished". But on initial scroll we want to land
    // on the LIVE loop (the one actively running), not on a DONE row
    // that's already in the past from a user's perspective. The user
    // can scroll up to see the DONE rows — we just don't park them
    // there by default.
    const currentLoopItems = Array.from(
      timelineContainer.querySelectorAll('.timeline-item.current-loop')
    ) as HTMLElement[];
    let targetItem: HTMLElement | null = currentLoopItems.find(
      item => !item.querySelector('.trip-completed-badge')
    ) ?? null;

    // Priority 2: any current-loop row (covers the case where ALL
    // current-loop rows are completed — rare but possible for a
    // 2-minute window after the last shuttle finishes its final loop).
    if (!targetItem && currentLoopItems.length > 0) {
      targetItem = currentLoopItems[0];
    }

    // Priority 3: the first non-past item (no current loop exists).
    if (!targetItem) {
      targetItem = Array.from(timelineContainer.querySelectorAll('.timeline-item')).find(item =>
        !item.classList.contains('past-time')
      ) as HTMLElement | null;
    }

    if (targetItem) {
      const containerRect = timelineContainer.getBoundingClientRect();
      const targetRect = targetItem.getBoundingClientRect();
      const targetOffsetInContainer = targetRect.top - containerRect.top + timelineContainer.scrollTop;
      timelineContainer.scrollTop = Math.max(0, targetOffsetInContainer);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRoute, selectedDay, schedule, hasTrips]);

  const toggleExpand = (key: string) => {
    setExpandedLoops(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  // P6: memoize so deps in downstream hooks stay stable across unrelated renders
  const times = useMemo(
    () => schedule[safeSelectedRoute as Route] || [],
    [schedule, safeSelectedRoute]
  );
  const currentLoopIndex = findCurrentLoopIndex(times);
  const isToday = selectedDay === now.getDay();

  // --- Timeline row model: one row per physical departure ---
  // When two shuttles concurrently run the same scheduled time, each gets
  // its own row. Injected (early / non-schedule) trips appear as their own
  // rows at their actual departure time. Scheduled slots with no trip
  // match render as a single "trip: null" row (static/scheduled display).
  type TimelineRow = {
    key: string;           // React key; unique per row
    time: string;          // "9:00 AM" display label
    timeDate: Date;        // for sorting + past-time comparison
    trip: Trip | null;     // null when no vehicle is assigned yet
    originalIndex: number; // index into `times`; -1 for injected
    isInjected: boolean;
  };

  const formatDepartureDisplay = (d: Date): string =>
    d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });

  // Group trips by their departure-time display string for O(1) lookup.
  // Trips are ALWAYS today's live data — never merge them into other days'
  // schedules, or tomorrow's schedule will show ghost vehicle badges.
  const tripsByTimeKey: Record<string, Trip[]> = {};
  if (isToday) {
    for (const t of routeTrips) {
      const key = formatDepartureDisplay(new Date(t.departure_time));
      (tripsByTimeKey[key] ||= []).push(t);
    }
  }

  const timelineRows: TimelineRow[] = [];
  const consumedTripIds = new Set<string>();

  // 1. Emit rows for each scheduled time. Hide past scheduled-only rows
  //    (older than 5 minutes) — they're clutter when an active shuttle
  //    is already running on the route. Scheduled rows with matching
  //    trip data are always shown regardless of age.
  const pastThreshold = new Date(now.getTime() - 5 * 60_000);
  times.forEach((time, i) => {
    const timeDate = timeToDate(time);
    const matching = tripsByTimeKey[time] || [];
    if (matching.length === 0) {
      if (isToday && timeDate < pastThreshold) return;
      timelineRows.push({
        key: `${time}|sched:${i}`,
        time,
        timeDate,
        trip: null,
        originalIndex: i,
        isInjected: false,
      });
    } else {
      for (const t of matching) {
        consumedTripIds.add(t.trip_id);
        timelineRows.push({
          key: `${time}|${t.vehicle_id ?? t.trip_id}`,
          time,
          timeDate,
          trip: t,
          originalIndex: i,
          isInjected: false,
        });
      }
    }
  });

  // 2. Append injected trips (non-schedule times) at their own slots.
  //    Only for today — tomorrow's view is static schedule only.
  if (isToday) {
    for (const t of routeTrips) {
      if (consumedTripIds.has(t.trip_id)) continue;
      const d = new Date(t.departure_time);
      const time = formatDepartureDisplay(d);
      timelineRows.push({
        key: `${time}|${t.vehicle_id ?? t.trip_id}|injected`,
        time,
        timeDate: d,
        trip: t,
        originalIndex: -1,
        isInjected: true,
      });
    }
  }

  // 3. Stable sort by timeDate so same-time rows stay grouped
  timelineRows.sort((a, b) => a.timeDate.getTime() - b.timeDate.getTime());

  // Rows considered "current" — either an active trip, or (fallback) the
  // most-recent scheduled row when no trips are present at all.
  const currentRowKeys = new Set<string>();
  if (isToday) {
    let sawActive = false;
    for (const row of timelineRows) {
      if (row.trip?.status === 'active' || row.trip?.status === 'completed') {
        currentRowKeys.add(row.key);
        if (row.trip?.status === 'active') sawActive = true;
      }
    }
    if (!sawActive && currentLoopIndex >= 0) {
      // Legacy fallback: no trip data yet → highlight the most recent scheduled row
      const legacyRow = timelineRows.find(
        r => r.trip === null && r.originalIndex === currentLoopIndex
      );
      if (legacyRow) currentRowKeys.add(legacyRow.key);
    }
  }

  // P6: precompute the stop→offset map ONCE per route. Every static ETA
  // computation reduces to `base + offsetMs`. Avoids walking route.STOPS and
  // casting/destructuring per-call.
  const stopOffsetMs = useMemo(() => {
    const out: Record<string, number> = {};
    if (!route?.STOPS) return out;
    for (const stopKey of route.STOPS) {
      const stopData = route[stopKey] as ShuttleStopData | undefined;
      if (!stopData) continue;
      out[stopKey] = stopData.OFFSET * 60_000;
    }
    return out;
  }, [route]);

  // Compute static ETAs from route offsets given an absolute departure Date.
  // Used directly for injected trips (which have no schedule index) and
  // indirectly via computeStaticETAs(loopIndex) for scheduled rows.
  const computeStaticETAsForDate = useCallback((departureTime: Date): StopETAs => {
    const stopETAs: StopETAs = {};
    const baseMs = departureTime.getTime();
    for (const stopKey in stopOffsetMs) {
      stopETAs[stopKey] = new Date(baseMs + stopOffsetMs[stopKey]).toLocaleTimeString(undefined, TIME_FORMAT);
    }
    return stopETAs;
  }, [stopOffsetMs]);

  const computeStaticETADatesForDate = useCallback((departureTime: Date): Record<string, Date> => {
    const dates: Record<string, Date> = {};
    const baseMs = departureTime.getTime();
    for (const stopKey in stopOffsetMs) {
      dates[stopKey] = new Date(baseMs + stopOffsetMs[stopKey]);
    }
    return dates;
  }, [stopOffsetMs]);

  // P6: cache static ETA/dates per loopIndex so repeated calls in the row
  // render loop are O(1). Render-scoped — re-created every render, which is
  // cheap and keeps deps stable for the useCallbacks below.
  const staticETAsByIndex = useMemo(() => new Map<number, StopETAs>(), [times, stopOffsetMs]);  // eslint-disable-line react-hooks/exhaustive-deps
  const staticETADatesByIndex = useMemo(() => new Map<number, Record<string, Date>>(), [times, stopOffsetMs]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Compute static ETAs from route offsets for a scheduled loop index.
  const computeStaticETAs = useCallback((loopIndex: number): StopETAs => {
    if (loopIndex < 0) return {};
    const cached = staticETAsByIndex.get(loopIndex);
    if (cached) return cached;
    const result = computeStaticETAsForDate(timeToDate(times[loopIndex]));
    staticETAsByIndex.set(loopIndex, result);
    return result;
  }, [times, timeToDate, computeStaticETAsForDate, staticETAsByIndex]);

  // Compute static ETA Dates for deviation comparison
  const computeStaticETADates = useCallback((loopIndex: number): Record<string, Date> => {
    if (loopIndex < 0) return {};
    const cached = staticETADatesByIndex.get(loopIndex);
    if (cached) return cached;
    const result = computeStaticETADatesForDate(timeToDate(times[loopIndex]));
    staticETADatesByIndex.set(loopIndex, result);
    return result;
  }, [times, timeToDate, computeStaticETADatesForDate, staticETADatesByIndex]);

  // Determine contextual message when no ETA data is available (D-08).
  // P6: computed ONCE per render instead of being called from the inner
  // per-stop loop — the value only depends on selectedDay/selectedRoute/now.
  const missingDataMessage: string = (() => {
    const daySchedule = aggregatedSchedule[selectedDay];
    const routeTimes = daySchedule?.[safeSelectedRoute as Route];
    if (!routeTimes || routeTimes.length === 0) return 'No shuttle in service';
    const firstDeparture = timeToDate(routeTimes[0]);
    const lastDeparture = timeToDate(routeTimes[routeTimes.length - 1]);
    if (now < firstDeparture) return `Service starts at ${routeTimes[0]}`;
    if (now > lastDeparture) return 'No shuttle in service';
    return 'No shuttle in service';
  })();

  // Compute early/late deviation for live ETAs (D-05, D-06, D-07, D-11, D-12).
  // Takes an explicit liveISO so trip-based rows compare against their own
  // per-vehicle ETA instead of the global per-stop aggregate. Falls back to
  // the legacy aggregate for rows without trip data.
  const computeDeviation = (
    stopKey: string,
    loopStaticDatesForStop: Record<string, Date>,
    liveISO?: string | null,
  ): { text: string; className: string } | null => {
    let effectiveISO: string | null | undefined = liveISO;
    if (!effectiveISO) {
      const detail = liveETADetails[stopKey];
      if (!detail?.etaISO) return null;
      effectiveISO = detail.etaISO;
    }

    const scheduledDate = loopStaticDatesForStop[stopKey];
    if (!scheduledDate) return null;

    const liveDate = new Date(effectiveISO);
    const diffMinutes = Math.round((liveDate.getTime() - scheduledDate.getTime()) / 60_000);

    // D-11: 2-minute dead zone (inclusive)
    if (Math.abs(diffMinutes) <= 2) return null;

    if (diffMinutes > 0) {
      return { text: `+${diffMinutes} min late`, className: 'eta-late' };
    } else {
      return { text: `${diffMinutes} min early`, className: 'eta-early' };
    }
  };

  const activeETAs = config.staticETAs ? computeStaticETAs(currentLoopIndex) : liveETAs;

  // --- Countdown summary ---
  // Parenthesized so the `as` cast applies to the property access, not the
  // result of &&. Previously eslint's no-unsafe-assignment flagged this.
  const selectedStopData = selectedStop
    ? (route?.[selectedStop] as ShuttleStopData | undefined)
    : undefined;
  const selectedStopName = selectedStopData?.NAME;

  // Compute next ETA at the selected stop (live or scheduled)
  let nextETAMinutes: number | null = null;
  let nextETATime: string | null = null;

  // Suppress unused variable warning — tick drives re-renders for countdown freshness
  void tick;

  // Track which trip rows have the soonest ETA for the selected stop, so
  // we can visually highlight them. When two shuttles have nearly-identical
  // ETAs at the same stop, both should be marked as "next arrivals" —
  // otherwise students looking at the un-marked row get misleading data.
  const soonestRowKeys = new Set<string>();
  let soonestETAMs = Infinity;
  // Tracks whether the soonest ETA is a next-loop projection (shuttle already
  // passed) rather than a direct current-trip ETA, so the summary can phrase
  // it as "Next arrival" vs "Arriving" for clarity.
  let soonestIsProjected = false;

  // Derive the route's loop duration (minutes) from the last stop's offset
  // so we can project the next loop's ETA when a shuttle has already
  // passed the selected stop.
  const routeLoopMinutes = (() => {
    if (!route?.STOPS || route.STOPS.length === 0) return 20;
    const lastStop = route.STOPS[route.STOPS.length - 1];
    const lastStopData = route[lastStop] as ShuttleStopData | undefined;
    return lastStopData?.OFFSET ?? 20;
  })();

  if (selectedStop && selectedDay === now.getDay()) {
    const nowMs = devNowMs();

    // First pass: find the absolute earliest ETA at selectedStop.
    // If a trip's ETA for this stop is stale or the shuttle has already
    // passed it, project the next-loop ETA by adding the route's loop
    // duration so the student still sees a useful countdown.
    const rowETAs: Array<{ rowKey: string; etaMs: number; etaISO: string; isProjected: boolean }> = [];
    for (const row of timelineRows) {
      if (!row.trip?.vehicle_id || !row.trip.stop_etas[selectedStop]) continue;
      // Completed trips are DONE — their shuttle has finished this loop
      // and any subsequent arrivals belong to a NEW trip row, not this
      // one. Projecting a next-loop ETA onto a completed trip caused
      // the row to get both the DONE badge and the NEXT highlight,
      // which contradicted the trip's finished state.
      if (row.trip.status === 'completed') continue;
      const stopInfo = row.trip.stop_etas[selectedStop];
      const etaISO = stopInfo.eta;
      let etaMs: number | null = null;
      let etaForDisplay: string | null | undefined = etaISO;
      let isProjected = false;

      const rawEtaMs = etaISO ? new Date(etaISO).getTime() : null;
      const baseMs = stopInfo.last_arrival ? new Date(stopInfo.last_arrival).getTime() : rawEtaMs;

      if (stopInfo.passed || stopInfo.last_arrival || (rawEtaMs !== null && rawEtaMs <= nowMs)) {
        // Stop is either passed or the ETA is stale — project to next loop
        if (baseMs !== null) {
          etaMs = baseMs + routeLoopMinutes * 60_000;
          etaForDisplay = new Date(etaMs).toISOString();
          isProjected = true;
        }
      } else if (rawEtaMs !== null) {
        etaMs = rawEtaMs;
        etaForDisplay = etaISO;
      }

      if (etaMs !== null && etaMs > nowMs && etaForDisplay) {
        rowETAs.push({ rowKey: row.key, etaMs, etaISO: etaForDisplay, isProjected });
        if (etaMs < soonestETAMs) soonestETAMs = etaMs;
      }
    }

    // Second pass: mark all rows within 60s of the earliest as "next"
    for (const r of rowETAs) {
      if (r.etaMs - soonestETAMs <= 60_000) {
        soonestRowKeys.add(r.rowKey);
        if (r.etaMs === soonestETAMs) {
          const mins = Math.round((r.etaMs - nowMs) / 60_000);
          nextETAMinutes = mins;
          nextETATime = new Date(r.etaISO).toLocaleTimeString(undefined, TIME_FORMAT);
          soonestIsProjected = r.isProjected;
        }
      }
    }

    // Fallback: try legacy per-stop aggregate
    if (nextETAMinutes === null) {
      const liveDetail = liveETADetails[selectedStop];
      if (liveDetail?.etaISO) {
        const etaDate = new Date(liveDetail.etaISO);
        const mins = Math.round((etaDate.getTime() - nowMs) / 60_000);
        if (mins > 0) {
          nextETAMinutes = mins;
          nextETATime = liveDetail.eta;
        }
      }
    }

    // Fallback: static scheduled time.
    // Scan all schedule times and find the earliest upcoming ETA for the
    // selected stop. This handles the pre-service-hours case (e.g., 3 AM
    // before 7 AM departures) where currentLoopIndex is -1.
    if (nextETAMinutes === null) {
      for (let i = 0; i < times.length; i++) {
        const dates = computeStaticETADates(i);
        const formatted = computeStaticETAs(i);
        const stopDate = dates[selectedStop];
        if (!stopDate) continue;
        const mins = Math.round((stopDate.getTime() - nowMs) / 60_000);
        if (mins > 0) {
          nextETAMinutes = mins;
          nextETATime = formatted[selectedStop];
          break;
        }
      }
    }
  }

  // --- Truncated view ---
  const shouldTruncate = isToday && !showFullSchedule && timelineRows.length > 7;

  // Find the index of the first current-row (earliest active shuttle) as
  // the anchor for the truncation window. If there's no active row, fall
  // back to the row whose originalIndex matches currentLoopIndex.
  let anchorIdx = timelineRows.findIndex(r => currentRowKeys.has(r.key));
  if (anchorIdx === -1 && currentLoopIndex >= 0) {
    anchorIdx = timelineRows.findIndex(r => r.originalIndex === currentLoopIndex);
  }
  if (anchorIdx === -1) anchorIdx = 0;

  const visibleItems: TimelineRow[] = shouldTruncate
    ? timelineRows.slice(Math.max(0, anchorIdx - 1), anchorIdx + 6)
    : timelineRows;

  return (
    <div className="schedule-container">
      <div className="schedule-header">
        <h2>Schedule</h2>
        <div className="schedule-controls">
          <div className="control-group">
            <label htmlFor='weekday-dropdown'>Day:</label>
            <select id='weekday-dropdown' className="schedule-dropdown" value={selectedDay} onChange={handleDayChange}>
              {daysOfTheWeek.map((day, index) => (
                <option key={index} value={index}>{day}</option>
              ))}
            </select>
          </div>
          <div className="control-group">
            <label htmlFor='route-toggle'>Route:</label>
            <div className="route-toggle">
              {routeNames.map((routeName, index) => (
                <button
                  key={index}
                  className={`route-toggle-button ${safeSelectedRoute === routeName ? 'active' : ''}`}
                  onClick={() => setSelectedRoute(routeName)}
                >
                  {routeName}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Countdown summary */}
      {isToday && selectedStopName && nextETAMinutes !== null && (
        <div className="next-shuttle-summary" role="status" aria-live="polite">
          {nextETAMinutes === 0 && !soonestIsProjected ? (
            <>
              Shuttle <strong>arriving now</strong> at{' '}
              <span className="summary-stop">{selectedStopName}</span>
            </>
          ) : (
            <>
              {soonestIsProjected ? 'Next loop at ' : 'Next at '}
              <span className="summary-stop">{selectedStopName}</span>
              {' '}in <strong>{
                nextETAMinutes >= 60
                  ? `${Math.floor(nextETAMinutes / 60)}h ${nextETAMinutes % 60}m`
                  : `${nextETAMinutes} min`
              }</strong>
              {nextETATime && <span className="summary-time"> ({nextETATime})</span>}
              {soonestIsProjected && (
                <div className="summary-note">
                  Shuttle just passed — this is the next loop.
                </div>
              )}
            </>
          )}
        </div>
      )}


      <div className="timeline-container">
        <div className="timeline-content">
          {(() => {
            // Auto-expand a SINGLE current row to show secondary stops.
            // Priority:
            //   1. The row with the soonest ETA at the selected stop
            //   2. Otherwise the first current row
            // Other current rows collapse by default so users can compare
            // shuttles at-a-glance without scrolling through huge lists.
            const currentItems = visibleItems.filter(r => currentRowKeys.has(r.key));
            let autoExpandKey: string | null = null;
            if (currentItems.length > 0) {
              // Prefer a row marked as soonest
              const soonest = currentItems.find(r => soonestRowKeys.has(r.key));
              autoExpandKey = (soonest ?? currentItems[0]).key;
            }
            return visibleItems.map((row) => {
          const { key: rowKey, time, timeDate, trip, originalIndex } = row;
          const isCurrentLoop = currentRowKeys.has(rowKey);
          // A row is "past-time" only if its scheduled time is in the past
          // AND it isn't an active trip. Active shuttles that departed
          // earlier than now are NOT past — they're currently running.
          const isPastTime = isToday && timeDate < now && !isCurrentLoop;
          const isExpanded = expandedLoops.has(rowKey);
          // Only the auto-expanded current row shows stops by default.
          // Other current rows collapse unless the user manually expands.
          const showSecondary = (isCurrentLoop && rowKey === autoExpandKey) || isExpanded;

            // Get first stop info
            const firstStop = route?.STOPS?.[0];
            const firstStopData = firstStop ? route[firstStop] as ShuttleStopData : null;

            // Relative time label (upcoming departures within 30 min)
            const minutesUntil = Math.round((timeDate.getTime() - now.getTime()) / 60_000);
            const showRelative = isToday && !isPastTime && !isCurrentLoop && minutesUntil > 0 && minutesUntil <= 30;

            // This row's trip (null for scheduled-only rows with no vehicle)
            const loopTrip: Trip | null = trip;
            // Static ETAs for this row — use loopIndex when it's a scheduled
            // row, else compute from the injected trip's actual departure time.
            const staticForRow: StopETAs = originalIndex >= 0
              ? computeStaticETAs(originalIndex)
              : computeStaticETAsForDate(timeDate);
            const staticDatesForRow: Record<string, Date> = originalIndex >= 0
              ? computeStaticETADates(originalIndex)
              : computeStaticETADatesForDate(timeDate);
            // Legacy fallback ETAs (global per-stop aggregate, used when no trip)
            const loopETAs: StopETAs = isCurrentLoop ? activeETAs : staticForRow;
            const loopStaticETAs: StopETAs = staticForRow;
            const loopStaticDates: Record<string, Date> = staticDatesForRow;

            // Helper: get trip-based ETA info for a stop. Only returns data
            // for trips with an assigned vehicle — scheduled-only trips
            // (no vehicle_id) have static stop_etas that shouldn't be
            // displayed as LIVE.
            const isCompletedTrip = loopTrip?.status === 'completed';

            const getTripStopInfo = (stop: string) => {
              if (!loopTrip || !loopTrip.vehicle_id || !loopTrip.stop_etas[stop]) return null;
              const info = loopTrip.stop_etas[stop];
              const nowMs = devNowMs();
              let etaTime: string | undefined;
              // Interpolated passes come from the backend's detection-gap
              // backfill — the `last_arrival` timestamp is a linear guess
              // between two real anchors, not a real detection. Surface the
              // "passed" state but don't render a fabricated "Last: HH:MM"
              // that the user might take as a real arrival time.
              const interpolated = !!info.passed_interpolated;
              // Completed trips: suppress all future ETAs — the loop is done.
              if (isCompletedTrip) {
                return {
                  etaTime: undefined,
                  lastArrival: !interpolated && info.last_arrival
                    ? new Date(info.last_arrival).toLocaleTimeString(undefined, TIME_FORMAT)
                    : undefined,
                  passed: true,
                  passedInterpolated: interpolated,
                };
              }
              if (info.eta) {
                const etaDate = new Date(info.eta);
                // Grace window: keep showing the ETA for up to 120s after it
                // expires so stops don't flip to "Passed" just because the
                // backend hasn't re-computed yet. If the shuttle's been stuck
                // or hasn't triggered detection, the stale ETA is still more
                // useful than an inferred "Passed".
                if (etaDate.getTime() > nowMs - 120_000) {
                  etaTime = etaDate.toLocaleTimeString(undefined, TIME_FORMAT);
                }
              }
              let lastArrival: string | undefined;
              if (info.last_arrival && !interpolated) {
                // Hide egregiously stale "Last:" timestamps. With the
                // segment-based drive-by detection in ml/data/stops.py
                // (add_stops_from_segments), most passes now register
                // even when a shuttle moves faster than the 20m radius
                // threshold, so a missed detection is rare. The 60-min
                // ceiling catches the unlikely case where a stop is
                // missed across multiple loops AND the shuttle is still
                // running (idle shuttles are already filtered upstream
                // by backend/worker/trips.py).
                const laMs = new Date(info.last_arrival).getTime();
                if (laMs > nowMs - 60 * 60_000) {
                  lastArrival = new Date(info.last_arrival).toLocaleTimeString(undefined, TIME_FORMAT);
                }
              }
              return { etaTime, lastArrival, passed: info.passed, passedInterpolated: interpolated };
            };

            const firstStopIsSelected = !!firstStop && firstStop === selectedStop;

            return (
              <div key={rowKey} className="timeline-route-group">
                {/* First stop - main timeline item.
                    Click behavior:
                      - Current/past rows: click the stop NAME to select,
                        click anywhere else is a no-op
                      - Expandable (future) rows: click anywhere to expand,
                        but click on the stop NAME still selects it */}
                <div
                  className={`timeline-item first-stop ${isCurrentLoop ? 'current-loop' : ''} ${isPastTime && !isCurrentLoop ? 'past-time' : ''} ${soonestRowKeys.has(rowKey) ? 'soonest-arrival' : ''} ${firstStopIsSelected ? 'first-stop-selected' : ''}`}
                  role="button"
                  tabIndex={isPastTime || (isCurrentLoop && rowKey === autoExpandKey) ? -1 : 0}
                  aria-expanded={isExpanded}
                  onClick={() => {
                    // Toggle expand unless this is a past row or the
                    // auto-expanded current row (which stays open).
                    if (isPastTime) return;
                    if (isCurrentLoop && rowKey === autoExpandKey) return;
                    toggleExpand(rowKey);
                  }}
                  onKeyDown={(e) => {
                    if (isPastTime) return;
                    if (isCurrentLoop && rowKey === autoExpandKey) return;
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      toggleExpand(rowKey);
                    }
                  }}
                  style={!isPastTime && !(isCurrentLoop && rowKey === autoExpandKey) ? POINTER_CURSOR_STYLE : undefined}
                >
                  <div className="timeline-dot"></div>
                  <div className="timeline-content-item">
                    <div className="timeline-time">
                      <span className="timeline-time-text">{time}</span>
                      {(() => {
                        // Departure deviation label (260415-3ec).
                        // Hugs the row anchor time so late/early/off-schedule
                        // trips are clearly communicated without disturbing
                        // row sort order or the expand/current-loop logic.
                        const lbl = loopTrip ? getDepartureLabel(loopTrip, times, selectedDay, timeToDate) : null;
                        if (!lbl) return null;
                        return (
                          <span className={`timeline-deviation deviation-${lbl.kind}`}>
                            {lbl.text}
                          </span>
                        );
                      })()}
                      {loopTrip?.vehicle_id && (
                        <span className="vehicle-badge" aria-label={`Shuttle ${loopTrip.vehicle_id.slice(-3)}`}>#{loopTrip.vehicle_id.slice(-3)}</span>
                      )}
                      {showRelative && <span className="relative-time">in {minutesUntil} min</span>}
                      {isCompletedTrip && <span className="trip-completed-badge">DONE</span>}
                      {(() => {
                        if (isCompletedTrip) return null;
                        // Prefer trip-based first-stop ETA if loop has a trip
                        // WITH a vehicle (live prediction source).
                        if (loopTrip && firstStop) {
                          const ti = getTripStopInfo(firstStop);
                          if (ti?.etaTime) {
                            return (
                              <>
                                <span className="live-eta"> - ETA: {ti.etaTime}</span>
                                <span className="source-badge source-live">LIVE</span>
                              </>
                            );
                          }
                          // loopTrip exists but getTripStopInfo returned no
                          // etaTime. Two possible reasons:
                          //   1. Unassigned scheduled trip (vehicle_id=null) —
                          //      there's NO live prediction source for this
                          //      row, so a "LIVE" badge is a lie regardless of
                          //      what activeETAs says. Show nothing.
                          //   2. loopTrip has a vehicle but STUDENT_UNION.eta
                          //      is null because the shuttle already departed
                          //      (passed=True with a real last_arrival). In
                          //      that case the last_arrival is rendered
                          //      elsewhere; nothing to show here either.
                          // In both cases, skip the fallback to prevent
                          // activeETAs from surfacing another trip's static
                          // departure time with a misleading "LIVE" badge.
                          return null;
                        }
                        // Pure scheduled row with NO loopTrip at all (no trip
                        // matched this row). Legacy fallback: if the global
                        // activeETAs has a live prediction for this route's
                        // first stop AND this is the current loop, show it.
                        // This only fires on routes where the backend hasn't
                        // produced a trip for this row, which is rare.
                        const fRouteKey = `${firstStop}:${safeSelectedRoute}`;
                        const fEta = activeETAs[fRouteKey] || activeETAs[firstStop];
                        const fDetails = liveETADetails[fRouteKey] || liveETADetails[firstStop];
                        const fMatch = !fDetails?.route || fDetails.route === safeSelectedRoute;
                        return isCurrentLoop && fEta && fMatch ? (
                          <>
                            <span className="live-eta"> - ETA: {fEta}</span>
                            <span className="source-badge source-live">LIVE</span>
                          </>
                        ) : null;
                      })()}
                      {!isPastTime && (!isCurrentLoop || rowKey !== autoExpandKey) && (
                        <span className="expand-indicator">{isExpanded ? '\u25B4' : '\u25BE'}</span>
                      )}
                    </div>
                    <div
                      className="timeline-stop"
                      role="button"
                      tabIndex={0}
                      aria-label={`Select ${firstStopData?.NAME || 'stop'} for arrival countdown`}
                      aria-pressed={firstStopIsSelected}
                      onClick={(e) => {
                        if (firstStop) {
                          e.stopPropagation();
                          handleStopSelect(firstStop);
                        }
                      }}
                      onKeyDown={(e) => {
                        if (firstStop && (e.key === 'Enter' || e.key === ' ')) {
                          e.preventDefault();
                          handleStopSelect(firstStop);
                        }
                      }}
                      style={POINTER_CURSOR_STYLE}
                    >
                      {firstStopData?.NAME || 'Unknown Stop'}
                    </div>
                  </div>
                </div>

                {/* Secondary stops */}
                {showSecondary && route?.STOPS && route.STOPS.length > 1 && (() => {
                  // Pre-compute stop live info. Prefer trip-based ETAs if this
                  // loop has an assigned trip; otherwise fall back to global ETAs.
                  const secondaryStops = route.STOPS.slice(1);
                  const stopInfo = secondaryStops.map(stop => {
                    if (loopTrip) {
                      const ti = getTripStopInfo(stop);
                      if (ti) {
                        const hasETA = !!ti.etaTime;
                        const hasLast = !!ti.lastArrival;
                        return {
                          stop,
                          hasETA,
                          hasLast,
                          hasAnyLive: hasETA || hasLast,
                          etaTime: ti.etaTime,
                          lastArrival: ti.lastArrival,
                          passed: ti.passed,
                        } as const;
                      }
                    }
                    // Legacy fallback. Only fires when loopTrip is null or
                    // getTripStopInfo returned null (unassigned scheduled row
                    // or stop missing from trip.stop_etas). `activeETAs` and
                    // `liveETADetails` from deriveStopEtasFromTrips are
                    // aggregated ACROSS routes — a stop that appears on both
                    // NORTH and WEST (Student Union, Student Union Return) is
                    // stored once, owned by whichever route has the earliest
                    // future ETA or most recent last_arrival. Gate BOTH the
                    // ETA and the last_arrival reads on `routeMatch` so a
                    // NORTH shuttle's data never leaks onto a WEST row.
                    const rk = `${stop}:${safeSelectedRoute}`;
                    const eta = activeETAs[rk] || activeETAs[stop];
                    const det = liveETADetails[rk] || liveETADetails[stop];
                    const routeMatch = !det?.route || det.route === safeSelectedRoute;
                    const hasETA = !!(isCurrentLoop && eta && routeMatch);
                    const hasLast = !!(isCurrentLoop && det?.lastArrival && routeMatch);
                    return {
                      stop,
                      hasETA,
                      hasLast,
                      hasAnyLive: hasETA || hasLast,
                      etaTime: routeMatch ? eta : undefined,
                      lastArrival: routeMatch ? det?.lastArrival : undefined,
                      passed: false,
                    } as const;
                  });

                  // Single-pass inferredPassed: scan forward/backward once
                  const n = stopInfo.length;
                  // hasLiveBefore[i] = true if any stop before i has live data
                  const hasLiveBefore: boolean[] = Array.from({ length: n }, () => false);
                  hasLiveBefore[0] = true; // first secondary stop: departure serves as "before"
                  for (let i = 1; i < n; i++) {
                    hasLiveBefore[i] = hasLiveBefore[i - 1] || stopInfo[i - 1].hasAnyLive;
                  }
                  // hasLiveAfter[i] = true if any stop after i has live data
                  const hasLiveAfter: boolean[] = Array.from({ length: n }, () => false);
                  for (let i = n - 2; i >= 0; i--) {
                    hasLiveAfter[i] = hasLiveAfter[i + 1] || stopInfo[i + 1].hasAnyLive;
                  }
                  // allPriorLive[i] = true if every stop before i has live data
                  const allPriorLive: boolean[] = Array.from({ length: n }, () => false);
                  allPriorLive[0] = true;
                  for (let i = 1; i < n; i++) {
                    allPriorLive[i] = allPriorLive[i - 1] && stopInfo[i - 1].hasAnyLive;
                  }

                  return (
                  <div className="secondary-timeline">
                    {stopInfo.map((si, stopIndex) => {
                      const stop = si.stop;
                      const stopData = route[stop] as ShuttleStopData;
                      const etaTime = si.etaTime;
                      const hasLiveETA = si.hasETA;
                      // Show last_arrival for any loop with trip data, or current loop with legacy data
                      const showLast = !!loopTrip || isCurrentLoop;
                      const lastArrival = showLast ? si.lastArrival : undefined;
                      const inferredPassed = showLast && !hasLiveETA && !lastArrival
                        && hasLiveBefore[stopIndex] && (hasLiveAfter[stopIndex] || (allPriorLive[stopIndex] && stopIndex > 0));
                      const isSelected = stop === selectedStop;

                      return (
                        <div
                          key={stopIndex}
                          className={`secondary-timeline-item ${isPastTime && !isCurrentLoop ? 'past-time' : ''} ${hasLiveETA ? 'has-eta' : ''} ${isSelected ? 'selected-stop' : ''}`}
                          onClick={() => handleStopSelect(stop)}
                          style={POINTER_CURSOR_STYLE}
                          role="button"
                          tabIndex={0}
                          aria-label={`Select ${stopData?.NAME || stop} for arrival countdown`}
                          aria-pressed={isSelected}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              handleStopSelect(stop);
                            }
                          }}
                        >
                          <div className="secondary-timeline-dot"></div>
                          <div className="secondary-timeline-content">
                            <div className="secondary-timeline-time">
                              {hasLiveETA ? (
                                <>
                                  <span className="live-eta">{etaTime}</span>
                                  <span className="source-badge source-live">LIVE</span>
                                  {(() => {
                                    // Suppress deviation for injected trips — they
                                    // have no real schedule to be late/early against,
                                    // only a projection from their own actual departure.
                                    if (loopTrip && !loopTrip.scheduled) return null;
                                    // Suppress deviation when the shuttle has clearly
                                    // looped past its originally-matched departure —
                                    // the backend keeps the same trip object across
                                    // loops, so the "+NN min late" labels become
                                    // cumulative and misleading. Detection: if the
                                    // current ETA to any NEAR stop exceeds loop
                                    // duration (~15 min) from the trip's scheduled
                                    // departure, the shuttle is on a later loop.
                                    if (loopTrip?.departure_time) {
                                      const schedMs = new Date(loopTrip.departure_time).getTime();
                                      const tripEtaISOCheck = loopTrip.stop_etas[stop]?.eta;
                                      if (tripEtaISOCheck) {
                                        const etaMs = new Date(tripEtaISOCheck).getTime();
                                        // ETA is > routeLoopMinutes after scheduled
                                        // departure + this stop's offset → next loop
                                        const stopOffsetMin = loopStaticDates[stop]
                                          ? (loopStaticDates[stop].getTime() - schedMs) / 60_000
                                          : 0;
                                        const expectedEtaMs = schedMs + (stopOffsetMin + routeLoopMinutes) * 60_000;
                                        if (etaMs >= expectedEtaMs) return null;
                                      }
                                    }
                                    // For scheduled trip rows, use the trip's own ETA
                                    // so deviation matches the displayed time.
                                    const tripEtaISO = loopTrip?.stop_etas[stop]?.eta ?? null;
                                    const dev = computeDeviation(stop, loopStaticDates, tripEtaISO);
                                    return dev ? <span className={dev.className}>{dev.text}</span> : null;
                                  })()}
                                </>
                              ) : lastArrival ? (
                                <>
                                  <span className="last-arrival">Last: {lastArrival}</span>
                                  <span className="source-badge source-live">LIVE</span>
                                </>
                              ) : inferredPassed ? (
                                <>
                                  <span className="last-arrival">Passed</span>
                                  <span className="source-badge source-live">LIVE</span>
                                </>
                              ) : (isCurrentLoop || isExpanded) && (loopETAs[stop] || loopStaticETAs[stop]) ? (
                                <>
                                  <span className="scheduled-fallback">{loopETAs[stop] || loopStaticETAs[stop]}</span>
                                  <span className="source-badge source-sched">SCHED</span>
                                </>
                              ) : (
                                <span className="no-service-message">{missingDataMessage}</span>
                              )}
                            </div>
                            <div className="secondary-timeline-stop">{stopData?.NAME || stop}</div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  );
                })()}
              </div>
            );
          });
          })()}
        </div>

        {/* Show full schedule link */}
        {shouldTruncate && (
          <button className="show-full-schedule" onClick={() => setShowFullSchedule(true)}>
            Show full schedule ({timelineRows.length - visibleItems.length} more)
          </button>
        )}
        {showFullSchedule && isToday && timelineRows.length > 7 && (
          <button className="show-full-schedule" onClick={() => setShowFullSchedule(false)}>
            Show less
          </button>
        )}
      </div>
    </div>
  );
}
