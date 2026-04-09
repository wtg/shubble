import { useState, useEffect, useCallback } from 'react';
import './styles/Schedule.css';
import rawRouteData from '../shared/routes.json';
import rawAggregatedSchedule from '../shared/aggregated_schedule.json';
import config from '../utils/config';
import type { AggregatedDaySchedule, AggregatedScheduleType, Route } from '../types/schedule';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';
import { useStopETAs, type StopETAs, type StopETADetails } from '../hooks/useStopETAs';
import { useTrips, type Trip } from '../hooks/useTrips';
import { useClosestStop } from '../hooks/useClosestStop';

const aggregatedSchedule: AggregatedScheduleType = rawAggregatedSchedule as unknown as AggregatedScheduleType;
const routeData = rawRouteData as unknown as ShuttleRouteData;

import { devNow, devNowMs } from '../utils/devTime';

const TIME_FORMAT: Intl.DateTimeFormatOptions = { hour: 'numeric', minute: '2-digit' };

type ScheduleProps = {
  selectedRoute: string | null;
  setSelectedRoute: (route: string | null) => void;
  stopETAs?: StopETAs;
  stopETADetails?: StopETADetails;
};

export default function Schedule({ selectedRoute, setSelectedRoute, stopETAs: externalStopETAs, stopETADetails: externalDetails }: ScheduleProps) {
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }

  const now = devNow();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [schedule, setSchedule] = useState<AggregatedDaySchedule>(aggregatedSchedule[selectedDay]);
  const [showFullSchedule, setShowFullSchedule] = useState(false);
  const [expandedLoops, setExpandedLoops] = useState<Set<string>>(new Set());
  const [selectedStop, setSelectedStop] = useState<string | null>(
    () => localStorage.getItem('shubble-stop')
  );
  const [tick, setTick] = useState(0); // for countdown re-renders

  // Use external ETAs if provided (from parent's shared hook), otherwise fetch our own
  const hasExternalETAs = externalStopETAs !== undefined;
  const { stopETAs: internalStopETAs, stopETADetails: internalDetails } = useStopETAs(
    !config.staticETAs && !hasExternalETAs
  );
  const liveETAs = externalStopETAs ?? internalStopETAs;
  const liveETADetails = externalDetails ?? internalDetails;

  // Per-trip ETAs from the new /api/trips endpoint. Fetched regardless of
  // whether stopETAs are passed externally — trips are always needed to
  // render the per-vehicle row model.
  const trips = useTrips(!config.staticETAs);

  const safeSelectedRoute = selectedRoute || routeNames[0];

  // Trips for the selected route only (row model built below, after `times`).
  const routeTrips: Trip[] = trips.filter(t => t.route === safeSelectedRoute);

  // Auto-detect closest stop via geolocation
  const { closestStop } = useClosestStop(safeSelectedRoute);

  // Initialize selectedStop from geolocation if not already set
  useEffect(() => {
    if (!selectedStop && closestStop) {
      setSelectedStop(closestStop.id);
      localStorage.setItem('shubble-stop', closestStop.id);
    }
  }, [closestStop, selectedStop]);

  // When the user switches routes, the previously-selected stop may not
  // exist on the new route (e.g. City Station → NORTH). Without a reset
  // the countdown summary silently disappears. Switch to the closest stop
  // on the new route so the student always has a live countdown.
  useEffect(() => {
    const currentRoute = routeData[safeSelectedRoute as keyof typeof routeData];
    if (!currentRoute?.STOPS) return;
    if (selectedStop && !currentRoute.STOPS.includes(selectedStop)) {
      // Prefer the geo-based closest stop when available. Otherwise pick
      // the second stop (index 1) rather than index 0 — the first stop is
      // Student Union, which shuttles pass through continuously, so its
      // ETA is almost always "passed" and produces no countdown.
      let fallback: string;
      if (closestStop?.id && currentRoute.STOPS.includes(closestStop.id)) {
        fallback = closestStop.id;
      } else if (currentRoute.STOPS.length > 1) {
        fallback = currentRoute.STOPS[1];
      } else {
        fallback = currentRoute.STOPS[0];
      }
      setSelectedStop(fallback);
      localStorage.setItem('shubble-stop', fallback);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeSelectedRoute]);

  // Persist stop selection
  const handleStopSelect = useCallback((stopId: string) => {
    setSelectedStop(stopId);
    localStorage.setItem('shubble-stop', stopId);
  }, []);

  // Re-render countdown every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => setTick(t => t + 1), 30_000);
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

  const timeToDate = (timeStr: string): Date => {
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
    return dateObj;
  };

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

  // Scroll to current time on route/day change (only within the timeline container)
  useEffect(() => {
    const timelineContainer = document.querySelector('.timeline-container') as HTMLElement;
    if (!timelineContainer) return;
    if (selectedDay !== now.getDay()) return;

    // Prioritize the item with current-loop class (where ETAs are displayed)
    let targetItem = timelineContainer.querySelector('.timeline-item.current-loop') as HTMLElement | null;

    // Fall back to first non-past item if no current loop
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
  }, [selectedRoute, selectedDay, schedule]);

  const toggleExpand = (key: string) => {
    setExpandedLoops(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  const times = schedule[safeSelectedRoute as Route] || [];
  const route = routeData[safeSelectedRoute as keyof typeof routeData];
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

  // Compute static ETAs from route offsets given an absolute departure Date.
  // Used directly for injected trips (which have no schedule index) and
  // indirectly via computeStaticETAs(loopIndex) for scheduled rows.
  const computeStaticETAsForDate = (departureTime: Date): StopETAs => {
    if (!route?.STOPS) return {};
    const stopETAs: StopETAs = {};
    for (const stopKey of route.STOPS) {
      const stopData = route[stopKey] as ShuttleStopData;
      if (!stopData) continue;
      const etaDate = new Date(departureTime.getTime() + stopData.OFFSET * 60_000);
      stopETAs[stopKey] = etaDate.toLocaleTimeString(undefined, TIME_FORMAT);
    }
    return stopETAs;
  };

  const computeStaticETADatesForDate = (departureTime: Date): Record<string, Date> => {
    if (!route?.STOPS) return {};
    const dates: Record<string, Date> = {};
    for (const stopKey of route.STOPS) {
      const stopData = route[stopKey] as ShuttleStopData;
      if (!stopData) continue;
      dates[stopKey] = new Date(departureTime.getTime() + stopData.OFFSET * 60_000);
    }
    return dates;
  };

  // Compute static ETAs from route offsets for a scheduled loop index.
  const computeStaticETAs = (loopIndex: number): StopETAs => {
    if (loopIndex < 0) return {};
    return computeStaticETAsForDate(timeToDate(times[loopIndex]));
  };

  // Compute static ETA Dates for deviation comparison
  const computeStaticETADates = (loopIndex: number): Record<string, Date> => {
    if (loopIndex < 0) return {};
    return computeStaticETADatesForDate(timeToDate(times[loopIndex]));
  };

  // Determine contextual message when no ETA data is available (D-08)
  const getMissingDataMessage = (routeName: string): string => {
    const daySchedule = aggregatedSchedule[selectedDay];
    const routeTimes = daySchedule?.[routeName as Route];

    if (!routeTimes || routeTimes.length === 0) {
      return 'No shuttle in service';
    }

    const firstDeparture = timeToDate(routeTimes[0]);
    const lastDeparture = timeToDate(routeTimes[routeTimes.length - 1]);

    if (now < firstDeparture) {
      return `Service starts at ${routeTimes[0]}`;
    }

    if (now > lastDeparture) {
      return 'No shuttle in service';
    }

    // During service hours but no data for this stop
    return 'No shuttle in service';
  };

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
  const staticETAs = computeStaticETAs(currentLoopIndex);
  const staticETADates = computeStaticETADates(currentLoopIndex);

  // --- Countdown summary ---
  const selectedStopData = selectedStop && route?.[selectedStop] as ShuttleStopData | undefined;
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
    const rowETAs: Array<{ rowKey: string; etaMs: number; etaISO: string }> = [];
    for (const row of timelineRows) {
      if (!row.trip?.vehicle_id || !row.trip.stop_etas[selectedStop]) continue;
      const stopInfo = row.trip.stop_etas[selectedStop];
      const etaISO = stopInfo.eta;
      let etaMs: number | null = null;
      let etaForDisplay: string | null | undefined = etaISO;

      const rawEtaMs = etaISO ? new Date(etaISO).getTime() : null;
      const baseMs = stopInfo.last_arrival ? new Date(stopInfo.last_arrival).getTime() : rawEtaMs;

      if (stopInfo.passed || stopInfo.last_arrival || (rawEtaMs !== null && rawEtaMs <= nowMs)) {
        // Stop is either passed or the ETA is stale — project to next loop
        if (baseMs !== null) {
          etaMs = baseMs + routeLoopMinutes * 60_000;
          etaForDisplay = new Date(etaMs).toISOString();
        }
      } else if (rawEtaMs !== null) {
        etaMs = rawEtaMs;
        etaForDisplay = etaISO;
      }

      if (etaMs !== null && etaMs > nowMs && etaForDisplay) {
        rowETAs.push({ rowKey: row.key, etaMs, etaISO: etaForDisplay });
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

    // Fallback: static scheduled time
    if (nextETAMinutes === null && staticETADates[selectedStop]) {
      const scheduledDate = staticETADates[selectedStop];
      const mins = Math.round((scheduledDate.getTime() - nowMs) / 60_000);
      if (mins > 0) {
        nextETAMinutes = mins;
        nextETATime = staticETAs[selectedStop];
      } else {
        const nextLoopIdx = currentLoopIndex + 1;
        if (nextLoopIdx < times.length) {
          const nextDates = computeStaticETADates(nextLoopIdx);
          const nextETAsFormatted = computeStaticETAs(nextLoopIdx);
          if (nextDates[selectedStop]) {
            const nextMins = Math.round((nextDates[selectedStop].getTime() - nowMs) / 60_000);
            if (nextMins > 0) {
              nextETAMinutes = nextMins;
              nextETATime = nextETAsFormatted[selectedStop];
            }
          }
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
        <div className="next-shuttle-summary">
          Next at{' '}
          <span className="summary-stop">{selectedStopName}</span>
          {' '}in <strong>{nextETAMinutes} min</strong>
          {nextETATime && <span className="summary-time"> ({nextETATime})</span>}
        </div>
      )}

      {/* First-time hint: appears only when no stop has been selected yet,
          teaching the user that tapping a stop gives them a countdown. */}
      {isToday && !selectedStop && (
        <div className="stop-select-hint" role="note">
          Tap a stop below to see the live countdown for that stop.
        </div>
      )}

      <div className="timeline-container">
        <div className="timeline-content">
          {visibleItems.map((row) => {
            const { key: rowKey, time, timeDate, trip, originalIndex } = row;
            const isPastTime = isToday && timeDate < now;
            const isCurrentLoop = currentRowKeys.has(rowKey);
            const isExpanded = expandedLoops.has(rowKey);
            const showSecondary = isCurrentLoop || isExpanded;

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
              // Completed trips: suppress all future ETAs — the loop is done.
              if (isCompletedTrip) {
                return { etaTime: undefined, lastArrival: info.last_arrival ? new Date(info.last_arrival).toLocaleTimeString(undefined, TIME_FORMAT) : undefined, passed: true };
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
              if (info.last_arrival) {
                lastArrival = new Date(info.last_arrival).toLocaleTimeString(undefined, TIME_FORMAT);
              }
              return { etaTime, lastArrival, passed: info.passed };
            };

            return (
              <div key={rowKey} className="timeline-route-group">
                {/* First stop - main timeline item */}
                <div
                  className={`timeline-item first-stop ${isCurrentLoop ? 'current-loop' : ''} ${isPastTime && !isCurrentLoop ? 'past-time' : ''} ${soonestRowKeys.has(rowKey) ? 'soonest-arrival' : ''}`}
                  onClick={() => !isPastTime && !isCurrentLoop && toggleExpand(rowKey)}
                  style={!isPastTime && !isCurrentLoop ? { cursor: 'pointer' } : undefined}
                >
                  <div className="timeline-dot"></div>
                  <div className="timeline-content-item">
                    <div className="timeline-time">
                      <span className="timeline-time-text">{time}</span>
                      {loopTrip?.vehicle_id && (
                        <span className="vehicle-badge" aria-label={`Shuttle ${loopTrip.vehicle_id.slice(-3)}`}>#{loopTrip.vehicle_id.slice(-3)}</span>
                      )}
                      {showRelative && <span className="relative-time">in {minutesUntil} min</span>}
                      {isCompletedTrip && <span className="trip-completed-badge">DONE</span>}
                      {(() => {
                        if (isCompletedTrip) return null;
                        // Prefer trip-based first-stop ETA if loop has a trip
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
                        }
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
                      {!isCurrentLoop && !isPastTime && (
                        <span className="expand-indicator">{isExpanded ? '\u25B4' : '\u25BE'}</span>
                      )}
                    </div>
                    <div className="timeline-stop">{firstStopData?.NAME || 'Unknown Stop'}</div>
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
                    // Legacy fallback
                    const rk = `${stop}:${safeSelectedRoute}`;
                    const eta = activeETAs[rk] || activeETAs[stop];
                    const det = liveETADetails[rk] || liveETADetails[stop];
                    const routeMatch = !det?.route || det.route === safeSelectedRoute;
                    const hasETA = !!(isCurrentLoop && eta && routeMatch);
                    const hasLast = !!(isCurrentLoop && det?.lastArrival);
                    return {
                      stop,
                      hasETA,
                      hasLast,
                      hasAnyLive: hasETA || hasLast,
                      etaTime: eta,
                      lastArrival: det?.lastArrival,
                      passed: false,
                    } as const;
                  });

                  // Single-pass inferredPassed: scan forward/backward once
                  const n = stopInfo.length;
                  // hasLiveBefore[i] = true if any stop before i has live data
                  const hasLiveBefore: boolean[] = new Array(n);
                  hasLiveBefore[0] = true; // first secondary stop: departure serves as "before"
                  for (let i = 1; i < n; i++) {
                    hasLiveBefore[i] = hasLiveBefore[i - 1] || stopInfo[i - 1].hasAnyLive;
                  }
                  // hasLiveAfter[i] = true if any stop after i has live data
                  const hasLiveAfter: boolean[] = new Array(n);
                  hasLiveAfter[n - 1] = false;
                  for (let i = n - 2; i >= 0; i--) {
                    hasLiveAfter[i] = hasLiveAfter[i + 1] || stopInfo[i + 1].hasAnyLive;
                  }
                  // allPriorLive[i] = true if every stop before i has live data
                  const allPriorLive: boolean[] = new Array(n);
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
                          style={{ cursor: 'pointer' }}
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
                                <span className="no-service-message">{getMissingDataMessage(safeSelectedRoute)}</span>
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
          })}
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
