import { useState, useEffect, useCallback } from 'react';
import './styles/Schedule.css';
import rawRouteData from '../shared/routes.json';
import rawAggregatedSchedule from '../shared/aggregated_schedule.json';
import config from '../utils/config';
import type { AggregatedDaySchedule, AggregatedScheduleType, Route } from '../types/schedule';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';
import { useStopETAs, type StopETAs, type StopETADetails } from '../hooks/useStopETAs';
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
  const [expandedLoops, setExpandedLoops] = useState<Set<number>>(new Set());
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

  const safeSelectedRoute = selectedRoute || routeNames[0];

  // Auto-detect closest stop via geolocation
  const { closestStop } = useClosestStop(safeSelectedRoute);

  // Initialize selectedStop from geolocation if not already set
  useEffect(() => {
    if (!selectedStop && closestStop) {
      setSelectedStop(closestStop.id);
      localStorage.setItem('shubble-stop', closestStop.id);
    }
  }, [closestStop, selectedStop]);

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
    const [time, modifier] = timeStr.trim().split(" ");
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

  const toggleExpand = (idx: number) => {
    setExpandedLoops(prev => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  };

  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  const times = schedule[safeSelectedRoute as Route] || [];
  const route = routeData[safeSelectedRoute as keyof typeof routeData];
  const currentLoopIndex = findCurrentLoopIndex(times);

  // Compute static ETAs from route offsets for a given loop
  const computeStaticETAs = (loopIndex: number): StopETAs => {
    if (loopIndex < 0 || !route?.STOPS) return {};

    const departureTime = timeToDate(times[loopIndex]);
    const stopETAs: StopETAs = {};

    for (const stopKey of route.STOPS) {
      const stopData = route[stopKey] as ShuttleStopData;
      if (!stopData) continue;

      const etaDate = new Date(departureTime.getTime() + stopData.OFFSET * 60_000);
      stopETAs[stopKey] = etaDate.toLocaleTimeString(undefined, TIME_FORMAT);
    }

    return stopETAs;
  };

  // Compute static ETA Dates for deviation comparison
  const computeStaticETADates = (loopIndex: number): Record<string, Date> => {
    if (loopIndex < 0 || !route?.STOPS) return {};
    const departureTime = timeToDate(times[loopIndex]);
    const dates: Record<string, Date> = {};
    for (const stopKey of route.STOPS) {
      const stopData = route[stopKey] as ShuttleStopData;
      if (!stopData) continue;
      dates[stopKey] = new Date(departureTime.getTime() + stopData.OFFSET * 60_000);
    }
    return dates;
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

  // Compute early/late deviation for live ETAs (D-05, D-06, D-07, D-11, D-12)
  const computeDeviation = (
    stopKey: string,
    loopStaticDatesForStop: Record<string, Date>
  ): { text: string; className: string } | null => {
    const detail = liveETADetails[stopKey];
    if (!detail?.etaISO) return null;

    const scheduledDate = loopStaticDatesForStop[stopKey];
    if (!scheduledDate) return null;

    const liveDate = new Date(detail.etaISO);
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

  if (selectedStop && selectedDay === now.getDay()) {
    // Try live ETA first
    const liveDetail = liveETADetails[selectedStop];
    if (liveDetail?.etaISO) {
      const etaDate = new Date(liveDetail.etaISO);
      const mins = Math.round((etaDate.getTime() - devNowMs()) / 60_000);
      if (mins > 0) {
        nextETAMinutes = mins;
        nextETATime = liveDetail.eta;
      }
    }
    // Fallback to static scheduled time
    if (nextETAMinutes === null && staticETADates[selectedStop]) {
      const scheduledDate = staticETADates[selectedStop];
      const mins = Math.round((scheduledDate.getTime() - devNowMs()) / 60_000);
      if (mins > 0) {
        nextETAMinutes = mins;
        nextETATime = staticETAs[selectedStop];
      } else {
        // Current loop's time has passed — check next loop
        const nextLoopIdx = currentLoopIndex + 1;
        if (nextLoopIdx < times.length) {
          const nextDates = computeStaticETADates(nextLoopIdx);
          const nextETAsFormatted = computeStaticETAs(nextLoopIdx);
          if (nextDates[selectedStop]) {
            const nextMins = Math.round((nextDates[selectedStop].getTime() - devNowMs()) / 60_000);
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
  const isToday = selectedDay === now.getDay();
  const shouldTruncate = isToday && !showFullSchedule && times.length > 7;

  const visibleItems = shouldTruncate
    ? times
        .map((t, i) => ({ time: t, originalIndex: i }))
        .slice(Math.max(0, currentLoopIndex - 1), currentLoopIndex + 6)
    : times.map((t, i) => ({ time: t, originalIndex: i }));

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

      <div className="timeline-container">
        <div className="timeline-content">
          {visibleItems.map(({ time, originalIndex }) => {
            const timeDate = timeToDate(time);
            const isPastTime = isToday && timeDate < now;
            const isCurrentLoop = originalIndex === currentLoopIndex;
            const isExpanded = expandedLoops.has(originalIndex);
            const showSecondary = isCurrentLoop || isExpanded;

            // Get first stop info
            const firstStop = route?.STOPS?.[0];
            const firstStopData = firstStop ? route[firstStop] as ShuttleStopData : null;

            // Relative time label (upcoming departures within 30 min)
            const minutesUntil = Math.round((timeDate.getTime() - now.getTime()) / 60_000);
            const showRelative = isToday && !isPastTime && !isCurrentLoop && minutesUntil > 0 && minutesUntil <= 30;

            // ETAs for expanded (non-current) loops use static schedule
            const loopETAs = isCurrentLoop ? activeETAs : computeStaticETAs(originalIndex);
            const loopStaticETAs = computeStaticETAs(originalIndex);
            const loopStaticDates = isCurrentLoop ? staticETADates : computeStaticETADates(originalIndex);

            return (
              <div key={originalIndex} className="timeline-route-group">
                {/* First stop - main timeline item */}
                <div
                  className={`timeline-item first-stop ${isCurrentLoop ? 'current-loop' : ''} ${isPastTime && !isCurrentLoop ? 'past-time' : ''}`}
                  onClick={() => !isPastTime && !isCurrentLoop && toggleExpand(originalIndex)}
                  style={!isPastTime && !isCurrentLoop ? { cursor: 'pointer' } : undefined}
                >
                  <div className="timeline-dot"></div>
                  <div className="timeline-content-item">
                    <div className="timeline-time">
                      {time}
                      {showRelative && <span className="relative-time">in {minutesUntil} min</span>}
                      {isCurrentLoop && activeETAs[firstStop] && (
                        <>
                          <span className="live-eta"> - ETA: {activeETAs[firstStop]}</span>
                          <span className="source-badge source-live">LIVE</span>
                        </>
                      )}
                      {!isCurrentLoop && !isPastTime && (
                        <span className="expand-indicator">{isExpanded ? '\u25B4' : '\u25BE'}</span>
                      )}
                    </div>
                    <div className="timeline-stop">{firstStopData?.NAME || 'Unknown Stop'}</div>
                  </div>
                </div>

                {/* Secondary stops */}
                {showSecondary && route?.STOPS && route.STOPS.length > 1 && (
                  <div className="secondary-timeline">
                    {route.STOPS.slice(1).map((stop, stopIndex) => {
                      const stopData = route[stop] as ShuttleStopData;
                      const hasLiveETA = isCurrentLoop && activeETAs[stop];
                      const details = liveETADetails[stop];
                      const lastArrival = isCurrentLoop ? details?.lastArrival : undefined;
                      const isSelected = stop === selectedStop;

                      return (
                        <div
                          key={stopIndex}
                          className={`secondary-timeline-item ${isPastTime && !isCurrentLoop ? 'past-time' : ''} ${hasLiveETA ? 'has-eta' : ''} ${isSelected ? 'selected-stop' : ''}`}
                          onClick={() => handleStopSelect(stop)}
                          style={{ cursor: 'pointer' }}
                        >
                          <div className="secondary-timeline-dot"></div>
                          <div className="secondary-timeline-content">
                            <div className="secondary-timeline-time">
                              {hasLiveETA ? (
                                <>
                                  <span className="live-eta">{activeETAs[stop]}</span>
                                  <span className="source-badge source-live">LIVE</span>
                                  {(() => {
                                    const dev = computeDeviation(stop, loopStaticDates);
                                    return dev ? <span className={dev.className}>{dev.text}</span> : null;
                                  })()}
                                </>
                              ) : lastArrival ? (
                                <>
                                  <span className="last-arrival">Last: {lastArrival}</span>
                                  <span className="source-badge source-live">LIVE</span>
                                </>
                              ) : (isCurrentLoop || isExpanded) && (loopETAs[stop] || loopStaticETAs[stop]) ? (
                                <>
                                  <span className="scheduled-fallback">Sched: {loopETAs[stop] || loopStaticETAs[stop]}</span>
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
                )}
              </div>
            );
          })}
        </div>

        {/* Show full schedule link */}
        {shouldTruncate && (
          <button className="show-full-schedule" onClick={() => setShowFullSchedule(true)}>
            Show full schedule ({times.length - visibleItems.length} more)
          </button>
        )}
        {showFullSchedule && isToday && times.length > 7 && (
          <button className="show-full-schedule" onClick={() => setShowFullSchedule(false)}>
            Show less
          </button>
        )}
      </div>
    </div>
  );
}
