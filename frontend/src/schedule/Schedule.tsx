import { useState, useEffect } from 'react';
import './styles/Schedule.css';
import rawRouteData from '../shared/routes.json';
import rawAggregatedSchedule from '../shared/aggregated_schedule.json';
import config from '../utils/config';
import type { AggregatedDaySchedule, AggregatedScheduleType, Route } from '../types/schedule';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';
import type { VehicleETAMap } from '../types/vehicleLocation';

const aggregatedSchedule: AggregatedScheduleType = rawAggregatedSchedule as unknown as AggregatedScheduleType;
const routeData = rawRouteData as unknown as ShuttleRouteData;

type ScheduleProps = {
  selectedRoute: string | null;
  setSelectedRoute: (route: string | null) => void;
};

// ETAs mapped by stop key -> formatted time string
type StopETAs = Record<string, string>;

export default function Schedule({ selectedRoute, setSelectedRoute }: ScheduleProps) {
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }

  const now = new Date();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [schedule, setSchedule] = useState<AggregatedDaySchedule>(aggregatedSchedule[selectedDay]);
  const [liveETAs, setLiveETAs] = useState<StopETAs>({});

  const safeSelectedRoute = selectedRoute || routeNames[0];

  // Fetch live ETAs from backend etas endpoint
  useEffect(() => {
    const fetchETAs = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/etas`, { cache: 'no-store' });
        if (!response.ok) return;

        const data: VehicleETAMap = await response.json();

        // Aggregate ETAs from all vehicles - use earliest ETA for each stop
        const stopETAs: StopETAs = {};

        Object.values(data).forEach((etaData) => {
          if (etaData.stop_times) {
            Object.entries(etaData.stop_times).forEach(([stopKey, isoTime]) => {
              const etaDate = new Date(isoTime);
              const formattedTime = etaDate.toLocaleTimeString(undefined, {
                hour: 'numeric',
                minute: '2-digit'
              });

              // Use earliest ETA if multiple vehicles
              if (!stopETAs[stopKey] || etaDate < new Date(stopETAs[stopKey])) {
                stopETAs[stopKey] = formattedTime;
              }
            });
          }
        });

        setLiveETAs(stopETAs);
      } catch (error) {
        console.error('Failed to fetch ETAs:', error);
      }
    };

    fetchETAs();
    const interval = setInterval(fetchETAs, 30000); // Poll every 30 seconds

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
  }, [selectedDay, selectedRoute, setSelectedRoute]);

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
      // Calculate scroll position using getBoundingClientRect for accurate positioning
      const containerRect = timelineContainer.getBoundingClientRect();
      const targetRect = targetItem.getBoundingClientRect();
      // Convert visual offset to scroll position within the container
      const targetOffsetInContainer = targetRect.top - containerRect.top + timelineContainer.scrollTop;
      // Position the current loop at the top of the container
      timelineContainer.scrollTop = Math.max(0, targetOffsetInContainer);
    }
  }, [selectedRoute, selectedDay, schedule]);

  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  const times = schedule[safeSelectedRoute as Route] || [];
  const route = routeData[safeSelectedRoute as keyof typeof routeData];
  const currentLoopIndex = findCurrentLoopIndex(times);

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
            <label>Route:</label>
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

      <div className="timeline-container">
        <div className="timeline-content">
          {times.map((time: string, index: number) => {
            const timeDate = timeToDate(time);
            const isPastTime = selectedDay === now.getDay() && timeDate < now;
            const isCurrentLoop = index === currentLoopIndex;

            // Get first stop info
            const firstStop = route?.STOPS?.[0];
            const firstStopData = firstStop ? route[firstStop] as ShuttleStopData : null;

            return (
              <div key={index} className="timeline-route-group">
                {/* First stop - main timeline item */}
                <div className={`timeline-item first-stop ${isCurrentLoop ? 'current-loop' : ''} ${isPastTime && !isCurrentLoop ? 'past-time' : ''}`}>
                  <div className="timeline-dot"></div>
                  <div className="timeline-content-item">
                    <div className="timeline-time">
                      {time}
                      {isCurrentLoop && liveETAs[firstStop] && (
                        <span className="live-eta"> - ETA: {liveETAs[firstStop]}</span>
                      )}
                    </div>
                    <div className="timeline-stop">{firstStopData?.NAME || 'Unknown Stop'}</div>
                  </div>
                </div>

                {/* Secondary stops - only show for current loop with ETAs */}
                {route?.STOPS && route.STOPS.length > 1 && (
                  <div className="secondary-timeline">
                    {route.STOPS.slice(1).map((stop, stopIndex) => {
                      const stopData = route[stop] as ShuttleStopData;
                      const hasETA = isCurrentLoop && liveETAs[stop];

                      return (
                        <div
                          key={stopIndex}
                          className={`secondary-timeline-item ${isPastTime && !isCurrentLoop ? 'past-time' : ''} ${hasETA ? 'has-eta' : ''}`}
                        >
                          <div className="secondary-timeline-dot"></div>
                          <div className="secondary-timeline-content">
                            <div className="secondary-timeline-time">
                              {hasETA ? (
                                <span className="live-eta">{liveETAs[stop]}</span>
                              ) : (
                                <span className="no-eta">--:--</span>
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
      </div>
    </div>
  );
}
