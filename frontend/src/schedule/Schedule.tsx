import { useState, useEffect, useMemo } from 'react';
import './styles/Schedule.css';
import rawRouteData from '../shared/routes.json';
import rawAggregatedSchedule from '../shared/aggregated_schedule.json';
import type { AggregatedDaySchedule, AggregatedScheduleType} from '../types/schedule';
import type { ShuttleRouteData, ShuttleStopData } from '../types/route';
import type { VehicleETAs } from '../types/vehicleLocation';
import {buildAllStops, findClosestStop, type Stop, type ClosestStop, } from '../types/ClosestStop';



const aggregatedSchedule: AggregatedScheduleType = rawAggregatedSchedule as unknown as AggregatedScheduleType;


const routeData = rawRouteData as unknown as ShuttleRouteData;
type ScheduleProps = {
  selectedRoute: string | null;
  setSelectedRoute: (route: string | null) => void;
  stopTimes?: VehicleETAs;
  vehiclesAtStops?: Record<string, string[]>;
};

export default function Schedule({ selectedRoute, setSelectedRoute, stopTimes = {}, vehiclesAtStops = {} }: ScheduleProps) {
  // Validate props once at the top
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }

  const now = new Date();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [schedule, setSchedule] = useState<AggregatedDaySchedule>(aggregatedSchedule[selectedDay]);

  const [allStops, setAllStops] = useState<Stop[]>([]);
  const [closestStop, setClosestStop] = useState<ClosestStop | null>(null);

  // Compute minimum stop time for each stop across all shuttles
  const minStopTimes = useMemo(() => {
    const stopTimeMap: Record<string, Date> = {};

    // Iterate through all vehicles and their stop times
    Object.values(stopTimes).forEach((vehicleStopTimes) => {
      Object.entries(vehicleStopTimes).forEach(([stopKey, timeString]) => {
        const timeDate = new Date(timeString);

        // If this stop doesn't have a time yet, or this time is earlier, update it
        if (!stopTimeMap[stopKey] || timeDate < stopTimeMap[stopKey]) {
          stopTimeMap[stopKey] = timeDate;
        }
      });
    });

    return stopTimeMap;
  }, [stopTimes]);

  // Whenever selectedDay changes, recompute today's stops
  useEffect(() => {
    const stopsToday = buildAllStops(routeData, aggregatedSchedule, selectedDay);
    setAllStops(stopsToday);
  }, [selectedDay]);

  // Define safe values to avoid repeated null checks
  const safeSelectedRoute = selectedRoute || routeNames[0];

  // Update schedule and routeNames when selectedDay changes
  useEffect(() => {
    setSchedule(aggregatedSchedule[selectedDay]);
    setRouteNames(Object.keys(aggregatedSchedule[selectedDay]));
    // If parent hasn't provided a selectedRoute yet, pick the first available one
    const firstRoute = Object.keys(aggregatedSchedule[selectedDay])[0];
    if (!selectedRoute || !(selectedRoute in aggregatedSchedule[selectedDay])) {
      setSelectedRoute(firstRoute);
    }
  }, [selectedDay, selectedRoute, setSelectedRoute]);

  // Handle day change from dropdown
  const handleDayChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDay(parseInt(e.target.value));
  }

  const timeToDate = (timeStr: string): Date => {
    const [time, modifier] = timeStr.trim().split(" ");

    // eslint-disable-next-line prefer-const
    let [hours, minutes] = time.split(":").map(Number);
    if (modifier.toUpperCase() === "PM" && hours !== 12) {
      hours += 12;
    }
    else if (modifier.toUpperCase() === "AM" && hours === 12) {
      hours = 0;
    }
    const dateObj = new Date();
    dateObj.setHours(hours);
    dateObj.setMinutes(minutes);
    dateObj.setSeconds(0);
    return dateObj;
  }

  // Function to offset schedule time by given minutes
  const offsetTime = (time: string, offset: number) => {
    const date = timeToDate(time);
    date.setMinutes(date.getMinutes() + offset);
    return date;
  }

// Use user location and get closest stop to them
  useEffect(() => {
    if (!('geolocation' in navigator)) return;

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const userPoint = {
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
        };

        const closest = findClosestStop(userPoint, allStops);
        setClosestStop(closest);
      },
      (err) => {
        console.error('Error getting user location', err);
      }
    );
  }, [allStops]);


  // scroll to the current time on route change
  useEffect(() => {
    const scheduleDiv = document.querySelector('.schedule-scroll');
    if (!scheduleDiv) return;

    if (selectedDay !== now.getDay()) return; // only scroll if viewing today's schedule

    // Find all times before now and get the last one (excluding 12AM)
    const timesBeforeNow = Array.from(scheduleDiv.querySelectorAll('td.outdented')).filter(td => {
      const timeStr = td.textContent?.trim();

      // Exclude 12AM times (end of schedule)
      if (timeStr?.includes('12:') && timeStr?.includes('AM')) {
        return false;
      }

      // Expect "H:MM AM/PM" → split at the first space
      const timeDate = timeToDate(timeStr || "");

      return timeDate < now;
    });

    const currentTimeRow = timesBeforeNow[timesBeforeNow.length - 1];

    if (currentTimeRow) {
      currentTimeRow.scrollIntoView({ behavior: "auto" });
    }
  }, [selectedRoute, selectedDay, schedule]);


  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  return (
    <div className="p-4">
      <h2>Schedule</h2>
      <div>
          {closestStop && (
          <div className="closest-stop-hint">
            Closest Stop: <strong>{closestStop.name}</strong>
          </div>
         )}
        <label htmlFor='weekday-dropdown'>Weekday:</label>
        <select id='weekday-dropdown' className="schedule-dropdown-style" value={selectedDay} onChange={handleDayChange}>
          {
            daysOfTheWeek.map((day, index) =>
              <option key={index} value={index}>
                {day}
              </option>
            )
          }
        </select>
      </div>
      <div>
        <label htmlFor='loop-dropdown'>Loop:</label>

        <select id='loop-dropdown' className="schedule-dropdown-style" value={safeSelectedRoute} onChange={(e) => setSelectedRoute(e.target.value)}>
          {
            routeNames.map((route, index) =>
              <option key={index} value={route}>
                {route}
              </option>
            )
          }
        </select>
      </div>
      <div className="schedule-scroll">
        <table>
          <thead>
            <tr>
              <th className="schedule-header"><span className="bold">Time</span> (estimated) | <span className="bold live-eta-header">Live ETA</span></th>
            </tr>
          </thead>
          <tbody>
            {(() => {
              const routeKey = safeSelectedRoute as keyof typeof routeData;
              const route = routeData[routeKey];
              const times = schedule[routeKey];

              // Find the current time loop index if viewing today's schedule
              // Use the last time before now (excluding 12AM)
              let currentLoopIdx = -1;
              if (selectedDay === now.getDay()) {
                for (let i = 0; i < times.length; i++) {
                  const time = times[i];
                  const firstStop = route.STOPS[0];
                  const firstStopData = route[firstStop] as ShuttleStopData;
                  const loopTime = offsetTime(time, firstStopData.OFFSET);

                  // Exclude 12AM times (end of schedule)
                  const timeStr = loopTime.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
                  const is12AM = timeStr.includes('12:') && timeStr.includes('AM');

                  if (loopTime < now && !is12AM) {
                    currentLoopIdx = i;
                    // Keep going to find the last one before now
                  } else if (loopTime >= now) {
                    break;
                  }
                }
              }

              // If no current loop found, use the last loop
              if (currentLoopIdx === -1) {
                currentLoopIdx = times.length - 1;
              }

              // Build the full list of rows
              const allRows: Array<{
                time: string;
                index: number;
                stop: string;
                sidx: number;
                stopData: ShuttleStopData;
                displayTime: string;
              }> = [];

              times.forEach((time, index) => {
                route.STOPS.forEach((stop, sidx) => {
                  const stopData = route[stop] as ShuttleStopData;
                  const displayTime = offsetTime(time, stopData.OFFSET).toLocaleTimeString(
                    undefined,
                    { hour: 'numeric', minute: '2-digit' }
                  );
                  allRows.push({ time, index, stop, sidx, stopData, displayTime });
                });
              });

              // Find occurrences of each stop in the current loop
              const currentLoopOccurrenceMap = new Map<string, number>();
              const stopsPerLoop = route.STOPS.length;
              const currentLoopStartIdx = currentLoopIdx * stopsPerLoop;
              const currentLoopEndIdx = currentLoopStartIdx + stopsPerLoop;

              for (let rowIdx = currentLoopStartIdx; rowIdx < currentLoopEndIdx && rowIdx < allRows.length; rowIdx++) {
                const row = allRows[rowIdx];
                currentLoopOccurrenceMap.set(row.stop, rowIdx);
              }

              // Render all rows, showing times/NOW only on current loop occurrence
              return allRows.map((row, rowIdx) => {
                const { stop, sidx, stopData, displayTime, index } = row;

                // Check if there are vehicles currently at this stop
                const vehiclesAtThisStop = vehiclesAtStops[stopData.NAME] || [];
                const hasVehicleNow = vehiclesAtThisStop.length > 0;

                // Get stop time for this stop using the stop key
                const stopTime = minStopTimes[stop];

                // Only show time on the current loop occurrence of this stop
                const isCurrentLoopOccurrence = currentLoopOccurrenceMap.get(stop) === rowIdx;
                const shouldShowTime = stopTime && isCurrentLoopOccurrence;

                // Check if stop time is in the past
                const isStopTimePast = stopTime && stopTime < now;

                // For historical stops, show "---" instead of the time
                const timeDisplay = shouldShowTime
                  ? (isStopTimePast
                      ? "---"
                      : stopTime.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' }))
                  : "";

                const etaClassName = isStopTimePast ? "inline-eta-past" : "inline-eta";

                // Only show NOW on current loop occurrence as well
                const shouldShowNow = hasVehicleNow && isCurrentLoopOccurrence;

                return (
                  <tr key={`${index}-${sidx}`}>
                    <td className={sidx === 0 ? "outdented" : "indented-time"}>
                      {sidx === 0 ? displayTime : ""} {shouldShowNow && <span className="inline-now">NOW </span>}{timeDisplay && <span className={etaClassName}>{timeDisplay} </span>}{stopData.NAME}
                    </td>
                  </tr>
                );
              });
            })()}

          </tbody>
        </table>
      </div>

    </div>
  );
}
