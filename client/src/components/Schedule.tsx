import { useState, useEffect } from 'react';
import '../styles/Schedule.css';
import routeData from '../data/routes.json';
import { aggregatedSchedule } from '../data/parseSchedule';
import RouteToggle from './RouteToggle';

const aggregatedSchedule: AggregatedScheduleType = rawAggregatedSchedule as unknown as AggregatedScheduleType;


const routeData = rawRouteData as unknown as ShuttleRouteData;
type ScheduleProps = {
  selectedRoute: string | null;
  setSelectedRoute: (route: string | null) => void;
};

export default function Schedule({ selectedRoute, setSelectedRoute }: ScheduleProps) {
  // Validate props once at the top
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }

  const now = new Date();
  const [selectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [stopNames, setStopNames] = useState<string[]>([]);
  const [schedule, setSchedule] = useState<AggregatedDaySchedule>(aggregatedSchedule[selectedDay]);

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

  // Update stopNames when selectedRoute changes
  useEffect(() => {
    if (!safeSelectedRoute || !(safeSelectedRoute in routeData)) return;
    setStopNames(routeData[safeSelectedRoute as keyof typeof routeData].STOPS);
  }, [selectedRoute]);

  // Function to offset schedule time by given minutes
  const offsetTime = (time: string, offset: number) => {
    const date = timeToDate(time);
    date.setMinutes(date.getMinutes() + offset);
    return date;
  }

  // scroll to the current time on route change
  useEffect(() => {
    const scheduleDiv = document.querySelector('.schedule-scroll');
    if (!scheduleDiv) return;

    const currentTimeRow = Array.from(scheduleDiv.querySelectorAll('td.outdented')).find(td => {
      const timeStr = td.textContent?.trim();

      // Expect "H:MM AM/PM" → split at the first space
      const timeDate = timeToDate(timeStr || "");

      return timeDate >= now;
    });

    if (currentTimeRow) {
      currentTimeRow.scrollIntoView({
        block: 'nearest',
        inline: 'nearest',
        behavior: 'smooth'
      });
    }
  }, [selectedRoute, selectedDay, schedule]);


  return (
    <div className="p-4">
      <h2>Today's schedule</h2>
      <RouteToggle selectedRoute={selectedRoute} setSelectedRoute={setSelectedRoute} />
      
      <div>
        <label for='stop-dropdown'>Filter stops: </label>
        <select id='stop-dropdown' className="schedule-dropdown-style" value={selectedStop} onChange={(e) => setSelectedStop(e.target.value)}>
          <option value="all">All Stops</option>
          {
            stopNames.map((stop, index) =>
              <option key={index} value={stop}>
                {routeData[safeSelectedRoute][stop]?.NAME}
              </option>
            )
          }
        </select>
      </div>
      <div className="schedule-scroll">
        <table>
          <thead>
            <tr>
              <th className="schedule-header"><span className="bold">Time</span> (estimated)</th>
            </tr>
          </thead>
          <tbody>
            {(() => {
              const routeKey = safeSelectedRoute as keyof typeof routeData;
              const route = routeData[routeKey];
              const times = schedule[routeKey];

              return times.map((time, index) =>
                route.STOPS.map((stop, sidx) => {
                  const stopData = route[stop] as ShuttleStopData;
                  const displayTime = offsetTime(time, stopData.OFFSET).toLocaleTimeString(
                    undefined,
                    { timeStyle: "short" }
                  );
                  return (
                    <tr key={`${index}-${sidx}`}>
                      <td className={sidx === 0 ? "outdented" : "indented-time"}>
                        {sidx === 0 ? displayTime : ""} {stopData.NAME}
                      </td>
                    </tr>
                  );
                })
              );
            })()}
          </tbody>
        </table>
      </div>
    </div>
  );
}
