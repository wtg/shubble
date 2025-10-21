import { useState, useEffect } from 'react';
import '../styles/Schedule.css';
import rawRouteData from '../data/routes.json';
import rawAggregatedSchedule from '../data/aggregated_schedule.json';
import type { AggregatedDaySchedule, AggregatedScheduleType } from '../ts/types/schedule';
import type { ShuttleRouteData, ShuttleStopData } from '../ts/types/route';

const aggregatedSchedule: AggregatedScheduleType = rawAggregatedSchedule as unknown as AggregatedScheduleType;


const routeData = rawRouteData as unknown as ShuttleRouteData;
type ScheduleProps = {
  selectedRoute: string | null;
  setSelectedRoute: (route: string | null) => void;
  selectedStop: string;
  setSelectedStop: (stop: string) => void;
};

export default function Schedule({ selectedRoute, setSelectedRoute, selectedStop, setSelectedStop }: ScheduleProps) {
  // Validate props once at the top
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }
  if (typeof setSelectedStop !== 'function') {
    throw new Error('setSelectedStop must be a function');
  }

  const now = new Date();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [stopNames, setStopNames] = useState<string[]>([]);
  const [schedule, setSchedule] = useState<AggregatedDaySchedule>(aggregatedSchedule[selectedDay]);

  // Define safe values to avoid repeated null checks
  const safeSelectedStop = selectedStop || "all";
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

  // Update stopNames and selectedStop when selectedRoute changes
  useEffect(() => {
    if (!safeSelectedRoute || !(safeSelectedRoute in routeData)) return;
    if (selectedStop && !(selectedStop in routeData[safeSelectedRoute as keyof typeof routeData])) {
      setSelectedStop("all");
    }
    setStopNames(routeData[safeSelectedRoute as keyof typeof routeData].STOPS);
  }, [selectedRoute]);

  // Handle day change from dropdown
  const handleDayChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDay(parseInt(e.target.value));
  }

  // Function to offset schedule time by given minutes
  const offsetTime = (time: Date, offset: number) => {
    const date = new Date(time);
    date.setMinutes(date.getMinutes() + offset);
    return date;
  }

  // scroll to the current time on route change
  useEffect(() => {
    const scheduleDiv = document.querySelector('.schedule-scroll');
    if (!scheduleDiv) return;

    if (selectedDay !== now.getDay()) return; // only scroll if viewing today's schedule
    const currentTimeRow = Array.from(scheduleDiv.querySelectorAll('td.outdented')).find(td => {
      const text = td.textContent.trim();

      // Expect "H:MM AM/PM ..." → split at the first space
      const [timePart, meridian] = text.split(" ");
      if (!timePart || !meridian) return false;

      const [rawHours, rawMinutes] = timePart.split(":");
      let hours = parseInt(rawHours, 10);
      const minutes = parseInt(rawMinutes, 10);

      // Convert to 24h
      if (meridian.toUpperCase() === "PM" && hours < 12) {
        hours += 12;
      }
      if (meridian.toUpperCase() === "AM" && hours === 12) {
        hours = 0;
      }

      const timeDate = new Date();
      timeDate.setHours(hours, minutes, 0, 0);

      return timeDate >= now;
    });

    if (currentTimeRow) {
      currentTimeRow.scrollIntoView({ behavior: "auto" });
    }
  }, [selectedRoute, selectedDay, selectedStop, schedule]);


  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  return (
    <div className="p-4">
      <h2>Schedule</h2>
      <div>
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
      <div>
        <label htmlFor='stop-dropdown'>Stop:</label>
        <select id='stop-dropdown' className="schedule-dropdown-style" value={safeSelectedStop} onChange={(e) => setSelectedStop(e.target.value)}>
          <option value="all">All Stops</option>
          {
            stopNames.map((stop, index) =>
              <option key={index} value={stop}>
                {(routeData[safeSelectedRoute as keyof typeof routeData][stop] as ShuttleStopData)?.NAME}
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
            {
              safeSelectedStop === "all" ?
                schedule[safeSelectedRoute as keyof typeof schedule].map((time, index) => (
                  routeData[safeSelectedRoute as keyof typeof routeData].STOPS.map((stop, sidx) => (
                    <tr key={`${index}-${sidx}`} className="">
                      <td className={sidx === 0 ? "outdented" : "indented-time"}>{offsetTime(time, (routeData[safeSelectedRoute as keyof typeof routeData][stop] as ShuttleStopData).OFFSET).toLocaleTimeString(undefined, { timeStyle: 'short' })} {(routeData[safeSelectedRoute as keyof typeof routeData][stop as keyof ShuttleStopData] as ShuttleStopData).NAME}</td>
                    </tr>
                  ))
                )) :
                schedule[safeSelectedRoute as keyof typeof schedule].map((time, index) => (
                  <tr key={index} className="">
                    <td className="outdented">{offsetTime(time, (routeData[safeSelectedRoute as keyof typeof routeData][selectedStop] as ShuttleStopData).OFFSET).toLocaleTimeString(undefined, { timeStyle: 'short' })}</td>
                  </tr>
                ))
            }
          </tbody>
        </table>
      </div>
    </div>
  );
}
