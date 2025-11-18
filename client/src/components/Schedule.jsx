import { useState, useEffect, useLayoutEffect } from 'react';
import '../styles/Schedule.css';
import scheduleData from '../data/schedule.json';
import routeData from '../data/routes.json';
import { aggregatedSchedule } from '../data/parseSchedule';
import RouteToggle from './RouteToggle';

export default function Schedule() {
  const now = new Date();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [selectedRoute, setSelectedRoute] = useState(routeNames[0]);
  const [selectedStop, setSelectedStop] = useState("all");
  const [stopNames, setStopNames] = useState([]);
  const [schedule, setSchedule] = useState([]);

  // Update schedule and routeNames when selectedDay changes
  useEffect(() => {
    setSchedule(aggregatedSchedule[selectedDay]);
    setRouteNames(Object.keys(aggregatedSchedule[selectedDay]));
  }, [selectedDay]);

  // Update stopNames and selectedStop when selectedRoute changes
  useEffect(() => {
    if (!(selectedStop in routeData[selectedRoute])) {
      setSelectedStop("all");
    }
    setStopNames(routeData[selectedRoute].STOPS);
  }, [selectedRoute]);

  // Handle day change from dropdown
  const handleDayChange = (e) => {
    setSelectedDay(parseInt(e.target.value));
  }

  // Function to offset schedule time by given minutes
  const offsetTime = (time, offset) => {
    const date = new Date(time);
    date.setMinutes(date.getMinutes() + offset);
    return date;
  }

  // scroll to the current time on route change
  useEffect(() => {
    const scheduleDiv = document.querySelector('.schedule-scroll');
    if (!scheduleDiv) return;

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
      currentTimeRow.scrollIntoView({
        block: 'nearest',
        inline: 'nearest',
        behavior: 'smooth'
      });
    }
  }, [selectedRoute, selectedDay, selectedStop, schedule]);


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
                {routeData[selectedRoute][stop]?.NAME}
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
              selectedStop === "all" ?
                schedule[selectedRoute]?.map((time, index) => (
                  routeData[selectedRoute].STOPS.map((stop, index) => (
                    <tr key={index} className="">
                      <td className={index === 0 ? "outdented" : "indented-time"}>{offsetTime(time, routeData[selectedRoute][stop].OFFSET).toLocaleTimeString(undefined, { timeStyle: 'short' })} {routeData[selectedRoute][stop].NAME}</td>
                    </tr>
                  ))
                )) :
                schedule[selectedRoute]?.map((time, index) => (
                  <tr key={index} className="">
                    <td className="outdented">{offsetTime(time, routeData[selectedRoute][selectedStop]?.OFFSET).toLocaleTimeString(undefined, { timeStyle: 'short' })}</td>
                  </tr>
                ))
            }
          </tbody>
        </table>
      </div>
    </div>
  );
}
