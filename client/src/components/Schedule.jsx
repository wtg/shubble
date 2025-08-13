import { useState, useEffect } from 'react';
import '../styles/Schedule.css';
import scheduleData from '../data/schedule.json';
import routeData from '../data/routes.json';
import { parseSchedule } from '../data/parseSchedule';

export default function Schedule() {
    const now = new Date();
    const routeNames = Object.keys(routeData);
    const [selectedDay, setSelectedDay] = useState(now.getDay());
    const [selectedRoute, setSelectedRoute] = useState(routeNames[0]);
    const [selectedStop, setSelectedStop] = useState("all");
    const [stopNames, setStopNames] = useState(routeData[selectedRoute].STOPS || []);
    const [schedule, setSchedule] = useState([]);

    useEffect(() => {
        setSchedule(parseSchedule(scheduleData, selectedDay));
    }, [selectedDay]);

    useEffect(() => {
        if (!(selectedStop in routeData[selectedRoute])) {
            setSelectedStop("all");
        }
        setStopNames(routeData[selectedRoute].STOPS);
    }, [selectedRoute]);

    const handleDayChange = (e) => {
        setSelectedDay(parseInt(e.target.value));
    }

    const offsetTime = (time, offset) => {
        const date = new Date(time);
        date.setMinutes(date.getMinutes() + offset);
        return date;
    }

    const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

    return (
        <div className="p-4">
            <h2>Schedule</h2>
            <div>
                <label for='weekday-dropdown'>Weekday:</label>
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
                <label for='loop-dropdown'>Loop:</label>
                <select id='loop-dropdown' className="schedule-dropdown-style" value={selectedRoute} onChange={(e) => setSelectedRoute(e.target.value)}>
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
                <label for='stop-dropdown'>Stop:</label>
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
            <div className = "schedule-scroll">
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
                                        <td className={ index === 0 ? "" : "indented-time" }>{offsetTime(time, routeData[selectedRoute][stop].OFFSET).toLocaleTimeString(undefined, { timeStyle: 'short' })} {routeData[selectedRoute][stop].NAME}</td>
                                    </tr>
                                ))
                            )) :
                            schedule[selectedRoute]?.map((time, index) => (
                                <tr key={index} className="">
                                    <td className="">{offsetTime(time, routeData[selectedRoute][selectedStop]?.OFFSET).toLocaleTimeString(undefined, { timeStyle: 'short' })}</td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>
            </div>
        </div>
    );
}
