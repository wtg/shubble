import { useState, useEffect } from 'react';
import '../styles/Schedule.css';
import scheduleData from '../data/schedule.json';
import routeData from '../data/routes.json';

export default function Schedule() {
    const weeklySchedule = [
        'SUNDAY',
        'MONDAY',
        'TUESDAY',
        'WEDNESDAY',
        'THURSDAY',
        'FRIDAY',
        'SATURDAY',
    ].map((day) => scheduleData[scheduleData[day]] || []);
    const now = new Date();

    const parseTimeString = (timeStr) => {
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
        dateObj.setMilliseconds(0);
        return dateObj;
    }

    const routeNames = Object.keys(routeData);
    const [selectedRoute, setSelectedRoute] = useState(routeNames[0]);
    const [stopNames, setStopNames] = useState(routeData[selectedRoute]["STOPS"] || []);
    const [selectedDay, setSelectedDay] = useState(now.getDay());
    const [schedule, setSchedule] = useState([]);

    useEffect(() => {
        const scheduleByLoop = {};
        Object.values(weeklySchedule[selectedDay]).forEach((busSchedule) => {
            busSchedule.forEach(([time, loop]) => {
                const timeObj = parseTimeString(time);
                if (loop in scheduleByLoop) {
                    scheduleByLoop[loop].push(timeObj);
                } else {
                    scheduleByLoop[loop] = [timeObj];
                }
            });
        });
        Object.values(scheduleByLoop).forEach((times) => {
            times.sort((a, b) => {
                const isA12AM = a.getHours() === 0 && a.getMinutes() === 0;
                const isB12AM = b.getHours() === 0 && b.getMinutes() === 0;

                if (isA12AM && !isB12AM) return 1;   // a goes after b
                if (!isA12AM && isB12AM) return -1;  // a goes before b
                return a - b;                        // otherwise, normal sort
            });
            });
        setSchedule(scheduleByLoop);
    }, [selectedDay]);

    useEffect(() => {
        setStopNames(routeData[selectedRoute]["STOPS"]);
    }, [selectedRoute]);

    const handleDayChange = (e) => {
        setSelectedDay(parseInt(e.target.value));
    }

    const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

    // Converting Date object into String comparable with format 'HH:MM'
    const currentHours = now.getHours();
    const currentMinutes = now.getMinutes();
    const formattedHours = currentHours < 10 ? '0' + currentHours : String(currentHours);
    const formattedMinutes = currentMinutes < 10 ? '0' + currentMinutes : String(currentMinutes);
    const currentTimeString = formattedHours + ":" + formattedMinutes;

    return (
        <div className="p-4">
            <h2>Schedule</h2>
            Weekday:
            <select value={selectedDay} onChange={handleDayChange}>
                {
                    daysOfTheWeek.map((day, index) =>
                        <option key={index} value={index}>
                            {day}
                        </option>
                    )
                }
            </select>
            Loop:
            <select value={selectedRoute} onChange={(e) => setSelectedRoute(e.target.value)}>
                {
                    routeNames.map((route, index) =>
                        <option key={index} value={route}>
                            {route}
                        </option>
                    )
                }
            </select>
            Stop:
            <select>
                {
                    stopNames.map((stop, index) =>
                        <option key={index} value={stop}>
                            {stop}
                        </option>
                    )
                }
            </select>
            <div className = "schedule-scroll">
                <table>
                    <thead>
                        <tr>
                            <th className="">Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {
                            schedule[selectedRoute]?.map((time, index) => (
                            <tr key={index} className="">
                                <td className="">{time.toLocaleTimeString(undefined, { timeStyle: 'short' })}</td>
                            </tr>
                            ))
                        }
                    </tbody>
                </table>
            </div>
        </div>
    );
}
