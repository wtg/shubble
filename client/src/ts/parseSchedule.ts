import rawScheduleData from '../data/schedule.json';
import type { AggregatedScheduleType, Route, ShuttleScheduleData } from './types/schedule';

function aggregateSchedule(scheduleData: ShuttleScheduleData) {
    const aggregatedSchedule: AggregatedScheduleType = [
        { 'WEST': [], 'NORTH': [] }, // SUNDAY
        { 'WEST': [], 'NORTH': [] }, // MONDAY
        { 'WEST': [], 'NORTH': [] }, // TUESDAY
        { 'WEST': [], 'NORTH': [] }, // WEDNESDAY
        { 'WEST': [], 'NORTH': [] }, // THURSDAY
        { 'WEST': [], 'NORTH': [] }, // FRIDAY
        { 'WEST': [], 'NORTH': [] }, // SATURDAY
    ];
    const weeklySchedule = [
        'SUNDAY',
        'MONDAY',
        'TUESDAY',
        'WEDNESDAY',
        'THURSDAY',
        'FRIDAY',
        'SATURDAY',
    ].map((day: string) => scheduleData[scheduleData[day as keyof ShuttleScheduleData] as keyof ShuttleScheduleData] || []);

    for (let i = 0; i < weeklySchedule.length; i++) {
        Object.values(weeklySchedule[i]).forEach((busSchedule) => {
            busSchedule.forEach(([time, loop]: [string, Route]) => {
                const timeObj = parseTimeString(time);
                aggregatedSchedule[i][loop].push(timeObj);
            });
        });
        Object.values(aggregatedSchedule[i]).forEach((times) => {
            times.sort((a, b) => {
                const isA12AM = a.getHours() === 0 && a.getMinutes() === 0;
                const isB12AM = b.getHours() === 0 && b.getMinutes() === 0;

                if (isA12AM && !isB12AM) return 1;   // a goes after b
                if (!isA12AM && isB12AM) return -1;  // a goes before b
                return a.getTime() - b.getTime();                        // otherwise, normal sort
            });
        });
    }
    return aggregatedSchedule;
}

function parseTimeString(timeStr: string): Date {
    const [time, modifier] = timeStr.trim().split(" ");

    // hour changes but minutes stay the same
    // eslint-disable-next-line prefer-const
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

// We should really trust our JSON data at this point
export const aggregatedSchedule: AggregatedScheduleType = aggregateSchedule(rawScheduleData as unknown as ShuttleScheduleData);
