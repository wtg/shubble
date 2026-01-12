import { timeToDate } from './timeUtils.js';
import fs from 'fs';
import path from 'path';
import scheduleData from './schedule.json' with { type: 'json' };

function aggregateSchedule(scheduleData) {
    const aggregatedSchedule = [];
    const weeklySchedule = [
        'SUNDAY',
        'MONDAY',
        'TUESDAY',
        'WEDNESDAY',
        'THURSDAY',
        'FRIDAY',
        'SATURDAY',
    ].map((day) => scheduleData[scheduleData[day]] || []);

    for (let i = 0; i < weeklySchedule.length; i++) {
        aggregatedSchedule.push({});
        Object.values(weeklySchedule[i]).forEach((busSchedule) => {
            busSchedule.forEach(([time, loop]) => {
                if (loop in aggregatedSchedule[i]) {
                    aggregatedSchedule[i][loop].push(time);
                } else {
                    aggregatedSchedule[i][loop] = [time];
                }
            });
        });
        Object.values(aggregatedSchedule[i]).forEach((times) => {
            times.sort((a, b) => {
                a = timeToDate(a);
                b = timeToDate(b);
                const isA12AM = a.getHours() === 0 && a.getMinutes() === 0;
                const isB12AM = b.getHours() === 0 && b.getMinutes() === 0;

                if (isA12AM && !isB12AM) return 1;
                if (!isA12AM && isB12AM) return -1;
                return a - b;
            });
        });
    }
    return aggregatedSchedule;
}

const aggregatedSchedule = aggregateSchedule(scheduleData);

// Write to the shared directory (where this script is located)
const outputPath = path.join(path.dirname(new URL(import.meta.url).pathname), 'aggregated_schedule.json');
fs.writeFileSync(outputPath, JSON.stringify(aggregatedSchedule, null, 2));
console.log(`aggregatedSchedule.json generated at ${outputPath}`);