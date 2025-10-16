const fs = require('fs');
const path = require('path');

const schedulePath = path.resolve(__dirname, './schedule.json');
const scheduleData = JSON.parse(fs.readFileSync(schedulePath, 'utf-8'));

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
                const timeObj = parseTimeString(time);
                if (loop in aggregatedSchedule[i]) {
                    aggregatedSchedule[i][loop].push(timeObj);
                } else {
                    aggregatedSchedule[i][loop] = [timeObj];
                }
            });
        });
        Object.values(aggregatedSchedule[i]).forEach((times) => {
            times.sort((a, b) => {
                a = new Date(a);
                b = new Date(b);
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

function parseTimeString(timeStr) {
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
    const timezone = process.env.VITE_TIMEZONE || 'America/New_York';
    return dateObj.toLocaleTimeString('en-US', { timeZone: timezone, hour12: true, hour: 'numeric', minute: '2-digit' });
}

const aggregatedSchedule = aggregateSchedule(scheduleData);

const outputPath = path.resolve(__dirname, './aggregated_schedule.json');
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(aggregatedSchedule, null, 2));
console.log(`aggregatedSchedule.json generated at ${outputPath}`);
