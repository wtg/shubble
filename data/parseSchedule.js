export function parseSchedule(scheduleData, selectedDay) {
    const scheduleByLoop = {};
    const weeklySchedule = [
        'SUNDAY',
        'MONDAY',
        'TUESDAY',
        'WEDNESDAY',
        'THURSDAY',
        'FRIDAY',
        'SATURDAY',
    ].map((day) => scheduleData[scheduleData[day]] || []);
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
    return scheduleByLoop;
}

export function parseTimeString(timeStr) {
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
