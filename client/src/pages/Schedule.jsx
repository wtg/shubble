import { useState } from 'react';

export default function Schedule() {

    // schedules store the times for each shuttle
    // weekdaySchedule is Monday-Saturday and sundaySchedule is Sunday
    const weekdaySchedule = [
        [
            '11:00',
            '11:20',
            '11:40',
            '12:00',
            '12:20',
            '12:40',
            '13:00',
            '13:20',
            '13:40',
            '14:30',
            '14:50',
            '15:10',
            '15:30',
            '15:50',
            '16:10',
            '16:30',
            '16:50',
            '17:10',
            '17:30',
            '17:50',
        ],
        [
            '11:10',
            '11:30',
            '11:50',
            '12:10',
            '12:30',
            '12:50',
            '13:10',
            '13:30',
            '13:50',
            '14:10',
            '15:00',
            '15:20',
            '15:40',
            '16:00',
            '16:20',
            '16:40',
            '17:00',
            '17:20',
            '17:40',
            '18:00',
        ]
    ];

    const sundaySchedule = [
        '11:00',
        '11:20',
        '11:40',
        '12:00',
        '12:20',
        '12:40',
        '13:00',
        '13:40',
        '14:00',
        '14:20',
        '14:40',
        '15:00',
        '15:20',
        '15:40',
        '16:00',
    ]

    const [day, setDay] = useState(0);
    const handleDayChange = (e) => {
        setDay(parseInt(e.target.value));
    }

    const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

    const schedule = day === 0 ? sundaySchedule : weekdaySchedule;

    // Flatten the schedule for display
    const flatSchedule = schedule.flat().sort();

    return (
        <>
        <div className="p-4">
            <h2>Schedule</h2>
            <p>
                Expected departure times from the student union.
            </p>
            <p>
                Show the times for
                <select value={day} onChange={handleDayChange}>
                    {
                        daysOfTheWeek.map((day, index) =>
                            <option key={index} value={index}>
                                {day}
                            </option>
                        )
                    }
                </select>
            </p>
            <table>
                <thead>
                <tr>
                    <th className="">Time</th>
                </tr>
                </thead>
                <tbody>
                {
                    flatSchedule.map((time, index) => (
                        <tr key={index} className="">
                        <td className="">{time}</td>
                        </tr>
                    ))
                }
                </tbody>
            </table>
            </div>
        </>
    );
}
