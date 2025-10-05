type DayOfTheWeek = "MONDAY" | "TUESDAY" | "WEDNESDAY" | "THURSDAY" | "FRIDAY" | "SATURDAY" | "SUNDAY";

export type Route = 'WEST' | 'NORTH';

type FleetSchedule = {
  [loop: string]: [string, Route][];
}

// schedule.json
export type ShuttleScheduleData = {
  [day in DayOfTheWeek]: keyof ShuttleScheduleData;
} & {
  // this is keyof ShuttleScheduleData. If we have more specific schedule for days, 
  // we can add them here
  "weekday": FleetSchedule;
  "saturday": FleetSchedule;
  "sunday": FleetSchedule;
};

export type AggregatedDaySchedule = {
  [route in Route]: Date[];
}

// data type after getting exported from parseSchedule.ts
export type AggregatedScheduleType = [
  AggregatedDaySchedule, // SUNDAY
  AggregatedDaySchedule, // MONDAY
  AggregatedDaySchedule, // TUESDAY
  AggregatedDaySchedule, // WEDNESDAY
  AggregatedDaySchedule, // THURSDAY
  AggregatedDaySchedule, // FRIDAY
  AggregatedDaySchedule, // SATURDAY
]
