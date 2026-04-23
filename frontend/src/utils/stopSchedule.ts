import type { ShuttleRouteData, ShuttleStopData, StopSchedule } from '../types/route';

const DAY_NAMES: StopSchedule['days'][number][] = [
  'SUNDAY',
  'MONDAY',
  'TUESDAY',
  'WEDNESDAY',
  'THURSDAY',
  'FRIDAY',
  'SATURDAY',
];

/**
 * Parse a "HH:MM" string into total minutes from midnight.
 */
function parseMinutes(hhmm: string): number {
  const [h, m] = hhmm.split(':').map(Number);
  return h * 60 + m;
}

/**
 * Check whether a specific stop is active at the given date/time.
 *
 * If the stop has no SCHEDULE constraint, it is considered always active
 * (its availability is governed solely by whether the route itself is running).
 */
export function isStopActive(stopData: ShuttleStopData, now: Date): boolean {
  const { SCHEDULE } = stopData;
  if (!SCHEDULE) return true;

  const todayName = DAY_NAMES[now.getDay()];
  if (!SCHEDULE.days.includes(todayName)) return false;

  const nowMinutes = now.getHours() * 60 + now.getMinutes();
  const startMinutes = parseMinutes(SCHEDULE.hours.start);
  const endMinutes = parseMinutes(SCHEDULE.hours.end);

  return nowMinutes >= startMinutes && nowMinutes <= endMinutes;
}

/**
 * Return a set of stop keys (e.g. "WEST:CHASAN") that are currently inactive
 * based on their SCHEDULE constraints and the given time.
 *
 * Stops without a SCHEDULE are always considered active.
 */
export function getInactiveStops(routeData: ShuttleRouteData, now: Date): Set<string> {
  const inactive = new Set<string>();

  for (const [routeKey, routeInfo] of Object.entries(routeData)) {
    for (const stopKey of routeInfo.STOPS) {
      const stopData = routeInfo[stopKey] as ShuttleStopData;
      if (!isStopActive(stopData, now)) {
        inactive.add(`${routeKey}:${stopKey}`);
      }
    }
  }

  return inactive;
}
