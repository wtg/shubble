import { useMemo } from 'react';
import { Link } from 'react-router';
import type { Trip } from '../../hooks/useTrips';
import { useCurrentTime } from '../../hooks/useCurrentTime';
import { useDataFreshness } from '../../hooks/useDataFreshness';
import { FreshnessIndicator } from '../../components/FreshnessIndicator';
import { devNowMs } from '../../utils/devTime';
import routeData from '../../shared/routes.json';
import type { ShuttleRouteData, ShuttleStopData } from '../../types/route';

const TIME_FORMAT: Intl.DateTimeFormatOptions = { hour: 'numeric', minute: '2-digit' };

interface HomeCountdownProps {
  trips: Trip[];
  selectedRoute: string | null;
  lastUpdateAt: Date | null;
}

/**
 * Compact one-liner shown below the map on mobile: answers "when is
 * the next shuttle at my stop?" without requiring a page switch.
 * Taps through to /schedule for full details.
 *
 * Hidden on desktop (the full schedule panel is already visible there).
 * Renders nothing if no stop is selected or no upcoming ETA exists.
 */
export function HomeCountdown({ trips, selectedRoute, lastUpdateAt }: HomeCountdownProps) {
  useCurrentTime('minute');
  const freshness = useDataFreshness(lastUpdateAt);

  const selectedStop = typeof window !== 'undefined'
    ? localStorage.getItem('shubble-stop') : null;

  const result = useMemo(() => {
    if (!selectedRoute || !selectedStop || trips.length === 0) return null;

    const allRoutes = routeData as unknown as ShuttleRouteData;
    const route = allRoutes[selectedRoute as keyof ShuttleRouteData];
    if (!route) return null;
    const stops = (route as Record<string, unknown>)['STOPS'] as string[] | undefined;
    if (!stops) return null;

    const stopData = (route as Record<string, unknown>)[selectedStop] as ShuttleStopData | undefined;
    if (!stopData?.NAME) return null;

    const nowMs = devNowMs();
    const isOrigin = stops[0] === selectedStop;

    let bestEtaMs: number | null = null;
    let bestEtaISO: string | null = null;

    for (const trip of trips) {
      if (trip.status === 'completed') continue;
      const info = trip.stop_etas[selectedStop];
      if (!info || info.passed || !info.eta) continue;
      const etaMs = new Date(info.eta).getTime();
      if (etaMs > nowMs && (bestEtaMs === null || etaMs < bestEtaMs)) {
        bestEtaMs = etaMs;
        bestEtaISO = info.eta;
      }
    }

    if (bestEtaMs === null || bestEtaISO === null) return null;

    const minutes = Math.ceil((bestEtaMs - nowMs) / 60_000);
    const timeStr = new Date(bestEtaISO).toLocaleTimeString(undefined, TIME_FORMAT);
    const stopName = stopData.NAME;
    const prefix = isOrigin ? 'Next departure from' : 'Next at';

    return { minutes, timeStr, stopName, prefix };
  }, [trips, selectedRoute, selectedStop]);

  if (!result) return null;

  return (
    <Link to="/schedule" className="home-countdown">
      {result.prefix} <strong>{result.stopName}</strong> in{' '}
      <strong>
        {result.minutes >= 60
          ? `${Math.floor(result.minutes / 60)}h ${result.minutes % 60}m`
          : `${result.minutes} min`}
      </strong>
      <span className="home-countdown-time"> ({result.timeStr})</span>
      <FreshnessIndicator state={freshness} />
    </Link>
  );
}
