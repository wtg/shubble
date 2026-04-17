import type { FreshnessState } from '../hooks/useDataFreshness';
import './styles/FreshnessIndicator.css';

interface FreshnessIndicatorProps {
  state: FreshnessState;
  /** Optional text label shown next to the dot. Omit for dot-only mode. */
  label?: string;
}

const STATE_LABEL: Record<Exclude<FreshnessState, 'none'>, string> = {
  fresh: 'Live data current',
  aging: 'Live data recent',
  stale: 'Live data may be delayed',
};

/**
 * Small pulsing dot that visualizes whether live data is fresh, aging,
 * or stale (TRUST-04). Renders nothing when state === 'none' (e.g.
 * before the first fetch completes).
 *
 * A11y: role="status" with `aria-live="polite"` so screen readers
 * announce transitions (e.g. "Live data may be delayed") without
 * interrupting. Decorative ring animation is suppressed when
 * `prefers-reduced-motion` is set.
 */
export function FreshnessIndicator({ state, label }: FreshnessIndicatorProps) {
  if (state === 'none') return null;

  const text = STATE_LABEL[state];
  return (
    <span
      className={`freshness-indicator freshness-${state}`}
      role="status"
      aria-live="polite"
      aria-label={text}
      title={text}
    >
      <span className="freshness-dot" aria-hidden="true" />
      {label !== undefined && <span className="freshness-label">{label}</span>}
    </span>
  );
}
