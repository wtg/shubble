import { useBreakPredictions, type BreakPrediction } from '../../hooks/useBreakPredictions';
import '../styles/UpcomingBreaks.css';

/** Format ISO → "11:28 AM" in campus time (server already sends campus-local). */
function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

function formatLead(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)} min`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return m ? `${h}h ${m}m` : `${h}h`;
}

function sourceLabel(p: BreakPrediction): string {
  if (p.source.endsWith('-driver') || p.source === 'bimodal-mode-driver') {
    return 'driver-matched';
  }
  switch (p.source) {
    case 'scheduled-active':
      return 'printed';
    case 'bimodal-mode':
      return 'learned cluster';
    case 'discovered':
      return 'discovered';
    case 'scheduled-rare':
      return 'rare slot';
    default:
      return p.source;
  }
}

function confidenceBar(c: number): string {
  const filled = Math.round(c * 10);
  return '█'.repeat(filled) + '░'.repeat(10 - filled);
}

export default function UpcomingBreaks() {
  const { data, loading, error } = useBreakPredictions({
    lookaheadMin: 240,
    pollIntervalMs: 60_000,
  });

  if (loading && !data) {
    return (
      <div className="upcoming-breaks">
        <h3>Upcoming breaks</h3>
        <p className="upcoming-breaks-loading">Loading predictions…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="upcoming-breaks">
        <h3>Upcoming breaks</h3>
        <p className="upcoming-breaks-error">Error: {error}</p>
      </div>
    );
  }

  const preds = data?.predictions ?? [];
  const reactive = data?.reactive_observed ?? [];

  return (
    <div className="upcoming-breaks">
      <h3>
        Upcoming breaks
        <span className="upcoming-breaks-count">
          {preds.length} next {data?.lookahead_min ?? 240} min
          {data?.active_drivers_matched ? (
            <> · {data.active_drivers_matched} drivers matched</>
          ) : null}
        </span>
      </h3>

      {reactive.length > 0 && (
        <div className="upcoming-breaks-happening-now">
          <h4>Happening now</h4>
          <ul className="upcoming-breaks-list">
            {reactive.map((r) => (
              <li key={`reactive-${r.vehicle_id}-${r.observed_at}`} className="upcoming-break-card reactive">
                <div className="upcoming-break-row-main">
                  <span className="upcoming-break-run">Shuttle {r.vehicle_id}</span>
                  <span className="upcoming-break-time">on break</span>
                </div>
                <div className="upcoming-break-row-meta">
                  <span className="upcoming-break-source" title="Detected in real time (no prior prediction)">
                    reactive-observed
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {preds.length === 0 ? (
        <p className="upcoming-breaks-empty">
          No predicted breaks in the next {data?.lookahead_min ?? 240} minutes.
        </p>
      ) : (
        <ul className="upcoming-breaks-list">
          {preds.map((p, idx) => (
            <li key={`${p.run}-${p.predicted_start}-${idx}`} className="upcoming-break-card">
              <div className="upcoming-break-row-main">
                <span className="upcoming-break-run">{p.run}</span>
                <span className="upcoming-break-time">{formatTime(p.predicted_start)}</span>
              </div>
              <div className="upcoming-break-row-meta">
                <span className="upcoming-break-lead">in {formatLead(p.lead_min)}</span>
                <span className="upcoming-break-sigma">±{Math.round(p.sigma_min)}m</span>
                <span
                  className="upcoming-break-source"
                  title={`Source: ${p.source}`}
                >
                  {sourceLabel(p)}
                </span>
                {p.db_verified === false && (
                  <span
                    className="upcoming-break-unverified"
                    title="Slot not found in live DB schedule — may be stale."
                  >
                    ⚠ unverified
                  </span>
                )}
              </div>
              <div className="upcoming-break-conf" title={`confidence ${(p.confidence * 100).toFixed(0)}%`}>
                <span className="upcoming-break-conf-bar">{confidenceBar(p.confidence)}</span>
                <span className="upcoming-break-conf-pct">{Math.round(p.confidence * 100)}%</span>
              </div>
            </li>
          ))}
        </ul>
      )}

      {data?.generated_at && (
        <p className="upcoming-breaks-updated">
          Updated {formatTime(data.generated_at)}
        </p>
      )}
    </div>
  );
}
