import { useBreakPredictions, type BreakPrediction } from '../../hooks/useBreakPredictions';
import '../styles/UpcomingBreaks.css';

/** Format ISO → "11:28 AM" in campus time (server already sends campus-local). */
function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

function likelihoodLabel(c: number): string {
  if (c >= 0.6) return 'Likely';
  if (c >= 0.3) return 'Possible';
  return 'Occasional';
}

function likelihoodClass(c: number): string {
  if (c >= 0.6) return 'likelihood-likely';
  if (c >= 0.3) return 'likelihood-possible';
  return 'likelihood-occasional';
}

/** Compact "in Xh Ym" / "in X min". Dropped if lead is tiny (< 1 min). */
function leadLabel(minutes: number): string {
  if (minutes < 1) return 'soon';
  if (minutes < 60) return `in ${Math.round(minutes)} min`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return m ? `in ${h}h ${m}m` : `in ${h}h`;
}

interface UpcomingBreaksProps {
  /** Filter predictions to runs whose name contains this substring
   *  (e.g. "North" or "West"). Empty string or undefined = show all. */
  routeFilter?: string;
}

export default function UpcomingBreaks({ routeFilter }: UpcomingBreaksProps = {}) {
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

  const allPreds = data?.predictions ?? [];
  const allReactive = data?.reactive_observed ?? [];
  // Route filter: substring match on p.run. Reactive entries have no run
  // attribution yet, so they pass through unfiltered.
  const preds = routeFilter
    ? allPreds.filter((p) => p.run.includes(routeFilter))
    : allPreds;
  const reactive = routeFilter ? [] : allReactive;

  return (
    <div className="upcoming-breaks">
      <h3>
        Upcoming breaks
        <span className="upcoming-breaks-count">
          {preds.length} in next {Math.round((data?.lookahead_min ?? 240) / 60)}h
        </span>
      </h3>

      {data?.artifact_stale && (
        <div
          className="upcoming-breaks-stale-banner"
          title={`Prediction model last trained ${data.artifact_age_days} days ago`}
        >
          Model is getting old — predictions may be less accurate.
        </div>
      )}
      {data?.artifact_age_days === null && (
        <div className="upcoming-breaks-stale-banner">
          Prediction model not yet trained — only live detection available.
        </div>
      )}

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
                <span className="upcoming-break-time">
                  {formatTime(p.predicted_start)}
                  <span className="upcoming-break-sigma">±{Math.round(p.sigma_min)} min</span>
                </span>
              </div>
              <div className="upcoming-break-row-meta">
                <span
                  className={`upcoming-break-likelihood ${likelihoodClass(p.confidence)}`}
                  title={`confidence ${Math.round(p.confidence * 100)}%`}
                >
                  {likelihoodLabel(p.confidence)}
                </span>
                <span className="upcoming-break-lead">{leadLabel(p.lead_min)}</span>
                {p.driver_id !== null && (
                  <span
                    className="upcoming-break-driver"
                    title="Prediction narrowed using today's driver"
                  >
                    driver-matched
                  </span>
                )}
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
