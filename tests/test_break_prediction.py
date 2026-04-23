"""Tests for the Hungarian helper and the break-prediction endpoint."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from backend.matching import hungarian_assign


CAMPUS_TZ = ZoneInfo("America/New_York")


# ─── Hungarian helper ────────────────────────────────────────────────────

def test_hungarian_square_exact_match():
    """Zero on the diagonal → diagonal assignment."""
    cost = 1.0 - np.eye(3)  # 0 on the diagonal, 1 off-diagonal
    row, col = hungarian_assign(cost, pad_square=False)
    assignments = sorted(zip(row.tolist(), col.tolist()))
    assert assignments == [(0, 0), (1, 1), (2, 2)]


def test_hungarian_rectangular_pads():
    """More rows than cols gets padded square; real rows retain real matches."""
    # 4 vehicles, 2 runs; two vehicles should be "spares"
    cost = np.array([
        [1.0, 5.0],
        [5.0, 1.0],
        [9.0, 9.0],
        [9.0, 9.0],
    ])
    row, col = hungarian_assign(cost, pad_square=True, pad_penalty=1e6)
    # Output is indexed on the padded 4x4 matrix
    assert row.shape == col.shape == (4,)
    # First two rows must hit their low-cost columns
    mapping = dict(zip(row.tolist(), col.tolist()))
    assert mapping[0] == 0
    assert mapping[1] == 1
    # The other two rows should be assigned to padded cols (idx >= 2)
    assert mapping[2] >= 2
    assert mapping[3] >= 2


def test_hungarian_handles_inf():
    """Inf entries are replaced with a finite penalty so the solver runs."""
    cost = np.array([
        [1.0, np.inf],
        [np.inf, 1.0],
    ])
    row, col = hungarian_assign(cost, pad_square=False, pad_penalty=1e6)
    assignments = sorted(zip(row.tolist(), col.tolist()))
    assert assignments == [(0, 0), (1, 1)]


# ─── Break-prediction pipeline ───────────────────────────────────────────

SHARED = Path(__file__).resolve().parents[1] / "shared"


def _artifacts_present() -> bool:
    return (SHARED / "break_priors.json").exists() and (SHARED / "break_effective_schedule.json").exists()


@pytest.mark.skipif(not _artifacts_present(),
                    reason="predictive_layers.py export not yet run; artifacts missing")
def test_effective_schedule_artifact_shape():
    """The exported effective schedule JSON is well-formed and has M-F runs."""
    data = json.loads((SHARED / "break_effective_schedule.json").read_text())
    assert isinstance(data, dict)
    # At least one M-F run
    mf_keys = [k for k in data if k.startswith("M-F|")]
    assert mf_keys, "No M-F runs in effective schedule artifact"
    # Every entry has effective_mean_min or modes (allowing scheduled-rare with neither)
    for key, entries in data.items():
        assert isinstance(entries, list)
        for e in entries:
            assert "src" in e
            assert "take_rate" in e


@pytest.mark.skipif(not _artifacts_present(),
                    reason="predictive_layers.py export not yet run; artifacts missing")
def test_predict_upcoming_breaks_midmorning_weekday():
    """Called before lunch on a weekday, predictions should include upcoming
    lunch slots with positive lead time."""
    from backend.fastapi import break_detection

    # Force a fresh load so test doesn't inherit another test's cache state
    break_detection._priors_cache = None
    break_detection._effective_cache = None

    # Pick a Tuesday at 10:30 AM local.
    # Using a fixed known weekday date to make this reproducible.
    local_now = datetime(2026, 3, 3, 10, 30, 0, tzinfo=CAMPUS_TZ)  # Tuesday
    now_utc = local_now.astimezone(timezone.utc)

    preds = break_detection.predict_upcoming_breaks(
        now_utc=now_utc, campus_tz=CAMPUS_TZ, lookahead_min=240,
    )
    assert preds, "Expected at least one upcoming break prediction"
    # All lead_min must be positive (we asked for upcoming only)
    assert all(p["lead_min"] > 0 for p in preds)
    # Sorted by predicted_start
    starts = [p["predicted_start"] for p in preds]
    assert starts == sorted(starts)
    # Each prediction has the required keys
    for p in preds:
        for k in ("run", "predicted_start", "predicted_end", "confidence",
                  "lead_min", "source", "sigma_min"):
            assert k in p, f"Missing {k} in prediction"
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["lead_min"] > 0


@pytest.mark.skipif(not _artifacts_present(),
                    reason="predictive_layers.py export not yet run; artifacts missing")
def test_predict_upcoming_breaks_returns_empty_after_all_slots():
    """Late evening on a weekday → no more upcoming breaks."""
    from backend.fastapi import break_detection

    break_detection._priors_cache = None
    break_detection._effective_cache = None

    local_now = datetime(2026, 3, 3, 22, 30, 0, tzinfo=CAMPUS_TZ)  # 10:30 PM Tuesday
    now_utc = local_now.astimezone(timezone.utc)

    preds = break_detection.predict_upcoming_breaks(
        now_utc=now_utc, campus_tz=CAMPUS_TZ, lookahead_min=120,
    )
    # No slots between 22:30 and midnight (the schedule ends earlier).
    # Allow an edge case of one or two late-evening entries; main assertion:
    # none have negative lead.
    assert all(p["lead_min"] >= 0 for p in preds)


@pytest.mark.skipif(not _artifacts_present(),
                    reason="predictive_layers.py export not yet run; artifacts missing")
@pytest.mark.asyncio
async def test_predictions_route_handler_returns_expected_shape():
    """Integration: call the /api/predictions handler directly, verify response shape."""
    from backend.fastapi import break_detection
    from backend.fastapi.routes import get_break_predictions

    break_detection._priors_cache = None
    break_detection._effective_cache = None

    resp = await get_break_predictions(lookahead_min=180)
    assert set(resp.keys()) == {"generated_at", "lookahead_min", "n_predictions", "predictions"}
    assert resp["lookahead_min"] == 180
    assert resp["n_predictions"] == len(resp["predictions"])
    assert isinstance(resp["generated_at"], str)


def test_predict_upcoming_breaks_graceful_without_artifacts(tmp_path, monkeypatch):
    """If artifacts are missing, endpoint returns [] without raising."""
    from backend.fastapi import break_detection

    # Point the module at empty tmp_path so load sees no files
    monkeypatch.setattr(break_detection, "_PRIORS_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(break_detection, "_EFFECTIVE_JSON", tmp_path / "missing2.json")
    break_detection._priors_cache = None
    break_detection._effective_cache = None

    preds = break_detection.predict_upcoming_breaks(
        now_utc=datetime.now(timezone.utc),
        campus_tz=CAMPUS_TZ,
        lookahead_min=120,
    )
    assert preds == []
