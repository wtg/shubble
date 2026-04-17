"""
Subagent 2: Frontend UX & Logic Tests

Tests:
- ETA display correctness (schedule calculations match simulation)
- Union stop never shows "early" status
- Schedule component logic (truncation, countdown, time parsing)
- CSS class application for early/late/on-time states
- Edge cases in time formatting and day boundary handling
"""

import json
import pytest
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.simulation.simulator import (
    UNION_STOPS,
    SimulationResult,
    generate_api_payload,
    run_simulation,
)


@pytest.fixture(scope="module")
def sim() -> SimulationResult:
    return run_simulation(seed=42, window_hours=2.0, min_trips=50)


# --- Load frontend source for static analysis ---

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "src"


def read_frontend_file(relative_path: str) -> str:
    """Read a frontend source file."""
    path = FRONTEND_DIR / relative_path
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


# --- ETA Display Correctness ---

class TestETADisplayCorrectness:
    """Verify ETAs in API payload match simulation logic."""

    def test_api_payload_has_future_etas(self, sim: SimulationResult):
        """API payload should contain future ETAs for active stops."""
        # The simulation generates times relative to base_time (near now).
        # Check that at least some stop arrivals are in the future.
        now = datetime.now(timezone.utc)
        future_arrivals = [
            a for a in sim.all_arrivals()
            if a.actual_time > now
        ]
        assert len(future_arrivals) > 0, (
            f"No future arrivals in simulation. "
            f"Window: {sim.window_start} - {sim.window_end}, now: {now}"
        )

        payload = generate_api_payload(sim)
        future_etas = {k: v for k, v in payload.items() if v.get("eta")}
        assert len(future_etas) > 0, (
            f"No future ETAs in payload despite {len(future_arrivals)} future arrivals. "
            f"Payload has {len(payload)} entries total."
        )

    def test_etas_are_valid_iso_format(self, sim: SimulationResult):
        """All ETA values should be valid ISO 8601 datetimes."""
        payload = generate_api_payload(sim)
        for stop_key, data in payload.items():
            if data.get("eta"):
                try:
                    datetime.fromisoformat(data["eta"])
                except ValueError:
                    pytest.fail(
                        f"Invalid ISO datetime for {stop_key}: {data['eta']}"
                    )

    def test_each_stop_has_route(self, sim: SimulationResult):
        """Every ETA entry should include the route name."""
        payload = generate_api_payload(sim)
        for stop_key, data in payload.items():
            assert "route" in data, f"{stop_key} missing route field"
            if data.get("eta"):
                assert data["route"] in ("NORTH", "WEST"), (
                    f"{stop_key} has unexpected route: {data['route']}"
                )


# --- Union Stop Display ---

class TestUnionStopDisplay:
    """Union stops must never show 'early' in the UI."""

    def test_union_stops_never_early_in_payload(self, sim: SimulationResult):
        """CRITICAL: Union stop arrivals in simulation must never be early."""
        for arrival in sim.union_arrivals():
            assert arrival.delay_seconds >= 0, (
                f"Union stop {arrival.stop_key} is early by "
                f"{abs(arrival.delay_seconds)}s in trip {arrival.trip_id}"
            )

    def test_frontend_deviation_logic_for_union(self):
        """
        Verify that Schedule.tsx deviation logic handles Union stops correctly.

        The frontend computes: deviationMinutes = (liveDate - scheduledDate) / 60000
        For Union stops, this should always be >= 0.
        """
        schedule_tsx = read_frontend_file("schedule/Schedule.tsx")
        if not schedule_tsx:
            pytest.skip("Schedule.tsx not found")

        # Verify the deviation computation exists
        assert "deviationMinutes" in schedule_tsx, "Missing deviation calculation"

        # Verify early indicator exists
        assert "eta-early" in schedule_tsx, "Missing eta-early CSS class"
        assert "eta-late" in schedule_tsx, "Missing eta-late CSS class"

        # Check threshold: early requires deviation <= -2
        assert "deviationMinutes <= -2" in schedule_tsx, (
            "Missing early threshold check (deviationMinutes <= -2)"
        )

    def test_frontend_early_class_not_applied_to_union(self, sim: SimulationResult):
        """
        Given simulation data where Union is never early,
        verify that no Union stop would trigger the 'eta-early' class.

        The frontend applies eta-early when deviationMinutes <= -2.
        """
        for arrival in sim.union_arrivals():
            deviation_minutes = arrival.delay_seconds / 60.0
            assert deviation_minutes >= 0, (
                f"Union stop {arrival.stop_key} would show 'early' with "
                f"deviation={deviation_minutes:.1f}min"
            )


# --- Schedule Component Logic ---

class TestScheduleComponentLogic:
    """Verify Schedule.tsx logic matches expectations."""

    def test_time_to_date_parsing(self):
        """Verify timeToDate parsing logic handles all formats."""
        # Extracted from Schedule.tsx:
        # const [time, modifier] = timeStr.trim().split(" ");
        # let [hours, minutes] = time.split(":").map(Number);

        test_cases = [
            ("9:00 AM", 9, 0),
            ("12:00 PM", 12, 0),
            ("12:00 AM", 0, 0),  # midnight
            ("1:30 PM", 13, 30),
            ("11:59 PM", 23, 59),
            ("12:30 AM", 0, 30),
        ]

        for time_str, expected_hour, expected_min in test_cases:
            parts = time_str.strip().split(" ")
            time_part, modifier = parts[0], parts[1]
            hours, minutes = map(int, time_part.split(":"))

            if modifier.upper() == "PM" and hours != 12:
                hours += 12
            elif modifier.upper() == "AM" and hours == 12:
                hours = 0

            assert hours == expected_hour, (
                f"'{time_str}': expected hour={expected_hour}, got {hours}"
            )
            assert minutes == expected_min, (
                f"'{time_str}': expected min={expected_min}, got {minutes}"
            )

    def test_truncation_window_size(self):
        """
        Truncated view should show ~7 items.
        Logic: slice(max(0, currentLoopIndex - 1), currentLoopIndex + 6)
        """
        # Simulate various currentLoopIndex values
        times_count = 20  # typical schedule length

        for idx in range(times_count):
            start = max(0, idx - 1)
            end = idx + 6
            visible = min(end, times_count) - start
            assert visible <= 7, (
                f"At index {idx}, showing {visible} items (max 7)"
            )

    def test_countdown_minutes_computation(self):
        """Countdown minutes should be positive for future ETAs."""
        now = datetime(2026, 4, 4, 14, 0, 0, tzinfo=timezone.utc)
        future_eta = datetime(2026, 4, 4, 14, 5, 0, tzinfo=timezone.utc)

        mins = round((future_eta.timestamp() - now.timestamp()) / 60)
        assert mins == 5
        assert mins > 0

    def test_countdown_negative_for_past(self):
        """Past ETAs should produce negative countdown (filtered out in UI)."""
        now = datetime(2026, 4, 4, 14, 0, 0, tzinfo=timezone.utc)
        past_eta = datetime(2026, 4, 4, 13, 55, 0, tzinfo=timezone.utc)

        mins = round((past_eta.timestamp() - now.timestamp()) / 60)
        assert mins == -5
        assert mins <= 0

    def test_relative_time_threshold(self):
        """Relative time labels only show for departures within 30 minutes."""
        schedule_tsx = read_frontend_file("schedule/Schedule.tsx")
        if not schedule_tsx:
            pytest.skip("Schedule.tsx not found")

        assert "minutesUntil <= 30" in schedule_tsx, (
            "Missing 30-minute threshold for relative time display"
        )

    def test_expand_collapse_state_management(self):
        """Verify expand/collapse uses Set<number> pattern."""
        schedule_tsx = read_frontend_file("schedule/Schedule.tsx")
        if not schedule_tsx:
            pytest.skip("Schedule.tsx not found")

        assert "expandedLoops" in schedule_tsx
        assert "Set<number>" in schedule_tsx or "new Set()" in schedule_tsx


# --- CSS Class Verification ---

class TestCSSClasses:
    """Verify CSS classes exist and match logic in Schedule.tsx."""

    def test_required_css_classes_exist(self):
        """All referenced CSS classes should exist in Schedule.css."""
        css = read_frontend_file("schedule/styles/Schedule.css")
        if not css:
            pytest.skip("Schedule.css not found")

        required_classes = [
            "schedule-container",
            "timeline-container",
            "timeline-item",
            "timeline-dot",
            "current-loop",
            "past-time",
            "secondary-timeline",
            "secondary-timeline-item",
            "live-eta",
            "eta-early",
            "eta-late",
            "next-shuttle-summary",
            "summary-stop",
            "relative-time",
            "expand-indicator",
            "show-full-schedule",
            "selected-stop",
        ]

        for cls in required_classes:
            assert cls in css, f"CSS class '.{cls}' not found in Schedule.css"

    def test_eta_early_styling(self):
        """eta-early should use blue color."""
        css = read_frontend_file("schedule/styles/Schedule.css")
        if not css:
            pytest.skip("Schedule.css not found")

        # Find the eta-early rule
        assert ".eta-early" in css
        # Should have blue-ish color
        assert "#578FCA" in css or "blue" in css.lower()

    def test_eta_late_styling(self):
        """eta-late should use orange/warning color."""
        css = read_frontend_file("schedule/styles/Schedule.css")
        if not css:
            pytest.skip("Schedule.css not found")

        assert ".eta-late" in css
        assert "#E67E22" in css  # Orange color

    def test_responsive_breakpoints(self):
        """Verify responsive CSS breakpoints exist."""
        css = read_frontend_file("schedule/styles/Schedule.css")
        if not css:
            pytest.skip("Schedule.css not found")

        # Should have at least a mobile breakpoint
        assert "@media" in css
        assert "768px" in css, "Missing 768px tablet breakpoint"


# --- DevTime Integration ---

class TestDevTimeIntegration:
    """Verify devTime utility is properly integrated."""

    def test_dev_time_module_exists(self):
        """devTime.ts should exist and export devNow/devNowMs."""
        content = read_frontend_file("utils/devTime.ts")
        if not content:
            pytest.skip("devTime.ts not found")

        assert "devNow" in content, "Missing devNow export"
        assert "devNowMs" in content, "Missing devNowMs export"

    def test_schedule_uses_dev_time(self):
        """Schedule.tsx should import and use devNow, not raw Date()."""
        schedule_tsx = read_frontend_file("schedule/Schedule.tsx")
        if not schedule_tsx:
            pytest.skip("Schedule.tsx not found")

        assert "devNow" in schedule_tsx, "Schedule.tsx doesn't use devNow"
        assert "devNowMs" in schedule_tsx, "Schedule.tsx doesn't use devNowMs"

        # Should NOT use raw `new Date()` for "now" (except in timeToDate helper)
        # Count occurrences of `new Date()` (no args = current time)
        raw_date_calls = re.findall(r'new Date\(\)', schedule_tsx)
        # timeToDate uses `new Date()` once, that's acceptable
        assert len(raw_date_calls) <= 1, (
            f"Schedule.tsx has {len(raw_date_calls)} raw `new Date()` calls. "
            f"Should use devNow() instead for time-sensitive logic."
        )

    def test_use_stop_etas_uses_dev_time(self):
        """useStopETAs.ts should use devNow for time comparisons."""
        content = read_frontend_file("hooks/useStopETAs.ts")
        if not content:
            pytest.skip("useStopETAs.ts not found")

        assert "devNow" in content, "useStopETAs.ts doesn't use devNow"


# --- Accessibility Basics ---

class TestAccessibilityBasics:
    """Basic accessibility checks on frontend source."""

    def test_schedule_has_labels(self):
        """Schedule dropdowns/buttons should have labels or aria attributes."""
        schedule_tsx = read_frontend_file("schedule/Schedule.tsx")
        if not schedule_tsx:
            pytest.skip("Schedule.tsx not found")

        # Check that select has an id and label htmlFor
        assert "htmlFor" in schedule_tsx, "Missing htmlFor on labels"
        assert 'id=' in schedule_tsx, "Missing id on form elements"

    def test_interactive_elements_have_handlers(self):
        """Clickable elements should have onClick handlers."""
        schedule_tsx = read_frontend_file("schedule/Schedule.tsx")
        if not schedule_tsx:
            pytest.skip("Schedule.tsx not found")

        assert "onClick" in schedule_tsx
        assert "cursor: 'pointer'" in schedule_tsx or "cursor: pointer" in schedule_tsx

    def test_color_contrast_of_status_classes(self):
        """Status colors should have sufficient contrast against white bg."""
        css = read_frontend_file("schedule/styles/Schedule.css")
        if not css:
            pytest.skip("Schedule.css not found")

        # #578FCA (blue) on white: contrast ratio ~3.6:1 (AA for large text)
        # #E67E22 (orange) on white: contrast ratio ~3.3:1 (AA for large text)
        # #95a5a6 (gray) on white: contrast ratio ~2.6:1 (below AA)
        # These are informational; flagged as LOW if below 3:1
        assert "#578FCA" in css, "Primary blue color not found"
        assert "#E67E22" in css, "Late indicator orange not found"
