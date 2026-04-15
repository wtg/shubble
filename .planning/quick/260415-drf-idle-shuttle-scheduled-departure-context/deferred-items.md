---
task: 260415-drf
created: 2026-04-15
---

# Deferred Items (Out of Scope)

Items discovered during execution of quick task 260415-drf that are
pre-existing and NOT caused by changes in this task. Logged here per
GSD SCOPE BOUNDARY rule — only auto-fix issues DIRECTLY caused by the
current task's changes.

## Pre-existing test failures

### `tests/simulation/test_frontend_ux.py::TestUnionStopDisplay::test_frontend_deviation_logic_for_union`

- **Status:** Pre-existing failure (reproduces on base commit before
  any task 260415-drf edits — verified via `git stash` isolation).
- **Assertion:** `"deviationMinutes" in schedule_tsx` — the test
  expects the Schedule.tsx source to contain a token named
  `deviationMinutes`. Current source uses a different naming
  (likely computed inline as `delta` / `deviationSec` inside
  `getDepartureLabel`).
- **Out of scope:** Not introduced by this task; the Schedule.tsx
  change in Task 3 only adds a waiting pill and does not touch
  deviation-calculation logic.
- **Suggested follow-up:** Either rename the helper to include a
  `deviationMinutes` local var for grep-ability, or update the
  simulation test to match the current source.
