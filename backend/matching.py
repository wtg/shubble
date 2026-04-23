"""Shared matching utilities — house Hungarian solver for vehicle↔schedule problems.

This module is the single place where vehicle-to-schedule assignment via the
Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) is implemented
for this codebase. Callers build a cost matrix for their specific problem and
hand it to :func:`hungarian_assign`.

**Attribution.** The Hungarian pattern in this codebase was originally authored
by the project lead and team members in ``shared/schedules.py``
(``Schedule.match_shuttles_to_schedules``) for matching shuttle stop patterns
to scheduled routes. The pattern is extracted here so every caller depends on
one canonical helper rather than duplicating the ``linear_sum_assignment`` call
+ padding logic.

Current callers:
  * ``shared/schedules.py`` — matches shuttle stop patterns to scheduled routes
    (retrospective labelling, production, Redis-cached).
  * ``.planning/debug/vehicle_to_run.py`` — matches vehicles to scheduled runs
    by morning Union-departure timings (offline research, feeds the predictive
    break-forecasting stack).
  * ``backend/fastapi/break_detection.py::predict_upcoming_break`` — wraps the
    offline-trained priors + effective schedule to announce upcoming breaks
    ahead of time (production predictive path).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_assign(
    cost: np.ndarray,
    pad_square: bool = True,
    pad_penalty: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the Hungarian algorithm on a cost matrix.

    Args:
        cost: Rectangular cost matrix, shape ``(n_rows, n_cols)``. Rows and
            columns are whatever the caller is matching (e.g. vehicles vs
            scheduled runs). Lower cost = better match. Infinite costs are
            replaced with ``pad_penalty`` so the solver still runs.
        pad_square: If True (default), pad the matrix with a dummy row or
            column of ``pad_penalty`` so the solver gets a square input. The
            caller is responsible for filtering padded-row/col assignments
            out of the result (check against original dimensions and against
            ``pad_penalty`` cost).
        pad_penalty: Finite penalty used for (a) replacing ``inf`` entries in
            ``cost`` and (b) filling padded rows/cols when ``pad_square``.

    Returns:
        ``(row_ind, col_ind)`` — the assignment arrays from
        :func:`scipy.optimize.linear_sum_assignment`. Indices correspond to
        the *padded* matrix when ``pad_square`` is True; callers should
        compare each ``(r, c)`` against the original ``(n_rows, n_cols)`` and
        the padded cost to tell real assignments from padding artifacts.
    """
    cost = np.asarray(cost, dtype=float)
    cost = np.where(np.isinf(cost), pad_penalty, cost)

    n_rows, n_cols = cost.shape
    if pad_square and n_rows != n_cols:
        if n_rows > n_cols:
            pad = np.full((n_rows, n_rows - n_cols), pad_penalty)
            cost = np.hstack([cost, pad])
        else:
            pad = np.full((n_cols - n_rows, n_cols), pad_penalty)
            cost = np.vstack([cost, pad])

    return linear_sum_assignment(cost)
