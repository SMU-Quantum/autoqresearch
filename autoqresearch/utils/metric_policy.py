"""Fixed metric semantics for the active knapsack policy-discovery workflow."""

from __future__ import annotations

import math


PRIMARY_METRIC = "suite_average_gap"
LOWER_IS_BETTER = True
DEV_REGRESSION_TOLERANCE = 0.02


def _normalize_gap(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_strict_improvement(
    candidate: float | None,
    incumbent: float | None,
    eps: float = 1e-9,
) -> bool:
    """Return True when the candidate gap is strictly better than the incumbent."""

    candidate_gap = _normalize_gap(candidate)
    incumbent_gap = _normalize_gap(incumbent)
    if candidate_gap is None:
        return False
    if incumbent_gap is None:
        return True
    return candidate_gap < (incumbent_gap - eps)


def passes_dev_guardrail(
    candidate_dev: float | None,
    incumbent_dev: float | None,
    tolerance: float = DEV_REGRESSION_TOLERANCE,
) -> bool:
    """Reject candidates that regress too far on the dev split."""

    candidate_gap = _normalize_gap(candidate_dev)
    incumbent_gap = _normalize_gap(incumbent_dev)
    if candidate_gap is None:
        candidate_gap = 1.0
    if incumbent_gap is None:
        return True
    return candidate_gap <= (incumbent_gap + float(tolerance))


def accept_candidate(
    candidate_train: float | None,
    candidate_dev: float | None,
    incumbent_train: float | None,
    incumbent_dev: float | None,
) -> bool:
    """Keep only strict train improvements that do not regress badly on dev."""

    return is_strict_improvement(candidate_train, incumbent_train) and passes_dev_guardrail(
        candidate_dev,
        incumbent_dev,
    )
