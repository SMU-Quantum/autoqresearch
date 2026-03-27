#!/usr/bin/env python3
"""
Adaptive multi-attempt experiment for constrained quantum optimization (knapsack).

Architecture (from Research Plan v4):

  POLICY SURFACE (agent edits)
    choose_solver_family(problem)           → family string
    build_base_policy(problem, family)      → base policy dict
    should_continue(attempt, history, problem, ...) → bool
    adapt_policy(attempt, history, problem, ...)    → adapted policy dict

  INFRASTRUCTURE (fixed)
    while should_continue(...):
        policy = adapt_policy(attempt, history, problem, ...)
        result = solver.solve(problem, policy, backend)
        raw_eval = evaluate(result, problem)
        improvement, stagnation, final = _normalize_convergence(result.convergence_history)
        learning = _compute_learning_score(raw_eval, feasibility_rate, best_feasible_ar)
        outcome = AttemptOutcome(raw metrics + learning_score + convergence stats)
        history.append(outcome)

    repaired = _fixed_repair(best_raw_result.bitstring, problem)
    repaired_eval = evaluate(repaired, problem)
    print(raw metrics, repaired metrics)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


DEPTH_PENALTY_WEIGHT = 0.05
TWO_QUBIT_PENALTY_WEIGHT = 0.05
DEPTH_PENALTY_SCALE = 500.0
TWO_QUBIT_PENALTY_SCALE = 200.0
DEFAULT_ESTIMATOR_SHOTS = 1000
DEFAULT_SAMPLER_SHOTS = 1000
DEFAULT_PROBLEM_SPEC = "knapsack_12"
DEFAULT_RESULTS_PATH = Path("results.tsv")
DEFAULT_PROGRESS_PATH = Path("instance_progress.png")


# ─── AttemptOutcome ──────────────────────────────────────────────


@dataclass
class AttemptOutcome:
    """Observation given to the agent after each solver attempt."""

    attempt: int
    learning_score: float       # feasibility-shaped signal (lower is better)
    optimality_gap: float       # (optimal - found) / optimal. Lower is better. THE METRIC.
    raw_feasible: bool          # did the solver produce a feasible best bitstring?
    raw_feasibility_rate: float # how close the distribution is to feasible
    raw_ar: float               # approximation ratio of best raw solution

    convergence_improvement: float  # (start - end) / |start|; > 0 means cost decreased
    convergence_stagnation: float   # > 0.8 reliably means stuck (cross-optimizer)
    final_cost: float               # terminal cost value

    policy_used: dict = field(default_factory=dict)
    wall_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# POLICY SURFACE — the agent edits these four functions
# ═══════════════════════════════════════════════════════════════════


def _tightness(problem) -> float:
    weights = problem.metadata.get("weights")
    capacity = problem.metadata.get("capacity")
    if weights is None or capacity is None:
        return 1.0
    total_weight = float(np.sum(np.asarray(weights, dtype=float)))
    if total_weight <= 0.0:
        return 1.0
    return float(capacity) / total_weight


def _slack_ratio(problem) -> float:
    n_items = int(problem.metadata.get("num_items", problem.num_variables))
    return float(problem.num_variables) / max(1.0, float(n_items))


def _shot_budget(problem) -> tuple[int, int]:
    return DEFAULT_ESTIMATOR_SHOTS, DEFAULT_SAMPLER_SHOTS


def choose_solver_family(problem) -> str:
    """Choose the initial solver family based on problem characteristics.

    `adapt_policy()` may switch families on later attempts after observing the
    trajectory. Knapsack starts from the simplest VQE baseline.
    """
    n_qubo = int(problem.num_variables)

    if problem.problem_type == "knapsack":
        return "vqe"

    if n_qubo <= 14:
        return "qaoa"
    return "vqe"


def build_base_policy(problem, family: str) -> dict:
    """Build initial policy for the chosen solver family."""
    est_shots, samp_shots = _shot_budget(problem)
    tightness = _tightness(problem)
    n_qubo = int(problem.num_variables)

    if family == "qaoa":
        reps = 1 if n_qubo <= 12 else 2 if n_qubo <= 20 else 3
        return {
            "variant": "standard",
            "measurement_mode": "expectation",
            "reps": reps,
            "initialization": "random",
            "ws_source": "relaxation",
            "ws_epsilon": 0.25,
            "ma_tying": "none",
            "optimizer_method": "COBYLA",
            "optimizer_maxiter": 120 if n_qubo <= 14 else 180,
            "optimizer_tol": 5e-4,
            "cvar_alpha": 0.18 if tightness <= 0.42 else 0.25,
            "estimator_shots": est_shots,
            "sampler_shots": samp_shots,
            "learning_rate": 0.05,
            "seed": 42,
            "solver_family": family,
        }
    if family == "vqe":
        return {
            "variant": "standard",
            "measurement_mode": "expectation",
            "ansatz_type": "real_amplitudes",
            "vqe_reps": 1,
            "entanglement": "linear",
            "optimizer_method": "COBYLA",
            "optimizer_maxiter": 140 if n_qubo <= 16 else 200,
            "optimizer_tol": 1e-3,
            "cvar_alpha": 0.20 if tightness <= 0.42 else 0.30,
            "estimator_shots": est_shots,
            "sampler_shots": samp_shots,
            "learning_rate": 0.05,
            "seed": 42,
            "solver_family": family,
        }
    if family == "qrao":
        reps = 1 if n_qubo <= 20 else 2
        return {
            "qrac_type": 3 if _slack_ratio(problem) >= 1.8 else 2,
            "rounding": "semideterministic",
            "measurement_mode": "expectation",
            "ansatz_type": "real_amplitudes",
            "entanglement": "linear",
            "vqe_reps": reps,
            "optimizer_method": "COBYLA",
            "optimizer_maxiter": 120,
            "optimizer_tol": 1e-3,
            "estimator_shots": est_shots,
            "sampler_shots": samp_shots,
            "learning_rate": 0.05,
            "seed": 42,
            "solver_family": family,
        }
    if family == "pce":
        return {
            "pce_k": 2 if n_qubo <= 16 else 3,
            "pce_depth": 2 if n_qubo <= 16 else 3,
            "measurement_mode": "expectation",
            "ansatz_type": "brickwork",
            "entanglement": "linear",
            "initialization": "random",
            "optimizer_method": "COBYLA",
            "optimizer_maxiter": 80 if n_qubo <= 16 else 120,
            "optimizer_tol": 1e-3,
            "pce_beta": 0.5,
            "pce_local_search": True,
            "estimator_shots": est_shots,
            "sampler_shots": samp_shots,
            "learning_rate": 0.05,
            "seed": 42,
            "solver_family": family,
        }
    raise ValueError(f"Unknown solver family: {family}")


def _summarize_family_history(history: list[AttemptOutcome]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for outcome in history:
        family = str((outcome.policy_used or {}).get("solver_family", "")).lower()
        if not family:
            continue
        stats = summary.setdefault(
            family,
            {
                "attempts": 0,
                "best_gap": float("inf"),
                "best_feasibility_rate": 0.0,
                "best_ar": 0.0,
                "last_attempt": -1,
            },
        )
        stats["attempts"] = int(stats["attempts"]) + 1
        stats["best_gap"] = min(float(stats["best_gap"]), float(outcome.optimality_gap))
        stats["best_feasibility_rate"] = max(
            float(stats["best_feasibility_rate"]),
            float(outcome.raw_feasibility_rate),
        )
        stats["best_ar"] = max(float(stats["best_ar"]), float(outcome.raw_ar))
        stats["last_attempt"] = int(outcome.attempt)
    return summary


def _score_family_candidate(
    candidate: str,
    current_family: str,
    history: list[AttemptOutcome],
    last: AttemptOutcome,
    improving: bool,
    plateaued: bool,
    stalled: bool,
) -> float:
    family_stats = _summarize_family_history(history).get(candidate, {})
    attempts = int(family_stats.get("attempts", 0))
    best_gap = float(family_stats.get("best_gap", float("inf")))
    best_feasibility = float(family_stats.get("best_feasibility_rate", 0.0))
    best_ar = float(family_stats.get("best_ar", 0.0))
    last_attempt = int(family_stats.get("last_attempt", -100))

    score = 0.0
    if candidate == current_family:
        score += 0.30
    elif attempts == 0:
        score += 0.25
    else:
        score += 0.05

    if improving and candidate == current_family and not stalled:
        score += 0.50
    if plateaued or stalled:
        score += 0.40 if candidate != current_family else -0.35

    if attempts > 0:
        if best_gap < float("inf"):
            score += 0.35 * max(0.0, 1.0 - best_gap)
        score += 0.12 * best_feasibility
        score += 0.10 * best_ar
        score -= 0.10 * max(0, attempts - 1)
        if candidate != current_family and history:
            score -= 0.08 * max(0, 2 - (last.attempt - last_attempt))

    if not last.raw_feasible:
        if candidate == "qaoa":
            score += 0.65 if last.raw_feasibility_rate < 0.10 else 0.30
        elif candidate == "qrao":
            score += 0.35 if last.raw_feasibility_rate < 0.20 else 0.15
        elif candidate == "pce":
            score += 0.45 if last.raw_feasibility_rate < 0.05 or stalled else 0.18
        elif candidate == "vqe":
            score += 0.20 if len(history) == 1 else 0.05
    else:
        if candidate == "vqe":
            score += 0.45 if last.raw_feasibility_rate > 0.70 and last.raw_ar < 0.92 else 0.12
        elif candidate == "qaoa":
            score += 0.45 if last.raw_ar < 0.90 else 0.10
            if last.raw_feasibility_rate > 0.60 and (plateaued or stalled):
                score += 0.10
        elif candidate == "qrao":
            score += 0.25 if last.raw_ar < 0.80 and last.raw_feasibility_rate < 0.95 else 0.08
        elif candidate == "pce":
            score += 0.25 if plateaued or stalled else 0.08

    if candidate == current_family and len(history) == 1 and last.raw_feasibility_rate >= 0.10:
        score += 0.20

    return score


def _choose_followup_family(
    attempt: int,
    history: list[AttemptOutcome],
    problem,
    current_family: str,
    improving: bool,
    plateaued: bool,
    stalled: bool,
) -> str:
    if problem is None or getattr(problem, "problem_type", "") != "knapsack" or not history:
        return current_family

    last = history[-1]
    candidates = ["vqe", "qaoa", "qrao", "pce"]
    scores = {
        candidate: _score_family_candidate(
            candidate, current_family, history, last, improving, plateaued, stalled
        )
        for candidate in candidates
    }
    best_family = max(scores, key=scores.get)
    switch_margin = 0.10 if stalled or plateaued or not last.raw_feasible else 0.20
    if best_family != current_family and scores[best_family] >= scores[current_family] + switch_margin:
        return best_family
    return current_family


def _summarize_family_ansatzes(
    history: list[AttemptOutcome],
    family: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in history:
        used = outcome.policy_used or {}
        if str(used.get("solver_family", "")).lower() != family:
            continue
        ansatz = str(used.get("ansatz_type", "")).lower()
        if ansatz:
            counts[ansatz] = counts.get(ansatz, 0) + 1
    return counts


def _score_ansatz_candidate(
    candidate: str,
    current_ansatz: str,
    family: str,
    tried_counts: dict[str, int],
    last: AttemptOutcome,
    improving: bool,
    plateaued: bool,
    stalled: bool,
) -> float:
    tried = int(tried_counts.get(candidate, 0))
    score = 0.0

    if candidate == current_ansatz:
        score += 0.20
    elif tried == 0:
        score += 0.25
    else:
        score -= 0.08 * tried

    if improving and candidate == current_ansatz and not stalled:
        score += 0.35
    if plateaued or stalled:
        score += 0.20 if candidate != current_ansatz else -0.15

    if candidate == "real_amplitudes":
        score += 0.35 if (not last.raw_feasible or last.raw_feasibility_rate < 0.25) else 0.05
        if family in {"vqe", "qrao"}:
            score += 0.05
    elif candidate == "efficient_su2":
        score += 0.35 if (last.raw_feasible and last.raw_ar < 0.92) else 0.15
        if stalled or plateaued:
            score += 0.15
    elif candidate == "pauli_two_design":
        score += 0.30 if (last.raw_feasible and last.raw_ar < 0.90) else 0.12
        if stalled or plateaued:
            score += 0.18
    elif candidate == "brickwork":
        score += 0.35 if (stalled or (last.raw_feasible and last.raw_feasibility_rate > 0.60 and last.raw_ar < 0.90)) else -0.05
        if family == "pce":
            score += 0.20

    return score


def _choose_followup_ansatz(
    family: str,
    current_ansatz: str,
    history: list[AttemptOutcome],
    last: AttemptOutcome,
    improving: bool,
    plateaued: bool,
    stalled: bool,
) -> str:
    if family not in {"vqe", "qrao", "pce"}:
        return current_ansatz

    candidates = ["real_amplitudes", "efficient_su2", "pauli_two_design", "brickwork"]
    tried_counts = _summarize_family_ansatzes(history, family)
    scores = {
        candidate: _score_ansatz_candidate(
            candidate,
            current_ansatz,
            family,
            tried_counts,
            last,
            improving,
            plateaued,
            stalled,
        )
        for candidate in candidates
    }
    best_ansatz = max(scores, key=scores.get)
    switch_margin = 0.10 if stalled or plateaued else 0.20
    if best_ansatz != current_ansatz and scores[best_ansatz] >= scores[current_ansatz] + switch_margin:
        return best_ansatz
    return current_ansatz


def _choose_followup_optimizer(
    current_optimizer: str,
    last: AttemptOutcome,
    improving: bool,
    plateaued: bool,
    stalled: bool,
) -> str:
    normalized = current_optimizer.upper()
    scores = {
        "COBYLA": 0.25,
        "SPSA": 0.15,
        "ADAM": 0.10,
    }
    scores[normalized] = scores.get(normalized, 0.0) + 0.10

    if improving and not stalled:
        scores[normalized] = scores.get(normalized, 0.0) + 0.30
    if stalled:
        scores["SPSA"] += 0.50
        scores["ADAM"] += 0.25
        scores["COBYLA"] -= 0.40
    elif plateaued:
        scores["SPSA"] += 0.25 if last.raw_feasibility_rate < 0.25 else 0.10
        scores["ADAM"] += 0.20 if last.raw_feasible else 0.05

    return max(scores, key=scores.get)


def should_continue(
    attempt: int,
    history: list[AttemptOutcome],
    problem=None,
    max_attempts: int = 5,
) -> bool:
    """Decide whether to continue with another attempt."""
    if isinstance(problem, (int, np.integer)):
        max_attempts = int(problem)
        problem = None

    if attempt >= max_attempts:
        return False
    if not history:
        return True

    last = history[-1]
    best_gap = min(outcome.optimality_gap for outcome in history)
    if (
        len(history) >= 4
        and last.raw_feasible
        and last.raw_ar >= 0.95
        and last.optimality_gap <= best_gap + 1e-9
    ):
        return False

    if len(history) >= 2:
        prev = history[-2]
        learning_delta = prev.learning_score - last.learning_score
        feasibility_delta = last.raw_feasibility_rate - prev.raw_feasibility_rate
        ar_delta = last.raw_ar - prev.raw_ar

        if (
            len(history) >= 4
            and last.raw_feasible
            and prev.raw_feasible
            and ar_delta <= 0.01
            and last.optimality_gap >= prev.optimality_gap - 0.002
        ):
            return False

        if (
            not last.raw_feasible
            and not prev.raw_feasible
            and feasibility_delta < 0.02
            and learning_delta <= 0.01
            and max(last.convergence_stagnation, prev.convergence_stagnation) > 0.80
        ):
            return False

    if len(history) >= 3:
        recent = history[-3:]
        if (
            all(not outcome.raw_feasible for outcome in recent)
            and recent[-1].raw_feasibility_rate - recent[0].raw_feasibility_rate < 0.05
            and recent[-1].learning_score
            >= min(outcome.learning_score for outcome in recent[:-1]) - 0.01
        ):
            return False

    if len(history) >= 2:
        prev = history[-2]
        if (
            last.convergence_stagnation > 0.85
            and prev.convergence_stagnation > 0.85
            and max(last.raw_feasibility_rate, prev.raw_feasibility_rate) < 0.05
            and last.learning_score >= prev.learning_score - 1e-6
        ):
            return False

    return True


def adapt_policy(
    attempt: int,
    history: list[AttemptOutcome],
    problem,
    base_policy: dict | None = None,
) -> dict:
    """Adapt the policy based on history.

    This is where the agent expresses state-dependent control logic:
    - alpha scheduling based on feasibility_rate
    - optimizer switching on stagnation
    - depth escalation
    - shot allocation
    """
    if base_policy is None:
        base_policy = problem
        problem = None

    if not history:
        return base_policy.copy()

    last = history[-1]
    prev = history[-2] if len(history) >= 2 else None
    learning_delta = prev.learning_score - last.learning_score if prev is not None else 0.0
    feasibility_delta = (
        last.raw_feasibility_rate - prev.raw_feasibility_rate
        if prev is not None
        else 0.0
    )
    ar_delta = last.raw_ar - prev.raw_ar if prev is not None else 0.0
    stalled = last.convergence_stagnation > 0.80
    plateaued = prev is not None and feasibility_delta < 0.02 and learning_delta <= 0.01
    improving = prev is not None and feasibility_delta > 0.05 and learning_delta > 0.01

    current_family = str(
        (last.policy_used or {}).get(
            "solver_family",
            base_policy.get("solver_family", "qaoa"),
        )
    ).lower()
    target_family = _choose_followup_family(
        attempt,
        history,
        problem,
        current_family,
        improving,
        plateaued,
        stalled,
    )
    switched_family = target_family != current_family

    family_base = build_base_policy(problem, target_family)
    policy = family_base.copy()
    if not switched_family:
        for key, value in (last.policy_used or {}).items():
            if key not in {"solver_family", "qrac_type", "pce_k"}:
                policy[key] = value
    else:
        for key in (
            "estimator_shots",
            "sampler_shots",
            "optimizer_method",
            "optimizer_maxiter",
            "optimizer_tol",
            "learning_rate",
            "seed",
        ):
            if key in (last.policy_used or {}):
                policy[key] = last.policy_used[key]

    family = target_family
    policy["solver_family"] = family
    policy["measurement_mode"] = str(
        policy.get("measurement_mode", family_base.get("measurement_mode", "expectation"))
    ).lower()
    if family == "qrao" and "qrac_type" in family_base:
        policy["qrac_type"] = family_base["qrac_type"]
    if family == "pce" and "pce_k" in family_base:
        policy["pce_k"] = family_base["pce_k"]
    base_estimator = int(family_base.get("estimator_shots", DEFAULT_ESTIMATOR_SHOTS))
    base_sampler = int(family_base.get("sampler_shots", DEFAULT_SAMPLER_SHOTS))

    if family == "qaoa":
        depth_key = "reps"
        max_depth = 4
    elif family in {"vqe", "qrao"}:
        depth_key = "vqe_reps"
        max_depth = 3
    else:
        depth_key = "pce_depth"
        max_depth = 4

    if switched_family:
        if family == "qaoa":
            if not last.raw_feasible:
                if last.raw_feasibility_rate < 0.08:
                    policy["variant"] = "standard"
                    policy["measurement_mode"] = "cvar"
                    policy["cvar_alpha"] = 0.50
                elif plateaued or stalled or attempt >= 2:
                    policy["variant"] = "warmstart"
                    policy["measurement_mode"] = "cvar"
                    policy["ws_source"] = "relaxation"
                    policy["ws_epsilon"] = 0.20
            elif last.raw_ar < 0.90:
                if plateaued or stalled or ar_delta < 0.02:
                    policy["variant"] = "multiangle"
                    policy["measurement_mode"] = "cvar"
                    policy["ma_tying"] = "none"
                else:
                    policy["variant"] = "standard"
                    policy["measurement_mode"] = "cvar"
                    policy["cvar_alpha"] = 0.20
        elif family == "vqe":
            if not last.raw_feasible and last.raw_feasibility_rate < 0.10:
                policy["variant"] = "standard"
                policy["measurement_mode"] = "cvar"
                policy["cvar_alpha"] = 0.50
            elif last.raw_feasible and last.raw_feasibility_rate > 0.40 and last.raw_ar < 0.90:
                policy["variant"] = "standard"
                policy["measurement_mode"] = "cvar"
                policy["cvar_alpha"] = 0.20
        elif family == "qrao":
            if not last.raw_feasible or plateaued or stalled:
                policy["measurement_mode"] = "cvar"
                policy["cvar_alpha"] = 0.35 if last.raw_feasibility_rate < 0.15 else 0.25
            if last.raw_feasibility_rate < 0.15:
                policy["qrac_type"] = 3
            if last.raw_feasible and last.raw_ar < 0.75:
                policy["rounding"] = "magic"
        elif family == "pce":
            if not last.raw_feasible or plateaued or stalled:
                policy["measurement_mode"] = "cvar"
                policy["cvar_alpha"] = 0.35 if last.raw_feasibility_rate < 0.10 else 0.25
            if stalled or plateaued:
                policy["pce_depth"] = min(
                    max_depth,
                    int(policy.get("pce_depth", family_base.get("pce_depth", 2))) + 1,
                )
            if not last.raw_feasible and last.raw_feasibility_rate < 0.05:
                policy["pce_beta"] = min(0.80, float(policy.get("pce_beta", 0.5)) + 0.10)

    if not last.raw_feasible:
        if last.raw_feasibility_rate < 0.10:
            if family in {"qaoa", "vqe"}:
                if family == "qaoa":
                    policy["variant"] = "standard"
                policy["measurement_mode"] = "cvar"
                if plateaued:
                    policy["cvar_alpha"] = max(0.15, float(policy.get("cvar_alpha", 0.25)) * 0.6)
                else:
                    policy["cvar_alpha"] = 0.50
            elif family in {"qrao", "pce"}:
                policy["measurement_mode"] = "cvar"
                policy["cvar_alpha"] = 0.35 if last.raw_feasibility_rate < 0.05 else 0.25
            if family == "qaoa":
                if plateaued and attempt >= 1:
                    policy["variant"] = "warmstart"
                    policy["measurement_mode"] = "cvar"
                    policy["ws_source"] = "relaxation"
                    policy["ws_epsilon"] = 0.20
                elif improving and last.raw_feasibility_rate >= 0.10:
                    policy["variant"] = "standard"
                    policy["measurement_mode"] = "cvar"
                    policy["cvar_alpha"] = 0.35
            policy["estimator_shots"] = min(
                8192,
                max(base_estimator, int(policy.get("estimator_shots", base_estimator))) * 2,
            )
            policy["sampler_shots"] = min(
                8192,
                max(base_sampler, int(policy.get("sampler_shots", base_sampler))) * 2,
            )
        elif improving:
            policy["estimator_shots"] = max(
                base_estimator,
                int(policy.get("estimator_shots", base_estimator)),
            )
            policy["sampler_shots"] = min(
                8192,
                max(base_sampler, int(policy.get("sampler_shots", base_sampler))) + base_sampler,
            )

        if plateaued:
            policy[depth_key] = min(
                max_depth,
                int(policy.get(depth_key, family_base.get(depth_key, 1))) + 1,
            )

    if last.raw_feasible and last.raw_ar < 0.90:
        if family == "qaoa":
            if prev is not None and last.raw_feasibility_rate > 0.75 and last.raw_ar < 0.90:
                if plateaued or ar_delta < 0.02:
                    policy["variant"] = "multiangle"
                    policy["measurement_mode"] = "cvar"
                    policy["ma_tying"] = "none"
                else:
                    policy["variant"] = "standard"
                    policy["measurement_mode"] = "cvar"
                    policy["cvar_alpha"] = min(
                        float(policy.get("cvar_alpha", family_base.get("cvar_alpha", 0.25))),
                        0.20,
                    )
            elif last.raw_feasibility_rate > 0.35:
                policy["variant"] = "standard"
                policy["measurement_mode"] = "expectation"
        elif family == "vqe":
            if (
                prev is not None
                and last.raw_feasibility_rate > 0.90
                and ar_delta > 0.02
                and learning_delta > 0.01
            ):
                policy["variant"] = "standard"
                policy["measurement_mode"] = "cvar"
                policy["cvar_alpha"] = min(
                    float(policy.get("cvar_alpha", family_base.get("cvar_alpha", 0.25))),
                    0.20,
                )
                policy["estimator_shots"] = min(
                    4096,
                    max(base_estimator, int(policy.get("estimator_shots", base_estimator)))
                    + max(base_estimator // 2, 256),
                )
                policy["sampler_shots"] = min(
                    8192,
                    max(base_sampler, int(policy.get("sampler_shots", base_sampler)))
                    + base_sampler,
                )
            elif last.raw_feasibility_rate > 0.35:
                policy["variant"] = "standard"
                policy["measurement_mode"] = "expectation"
        elif family in {"qrao", "pce"} and (plateaued or stalled or last.raw_ar < 0.75):
            policy["measurement_mode"] = "cvar"
            policy["cvar_alpha"] = min(
                float(policy.get("cvar_alpha", family_base.get("cvar_alpha", 0.25))),
                0.25,
            )
        if plateaued or ar_delta < 0.02:
            policy[depth_key] = min(
                max_depth,
                int(policy.get(depth_key, family_base.get(depth_key, 1))) + 1,
            )
        if family == "qrao" and last.raw_ar < 0.70:
            policy["rounding"] = "magic"

    current_optimizer = str(
        policy.get(
            "optimizer_method",
            (last.policy_used or {}).get("optimizer_method", family_base.get("optimizer_method", "COBYLA")),
        )
    ).upper()
    policy["optimizer_method"] = _choose_followup_optimizer(
        current_optimizer,
        last,
        improving,
        plateaued,
        stalled,
    )
    if policy["optimizer_method"] == "SPSA":
        policy["learning_rate"] = 0.03 if last.raw_feasibility_rate < 0.10 else 0.05
        policy["optimizer_maxiter"] = max(int(policy.get("optimizer_maxiter", 100)), 100 if stalled else 80)
    elif policy["optimizer_method"] == "ADAM":
        policy["learning_rate"] = 0.02 if last.raw_feasibility_rate < 0.10 else 0.04
        policy["optimizer_maxiter"] = max(int(policy.get("optimizer_maxiter", 100)), 120)
    else:
        if plateaued:
            policy["optimizer_tol"] = min(float(policy.get("optimizer_tol", 1e-3)), 5e-4)

    ansatz_switched = False
    if family in {"vqe", "qrao", "pce"}:
        current_ansatz = str(policy.get("ansatz_type", family_base.get("ansatz_type", "real_amplitudes"))).lower()
        chosen_ansatz = _choose_followup_ansatz(
            family,
            current_ansatz,
            history,
            last,
            improving,
            plateaued,
            stalled,
        )
        if chosen_ansatz != current_ansatz:
            policy["ansatz_type"] = chosen_ansatz
            if family in {"vqe", "qrao"}:
                policy["vqe_reps"] = int(family_base.get("vqe_reps", 1))
            elif family == "pce":
                policy["pce_depth"] = int(family_base.get("pce_depth", 2))
            policy["estimator_shots"] = base_estimator
            policy["sampler_shots"] = base_sampler
            ansatz_switched = True

        if chosen_ansatz not in {"brickwork", "pauli_two_design"}:
            entanglement = str(policy.get("entanglement", family_base.get("entanglement", "linear"))).lower()
            if (
                (last.raw_feasible and last.raw_feasibility_rate > 0.60 and last.raw_ar < 0.90)
                or stalled
                or plateaued
            ):
                policy["entanglement"] = "circular"
            elif not last.raw_feasible and last.raw_feasibility_rate < 0.15:
                policy["entanglement"] = "linear"
            else:
                policy["entanglement"] = entanglement

    if last.raw_feasibility_rate > 0.70 and last.raw_feasible and not ansatz_switched and not switched_family:
        policy["estimator_shots"] = max(
            base_estimator,
            int(policy.get("estimator_shots", base_estimator)) // 2,
        )
        policy["sampler_shots"] = max(
            base_sampler,
            int(policy.get("sampler_shots", base_sampler)) // 2,
        )

    return policy


# ═══════════════════════════════════════════════════════════════════
# INFRASTRUCTURE — fixed; do not edit below this line
# ═══════════════════════════════════════════════════════════════════


def _normalize_convergence(history: list[float]) -> tuple[float, float, float]:
    """Compute normalized convergence statistics from optimizer trace.

    Returns:
        (improvement, stagnation, final_cost)

    - improvement: (start - end) / |start|. > 0 means cost decreased.
    - stagnation: fraction of final 25% of trace with < 1% relative change.
      > 0.8 reliably means the optimizer was stuck.
    - final_cost: terminal cost value.
    """
    if not history:
        return 0.0, 1.0, 0.0
    if len(history) < 2:
        return 0.0, 0.0, history[-1]

    start_cost = history[0]
    end_cost = history[-1]
    final_cost = end_cost

    # Improvement
    if abs(start_cost) > 1e-10:
        improvement = (start_cost - end_cost) / abs(start_cost)
    else:
        improvement = 0.0

    # Stagnation: fraction of final 25% with < 1% relative change
    tail_start = max(1, int(0.75 * len(history)))
    tail = history[tail_start:]
    if len(tail) < 2:
        stagnation = 0.0
    else:
        stagnant = 0
        for i in range(1, len(tail)):
            if abs(tail[i - 1]) > 1e-10:
                rel_change = abs(tail[i] - tail[i - 1]) / abs(tail[i - 1])
            else:
                rel_change = abs(tail[i] - tail[i - 1])
            if rel_change < 0.01:
                stagnant += 1
        stagnation = stagnant / max(len(tail) - 1, 1)

    return float(improvement), float(stagnation), float(final_cost)


def _compute_learning_score(
    optimality_gap: float,
    is_feasible: bool,
    feasibility_rate: float,
    best_feasible_ar: float,
    result=None,
) -> float:
    """Compute feasibility-shaped learning score. Lower is better.

    When feasible: learning_score = optimality_gap.
    When infeasible: start from 1.0 and add a shaped miss term, plus optional
    circuit-cost penalties when resource data is available.
    """
    if is_feasible:
        return optimality_gap
    depth_penalty, two_qubit_penalty = _resource_penalties(result)
    return (
        1.0
        + 0.4 * (1.0 - feasibility_rate)
        + 0.1 * (1.0 - best_feasible_ar)
        + depth_penalty
        + two_qubit_penalty
    )


def _get_solver_fn(family: str):
    """Get the appropriate knapsack solver function."""
    if family == "qaoa":
        from autoqresearch.solvers.qubo_primitives import solve_qubo_qaoa
        return solve_qubo_qaoa
    if family == "vqe":
        from autoqresearch.solvers.qubo_primitives import solve_qubo_vqe
        return solve_qubo_vqe
    if family == "qrao":
        from autoqresearch.solvers.qrao_solver import QRAOSolver

        solver = QRAOSolver()

        def _solve(problem, policy, backend):
            shots = int(
                policy.get(
                    "sampler_shots",
                    policy.get("estimator_shots", DEFAULT_SAMPLER_SHOTS),
                )
            )
            return solver.solve(problem, policy, backend, shots=shots)

        return _solve
    if family == "pce":
        from autoqresearch.solvers.pce_solver import PCESolver

        solver = PCESolver()

        def _solve(problem, policy, backend):
            shots = int(
                policy.get(
                    "sampler_shots",
                    policy.get("estimator_shots", DEFAULT_SAMPLER_SHOTS),
                )
            )
            return solver.solve(problem, policy, backend, shots=shots)

        return _solve
    raise ValueError(f"Unknown solver family for knapsack: {family}")


def _make_backend(policy: dict, mode: str):
    """Create execution context from policy."""
    from autoqresearch.backends.factory import BackendConfig, create_execution_context

    return create_execution_context(
        BackendConfig(
            mode=mode,
            shots=int(policy.get("estimator_shots", DEFAULT_ESTIMATOR_SHOTS)),
            sampler_shots=int(policy.get("sampler_shots", DEFAULT_SAMPLER_SHOTS)),
        )
    )


def _parse_problem_spec(spec: str) -> tuple[str, int, int]:
    """Parse problem specification string (e.g., knapsack_12_s3)."""
    parts = spec.split("_")
    problem_type = parts[0]
    size = int(parts[1])
    seed = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("s") else 0
    return problem_type, size, seed


def _compute_optimality_gap(ar: float, is_feasible: bool) -> float:
    """Optimality gap: (optimal - found) / optimal = 1 - AR.

    Lower is better. 0.0 = optimal solution found. 1.0 = nothing useful.
    Infeasible solutions score 1.0 regardless of AR.
    """
    if not is_feasible:
        return 1.0
    return 1.0 - float(ar)


def _resource_penalties(result) -> tuple[float, float]:
    """Return the depth and two-qubit penalties used by the knapsack metrics."""

    if result is None:
        return 0.0, 0.0

    depth = int(getattr(result, "circuit_depth", 0))
    two_qubit = int(
        getattr(result, "two_qubit_gate_count", 0)
        or getattr(result, "cnot_count", 0)
    )
    return (
        DEPTH_PENALTY_WEIGHT * min(1.0, depth / DEPTH_PENALTY_SCALE),
        TWO_QUBIT_PENALTY_WEIGHT * min(1.0, two_qubit / TWO_QUBIT_PENALTY_SCALE),
    )


POLICY_SNAPSHOT_KEYS = (
    "solver_family",
    "variant",
    "reps",
    "vqe_reps",
    "pce_depth",
    "ansatz_type",
    "entanglement",
    "measurement_mode",
    "cvar_alpha",
    "optimizer_method",
    "optimizer_maxiter",
    "optimizer_tol",
    "learning_rate",
    "estimator_shots",
    "sampler_shots",
    "rounding",
    "qrac_type",
    "pce_k",
    "pce_alpha",
    "pce_beta",
    "pce_local_search",
    "ws_source",
    "ws_epsilon",
    "ma_tying",
    "initialization",
    "seed",
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def snapshot_policy(policy: dict | None) -> dict:
    """Return a JSON-safe normalized policy snapshot."""
    snapshot: dict[str, object] = {}
    for key in POLICY_SNAPSHOT_KEYS:
        if policy is None or key not in policy:
            continue
        value = policy.get(key)
        if value is None:
            continue
        snapshot[key] = _json_safe(value)
    return snapshot


def _policies_match(left: dict | None, right: dict | None) -> bool:
    return snapshot_policy(left) == snapshot_policy(right)


def _load_policy_override(
    policy_file: Path | None = None,
    policy_json: str | None = None,
) -> dict | None:
    if policy_file is not None and policy_json is not None:
        raise ValueError("Specify at most one of --policy-file or --policy-json.")

    payload = None
    if policy_file is not None:
        payload = policy_file.read_text()
    elif policy_json is not None:
        payload = policy_json

    if payload is None:
        return None

    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("Policy override must decode to a JSON object.")
    return {str(key): value for key, value in loaded.items()}


def _merge_policy_override(base_policy: dict, override: dict | None) -> dict:
    merged = base_policy.copy()
    if override:
        for key, value in override.items():
            merged[str(key)] = value
    return merged


def _description_with_run_tag(run_tag: str, family: str, policy: dict) -> str:
    description = _build_description(family, policy)
    if not run_tag:
        return description
    return f"{run_tag} {description}"


def _write_json(path: Path | None, payload: dict | list) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path | None, records: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _attempt_shot_accounting(policy: dict, result) -> dict[str, int]:
    estimator_shots = int(policy.get("estimator_shots", DEFAULT_ESTIMATOR_SHOTS))
    sampler_shots = int(policy.get("sampler_shots", DEFAULT_SAMPLER_SHOTS))
    optimizer_iterations = int(getattr(result, "optimizer_iterations", 0) or 0)
    optimization_shots = estimator_shots * max(optimizer_iterations, 0)
    sampling_shots = sampler_shots
    return {
        "estimator_shots": estimator_shots,
        "sampler_shots": sampler_shots,
        "optimizer_iterations": optimizer_iterations,
        "optimization_shots": optimization_shots,
        "sampling_shots": sampling_shots,
        "total_attempt_shots": optimization_shots + sampling_shots,
    }


def _first_matching_attempt(
    attempt_records: list[dict],
    predicate,
) -> tuple[int | None, int | None]:
    cumulative_shots = 0
    for record in attempt_records:
        cumulative_shots += int(record.get("total_attempt_shots", 0) or 0)
        if predicate(record):
            return int(record["attempt"]), cumulative_shots
    return None, None


def _first_changed_policy(
    attempt_records: list[dict],
    base_policy: dict,
) -> tuple[dict | None, int | None]:
    base_snapshot = snapshot_policy(base_policy)
    for record in attempt_records:
        policy = record.get("policy_used")
        if not _policies_match(policy, base_snapshot):
            return snapshot_policy(policy), int(record["attempt"])
    return None, None


def _results_header() -> list[str]:
    return [
        "experiment_id",
        "timestamp",
        "problem",
        "solver",
        "status",
        "description",
        "optimality_gap",
        "approx_ratio",
        "feasible",
        "feasibility_rate",
        "depth",
        "cnots",
        "two_qubit_gates",
        "total_gates",
        "qubits",
        "num_parameters",
        "opt_iters",
        "wall_time_s",
    ]


def _ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(_results_header())


def _reset_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(_results_header())


def _read_existing_metrics(path: Path) -> tuple[int, float | None]:
    if not path.exists() or path.stat().st_size == 0:
        return 0, None

    max_id = 0
    best_metric = None
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                max_id = max(max_id, int(str(row.get("experiment_id", "0")).strip() or 0))
            except ValueError:
                pass
            try:
                metric = float(str(row.get("optimality_gap", "")).strip())
            except ValueError:
                continue
            if best_metric is None or metric < best_metric:
                best_metric = metric
    return max_id, best_metric


def _build_description(family: str, policy: dict) -> str:
    family = family.lower()
    parts = [family]

    if family == "qaoa":
        parts.append(f"variant={policy.get('variant', 'standard')}")
        parts.append(f"reps={int(policy.get('reps', 1))}")
        variant = str(policy.get("variant", "standard")).lower()
        measurement_mode = str(
            policy.get(
                "measurement_mode",
                "cvar" if variant == "cvar" else "expectation",
            )
        ).lower()
        if measurement_mode == "cvar":
            parts.append(f"cvar_alpha={float(policy.get('cvar_alpha', 0.25)):.2f}")
        if variant == "warmstart":
            parts.append(f"ws_source={policy.get('ws_source', 'relaxation')}")
        elif variant == "multiangle":
            parts.append(f"ma_tying={policy.get('ma_tying', 'none')}")
    elif family == "vqe":
        parts.append(f"variant={policy.get('variant', 'standard')}")
        parts.append(f"ansatz_type={policy.get('ansatz_type', 'efficient_su2')}")
        parts.append(f"vqe_reps={int(policy.get('vqe_reps', 1))}")
        if str(policy.get("measurement_mode", "expectation")).lower() == "cvar":
            parts.append(f"cvar_alpha={float(policy.get('cvar_alpha', 0.25)):.2f}")
    elif family == "qrao":
        parts.append(f"qrac_type={int(policy.get('qrac_type', 2))}")
        parts.append(f"rounding={policy.get('rounding', 'semideterministic')}")
        parts.append(f"ansatz_type={policy.get('ansatz_type', 'real_amplitudes')}")
        parts.append(f"vqe_reps={int(policy.get('vqe_reps', 1))}")
        if str(policy.get("measurement_mode", "expectation")).lower() == "cvar":
            parts.append(f"cvar_alpha={float(policy.get('cvar_alpha', 0.25)):.2f}")
    elif family == "pce":
        parts.append(f"pce_k={int(policy.get('pce_k', 2))}")
        parts.append(f"pce_depth={int(policy.get('pce_depth', 1))}")
        parts.append(f"ansatz_type={policy.get('ansatz_type', 'brickwork')}")
        if str(policy.get("measurement_mode", "expectation")).lower() == "cvar":
            parts.append(f"cvar_alpha={float(policy.get('cvar_alpha', 0.25)):.2f}")

    return " ".join(str(part) for part in parts)


def _append_results_row(path: Path, row: list[object]) -> tuple[int, str]:
    _ensure_results_file(path)
    last_id, best_metric = _read_existing_metrics(path)

    metric = row[_results_header().index("optimality_gap")]
    try:
        metric_value = float(metric)
    except (TypeError, ValueError):
        metric_value = None

    if metric_value is None:
        status = "crash"
    else:
        status = "keep" if best_metric is None or metric_value < best_metric else "discard"

    experiment_id = last_id + 1
    row[0] = experiment_id
    row[4] = status

    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)

    return experiment_id, status


def _update_progress_plot(results_path: Path, output_path: Path) -> None:
    from analysis import load_results, make_progress_plot

    data = load_results(results_path, metric="optimality_gap")
    make_progress_plot(
        data,
        output_path=output_path,
        title="Knapsack Experiment Progress (optimality gap, lower is better)",
    )


def run_experiment(
    problem_spec: str,
    backend_mode: str = "ideal_mps",
    solver_family: str | None = None,
    max_attempts: int = 5,
    timeout: int = 600,
    results_file: Path = DEFAULT_RESULTS_PATH,
    plot_output: Path = DEFAULT_PROGRESS_PATH,
    no_results_log: bool = False,
    no_progress_plot: bool = False,
    policy_file: Path | None = None,
    policy_json: str | None = None,
    run_tag: str = "interactive",
    summary_json: Path | None = None,
    attempts_jsonl: Path | None = None,
    winning_policy_json: Path | None = None,
) -> dict:
    problem_type, size, seed = _parse_problem_spec(problem_spec)
    from autoqresearch.problems.registry import get_single_instance
    from autoqresearch.solvers.qubo_primitives import (
        check_qubo_feasibility,
        compute_best_feasible_ar,
        compute_feasibility_rate,
        fixed_repair,
        qubo_objective_value,
    )

    if not no_results_log:
        _reset_results_file(results_file)

    policy_override = _load_policy_override(policy_file, policy_json)
    problem = get_single_instance(problem_type, size, seed)
    initial_family = (
        solver_family
        or (str(policy_override.get("solver_family")) if policy_override and policy_override.get("solver_family") else None)
        or choose_solver_family(problem)
    )
    base_policy = build_base_policy(problem, initial_family)
    base_policy = _merge_policy_override(base_policy, policy_override)
    base_policy["solver_family"] = initial_family
    policy_mode = "static" if policy_override is not None else "adaptive"
    policy_source = "policy_file" if policy_file is not None else "policy_json" if policy_json is not None else "default"

    print(
        f"Problem: {problem.name} (n_items={problem.metadata.get('num_items', '?')}, "
        f"n_qubo={problem.num_variables}, optimal={problem.optimal_value:.2f})"
    )
    print(
        f"\nRunning {policy_mode} loop (max_attempts={max_attempts}, "
        f"start_family={initial_family}, run_tag={run_tag})\n"
    )

    history: list[AttemptOutcome] = []
    attempt_records: list[dict] = []
    best_raw_result = None
    best_outcome = None
    best_gap = float("inf")
    best_feasible_ar_global = 0.0
    attempt = 0
    t_total = time.time()

    while should_continue(attempt, history, problem, max_attempts):
        if policy_mode == "static":
            policy = base_policy.copy()
        else:
            policy = adapt_policy(attempt, history, problem, base_policy)
        attempt_family = str(policy.get("solver_family", initial_family)).lower()
        policy["solver_family"] = attempt_family
        attempt_base = build_base_policy(problem, attempt_family)
        if attempt_family == "qrao" and "qrac_type" in attempt_base and "qrac_type" not in policy:
            policy["qrac_type"] = attempt_base["qrac_type"]
        if attempt_family == "pce" and "pce_k" in attempt_base and "pce_k" not in policy:
            policy["pce_k"] = attempt_base["pce_k"]
        solve_fn = _get_solver_fn(attempt_family)

        backend = _make_backend(policy, backend_mode)

        t0 = time.time()
        try:
            result = solve_fn(problem, policy, backend)
        except Exception as exc:
            print(f"  Attempt {attempt}: FAILED ({exc})")
            attempt_records.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error": str(exc),
                    "policy_used": snapshot_policy(policy),
                    "solver_family": attempt_family,
                }
            )
            attempt += 1
            continue
        elapsed = time.time() - t0

        feas_rate = compute_feasibility_rate(result.counts, problem)
        is_feasible = check_qubo_feasibility(result.best_bitstring, problem)
        found_value = qubo_objective_value(result.best_bitstring, problem)
        ar = found_value / problem.optimal_value if problem.optimal_value > 0 else 0.0
        ar = min(1.0, max(0.0, ar))
        gap = _compute_optimality_gap(ar, is_feasible)
        attempt_best_feas_ar = compute_best_feasible_ar(result.counts, problem)
        best_feasible_ar_global = max(best_feasible_ar_global, attempt_best_feas_ar)
        improvement, stagnation, final_cost = _normalize_convergence(
            result.convergence_history
        )
        learning = _compute_learning_score(
            gap, is_feasible, feas_rate, best_feasible_ar_global, result
        )

        outcome = AttemptOutcome(
            attempt=attempt,
            learning_score=learning,
            optimality_gap=gap,
            raw_feasible=is_feasible,
            raw_feasibility_rate=feas_rate,
            raw_ar=ar,
            convergence_improvement=improvement,
            convergence_stagnation=stagnation,
            final_cost=final_cost,
            policy_used=policy,
            wall_time=elapsed,
        )
        history.append(outcome)

        if gap < best_gap:
            best_gap = gap
            best_raw_result = result
            best_outcome = outcome

        attempt_stats = _attempt_shot_accounting(policy, result)
        attempt_records.append(
            {
                "attempt": attempt,
                "status": "completed",
                "solver_name": str(getattr(result, "solver_name", attempt_family) or attempt_family),
                "policy_used": snapshot_policy(policy),
                "solver_family": attempt_family,
                "learning_score": learning,
                "optimality_gap": gap,
                "raw_feasible": bool(is_feasible),
                "raw_feasibility_rate": feas_rate,
                "raw_ar": ar,
                "convergence_improvement": improvement,
                "convergence_stagnation": stagnation,
                "final_cost": final_cost,
                "best_feasible_ar_global": best_feasible_ar_global,
                "wall_time_s": elapsed,
                "circuit_depth": int(getattr(result, "circuit_depth", 0)),
                "cnot_count": int(getattr(result, "cnot_count", 0)),
                "two_qubit_gate_count": int(getattr(result, "two_qubit_gate_count", 0)),
                "total_gate_count": int(getattr(result, "total_gate_count", 0)),
                "num_qubits": int(getattr(result, "num_qubits", 0)),
                "num_parameters": int(getattr(result, "num_parameters", 0)),
                **attempt_stats,
            }
        )

        marker = "*" if gap == best_gap and gap < 1.0 else " "
        print(
            f"  Attempt {attempt}: "
            f"optimality_gap={gap:.4f} "
            f"learning={learning:.4f} "
            f"feas_rate={feas_rate:.3f} "
            f"AR={ar:.3f} "
            f"stagnation={stagnation:.2f} "
            f"time={elapsed:.1f}s {marker}"
        )

        attempt += 1
        if time.time() - t_total > timeout:
            print(f"\n  TIMEOUT after {time.time() - t_total:.1f}s")
            break

    total_wall_time = time.time() - t_total

    first_feasible_attempt, shots_to_first_feasible = _first_matching_attempt(
        attempt_records,
        lambda record: bool(record.get("raw_feasible")),
    )
    first_ar_ge_0_5_attempt, shots_to_first_ar_ge_0_5 = _first_matching_attempt(
        attempt_records,
        lambda record: float(record.get("raw_ar", 0.0) or 0.0) >= 0.5,
    )
    total_run_shots = int(
        sum(int(record.get("total_attempt_shots", 0) or 0) for record in attempt_records)
    )
    direct_stage2_policy, direct_stage2_attempt = _first_changed_policy(
        attempt_records,
        base_policy,
    )

    print("\n" + "=" * 60)
    print("Pre-repair results (primary — drives learning)")
    print("=" * 60)

    status = "completed"
    repaired_score = None
    repaired_ar = None
    repaired_feasible = None
    repair_changed = None
    best_ar = None
    best_feasible = None
    best_feas_rate = None
    best_learning = min((h.learning_score for h in history), default=1.0)

    if best_raw_result is not None and best_outcome is not None:
        best_ar = qubo_objective_value(
            best_raw_result.best_bitstring, problem
        ) / max(problem.optimal_value, 1e-10)
        best_feasible = check_qubo_feasibility(
            best_raw_result.best_bitstring, problem
        )
        best_feas_rate = compute_feasibility_rate(best_raw_result.counts, problem)

        print(f"optimality_gap: {best_gap:.6f}")
        print(f"raw_ar: {best_ar:.4f}")
        print(f"raw_feasible: {int(best_feasible)}")
        print(f"raw_feasibility_rate: {best_feas_rate:.4f}")
        print(f"learning_score: {best_learning:.6f}")

        repaired_x, repair_changed = fixed_repair(
            best_raw_result.best_bitstring, problem
        )
        values = problem.metadata["values"]
        weights = problem.metadata["weights"]
        capacity = problem.metadata["capacity"]
        n_items = problem.metadata["num_items"]

        repaired_value = float(np.dot(values, repaired_x[:n_items]))
        repaired_feasible = float(np.dot(weights, repaired_x[:n_items])) <= capacity
        repaired_ar = repaired_value / max(problem.optimal_value, 1e-10)
        repaired_score = _compute_optimality_gap(
            repaired_ar,
            repaired_feasible,
        )

        print(f"\n{'=' * 60}")
        print("Post-repair results (secondary — for completeness)")
        print("=" * 60)
        print(f"repaired_optimality_gap: {repaired_score:.6f}")
        print(f"repaired_ar: {repaired_ar:.4f}")
        print(f"repaired_feasible: {int(repaired_feasible)}")
        print(f"repair_changed: {int(repair_changed)}")
    else:
        status = "crash"
        if not no_results_log:
            row = [
                0,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                problem.name,
                initial_family,
                "",
                _description_with_run_tag(run_tag, initial_family, base_policy),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                len(history),
                f"{total_wall_time:.1f}",
            ]
            experiment_id, results_status = _append_results_row(results_file, row)
            print(f"results_file: {results_file}")
            print(f"results_experiment_id: {experiment_id}")
            print(f"results_status: {results_status}")
        if not no_progress_plot and results_file.exists():
            try:
                _update_progress_plot(results_file, plot_output)
                print(f"progress_plot: {plot_output}")
            except Exception as exc:
                print(f"progress_plot: unavailable ({exc})")
        print("No successful attempts.")

    for record in attempt_records:
        record["is_best_attempt"] = bool(
            best_outcome is not None and int(record.get("attempt", -1)) == best_outcome.attempt
        )
        record["run_tag"] = run_tag
        record["problem"] = problem.name
        record["solver_family"] = str(
            record.get("solver_family")
            or (record.get("policy_used") or {}).get("solver_family")
            or initial_family
        ).lower()

    winning_policy = snapshot_policy(best_outcome.policy_used) if best_outcome is not None else None
    winning_family = (
        str(winning_policy.get("solver_family", initial_family)).lower()
        if winning_policy is not None
        else initial_family
    )
    winning_attempt_record = next(
        (
            record
            for record in attempt_records
            if best_outcome is not None and int(record.get("attempt", -1)) == best_outcome.attempt
        ),
        None,
    )
    summary = {
        "status": status,
        "run_tag": run_tag,
        "policy_mode": policy_mode,
        "policy_source": policy_source,
        "problem_spec": problem_spec,
        "problem": problem.name,
        "problem_type": problem_type,
        "size": size,
        "seed": seed,
        "backend": backend_mode,
        "solver_family": initial_family,
        "initial_solver_family": initial_family,
        "winning_solver_family": winning_family,
        "base_policy": snapshot_policy(base_policy),
        "winning_policy": winning_policy,
        "direct_stage2_policy": direct_stage2_policy,
        "direct_stage2_attempt": direct_stage2_attempt,
        "best_attempt_index": best_outcome.attempt if best_outcome is not None else None,
        "total_attempts": len(history),
        "total_run_shots": total_run_shots,
        "total_wall_time_s": total_wall_time,
        "first_feasible_attempt": first_feasible_attempt,
        "shots_to_first_feasible": shots_to_first_feasible,
        "first_ar_ge_0_5_attempt": first_ar_ge_0_5_attempt,
        "shots_to_ar_ge_0_5": shots_to_first_ar_ge_0_5,
        "best_optimality_gap": best_gap if best_outcome is not None else None,
        "optimality_gap": best_gap if best_outcome is not None else None,
        "raw_ar": best_ar,
        "raw_feasible": bool(best_feasible) if best_feasible is not None else None,
        "raw_feasibility_rate": best_feas_rate,
        "learning_score": best_learning if history else None,
        "winning_solver_name": winning_attempt_record.get("solver_name") if winning_attempt_record else None,
        "winning_optimizer_iterations": (
            int(winning_attempt_record.get("optimizer_iterations", 0))
            if winning_attempt_record
            else None
        ),
        "winning_attempt_shots": (
            int(winning_attempt_record.get("total_attempt_shots", 0))
            if winning_attempt_record
            else None
        ),
        "repaired_optimality_gap": repaired_score,
        "repaired_ar": repaired_ar,
        "repaired_feasible": bool(repaired_feasible) if repaired_feasible is not None else None,
        "repair_changed": bool(repair_changed) if repair_changed is not None else None,
        "attempts": attempt_records,
    }

    if winning_policy_json is not None and winning_policy is not None:
        summary["winning_policy_path"] = str(winning_policy_json)
        _write_json(winning_policy_json, winning_policy)

    if attempts_jsonl is not None:
        summary["attempts_jsonl_path"] = str(attempts_jsonl)
        _write_jsonl(attempts_jsonl, attempt_records)

    if summary_json is not None:
        summary["summary_json_path"] = str(summary_json)
        _write_json(summary_json, summary)

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"run_tag: {run_tag}")
    print(f"policy_mode: {policy_mode}")
    print(f"total_attempts: {len(history)}")
    print(f"total_time: {total_wall_time:.1f}")
    print(f"total_run_shots: {total_run_shots}")
    print(f"first_feasible_attempt: {first_feasible_attempt}")
    print(f"first_ar_ge_0_5_attempt: {first_ar_ge_0_5_attempt}")
    print(f"best_attempt_index: {summary['best_attempt_index']}")
    print(
        "best_optimality_gap: "
        f"{best_gap:.6f}" if best_outcome is not None else "best_optimality_gap: "
    )
    print(f"optimal_value: {problem.optimal_value:.4f}")
    print(f"solver_family: {winning_family}")
    print(f"initial_solver_family: {initial_family}")
    print(f"winning_solver_family: {winning_family}")
    print(f"backend: {backend_mode}")
    print(f"problem: {problem.name}")

    if not no_results_log and best_raw_result is not None and best_outcome is not None:
        policy_for_log = best_outcome.policy_used
        solver_name = str(getattr(best_raw_result, "solver_name", winning_family) or winning_family)
        row = [
            0,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            problem.name,
            solver_name,
            "",
            _description_with_run_tag(run_tag, winning_family, policy_for_log),
            f"{best_gap:.6f}",
            f"{best_ar:.4f}",
            int(best_feasible),
            f"{best_feas_rate:.4f}",
            int(getattr(best_raw_result, "circuit_depth", 0)),
            int(getattr(best_raw_result, "cnot_count", 0)),
            int(getattr(best_raw_result, "two_qubit_gate_count", 0)),
            int(getattr(best_raw_result, "total_gate_count", 0)),
            int(getattr(best_raw_result, "num_qubits", 0)),
            int(getattr(best_raw_result, "num_parameters", 0)),
            int(getattr(best_raw_result, "optimizer_iterations", 0)),
            f"{total_wall_time:.1f}",
        ]
        experiment_id, results_status = _append_results_row(results_file, row)
        print(f"results_file: {results_file}")
        print(f"results_experiment_id: {experiment_id}")
        print(f"results_status: {results_status}")

    if not no_progress_plot and not no_results_log:
        try:
            _update_progress_plot(results_file, plot_output)
            print(f"progress_plot: {plot_output}")
        except Exception as exc:
            print(f"progress_plot: unavailable ({exc})")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive knapsack experiment")
    parser.add_argument(
        "--problem", type=str, default=DEFAULT_PROBLEM_SPEC,
        help="Problem spec (e.g., knapsack_12, knapsack_12_s3)",
    )
    parser.add_argument("--backend", type=str, default="ideal_mps")
    parser.add_argument(
        "--solver-family",
        type=str,
        choices=("qaoa", "vqe", "qrao", "pce"),
        default=None,
        help="Force a solver family instead of using choose_solver_family(problem).",
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="TSV ledger rewritten at the start of each logged run.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PROGRESS_PATH,
        help="Progress plot regenerated after appending to the results file.",
    )
    parser.add_argument(
        "--no-results-log",
        action="store_true",
        help="Do not append this run to the TSV ledger.",
    )
    parser.add_argument(
        "--no-progress-plot",
        action="store_true",
        help="Do not regenerate the progress plot after the run.",
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help="JSON file containing a fixed policy to execute from attempt 0.",
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="JSON object containing a fixed policy to execute from attempt 0.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="interactive",
        help="Short label recorded in machine-readable outputs and results descriptions.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON file for the normalized run summary.",
    )
    parser.add_argument(
        "--attempts-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file with one machine-readable record per attempt.",
    )
    parser.add_argument(
        "--winning-policy-json",
        type=Path,
        default=None,
        help="Optional JSON file for the winning final policy snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_experiment(
        problem_spec=args.problem,
        backend_mode=args.backend,
        solver_family=args.solver_family,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        results_file=args.results_file,
        plot_output=args.plot_output,
        no_results_log=args.no_results_log,
        no_progress_plot=args.no_progress_plot,
        policy_file=args.policy_file,
        policy_json=args.policy_json,
        run_tag=args.run_tag,
        summary_json=args.summary_json,
        attempts_jsonl=args.attempts_jsonl,
        winning_policy_json=args.winning_policy_json,
    )
    return 0 if summary.get("status") != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
