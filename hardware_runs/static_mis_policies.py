#!/usr/bin/env python3
"""Static per-instance MIS hardware replay policies."""

from __future__ import annotations

from typing import Any


COMMON_DEFAULTS: dict[str, Any] = {
    "optimizer_method": "COBYLA",
    "optimizer_maxiter": 150,
    "optimizer_tol": 1e-3,
    "learning_rate": 0.05,
    "entanglement": "linear",
    "estimator_shots": 1024,
    "sampler_shots": 1024,
    "seed": 17,
    "penalty": None,
    "pce_local_search": False,
    "final_local_search": False,
}


def _policy(**overrides: Any) -> dict[str, Any]:
    return {**COMMON_DEFAULTS, **overrides}


STATIC_RETAINED_POLICIES: dict[str, dict[str, Any]] = {
    "1tc.16": _policy(
        solver_family="qaoa",
        variant="warmstart",
        reps=1,
        ws_epsilon=0.25,
        ws_source="relaxation",
        measurement_mode="cvar",
        cvar_alpha=0.25,
    ),
    "1tc.32": _policy(
        solver_family="qrao",
        qrao_max_vars_per_qubit=3,
        qrac_type=3,
        rounding="magic",
        ansatz_type="real_amplitudes",
        vqe_reps=1,
        measurement_mode="expectation",
    ),
    "p1tc.48": _policy(
        solver_family="qrao",
        qrao_max_vars_per_qubit=2,
        qrac_type=2,
        rounding="semideterministic",
        ansatz_type="real_amplitudes",
        vqe_reps=1,
        measurement_mode="expectation",
    ),
    "1tc.64": _policy(
        solver_family="qrao",
        qrao_max_vars_per_qubit=2,
        qrac_type=2,
        rounding="semideterministic",
        ansatz_type="real_amplitudes",
        vqe_reps=1,
        measurement_mode="expectation",
    ),
}


STATIC_POLICY_NOTES: dict[str, str] = {
    "1tc.16": "Retained 16-node winner: QAOA warmstart CVaR.",
    "1tc.32": "Retained 32-node winner: QRAO 3:1 magic.",
    "p1tc.48": (
        "Pinned to QRAO 2:1 semideterministic because retained evidence shows "
        "the 48-node sparse win came from the fallback branch, while 3:1 was a dead branch."
    ),
    "1tc.64": (
        "Pinned to QRAO 2:1 semideterministic as the closest static replay of the retained "
        "large-instance sparse path; this is an inference from the preserved 48/64 evidence."
    ),
}


def get_static_policy_for_instance(stem: str) -> dict[str, Any] | None:
    policy = STATIC_RETAINED_POLICIES.get(str(stem))
    if policy is None:
        return None
    return dict(policy)


def get_static_policy_note(stem: str) -> str | None:
    note = STATIC_POLICY_NOTES.get(str(stem))
    return None if note is None else str(note)
