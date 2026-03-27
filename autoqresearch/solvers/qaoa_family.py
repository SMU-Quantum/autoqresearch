"""QAOA-family solver wrappers for the MaxCut-first implementation."""

from __future__ import annotations

import time

from .base import BaseSolver, SolverResult
from .maxcut_primitives import normalize_measurement_policy, solve_qaoa_variant


DEFAULT_QAOA_POLICY = {
    "variant": "standard",
    "p": 1,
    "reps": 1,
    "optimizer": "COBYLA",
    "optimizer_method": "COBYLA",
    "optimizer_maxiter": 200,
    "optimizer_tol": 1e-3,
    "num_restarts": 1,
    "initialization": "random",
    "shots": 1000,
    "estimator_shots": 1000,
    "sampler_shots": 1000,
    "mixer": "x",
    "circuit_type": "qaoa_ansatz",
    "decompose_reps": 2,
    "measurement_mode": "expectation",
    "cvar_alpha": 0.25,
    "alpha_schedule": "fixed",
    "ws_source": "greedy",
    "ws_epsilon": 0.25,
    "ma_tying": "none",
    "ma_transfer_from_standard": False,
    "final_local_search": False,
}


class QAOAFamilySolver(BaseSolver):
    """Unified wrapper over the MaxCut QAOA variants."""

    name = "qaoa_family"

    def solve(
        self,
        problem,
        policy: dict,
        backend,
        shots: int = 1000,
    ) -> SolverResult:
        pol = {**DEFAULT_QAOA_POLICY, **policy}
        pol["shots"] = int(pol.get("shots", shots))
        pol["estimator_shots"] = int(pol.get("estimator_shots", pol["shots"]))
        pol["sampler_shots"] = int(pol.get("sampler_shots", pol["shots"]))
        pol["reps"] = int(pol.get("reps", pol.get("p", 1)))
        pol["p"] = pol["reps"]
        variant, measurement_mode, _ = normalize_measurement_policy(pol)
        pol["variant"] = variant
        pol["measurement_mode"] = measurement_mode

        t0 = time.time()
        result = solve_qaoa_variant(problem, pol, backend, str(pol["variant"]).lower())
        result.wall_time_seconds = time.time() - t0
        return result
