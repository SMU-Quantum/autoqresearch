"""QRAO solver wrapper for the MaxCut-first implementation."""

from __future__ import annotations

import time

from .base import BaseSolver, SolverResult
from .maxcut_primitives import solve_qrao


DEFAULT_QRAO_POLICY = {
    "qrao_max_vars_per_qubit": 3,
    "qrac_type": 3,
    "rounding": "semideterministic",
    "ansatz": "real_amplitudes",
    "ansatz_type": "real_amplitudes",
    "entanglement": "linear",
    "reps": 2,
    "vqe_reps": 2,
    "optimizer": "COBYLA",
    "optimizer_method": "COBYLA",
    "optimizer_maxiter": 200,
    "measurement_mode": "expectation",
    "cvar_alpha": 0.25,
    "shots": 1000,
    "estimator_shots": 1000,
    "sampler_shots": 1000,
    "final_local_search": False,
}


class QRAOSolver(BaseSolver):
    """Thin wrapper over the MaxCut QRAO implementation."""

    name = "qrao"

    def solve(
        self,
        problem,
        policy: dict,
        backend,
        shots: int = 1000,
    ) -> SolverResult:
        pol = {**DEFAULT_QRAO_POLICY, **policy}
        pol["shots"] = int(pol.get("shots", shots))
        pol["estimator_shots"] = int(pol.get("estimator_shots", pol["shots"]))
        pol["sampler_shots"] = int(pol.get("sampler_shots", pol["shots"]))
        pol["qrao_max_vars_per_qubit"] = int(
            pol.get("qrao_max_vars_per_qubit", pol.get("qrac_type", 3))
        )
        pol["qrac_type"] = pol["qrao_max_vars_per_qubit"]
        pol["vqe_reps"] = int(pol.get("vqe_reps", pol.get("reps", 2)))
        pol["reps"] = pol["vqe_reps"]
        pol["ansatz_type"] = pol.get("ansatz_type", pol.get("ansatz", "real_amplitudes"))

        t0 = time.time()
        result = solve_qrao(problem, pol, backend)
        result.wall_time_seconds = time.time() - t0
        return result
