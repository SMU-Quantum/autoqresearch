"""VQE-family solver wrappers for the MaxCut-first implementation."""

from __future__ import annotations

import time

from .base import BaseSolver, SolverResult
from .maxcut_primitives import normalize_measurement_policy, solve_vqe_variant


DEFAULT_VQE_POLICY = {
    "variant": "standard",
    "ansatz": "efficient_su2",
    "ansatz_type": "efficient_su2",
    "entanglement": "linear",
    "reps": 2,
    "vqe_reps": 2,
    "rotation_blocks": ["ry", "rz"],
    "optimizer": "COBYLA",
    "optimizer_method": "COBYLA",
    "optimizer_maxiter": 200,
    "optimizer_tol": 1e-3,
    "initialization": "random",
    "num_restarts": 1,
    "shots": 1000,
    "estimator_shots": 1000,
    "sampler_shots": 1000,
    "measurement_mode": "expectation",
    "cvar_alpha": 0.25,
    "final_local_search": False,
}


class VQEFamilySolver(BaseSolver):
    """Unified wrapper over the MaxCut VQE variants."""

    name = "vqe_family"

    def solve(
        self,
        problem,
        policy: dict,
        backend,
        shots: int = 1000,
    ) -> SolverResult:
        pol = {**DEFAULT_VQE_POLICY, **policy}
        pol["shots"] = int(pol.get("shots", shots))
        pol["estimator_shots"] = int(pol.get("estimator_shots", pol["shots"]))
        pol["sampler_shots"] = int(pol.get("sampler_shots", pol["shots"]))
        pol["vqe_reps"] = int(pol.get("vqe_reps", pol.get("reps", 2)))
        pol["reps"] = pol["vqe_reps"]
        pol["ansatz_type"] = pol.get("ansatz_type", pol.get("ansatz", "efficient_su2"))
        variant, measurement_mode, _ = normalize_measurement_policy(pol)
        pol["variant"] = variant
        pol["measurement_mode"] = measurement_mode

        t0 = time.time()
        result = solve_vqe_variant(problem, pol, backend, str(pol["variant"]).lower())
        result.wall_time_seconds = time.time() - t0
        return result
