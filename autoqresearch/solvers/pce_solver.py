"""PCE solver wrapper for the supported problem families."""

from __future__ import annotations

import time

from .base import BaseSolver, SolverResult
from .maxcut_primitives import solve_pce
from .qubo_primitives import solve_qubo_pce


DEFAULT_PCE_POLICY = {
    "k": 2,
    "pce_k": 2,
    "depth": 10,
    "pce_depth": 10,
    "optimizer": "COBYLA",
    "optimizer_method": "COBYLA",
    "optimizer_maxiter": 100,
    "optimizer_tol": 1e-3,
    "learning_rate": 0.05,
    "initialization": "random",
    "shots": 1000,
    "estimator_shots": 1000,
    "sampler_shots": 1000,
    "ansatz_type": "brickwork",
    "entanglement": "linear",
    "measurement_mode": "expectation",
    "cvar_alpha": 0.25,
    "pce_alpha": None,
    "pce_beta": 0.5,
    "pce_local_search": False,
    "final_local_search": False,
}


class PCESolver(BaseSolver):
    """Thin wrapper over the PCE implementations."""

    name = "pce"

    def solve(
        self,
        problem,
        policy: dict,
        backend,
        shots: int = 1000,
    ) -> SolverResult:
        pol = {**DEFAULT_PCE_POLICY, **policy}
        pol["shots"] = int(pol.get("shots", shots))
        pol["estimator_shots"] = int(pol.get("estimator_shots", pol["shots"]))
        pol["sampler_shots"] = int(pol.get("sampler_shots", pol["shots"]))
        pol["pce_k"] = int(pol.get("pce_k", pol.get("k", 2)))
        pol["k"] = pol["pce_k"]
        pol["pce_depth"] = int(pol.get("pce_depth", pol.get("depth", 10)))
        pol["depth"] = pol["pce_depth"]

        t0 = time.time()
        if problem.problem_type == "maxcut":
            result = solve_pce(problem, pol, backend)
        elif problem.problem_type in ("knapsack", "mis", "cvrp", "cvrp_tsp"):
            # Both knapsack and MIS have a QUBO that can be converted to
            # weighted MaxCut for PCE.  The helper _qubo_pce_graph()
            # is a generic QUBO → weighted-MaxCut reduction.
            result = solve_qubo_pce(problem, pol, backend)
        else:
            raise NotImplementedError(
                f"PCE is not wired for problem type '{problem.problem_type}'."
            )
        result.wall_time_seconds = time.time() - t0
        return result
