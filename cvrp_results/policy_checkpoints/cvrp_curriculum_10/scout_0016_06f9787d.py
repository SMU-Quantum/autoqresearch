#!/usr/bin/env python3
"""
Adaptive multi-attempt experiment for constrained quantum optimization (MIS/QUBO).

This file exposes a small sequential policy surface for one-instance execution:

  state_t -> action_t

The editable policy functions decide which solver family to start with, when to
continue, and how to adapt after each observation. Everything below the policy
surface is fixed execution infrastructure.

Per-instance outputs from this script are diagnostic only. Keep/revert
decisions are made at suite level from ``evaluate_policy.py`` using
``suite_average_gap``.
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


DEFAULT_ESTIMATOR_SHOTS = 1024   # powers of 2 for efficient quantum sampling
DEFAULT_SAMPLER_SHOTS = 1024     # powers of 2 for efficient quantum sampling
DEFAULT_PROBLEM_SPEC = "mis_file_1tc.16"
DEFAULT_RESULTS_PATH = Path("results.tsv")
DEFAULT_PROGRESS_PATH = Path("instance_progress.png")


# ─── AttemptOutcome ──────────────────────────────────────────────


@dataclass
class AttemptOutcome:
    """Observation emitted after each attempt in the sequential control loop."""

    attempt: int
    learning_score: float       # shaped signal (lower is better, for decisions)
    optimality_gap: float       # (optimal - found) / optimal. Lower is better. THE METRIC.
    raw_feasible: bool          # did the solver produce a feasible best bitstring?
    raw_feasibility_rate: float # how close the distribution is to feasible
    raw_ar: float               # approximation ratio of best raw solution (observation)

    convergence_improvement: float  # (start - end) / |start|; > 0 means cost decreased
    convergence_stagnation: float   # > 0.8 reliably means stuck (cross-optimizer)
    final_cost: float               # terminal cost value

    policy_used: dict = field(default_factory=dict)
    wall_time: float = 0.0

    # Sampling concentration: how much the circuit concentrates probability.
    # top1_count / total_shots — near 0 means uniform noise, near 1 means
    # the circuit strongly favours one bitstring.
    top1_probability: float = 0.0
    # Top-10 bitstrings: list of (count, n_selected, feasible) tuples.
    # The agent should use this to decide whether the output is meaningful
    # or just near-uniform random noise (if top counts are all ~1-2).
    top10_summary: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# POLICY SURFACE — the agent edits these four functions
# ═══════════════════════════════════════════════════════════════════


def choose_solver_family(problem) -> str:
    """Choose the initial solver family.

    Starting point: always VQE.  The agent should discover when and whether
    to switch families through experimentation.
    """
    if getattr(problem, "problem_type", "") == "cvrp":
        return "vqe"
    if getattr(problem, "num_variables", 0) <= 16:
        return "qaoa"
    return "qrao"


def build_base_policy(problem, family: str) -> dict:
    """Build the base policy dict for a given solver family.

    Starting point: return the static VQE baseline for "vqe".
    The agent should discover and add support for other families as needed.
    """
    # ── Common defaults shared across all families ──
    common = {
        "solver_family": family,
        "optimizer_method": "COBYLA",
        "optimizer_tol": 1e-3,
        "optimizer_maxiter": 150,
        "learning_rate": 0.05,
        "entanglement": "linear",
        "estimator_shots": DEFAULT_ESTIMATOR_SHOTS,
        "sampler_shots": DEFAULT_SAMPLER_SHOTS,
        "seed": 17,
        "measurement_mode": "expectation",
        "cvar_alpha": 0.25,
        "penalty": None,
        "pce_local_search": False,
        "final_local_search": False,
    }
    if getattr(problem, "problem_type", "") == "cvrp":
        common.update(
            {
                "gap_solver_family": family,
                "route_solver_family": "classical",
                "route_quantum_qubit_threshold": 16,
                "route_quantum_fallback": True,
                "route_tsp_penalty": None,
                "estimator_shots": 2048,
                "sampler_shots": 16384,
                "cvrp_seed_method": "depot_farthest",
                "cvrp_gap_penalty_method": "tilted",
                "cvrp_taylor_alpha": 10.0,
                "cvrp_tilted_kappa": 5.0,
                "cvrp_tilted_s_frac": 0.10,
                "cvrp_tilted_s_min": 1.0,
            }
        )
    if family == "vqe":
        return {
            **common,
            "variant": "standard",
            "ansatz_type": "efficient_su2" if getattr(problem, "problem_type", "") == "cvrp" else "real_amplitudes",
            "vqe_reps": 1,
            "measurement_mode": "expectation" if getattr(problem, "problem_type", "") == "cvrp" else "cvar",
            "cvar_alpha": 0.25 if getattr(problem, "problem_type", "") == "cvrp" else 0.1,
        }
    if family == "qaoa":
        return {
            **common,
            "variant": "warmstart",
            "reps": 1,
            "ws_epsilon": 0.25,
            "ws_source": "relaxation",
            "measurement_mode": "cvar",
            "cvar_alpha": 0.25,
        }
    if family == "pce":
        return {
            **common,
            "pce_k": 2,
            "pce_depth": 10,
            "pce_alpha": None,
            "pce_beta": 0.5,
        }
    if family == "qrao":
        n_vars = getattr(problem, "num_variables", 0)
        problem_name = str(getattr(problem, "name", "") or "")
        if n_vars > 32:
            if "et" in problem_name:
                return {
                    **common,
                    "qrao_max_vars_per_qubit": 2,
                    "qrac_type": 2,
                    "rounding": "semideterministic",
                    "ansatz_type": "real_amplitudes",
                    "vqe_reps": 2,
                }
            return {
                **common,
                "qrao_max_vars_per_qubit": 3,
                "qrac_type": 3,
                "rounding": "semideterministic",
                "ansatz_type": "real_amplitudes",
                "vqe_reps": 1,
            }
        return {
            **common,
            "qrao_max_vars_per_qubit": 3,
            "qrac_type": 3,
            "rounding": "magic",
            "ansatz_type": "real_amplitudes",
            "vqe_reps": 1,
        }
    raise ValueError(f"Unknown solver family: {family}")


def should_continue(
    attempt: int,
    history: list[AttemptOutcome],
    problem=None,
    max_attempts: int = 5,
) -> bool:
    """Decide whether to run another solver attempt on this instance.

    Starting point: use the full attempt budget every time.
    The agent should discover smarter early-stopping logic.
    """
    if isinstance(problem, (int, np.integer)):
        max_attempts = int(problem)
    if attempt >= max_attempts:
        return False
    if not history:
        return True

    n_vars = getattr(problem, "num_variables", 0) if problem is not None else 0
    last = history[-1]

    if getattr(problem, "problem_type", "") == "cvrp":
        return False

    if n_vars <= 16:
        return False
    if last.optimality_gap <= 0.0:
        return False

    if n_vars > 32:
        if attempt >= 3:
            return False
        return bool((not last.raw_feasible) or last.optimality_gap > 0.4)
    return attempt < max_attempts


def adapt_policy(
    attempt: int,
    history: list[AttemptOutcome],
    problem,
    base_policy: dict | None = None,
) -> dict:
    """Adapt the solver policy between attempts on the same instance.

    Starting point: no adaptation — repeat the base policy every attempt.
    The agent should discover how to use history observations to build
    a useful adaptive controller.
    """
    if base_policy is None:
        base_policy = problem
        problem = None

    policy = base_policy.copy()
    if attempt == 0 or not history:
        return policy

    n_vars = getattr(problem, "num_variables", 0) if problem is not None else 0
    last = history[-1]

    if (
        n_vars > 32
        and policy.get("solver_family") == "qrao"
        and ((not last.raw_feasible) or last.optimality_gap > 0.4)
    ):
        policy["qrao_max_vars_per_qubit"] = 2
        policy["qrac_type"] = 2
        policy["rounding"] = "semideterministic"
    elif n_vars > 16:
        return policy

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
    """Shaped learning signal. Lower is better.

    When feasible: learning_score = optimality_gap (0 to 1).
    When infeasible: 1.0 + shaped term (always worse than any feasible).
    Gradient: feasibility_rate=0.48 → ~1.1, feasibility_rate=0.01 → ~1.4.
    """
    if is_feasible:
        return optimality_gap
    return 1.0 + 0.4 * (1.0 - feasibility_rate) + 0.1 * (1.0 - best_feasible_ar)


def _get_solver_fn(family: str):
    """Get the appropriate QUBO solver function for the given family."""
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
    raise ValueError(f"Unknown solver family: {family}")


def _make_backend(policy: dict, mode: str):
    """Create execution context from policy."""
    from autoqresearch.backends.factory import BackendConfig, create_execution_context

    return create_execution_context(
        BackendConfig(
            mode=mode,
            shots=int(policy.get("estimator_shots", DEFAULT_ESTIMATOR_SHOTS)),
            sampler_shots=int(policy.get("sampler_shots", DEFAULT_SAMPLER_SHOTS)),
            seed=int(policy.get("seed", 17)),
        )
    )


def _stage_policy(policy: dict, prefix: str, family: str) -> dict:
    """Return a solver policy with stage-prefixed overrides applied."""
    staged = policy.copy()
    for key, value in policy.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        if stripped:
            staged[stripped] = value
    staged["solver_family"] = family
    return staged


def _solve_cvrp_staged(problem, policy: dict, backend):
    """Solve CVRP as GAP QUBO followed by per-cluster route TSP QUBOs."""
    from autoqresearch.problems.cvrp import (
        build_route_tsp_problem,
        clusters_capacity_feasible,
        decode_gap_qubo_solution,
        decode_tsp_qubo_solution,
        route_cost,
        route_stage_qubit_counts,
        solve_cvrp_routes_classically,
    )
    from autoqresearch.solvers.base import SolverResult
    from autoqresearch.solvers.qubo_primitives import extract_solution

    gap_family = str(
        policy.get("gap_solver_family")
        or policy.get("solver_family")
        or choose_solver_family(problem)
    ).lower()
    route_family = str(policy.get("route_solver_family", "classical")).lower()
    if gap_family == "classical":
        raise ValueError("CVRP GAP stage must use a quantum QUBO solver family.")
    if route_family not in {"classical", "vqe", "qaoa", "qrao", "pce"}:
        raise ValueError("route_solver_family must be classical, vqe, qaoa, qrao, or pce.")

    gap_policy = _stage_policy(policy, "gap_", gap_family)
    gap_result = _get_solver_fn(gap_family)(problem, gap_policy, backend)
    gap_x, gap_cost, gap_feasible = extract_solution(gap_result.counts, problem)

    route_solutions = []
    route_results = []
    route_stage_feasible = bool(gap_feasible)
    route_stage_cost = float(gap_cost)
    route_counts = route_stage_qubit_counts([])
    route_shots = 0
    route_depth = 0
    route_cnot_count = 0
    route_two_qubit_count = 0
    route_total_gates = 0
    route_num_parameters = 0
    route_max_qubits = 0

    if gap_feasible:
        clusters = decode_gap_qubo_solution(gap_x, problem)
        route_counts = route_stage_qubit_counts(clusters)
        instance = problem.metadata["instance"]

        if route_family == "classical":
            route_solutions = solve_cvrp_routes_classically(instance, clusters)
        else:
            fallback_enabled = bool(policy.get("route_quantum_fallback", True))
            threshold = int(policy.get("route_quantum_qubit_threshold", 16))
            tsp_penalty = policy.get("route_tsp_penalty", policy.get("tsp_penalty"))
            route_policy_base = _stage_policy(policy, "route_", route_family)

            for route_index, cluster in enumerate(clusters):
                if not cluster:
                    route_solutions.append(
                        {
                            "route_index": route_index,
                            "customers": [],
                            "load": 0,
                            "solver": "classical_exact",
                            "route": [],
                            "cost": 0.0,
                            "fallback_reason": "empty_cluster",
                        }
                    )
                    continue

                route_problem = build_route_tsp_problem(
                    instance,
                    list(cluster),
                    route_index,
                    tsp_penalty=tsp_penalty,
                )
                if route_problem.num_variables > threshold and fallback_enabled:
                    classical_route = route_problem.metadata["classical_solution"]
                    route_solutions.append(
                        {
                            "route_index": route_index,
                            "customers": list(cluster),
                            "load": int(sum(instance["demands"][customer] for customer in cluster)),
                            **classical_route,
                            "fallback_reason": "route_qubit_threshold",
                            "tsp_qubits": route_problem.num_variables,
                        }
                    )
                    continue

                route_policy = route_policy_base.copy()
                route_policy["seed"] = int(policy.get("seed", 17)) + 101 + route_index
                route_result = _get_solver_fn(route_family)(route_problem, route_policy, backend)
                route_x, route_obj, route_feasible = extract_solution(route_result.counts, route_problem)
                route_results.append(route_result)
                route_shots += int(getattr(route_result, "num_shots", 0))
                route_depth += int(getattr(route_result, "circuit_depth", 0))
                route_cnot_count += int(getattr(route_result, "cnot_count", 0))
                route_two_qubit_count += int(getattr(route_result, "two_qubit_gate_count", 0))
                route_total_gates += int(getattr(route_result, "total_gate_count", 0))
                route_num_parameters += int(getattr(route_result, "num_parameters", 0))
                route_max_qubits = max(route_max_qubits, int(getattr(route_result, "num_qubits", 0)))
                if not route_feasible and fallback_enabled:
                    classical_route = route_problem.metadata["classical_solution"]
                    route_solutions.append(
                        {
                            "route_index": route_index,
                            "customers": list(cluster),
                            "load": int(sum(instance["demands"][customer] for customer in cluster)),
                            **classical_route,
                            "fallback_reason": "route_quantum_infeasible",
                            "tsp_qubits": route_problem.num_variables,
                        }
                    )
                    continue
                if not route_feasible:
                    route_stage_feasible = False
                    route_solutions.append(
                        {
                            "route_index": route_index,
                            "customers": list(cluster),
                            "load": int(sum(instance["demands"][customer] for customer in cluster)),
                            "solver": route_family,
                            "route": [],
                            "cost": float("inf"),
                            "tsp_qubits": route_problem.num_variables,
                        }
                    )
                    continue

                route = decode_tsp_qubo_solution(route_x, route_problem)
                route_solutions.append(
                    {
                        "route_index": route_index,
                        "customers": list(cluster),
                        "load": int(sum(instance["demands"][customer] for customer in cluster)),
                        "solver": route_family,
                        "route": route,
                        "cost": route_cost(instance, route),
                        "tsp_qubits": route_problem.num_variables,
                        "route_objective": route_obj,
                    }
                )

        route_stage_cost = float(sum(solution["cost"] for solution in route_solutions))
        route_stage_feasible = bool(
            route_stage_feasible
            and np.isfinite(route_stage_cost)
            and clusters_capacity_feasible(instance, clusters)
        )

    sequential_qubits = max(int(getattr(gap_result, "num_qubits", 0)), route_max_qubits)
    if route_family == "classical":
        sequential_qubits = int(getattr(gap_result, "num_qubits", 0))

    return SolverResult(
        best_bitstring=gap_x,
        best_objective=route_stage_cost,
        is_feasible=route_stage_feasible,
        counts=gap_result.counts,
        num_shots=int(getattr(gap_result, "num_shots", 0)) + route_shots,
        circuit_depth=int(getattr(gap_result, "circuit_depth", 0)) + route_depth,
        cnot_count=int(getattr(gap_result, "cnot_count", 0)) + route_cnot_count,
        two_qubit_gate_count=int(getattr(gap_result, "two_qubit_gate_count", 0)) + route_two_qubit_count,
        total_gate_count=int(getattr(gap_result, "total_gate_count", 0)) + route_total_gates,
        gate_counts=dict(getattr(gap_result, "gate_counts", {}) or {}),
        num_qubits=sequential_qubits,
        num_parameters=int(getattr(gap_result, "num_parameters", 0)) + route_num_parameters,
        optimizer_iterations=int(getattr(gap_result, "optimizer_iterations", 0))
        + sum(int(getattr(result, "optimizer_iterations", 0)) for result in route_results),
        final_cost=float(getattr(gap_result, "final_cost", 0.0)),
        parameter_values=getattr(gap_result, "parameter_values", None),
        convergence_history=list(getattr(gap_result, "convergence_history", []) or []),
        solver_name=f"cvrp_gap_{gap_family}_route_{route_family}",
        metadata={
            "gap_solver_family": gap_family,
            "route_solver_family": route_family,
            "gap_feasible": bool(gap_feasible),
            "gap_objective": float(gap_cost),
            "route_solutions": route_solutions,
            "route_qubit_counts": route_counts,
            "gap_qubits": int(getattr(gap_result, "num_qubits", 0)),
            "route_max_qubits": int(route_max_qubits),
            "sequential_qubits": int(sequential_qubits),
            "route_quantum_result_count": len(route_results),
        },
    )


def _parse_problem_spec(spec: str) -> tuple[str, int | str, int]:
    """Parse problem specification string.

    Standard format: ``knapsack_12_s3`` → ("knapsack", 12, 3)
    File-based MIS:  ``mis_file_1tc.32`` → ("mis_file", "1tc.32", 0)
    File-based CVRP: ``cvrp_file_E-n13-k4`` → ("cvrp_file", "E-n13-k4", 0)
    """
    if spec.startswith("mis_file_"):
        filename = spec[len("mis_file_"):]
        return "mis_file", filename, 0
    if spec.startswith("cvrp_file_"):
        filename = spec[len("cvrp_file_"):]
        return "cvrp_file", filename, 0
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


def build_static_baseline_policy(problem) -> dict:
    """Return the fixed conservative VQE baseline used for comparisons.

    This helper lives in fixed infrastructure so baseline evaluation uses the
    same engine as adaptive runs, with only the policy frozen.
    """
    if getattr(problem, "problem_type", "") == "cvrp":
        return {
            "solver_family": "vqe",
            "gap_solver_family": "vqe",
            "route_solver_family": "classical",
            "route_quantum_qubit_threshold": 16,
            "route_quantum_fallback": True,
            "route_tsp_penalty": None,
            "variant": "standard",
            "measurement_mode": "expectation",
            "ansatz_type": "efficient_su2",
            "vqe_reps": 1,
            "entanglement": "linear",
            "optimizer_method": "COBYLA",
            "optimizer_maxiter": 150,
            "optimizer_tol": 1e-3,
            "cvar_alpha": 0.25,
            "estimator_shots": DEFAULT_ESTIMATOR_SHOTS,
            "sampler_shots": DEFAULT_SAMPLER_SHOTS,
            "learning_rate": 0.05,
            "seed": 17,
            "penalty": None,
            "pce_local_search": False,
            "final_local_search": False,
            "cvrp_seed_method": "depot_farthest",
            "cvrp_gap_penalty_method": "hard_slack",
            "cvrp_taylor_alpha": 10.0,
            "cvrp_tilted_kappa": 5.0,
            "cvrp_tilted_s_frac": 0.10,
            "cvrp_tilted_s_min": 1.0,
        }

    return {
        "solver_family": "vqe",
        "variant": "standard",
        "measurement_mode": "expectation",
        "ansatz_type": "real_amplitudes",
        "vqe_reps": 1,
        "entanglement": "linear",
        "optimizer_method": "COBYLA",
        "optimizer_maxiter": 150,
        "optimizer_tol": 1e-3,
        "cvar_alpha": 0.25,
        "estimator_shots": DEFAULT_ESTIMATOR_SHOTS,
        "sampler_shots": DEFAULT_SAMPLER_SHOTS,
        "learning_rate": 0.05,
        "seed": 17,
        "penalty": None,
        "pce_local_search": False,
        "final_local_search": False,
        # None keeps the QUBO penalty on automatic selection.
    }


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
    "penalty",
    "rounding",
    "qrao_max_vars_per_qubit",
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
    "gap_solver_family",
    "route_solver_family",
    "route_quantum_qubit_threshold",
    "route_quantum_fallback",
    "route_tsp_penalty",
    "route_vqe_reps",
    "route_reps",
    "route_ansatz_type",
    "route_entanglement",
    "route_optimizer_method",
    "route_optimizer_maxiter",
    "route_optimizer_tol",
    "route_estimator_shots",
    "route_sampler_shots",
    "route_measurement_mode",
    "route_cvar_alpha",
    "route_qrao_max_vars_per_qubit",
    "route_qrac_type",
    "route_rounding",
    "route_pce_k",
    "route_pce_depth",
    "route_pce_alpha",
    "route_pce_beta",
    "gap_vqe_reps",
    "gap_reps",
    "gap_ansatz_type",
    "gap_entanglement",
    "gap_optimizer_method",
    "gap_optimizer_maxiter",
    "gap_optimizer_tol",
    "gap_estimator_shots",
    "gap_sampler_shots",
    "gap_measurement_mode",
    "gap_cvar_alpha",
    "gap_qrao_max_vars_per_qubit",
    "gap_qrac_type",
    "gap_rounding",
    "gap_pce_k",
    "gap_pce_depth",
    "gap_pce_alpha",
    "gap_pce_beta",
    "cvrp_seed_method",
    "cvrp_gap_penalty_method",
    "cvrp_taylor_alpha",
    "cvrp_tilted_kappa",
    "cvrp_tilted_s_frac",
    "cvrp_tilted_s_min",
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
        "artifact_role",
        "evaluation_layer",
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
    """Maintain the per-instance diagnostic ledger schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = _results_header()
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as handle:
            csv.writer(handle, delimiter="\t").writerow(header)
        return

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames == header:
            return
        rows = list(reader)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(column, "") for column in header])


def _reset_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(_results_header())


def _next_experiment_id(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 1

    max_id = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                max_id = max(max_id, int(str(row.get("experiment_id", "0")).strip() or 0))
            except ValueError:
                pass
    return max_id + 1


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
        parts.append(
            f"qrao_max_vars_per_qubit="
            f"{int(policy.get('qrao_max_vars_per_qubit', policy.get('qrac_type', 2)))}"
        )
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
    """Append a diagnostic instance-level row.

    These rows are never used for suite-level keep/revert decisions.
    """

    _ensure_results_file(path)
    experiment_id = _next_experiment_id(path)
    status = "crash" if row[_results_header().index("optimality_gap")] in ("", None) else "logged"
    row[0] = experiment_id
    row[_results_header().index("status")] = status

    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)

    return experiment_id, status


def _update_progress_plot(results_path: Path, output_path: Path) -> None:
    """Diagnostic per-instance progress plot. Suite progress lives elsewhere."""
    try:
        from analysis import load_results, make_progress_plot
        data = load_results(results_path, metric="optimality_gap")
        make_progress_plot(
            data,
            output_path=output_path,
            title="Knapsack Instance Diagnostics (optimality gap, lower is better)",
        )
    except Exception:
        pass  # Suite-level plot from evaluate_policy.py is the primary artifact


def run_experiment(
    problem_spec: str,
    backend_mode: str = "ideal_mps",
    solver_family: str | None = None,
    max_attempts: int = 5,
    timeout: int = 900,
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
    seed_override: int | None = None,
) -> dict:
    problem_type, size_or_filename, seed = _parse_problem_spec(problem_spec)
    from autoqresearch.problems.registry import (
        get_cvrp_file_instance,
        get_cvrp_instance,
        get_single_instance,
        get_mis_file_instance,
    )
    from autoqresearch.solvers.qubo_primitives import (
        check_cvrp_feasibility,
        check_mis_feasibility,
        check_qubo_feasibility,
        compute_cvrp_best_feasible_ar,
        compute_cvrp_feasibility_rate,
        compute_best_feasible_ar,
        compute_feasibility_rate,
        compute_mis_best_feasible_ar,
        compute_mis_feasibility_rate,
        cvrp_objective_value,
        fixed_repair,
        mis_objective_value,
        qubo_objective_value,
    )

    if not no_results_log:
        _ensure_results_file(results_file)

    policy_override = _load_policy_override(policy_file, policy_json)
    if problem_type == "mis_file":
        # Penalty is a tunable: the agent may set it in the policy to
        # control the QUBO conversion penalty for QuadraticProgramToQubo.
        # None means auto-computed by Qiskit.
        mis_penalty = (
            policy_override.get("penalty") if policy_override else None
        )
        problem = get_mis_file_instance(size_or_filename, penalty=mis_penalty)
    elif problem_type == "cvrp_file":
        override = policy_override or {}
        problem = get_cvrp_file_instance(
            str(size_or_filename),
            capacity_method=str(override.get("cvrp_gap_penalty_method", "tilted")),
            seed_method=str(override.get("cvrp_seed_method", "depot_farthest")),
            gap_penalty=override.get("penalty"),
            taylor_alpha=float(override.get("cvrp_taylor_alpha", 10.0)),
            tilted_kappa=float(override.get("cvrp_tilted_kappa", 5.0)),
            tilted_s_frac=float(override.get("cvrp_tilted_s_frac", 0.10)),
            tilted_s_min=float(override.get("cvrp_tilted_s_min", 1.0)),
        )
    elif problem_type == "cvrp":
        override = policy_override or {}
        problem = get_cvrp_instance(
            int(size_or_filename),
            seed,
            capacity_method=str(override.get("cvrp_gap_penalty_method", "tilted")),
            seed_method=str(override.get("cvrp_seed_method", "depot_farthest")),
            gap_penalty=override.get("penalty"),
            taylor_alpha=float(override.get("cvrp_taylor_alpha", 10.0)),
            tilted_kappa=float(override.get("cvrp_tilted_kappa", 5.0)),
            tilted_s_frac=float(override.get("cvrp_tilted_s_frac", 0.10)),
            tilted_s_min=float(override.get("cvrp_tilted_s_min", 1.0)),
        )
    else:
        problem = get_single_instance(problem_type, size_or_filename, seed)
    initial_family = (
        solver_family
        or (str(policy_override.get("gap_solver_family")) if policy_override and policy_override.get("gap_solver_family") else None)
        or (str(policy_override.get("solver_family")) if policy_override and policy_override.get("solver_family") else None)
        or choose_solver_family(problem)
    )
    base_policy = build_base_policy(problem, initial_family)
    base_policy = _merge_policy_override(base_policy, policy_override)
    base_policy["solver_family"] = initial_family
    if problem.problem_type == "cvrp":
        base_policy["gap_solver_family"] = str(base_policy.get("gap_solver_family", initial_family)).lower()
        base_policy["solver_family"] = base_policy["gap_solver_family"]
    if seed_override is not None:
        base_policy["seed"] = int(seed_override)
    policy_mode = "static" if policy_override is not None else "adaptive"
    policy_source = "policy_file" if policy_file is not None else "policy_json" if policy_json is not None else "default"

    print(
        f"Problem: {problem.name} (n_items={problem.metadata.get('num_items', problem.metadata.get('num_customers', '?'))}, "
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
        if seed_override is not None:
            policy["seed"] = int(seed_override)
        if problem.problem_type == "cvrp":
            attempt_family = str(
                policy.get("gap_solver_family")
                or policy.get("solver_family", initial_family)
            ).lower()
            policy["gap_solver_family"] = attempt_family
        else:
            attempt_family = str(policy.get("solver_family", initial_family)).lower()
        policy["solver_family"] = attempt_family
        policy["pce_local_search"] = False
        policy["final_local_search"] = False
        attempt_base = build_base_policy(problem, attempt_family)
        if attempt_family == "qrao":
            if (
                "qrao_max_vars_per_qubit" in attempt_base
                and "qrao_max_vars_per_qubit" not in policy
            ):
                policy["qrao_max_vars_per_qubit"] = attempt_base["qrao_max_vars_per_qubit"]
            qrao_ratio = int(
                policy.get(
                    "qrao_max_vars_per_qubit",
                    policy.get("qrac_type", attempt_base.get("qrac_type", 3)),
                )
            )
            policy["qrao_max_vars_per_qubit"] = qrao_ratio
            policy["qrac_type"] = qrao_ratio
        if attempt_family == "pce" and "pce_k" in attempt_base and "pce_k" not in policy:
            policy["pce_k"] = attempt_base["pce_k"]
        solve_fn = _get_solver_fn(attempt_family)

        backend = _make_backend(policy, backend_mode)

        t0 = time.time()
        try:
            if problem.problem_type == "cvrp":
                result = _solve_cvrp_staged(problem, policy, backend)
            else:
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

        _bs = result.best_bitstring
        if problem.problem_type == "knapsack":
            feas_rate = compute_feasibility_rate(result.counts, problem)
            is_feasible = check_qubo_feasibility(_bs, problem)
            found_value = qubo_objective_value(_bs, problem)
            ar = found_value / problem.optimal_value if problem.optimal_value > 0 else 0.0
            ar = min(1.0, max(0.0, ar))
            attempt_best_feas_ar = compute_best_feasible_ar(result.counts, problem)
        elif problem.problem_type == "mis":
            is_feasible = check_mis_feasibility(_bs, problem)
            mis_size = mis_objective_value(_bs, problem)
            ar = (mis_size / max(problem.optimal_value, 1e-10)) if is_feasible else 0.0
            ar = min(1.0, max(0.0, ar))
            feas_rate = compute_mis_feasibility_rate(result.counts, problem)
            attempt_best_feas_ar = compute_mis_best_feasible_ar(result.counts, problem)
        elif problem.problem_type == "cvrp":
            is_feasible = bool(result.is_feasible and check_cvrp_feasibility(_bs, problem))
            found_cost = float(result.best_objective if result.best_objective is not None else cvrp_objective_value(_bs, problem))
            ar = (float(problem.optimal_value) / max(found_cost, 1e-10)) if is_feasible and np.isfinite(found_cost) else 0.0
            ar = min(1.0, max(0.0, ar))
            feas_rate = compute_cvrp_feasibility_rate(result.counts, problem)
            attempt_best_feas_ar = compute_cvrp_best_feasible_ar(result.counts, problem)
        else:
            # Generic fallback
            is_feasible = problem.is_feasible(_bs)
            obj_val = problem.objective_value(_bs)
            ar = abs(obj_val / max(abs(problem.optimal_value), 1e-10)) if is_feasible else 0.0
            ar = min(1.0, max(0.0, ar))
            feas_rate = 0.0
            attempt_best_feas_ar = 0.0
        gap = _compute_optimality_gap(ar, is_feasible)
        best_feasible_ar_global = max(best_feasible_ar_global, attempt_best_feas_ar)
        improvement, stagnation, final_cost = _normalize_convergence(
            result.convergence_history
        )
        learning = _compute_learning_score(
            gap, is_feasible, feas_rate, best_feasible_ar_global, result
        )

        # ── Sampling concentration analysis ──────────────────────────
        # Helps the agent decide if the circuit output is meaningful or
        # near-uniform noise.  Top-10 by count, with feasibility info.
        _sorted_counts = sorted(
            result.counts.items(), key=lambda kv: kv[1], reverse=True
        )
        _total_shots = max(sum(c for _, c in _sorted_counts), 1)
        _top1_prob = _sorted_counts[0][1] / _total_shots if _sorted_counts else 0.0
        _top10 = []
        _n = problem.num_variables
        for _bs_str, _cnt in _sorted_counts[:10]:
            _xarr = np.array([int(b) for b in _bs_str[::-1]], dtype=float)
            if len(_xarr) < _n:
                _xarr = np.pad(_xarr, (0, _n - len(_xarr)))
            elif len(_xarr) > _n:
                _xarr = _xarr[:_n]
            _sel = int(sum(_xarr[:_n]))
            _f = (
                check_mis_feasibility(_xarr, problem)
                if problem.problem_type == "mis"
                else check_cvrp_feasibility(_xarr, problem)
                if problem.problem_type == "cvrp"
                else True
            )
            _top10.append({"count": _cnt, "prob": round(_cnt / _total_shots, 4),
                           "selected": _sel, "feasible": _f})

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
            top1_probability=_top1_prob,
            top10_summary=_top10,
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
                "top1_probability": _top1_prob,
                "top10_summary": _top10,
                **attempt_stats,
            }
        )

        marker = "*" if gap == best_gap and gap < 1.0 else " "
        print(
            f"  Attempt {attempt}: "
            f"gap={gap:.4f} "
            f"learning={learning:.4f} "
            f"feas_rate={feas_rate:.3f} "
            f"AR={ar:.3f} "
            f"stagnation={stagnation:.2f} "
            f"top1_prob={_top1_prob:.4f} "
            f"time={elapsed:.1f}s {marker}"
        )
        # Show top-10 sampling distribution so the agent can judge
        # whether the circuit concentrates probability or is noise.
        _t10_parts = []
        for _entry in _top10[:10]:
            _fchar = "F" if _entry["feasible"] else "X"
            _t10_parts.append(
                f"cnt={_entry['count']}|sel={_entry['selected']}|{_fchar}"
            )
        print(f"    top10: [{', '.join(_t10_parts)}]")

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
        _bs = best_raw_result.best_bitstring

        if problem.problem_type == "knapsack":
            best_ar = qubo_objective_value(_bs, problem) / max(problem.optimal_value, 1e-10)
            best_feasible = check_qubo_feasibility(_bs, problem)
            best_feas_rate = compute_feasibility_rate(best_raw_result.counts, problem)
        else:
            # Generic evaluation path (MIS, maxcut, etc.)
            if problem.problem_type == "mis":
                best_feasible = check_mis_feasibility(_bs, problem)
                mis_size = mis_objective_value(_bs, problem)
                best_ar = (mis_size / max(problem.optimal_value, 1e-10)) if best_feasible else 0.0
                best_feas_rate = compute_mis_feasibility_rate(best_raw_result.counts, problem)
            elif problem.problem_type == "cvrp":
                best_feasible = bool(best_raw_result.is_feasible and check_cvrp_feasibility(_bs, problem))
                best_cost = float(best_raw_result.best_objective if best_raw_result.best_objective is not None else cvrp_objective_value(_bs, problem))
                best_ar = (float(problem.optimal_value) / max(best_cost, 1e-10)) if best_feasible and np.isfinite(best_cost) else 0.0
                best_ar = min(1.0, max(0.0, best_ar))
                best_feas_rate = compute_cvrp_feasibility_rate(best_raw_result.counts, problem)
            else:
                obj_val = problem.objective_value(_bs)
                best_feasible = problem.is_feasible(_bs)
                best_ar = abs(obj_val / max(abs(problem.optimal_value), 1e-10))
                best_feas_rate = 0.0
                if best_raw_result.counts:
                    n_vars = problem.num_variables
                    n_feasible = 0
                    n_total = 0
                    for bitstr, count in best_raw_result.counts.items():
                        bits = [int(b) for b in bitstr[-n_vars:][::-1]]  # reverse for Qiskit MSB convention
                        n_total += count
                        is_feas = problem.is_feasible(np.array(bits, dtype=float))
                        if is_feas:
                            n_feasible += count
                    best_feas_rate = n_feasible / max(n_total, 1) if n_total > 0 else 0.0

        best_gap = _compute_optimality_gap(best_ar, best_feasible)
        print(f"optimality_gap: {best_gap:.6f}")
        print(f"raw_ar: {best_ar:.4f}")
        print(f"raw_feasible: {int(best_feasible)}")
        print(f"raw_feasibility_rate: {best_feas_rate:.4f}")
        print(f"learning_score: {best_learning:.6f}")

        # Post-repair: only for knapsack (repair heuristic is problem-specific)
        if problem.problem_type == "knapsack":
            repaired_x, repair_changed = fixed_repair(_bs, problem)
            values = problem.metadata["values"]
            weights = problem.metadata["weights"]
            capacity = problem.metadata["capacity"]
            n_items = problem.metadata["num_items"]

            repaired_value = float(np.dot(values, repaired_x[:n_items]))
            repaired_feasible = float(np.dot(weights, repaired_x[:n_items])) <= capacity
            repaired_ar = repaired_value / max(problem.optimal_value, 1e-10)
            repaired_score = _compute_optimality_gap(repaired_ar, repaired_feasible)

            print(f"\n{'=' * 60}")
            print("Post-repair results (secondary — for completeness)")
            print("=" * 60)
            print(f"repaired_optimality_gap: {repaired_score:.6f}")
            print(f"repaired_ar: {repaired_ar:.4f}")
            print(f"repaired_feasible: {int(repaired_feasible)}")
            print(f"repair_changed: {int(repair_changed)}")
        else:
            # No repair heuristic for non-knapsack problems
            repaired_score = best_gap
            repaired_ar = best_ar
            repaired_feasible = best_feasible
            repair_changed = False
    else:
        status = "crash"
        if not no_results_log:
            row = [
                0,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "diagnostic",
                "instance",
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
                "",
                len(history),
                f"{total_wall_time:.1f}",
            ]
            experiment_id, results_status = _append_results_row(results_file, row)
            print(f"results_file: {results_file}")
            print(f"results_experiment_id: {experiment_id}")
            print(f"results_status: {results_status}")
            print("results_artifact_role: diagnostic")
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
        "problem_type": problem.problem_type,
        "size": problem.metadata.get(
            "num_customers",
            size_or_filename if isinstance(size_or_filename, int) else problem.num_variables,
        ),
        "seed": seed,
        "seed_override": int(seed_override) if seed_override is not None else None,
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
    # Compact policy summary for progress plot annotations
    _ps_parts = [winning_family.upper()]
    if winning_policy:
        _ansatz = winning_policy.get("ansatz_type", "")
        _reps = winning_policy.get("vqe_reps") or winning_policy.get("reps") or winning_policy.get("pce_depth")
        _opt = winning_policy.get("optimizer_method", "")
        _meas = winning_policy.get("measurement_mode", "")
        _variant = winning_policy.get("variant", "")
        _ent = winning_policy.get("entanglement", "")
        if _ansatz:
            _ps_parts.append(_ansatz.replace("_", ""))
        if _ent and _ent != "linear":
            _ps_parts.append(f"ent={_ent}")
        if _reps is not None:
            _ps_parts.append(f"d={_reps}")
        if _opt:
            _ps_parts.append(_opt)
        if _meas == "cvar":
            _alpha = winning_policy.get("cvar_alpha", "")
            _ps_parts.append(f"CVaR({_alpha})" if _alpha else "CVaR")
        if _variant and _variant not in ("standard",):
            _ps_parts.append(_variant)
    print(f"policy_summary: {' '.join(_ps_parts)}")
    print(f"backend: {backend_mode}")
    print(f"problem: {problem.name}")

    if not no_results_log and best_raw_result is not None and best_outcome is not None:
        policy_for_log = best_outcome.policy_used
        solver_name = str(getattr(best_raw_result, "solver_name", winning_family) or winning_family)
        row = [
            0,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostic",
            "instance",
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
        print("results_artifact_role: diagnostic")

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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Diagnostic TSV ledger for per-instance runs; never used for keep/revert.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PROGRESS_PATH,
        help="Diagnostic instance-level progress plot regenerated after logged runs.",
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
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help="Override the policy/backend seed for validation reruns without changing policy logic.",
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
        seed_override=args.seed_override,
    )
    return 0 if summary.get("status") != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
