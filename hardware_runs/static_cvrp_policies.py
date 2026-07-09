#!/usr/bin/env python3
"""Static CVRP hardware replay policies."""

from __future__ import annotations

from typing import Any


CVRP_E13_POLICY: dict[str, Any] = {
    "solver_family": "hybrid",
    "gap_solver_family": "hybrid",
    "hybrid_sub_family": "vqe",
    "hybrid_ambiguity_threshold": 0.5,
    "route_solver_family": "classical",
    "route_quantum_qubit_threshold": 16,
    "route_quantum_fallback": True,
    "route_tsp_penalty": None,
    "variant": "standard",
    "ansatz_type": "efficient_su2",
    "vqe_reps": 1,
    "entanglement": "linear",
    "optimizer_method": "COBYLA",
    "optimizer_tol": 1e-3,
    "optimizer_maxiter": 150,
    "learning_rate": 0.05,
    "measurement_mode": "expectation",
    "cvar_alpha": 0.25,
    "estimator_shots": 2048,
    "sampler_shots": 16384,
    "seed": 17,
    "penalty": None,
    "pce_local_search": False,
    "final_local_search": False,
    "cvrp_seed_method": "depot_farthest",
    "cvrp_gap_penalty_method": "tilted",
    "cvrp_taylor_alpha": 10.0,
    "cvrp_tilted_kappa": 5.0,
    "cvrp_tilted_s_frac": 0.10,
    "cvrp_tilted_s_min": 1.0,
}


CVRP_E13_POLICY_NOTE = (
    "Retained E-n13-k4 policy: hybrid classical greedy GAP, VQE refinement "
    "on ambiguous customers, and classical exact route TSP."
)


def get_cvrp_e13_policy() -> dict[str, Any]:
    return dict(CVRP_E13_POLICY)


def get_cvrp_e13_policy_note() -> str:
    return CVRP_E13_POLICY_NOTE
