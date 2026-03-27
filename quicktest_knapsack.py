#!/usr/bin/env python3
"""
quicktest_knapsack.py - Smoke test for the knapsack-first adaptive stack.

Tests:
  Phase 1: Unit tests (problem generation, feasibility, convergence stats,
           learning score, repair, adaptive policy logic)
  Phase 2: Solver tests (QAOA, VQE, QRAO, PCE smokes)
  Phase 3: End-to-end adaptive experiment loop
  Phase 4: CLI quickrun + TSV + plots

Usage:
    ./.venv/bin/python quicktest_knapsack.py
    ./.venv/bin/python quicktest_knapsack.py --phase 1
    ./.venv/bin/python quicktest_knapsack.py --phase 4
    ./.venv/bin/python quicktest_knapsack.py --size 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


PASS = 0
FAIL = 0
WARN = 0
SKIP = 0

DEFAULT_RESULTS_PATH = Path("quicktest_knapsack_results.tsv")
DEFAULT_PLOTS_DIR = Path("quicktest_knapsack_figures")


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  [FAIL] {name}{suffix}")


def warn(name: str, detail: str = "") -> None:
    global WARN
    WARN += 1
    suffix = f"  {detail}" if detail else ""
    print(f"  [WARN] {name}{suffix}")


def skip(name: str, detail: str = "") -> None:
    global SKIP
    SKIP += 1
    suffix = f"  {detail}" if detail else ""
    print(f"  [SKIP] {name}{suffix}")


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"{title}")
    print("=" * 68)


def make_backend(est_shots: int = 128, samp_shots: int = 128):
    from autoqresearch.backends.factory import BackendConfig, create_execution_context

    return create_execution_context(
        BackendConfig(mode="ideal_mps", shots=est_shots, sampler_shots=samp_shots)
    )


def parse_metric_output(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        metrics[key.strip()] = value.strip()
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Unit Tests
# ═══════════════════════════════════════════════════════════════════


def phase1_unit_tests(size: int) -> None:
    section("Phase 1: Unit Tests")

    # ── 1a. Knapsack problem generation ──────────────────────────
    print("\n  [Knapsack problem generation]")
    from autoqresearch.problems.knapsack import KnapsackGenerator
    from autoqresearch.problems.registry import get_single_instance

    gen = KnapsackGenerator(tightness=0.6)
    problem = gen.generate(size, seed=0)
    larger_problem = gen.generate(max(size, 14), seed=0)

    check("Knapsack instance created", problem is not None)
    check("Problem type is knapsack", problem.problem_type == "knapsack")
    check(
        "Num items matches requested size",
        problem.metadata["num_items"] == size,
        f"got {problem.metadata['num_items']}",
    )
    check(
        "QUBO has more variables than items (slack vars added)",
        problem.num_variables > size,
        f"n_qubo={problem.num_variables} n_items={size}",
    )
    check(
        "Optimal value is positive",
        problem.optimal_value > 0,
        f"got {problem.optimal_value}",
    )
    check("Optimal solution exists", problem.optimal_solution is not None)

    # Verify optimal solution is feasible
    weights = problem.metadata["weights"]
    capacity = problem.metadata["capacity"]
    opt_x = problem.optimal_solution
    opt_weight = np.dot(weights, opt_x)
    check(
        "Optimal solution is feasible",
        opt_weight <= capacity,
        f"weight={opt_weight} capacity={capacity}",
    )
    opt_val = np.dot(problem.metadata["values"], opt_x)
    check(
        "Optimal value matches optimal solution",
        abs(opt_val - problem.optimal_value) < 1e-9,
        f"computed={opt_val} stored={problem.optimal_value}",
    )

    # Registry lookup works
    problem2 = get_single_instance("knapsack", size, seed=0)
    check("Registry returns knapsack instance", problem2.problem_type == "knapsack")

    # ── 1b. Feasibility checking ─────────────────────────────────
    print("\n  [Feasibility checking]")
    from autoqresearch.solvers.qubo_primitives import (
        check_qubo_feasibility,
        compute_feasibility_rate,
        qubo_objective_value,
    )

    # Build a QUBO-space bitstring from optimal original solution
    n_qubo = problem.num_variables
    n_items = problem.metadata["num_items"]

    # Feasible solution: optimal
    qubo_feasible = np.zeros(n_qubo)
    qubo_feasible[:n_items] = opt_x
    check(
        "Optimal solution passes feasibility check",
        check_qubo_feasibility(qubo_feasible, problem),
    )

    # Infeasible solution: all items selected (likely exceeds capacity)
    qubo_all = np.zeros(n_qubo)
    qubo_all[:n_items] = 1.0
    all_weight = np.dot(weights, np.ones(n_items))
    if all_weight > capacity:
        check(
            "All-items solution is infeasible",
            not check_qubo_feasibility(qubo_all, problem),
        )
    else:
        skip("All-items-infeasible test", "all items fit in this instance")

    # Objective value
    obj = qubo_objective_value(qubo_feasible, problem)
    check(
        "Objective value matches expected",
        abs(obj - problem.optimal_value) < 1e-6,
        f"got {obj} expected {problem.optimal_value}",
    )

    # Feasibility rate from counts
    feasible_bs = "".join(str(int(b)) for b in qubo_feasible[::-1].astype(int))
    zero_bs = "0" * n_qubo
    counts = {feasible_bs: 7, zero_bs: 3}
    feas_rate = compute_feasibility_rate(counts, problem)
    check(
        "Feasibility rate computed correctly",
        0.0 <= feas_rate <= 1.0,
        f"got {feas_rate}",
    )

    # ── 1c. Convergence normalization ────────────────────────────
    print("\n  [Convergence normalization]")
    from experiment import _compute_optimality_gap, _normalize_convergence

    # Decreasing trace → positive improvement
    trace1 = [10.0, 8.0, 6.0, 5.0, 4.5, 4.3, 4.2, 4.15]
    imp1, stag1, fc1 = _normalize_convergence(trace1)
    check("Improvement > 0 for decreasing trace", imp1 > 0, f"got {imp1:.4f}")
    check("Final cost matches last entry", abs(fc1 - 4.15) < 1e-6, f"got {fc1}")

    # Flat trace → high stagnation
    trace2 = [5.0] * 20
    imp2, stag2, fc2 = _normalize_convergence(trace2)
    check("Stagnation high for flat trace", stag2 >= 0.8, f"got {stag2:.2f}")

    # Empty trace
    imp3, stag3, fc3 = _normalize_convergence([])
    check(
        "Empty trace is treated as fully stagnated",
        imp3 == 0.0 and stag3 == 1.0 and fc3 == 0.0,
    )

    # Single entry
    imp4, stag4, fc4 = _normalize_convergence([7.0])
    check("Single-entry trace returns zero improvement", imp4 == 0.0)
    check("Single-entry trace final_cost correct", abs(fc4 - 7.0) < 1e-6)

    # ── 1d. Learning score ───────────────────────────────────────
    print("\n  [Learning score]")
    from experiment import _compute_learning_score

    # Feasible → learning_score = optimality_gap
    ls1 = _compute_learning_score(0.2, True, 0.9, 0.7)
    check("Feasible: learning_score = optimality_gap", abs(ls1 - 0.2) < 1e-6, f"got {ls1}")

    # Infeasible with high feasibility_rate remains worse than any feasible run.
    ls2 = _compute_learning_score(0.0, False, 1.0, 0.5)
    expected2 = 1.0 + 0.4 * (1.0 - 1.0) + 0.1 * (1.0 - 0.5)
    check(
        "Infeasible high feas_rate: learning is shaped above 1.0",
        abs(ls2 - expected2) < 1e-6,
        f"got {ls2} expected {expected2}",
    )

    # Infeasible with zero feasibility is penalized more heavily.
    ls3 = _compute_learning_score(0.0, False, 0.0, 0.0)
    expected3 = 1.0 + 0.4 * (1.0 - 0.0) + 0.1 * (1.0 - 0.0)
    check("Infeasible zero feas_rate: learning > 1.0", abs(ls3 - expected3) < 1e-6)

    # Near-miss gradient
    ls4 = _compute_learning_score(0.0, False, 0.48, 0.0)
    check("Near-miss has gradient", ls4 < ls3, f"got {ls4} vs {ls3}")
    check("Near-miss still worse than any feasible run", ls4 > 1.0, f"got {ls4}")

    gap1 = _compute_optimality_gap(0.8, True)
    check("Optimality gap = 1 - AR when feasible", abs(gap1 - 0.2) < 1e-9, f"got {gap1}")
    gap2 = _compute_optimality_gap(0.8, False)
    check("Optimality gap = 1.0 when infeasible", abs(gap2 - 1.0) < 1e-9, f"got {gap2}")

    # ── 1e. Fixed repair ─────────────────────────────────────────
    print("\n  [Fixed repair]")
    from autoqresearch.solvers.qubo_primitives import fixed_repair

    # Feasible solution → no change
    repaired, changed = fixed_repair(qubo_feasible, problem)
    check("Feasible solution: no repair needed", not changed)
    check(
        "Feasible solution: values preserved",
        np.array_equal(repaired[:n_items], opt_x),
    )

    # Infeasible solution → repair drops items
    if all_weight > capacity:
        repaired_inf, changed_inf = fixed_repair(qubo_all, problem)
        check("Infeasible all-items: repair changed", changed_inf)
        repaired_weight = np.dot(weights, repaired_inf[:n_items])
        check(
            "Repaired solution is feasible",
            repaired_weight <= capacity,
            f"weight={repaired_weight} capacity={capacity}",
        )
        check(
            "Repaired solution drops some items",
            np.sum(repaired_inf[:n_items]) < n_items,
        )

    # ── 1f. Adaptive policy surface ──────────────────────────────
    print("\n  [Adaptive policy surface]")
    from experiment import (
        AttemptOutcome,
        _load_policy_override,
        adapt_policy,
        build_base_policy,
        choose_solver_family,
        snapshot_policy,
        should_continue,
    )
    from study_analysis import (
        summarize_ablation,
        summarize_budget,
        summarize_same_seed,
        summarize_transfer,
    )
    from study_runner import (
        build_followup_policies,
        build_static_variant_policy,
        expand_study_cases,
        rebudget_policy,
    )

    family = choose_solver_family(problem)
    larger_family = choose_solver_family(larger_problem)
    check("choose_solver_family returns valid family", family in {"qaoa", "vqe", "qrao", "pce"}, f"got {family}")
    check("choose_solver_family returns valid family for larger problem", larger_family in {"qaoa", "vqe", "qrao", "pce"}, f"got {larger_family}")

    base_policy = build_base_policy(problem, family)
    check("build_base_policy returns dict", isinstance(base_policy, dict))
    check("Base policy stores solver family", base_policy.get("solver_family") == family)
    if family == "vqe":
        check(
            "Base VQE ansatz is real_amplitudes or efficient_su2",
            base_policy.get("ansatz_type") in {"real_amplitudes", "efficient_su2"},
            f"got {base_policy.get('ansatz_type')}",
        )
        check(
            "Base VQE reps >= 1",
            int(base_policy.get("vqe_reps", 0)) >= 1,
            f"got {base_policy.get('vqe_reps')}",
        )
    elif family == "qaoa":
        check(
            "Base QAOA reps >= 1",
            int(base_policy.get("reps", 0)) >= 1,
            f"got {base_policy.get('reps')}",
        )
    check(
        "Base policy default estimator shots = 1000",
        int(base_policy.get("estimator_shots", 0)) == 1000,
        f"got {base_policy.get('estimator_shots')}",
    )
    check(
        "Base policy default sampler shots = 1000",
        int(base_policy.get("sampler_shots", 0)) == 1000,
        f"got {base_policy.get('sampler_shots')}",
    )

    # -- Structural invariants: must hold for ANY adapt_policy implementation --
    # These tests verify contracts that the infrastructure depends on.
    # They do NOT test specific adaptive behaviors (those change as the agent evolves).

    # Synthetic histories should stay semantically consistent with
    # experiment._compute_learning_score, even when tests only assert structure.
    low_feas_history = [
        AttemptOutcome(
            attempt=0,
            learning_score=1.482,
            optimality_gap=1.0,
            raw_feasible=False,
            raw_feasibility_rate=0.02,
            raw_ar=0.1,
            convergence_improvement=0.02,
            convergence_stagnation=0.10,
            final_cost=1.0,
            policy_used=base_policy.copy(),
            wall_time=1.0,
        )
    ]
    adapted = adapt_policy(1, low_feas_history, problem, base_policy)
    check(
        "adapt_policy returns a dict",
        isinstance(adapted, dict),
    )
    check(
        "adapt_policy result contains solver_family",
        "solver_family" in adapted or "solver_family" in base_policy,
    )
    check(
        "adapt_policy does not mutate base_policy object",
        adapted is not base_policy,
    )

    stuck_history = [
        AttemptOutcome(
            attempt=0,
            learning_score=1.40,
            optimality_gap=1.0,
            raw_feasible=False,
            raw_feasibility_rate=0.20,
            raw_ar=0.2,
            convergence_improvement=0.01,
            convergence_stagnation=0.92,
            final_cost=0.8,
            policy_used=base_policy.copy(),
            wall_time=1.0,
        )
    ]
    adapted_stuck = adapt_policy(1, stuck_history, problem, base_policy)
    check(
        "adapt_policy returns dict on stagnation input",
        isinstance(adapted_stuck, dict),
    )

    good_history = [
        AttemptOutcome(
            attempt=0,
            learning_score=0.18,
            optimality_gap=0.18,
            raw_feasible=True,
            raw_feasibility_rate=0.95,
            raw_ar=0.84,
            convergence_improvement=0.40,
            convergence_stagnation=0.10,
            final_cost=0.2,
            policy_used=base_policy.copy(),
            wall_time=1.0,
        )
    ]
    adapted_good = adapt_policy(1, good_history, problem, base_policy)
    check(
        "adapt_policy returns dict on good-feasible input",
        isinstance(adapted_good, dict),
    )
    check(
        "should_continue returns bool",
        isinstance(should_continue(1, good_history, problem, max_attempts=5), bool),
    )
    check(
        "should_continue allows more attempts within budget",
        should_continue(0, [], problem, max_attempts=5),
    )

    great_history = good_history + [
        AttemptOutcome(
            attempt=1,
            learning_score=0.03,
            optimality_gap=0.03,
            raw_feasible=True,
            raw_feasibility_rate=0.98,
            raw_ar=0.98,
            convergence_improvement=0.45,
            convergence_stagnation=0.05,
            final_cost=0.1,
            policy_used=base_policy.copy(),
            wall_time=1.0,
        )
    ]

    mature_great_history = great_history + [
        AttemptOutcome(
            attempt=2,
            learning_score=0.029,
            optimality_gap=0.029,
            raw_feasible=True,
            raw_feasibility_rate=0.985,
            raw_ar=0.981,
            convergence_improvement=0.46,
            convergence_stagnation=0.05,
            final_cost=0.09,
            policy_used=base_policy.copy(),
            wall_time=1.0,
        ),
        AttemptOutcome(
            attempt=3,
            learning_score=0.029,
            optimality_gap=0.029,
            raw_feasible=True,
            raw_feasibility_rate=0.986,
            raw_ar=0.981,
            convergence_improvement=0.46,
            convergence_stagnation=0.05,
            final_cost=0.09,
            policy_used=base_policy.copy(),
            wall_time=1.0,
        ),
    ]
    check(
        "should_continue stops at budget ceiling",
        not should_continue(5, mature_great_history, problem, max_attempts=5),
    )

    # -- Structural key preservation: qrac_type and pce_k must never change in adapt_policy --

    qrao_policy = build_base_policy(larger_problem, "qrao")
    qrao_history = [
        AttemptOutcome(
            attempt=0,
            learning_score=0.18,
            optimality_gap=0.18,
            raw_feasible=True,
            raw_feasibility_rate=0.95,
            raw_ar=0.84,
            convergence_improvement=0.40,
            convergence_stagnation=0.10,
            final_cost=0.2,
            policy_used=qrao_policy.copy(),
            wall_time=1.0,
        )
    ]
    adapted_qrao = adapt_policy(1, qrao_history, larger_problem, qrao_policy)
    check(
        "QRAO qrac_type remains fixed across attempts",
        adapted_qrao.get("qrac_type") == qrao_policy.get("qrac_type"),
        f"base={qrao_policy.get('qrac_type')} adapted={adapted_qrao.get('qrac_type')}",
    )

    pce_policy = build_base_policy(problem, "pce")
    low_quality_feasible = [
        AttemptOutcome(
            attempt=0,
            learning_score=0.80,
            optimality_gap=0.80,
            raw_feasible=True,
            raw_feasibility_rate=0.90,
            raw_ar=0.25,
            convergence_improvement=0.05,
            convergence_stagnation=0.10,
            final_cost=0.4,
            policy_used=pce_policy.copy(),
            wall_time=1.0,
        )
    ]
    adapted_pce = adapt_policy(1, low_quality_feasible, problem, pce_policy)
    check(
        "PCE k remains fixed across attempts",
        adapted_pce.get("pce_k") == pce_policy.get("pce_k"),
        f"base={pce_policy.get('pce_k')} adapted={adapted_pce.get('pce_k')}",
    )

    # ── 1g. Machine-readable policy snapshots and study helpers ──
    print("\n  [Machine-readable study helpers]")
    snap = snapshot_policy({**base_policy, "unknown_key": "ignore", "cvar_alpha": None})
    check("snapshot_policy keeps solver family", snap.get("solver_family") == family)
    check("snapshot_policy omits unknown keys", "unknown_key" not in snap)

    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.json"
        policy_path.write_text(json.dumps(snap))
        loaded_policy = _load_policy_override(policy_file=policy_path)
        check("Policy snapshot reloads from JSON file", loaded_policy == snap)

    static_qaoa_family, static_qaoa_policy = build_static_variant_policy(problem, "static_qaoa_warmstart")
    check("Static warmstart policy uses QAOA", static_qaoa_family == "qaoa", f"got {static_qaoa_family}")
    check(
        "Static warmstart policy sets warmstart variant",
        static_qaoa_policy.get("variant") == "warmstart",
        f"got {static_qaoa_policy.get('variant')}",
    )
    static_qaoa_cvar_family, static_qaoa_cvar_policy = build_static_variant_policy(
        problem,
        "static_qaoa_warmstart_cvar",
    )
    check(
        "Static warmstart CVaR policy keeps warmstart circuit",
        static_qaoa_cvar_family == "qaoa" and static_qaoa_cvar_policy.get("variant") == "warmstart",
        f"got {static_qaoa_cvar_family} / {static_qaoa_cvar_policy.get('variant')}",
    )
    check(
        "Static warmstart CVaR policy enables cvar measurement",
        static_qaoa_cvar_policy.get("measurement_mode") == "cvar",
        f"got {static_qaoa_cvar_policy.get('measurement_mode')}",
    )

    manifest_cases = expand_study_cases(
        {"problem_type": "knapsack", "sizes": [5], "splits": ["train", "dev"]}
    )
    seeds = sorted(case["seed"] for case in manifest_cases)
    check("Manifest expansion creates 10 train+dev cases", len(manifest_cases) == 10, f"got {len(manifest_cases)}")
    check("Manifest expansion includes train seeds", seeds[:5] == [0, 1, 2, 3, 4], f"got {seeds[:5]}")
    check("Manifest expansion includes dev seeds", seeds[5:] == [100, 101, 102, 103, 104], f"got {seeds[5:]}")

    rebudgeted = rebudget_policy(
        {"solver_family": "vqe", "estimator_shots": 100, "sampler_shots": 100, "optimizer_maxiter": 10},
        target_total_shots=2200,
        reference_iterations=10,
    )
    check(
        "Rebudgeting scales estimator shots upward",
        int(rebudgeted.get("estimator_shots", 0)) > 100,
        f"got {rebudgeted.get('estimator_shots')}",
    )

    adaptive_summary = {
        "winning_policy": {"solver_family": "vqe", "ansatz_type": "brickwork", "vqe_reps": 2, "entanglement": "linear"},
        "direct_stage2_policy": {"solver_family": "vqe", "ansatz_type": "efficient_su2", "vqe_reps": 1, "entanglement": "linear"},
    }
    followups = build_followup_policies(adaptive_summary)
    check("Followup policies include static_final", "static_final" in followups)
    check("Followup policies include static_direct_stage2", "static_direct_stage2" in followups)

    synthetic_rows = [
        {
            "run_id": "1",
            "status": "completed",
            "problem": "knapsack_12_s0",
            "size": "12",
            "seed": "0",
            "split": "train",
            "variant": "adaptive_full",
            "prompt_variant": "full",
            "budget_mode": "trajectory",
            "optimality_gap": "0.10",
            "raw_ar": "0.91",
            "raw_feasible": "1",
            "shots_to_first_feasible": "1200",
            "shots_to_ar_ge_0_5": "1200",
            "total_run_shots": "4000",
            "total_wall_time_s": "12.0",
        },
        {
            "run_id": "2",
            "status": "completed",
            "problem": "knapsack_12_s0",
            "size": "12",
            "seed": "0",
            "split": "train",
            "variant": "static_final",
            "prompt_variant": "full",
            "budget_mode": "trajectory",
            "optimality_gap": "0.20",
            "raw_ar": "0.80",
            "raw_feasible": "1",
            "shots_to_first_feasible": "2000",
            "shots_to_ar_ge_0_5": "2000",
            "total_run_shots": "2000",
            "total_wall_time_s": "8.0",
        },
        {
            "run_id": "3",
            "status": "completed",
            "problem": "knapsack_12_s0",
            "size": "12",
            "seed": "0",
            "split": "train",
            "variant": "static_direct_stage2",
            "prompt_variant": "full",
            "budget_mode": "trajectory",
            "optimality_gap": "0.15",
            "raw_ar": "0.86",
            "raw_feasible": "1",
            "shots_to_first_feasible": "1800",
            "shots_to_ar_ge_0_5": "1800",
            "total_run_shots": "2200",
            "total_wall_time_s": "8.5",
        },
        {
            "run_id": "4",
            "status": "completed",
            "problem": "knapsack_12_s100",
            "size": "12",
            "seed": "100",
            "split": "dev",
            "variant": "adaptive_full",
            "prompt_variant": "full",
            "budget_mode": "trajectory",
            "optimality_gap": "0.12",
            "raw_ar": "0.89",
            "raw_feasible": "1",
            "shots_to_first_feasible": "1300",
            "shots_to_ar_ge_0_5": "1300",
            "total_run_shots": "4100",
            "total_wall_time_s": "12.5",
        },
        {
            "run_id": "5",
            "status": "completed",
            "problem": "knapsack_12_s200",
            "size": "12",
            "seed": "200",
            "split": "test",
            "variant": "adaptive_full",
            "prompt_variant": "full",
            "budget_mode": "trajectory",
            "optimality_gap": "0.13",
            "raw_ar": "0.88",
            "raw_feasible": "1",
            "shots_to_first_feasible": "1400",
            "shots_to_ar_ge_0_5": "1400",
            "total_run_shots": "4200",
            "total_wall_time_s": "13.0",
        },
    ]
    same_seed_summary = summarize_same_seed(synthetic_rows)
    transfer_summary = summarize_transfer(synthetic_rows)
    budget_summary = summarize_budget(synthetic_rows)
    ablation_summary = summarize_ablation(synthetic_rows)
    check("Same-seed summary contains static_final comparison", any(row["challenger"] == "static_final" for row in same_seed_summary))
    check("Transfer summary includes adaptive_full trajectory row", any(row["variant"] == "adaptive_full" and row["budget_mode"] == "trajectory" for row in transfer_summary))
    check("Budget summary includes adaptive_full", any(row["variant"] == "adaptive_full" for row in budget_summary))
    check("Ablation summary groups prompt_variant", any(row["prompt_variant"] == "full" for row in ablation_summary))


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Solver Tests
# ═══════════════════════════════════════════════════════════════════


def run_solver_test(name, solve_fn, problem, policy, backend):
    """Run a single solver test and check basic invariants."""
    from autoqresearch.solvers.qubo_primitives import (
        compute_feasibility_rate,
        qubo_objective_value,
    )

    try:
        t0 = time.time()
        result = solve_fn(problem, policy, backend)
        elapsed = time.time() - t0

        conditions = [
            math.isfinite(result.best_objective),
            result.num_qubits > 0,
            result.num_parameters >= 0,
            isinstance(result.counts, dict),
            len(result.counts) > 0,
            isinstance(result.convergence_history, list),
        ]
        ok = all(conditions)

        feas_rate = compute_feasibility_rate(result.counts, problem)
        obj = qubo_objective_value(result.best_bitstring, problem)
        ar = obj / problem.optimal_value if problem.optimal_value > 0 else 0.0
        optimizer_iters = str(result.optimizer_iterations)
        if str(getattr(result, "solver_name", "")).startswith("qrao") and int(result.optimizer_iterations) == 0:
            optimizer_iters = "n/a"

        detail = (
            f"AR={ar:.3f} "
            f"feasible={int(result.is_feasible)} "
            f"feas_rate={feas_rate:.3f} "
            f"depth={result.circuit_depth} "
            f"2q={result.two_qubit_gate_count} "
            f"qubits={result.num_qubits} "
            f"iters={optimizer_iters} "
            f"time={elapsed:.1f}s"
        )
        check(name, ok, detail if not ok else "")
        print(f"       {detail}")
        if "compression_ratio" in result.metadata:
            original_variables = int(
                result.metadata.get("original_variables", problem.num_variables)
            )
            encoded_qubits = int(
                result.metadata.get("encoded_qubits", result.num_qubits)
            )
            compression_ratio = float(result.metadata.get("compression_ratio", 1.0))
            print(
                "       "
                f"encoding={original_variables} binary vars : {encoded_qubits} qubits "
                f"≈ {compression_ratio:.3f}"
            )
        return result, elapsed
    except Exception as exc:
        check(name, False, f"EXCEPTION: {exc}")
        traceback.print_exc()
        return None, None


def phase2_solver_tests(size: int) -> None:
    section("Phase 2: Solver Tests")

    from autoqresearch.problems.registry import get_single_instance
    from autoqresearch.solvers.qubo_primitives import (
        solve_qubo_qaoa,
        solve_qubo_vqe,
    )
    from autoqresearch.solvers.pce_solver import PCESolver
    from autoqresearch.solvers.qrao_solver import QRAOSolver

    problem = get_single_instance("knapsack", size, seed=0)
    backend = make_backend(est_shots=128, samp_shots=128)
    qrao_solver = QRAOSolver()
    pce_solver = PCESolver()

    print(
        f"\n  Problem: {problem.name} "
        f"(n_items={problem.metadata['num_items']}, "
        f"n_qubo={problem.num_variables}, "
        f"optimal={problem.optimal_value})\n"
    )

    cases = [
        (
            "QAOA standard reps=1",
            solve_qubo_qaoa,
            {
                "variant": "standard",
                "reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QAOA CVaR alpha=0.25 reps=1",
            solve_qubo_qaoa,
            {
                "variant": "cvar",
                "reps": 1,
                "cvar_alpha": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QAOA standard reps=2",
            solve_qubo_qaoa,
            {
                "variant": "standard",
                "reps": 2,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QAOA warmstart reps=1",
            solve_qubo_qaoa,
            {
                "variant": "warmstart",
                "reps": 1,
                "ws_source": "relaxation",
                "ws_epsilon": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QAOA warmstart CVaR reps=1",
            solve_qubo_qaoa,
            {
                "variant": "warmstart",
                "measurement_mode": "cvar",
                "cvar_alpha": 0.25,
                "reps": 1,
                "ws_source": "relaxation",
                "ws_epsilon": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QAOA multiangle reps=1",
            solve_qubo_qaoa,
            {
                "variant": "multiangle",
                "reps": 1,
                "ma_tying": "none",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QAOA multiangle CVaR reps=1",
            solve_qubo_qaoa,
            {
                "variant": "multiangle",
                "measurement_mode": "cvar",
                "cvar_alpha": 0.25,
                "reps": 1,
                "ma_tying": "none",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "VQE efficient_su2",
            solve_qubo_vqe,
            {
                "variant": "standard",
                "ansatz_type": "efficient_su2",
                "vqe_reps": 1,
                "entanglement": "linear",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "VQE brickwork",
            solve_qubo_vqe,
            {
                "variant": "standard",
                "ansatz_type": "brickwork",
                "vqe_reps": 2,
                "entanglement": "linear",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 10,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "VQE PauliTwoDesign",
            solve_qubo_vqe,
            {
                "variant": "standard",
                "ansatz_type": "pauli_two_design",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 10,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "CVaR-VQE alpha=0.25",
            solve_qubo_vqe,
            {
                "variant": "cvar",
                "ansatz_type": "efficient_su2",
                "vqe_reps": 1,
                "entanglement": "linear",
                "cvar_alpha": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QRAO qrac=2",
            lambda p, pol, b: qrao_solver.solve(p, pol, b, shots=128),
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "ansatz_type": "real_amplitudes",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QRAO qrac=2 brickwork",
            lambda p, pol, b: qrao_solver.solve(p, pol, b, shots=128),
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "ansatz_type": "brickwork",
                "vqe_reps": 2,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 10,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QRAO qrac=2 PauliTwoDesign",
            lambda p, pol, b: qrao_solver.solve(p, pol, b, shots=128),
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "ansatz_type": "pauli_two_design",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 10,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "QRAO qrac=2 CVaR",
            lambda p, pol, b: qrao_solver.solve(p, pol, b, shots=128),
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "measurement_mode": "cvar",
                "cvar_alpha": 0.25,
                "ansatz_type": "real_amplitudes",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "seed": 42,
            },
        ),
        (
            "PCE k=2",
            lambda p, pol, b: pce_solver.solve(p, pol, b, shots=128),
            {
                "pce_k": 2,
                "pce_depth": 2,
                "ansatz_type": "brickwork",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "pce_local_search": True,
                "seed": 42,
            },
        ),
        (
            "PCE k=2 CVaR",
            lambda p, pol, b: pce_solver.solve(p, pol, b, shots=128),
            {
                "pce_k": 2,
                "pce_depth": 2,
                "measurement_mode": "cvar",
                "cvar_alpha": 0.25,
                "ansatz_type": "brickwork",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "pce_local_search": True,
                "seed": 42,
            },
        ),
        (
            "PCE k=2 efficient_su2",
            lambda p, pol, b: pce_solver.solve(p, pol, b, shots=128),
            {
                "pce_k": 2,
                "pce_depth": 1,
                "ansatz_type": "efficient_su2",
                "entanglement": "linear",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "pce_local_search": True,
                "seed": 42,
            },
        ),
        (
            "PCE k=2 PauliTwoDesign",
            lambda p, pol, b: pce_solver.solve(p, pol, b, shots=128),
            {
                "pce_k": 2,
                "pce_depth": 1,
                "ansatz_type": "pauli_two_design",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "pce_local_search": True,
                "seed": 42,
            },
        ),
        (
            "PCE k=3",
            lambda p, pol, b: pce_solver.solve(p, pol, b, shots=128),
            {
                "pce_k": 3,
                "pce_depth": 2,
                "ansatz_type": "real_amplitudes",
                "entanglement": "linear",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "estimator_shots": 128,
                "sampler_shots": 128,
                "pce_local_search": True,
                "seed": 42,
            },
        ),
    ]

    for name, solve_fn, policy in cases:
        run_solver_test(name, solve_fn, problem, policy, backend)


# ═══════════════════════════════════════════════════════════════════
# Phase 3: End-to-End Adaptive Loop
# ═══════════════════════════════════════════════════════════════════


def phase3_end_to_end(size: int) -> None:
    section("Phase 3: End-to-End Adaptive Experiment")

    from autoqresearch.problems.registry import get_single_instance
    from autoqresearch.solvers.qubo_primitives import (
        check_qubo_feasibility,
        compute_best_feasible_ar,
        compute_feasibility_rate,
        fixed_repair,
        qubo_objective_value,
    )
    from experiment import (
        AttemptOutcome,
        _compute_learning_score,
        _compute_optimality_gap,
        _get_solver_fn,
        _normalize_convergence,
        adapt_policy,
        build_base_policy,
        choose_solver_family,
        should_continue,
    )

    problem = get_single_instance("knapsack", size, seed=0)
    print(
        f"\n  Problem: {problem.name} "
        f"(n_items={problem.metadata['num_items']}, "
        f"n_qubo={problem.num_variables}, "
        f"optimal={problem.optimal_value})\n"
    )

    family = choose_solver_family(problem)
    check("choose_solver_family returns valid family", family in ("qaoa", "vqe", "qrao"))

    base_policy = build_base_policy(problem, family)
    check("build_base_policy returns dict", isinstance(base_policy, dict))
    check("Base policy keeps solver family", base_policy.get("solver_family") == family)
    solve_fn = _get_solver_fn(family)

    # Run 2 attempts with small budget
    test_policy = dict(base_policy)
    test_policy.update(
        {
            "optimizer_maxiter": 5 if family != "qrao" else 8,
            "estimator_shots": 128,
            "sampler_shots": 128,
            "seed": 42,
        }
    )
    if family == "qaoa":
        test_policy["reps"] = 1
    elif family in {"vqe", "qrao"}:
        test_policy["vqe_reps"] = 1

    backend = make_backend(est_shots=128, samp_shots=128)
    history: list[AttemptOutcome] = []
    best_raw_result = None
    best_gap = float("inf")
    best_feasible_ar_global = 0.0
    max_attempts = 2

    print(f"  Running {max_attempts} attempts...\n")
    for attempt in range(max_attempts):
        policy = adapt_policy(attempt, history, problem, test_policy)
        check(f"adapt_policy returns dict (attempt {attempt})", isinstance(policy, dict))

        t0 = time.time()
        result = solve_fn(problem, policy, backend)
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

        print(
            f"    Attempt {attempt}: "
            f"gap={gap:.4f} "
            f"learning={learning:.4f} "
            f"feas_rate={feas_rate:.3f} "
            f"AR={ar:.3f} "
            f"time={elapsed:.1f}s"
        )

    check("History has correct length", len(history) == max_attempts)
    check(
        "should_continue stops at max",
        not should_continue(max_attempts, history, problem, max_attempts),
    )
    check("AttemptOutcome has learning_score", hasattr(history[0], "learning_score"))
    check("AttemptOutcome has convergence_stagnation", hasattr(history[0], "convergence_stagnation"))
    check(
        "Learning scores are finite",
        all(math.isfinite(o.learning_score) for o in history),
    )
    check(
        "Feasibility rates in [0,1]",
        all(0.0 <= o.raw_feasibility_rate <= 1.0 for o in history),
    )
    check(
        "Optimality gaps are bounded",
        all(0.0 <= o.optimality_gap <= 1.0 for o in history),
    )

    # Test repair
    if best_raw_result is not None:
        repaired_x, repair_changed = fixed_repair(
            best_raw_result.best_bitstring, problem
        )
        n_items = problem.metadata["num_items"]
        values = problem.metadata["values"]
        weights = problem.metadata["weights"]
        capacity = problem.metadata["capacity"]

        repaired_weight = np.dot(weights, repaired_x[:n_items])
        check("Repaired solution is feasible", repaired_weight <= capacity)

        repaired_value = np.dot(values, repaired_x[:n_items])
        repaired_ar = repaired_value / max(problem.optimal_value, 1e-10)
        print(f"\n    Repair: changed={repair_changed} "
              f"repaired_AR={repaired_ar:.3f} "
              f"repaired_value={repaired_value:.1f}")
        check("Repaired AR is finite", math.isfinite(repaired_ar))


# ═══════════════════════════════════════════════════════════════════
# Phase 4: CLI Quickrun + Plots
# ═══════════════════════════════════════════════════════════════════


def write_tsv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "experiment_id",
                "timestamp",
                "problem",
                "solver",
                "status",
                "description",
                "optimality_gap",
                "learning_score",
                "raw_ar",
                "raw_feasible",
                "raw_feasibility_rate",
                "repaired_optimality_gap",
                "repaired_ar",
                "repaired_feasible",
                "repair_changed",
                "total_attempts",
                "total_time_s",
                "backend",
            ]
        )


def append_cli_tsv_row(
    path: Path,
    experiment_id: int,
    metrics: dict[str, str],
    description: str,
) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        total_time = str(metrics.get("total_time", "")).rstrip("s")
        writer.writerow(
            [
                experiment_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metrics.get("problem", ""),
                metrics.get("solver_family", ""),
                "logged",
                description,
                metrics.get("optimality_gap", ""),
                metrics.get("learning_score", ""),
                metrics.get("raw_ar", ""),
                metrics.get("raw_feasible", ""),
                metrics.get("raw_feasibility_rate", ""),
                metrics.get("repaired_optimality_gap", ""),
                metrics.get("repaired_ar", ""),
                metrics.get("repaired_feasible", ""),
                metrics.get("repair_changed", ""),
                metrics.get("total_attempts", ""),
                total_time,
                metrics.get("backend", ""),
            ]
        )


def phase4_cli_and_plots(size: int, results_path: Path, plots_dir: Path) -> None:
    section("Phase 4: CLI Quickrun and Plots")

    repo_root = Path(__file__).resolve().parent
    experiments = [
        ("Policy train seed0", f"knapsack_{size}"),
        ("Policy dev seed100", f"knapsack_{size}_s100"),
        ("Policy dev seed101", f"knapsack_{size}_s101"),
    ]

    write_tsv_header(results_path)
    machine_dir = plots_dir / "machine_outputs"
    machine_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running {len(experiments)} CLI experiments via experiment.py\n")
    for experiment_id, (description, problem_name) in enumerate(experiments, start=1):
        sys.stdout.write(f"  [{experiment_id:02d}/{len(experiments):02d}] {description:<18s} ")
        sys.stdout.flush()
        summary_path = machine_dir / f"run_{experiment_id:02d}_summary.json"
        attempts_path = machine_dir / f"run_{experiment_id:02d}_attempts.jsonl"
        winning_policy_path = machine_dir / f"run_{experiment_id:02d}_winning_policy.json"
        command = [
            sys.executable,
            "experiment.py",
            "--problem",
            problem_name,
            "--backend",
            "ideal_mps",
            "--max-attempts",
            "3",
            "--timeout",
            "180",
            "--no-results-log",
            "--no-progress-plot",
            "--run-tag",
            f"quicktest-{experiment_id}",
            "--summary-json",
            str(summary_path),
            "--attempts-jsonl",
            str(attempts_path),
            "--winning-policy-json",
            str(winning_policy_path),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=240,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}")

        metrics = parse_metric_output(completed.stdout)
        required = {
            "optimality_gap",
            "learning_score",
            "raw_ar",
            "raw_feasible",
            "raw_feasibility_rate",
            "repaired_optimality_gap",
            "repaired_ar",
            "repaired_feasible",
            "repair_changed",
            "total_attempts",
            "total_time",
            "total_run_shots",
            "first_feasible_attempt",
            "first_ar_ge_0_5_attempt",
            "best_attempt_index",
            "run_tag",
            "policy_mode",
            "solver_family",
            "problem",
            "backend",
        }
        missing = sorted(required - set(metrics))
        if missing:
            raise RuntimeError(f"missing output fields: {missing}")

        summary_payload = json.loads(summary_path.read_text())
        attempt_lines = [line for line in attempts_path.read_text().splitlines() if line.strip()]
        winning_policy = json.loads(winning_policy_path.read_text())
        if int(summary_payload.get("total_attempts", 0)) <= 0:
            raise RuntimeError("summary JSON missing total_attempts")
        if not attempt_lines:
            raise RuntimeError("attempts JSONL is empty")
        if not isinstance(winning_policy, dict) or not winning_policy:
            raise RuntimeError("winning policy JSON missing policy payload")

        append_cli_tsv_row(results_path, experiment_id, metrics, description)
        print(
            f"gap={float(metrics['optimality_gap']):.4f} "
            f"repair_gap={float(metrics['repaired_optimality_gap']):.4f} "
            f"feasible={metrics['raw_feasible']} "
            f"attempts={metrics['total_attempts']} "
            f"time={float(str(metrics['total_time']).rstrip('s')):.1f}s"
        )

    generate_plots(results_path, plots_dir)


def generate_plots(results_path: Path, plots_dir: Path) -> None:
    section("Generating Quicktest Plots")

    try:
        from analysis import load_results, make_progress_plot, print_summary
        import matplotlib.pyplot as plt
    except Exception as exc:
        skip("Plot generation", f"matplotlib/analysis unavailable: {exc}")
        return

    if not results_path.exists():
        skip("Plot generation", f"results file not found: {results_path}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis_data = load_results(results_path, metric="optimality_gap")
    print_summary(analysis_data)
    make_progress_plot(
        analysis_data,
        output_path=plots_dir / "instance_progress.png",
        title="Knapsack Quicktest Instance Diagnostics",
    )

    rows = []
    with results_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    if rows:
        labels = [str(row["description"]).strip() for row in rows]
        raw_scores = [float(row["optimality_gap"]) for row in rows]
        repaired_scores = [float(row["repaired_optimality_gap"]) for row in rows]

        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, raw_scores, width=width, label="raw gap")
        plt.bar(x + width / 2, repaired_scores, width=width, label="repaired gap")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Optimality gap")
        plt.title("Raw vs repaired knapsack quicktest gaps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "raw_vs_repaired_gap.png", dpi=200)
        plt.close()

    check("Plot generation completed", True)
    print(f"  Plots saved to {plots_dir}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="Knapsack quicktest")
    parser.add_argument("--size", type=int, default=6, help="Knapsack size (num items)")
    parser.add_argument("--phase", type=int, default=0, help="Run only phase N (1-4)")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()

    print("=" * 68)
    print("AutoQResearch Knapsack Quicktest")
    print("=" * 68)
    print(f"Scope: Knapsack (n_items={args.size}), ideal_mps, QAOA variants/VQE/QRAO/PCE smokes")
    print(f"       Adaptive multi-attempt loop with AttemptOutcome")

    started = time.time()

    try:
        if args.plots_only:
            generate_plots(args.results_file, args.plots_dir)
            total = time.time() - started
            print(f"\nFinished in {total:.1f}s")
            return 0

        if args.phase in (0, 1):
            phase1_unit_tests(args.size)

        if args.phase in (0, 2):
            phase2_solver_tests(args.size)

        if args.phase in (0, 3):
            phase3_end_to_end(args.size)

        if args.phase in (0, 4):
            phase4_cli_and_plots(args.size, args.results_file, args.plots_dir)

    except Exception as exc:
        traceback.print_exc()
        print(f"\nFatal quicktest failure: {exc}")
        return 1

    total = time.time() - started
    print("\n" + "=" * 68)
    print(f"Results: {PASS} passed, {FAIL} failed, {WARN} warnings, {SKIP} skipped")
    print(f"Total time: {total:.1f}s")
    print("STATUS: PASS" if FAIL == 0 else "STATUS: FAIL")
    print("=" * 68)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
