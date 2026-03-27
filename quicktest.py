#!/usr/bin/env python3
"""
quicktest.py - Comprehensive smoke test for the current AutoQResearch stack.

LEGACY / NOT USED FOR KNAPSACK POLICY OBJECTIVE.

Current scope:
  - MaxCut-first
  - ideal_mps / statevector alias
  - QAOA, CVaR-QAOA, WS-QAOA, MA-QAOA
  - VQE, CVaR-VQE
  - QRAO
  - PCE with k in {2, 3}

Future-only paths such as noisy simulator and hardware are checked as expected
NotImplementedError cases.

Usage:
    ./.venv/bin/python quicktest.py
    ./.venv/bin/python quicktest.py --problem maxcut_10
    ./.venv/bin/python quicktest.py --phase 2
    ./.venv/bin/python quicktest.py --plots-only
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


PASS = 0
FAIL = 0
WARN = 0
SKIP = 0

DEFAULT_RESULTS_PATH = Path("quicktest_results.tsv")
DEFAULT_PLOTS_DIR = Path("quicktest_figures")


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


def parse_problem_spec(spec: str) -> tuple[str, int, int]:
    parts = spec.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid problem spec: {spec}")
    problem_type = parts[0]
    size = int(parts[1])
    seed = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("s") else 0
    return problem_type, size, seed


def require_maxcut_problem(spec: str) -> tuple[int, int]:
    problem_type, size, seed = parse_problem_spec(spec)
    if problem_type != "maxcut":
        raise ValueError(
            "quicktest.py is aligned with the current MaxCut-first codebase. "
            f"Received problem type '{problem_type}'."
        )
    return size, seed


def make_backend(estimator_shots: int = 128, sampler_shots: int = 128, mode: str = "ideal_mps"):
    from autoqresearch.backends.factory import BackendConfig, create_execution_context

    return create_execution_context(
        BackendConfig(
            mode=mode,
            shots=estimator_shots,
            sampler_shots=sampler_shots,
        )
    )


def phase1_unit_tests(problem_spec: str) -> None:
    section("Phase 1: Unit Tests")

    from autoqresearch.backends.factory import BackendConfig, create_execution_context
    from autoqresearch.evaluation.evaluator import Evaluator
    from autoqresearch.problems.registry import generate_split, get_single_instance
    from autoqresearch.solvers.base import SolverResult, extract_best_solution
    from autoqresearch.solvers.maxcut_primitives import compute_cvar_from_counts as compute_cvar_objective

    size, seed = require_maxcut_problem(problem_spec)

    print("\n  [Problem generation]")
    problem = get_single_instance("maxcut", size, seed=seed)
    check("MaxCut instance created", problem is not None)
    check("MaxCut has expected size", problem.num_variables == size, f"got {problem.num_variables}")
    check("MaxCut optimal value is positive", problem.optimal_value > 0.0, f"got {problem.optimal_value}")
    check("MaxCut graph metadata exists", problem.metadata.get("graph") is not None)

    graph = problem.metadata["graph"]
    manual_max = 0.0
    for integer in range(2 ** size):
        bitstring = [(integer >> bit) & 1 for bit in range(size)]
        cut = sum(
            graph[u][v].get("weight", 1.0)
            for u, v in graph.edges()
            if bitstring[u] != bitstring[v]
        )
        manual_max = max(manual_max, float(cut))
    check(
        "MaxCut brute force matches manual cut search",
        abs(problem.optimal_value - manual_max) < 1e-9,
        f"stored={problem.optimal_value} manual={manual_max}",
    )
    candidate_counts = {
        "".join(str(int(bit)) for bit in np.zeros(problem.num_variables, dtype=int)[::-1]): 1,
        "".join(str(int(bit)) for bit in problem.optimal_solution.astype(int)[::-1]): 1,
    }
    selected_bits, selected_obj, _ = extract_best_solution(candidate_counts, problem)
    check(
        "Best-solution extraction respects MAXIMIZE sense",
        np.array_equal(selected_bits.astype(int), problem.optimal_solution.astype(int)),
        f"selected={selected_bits.astype(int).tolist()} expected={problem.optimal_solution.astype(int).tolist()}",
    )
    check(
        "Best-solution extraction returns optimal MaxCut objective",
        abs(selected_obj - problem.objective_value(problem.optimal_solution)) < 1e-9,
        f"selected={selected_obj} expected={problem.objective_value(problem.optimal_solution)}",
    )
    cvar_counts = {
        "".join(str(int(bit)) for bit in np.zeros(problem.num_variables, dtype=int)[::-1]): 1,
        "".join(str(int(bit)) for bit in problem.optimal_solution.astype(int)[::-1]): 1,
    }
    cvar_objective = compute_cvar_objective(problem, cvar_counts, alpha=0.5)
    check(
        "CVaR objective uses the maximizing tail for MaxCut",
        abs(cvar_objective + problem.objective_value(problem.optimal_solution)) < 1e-9,
        f"got={cvar_objective} expected={-problem.objective_value(problem.optimal_solution)}",
    )

    print("\n  [Constrained decoding]")
    knapsack = get_single_instance("knapsack", 8, seed=0)
    check("Knapsack instance created", knapsack is not None)
    check("Knapsack QUBO adds slack variables", knapsack.num_variables > 8, f"got {knapsack.num_variables}")
    decoded_zero = knapsack.decode_solution(np.zeros(knapsack.num_variables))
    check("Knapsack decoder returns original item width", len(decoded_zero) == 8, f"got {len(decoded_zero)}")
    check("Knapsack zero solution is feasible", knapsack.is_feasible(np.zeros(knapsack.num_variables)))
    overweight = np.concatenate([np.ones(8), np.zeros(knapsack.num_variables - 8)])
    check("Knapsack overweight solution is infeasible", not knapsack.is_feasible(overweight))
    check("Knapsack decoded objective is non-negative", knapsack.original_objective_value(overweight) >= 0.0)

    print("\n  [Evaluator]")
    evaluator = Evaluator()
    reference_bits = (
        problem.optimal_solution.copy()
        if problem.optimal_solution is not None
        else np.zeros(problem.num_variables)
    )
    fake_result = SolverResult(
        best_bitstring=reference_bits,
        best_objective=float(problem.objective_value(reference_bits)),
        is_feasible=True,
        counts={"0" * problem.num_variables: 10},
        num_shots=10,
        num_qubits=problem.num_variables,
    )
    evaluation = evaluator.evaluate(fake_result, problem)
    check("Evaluator returned finite score", math.isfinite(evaluation.composite_score))
    check(
        "Evaluator approximation ratio stays in [0,1]",
        0.0 <= evaluation.approximation_ratio <= 1.0,
        f"got {evaluation.approximation_ratio}",
    )

    print("\n  [Splits]")
    split = generate_split("maxcut", size, instances_per_split=2)
    check("Train split length", len(split.train) == 2, f"got {len(split.train)}")
    check("Dev split length", len(split.dev) == 2, f"got {len(split.dev)}")
    check("Test split length", len(split.test) == 2, f"got {len(split.test)}")
    check("Train and dev names differ", split.train[0].name != split.dev[0].name)

    print("\n  [Backend creation]")
    context = create_execution_context(BackendConfig(mode="ideal_mps", shots=64, sampler_shots=64))
    check("ideal_mps backend created", context is not None)
    check("ideal_mps mode set", context.mode == "ideal_mps", f"got {context.mode}")

    alias_context = create_execution_context(BackendConfig(mode="statevector", shots=64, sampler_shots=64))
    check("statevector alias maps to ideal_mps", alias_context.mode == "ideal_mps", f"got {alias_context.mode}")

    try:
        create_execution_context(BackendConfig(mode="noisy_simulator", shots=64, sampler_shots=64))
        check("noisy_simulator placeholder raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        check("noisy_simulator placeholder raises NotImplementedError", True)

    try:
        create_execution_context(BackendConfig(mode="hardware", shots=64, sampler_shots=64))
        check("hardware placeholder raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        check("hardware placeholder raises NotImplementedError", True)


def run_solver_test(name, solver, problem, policy, backend):
    from autoqresearch.evaluation.evaluator import Evaluator

    try:
        t0 = time.time()
        result = solver.solve(problem, policy, backend, shots=policy.get("shots", 128))
        elapsed = time.time() - t0
        evaluation = Evaluator().evaluate(result, problem)

        conditions = [
            math.isfinite(evaluation.composite_score),
            math.isfinite(evaluation.approximation_ratio),
            0.0 <= evaluation.approximation_ratio <= 1.0,
            result.num_qubits > 0,
            result.num_parameters >= 0,
            result.num_shots >= 0,
            isinstance(result.counts, dict),
        ]
        ok = all(conditions)
        detail = (
            f"AR={evaluation.approximation_ratio:.3f} "
            f"score={evaluation.composite_score:.4f} "
            f"depth={evaluation.circuit_depth} "
            f"2q={evaluation.two_qubit_gate_count} "
            f"qubits={evaluation.num_qubits} "
            f"params={result.num_parameters} "
            f"time={elapsed:.1f}s"
        )
        check(name, ok, detail if not ok else "")
        print(f"       {detail}")
        return evaluation, result, elapsed
    except Exception as exc:
        check(name, False, f"EXCEPTION: {exc}")
        traceback.print_exc()
        return None, None, None


def phase2_solver_tests(problem_spec: str):
    section("Phase 2: Solver Tests")

    from autoqresearch.problems.registry import get_single_instance
    from autoqresearch.solvers.pce_solver import PCESolver
    from autoqresearch.solvers.qaoa_family import QAOAFamilySolver
    from autoqresearch.solvers.qrao_solver import QRAOSolver
    from autoqresearch.solvers.vqe_family import VQEFamilySolver

    size, seed = require_maxcut_problem(problem_spec)
    problem = get_single_instance("maxcut", size, seed=seed)
    backend = make_backend(estimator_shots=128, sampler_shots=128)

    print(f"\n  Problem: {problem.name} (opt={problem.optimal_value}, n={problem.num_variables})\n")

    qaoa = QAOAFamilySolver()
    vqe = VQEFamilySolver()
    qrao = QRAOSolver()
    pce = PCESolver()

    cases = [
        (
            "QAOA standard p=1",
            qaoa,
            {
                "variant": "standard",
                "reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "initialization": "pi_over_2",
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "QAOA standard pauli_evolution",
            qaoa,
            {
                "variant": "standard",
                "reps": 1,
                "circuit_type": "pauli_evolution",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "CVaR-QAOA alpha=0.25",
            qaoa,
            {
                "variant": "cvar",
                "reps": 1,
                "cvar_alpha": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "WS-QAOA greedy",
            qaoa,
            {
                "variant": "warmstart",
                "reps": 1,
                "ws_source": "greedy",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "WS-QAOA sdp",
            qaoa,
            {
                "variant": "warmstart",
                "reps": 1,
                "ws_source": "sdp",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "MA-QAOA degree tying",
            qaoa,
            {
                "variant": "multiangle",
                "reps": 1,
                "ma_tying": "degree",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 40,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "VQE efficient_su2",
            vqe,
            {
                "variant": "standard",
                "ansatz_type": "efficient_su2",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 40,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "CVaR-VQE alpha=0.25",
            vqe,
            {
                "variant": "cvar",
                "ansatz_type": "efficient_su2",
                "vqe_reps": 1,
                "cvar_alpha": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 40,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "QRAO qrac=2 semideterministic",
            qrao,
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "QRAO qrac=3 magic",
            qrao,
            {
                "qrac_type": 3,
                "rounding": "magic",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "PCE k=2",
            pce,
            {
                "pce_k": 2,
                "pce_depth": 2,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 20,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "PCE k=3",
            pce,
            {
                "pce_k": 3,
                "pce_depth": 2,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 20,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
    ]

    rows = []
    for name, solver, policy in cases:
        evaluation, result, elapsed = run_solver_test(name, solver, problem, policy, backend)
        if evaluation is not None:
            rows.append((name, evaluation, result, elapsed))
    return rows


def phase3_edge_cases(problem_spec: str) -> None:
    section("Phase 3: Edge Cases and Expected Failures")

    from autoqresearch.backends.factory import BackendConfig, create_execution_context
    from autoqresearch.problems.registry import get_single_instance
    from autoqresearch.solvers.pce_solver import PCESolver
    from autoqresearch.solvers.qaoa_family import QAOAFamilySolver
    from autoqresearch.solvers.qrao_solver import QRAOSolver
    from autoqresearch.solvers.vqe_family import VQEFamilySolver

    size, seed = require_maxcut_problem(problem_spec)
    maxcut = get_single_instance("maxcut", size, seed=seed)
    mis = get_single_instance("mis", 8, seed=0)
    mdkp = get_single_instance("mdkp", 10, seed=0)
    backend = make_backend(estimator_shots=64, sampler_shots=64)

    qaoa = QAOAFamilySolver()
    vqe = VQEFamilySolver()
    qrao = QRAOSolver()
    pce = PCESolver()

    print("\n  [Unsupported problem families]")
    try:
        qaoa.solve(mis, {"variant": "standard", "reps": 1, "shots": 64}, backend, shots=64)
        check("QAOA on MIS raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        check("QAOA on MIS raises NotImplementedError", True)

    try:
        vqe.solve(mdkp, {"variant": "standard", "vqe_reps": 1, "shots": 64}, backend, shots=64)
        check("VQE on MDKP raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        check("VQE on MDKP raises NotImplementedError", True)

    print("\n  [Invalid policy values]")
    try:
        qaoa.solve(
            maxcut,
            {"variant": "standard", "reps": 1, "optimizer_method": "NOPE", "shots": 64},
            backend,
            shots=64,
        )
        check("Invalid optimizer raises ValueError", False, "did not raise")
    except ValueError:
        check("Invalid optimizer raises ValueError", True)

    try:
        vqe.solve(
            maxcut,
            {"variant": "standard", "ansatz_type": "not_an_ansatz", "shots": 64},
            backend,
            shots=64,
        )
        check("Invalid ansatz raises ValueError", False, "did not raise")
    except ValueError:
        check("Invalid ansatz raises ValueError", True)

    try:
        qrao.solve(
            maxcut,
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "optimizer_method": "NOPE",
                "shots": 64,
            },
            backend,
            shots=64,
        )
        check("Invalid QRAO optimizer raises ValueError", False, "did not raise")
    except ValueError:
        check("Invalid QRAO optimizer raises ValueError", True)

    try:
        pce.solve(
            maxcut,
            {"pce_k": 4, "pce_depth": 2, "shots": 64},
            backend,
            shots=64,
        )
        check("Invalid PCE k raises ValueError", False, "did not raise")
    except ValueError:
        check("Invalid PCE k raises ValueError", True)

    print("\n  [Backend placeholders]")
    try:
        create_execution_context(BackendConfig(mode="noisy_simulator", shots=64, sampler_shots=64))
        check("Noisy simulator placeholder raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        check("Noisy simulator placeholder raises NotImplementedError", True)

    try:
        create_execution_context(BackendConfig(mode="hardware", shots=64, sampler_shots=64))
        check("Hardware placeholder raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        check("Hardware placeholder raises NotImplementedError", True)


def write_tsv_header(path: Path) -> None:
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
                "composite_score",
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
        )


def append_tsv_row(
    path: Path,
    experiment_id: int,
    evaluation,
    result,
    status: str,
    description: str,
    elapsed: float,
) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                experiment_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                evaluation.problem_name,
                evaluation.solver_name,
                status,
                description,
                f"{evaluation.composite_score:.6f}",
                f"{evaluation.approximation_ratio:.4f}",
                int(evaluation.is_feasible),
                f"{evaluation.feasibility_rate:.4f}",
                evaluation.circuit_depth,
                evaluation.cnot_count,
                evaluation.two_qubit_gate_count,
                evaluation.total_gate_count,
                evaluation.num_qubits,
                result.num_parameters,
                result.optimizer_iterations,
                f"{elapsed:.2f}",
            ]
        )


def phase4_end_to_end(problem_spec: str, results_path: Path):
    section("Phase 4: End-to-End Quickrun")

    from autoqresearch.evaluation.evaluator import Evaluator
    from autoqresearch.problems.registry import get_single_instance
    from autoqresearch.solvers.pce_solver import PCESolver
    from autoqresearch.solvers.qaoa_family import QAOAFamilySolver
    from autoqresearch.solvers.qrao_solver import QRAOSolver
    from autoqresearch.solvers.vqe_family import VQEFamilySolver

    size, seed = require_maxcut_problem(problem_spec)
    problem = get_single_instance("maxcut", size, seed=seed)
    backend = make_backend(estimator_shots=128, sampler_shots=128)
    evaluator = Evaluator()

    experiments = [
        (
            "QAOA p=1",
            QAOAFamilySolver(),
            {
                "variant": "standard",
                "reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "QAOA (interp init)",
            QAOAFamilySolver(),
            {
                "variant": "standard",
                "reps": 2,
                "initialization": "interp",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "CVaR QAOA",
            QAOAFamilySolver(),
            {
                "variant": "cvar",
                "reps": 1,
                "cvar_alpha": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "WS QAOA",
            QAOAFamilySolver(),
            {
                "variant": "warmstart",
                "reps": 1,
                "ws_source": "greedy",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 8,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "MA QAOA",
            QAOAFamilySolver(),
            {
                "variant": "multiangle",
                "reps": 1,
                "ma_tying": "degree",
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 40,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "VQE",
            VQEFamilySolver(),
            {
                "variant": "standard",
                "ansatz_type": "efficient_su2",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 40,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "CVaR VQE",
            VQEFamilySolver(),
            {
                "variant": "cvar",
                "ansatz_type": "efficient_su2",
                "vqe_reps": 1,
                "cvar_alpha": 0.25,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 40,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "QRAO (2,1)",
            QRAOSolver(),
            {
                "qrac_type": 2,
                "rounding": "semideterministic",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "QRAO (3,1)",
            QRAOSolver(),
            {
                "qrac_type": 3,
                "rounding": "semideterministic",
                "vqe_reps": 1,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 12,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "PCE (2)",
            PCESolver(),
            {
                "pce_k": 2,
                "pce_depth": 2,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 20,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
        (
            "PCE (3)",
            PCESolver(),
            {
                "pce_k": 3,
                "pce_depth": 2,
                "optimizer_method": "COBYLA",
                "optimizer_maxiter": 20,
                "shots": 128,
                "estimator_shots": 128,
                "sampler_shots": 128,
            },
        ),
    ]

    write_tsv_header(results_path)
    best_score = float("-inf")
    rows = []

    print(f"\n  Running {len(experiments)} quick experiments on {problem.name}\n")
    for experiment_id, (description, solver, policy) in enumerate(experiments, start=1):
        sys.stdout.write(f"  [{experiment_id:02d}/{len(experiments):02d}] {description:<18s} ")
        sys.stdout.flush()
        try:
            t0 = time.time()
            result = solver.solve(problem, policy, backend, shots=policy.get("shots", 128))
            elapsed = time.time() - t0
            evaluation = evaluator.evaluate(result, problem)
            improved = evaluation.composite_score > best_score
            status = "keep" if improved else "discard"
            marker = "*" if improved else " "
            if improved:
                best_score = evaluation.composite_score
            print(
                f"score={evaluation.composite_score:.4f} "
                f"AR={evaluation.approximation_ratio:.3f} "
                f"2q={evaluation.two_qubit_gate_count:3d} "
                f"depth={evaluation.circuit_depth:3d} "
                f"time={elapsed:.1f}s {marker}"
            )
            append_tsv_row(
                results_path,
                experiment_id,
                evaluation,
                result,
                status,
                description,
                elapsed,
            )
            rows.append(
                {
                    "description": description,
                    "evaluation": evaluation,
                    "result": result,
                    "elapsed": elapsed,
                }
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            warn(f"End-to-end experiment failed: {description}", str(exc))

    print(f"\n  Results written to {results_path}")
    return rows


def load_rows_from_tsv(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def _pick_annotation_offsets(xs, ys):
    patterns = [
        (8, 10),
        (8, -10),
        (18, 10),
        (-42, 10),
        (18, -10),
        (-42, -10),
        (8, 20),
        (-48, 20),
    ]
    if not xs:
        return []

    x_span = max(xs) - min(xs) if len(xs) > 1 else 0.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 0.0
    x_threshold = max(1.0, x_span * 0.12)
    y_threshold = max(0.01, y_span * 0.12)

    offsets = []
    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        crowding = 0
        for other_index, (other_x, other_y) in enumerate(zip(xs, ys)):
            if other_index == index:
                continue
            if abs(other_x - x_value) <= x_threshold and abs(other_y - y_value) <= y_threshold:
                crowding += 1
        offsets.append(patterns[(crowding + index) % len(patterns)])
    return offsets


def _relax_annotations(
    ax,
    annotations,
    step: float = 6.0,
    iterations: int = 60,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    if not annotations:
        return

    figure = ax.figure
    for _ in range(iterations):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = [
            annotation.get_window_extent(renderer=renderer).expanded(1.03, 1.18)
            for annotation in annotations
        ]

        moved = False
        for index in range(len(annotations)):
            for other_index in range(index):
                if not boxes[index].overlaps(boxes[other_index]):
                    continue
                offset_x, offset_y = annotations[index].get_position()
                direction = 1 if (index - other_index) % 2 == 0 else -1
                next_y = offset_y + direction * step
                next_x = offset_x
                if x_bounds is not None:
                    next_x = min(max(next_x, x_bounds[0]), x_bounds[1])
                if y_bounds is not None:
                    next_y = min(max(next_y, y_bounds[0]), y_bounds[1])
                annotations[index].set_position((next_x, next_y))
                moved = True

        if not moved:
            break


def _annotate_points(ax, xs, ys, labels, fontsize: float = 8.0) -> None:
    offsets = _pick_annotation_offsets(xs, ys)
    annotations = []
    for (x_value, y_value, label), (offset_x, offset_y) in zip(zip(xs, ys, labels), offsets):
        annotation = ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=fontsize,
            alpha=0.9,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.8},
            arrowprops={"arrowstyle": "-", "color": "#999999", "alpha": 0.45, "lw": 0.7},
        )
        annotations.append(annotation)

    _relax_annotations(
        ax,
        annotations,
        step=4.0,
        iterations=40,
        x_bounds=(-48.0, 22.0),
        y_bounds=(-12.0, 24.0),
    )


def _annotate_score_trace(ax, xs, ys, labels) -> None:
    if not xs:
        return

    y_span = max(ys) - min(ys) if len(ys) > 1 else 0.0
    y_threshold = max(0.02, y_span * 0.08)
    y_offsets = [10, 18, 26, 34]
    annotations = []

    for index, (x_value, y_value, label) in enumerate(zip(xs, ys, labels)):
        crowding = 0
        for other_index, (other_x, other_y) in enumerate(zip(xs, ys)):
            if other_index == index:
                continue
            if abs(other_x - x_value) <= 1 and abs(other_y - y_value) <= y_threshold:
                crowding += 1

        annotation = ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, y_offsets[(crowding + index) % len(y_offsets)]),
            fontsize=8.0,
            rotation=24,
            alpha=0.9,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.8},
            arrowprops={"arrowstyle": "-", "color": "#999999", "alpha": 0.4, "lw": 0.7},
        )
        annotations.append(annotation)

    _relax_annotations(ax, annotations, step=4.0, iterations=40)


def _best_solver_label(description: str, solver_name: str) -> str:
    normalized = description.strip().lower()

    if normalized.startswith("cvar qaoa"):
        return "CVaR QAOA"
    if normalized.startswith("ws qaoa"):
        return "WS QAOA"
    if normalized.startswith("ma qaoa"):
        return "MA QAOA"
    if normalized.startswith("cvar vqe"):
        return "CVaR VQE"
    if normalized.startswith("qrao (2,1)"):
        return "QRAO (2,1)"
    if normalized.startswith("qrao (3,1)"):
        return "QRAO (3,1)"
    if normalized.startswith("pce (2)"):
        return "PCE (2)"
    if normalized.startswith("pce (3)"):
        return "PCE (3)"
    if normalized.startswith("qaoa"):
        return "QAOA"
    if normalized.startswith("vqe"):
        return "VQE"

    fallback = solver_name.replace("_", " ").strip()
    return fallback or description


def generate_plots(results_path: Path, plots_dir: Path) -> None:
    section("Generating Quicktest Plots")

    try:
        from analysis import load_results, make_progress_plot, print_summary
        import matplotlib.pyplot as plt
    except Exception as exc:
        skip("Plot generation", f"matplotlib unavailable: {exc}")
        return

    rows = load_rows_from_tsv(results_path)
    if not rows:
        skip("Plot generation", f"no rows found in {results_path}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis_data = load_results(results_path, metric="composite_score")
    print_summary(analysis_data)
    make_progress_plot(
        analysis_data,
        output_path=plots_dir / "progress.png",
        title="Quicktest Progress",
    )

    experiment_ids = [int(row["experiment_id"]) for row in rows]
    scores = [float(row["composite_score"]) for row in rows]
    approx = [float(row["approx_ratio"]) for row in rows]
    two_qubit = [int(row["two_qubit_gates"]) for row in rows]
    labels = [row["description"] for row in rows]
    solver_names = [row["solver"] for row in rows]

    score_fig, score_ax = plt.subplots(figsize=(12, 6))
    score_ax.plot(experiment_ids, scores, marker="o")
    _annotate_score_trace(score_ax, experiment_ids, scores, labels)
    score_ax.set_xlabel("Experiment")
    score_ax.set_ylabel("Composite score")
    score_ax.set_title("Quicktest score progression")
    score_ax.grid(True, alpha=0.3)
    score_span = max(scores) - min(scores) if len(scores) > 1 else 0.0
    score_margin = max(0.08, score_span * 0.22)
    score_ax.set_ylim(min(scores) - score_margin * 0.25, max(scores) + score_margin)
    score_fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.88)
    score_fig.savefig(plots_dir / "score_trace.png", dpi=200, bbox_inches="tight")
    plt.close(score_fig)

    approx_fig, approx_ax = plt.subplots(figsize=(12, 6))
    approx_ax.scatter(two_qubit, approx)
    _annotate_points(approx_ax, two_qubit, approx, labels, fontsize=8.0)
    approx_ax.set_xlabel("Two-qubit gates")
    approx_ax.set_ylabel("Approximation ratio")
    approx_ax.set_title("Approximation ratio vs two-qubit gates")
    approx_ax.grid(True, alpha=0.3)
    approx_ax.margins(x=0.1, y=0.12)
    approx_fig.subplots_adjust(left=0.1, right=0.98, bottom=0.14, top=0.88)
    approx_fig.savefig(plots_dir / "approx_vs_two_qubit.png", dpi=200, bbox_inches="tight")
    plt.close(approx_fig)

    best_by_solver = {}
    for description, solver_name, score in zip(labels, solver_names, scores):
        display_name = _best_solver_label(description, solver_name)
        best_by_solver[display_name] = max(score, best_by_solver.get(display_name, float("-inf")))

    preferred_order = [
        "QAOA",
        "CVaR QAOA",
        "WS QAOA",
        "MA QAOA",
        "VQE",
        "CVaR VQE",
        "QRAO (2,1)",
        "QRAO (3,1)",
        "PCE (2)",
        "PCE (3)",
    ]

    plt.figure(figsize=(10, 5))
    solver_keys = [key for key in preferred_order if key in best_by_solver]
    solver_scores = [best_by_solver[key] for key in solver_keys]
    plt.bar(solver_keys, solver_scores)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Best composite score")
    plt.title("Best score by solver")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_by_solver.png", dpi=200)
    plt.close()

    check("Plot generation completed", True)
    print(f"  Plots saved to {plots_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoQResearch quick smoke test")
    parser.add_argument("--problem", type=str, default="maxcut_8")
    parser.add_argument("--phase", type=int, default=0, help="Run only phase N (1-4)")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()

    print("=" * 68)
    print("AutoQResearch Quicktest")
    print("=" * 68)
    print("Scope: MaxCut-first, ideal_mps, primitive-based solvers")

    started = time.time()

    if args.plots_only:
        generate_plots(args.results_file, args.plots_dir)
        total = time.time() - started
        print(f"\nFinished in {total:.1f}s")
        return 0

    try:
        if args.phase in (0, 1):
            phase1_unit_tests(args.problem)

        if args.phase in (0, 2):
            phase2_solver_tests(args.problem)

        if args.phase in (0, 3):
            phase3_edge_cases(args.problem)

        if args.phase in (0, 4):
            phase4_end_to_end(args.problem, args.results_file)
            generate_plots(args.results_file, args.plots_dir)
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
