#!/usr/bin/env python3
"""Run the retained E-n13 CVRP policy on IBM quantum hardware."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment as exp
from autoqresearch.problems.base import ProblemInstance
from autoqresearch.problems.registry import get_cvrp_file_instance
from autoqresearch.problems.cvrp import (
    build_reduced_gap_qubo,
    clusters_capacity_feasible,
    identify_ambiguous_customers,
    solve_cvrp_routes_classically,
    solve_gap_greedy,
)
from autoqresearch.solvers.qubo_primitives import (
    check_cvrp_feasibility,
    compute_cvrp_best_feasible_ar,
    compute_cvrp_feasibility_rate,
    cvrp_objective_value,
)

from autoq_hardware_backend import (
    AutoQHardwareBackendFactory,
    patch_autoq_primitives,
)
from static_cvrp_policies import (
    get_cvrp_e13_policy,
    get_cvrp_e13_policy_note,
)


LOGGER = logging.getLogger("hardware_runs.run_cvrp_e13_hardware")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results_hardware"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints_autoq"
DEFAULT_CREDENTIALS_JSON = SCRIPT_DIR / "ibm_credentials.template.json"
DEFAULT_INSTANCE = "E-n13-k4"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _configure_logging(*, run_dir: Path, log_level_name: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    level = getattr(logging, str(log_level_name).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level '{log_level_name}'.")
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    for noisy_logger in (
        "qiskit.transpiler.passes",
        "qiskit.transpiler.runningpassmanager",
        "qiskit.transpiler.passmanager",
        "qiskit.compiler.transpiler",
        "qiskit_ibm_runtime.base_primitive",
        "management.get",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    return log_path


def _has_placeholder_credentials(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    markers = ("YOUR_TOKEN_", "YOUR_CRN_", "YOUR_INSTANCE_")
    return any(marker in text for marker in markers)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(exp._json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(exp._json_safe(record), sort_keys=True) + "\n")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read checkpoint %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(exp._json_safe(checkpoint), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _dump_qubo_lp(problem, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(problem.qubo, "export_as_lp_string"):
        text = problem.qubo.export_as_lp_string()
    else:
        text = str(problem.qubo.prettyprint())
    path.write_text(str(text), encoding="utf-8")


def _load_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy = get_cvrp_e13_policy()
    if args.policy_file or args.policy_json:
        policy_file = Path(args.policy_file).expanduser().resolve() if args.policy_file else None
        override = exp._load_policy_override(policy_file, args.policy_json)
        policy = exp._merge_policy_override(policy, override)
    if args.estimator_shots is not None:
        policy["estimator_shots"] = int(args.estimator_shots)
    if args.sampler_shots is not None:
        policy["sampler_shots"] = int(args.sampler_shots)
    if args.optimizer_maxiter is not None:
        policy["optimizer_maxiter"] = int(args.optimizer_maxiter)
    if args.hybrid_ambiguity_threshold is not None:
        policy["hybrid_ambiguity_threshold"] = float(args.hybrid_ambiguity_threshold)
    if args.seed is not None:
        policy["seed"] = int(args.seed)

    policy["gap_solver_family"] = str(policy.get("gap_solver_family", "hybrid")).lower()
    policy["solver_family"] = str(policy.get("solver_family", policy["gap_solver_family"])).lower()
    policy["route_solver_family"] = str(policy.get("route_solver_family", "classical")).lower()
    policy["pce_local_search"] = False
    policy["final_local_search"] = False
    return policy


def _load_problem(policy: dict[str, Any], instance_name: str) -> ProblemInstance:
    return get_cvrp_file_instance(
        instance_name,
        capacity_method=str(policy.get("cvrp_gap_penalty_method", "tilted")),
        seed_method=str(policy.get("cvrp_seed_method", "depot_farthest")),
        gap_penalty=policy.get("penalty"),
        taylor_alpha=float(policy.get("cvrp_taylor_alpha", 10.0)),
        tilted_kappa=float(policy.get("cvrp_tilted_kappa", 5.0)),
        tilted_s_frac=float(policy.get("cvrp_tilted_s_frac", 0.10)),
        tilted_s_min=float(policy.get("cvrp_tilted_s_min", 1.0)),
    )


def _make_reduced_gap_plan(problem: ProblemInstance, policy: dict[str, Any]) -> dict[str, Any]:
    instance = problem.metadata["instance"]
    seeds = list(problem.metadata["seeds"])
    clusters = solve_gap_greedy(instance, seeds)
    ambiguous, fixed = identify_ambiguous_customers(
        instance,
        seeds,
        clusters,
        threshold=float(policy.get("hybrid_ambiguity_threshold", 0.5)),
    )
    route_solutions = (
        solve_cvrp_routes_classically(instance, clusters)
        if clusters_capacity_feasible(instance, clusters)
        else []
    )
    greedy_cost = float(sum(route["cost"] for route in route_solutions)) if route_solutions else None
    greedy_gap = (
        None
        if greedy_cost is None
        else exp._compute_optimality_gap(
            float(problem.optimal_value) / max(greedy_cost, 1e-10),
            True,
        )
    )

    reduced = None
    if ambiguous:
        reduced = build_reduced_gap_qubo(
            instance,
            seeds,
            fixed,
            ambiguous,
            capacity_method=str(policy.get("cvrp_gap_penalty_method", "tilted")),
            gap_penalty=policy.get("penalty"),
            taylor_alpha=float(policy.get("cvrp_taylor_alpha", 10.0)),
            tilted_kappa=float(policy.get("cvrp_tilted_kappa", 5.0)),
            tilted_s_frac=float(policy.get("cvrp_tilted_s_frac", 0.10)),
            tilted_s_min=float(policy.get("cvrp_tilted_s_min", 1.0)),
        )

    return {
        "instance": problem.name,
        "source_file": problem.metadata.get("source_file"),
        "full_gap_qubo_variables": int(problem.num_variables),
        "optimal_reference_cost": float(problem.optimal_value),
        "num_customers": int(problem.metadata.get("num_customers", 0)),
        "num_vehicles": int(problem.metadata.get("num_vehicles", 0)),
        "capacity": int(problem.metadata.get("capacity", 0)),
        "total_demand": int(problem.metadata.get("total_demand", 0)),
        "seed_method": problem.metadata.get("seed_method"),
        "seeds": [int(seed) for seed in seeds],
        "classical_greedy_clusters": [[int(customer) for customer in cluster] for cluster in clusters],
        "classical_greedy_capacity_feasible": bool(clusters_capacity_feasible(instance, clusters)),
        "classical_greedy_route_cost": greedy_cost,
        "classical_greedy_gap": greedy_gap,
        "classical_greedy_route_solutions": route_solutions,
        "hybrid_ambiguity_threshold": float(policy.get("hybrid_ambiguity_threshold", 0.5)),
        "ambiguous_customers": [int(customer) for customer in ambiguous],
        "ambiguous_customer_count": len(ambiguous),
        "fixed_assignments": {str(customer): int(vehicle) for customer, vehicle in fixed.items()},
        "reduced_gap_qubo_variables": None if reduced is None else int(reduced["qubo"].get_num_vars()),
        "reduced_capacity": None if reduced is None else [int(value) for value in reduced["reduced_capacity"]],
        "will_submit_quantum_reduced_gap": bool(
            ambiguous
            and len(ambiguous) * int(problem.metadata.get("num_vehicles", 0)) <= 30
            and str(policy.get("gap_solver_family", "")).lower() == "hybrid"
        ),
    }


def _persist_plan_artifacts(inst_dir: Path, problem: ProblemInstance, plan: dict[str, Any], policy: dict[str, Any]) -> None:
    _write_json(inst_dir / "hardware_plan.json", plan)
    _write_json(inst_dir / "policy.json", exp.snapshot_policy(policy))
    _dump_qubo_lp(problem, inst_dir / "full_gap_qubo.lp")

    if plan.get("ambiguous_customers"):
        instance = problem.metadata["instance"]
        reduced = build_reduced_gap_qubo(
            instance,
            list(problem.metadata["seeds"]),
            {int(k): int(v) for k, v in plan.get("fixed_assignments", {}).items()},
            [int(customer) for customer in plan["ambiguous_customers"]],
            capacity_method=str(policy.get("cvrp_gap_penalty_method", "tilted")),
            gap_penalty=policy.get("penalty"),
            taylor_alpha=float(policy.get("cvrp_taylor_alpha", 10.0)),
            tilted_kappa=float(policy.get("cvrp_tilted_kappa", 5.0)),
            tilted_s_frac=float(policy.get("cvrp_tilted_s_frac", 0.10)),
            tilted_s_min=float(policy.get("cvrp_tilted_s_min", 1.0)),
        )
        reduced_problem = ProblemInstance(
            name=f"{problem.name}_reduced",
            problem_type="cvrp",
            num_variables=reduced["qubo"].get_num_vars(),
            qubo=reduced["qubo"],
            optimal_value=problem.optimal_value,
            optimal_solution=None,
            metadata={
                **problem.metadata,
                "original_qp": reduced["qp"],
                "converter": reduced["converter"],
                "converter_penalty": reduced["converter_penalty"],
                "customers": list(plan["ambiguous_customers"]),
            },
        )
        _dump_qubo_lp(reduced_problem, inst_dir / "reduced_gap_qubo.lp")


def _top_counts_summary(counts: dict[str, int], problem: ProblemInstance) -> tuple[float, list[dict[str, Any]]]:
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = max(sum(count for _, count in sorted_counts), 1)
    top1_prob = sorted_counts[0][1] / total if sorted_counts else 0.0
    top: list[dict[str, Any]] = []
    for bitstring, count in sorted_counts[:10]:
        x = np.array([int(bit) for bit in bitstring[::-1]], dtype=float)
        if len(x) < problem.num_variables:
            x = np.pad(x, (0, problem.num_variables - len(x)))
        elif len(x) > problem.num_variables:
            x = x[: problem.num_variables]
        top.append(
            {
                "count": int(count),
                "probability": round(float(count) / total, 6),
                "selected": int(sum(x[: problem.num_variables])),
                "feasible_full_gap": bool(check_cvrp_feasibility(x, problem)),
            }
        )
    return float(top1_prob), top


def _raw_counts_summary(counts: dict[str, int]) -> tuple[float, list[dict[str, Any]]]:
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = max(sum(count for _, count in sorted_counts), 1)
    top1_prob = sorted_counts[0][1] / total if sorted_counts else 0.0
    top: list[dict[str, Any]] = []
    for bitstring, count in sorted_counts[:10]:
        top.append(
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": round(float(count) / total, 6),
                "selected": int(sum(int(bit) for bit in bitstring)),
            }
        )
    return float(top1_prob), top


def _evaluate_result(result, problem: ProblemInstance, policy: dict[str, Any], elapsed: float) -> tuple[dict[str, Any], dict[str, Any]]:
    best_x = result.best_bitstring
    feasible = bool(result.is_feasible and check_cvrp_feasibility(best_x, problem))
    cost = float(
        result.best_objective
        if result.best_objective is not None
        else cvrp_objective_value(best_x, problem)
    )
    ar = (
        float(problem.optimal_value) / max(cost, 1e-10)
        if feasible and np.isfinite(cost)
        else 0.0
    )
    ar = min(1.0, max(0.0, float(ar)))
    gap = exp._compute_optimality_gap(ar, feasible)
    feasibility_rate = compute_cvrp_feasibility_rate(result.counts, problem)
    best_feasible_ar = compute_cvrp_best_feasible_ar(result.counts, problem)
    improvement, stagnation, final_cost = exp._normalize_convergence(result.convergence_history)
    learning = exp._compute_learning_score(
        gap,
        feasible,
        feasibility_rate,
        best_feasible_ar,
        result,
    )
    top1_prob, top_counts = _top_counts_summary(result.counts, problem)
    attempt_stats = exp._attempt_shot_accounting(policy, result)
    metadata = dict(getattr(result, "metadata", {}) or {})
    reduced_counts = dict(metadata.get("reduced_gap_counts") or {})
    reduced_top1_prob, reduced_top_counts = _raw_counts_summary(reduced_counts)
    artifact_count_count = len(reduced_counts) if reduced_counts else len(result.counts)
    record = {
        "attempt": 0,
        "status": "completed",
        "solver_name": str(getattr(result, "solver_name", "cvrp_hardware")),
        "solver_family": str(policy.get("solver_family", policy.get("gap_solver_family", "hybrid"))),
        "policy_description": exp._build_description(str(policy.get("gap_solver_family", "hybrid")), policy),
        "policy_used": exp.snapshot_policy(policy),
        "optimality_gap": float(gap),
        "raw_ar": float(ar),
        "raw_feasible": bool(feasible),
        "raw_feasibility_rate": float(feasibility_rate),
        "best_feasible_ar_from_counts": float(best_feasible_ar),
        "learning_score": float(learning),
        "routed_cost": cost,
        "optimal_reference_cost": float(problem.optimal_value),
        "convergence_improvement": float(improvement),
        "convergence_stagnation": float(stagnation),
        "final_cost": float(final_cost),
        "wall_time_s": float(elapsed),
        "circuit_depth": int(getattr(result, "circuit_depth", 0)),
        "cnot_count": int(getattr(result, "cnot_count", 0)),
        "two_qubit_gate_count": int(getattr(result, "two_qubit_gate_count", 0)),
        "total_gate_count": int(getattr(result, "total_gate_count", 0)),
        "gate_counts": exp._json_safe(getattr(result, "gate_counts", {})),
        "num_qubits": int(getattr(result, "num_qubits", 0)),
        "num_parameters": int(getattr(result, "num_parameters", 0)),
        "optimizer_iterations": int(getattr(result, "optimizer_iterations", 0)),
        "top1_probability": float(top1_prob),
        "top10_summary": top_counts,
        "reduced_gap_top1_probability": float(reduced_top1_prob),
        "reduced_gap_top10_summary": reduced_top_counts,
        "post_repair_source": metadata.get("post_repair_source"),
        "repair_applied": bool(metadata.get("repair_applied", False)),
        "classical_fallback": bool(metadata.get("classical_fallback", False)),
        "raw_count_space": metadata.get("raw_count_space", "full_gap"),
        "hybrid_ambiguous_count": metadata.get("hybrid_ambiguous_count"),
        "hybrid_fixed_count": metadata.get("hybrid_fixed_count"),
        "route_solutions": metadata.get("route_solutions"),
        "route_qubit_counts": metadata.get("route_qubit_counts"),
        "result_metadata": exp._json_safe(metadata),
        **attempt_stats,
    }
    best_result = {
        "optimality_gap": float(gap),
        "approx_ratio": float(ar),
        "feasible": bool(feasible),
        "routed_cost": cost,
        "optimal_reference_cost": float(problem.optimal_value),
        "best_bitstring": np.asarray(best_x, dtype=int).tolist(),
        "solver_name": str(getattr(result, "solver_name", "")),
        "num_qubits": int(getattr(result, "num_qubits", 0)),
        "circuit_depth": int(getattr(result, "circuit_depth", 0)),
        "cnot_count": int(getattr(result, "cnot_count", 0)),
        "two_qubit_gate_count": int(getattr(result, "two_qubit_gate_count", 0)),
        "total_gate_count": int(getattr(result, "total_gate_count", 0)),
        "num_parameters": int(getattr(result, "num_parameters", 0)),
        "optimizer_iterations": int(getattr(result, "optimizer_iterations", 0)),
        "counts_unique_bitstrings": int(artifact_count_count),
        "metadata": exp._json_safe(metadata),
    }
    return record, best_result


def _run_hardware(args: argparse.Namespace) -> dict[str, Any]:
    policy = _load_policy(args)
    problem = _load_problem(policy, args.instance)
    plan = _make_reduced_gap_plan(problem, policy)

    credentials_path = Path(args.ibm_credentials_json).expanduser().resolve()
    if not credentials_path.is_file():
        raise FileNotFoundError(f"Credential file not found: {credentials_path}")
    if _has_placeholder_credentials(credentials_path):
        raise ValueError(f"Credential file still contains placeholders: {credentials_path}")

    checkpoint_path = Path(args.checkpoint_dir).expanduser().resolve() / "cvrp_e13.json"
    checkpoint = {} if args.force_rerun else _load_checkpoint(checkpoint_path)
    if checkpoint.get(DEFAULT_INSTANCE) and not args.force_rerun:
        print(f"Skipping E-n13 because checkpoint already has a completed result: {checkpoint_path}")
        return checkpoint[DEFAULT_INSTANCE]

    run_stamp = _utc_stamp()
    run_dir = Path(args.output_root).expanduser().resolve() / "cvrp" / f"e13_autoq_{run_stamp}"
    inst_dir = run_dir / "E_n13_k4"
    log_path = _configure_logging(run_dir=run_dir, log_level_name=args.log_level)
    _persist_plan_artifacts(inst_dir, problem, plan, policy)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Instance: %s", problem.name)
    LOGGER.info("Policy note: %s", get_cvrp_e13_policy_note())
    LOGGER.info(
        "Plan: full_qubits=%s reduced_qubits=%s ambiguous=%s will_submit_quantum=%s",
        plan.get("full_gap_qubo_variables"),
        plan.get("reduced_gap_qubo_variables"),
        plan.get("ambiguous_customer_count"),
        plan.get("will_submit_quantum_reduced_gap"),
    )
    LOGGER.info(
        "Hardware settings: estimator_shots=%s sampler_shots=%s optimizer_maxiter=%s",
        policy.get("estimator_shots"),
        policy.get("sampler_shots"),
        policy.get("optimizer_maxiter"),
    )

    factory = AutoQHardwareBackendFactory(
        ibm_credentials_json=credentials_path,
        ibm_min_runtime_seconds=float(args.ibm_min_runtime_seconds),
        qiskit_optimization_level=int(args.qiskit_optimization_level),
        job_status_log_interval=float(args.job_status_seconds),
        job_timeout_sec=args.job_timeout_sec,
        capture_calibration=bool(args.capture_calibration),
    )
    backend = factory.create_bundle(
        shots=int(policy.get("estimator_shots", exp.DEFAULT_ESTIMATOR_SHOTS)),
        sampler_shots=int(policy.get("sampler_shots", exp.DEFAULT_SAMPLER_SHOTS)),
        seed=policy.get("seed"),
    )

    job_start = factory.job_count
    t0 = time.perf_counter()
    status = "ERROR"
    attempt_records: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    raw_result = None
    error = None
    try:
        with patch_autoq_primitives():
            raw_result = exp._solve_cvrp_staged(problem, policy, backend)
        elapsed = float(time.perf_counter() - t0)
        attempt_record, best_result = _evaluate_result(raw_result, problem, policy, elapsed)
        attempt_record["job_metadata"] = factory.job_records(job_start)
        attempt_record["job_ids"] = [
            record.get("job_id") for record in attempt_record["job_metadata"] if record.get("job_id")
        ]
        attempt_record["backend_names"] = sorted(
            {
                str(record.get("backend_name"))
                for record in attempt_record["job_metadata"]
                if record.get("backend_name")
            }
        )
        attempt_records.append(attempt_record)
        status = "OK"
    except Exception as exc:
        elapsed = float(time.perf_counter() - t0)
        error = str(exc)
        attempt_records.append(
            {
                "attempt": 0,
                "status": "failed",
                "error": error,
                "wall_time_s": elapsed,
                "policy_used": exp.snapshot_policy(policy),
                "job_metadata": factory.job_records(job_start),
            }
        )

    total_sec = float(time.perf_counter() - t0)
    raw_metadata = dict(getattr(raw_result, "metadata", {}) or {}) if raw_result is not None else {}
    artifact_counts = dict(
        raw_metadata.get("reduced_gap_counts")
        or (dict(raw_result.counts) if raw_result is not None else {})
    )
    result_payload = {
        "schema_version": "1.0",
        "run_timestamp_utc": run_stamp,
        "run_directory": str(run_dir),
        "log_file": str(log_path),
        "problem": "cvrp",
        "problem_spec": f"cvrp_file_{DEFAULT_INSTANCE}",
        "instance_name": f"{DEFAULT_INSTANCE}.vrp",
        "instance_path": str(problem.metadata.get("source_file", "")),
        "execution": {
            "method": "autoqresearch_static_e13_hybrid_policy",
            "backend_mode": "hardware",
            "provider": "ibm",
            "policy_source": "retained_e13_policy",
            "ibm_credentials_json": str(credentials_path),
            "ibm_min_runtime_seconds": float(args.ibm_min_runtime_seconds),
            "job_status_log_interval_sec": float(args.job_status_seconds),
            "job_timeout_sec": None if args.job_timeout_sec is None else float(args.job_timeout_sec),
            "qiskit_optimization_level": int(args.qiskit_optimization_level),
            "policy_note": get_cvrp_e13_policy_note(),
        },
        "problem_info": {
            "name": problem.name,
            "num_variables": int(problem.num_variables),
            "optimal_reference_cost": float(problem.optimal_value),
            "num_customers": int(problem.metadata.get("num_customers", 0)),
            "num_vehicles": int(problem.metadata.get("num_vehicles", 0)),
        },
        "hardware_plan": plan,
        "policy": exp.snapshot_policy(policy),
        "timing": {"total_instance_sec": total_sec},
        "qpu_status_snapshot": factory.status_snapshot(),
        "device_calibration": factory.calibration_snapshot(),
        "job_metadata": factory.job_records(job_start),
        "attempts": attempt_records,
        "best_result": best_result,
        "counts": artifact_counts,
        "counts_space": raw_metadata.get("raw_count_space", "full_gap"),
        "config": vars(args),
        "status": status,
        "error": error,
    }

    _write_jsonl(inst_dir / "trace.jsonl", attempt_records)
    _write_json(inst_dir / "best_counts.json", result_payload["counts"])
    _write_json(inst_dir / "winning_policy.json", exp.snapshot_policy(policy))
    _write_json(inst_dir / "result.json", result_payload)
    _write_json(run_dir / "summary.json", result_payload)

    if status == "OK":
        checkpoint[DEFAULT_INSTANCE] = {
            "status": status,
            "result_dir": str(inst_dir),
            "optimality_gap": None if best_result is None else best_result.get("optimality_gap"),
            "feasible": None if best_result is None else best_result.get("feasible"),
            "jobs": len(result_payload["job_metadata"]),
            "run_directory": str(run_dir),
        }
        _save_checkpoint(checkpoint_path, checkpoint)

    LOGGER.info("\n%s", "=" * 70)
    LOGGER.info("CVRP E-n13 HARDWARE SUMMARY")
    LOGGER.info("%s", "=" * 70)
    if best_result is not None:
        LOGGER.info(
            "status=%s gap=%s feasible=%s cost=%s qubits=%s jobs=%s",
            status,
            best_result.get("optimality_gap"),
            best_result.get("feasible"),
            best_result.get("routed_cost"),
            best_result.get("num_qubits"),
            len(result_payload["job_metadata"]),
        )
    else:
        LOGGER.info("status=%s error=%s jobs=%s", status, error, len(result_payload["job_metadata"]))
    LOGGER.info("Result directory: %s", inst_dir)

    return result_payload


def _print_plan(args: argparse.Namespace) -> int:
    policy = _load_policy(args)
    problem = _load_problem(policy, args.instance)
    plan = _make_reduced_gap_plan(problem, policy)
    payload = {
        "problem": problem.name,
        "problem_spec": f"cvrp_file_{DEFAULT_INSTANCE}",
        "policy_note": get_cvrp_e13_policy_note(),
        "policy": exp.snapshot_policy(policy),
        "hardware_plan": plan,
    }
    print(json.dumps(exp._json_safe(payload), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the retained E-n13 CVRP hybrid policy on IBM hardware."
    )
    parser.add_argument("--instance", default=DEFAULT_INSTANCE, help="CVRP .vrp stem; default: E-n13-k4.")
    parser.add_argument(
        "--ibm-credentials-json",
        default=str(DEFAULT_CREDENTIALS_JSON),
        help="Credential pool JSON. The runner rotates when runtime drops below the threshold.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root for hardware artifacts.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR), help="Checkpoint directory.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing cvrp_e13 checkpoint.")
    parser.add_argument("--plan-only", action="store_true", help="Print E-n13 hardware plan without touching IBM Runtime.")
    parser.add_argument("--policy-file", default=None, help="Optional JSON override merged onto the retained policy.")
    parser.add_argument("--policy-json", default=None, help="Optional inline JSON override merged onto the retained policy.")
    parser.add_argument("--estimator-shots", type=int, default=None, help="Override retained estimator shots.")
    parser.add_argument("--sampler-shots", type=int, default=None, help="Override retained final sampler shots.")
    parser.add_argument("--optimizer-maxiter", type=int, default=None, help="Override retained VQE optimizer maxiter.")
    parser.add_argument("--hybrid-ambiguity-threshold", type=float, default=None, help="Override retained ambiguity threshold.")
    parser.add_argument("--seed", type=int, default=None, help="Override retained random seed.")
    parser.add_argument("--job-timeout-sec", type=float, default=None, help="Optional timeout for a single IBM primitive job wait.")
    parser.add_argument("--ibm-min-runtime-seconds", type=float, default=60.0, help="Rotate credentials below this remaining runtime.")
    parser.add_argument("--job-status-seconds", type=float, default=120.0, help="Minimum interval between IBM job status logs.")
    parser.add_argument("--qiskit-optimization-level", type=int, default=3, help="Transpilation optimization level.")
    parser.add_argument("--capture-calibration", action="store_true", help="Capture a calibration snapshot.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.instance != DEFAULT_INSTANCE:
        raise ValueError("This hardware runner is intentionally limited to the sole E-n13-k4 CVRP instance.")
    if args.plan_only:
        return _print_plan(args)
    payload = _run_hardware(args)
    return 0 if payload.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
