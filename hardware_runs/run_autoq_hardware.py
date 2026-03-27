#!/usr/bin/env python3
"""Run MIS instances on IBM hardware with the autoqresearch policy stack.

This runner keeps all hardware work under ``hardware_runs/``:

- it reuses the exact adaptive policy hooks from ``experiment.py``
- it submits through IBM Runtime V2 primitives
- it borrows the benchmark repo's credential rotation and artifact layout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import experiment as exp
from autoqresearch.problems.registry import get_mis_file_instance
from autoqresearch.solvers.qubo_primitives import (
    check_mis_feasibility,
    compute_mis_best_feasible_ar,
    compute_mis_feasibility_rate,
    mis_objective_value,
)

from autoq_hardware_backend import (
    AutoQHardwareBackendFactory,
    patch_autoq_primitives,
)
from static_mis_policies import (
    get_static_policy_for_instance,
    get_static_policy_note,
)


LOGGER = logging.getLogger("hardware_runs.run_autoq_hardware")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results_hardware"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints_autoq"
DEFAULT_CREDENTIALS_JSON = SCRIPT_DIR / "ibm_credentials.template.json"
DEFAULT_MIS_DIR = PROJECT_ROOT / "individual" / "mis"
RETAINED_SPARSE_STEMS = ("1tc.16", "1tc.32", "p1tc.48", "1tc.64")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arg_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _has_placeholder_credentials(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    markers = ("YOUR_TOKEN_", "YOUR_CRN_", "YOUR_INSTANCE_")
    return any(marker in text for marker in markers)


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


def _build_best_result_payload(best_outcome, best_raw_result, best_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "attempt": None if best_outcome is None else int(best_outcome.attempt),
        "optimality_gap": None if best_outcome is None else float(best_outcome.optimality_gap),
        "approx_ratio": None if best_outcome is None else float(best_outcome.raw_ar),
        "feasible": None if best_outcome is None else bool(best_outcome.raw_feasible),
        "feasibility_rate": None if best_outcome is None else float(best_outcome.raw_feasibility_rate),
        "objective_value": None
        if best_raw_result is None
        else float(getattr(best_raw_result, "best_objective", 0.0)),
        "solver_name": None
        if best_raw_result is None
        else str(getattr(best_raw_result, "solver_name", "")),
        "num_qubits": None
        if best_raw_result is None
        else int(getattr(best_raw_result, "num_qubits", 0)),
        "circuit_depth": None
        if best_raw_result is None
        else int(getattr(best_raw_result, "circuit_depth", 0)),
        "cnot_count": None
        if best_raw_result is None
        else int(getattr(best_raw_result, "cnot_count", 0)),
        "two_qubit_gate_count": None
        if best_raw_result is None
        else int(getattr(best_raw_result, "two_qubit_gate_count", 0)),
        "total_gate_count": None
        if best_raw_result is None
        else int(getattr(best_raw_result, "total_gate_count", 0)),
        "gate_counts": None
        if best_raw_result is None
        else exp._json_safe(getattr(best_raw_result, "gate_counts", {})),
        "num_parameters": None
        if best_raw_result is None
        else int(getattr(best_raw_result, "num_parameters", 0)),
        "wall_time_seconds": None
        if best_raw_result is None
        else float(getattr(best_raw_result, "wall_time_seconds", 0.0)),
        "best_bitstring": None
        if best_raw_result is None
        else np.asarray(best_raw_result.best_bitstring, dtype=int).tolist(),
        "counts_unique_bitstrings": int(len(best_counts)),
        "metadata": None if best_raw_result is None else exp._json_safe(best_raw_result.metadata),
    }


def _build_result_payload(
    *,
    args: argparse.Namespace,
    spec: str,
    instance_path: Path,
    inst_dir: Path,
    log_path: Path,
    run_stamp: str,
    problem,
    policy_mode: str,
    policy_source: str,
    policy_note: str | None,
    base_policy: dict[str, Any],
    initial_family: str,
    max_execution_attempts: int,
    total_instance_sec: float,
    best_raw_result,
    best_outcome,
    best_counts: dict[str, int],
    winning_policy: dict[str, Any],
    all_job_metadata: list[dict[str, Any]],
    attempt_records: list[dict[str, Any]],
    factory: AutoQHardwareBackendFactory,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_timestamp_utc": run_stamp,
        "run_directory": str(inst_dir),
        "log_file": str(log_path),
        "problem": "mis",
        "problem_spec": spec,
        "instance_name": instance_path.name,
        "instance_path": str(instance_path),
        "execution": {
            "method": (
                "autoqresearch_adaptive_policy"
                if policy_mode == "adaptive_controller"
                else "autoqresearch_static_policy"
            ),
            "backend_mode": "hardware",
            "provider": "ibm",
            "policy_mode": policy_mode,
            "policy_source": policy_source,
            "max_attempts": int(max_execution_attempts),
            "timeout_sec": float(args.timeout_sec),
            "seed_override": args.seed_override,
            "ibm_credentials_json": str(
                Path(args.ibm_credentials_json).expanduser().resolve()
            ),
            "ibm_min_runtime_seconds": float(args.ibm_min_runtime_seconds),
            "job_status_log_interval_sec": float(args.job_status_seconds),
            "job_timeout_sec": None if args.job_timeout_sec is None else float(args.job_timeout_sec),
            "qiskit_optimization_level": int(args.qiskit_optimization_level),
            "policy_note": policy_note,
        },
        "problem_info": {
            "name": problem.name,
            "num_variables": int(problem.num_variables),
            "optimal_value": float(problem.optimal_value),
        },
        "policy": {
            "initial_family": initial_family,
            "base_policy": exp.snapshot_policy(base_policy),
            "winning_policy": winning_policy,
        },
        "timing": {
            "total_instance_sec": total_instance_sec,
        },
        "circuit_metrics": {
            "logical_num_variables": int(problem.num_variables),
            "best_result_num_qubits": None
            if best_raw_result is None
            else int(getattr(best_raw_result, "num_qubits", 0)),
            "best_result_depth": None
            if best_raw_result is None
            else int(getattr(best_raw_result, "circuit_depth", 0)),
            "best_result_cnot_count": None
            if best_raw_result is None
            else int(getattr(best_raw_result, "cnot_count", 0)),
            "best_result_two_qubit_gate_count": None
            if best_raw_result is None
            else int(getattr(best_raw_result, "two_qubit_gate_count", 0)),
            "best_result_total_gate_count": None
            if best_raw_result is None
            else int(getattr(best_raw_result, "total_gate_count", 0)),
            "best_result_gate_counts": None
            if best_raw_result is None
            else exp._json_safe(getattr(best_raw_result, "gate_counts", {})),
        },
        "qpu_status_snapshot": factory.status_snapshot(),
        "device_calibration": factory.calibration_snapshot(),
        "job_metadata": all_job_metadata,
        "attempts": attempt_records,
        "best_result": _build_best_result_payload(best_outcome, best_raw_result, best_counts),
        "config": vars(args),
        "status": status,
    }


def _persist_instance_artifacts(
    *,
    args: argparse.Namespace,
    spec: str,
    instance_path: Path,
    inst_dir: Path,
    log_path: Path,
    run_stamp: str,
    problem,
    policy_mode: str,
    policy_source: str,
    policy_note: str | None,
    base_policy: dict[str, Any],
    initial_family: str,
    max_execution_attempts: int,
    total_instance_sec: float,
    best_raw_result,
    best_outcome,
    attempt_records: list[dict[str, Any]],
    factory: AutoQHardwareBackendFactory,
    instance_job_start: int,
    status: str,
) -> dict[str, Any]:
    best_counts = dict(best_raw_result.counts) if best_raw_result is not None else {}
    winning_policy = (
        exp.snapshot_policy(best_outcome.policy_used) if best_outcome is not None else {}
    )
    all_job_metadata = factory.job_records(instance_job_start)
    result_payload = _build_result_payload(
        args=args,
        spec=spec,
        instance_path=instance_path,
        inst_dir=inst_dir,
        log_path=log_path,
        run_stamp=run_stamp,
        problem=problem,
        policy_mode=policy_mode,
        policy_source=policy_source,
        policy_note=policy_note,
        base_policy=base_policy,
        initial_family=initial_family,
        max_execution_attempts=max_execution_attempts,
        total_instance_sec=total_instance_sec,
        best_raw_result=best_raw_result,
        best_outcome=best_outcome,
        best_counts=best_counts,
        winning_policy=winning_policy,
        all_job_metadata=all_job_metadata,
        attempt_records=attempt_records,
        factory=factory,
        status=status,
    )
    _write_jsonl(inst_dir / "trace.jsonl", attempt_records)
    _write_json(inst_dir / "best_counts.json", best_counts)
    _write_json(inst_dir / "winning_policy.json", winning_policy)
    _write_json(inst_dir / "result.json", result_payload)
    return result_payload


def _resolve_instance_spec(raw: str, instance_dir: Path) -> tuple[str, Path]:
    candidate = Path(raw)
    if candidate.suffix == ".txt":
        path = candidate if candidate.is_absolute() else (instance_dir / candidate.name)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MIS instance file not found: {path}")
        return f"mis_file_{path.stem}", path

    if raw.startswith("mis_file_"):
        stem = raw[len("mis_file_") :]
    else:
        stem = raw
    path = (instance_dir / f"{stem}.txt").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MIS instance file not found: {path}")
    return f"mis_file_{stem}", path


def _resolve_instances(args: argparse.Namespace) -> list[tuple[str, Path]]:
    instance_dir = Path(args.instance_dir).expanduser().resolve()
    resolved: list[tuple[str, Path]] = []
    seen: set[str] = set()

    raw_instances: list[str] = []
    if args.retained_only:
        raw_instances.extend(RETAINED_SPARSE_STEMS)
    if args.all_mis:
        raw_instances.extend(path.stem for path in sorted(instance_dir.glob("*.txt")))
    raw_instances.extend(args.instance or [])

    if not raw_instances:
        raise ValueError(
            "Specify at least one instance via --instance, or use --all-mis / --retained-only."
        )

    for raw in raw_instances:
        spec, path = _resolve_instance_spec(str(raw), instance_dir)
        if spec in seen:
            continue
        seen.add(spec)
        resolved.append((spec, path))
    return resolved


def _dump_qubo_lp(problem, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(problem.qubo, "export_as_lp_string"):
        text = problem.qubo.export_as_lp_string()
    else:
        text = str(problem.qubo.prettyprint())
    path.write_text(str(text), encoding="utf-8")


def _resolve_instance_execution(
    *,
    args: argparse.Namespace,
    problem,
    instance_stem: str,
    policy_override: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str, int, str | None]:
    if policy_override is not None:
        initial_family = (
            args.solver_family
            or (
                str(policy_override.get("solver_family"))
                if policy_override.get("solver_family")
                else None
            )
            or exp.choose_solver_family(problem)
        )
        base_policy = exp.build_base_policy(problem, initial_family)
        base_policy = exp._merge_policy_override(base_policy, policy_override)
        base_policy["solver_family"] = str(base_policy.get("solver_family", initial_family))
        return (
            base_policy,
            "static_override",
            "policy_file" if args.policy_file else "policy_json",
            1,
            "Static run from explicit override.",
        )

    if not args.static_retained:
        initial_family = (
            args.solver_family
            or exp.choose_solver_family(problem)
        )
        base_policy = exp.build_base_policy(problem, initial_family)
        base_policy["solver_family"] = initial_family
        return (
            base_policy,
            "adaptive_controller",
            "experiment_policy_surface",
            int(args.max_attempts),
            "Default adaptive multi-attempt controller mode.",
        )

    static_policy = get_static_policy_for_instance(instance_stem)
    if static_policy is not None:
        base_policy = exp._merge_policy_override(static_policy, None)
        return (
            base_policy,
            "static_retained_winner",
            "retained_instance_map",
            1,
            get_static_policy_note(instance_stem),
        )

    initial_family = args.solver_family or exp.choose_solver_family(problem)
    base_policy = exp.build_base_policy(problem, initial_family)
    base_policy["solver_family"] = initial_family
    return (
        base_policy,
        "adaptive_controller",
        "build_base_policy",
        int(args.max_attempts),
        "No retained static winner found; using the adaptive base-policy controller.",
    )


def _print_plan(
    instances: list[tuple[str, Path]],
    args: argparse.Namespace,
    policy_file: Path | None,
    policy_json: str | None,
) -> int:
    policy_override = exp._load_policy_override(policy_file, policy_json)
    plan: list[dict[str, Any]] = []
    for spec, path in instances:
        _, filename, _ = exp._parse_problem_spec(spec)
        penalty = policy_override.get("penalty") if policy_override else None
        problem = get_mis_file_instance(filename, penalty=penalty)
        stem = path.stem
        base_policy, plan_mode, policy_source, max_attempts, note = _resolve_instance_execution(
            args=args,
            problem=problem,
            instance_stem=stem,
            policy_override=policy_override,
        )
        family = str(base_policy.get("solver_family", exp.choose_solver_family(problem)))
        plan.append(
            {
                "instance": path.name,
                "problem_spec": spec,
                "num_variables": int(problem.num_variables),
                "optimal_value": float(problem.optimal_value),
                "initial_family": family,
                "policy_mode": plan_mode,
                "policy_source": policy_source,
                "max_attempts": max_attempts,
                "note": note,
                "base_policy": exp.snapshot_policy(base_policy),
            }
        )
    print(json.dumps(exp._json_safe(plan), indent=2, sort_keys=True))
    return 0


def _top10_summary(counts: dict[str, int], problem) -> tuple[float, list[dict[str, Any]]]:
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = max(sum(count for _, count in sorted_counts), 1)
    top1_prob = sorted_counts[0][1] / total if sorted_counts else 0.0
    top10: list[dict[str, Any]] = []
    for bitstring, count in sorted_counts[:10]:
        x = np.array([int(bit) for bit in bitstring[::-1]], dtype=float)
        if len(x) < problem.num_variables:
            x = np.pad(x, (0, problem.num_variables - len(x)))
        elif len(x) > problem.num_variables:
            x = x[: problem.num_variables]
        top10.append(
            {
                "count": int(count),
                "prob": round(float(count) / total, 4),
                "selected": int(sum(x[: problem.num_variables])),
                "feasible": bool(check_mis_feasibility(x, problem)),
            }
        )
    return float(top1_prob), top10


def _run_single_instance(
    *,
    args: argparse.Namespace,
    spec: str,
    instance_path: Path,
    factory: AutoQHardwareBackendFactory,
    run_dir: Path,
    run_stamp: str,
    log_path: Path,
) -> dict[str, Any]:
    _, filename, _ = exp._parse_problem_spec(spec)
    policy_file = Path(args.policy_file).expanduser().resolve() if args.policy_file else None
    policy_override = exp._load_policy_override(policy_file, args.policy_json)
    penalty = policy_override.get("penalty") if policy_override else None
    problem = get_mis_file_instance(filename, penalty=penalty)
    instance_stem = instance_path.stem

    inst_dir = run_dir / instance_path.stem.replace(".", "_")
    inst_dir.mkdir(parents=True, exist_ok=True)
    _dump_qubo_lp(problem, inst_dir / "qubo.lp")

    base_policy, policy_mode, policy_source, max_execution_attempts, policy_note = (
        _resolve_instance_execution(
            args=args,
            problem=problem,
            instance_stem=instance_stem,
            policy_override=policy_override,
        )
    )
    initial_family = str(base_policy.get("solver_family", exp.choose_solver_family(problem)))
    if args.seed_override is not None:
        base_policy["seed"] = int(args.seed_override)

    LOGGER.info("\n%s", "=" * 70)
    LOGGER.info("Instance: %s", instance_path.name)
    LOGGER.info(
        "Problem: %s | qubo_vars=%s | optimal=%s | initial_family=%s | policy_mode=%s",
        problem.name,
        problem.num_variables,
        problem.optimal_value,
        initial_family,
        policy_mode,
    )
    if policy_note:
        LOGGER.info("Policy note: %s", policy_note)
    LOGGER.info("%s", "=" * 70)

    history: list[exp.AttemptOutcome] = []
    attempt_records: list[dict[str, Any]] = []
    best_raw_result = None
    best_outcome = None
    best_gap = float("inf")
    best_feasible_ar_global = 0.0
    attempt = 0
    t_total = time.time()
    instance_job_start = factory.job_count

    while attempt < max_execution_attempts:
        if policy_mode == "adaptive_controller":
            if not exp.should_continue(attempt, history, problem, int(args.max_attempts)):
                break
            policy = exp.adapt_policy(attempt, history, problem, base_policy)
        else:
            policy = base_policy.copy()
        if args.seed_override is not None:
            policy["seed"] = int(args.seed_override)

        attempt_family = str(policy.get("solver_family", initial_family)).lower()
        policy["solver_family"] = attempt_family
        policy["pce_local_search"] = False
        policy["final_local_search"] = False

        attempt_base = exp.build_base_policy(problem, attempt_family)
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

        solve_fn = exp._get_solver_fn(attempt_family)
        backend = factory.create_bundle(
            shots=int(policy.get("estimator_shots", exp.DEFAULT_ESTIMATOR_SHOTS)),
            sampler_shots=int(policy.get("sampler_shots", exp.DEFAULT_SAMPLER_SHOTS)),
            seed=policy.get("seed"),
        )

        jobs_before = factory.job_count
        t0 = time.time()
        try:
            result = solve_fn(problem, policy, backend)
        except Exception as exc:
            elapsed = time.time() - t0
            job_metadata = factory.job_records(jobs_before)
            LOGGER.info("Attempt %s FAILED after %.1fs: %s", attempt, elapsed, exc)
            attempt_records.append(
                {
                    "attempt": int(attempt),
                    "status": "failed",
                    "error": str(exc),
                    "solver_family": attempt_family,
                    "policy_used": exp.snapshot_policy(policy),
                    "wall_time_s": float(elapsed),
                    "job_metadata": job_metadata,
                    "job_ids": [record.get("job_id") for record in job_metadata if record.get("job_id")],
                    "backend_names": sorted(
                        {
                            str(record.get("backend_name"))
                            for record in job_metadata
                            if record.get("backend_name")
                        }
                    ),
                }
            )
            _persist_instance_artifacts(
                args=args,
                spec=spec,
                instance_path=instance_path,
                inst_dir=inst_dir,
                log_path=log_path,
                run_stamp=run_stamp,
                problem=problem,
                policy_mode=policy_mode,
                policy_source=policy_source,
                policy_note=policy_note,
                base_policy=base_policy,
                initial_family=initial_family,
                max_execution_attempts=max_execution_attempts,
                total_instance_sec=float(time.time() - t_total),
                best_raw_result=best_raw_result,
                best_outcome=best_outcome,
                attempt_records=attempt_records,
                factory=factory,
                instance_job_start=instance_job_start,
                status="RUNNING",
            )
            attempt += 1
            if (time.time() - t_total) > float(args.timeout_sec):
                break
            continue

        elapsed = time.time() - t0
        best_x = result.best_bitstring
        is_feasible = check_mis_feasibility(best_x, problem)
        objective = mis_objective_value(best_x, problem)
        ar = (objective / max(problem.optimal_value, 1e-10)) if is_feasible else 0.0
        ar = min(1.0, max(0.0, float(ar)))
        feas_rate = compute_mis_feasibility_rate(result.counts, problem)
        attempt_best_feas_ar = compute_mis_best_feasible_ar(result.counts, problem)
        gap = exp._compute_optimality_gap(ar, is_feasible)
        best_feasible_ar_global = max(best_feasible_ar_global, attempt_best_feas_ar)
        improvement, stagnation, final_cost = exp._normalize_convergence(
            result.convergence_history
        )
        learning = exp._compute_learning_score(
            gap,
            is_feasible,
            feas_rate,
            best_feasible_ar_global,
            result,
        )
        top1_prob, top10 = _top10_summary(result.counts, problem)

        outcome = exp.AttemptOutcome(
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
            top1_probability=top1_prob,
            top10_summary=top10,
        )
        history.append(outcome)

        if gap < best_gap:
            best_gap = gap
            best_raw_result = result
            best_outcome = outcome

        job_metadata = factory.job_records(jobs_before)
        attempt_stats = exp._attempt_shot_accounting(policy, result)
        attempt_record = {
            "attempt": int(attempt),
            "status": "completed",
            "solver_name": str(getattr(result, "solver_name", attempt_family) or attempt_family),
            "solver_family": attempt_family,
            "policy_description": exp._build_description(attempt_family, policy),
            "policy_used": exp.snapshot_policy(policy),
            "learning_score": float(learning),
            "optimality_gap": float(gap),
            "raw_feasible": bool(is_feasible),
            "raw_feasibility_rate": float(feas_rate),
            "raw_ar": float(ar),
            "objective_value": float(objective),
            "convergence_improvement": float(improvement),
            "convergence_stagnation": float(stagnation),
            "final_cost": float(final_cost),
            "best_feasible_ar_global": float(best_feasible_ar_global),
            "wall_time_s": float(elapsed),
            "circuit_depth": int(getattr(result, "circuit_depth", 0)),
            "cnot_count": int(getattr(result, "cnot_count", 0)),
            "two_qubit_gate_count": int(getattr(result, "two_qubit_gate_count", 0)),
            "total_gate_count": int(getattr(result, "total_gate_count", 0)),
            "gate_counts": exp._json_safe(getattr(result, "gate_counts", {})),
            "num_qubits": int(getattr(result, "num_qubits", 0)),
            "num_parameters": int(getattr(result, "num_parameters", 0)),
            "top1_probability": float(top1_prob),
            "top10_summary": top10,
            "job_metadata": job_metadata,
            "job_ids": [record.get("job_id") for record in job_metadata if record.get("job_id")],
            "backend_names": sorted(
                {
                    str(record.get("backend_name"))
                    for record in job_metadata
                    if record.get("backend_name")
                }
            ),
            **attempt_stats,
        }
        attempt_records.append(attempt_record)

        LOGGER.info(
            "Attempt %s | gap=%.4f AR=%.4f feasible=%s top1=%.4f time=%.1fs backends=%s",
            attempt,
            gap,
            ar,
            is_feasible,
            top1_prob,
            elapsed,
            ",".join(attempt_record["backend_names"]) or "(none)",
        )
        _persist_instance_artifacts(
            args=args,
            spec=spec,
            instance_path=instance_path,
            inst_dir=inst_dir,
            log_path=log_path,
            run_stamp=run_stamp,
            problem=problem,
            policy_mode=policy_mode,
            policy_source=policy_source,
            policy_note=policy_note,
            base_policy=base_policy,
            initial_family=initial_family,
            max_execution_attempts=max_execution_attempts,
            total_instance_sec=float(time.time() - t_total),
            best_raw_result=best_raw_result,
            best_outcome=best_outcome,
            attempt_records=attempt_records,
            factory=factory,
            instance_job_start=instance_job_start,
            status="RUNNING",
        )
        attempt += 1
        if (time.time() - t_total) > float(args.timeout_sec):
            break

    total_instance_sec = float(time.time() - t_total)
    status = (
        "OK"
        if any(record.get("status") == "completed" for record in attempt_records)
        else "ERROR"
    )
    result_payload = _persist_instance_artifacts(
        args=args,
        spec=spec,
        instance_path=instance_path,
        inst_dir=inst_dir,
        log_path=log_path,
        run_stamp=run_stamp,
        problem=problem,
        policy_mode=policy_mode,
        policy_source=policy_source,
        policy_note=policy_note,
        base_policy=base_policy,
        initial_family=initial_family,
        max_execution_attempts=max_execution_attempts,
        total_instance_sec=total_instance_sec,
        best_raw_result=best_raw_result,
        best_outcome=best_outcome,
        attempt_records=attempt_records,
        factory=factory,
        instance_job_start=instance_job_start,
        status=status,
    )
    all_job_metadata = result_payload["job_metadata"]

    summary = {
        "instance": instance_path.name,
        "qubits": int(
            getattr(best_raw_result, "num_qubits", problem.num_variables)
            if best_raw_result is not None
            else problem.num_variables
        ),
        "status": status,
        "objective": None
        if best_raw_result is None
        else float(getattr(best_raw_result, "best_objective", 0.0)),
        "gap": None if best_outcome is None else float(best_outcome.optimality_gap),
        "approx_ratio": None if best_outcome is None else float(best_outcome.raw_ar),
        "feasible": None if best_outcome is None else bool(best_outcome.raw_feasible),
        "time_sec": total_instance_sec,
        "jobs": len(all_job_metadata),
        "backends": sorted(
            {
                str(record.get("backend_name"))
                for record in all_job_metadata
                if record.get("backend_name")
            }
        ),
        "result_dir": str(inst_dir),
        "policy_mode": policy_mode,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MIS instances on IBM hardware using the autoqresearch solver stack."
        )
    )
    parser.add_argument("--instance", action="append", help="MIS stem, spec, or .txt path. Repeatable.")
    parser.add_argument("--all-mis", action="store_true", help="Run every .txt under individual/mis.")
    parser.add_argument(
        "--retained-only",
        action="store_true",
        help="Run the retained sparse set: 1tc.16, 1tc.32, p1tc.48, 1tc.64.",
    )
    parser.add_argument(
        "--instance-dir",
        default=str(DEFAULT_MIS_DIR),
        help="Directory containing MIS instance .txt files.",
    )
    parser.add_argument(
        "--ibm-credentials-json",
        default=str(DEFAULT_CREDENTIALS_JSON),
        help="Credential pool JSON. The runner rotates when runtime drops below the threshold.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for benchmark-style run artifacts.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Checkpoint directory. Completed instances are skipped unless --force-rerun is set.",
    )
    parser.add_argument("--solver-family", default=None, help="Force the initial solver family.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Upper bound for the adaptive controller. Static override and --static-retained runs use one attempt.",
    )
    parser.add_argument(
        "--static-retained",
        action="store_true",
        help="Replay one fixed retained-instance winner per instance instead of the default adaptive controller.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=43200.0,
        help="Per-instance wall-clock timeout for the adaptive loop.",
    )
    parser.add_argument(
        "--job-timeout-sec",
        type=float,
        default=None,
        help="Optional timeout for a single IBM primitive job wait.",
    )
    parser.add_argument(
        "--ibm-min-runtime-seconds",
        type=float,
        default=60.0,
        help="Rotate to the next credential once the current account drops below this remaining QPU runtime.",
    )
    parser.add_argument(
        "--job-status-seconds",
        type=float,
        default=120.0,
        help="Minimum interval between IBM job status log lines.",
    )
    parser.add_argument(
        "--qiskit-optimization-level",
        type=int,
        default=3,
        help="Stored for parity with the benchmark manager; runtime primitives still transpile server-side here.",
    )
    parser.add_argument("--policy-file", default=None, help="Optional JSON file merged onto the base policy.")
    parser.add_argument("--policy-json", default=None, help="Optional inline JSON merged onto the base policy.")
    parser.add_argument("--seed-override", type=int, default=None, help="Override the policy seed.")
    parser.add_argument("--capture-calibration", action="store_true", help="Capture a calibration snapshot from the benchmark manager.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore checkpoint state and rerun completed instances.")
    parser.add_argument("--plan-only", action="store_true", help="Print resolved instances and starting policies, then exit.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    instances = _resolve_instances(args)
    policy_file = Path(args.policy_file).expanduser().resolve() if args.policy_file else None
    if args.plan_only:
        return _print_plan(instances, args, policy_file, args.policy_json)

    credentials_path = Path(args.ibm_credentials_json).expanduser().resolve()
    if not credentials_path.is_file():
        raise FileNotFoundError(f"Credential file not found: {credentials_path}")
    if _has_placeholder_credentials(credentials_path):
        raise ValueError(
            f"Credential file still contains placeholders: {credentials_path}"
        )

    output_root = Path(args.output_root).expanduser().resolve()
    run_stamp = _utc_stamp()
    run_dir = output_root / "mis" / f"mis_autoq_{run_stamp}"
    log_path = _configure_logging(run_dir=run_dir, log_level_name=args.log_level)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Resolved %d instance(s): %s", len(instances), [path.name for _, path in instances])

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_path = checkpoint_dir / "mis.json"
    checkpoint = {} if args.force_rerun else _load_checkpoint(checkpoint_path)
    if checkpoint and not args.force_rerun:
        LOGGER.info("Loaded checkpoint %s with %d completed instance(s).", checkpoint_path, len(checkpoint))

    factory = AutoQHardwareBackendFactory(
        ibm_credentials_json=credentials_path,
        ibm_min_runtime_seconds=args.ibm_min_runtime_seconds,
        qiskit_optimization_level=args.qiskit_optimization_level,
        job_status_log_interval=args.job_status_seconds,
        job_timeout_sec=args.job_timeout_sec,
        capture_calibration=args.capture_calibration,
    )

    results: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    with patch_autoq_primitives():
        for index, (spec, path) in enumerate(instances, 1):
            instance_name = path.name
            if instance_name in checkpoint and not args.force_rerun:
                LOGGER.info(
                    "\n[%d/%d] SKIPPING (checkpoint): %s",
                    index,
                    len(instances),
                    instance_name,
                )
                results.append(checkpoint[instance_name])
                continue

            LOGGER.info("\n[%d/%d] Starting: %s", index, len(instances), instance_name)
            result = _run_single_instance(
                args=args,
                spec=spec,
                instance_path=path,
                factory=factory,
                run_dir=run_dir,
                run_stamp=run_stamp,
                log_path=log_path,
            )
            results.append(result)

            if result.get("status") == "OK":
                checkpoint[instance_name] = result
                _save_checkpoint(checkpoint_path, checkpoint)
                LOGGER.info("Checkpoint updated: %s", checkpoint_path)

    total_runtime = float(time.perf_counter() - run_start)
    ok_count = sum(1 for result in results if result.get("status") == "OK")
    summary = {
        "problem": "mis",
        "method": (
            "autoqresearch_static_policy"
            if args.static_retained or args.policy_file or args.policy_json
            else "autoqresearch_adaptive_policy"
        ),
        "total_instances": len(results),
        "ok_count": ok_count,
        "total_runtime_sec": total_runtime,
        "results": results,
    }
    _write_json(run_dir / "summary.json", summary)

    LOGGER.info("\n%s", "=" * 70)
    LOGGER.info("AUTOQ HARDWARE SUMMARY")
    LOGGER.info("%s", "=" * 70)
    for result in results:
        LOGGER.info(
            "%-14s status=%-5s gap=%-8s feasible=%-5s time=%7.1fs backends=%s",
            result.get("instance", "?"),
            result.get("status", "?"),
            result.get("gap", "?"),
            result.get("feasible", "?"),
            float(result.get("time_sec", 0.0) or 0.0),
            ",".join(result.get("backends", [])) or "(none)",
        )
    LOGGER.info("Results: %s", run_dir)
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
