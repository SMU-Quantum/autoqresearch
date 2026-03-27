#!/usr/bin/env python3
"""Batch study runner for adaptive-vs-static solver control comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from experiment import (
    DEFAULT_ESTIMATOR_SHOTS,
    DEFAULT_SAMPLER_SHOTS,
    build_base_policy,
    build_static_baseline_policy,
    run_experiment,
    snapshot_policy,
)
from autoqresearch.problems.registry import get_single_instance


SPLIT_SEEDS = {
    "train": [0, 1, 2, 3, 4],
    "dev": [100, 101, 102, 103, 104],
    "test": [200, 201, 202, 203, 204],
}

STATIC_VARIANTS = {
    "static_basic_vqe",
    "static_qaoa_standard",
    "static_qaoa_cvar",
    "static_qaoa_warmstart",
    "static_qaoa_warmstart_cvar",
    "static_qaoa_multiangle",
    "static_qaoa_multiangle_cvar",
    "static_qrao",
    "static_qrao_cvar",
    "static_pce",
    "static_pce_cvar",
}

DERIVED_VARIANTS = {"static_final", "static_direct_stage2"}


def _study_runs_header() -> list[str]:
    return [
        "run_id",
        "study_id",
        "timestamp",
        "problem",
        "problem_type",
        "size",
        "seed",
        "split",
        "variant",
        "run_tag",
        "prompt_variant",
        "budget_mode",
        "matched_total_shots",
        "source_run_id",
        "status",
        "solver_family",
        "policy_mode",
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
        "best_attempt_index",
        "first_feasible_attempt",
        "first_ar_ge_0_5_attempt",
        "shots_to_first_feasible",
        "shots_to_ar_ge_0_5",
        "total_run_shots",
        "total_wall_time_s",
        "winning_policy_path",
        "summary_json_path",
        "attempts_jsonl_path",
    ]


def _ensure_tsv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Study manifest must decode to a JSON object.")
    return manifest


def _split_for_seed(seed: int) -> str:
    for split_name, seeds in SPLIT_SEEDS.items():
        if seed in seeds:
            return split_name
    return "custom"


def expand_study_cases(manifest: dict) -> list[dict]:
    problem_type = str(manifest.get("problem_type", "knapsack"))
    sizes = [int(size) for size in manifest.get("sizes", [])]
    if not sizes:
        raise ValueError("Manifest must include at least one problem size.")

    seeds: list[int] = []
    for split_name in manifest.get("splits", []):
        if split_name not in SPLIT_SEEDS:
            raise ValueError(f"Unknown split '{split_name}'. Expected one of {sorted(SPLIT_SEEDS)}.")
        seeds.extend(SPLIT_SEEDS[split_name])
    seeds.extend(int(seed) for seed in manifest.get("seeds", []))
    if not seeds:
        seeds = [0]

    cases = []
    for size in sizes:
        for seed in sorted(set(seeds)):
            split = _split_for_seed(seed)
            cases.append(
                {
                    "problem_type": problem_type,
                    "size": size,
                    "seed": seed,
                    "split": split,
                    "problem_spec": f"{problem_type}_{size}_s{seed}",
                }
            )
    return cases


def build_static_variant_policy(problem, variant: str) -> tuple[str, dict]:
    if variant == "static_basic_vqe":
        family = "vqe"
        policy = build_static_baseline_policy(problem)
    elif variant == "static_qaoa_standard":
        family = "qaoa"
        policy = build_base_policy(problem, family)
        policy["variant"] = "standard"
    elif variant == "static_qaoa_cvar":
        family = "qaoa"
        policy = build_base_policy(problem, family)
        policy["variant"] = "standard"
        policy["measurement_mode"] = "cvar"
    elif variant == "static_qaoa_warmstart":
        family = "qaoa"
        policy = build_base_policy(problem, family)
        policy["variant"] = "warmstart"
        policy["ws_source"] = "relaxation"
    elif variant == "static_qaoa_warmstart_cvar":
        family = "qaoa"
        policy = build_base_policy(problem, family)
        policy["variant"] = "warmstart"
        policy["measurement_mode"] = "cvar"
        policy["ws_source"] = "relaxation"
    elif variant == "static_qaoa_multiangle":
        family = "qaoa"
        policy = build_base_policy(problem, family)
        policy["variant"] = "multiangle"
        policy["ma_tying"] = "none"
    elif variant == "static_qaoa_multiangle_cvar":
        family = "qaoa"
        policy = build_base_policy(problem, family)
        policy["variant"] = "multiangle"
        policy["measurement_mode"] = "cvar"
        policy["ma_tying"] = "none"
    elif variant == "static_qrao":
        family = "qrao"
        policy = build_base_policy(problem, family)
    elif variant == "static_qrao_cvar":
        family = "qrao"
        policy = build_base_policy(problem, family)
        policy["measurement_mode"] = "cvar"
    elif variant == "static_pce":
        family = "pce"
        policy = build_base_policy(problem, family)
    elif variant == "static_pce_cvar":
        family = "pce"
        policy = build_base_policy(problem, family)
        policy["measurement_mode"] = "cvar"
    else:
        raise ValueError(f"Unsupported static variant: {variant}")

    policy["solver_family"] = family
    return family, snapshot_policy(policy)


def rebudget_policy(
    policy: dict,
    target_total_shots: int,
    reference_iterations: int | None = None,
) -> dict:
    adjusted = dict(policy)
    estimator_shots = int(adjusted.get("estimator_shots", DEFAULT_ESTIMATOR_SHOTS))
    sampler_shots = int(adjusted.get("sampler_shots", DEFAULT_SAMPLER_SHOTS))
    if reference_iterations is None or reference_iterations <= 0:
        reference_iterations = int(adjusted.get("optimizer_maxiter", 1) or 1)
    reference_iterations = max(reference_iterations, 1)
    base_total = estimator_shots * reference_iterations + sampler_shots
    if base_total <= 0 or target_total_shots <= 0:
        return adjusted

    scale = float(target_total_shots) / float(base_total)
    adjusted["estimator_shots"] = max(32, int(round(estimator_shots * scale)))
    adjusted["sampler_shots"] = max(32, int(round(sampler_shots * scale)))
    return snapshot_policy(adjusted)


def build_followup_policies(adaptive_summary: dict) -> dict[str, dict]:
    policies: dict[str, dict] = {}
    winning_policy = adaptive_summary.get("winning_policy")
    if isinstance(winning_policy, dict) and winning_policy:
        policies["static_final"] = snapshot_policy(winning_policy)

    direct_stage2_policy = adaptive_summary.get("direct_stage2_policy")
    if isinstance(direct_stage2_policy, dict) and direct_stage2_policy:
        policies["static_direct_stage2"] = snapshot_policy(direct_stage2_policy)
    return policies


def _next_run_id(runs_path: Path) -> int:
    if not runs_path.exists() or runs_path.stat().st_size == 0:
        return 1
    with runs_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        max_id = 0
        for row in reader:
            try:
                max_id = max(max_id, int(str(row.get("run_id", "0")).strip() or 0))
            except ValueError:
                continue
    return max_id + 1


def _append_run_row(path: Path, row: dict) -> None:
    _ensure_tsv(path, _study_runs_header())
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([row.get(column, "") for column in _study_runs_header()])


def _append_attempt_records(path: Path, run_id: int, metadata: dict, attempts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        for attempt in attempts:
            payload = {
                "run_id": run_id,
                **metadata,
                **attempt,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _summary_optimality_gap(summary: dict) -> object:
    return summary.get("optimality_gap", "")


def _summary_repaired_optimality_gap(summary: dict) -> object:
    return summary.get("repaired_optimality_gap", "")


def _study_row_from_summary(
    study_id: str,
    run_id: int,
    case: dict,
    variant: str,
    run_tag: str,
    prompt_variant: str,
    budget_mode: str,
    matched_total_shots: int | None,
    source_run_id: int | None,
    summary: dict,
) -> dict:
    return {
        "run_id": run_id,
        "study_id": study_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "problem": summary.get("problem", case["problem_spec"]),
        "problem_type": case["problem_type"],
        "size": case["size"],
        "seed": case["seed"],
        "split": case["split"],
        "variant": variant,
        "run_tag": run_tag,
        "prompt_variant": prompt_variant,
        "budget_mode": budget_mode,
        "matched_total_shots": matched_total_shots if matched_total_shots is not None else "",
        "source_run_id": source_run_id if source_run_id is not None else "",
        "status": summary.get("status", "completed"),
        "solver_family": summary.get("solver_family", ""),
        "policy_mode": summary.get("policy_mode", ""),
        "optimality_gap": _summary_optimality_gap(summary),
        "learning_score": summary.get("learning_score", ""),
        "raw_ar": summary.get("raw_ar", ""),
        "raw_feasible": summary.get("raw_feasible", ""),
        "raw_feasibility_rate": summary.get("raw_feasibility_rate", ""),
        "repaired_optimality_gap": _summary_repaired_optimality_gap(summary),
        "repaired_ar": summary.get("repaired_ar", ""),
        "repaired_feasible": summary.get("repaired_feasible", ""),
        "repair_changed": summary.get("repair_changed", ""),
        "total_attempts": summary.get("total_attempts", ""),
        "best_attempt_index": summary.get("best_attempt_index", ""),
        "first_feasible_attempt": summary.get("first_feasible_attempt", ""),
        "first_ar_ge_0_5_attempt": summary.get("first_ar_ge_0_5_attempt", ""),
        "shots_to_first_feasible": summary.get("shots_to_first_feasible", ""),
        "shots_to_ar_ge_0_5": summary.get("shots_to_ar_ge_0_5", ""),
        "total_run_shots": summary.get("total_run_shots", ""),
        "total_wall_time_s": summary.get("total_wall_time_s", ""),
        "winning_policy_path": summary.get("winning_policy_path", ""),
        "summary_json_path": summary.get("summary_json_path", ""),
        "attempts_jsonl_path": summary.get("attempts_jsonl_path", ""),
    }


def _register_summary(
    study_dir: Path,
    study_id: str,
    case: dict,
    variant: str,
    run_tag: str,
    prompt_variant: str,
    budget_mode: str,
    matched_total_shots: int | None,
    source_run_id: int | None,
    summary: dict,
) -> int:
    runs_path = study_dir / "runs.tsv"
    attempts_path = study_dir / "attempts.jsonl"
    run_id = _next_run_id(runs_path)
    row = _study_row_from_summary(
        study_id=study_id,
        run_id=run_id,
        case=case,
        variant=variant,
        run_tag=run_tag,
        prompt_variant=prompt_variant,
        budget_mode=budget_mode,
        matched_total_shots=matched_total_shots,
        source_run_id=source_run_id,
        summary=summary,
    )
    _append_run_row(runs_path, row)
    attempts = summary.get("attempts", [])
    if isinstance(attempts, list) and attempts:
        _append_attempt_records(
            attempts_path,
            run_id,
            {
                "study_id": study_id,
                "problem": summary.get("problem", case["problem_spec"]),
                "problem_type": case["problem_type"],
                "size": case["size"],
                "seed": case["seed"],
                "split": case["split"],
                "variant": variant,
                "run_tag": run_tag,
                "prompt_variant": prompt_variant,
                "budget_mode": budget_mode,
            },
            attempts,
        )
    return run_id


def _execute_variant(
    study_dir: Path,
    study_id: str,
    case: dict,
    variant: str,
    run_tag: str,
    prompt_variant: str,
    max_attempts: int,
    backend_mode: str,
    solver_family: str | None,
    policy: dict | None,
    budget_mode: str = "trajectory",
    matched_total_shots: int | None = None,
    source_run_id: int | None = None,
) -> tuple[int, dict]:
    run_id = _next_run_id(study_dir / "runs.tsv")
    artifact_dir = study_dir / "artifacts" / f"{run_id:04d}"
    summary_path = artifact_dir / "summary.json"
    attempts_path = artifact_dir / "attempts.jsonl"
    winning_policy_path = artifact_dir / "winning_policy.json"

    policy_json = json.dumps(policy) if policy is not None else None
    summary = run_experiment(
        problem_spec=case["problem_spec"],
        backend_mode=backend_mode,
        solver_family=solver_family,
        max_attempts=max_attempts,
        timeout=600,
        no_results_log=True,
        no_progress_plot=True,
        policy_json=policy_json,
        run_tag=run_tag,
        summary_json=summary_path,
        attempts_jsonl=attempts_path,
        winning_policy_json=winning_policy_path,
    )

    final_run_id = _register_summary(
        study_dir=study_dir,
        study_id=study_id,
        case=case,
        variant=variant,
        run_tag=run_tag,
        prompt_variant=prompt_variant,
        budget_mode=budget_mode,
        matched_total_shots=matched_total_shots,
        source_run_id=source_run_id,
        summary=summary,
    )
    return final_run_id, summary


def run_manifest(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    study_id = str(manifest.get("study_id") or manifest_path.stem)
    output_root = Path(manifest.get("output_root", "studies"))
    study_dir = output_root / study_id
    prompt_variant = str(manifest.get("prompt_variant", "full"))
    backend_mode = str(manifest.get("backend", "ideal_mps"))
    max_attempts = int(manifest.get("max_attempts", 5))
    include_budget_matched = bool(manifest.get("include_budget_matched", True))
    requested_variants = list(dict.fromkeys(str(item) for item in manifest.get("variants", ["adaptive_full"])))

    cases = expand_study_cases(manifest)
    print(f"Study: {study_id}")
    print(f"Output: {study_dir}")
    print(f"Cases: {len(cases)}")

    for case in cases:
        print(f"\n[{case['problem_spec']}]")
        adaptive_summary = None
        adaptive_run_id = None
        adaptive_needed = (
            "adaptive_full" in requested_variants
            or any(variant in DERIVED_VARIANTS for variant in requested_variants)
            or include_budget_matched
        )

        if adaptive_needed:
            adaptive_run_id, adaptive_summary = _execute_variant(
                study_dir=study_dir,
                study_id=study_id,
                case=case,
                variant="adaptive_full",
                run_tag="adaptive-full",
                prompt_variant=prompt_variant,
                max_attempts=max_attempts,
                backend_mode=backend_mode,
                solver_family=None,
                policy=None,
            )
            print(
                f"  adaptive_full: gap={_summary_optimality_gap(adaptive_summary)} "
                f"attempts={adaptive_summary.get('total_attempts')}"
            )

        followup_policies = build_followup_policies(adaptive_summary or {})
        problem = get_single_instance(case["problem_type"], case["size"], case["seed"])

        for variant in requested_variants:
            if variant == "adaptive_full":
                continue

            family = None
            policy = None
            source_run_id = adaptive_run_id
            if variant in DERIVED_VARIANTS:
                policy = followup_policies.get(variant)
                if not policy:
                    print(f"  {variant}: skipped (no derived policy)")
                    continue
                family = str(policy.get("solver_family", "vqe"))
            elif variant in STATIC_VARIANTS:
                family, policy = build_static_variant_policy(problem, variant)
            else:
                raise ValueError(f"Unsupported study variant: {variant}")

            run_id, summary = _execute_variant(
                study_dir=study_dir,
                study_id=study_id,
                case=case,
                variant=variant,
                run_tag=variant.replace("_", "-"),
                prompt_variant=prompt_variant,
                max_attempts=1,
                backend_mode=backend_mode,
                solver_family=family,
                policy=policy,
                source_run_id=source_run_id,
            )
            print(f"  {variant}: gap={_summary_optimality_gap(summary)} attempts={summary.get('total_attempts')}")

            if include_budget_matched and adaptive_summary is not None:
                target_total_shots = int(adaptive_summary.get("total_run_shots", 0) or 0)
                if target_total_shots > 0:
                    reference_iterations = None
                    if variant in DERIVED_VARIANTS:
                        reference_iterations = int(adaptive_summary.get("winning_optimizer_iterations", 0) or 0)
                    budget_policy = rebudget_policy(policy or {}, target_total_shots, reference_iterations)
                    budget_run_tag = f"{variant.replace('_', '-')}-equal-budget"
                    budget_variant = f"{variant}_equal_budget"
                    _, budget_summary = _execute_variant(
                        study_dir=study_dir,
                        study_id=study_id,
                        case=case,
                        variant=budget_variant,
                        run_tag=budget_run_tag,
                        prompt_variant=prompt_variant,
                        max_attempts=1,
                        backend_mode=backend_mode,
                        solver_family=str(budget_policy.get("solver_family", family or "vqe")),
                        policy=budget_policy,
                        budget_mode="equal_budget",
                        matched_total_shots=target_total_shots,
                        source_run_id=run_id,
                    )
                    print(
                        f"  {budget_variant}: gap={_summary_optimality_gap(budget_summary)} "
                        f"shots={budget_summary.get('total_run_shots')}"
                    )


def register_existing_run(
    study_dir: Path,
    study_id: str,
    variant: str,
    prompt_variant: str,
    split: str,
    summary_json: Path,
    attempts_jsonl: Path | None = None,
    run_tag: str | None = None,
    budget_mode: str = "trajectory",
    matched_total_shots: int | None = None,
    source_run_id: int | None = None,
) -> int:
    summary = json.loads(summary_json.read_text())
    if attempts_jsonl is not None and attempts_jsonl.exists():
        attempts = []
        with attempts_jsonl.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                attempts.append(json.loads(line))
        summary["attempts"] = attempts

    case = {
        "problem_type": str(summary.get("problem_type", "knapsack")),
        "size": int(summary.get("size", 0) or 0),
        "seed": int(summary.get("seed", 0) or 0),
        "split": split,
        "problem_spec": str(summary.get("problem_spec", summary.get("problem", ""))),
    }
    return _register_summary(
        study_dir=study_dir,
        study_id=study_id,
        case=case,
        variant=variant,
        run_tag=run_tag or variant.replace("_", "-"),
        prompt_variant=prompt_variant,
        budget_mode=budget_mode,
        matched_total_shots=matched_total_shots,
        source_run_id=source_run_id,
        summary=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run append-only adaptive-control study batches")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a study manifest")
    run_parser.add_argument("--manifest", type=Path, required=True)

    register_parser = subparsers.add_parser("register", help="Register an existing run into a study ledger")
    register_parser.add_argument("--study-dir", type=Path, required=True)
    register_parser.add_argument("--study-id", type=str, required=True)
    register_parser.add_argument("--variant", type=str, required=True)
    register_parser.add_argument("--prompt-variant", type=str, default="full")
    register_parser.add_argument("--split", type=str, default="custom")
    register_parser.add_argument("--summary-json", type=Path, required=True)
    register_parser.add_argument("--attempts-jsonl", type=Path, default=None)
    register_parser.add_argument("--run-tag", type=str, default=None)
    register_parser.add_argument("--budget-mode", type=str, default="trajectory")
    register_parser.add_argument("--matched-total-shots", type=int, default=None)
    register_parser.add_argument("--source-run-id", type=int, default=None)

    args = parser.parse_args()
    if args.command == "run":
        run_manifest(args.manifest)
    else:
        run_id = register_existing_run(
            study_dir=args.study_dir,
            study_id=args.study_id,
            variant=args.variant,
            prompt_variant=args.prompt_variant,
            split=args.split,
            summary_json=args.summary_json,
            attempts_jsonl=args.attempts_jsonl,
            run_tag=args.run_tag,
            budget_mode=args.budget_mode,
            matched_total_shots=args.matched_total_shots,
            source_run_id=args.source_run_id,
        )
        print(f"Registered run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
