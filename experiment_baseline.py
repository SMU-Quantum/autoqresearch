#!/usr/bin/env python3
"""Run the fixed conservative knapsack baseline through the shared engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PROBLEM_SPEC = "knapsack_12"
DEFAULT_RESULTS_PATH = Path("results.tsv")
DEFAULT_PROGRESS_PATH = Path("instance_progress.png")


def _parse_problem_spec(spec: str) -> tuple[str, int, int]:
    parts = spec.split("_")
    problem_type = parts[0]
    size = int(parts[1])
    seed = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("s") else 0
    return problem_type, size, seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the immutable static VQE baseline through experiment.py",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=DEFAULT_PROBLEM_SPEC,
        help="Problem spec (e.g., knapsack_12, knapsack_12_s3)",
    )
    parser.add_argument("--backend", type=str, default="ideal_mps")
    parser.add_argument("--max-attempts", type=int, default=1)
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
    parser.add_argument("--no-results-log", action="store_true")
    parser.add_argument("--no-progress-plot", action="store_true")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="baseline",
        help="Short label recorded in machine-readable outputs and results descriptions.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--attempts-jsonl", type=Path, default=None)
    parser.add_argument("--winning-policy-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from autoqresearch.problems.registry import get_single_instance
    from experiment import build_static_baseline_policy, run_experiment

    problem_type, size, seed = _parse_problem_spec(args.problem)
    problem = get_single_instance(problem_type, size, seed)
    baseline_policy = build_static_baseline_policy(problem)
    summary = run_experiment(
        problem_spec=args.problem,
        backend_mode=args.backend,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        results_file=args.results_file,
        plot_output=args.plot_output,
        no_results_log=args.no_results_log,
        no_progress_plot=args.no_progress_plot,
        policy_json=json.dumps(baseline_policy),
        run_tag=args.run_tag,
        summary_json=args.summary_json,
        attempts_jsonl=args.attempts_jsonl,
        winning_policy_json=args.winning_policy_json,
    )
    return 0 if summary.get("status") != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
