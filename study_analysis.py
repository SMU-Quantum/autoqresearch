#!/usr/bin/env python3
"""Grouped analysis for adaptive-control study ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _primary_metric(row: dict) -> float | None:
    return _safe_float(row.get("optimality_gap"))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def load_study_runs(study_dir: Path) -> list[dict]:
    runs_path = study_dir / "runs.tsv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Study runs file not found: {runs_path}")
    with runs_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def load_study_attempts(study_dir: Path) -> list[dict]:
    attempts_path = study_dir / "attempts.jsonl"
    if not attempts_path.exists():
        return []
    attempts = []
    with attempts_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            attempts.append(json.loads(line))
    return attempts


def _completed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("status", "")).strip().lower() == "completed"]


def same_seed_pair_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in _completed_rows(rows):
        if str(row.get("budget_mode", "trajectory")) != "trajectory":
            continue
        key = (
            row.get("problem"),
            row.get("size"),
            row.get("seed"),
            row.get("split"),
            row.get("prompt_variant"),
        )
        grouped[key][str(row.get("variant", ""))] = row

    pair_rows = []
    for key, variants in grouped.items():
        adaptive = variants.get("adaptive_full")
        if adaptive is None:
            continue
        adaptive_gap = _primary_metric(adaptive)
        if adaptive_gap is None:
            continue
        for challenger in ("static_basic_vqe", "static_final", "static_direct_stage2"):
            challenger_row = variants.get(challenger)
            challenger_gap = _primary_metric(challenger_row) if challenger_row else None
            if challenger_row is None or challenger_gap is None:
                continue
            pair_rows.append(
                {
                    "problem": key[0],
                    "size": key[1],
                    "seed": key[2],
                    "split": key[3],
                    "prompt_variant": key[4],
                    "challenger": challenger,
                    "adaptive_gap": adaptive_gap,
                    "challenger_gap": challenger_gap,
                    "challenger_minus_adaptive_gap": challenger_gap - adaptive_gap,
                    "adaptive_better": 1 if adaptive_gap < challenger_gap else 0,
                }
            )
    return pair_rows


def summarize_same_seed(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in same_seed_pair_rows(rows):
        grouped[str(row["challenger"])].append(row)

    summaries = []
    for challenger, items in sorted(grouped.items()):
        adaptive_gaps = [float(item["adaptive_gap"]) for item in items]
        challenger_gaps = [float(item["challenger_gap"]) for item in items]
        gap_advantages = [float(item["challenger_minus_adaptive_gap"]) for item in items]
        summaries.append(
            {
                "challenger": challenger,
                "count": len(items),
                "adaptive_mean_gap": _mean(adaptive_gaps),
                "adaptive_median_gap": _median(adaptive_gaps),
                "challenger_mean_gap": _mean(challenger_gaps),
                "challenger_median_gap": _median(challenger_gaps),
                "mean_gap_advantage": _mean(gap_advantages),
                "median_gap_advantage": _median(gap_advantages),
                "adaptive_win_rate": _mean([float(item["adaptive_better"]) for item in items]),
            }
        )
    return summaries


def summarize_transfer(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _completed_rows(rows):
        metric = _primary_metric(row)
        if metric is None:
            continue
        key = (
            str(row.get("variant", "")),
            str(row.get("prompt_variant", "")),
            str(row.get("budget_mode", "trajectory")),
        )
        grouped[key][str(row.get("split", "custom"))].append(metric)

    summaries = []
    for key, split_values in sorted(grouped.items()):
        train_mean = _mean(split_values.get("train", []))
        dev_mean = _mean(split_values.get("dev", []))
        test_mean = _mean(split_values.get("test", []))
        summaries.append(
            {
                "variant": key[0],
                "prompt_variant": key[1],
                "budget_mode": key[2],
                "train_mean": train_mean,
                "dev_mean": dev_mean,
                "test_mean": test_mean,
                "dev_transfer_gap": dev_mean - train_mean if train_mean is not None and dev_mean is not None else None,
                "test_transfer_gap": test_mean - train_mean if train_mean is not None and test_mean is not None else None,
            }
        )
    return summaries


def summarize_budget(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in _completed_rows(rows):
        key = (
            str(row.get("variant", "")),
            str(row.get("prompt_variant", "")),
            str(row.get("budget_mode", "trajectory")),
        )
        grouped[key].append(row)

    summaries = []
    for key, items in sorted(grouped.items()):
        summaries.append(
            {
                "variant": key[0],
                "prompt_variant": key[1],
                "budget_mode": key[2],
                "mean_optimality_gap": _mean(
                    [value for value in (_primary_metric(item) for item in items) if value is not None]
                ),
                "mean_total_run_shots": _mean(
                    [value for value in (_safe_float(item.get("total_run_shots")) for item in items) if value is not None]
                ),
                "mean_total_wall_time_s": _mean(
                    [value for value in (_safe_float(item.get("total_wall_time_s")) for item in items) if value is not None]
                ),
                "mean_shots_to_first_feasible": _mean(
                    [value for value in (_safe_float(item.get("shots_to_first_feasible")) for item in items) if value is not None]
                ),
                "mean_shots_to_ar_ge_0_5": _mean(
                    [value for value in (_safe_float(item.get("shots_to_ar_ge_0_5")) for item in items) if value is not None]
                ),
            }
        )
    return summaries


def summarize_ablation(rows: list[dict]) -> list[dict]:
    adaptive_rows = [
        row
        for row in _completed_rows(rows)
        if str(row.get("variant", "")) == "adaptive_full"
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in adaptive_rows:
        grouped[str(row.get("prompt_variant", ""))].append(row)

    transfer_lookup = {
        (entry["variant"], entry["prompt_variant"], entry["budget_mode"]): entry
        for entry in summarize_transfer(rows)
    }
    summaries = []
    for prompt_variant, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: _safe_int(item.get("run_id")) or 0)
        first_feasible_index = None
        first_ar_index = None
        for index, row in enumerate(ordered, start=1):
            feasible = str(row.get("raw_feasible", "")).lower() in {"1", "true"}
            raw_ar = _safe_float(row.get("raw_ar"))
            if first_feasible_index is None and feasible:
                first_feasible_index = index
            if first_ar_index is None and raw_ar is not None and raw_ar >= 0.5:
                first_ar_index = index

        transfer = transfer_lookup.get(("adaptive_full", prompt_variant, "trajectory"), {})
        scores = [
            value
            for value in (_primary_metric(row) for row in items)
            if value is not None
        ]
        summaries.append(
            {
                "prompt_variant": prompt_variant,
                "completed_runs": len(items),
                "runs_to_first_feasible": first_feasible_index,
                "runs_to_raw_ar_ge_0_5": first_ar_index,
                "final_best_optimality_gap": min(scores) if scores else None,
                "test_transfer_gap": transfer.get("test_transfer_gap"),
                "dev_transfer_gap": transfer.get("dev_transfer_gap"),
            }
        )
    return summaries


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    header = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(column, "") for column in header])


def plot_same_seed_scatter(rows: list[dict], output_path: Path) -> None:
    pairs = same_seed_pair_rows(rows)
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {
        "static_basic_vqe": "#7f8c8d",
        "static_final": "#1f77b4",
        "static_direct_stage2": "#ff7f0e",
    }
    for challenger in ("static_basic_vqe", "static_final", "static_direct_stage2"):
        subset = [row for row in pairs if row["challenger"] == challenger]
        if not subset:
            continue
        ax.scatter(
            [row["challenger_gap"] for row in subset],
            [row["adaptive_gap"] for row in subset],
            label=challenger,
            alpha=0.8,
            s=45,
            color=colors.get(challenger, "#555555"),
        )
    limits = [
        min(min(row["challenger_gap"], row["adaptive_gap"]) for row in pairs),
        max(max(row["challenger_gap"], row["adaptive_gap"]) for row in pairs),
    ]
    ax.plot(limits, limits, linestyle="--", color="#444444", linewidth=1.0)
    ax.set_xlabel("Static challenger gap")
    ax.set_ylabel("Adaptive gap")
    ax.set_title("Adaptive vs static on matched seeds")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_bars(rows: list[dict], output_path: Path) -> None:
    transfer = [
        row
        for row in summarize_transfer(rows)
        if row["budget_mode"] == "trajectory"
        and row["variant"] in {"adaptive_full", "static_basic_vqe", "static_final", "static_direct_stage2"}
    ]
    if not transfer:
        return
    variants = [row["variant"] for row in transfer]
    train_vals = [row["train_mean"] or 0.0 for row in transfer]
    dev_vals = [row["dev_mean"] or 0.0 for row in transfer]
    test_vals = [row["test_mean"] or 0.0 for row in transfer]
    x = np.arange(len(variants))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, train_vals, width=width, label="train")
    ax.bar(x, dev_vals, width=width, label="dev")
    ax.bar(x + width, test_vals, width=width, label="test")
    ax.set_xticks(x, variants, rotation=20, ha="right")
    ax.set_ylabel("Mean optimality gap")
    ax.set_title("Transfer by split")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_budget_frontier(rows: list[dict], output_path: Path) -> None:
    budget_rows = [row for row in summarize_budget(rows) if row["budget_mode"] == "trajectory"]
    if not budget_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in budget_rows:
        x = row["mean_total_run_shots"]
        y = row["mean_optimality_gap"]
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=55, alpha=0.85)
        ax.annotate(row["variant"], (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("Mean total_run_shots")
    ax.set_ylabel("Mean optimality gap")
    ax.set_title("Budget vs gap frontier")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_attempt_trajectory(attempts: list[dict], output_path: Path, run_tag: str = "adaptive-full") -> None:
    subset = [row for row in attempts if str(row.get("run_tag", "")) == run_tag and str(row.get("status", "")) == "completed"]
    if not subset:
        return
    subset = sorted(subset, key=lambda row: (_safe_int(row.get("run_id")) or 0, _safe_int(row.get("attempt")) or 0))
    first_run_id = subset[0]["run_id"]
    run_attempts = [row for row in subset if row["run_id"] == first_run_id]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [int(row["attempt"]) for row in run_attempts],
        [float(row.get("optimality_gap", 0.0)) for row in run_attempts],
        marker="o",
        label="optimality_gap",
    )
    ax.plot(
        [int(row["attempt"]) for row in run_attempts],
        [float(row.get("learning_score", 0.0)) for row in run_attempts],
        marker="s",
        label="learning_score",
    )
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Metric")
    ax.set_title(f"Representative attempt trajectory ({run_tag}, run_id={first_run_id})")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze append-only adaptive-control study ledgers")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    study_dir = args.study_dir
    analysis_dir = study_dir / "analysis"
    rows = load_study_runs(study_dir)
    attempts = load_study_attempts(study_dir)

    same_seed = summarize_same_seed(rows)
    transfer = summarize_transfer(rows)
    budget = summarize_budget(rows)
    ablation = summarize_ablation(rows)

    _write_tsv(analysis_dir / "same_seed.tsv", same_seed)
    _write_tsv(analysis_dir / "transfer.tsv", transfer)
    _write_tsv(analysis_dir / "budget.tsv", budget)
    _write_tsv(analysis_dir / "ablation.tsv", ablation)

    print(f"Wrote analysis tables to {analysis_dir}")

    if not args.no_plots:
        plot_same_seed_scatter(rows, analysis_dir / "adaptive_vs_static_same_seed_gap.png")
        plot_transfer_bars(rows, analysis_dir / "transfer_by_split_gap.png")
        plot_budget_frontier(rows, analysis_dir / "budget_vs_gap_frontier.png")
        plot_attempt_trajectory(attempts, analysis_dir / "representative_attempt_trajectory.png")
        print(f"Wrote plots to {analysis_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
