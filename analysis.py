#!/usr/bin/env python3
"""Analysis helpers for AutoQResearch diagnostic and suite-ledger plots."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATUS_KEEP = "KEEP"
STATUS_DISCARD = "DISCARD"
STATUS_CRASH = "CRASH"


def _safe_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalize_status(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {STATUS_KEEP, STATUS_DISCARD, STATUS_CRASH}:
        return normalized
    return None


def _infer_crash(row: dict, metric_column: str) -> bool:
    metric = _safe_float(row.get(metric_column))
    if metric is None:
        return True

    approx = _safe_float(row.get("approx_ratio"))
    depth = _safe_float(row.get("depth"))
    wall_time = _safe_float(row.get("wall_time_s"))
    if wall_time is None:
        wall_time = _safe_float(row.get("wall_time"))
    feasible = str(row.get("feasible", "")).strip()

    if metric_column == "composite_score":
        if (
            metric <= -1.0
            and approx == 0.0
            and depth == 0.0
            and wall_time == 0.0
            and feasible in {"0", "False", "false", ""}
        ):
            return True

    return False


def detect_metric(rows: list[dict], preferred: str | None = None) -> tuple[str, bool]:
    """Return the metric column and its optimization direction."""

    if not rows:
        raise ValueError("No rows available for analysis.")

    columns = set(rows[0].keys())
    if preferred:
        if preferred not in columns:
            raise ValueError(f"Metric '{preferred}' is not present in the TSV.")
        higher_is_better = preferred not in {
            "val_bpb",
            "loss",
            "error",
            "optimality_gap",
            "suite_average_gap",
            "repaired_optimality_gap",
            "mean_optimality_gap",
        }
        return preferred, higher_is_better

    if "suite_average_gap" in columns:
        return "suite_average_gap", False
    if "optimality_gap" in columns:
        return "optimality_gap", False
    # Legacy fallbacks for older generic diagnostics that are outside the
    # active knapsack objective.
    if "composite_score" in columns:
        return "composite_score", True
    if "raw_composite_score" in columns:
        return "raw_composite_score", True
    if "val_bpb" in columns:
        return "val_bpb", False
    if "approx_ratio" in columns:
        return "approx_ratio", True
    if "raw_ar" in columns:
        return "raw_ar", True
    raise ValueError(f"Could not infer metric column from columns: {sorted(columns)}")


def load_results(results_path: str | Path = "results.tsv", metric: str | None = None) -> dict:
    """Load results.tsv and normalize rows for plotting and summaries."""

    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    metric_column, higher_is_better = detect_metric(rows, preferred=metric)

    normalized_rows = []
    best_metric = None
    for index, row in enumerate(rows):
        status = _normalize_status(row.get("status"))
        metric_value = _safe_float(row.get(metric_column))
        if status is None:
            status = STATUS_CRASH if _infer_crash(row, metric_column) else None

        normalized_rows.append(
            {
                "raw_index": index,
                "row": row,
                "metric": metric_value,
                "status": status,
                "description": str(row.get("description", "")).strip(),
            }
        )

    for entry in normalized_rows:
        if entry["status"] == STATUS_CRASH:
            continue
        metric_value = entry["metric"]
        if metric_value is None:
            entry["status"] = STATUS_CRASH
            continue
        if best_metric is None:
            entry["status"] = STATUS_KEEP
            best_metric = metric_value
            continue

        improved = metric_value > best_metric if higher_is_better else metric_value < best_metric
        if entry["status"] is None:
            entry["status"] = STATUS_KEEP if improved else STATUS_DISCARD

        if entry["status"] == STATUS_KEEP:
            best_metric = metric_value

    valid_rows = [entry for entry in normalized_rows if entry["status"] != STATUS_CRASH]
    if not valid_rows:
        raise ValueError("All rows were classified as crashes; nothing to plot.")

    for plotted_index, entry in enumerate(valid_rows):
        entry["plot_index"] = plotted_index

    kept_rows = [entry for entry in valid_rows if entry["status"] == STATUS_KEEP]
    discarded_rows = [entry for entry in valid_rows if entry["status"] == STATUS_DISCARD]
    crash_rows = [entry for entry in normalized_rows if entry["status"] == STATUS_CRASH]

    return {
        "path": path,
        "metric_column": metric_column,
        "higher_is_better": higher_is_better,
        "rows": normalized_rows,
        "valid_rows": valid_rows,
        "kept_rows": kept_rows,
        "discarded_rows": discarded_rows,
        "crash_rows": crash_rows,
    }


def _truncate(text: str, length: int = 45) -> str:
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def _compact_keep_label(entry: dict) -> str:
    """Build a short annotation label for a kept experiment."""

    row = entry.get("row", {})
    solver = str(row.get("solver", "")).strip().lower()
    description = str(entry.get("description", "")).strip().lower()
    experiment_id = str(row.get("experiment_id", "")).strip() or str(entry["plot_index"] + 1)

    def find_depth_token() -> str:
        for token in ("vqe_reps=1", "vqe_reps=2", "vqe_reps=3", "reps=1", "reps=2", "reps=3"):
            if token in description:
                return " r" + token.split("=")[1]
        return ""

    if solver == "qaoa_standard":
        summary = "QAOA"
        if "reps=1" in description:
            summary += " r1"
        elif "reps=2" in description:
            summary += " r2"
        elif "reps=3" in description:
            summary += " r3"
        if "interp" in description:
            summary += " interp"
    elif solver == "vqe_standard":
        summary = "VQE"
        if "real_amplitudes" in description:
            summary += " RA"
        elif "efficient_su2" in description:
            summary += " eSU2"
        if "vqe_reps=1" in description:
            summary += " r1"
        elif "vqe_reps=2" in description:
            summary += " r2"
    elif solver.startswith("qrao_2_"):
        summary = "QRAO QRAC2" + find_depth_token()
    elif solver.startswith("qrao_3_"):
        summary = "QRAO QRAC3" + find_depth_token()
        if solver.endswith("_magic"):
            summary += " magic"
    elif solver == "qaoa_cvar":
        summary = "CVaR-QAOA"
    elif solver == "qaoa_warmstart":
        summary = "WS-QAOA"
    elif solver == "qaoa_multiangle":
        summary = "MA-QAOA"
    elif solver == "vqe_cvar":
        summary = "CVaR-VQE"
    elif solver.startswith("pce_k"):
        summary = solver.replace("_", " ").upper()
    else:
        summary = _truncate(str(row.get("solver", "")).strip() or "Experiment", 18)

    return f"#{experiment_id} {summary}"


def print_summary(data: dict) -> None:
    """Print a compact text summary mirroring Karpathy's notebook."""

    rows = data["rows"]
    kept_rows = data["kept_rows"]
    discarded_rows = data["discarded_rows"]
    crash_rows = data["crash_rows"]
    metric_column = data["metric_column"]
    higher_is_better = data["higher_is_better"]

    print(f"Total experiments: {len(rows)}")
    print(f"Metric: {metric_column} ({'higher' if higher_is_better else 'lower'} is better)")
    print(
        f"Outcomes: keep={len(kept_rows)} discard={len(discarded_rows)} crash={len(crash_rows)}"
    )

    if kept_rows:
        baseline = kept_rows[0]["metric"]
        best = max(row["metric"] for row in kept_rows) if higher_is_better else min(
            row["metric"] for row in kept_rows
        )
        delta = best - baseline if higher_is_better else baseline - best
        print(f"Baseline {metric_column}: {baseline:.6f}")
        print(f"Best {metric_column}:     {best:.6f}")
        print(f"Total improvement:        {delta:+.6f}")

        print("\nKept experiments:")
        for row in kept_rows:
            description = _truncate(row["description"] or "(no description)", 70)
            print(
                f"  #{row['plot_index']:3d}  {metric_column}={row['metric']:.6f}  {description}"
            )


def make_progress_plot(
    data: dict,
    output_path: str | Path = "progress.png",
    title: str | None = None,
    annotate: bool = True,
) -> Path:
    """Generate the Karpathy-style progress plot."""

    metric_column = data["metric_column"]
    higher_is_better = data["higher_is_better"]
    valid_rows = data["valid_rows"]
    kept_rows = data["kept_rows"]
    discarded_rows = data["discarded_rows"]

    baseline = valid_rows[0]["metric"]
    best = max(row["metric"] for row in valid_rows) if higher_is_better else min(
        row["metric"] for row in valid_rows
    )
    span = abs(best - baseline)
    margin = span * 0.15 if span > 0 else max(abs(baseline) * 0.02, 0.01)

    fig, ax = plt.subplots(figsize=(16, 8))

    disc_x = [row["plot_index"] for row in discarded_rows]
    disc_y = [row["metric"] for row in discarded_rows]
    if disc_x:
        ax.scatter(
            disc_x,
            disc_y,
            c="#cccccc",
            s=18,
            alpha=0.55,
            zorder=2,
            label="Discarded",
        )

    kept_x = [row["plot_index"] for row in kept_rows]
    kept_y = [row["metric"] for row in kept_rows]
    ax.scatter(
        kept_x,
        kept_y,
        c="#2ecc71",
        s=70,
        zorder=4,
        label="Kept",
        edgecolors="black",
        linewidths=0.6,
    )

    running_best = []
    current_best = None
    for value in kept_y:
        if current_best is None:
            current_best = value
        else:
            current_best = max(current_best, value) if higher_is_better else min(current_best, value)
        running_best.append(current_best)
    ax.step(
        kept_x,
        running_best,
        where="post",
        color="#27ae60",
        linewidth=2.4,
        alpha=0.72,
        zorder=3,
        label="Running best",
    )

    if annotate:
        max_plot_index = max(kept_x) if kept_x else 0
        y_offsets = [16, -20, 20, -24, 24, -28]
        for index, row in enumerate(kept_rows):
            prefer_left = row["plot_index"] >= max_plot_index * 0.7
            dx = -14 if prefer_left else 14
            dy = y_offsets[index % len(y_offsets)]
            label = _compact_keep_label(row)
            ax.annotate(
                label,
                (row["plot_index"], row["metric"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
                fontweight="bold",
                color="#14532d",
                ha="right" if prefer_left else "left",
                va="bottom" if dy > 0 else "top",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#2ecc71",
                    "linewidth": 0.8,
                    "alpha": 0.92,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#27ae60",
                    "linewidth": 0.8,
                    "alpha": 0.65,
                    "shrinkA": 4,
                    "shrinkB": 5,
                },
            )

    total_experiments = len(data["rows"])
    total_kept = len(kept_rows)
    axis_label = metric_column.replace("_", " ")
    if title is None:
        title = f"AutoQResearch Progress: {total_experiments} Experiments, {total_kept} Kept Improvements"

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Experiment #", fontsize=15)
    ax.set_ylabel(
        f"{axis_label} ({'higher' if higher_is_better else 'lower'} is better)",
        fontsize=15,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=11)
    ax.margins(x=0.05)

    if higher_is_better:
        ax.set_ylim(baseline - margin, best + margin)
    else:
        ax.set_ylim(best - margin, baseline + margin)

    plt.tight_layout()
    output = Path(output_path)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AutoQResearch experiment logs")
    parser.add_argument("--results", type=Path, default=Path("results.tsv"))
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("progress.png"))
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    data = load_results(args.results, metric=args.metric)
    print_summary(data)
    output = make_progress_plot(
        data,
        output_path=args.output,
        title=args.title,
        annotate=not args.no_annotate,
    )
    print(f"\nSaved progress plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
