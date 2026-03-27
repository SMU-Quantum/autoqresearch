#!/usr/bin/env python3
"""
prepare.py — One-time setup for AutoQResearch.

Run this once to:
  1. Validate your Qiskit installation
  2. Generate benchmark problem instances (train/dev/test splits)
  3. Compute classical ground-truth solutions
  4. Verify the knapsack-first adaptive experiment entrypoint runs correctly

Usage:
    ./.venv/bin/python prepare.py [--suite quick|standard|full] [--validate-only]
"""

import sys
import time
import json
import pickle
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Verify all required packages are installed."""
    deps = {
        "qiskit": "qiskit",
        "qiskit_aer": "qiskit-aer",
        "qiskit_algorithms": "qiskit-algorithms",
        "qiskit_optimization": "qiskit-optimization",
        "networkx": "networkx",
        "scipy": "scipy",
    }

    missing = []
    for module, pip_name in deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    # Print versions
    import qiskit
    print(f"[ok] qiskit {qiskit.__version__}")
    import qiskit_aer
    print(f"[ok] qiskit-aer {qiskit_aer.__version__}")
    print("[ok] All dependencies satisfied.\n")


def generate_instances(suite: str = "quick"):
    """Generate cached benchmark instances and save them to disk."""
    from autoqresearch.problems.registry import generate_all_splits

    cache_suite = "quick" if suite == "quick" else "core"
    print(f"[*] Generating benchmark instances (suite={suite}, cache_profile={cache_suite})...")
    t0 = time.time()
    splits = generate_all_splits(suite=cache_suite)

    # Save to disk
    cache_dir = Path(".cache/autoqresearch")
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ptype, size_dict in splits.items():
        for size, split in size_dict.items():
            key = f"{ptype}_{size}"
            fname = cache_dir / f"{key}.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(split, f)

            summary[key] = {
                "train": len(split.train),
                "dev": len(split.dev),
                "test": len(split.test),
                "train_optimal": [inst.optimal_value for inst in split.train],
            }
            print(f"  {key}: {len(split.train)} train, {len(split.dev)} dev, "
                  f"{len(split.test)} test  (optimal: {split.train[0].optimal_value:.2f})")

    # Save summary
    with open(cache_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"[ok] Generated {sum(len(v) for v in splits.values())} problem sets "
          f"in {elapsed:.1f}s")
    print(f"[ok] Cached to {cache_dir}/\n")
    return splits


def validate_entrypoint():
    """Run the active knapsack experiment CLI to verify the current stack."""
    print("[*] Running knapsack-first experiment validation...")

    repo_root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "experiment.py",
        "--problem",
        "knapsack_5",
        "--backend",
        "ideal_mps",
        "--max-attempts",
        "1",
        "--timeout",
        "120",
        "--no-results-log",
        "--no-progress-plot",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if completed.returncode != 0:
        print("[ERROR] experiment.py validation failed.")
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip())
        sys.exit(1)

    metrics = _parse_metric_output(completed.stdout)
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
        "solver_family",
        "problem",
        "backend",
    }
    missing = sorted(required - set(metrics))
    if missing:
        print(f"[ERROR] experiment.py output is missing fields: {missing}")
        sys.exit(1)

    print(f"  Problem: {metrics['problem']}")
    print(f"  Solver family: {metrics['solver_family']}")
    print(f"  Optimality gap: {float(metrics['optimality_gap']):.4f}")
    print(f"  Raw feasible: {metrics['raw_feasible']}")
    print(f"  Time: {float(str(metrics['total_time']).rstrip('s')):.1f}s")
    print("[ok] Knapsack experiment validation passed.\n")


def _parse_metric_output(text: str) -> dict[str, str]:
    """Parse `key: value` lines from the experiment CLI output."""

    metrics: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AutoQResearch setup")
    parser.add_argument("--suite", choices=["quick", "standard", "full"], default="quick",
                        help="Cache profile aligned with the active repo dialect; standard/full both use the core cached split set")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only check dependencies and run the experiment entrypoint smoke test")
    args = parser.parse_args()

    print("=" * 60)
    print("  AutoQResearch — Setup")
    print("=" * 60 + "\n")

    check_dependencies()

    if not args.validate_only:
        generate_instances(suite=args.suite)

    validate_entrypoint()

    print("=" * 60)
    print("  Setup complete. Ready for autonomous experimentation.")
    print("  Next: point your coding agent at program.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
