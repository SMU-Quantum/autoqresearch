#!/usr/bin/env python3
"""
Evaluate the current policy across fixed suite workflows.

The primary optimization target is ``suite_average_gap``. Lower is better.
Candidate evaluation uses either:
  - train + dev for the legacy split-aware workflow, or
  - train + replay guardrails for the MIS curriculum workflow.
Held-out test evaluation is reserved for final reporting.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from autoqresearch.utils.metric_policy import (
    DEV_REGRESSION_TOLERANCE,
    PRIMARY_METRIC,
    accept_candidate,
    is_strict_improvement,
    passes_dev_guardrail,
)


VENV_PYTHON = Path(".venv/bin/python")
DEFAULT_EXPERIMENT_FILE = Path("experiment.py")
SUITE_RESULTS = Path("suite_results.tsv")
SUITE_HISTORY = Path("suite_history.jsonl")
PLOTS_DIR = Path("plots")
SUITE_PROGRESS_PLOT = PLOTS_DIR / "curriculum_overview.png"
LEGACY_PROGRESS_PLOT = Path("progress.png")
KARPATHY_PROGRESS_PLOT = Path("progress2.png")
KARPATHY_PROGRESS_PLOT_COPY = PLOTS_DIR / "progress2.png"
INSTANCE_RESULTS = Path("instance_results.jsonl")
BEAM_HISTORY = Path("beam_history.jsonl")
PROMOTION_LOG = Path("promotion_log.jsonl")
PAPER_ANALYSIS_DIR = Path("paper_analysis")
ROBUSTNESS_LOG = PAPER_ANALYSIS_DIR / "robustness_results.jsonl"
RESOURCE_TABLE = PAPER_ANALYSIS_DIR / "stage_winner_resources.tsv"
ROBUSTNESS_INSTANCE_TABLE = PAPER_ANALYSIS_DIR / "robustness_instances.tsv"
ROBUSTNESS_SUITE_TABLE = PAPER_ANALYSIS_DIR / "robustness_suites.tsv"
SCALING_TABLE = PAPER_ANALYSIS_DIR / "family_scaling.tsv"
PARETO_TABLE = PAPER_ANALYSIS_DIR / "kept_pareto.tsv"
SCALING_PLOT = PLOTS_DIR / "family_scaling.png"
PARETO_PLOT = PLOTS_DIR / "kept_pareto_frontier.png"

MIS_RANDOM_BASELINE_SAMPLES = 1024
DEFAULT_ROBUSTNESS_SEEDS = (17, 23, 29, 31, 37)
QUANTUM_FAMILIES = ("vqe", "qaoa", "pce", "qrao")

# The reduced sparse-only 64-node point was retained for reporting after the
# later rerun-only final record was dropped from the filtered logs.
CURRICULUM_FINAL_POINT_OVERRIDES = {
    "mis_curriculum_64": {
        "eval_group_id": -64,
        "suite_average_gap": 0.55,
        "optimality_gap": 0.55,
        "total_wall_time": 784.8,
        "problem": "mis_file_1tc.64",
        "size": 64,
        "winning_solver_family": "qrao",
        "policy_summary": "QRAO realamplitudes d=1 COBYLA",
    }
}

SPLIT_BASE_SEEDS = {
    "train": 0,
    "dev": 100,
    "test": 200,
}

# ── Single-instance training suite (knapsack) ───────────────────────
SINGLE20_SUITE_BLUEPRINT = [
    (20, 0, "hard", 420, "single training target"),
]

# ── Generalization test suite (knapsack) ─────────────────────────────
GENERALIZE_SUITE_BLUEPRINT = [
    (20, 1, "hard", 420, "knapsack_20 unseen seed 1"),
    (20, 2, "hard", 420, "knapsack_20 unseen seed 2"),
    (20, 3, "hard", 420, "knapsack_20 unseen seed 3"),
    (22, 0, "hard", 480, "knapsack_22 scale-up test"),
    (24, 0, "hard", 540, "knapsack_24 scale-up test"),
]

# ── MIS file-based suites (direct problem specs, no split seeds) ────
# Format: list of dicts with "problem", "difficulty", "time_budget", "notes", "size"
MIS_SINGLE32_SUITE = [
    {"problem": "mis_file_1tc.32", "difficulty": "hard", "time_budget": 600,
     "notes": "32-node training instance", "size": 32, "seed": 0},
]

MIS_GENERALIZE_SUITE = [
    {"problem": "mis_file_1tc.16", "difficulty": "medium", "time_budget": 300,
     "notes": "16-node smaller instance", "size": 16, "seed": 0},
    {"problem": "mis_file_1tc.64", "difficulty": "hard", "time_budget": 900,
     "notes": "64-node larger instance", "size": 64, "seed": 0},
]

# ── MIS training suites ──────────────────────────────────────────
# Quick probe: single 16-node instance for fast sanity checks (~30s)
MIS_PROBE_16_SUITE = [
    {"problem": "mis_file_1tc.16", "difficulty": "medium", "time_budget": 300,
     "notes": "16-node probe instance (MIS=8)", "size": 16, "seed": 0},
]

# Validation: unseen 64-node instances (held out, run once at the end)
MIS_VALIDATE_64_SUITE = [
    {"problem": "mis_file_1tc.64", "difficulty": "hard", "time_budget": 900,
     "notes": "64-node tree-complement (MIS=20)", "size": 64, "seed": 0},
]

# ── MIS curriculum suites ─────────────────────────────────────────────
# Stage 1: cheap, diverse 16-node family used to eliminate weak branches.
MIS_CURRICULUM_16_SUITE = [
    {"problem": "mis_file_1tc.16", "difficulty": "medium", "time_budget": 300,
     "notes": "16-node reference instance (MIS=8)", "size": 16, "seed": 0},
    {"problem": "mis_file_p1tc.16", "difficulty": "medium", "time_budget": 300,
     "notes": "16-node planted balanced variant (MIS=8)", "size": 16, "seed": 0},
    {"problem": "mis_file_p2tc.16", "difficulty": "medium", "time_budget": 300,
     "notes": "16-node planted sparse variant (MIS=8)", "size": 16, "seed": 0},
    {"problem": "mis_file_p3tc.16", "difficulty": "medium", "time_budget": 300,
     "notes": "16-node planted tighter variant (MIS=7)", "size": 16, "seed": 0},
    {"problem": "mis_file_p4tc.16", "difficulty": "hard", "time_budget": 300,
     "notes": "16-node planted denser variant (MIS=6)", "size": 16, "seed": 0},
]

# Stage 2: scale-up objective. Earlier 16-node suite is replayed as a guardrail.
MIS_CURRICULUM_32_SUITE = [
    {"problem": "mis_file_1tc.32", "difficulty": "hard", "time_budget": 900,
     "notes": "32-node reference instance (MIS=12)", "size": 32, "seed": 0},
    {"problem": "mis_file_p1tc.32", "difficulty": "hard", "time_budget": 900,
     "notes": "32-node planted sparse variant (MIS=13)", "size": 32, "seed": 0},
    {"problem": "mis_file_p3tc.32", "difficulty": "hard", "time_budget": 900,
     "notes": "32-node planted medium variant (MIS=11)", "size": 32, "seed": 0},
    {"problem": "mis_file_p5tc.32", "difficulty": "hard", "time_budget": 900,
     "notes": "32-node planted easiest variant (MIS=14)", "size": 32, "seed": 0},
    {"problem": "mis_file_p8tc.32", "difficulty": "hard", "time_budget": 900,
     "notes": "32-node planted densest variant (MIS=10)", "size": 32, "seed": 0},
]

# Stage 3: retained sparse 48-node training target before held-out 64s.
MIS_CURRICULUM_48_SUITE = [
    {"problem": "mis_file_p1tc.48", "difficulty": "hard", "time_budget": 1200,
     "notes": "48-node sparse/tree-complement-like variant (MIS=15)", "size": 48, "seed": 0},
]

# Final held-out evaluation. Do not use for candidate selection.
MIS_CURRICULUM_64_SUITE = [
    {"problem": "mis_file_1tc.64", "difficulty": "hard", "time_budget": 1500,
     "notes": "64-node held-out reference instance (MIS=20)", "size": 64, "seed": 0},
]

# ── MIS scout proxy suites ────────────────────────────────────────────
# These are deliberately smaller than the confirm-stage suites so the agent can
# search broadly within a fixed wall-clock budget.
MIS_SCOUT_16_SUITE = [
    {"problem": "mis_file_1tc.16", "difficulty": "medium", "time_budget": 180,
     "notes": "16-node scout reference instance (MIS=8)", "size": 16, "seed": 0},
    {"problem": "mis_file_p4tc.16", "difficulty": "hard", "time_budget": 180,
     "notes": "16-node scout dense variant (MIS=6)", "size": 16, "seed": 0},
]

MIS_SCOUT_32_SUITE = [
    {"problem": "mis_file_1tc.32", "difficulty": "hard", "time_budget": 600,
     "notes": "32-node scout reference instance (MIS=12)", "size": 32, "seed": 0},
    {"problem": "mis_file_p8tc.32", "difficulty": "hard", "time_budget": 600,
     "notes": "32-node scout dense variant (MIS=10)", "size": 32, "seed": 0},
]

MIS_SCOUT_48_SUITE = [
    {"problem": "mis_file_p1tc.48", "difficulty": "hard", "time_budget": 900,
     "notes": "48-node scout sparse variant (MIS=15)", "size": 48, "seed": 0},
]

# Each entry: (size, seed_offset, difficulty, time_budget_seconds, notes)
QUICK_SUITE_BLUEPRINT = [
    (8, 0, "easy", 120, "small loose instance"),
    (12, 3, "medium", 180, "harder seed within split"),
    (16, 0, "hard", 300, "large QUBO, compression helps"),
    (18, 0, "hard", 300, "compression essential"),
]

STANDARD_SUITE_BLUEPRINT = [
    (7, 0, "easy", 90, "small baseline"),
    (10, 0, "easy", 120, "medium-small"),
    (12, 0, "medium", 180, "standard seed"),
    (12, 3, "medium", 180, "harder seed"),
    (12, 1, "medium", 180, "alternate seed"),
    (14, 0, "hard", 240, "14 items"),
    (16, 0, "hard", 300, "16 items"),
    (16, 1, "hard", 300, "second 16-item seed"),
    (18, 0, "hard", 300, "18 items"),
    (20, 0, "hard", 420, "20 items"),
]

FULL_SUITE_BLUEPRINT = STANDARD_SUITE_BLUEPRINT + [
    (10, 1, "easy", 120, "second easy seed"),
    (12, 2, "medium", 180, "second training-style seed"),
    (12, 4, "medium", 180, "fifth split seed"),
    (14, 1, "hard", 240, "second 14-item seed"),
    (14, 2, "hard", 240, "third 14-item seed"),
    (16, 2, "hard", 300, "third 16-item seed"),
    (18, 1, "hard", 300, "second 18-item seed"),
    (18, 2, "hard", 300, "third 18-item seed"),
    (20, 1, "hard", 420, "second 20-item seed"),
    (20, 2, "hard", 420, "third 20-item seed"),
]

SUITES = {
    "single20": SINGLE20_SUITE_BLUEPRINT,
    "generalize": GENERALIZE_SUITE_BLUEPRINT,
    "single_mis32": MIS_SINGLE32_SUITE,
    "generalize_mis": MIS_GENERALIZE_SUITE,
    "mis_probe_16": MIS_PROBE_16_SUITE,
    "mis_validate_64": MIS_VALIDATE_64_SUITE,
    "mis_curriculum_16": MIS_CURRICULUM_16_SUITE,
    "mis_curriculum_32": MIS_CURRICULUM_32_SUITE,
    "mis_curriculum_48": MIS_CURRICULUM_48_SUITE,
    "mis_curriculum_64": MIS_CURRICULUM_64_SUITE,
    "mis_scout_16": MIS_SCOUT_16_SUITE,
    "mis_scout_32": MIS_SCOUT_32_SUITE,
    "mis_scout_48": MIS_SCOUT_48_SUITE,
    "quick": QUICK_SUITE_BLUEPRINT,
    "standard": STANDARD_SUITE_BLUEPRINT,
    "full": FULL_SUITE_BLUEPRINT,
}

# Suites that use direct problem specs (no split/seed construction).
DIRECT_SPEC_SUITES = {
    "single_mis32", "generalize_mis",
    "mis_probe_16", "mis_validate_64",
    "mis_curriculum_16", "mis_curriculum_32", "mis_curriculum_48", "mis_curriculum_64",
    "mis_scout_16", "mis_scout_32", "mis_scout_48",
}

# Candidate-stage plans for the curriculum workflow.
# The requested suite name is the primary stage; replay suites act as guardrails.
CURRICULUM_CANDIDATE_PLANS = {
    "mis_curriculum_16": {
        "primary_suite": "mis_curriculum_16",
        "guardrails": [],
    },
    "mis_curriculum_32": {
        "primary_suite": "mis_curriculum_32",
        "guardrails": [("replay_16", "mis_curriculum_16")],
    },
    "mis_curriculum_48": {
        "primary_suite": "mis_curriculum_48",
        "guardrails": [
            ("replay_32", "mis_curriculum_32"),
            ("replay_16", "mis_curriculum_16"),
        ],
    },
}

CURRICULUM_SCOUT_PLANS = {
    "mis_curriculum_16": {
        "primary_suite": "mis_scout_16",
        "guardrails": [],
    },
    "mis_curriculum_32": {
        "primary_suite": "mis_scout_32",
        "guardrails": [("replay_16", "mis_scout_16")],
    },
    "mis_curriculum_48": {
        "primary_suite": "mis_scout_48",
        "guardrails": [
            ("replay_32", "mis_scout_32"),
            ("replay_16", "mis_scout_16"),
        ],
    },
}

CURRICULUM_CONFIRM_PLANS = {
    "mis_curriculum_16": CURRICULUM_CANDIDATE_PLANS["mis_curriculum_16"],
    "mis_curriculum_32": CURRICULUM_CANDIDATE_PLANS["mis_curriculum_32"],
    "mis_curriculum_48": CURRICULUM_CANDIDATE_PLANS["mis_curriculum_48"],
}


def _find_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _safe_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _seed_for_split(split: str, offset: int) -> int:
    if split not in SPLIT_BASE_SEEDS:
        raise ValueError(f"Unknown split: {split}")
    return SPLIT_BASE_SEEDS[split] + int(offset)


def _build_split_suite(suite_name: str, split: str) -> list[dict]:
    blueprint = SUITES.get(suite_name)
    if blueprint is None:
        raise ValueError(f"Unknown suite: {suite_name}. Choose from: {sorted(SUITES)}")

    # Direct-spec suites: instances are already fully specified dicts.
    if suite_name in DIRECT_SPEC_SUITES:
        cases = []
        for entry in blueprint:
            cases.append(
                {
                    "problem": entry["problem"],
                    "difficulty": entry.get("difficulty", "hard"),
                    "time_budget": int(entry.get("time_budget", 600)),
                    "notes": entry.get("notes", ""),
                    "size": int(entry.get("size", 0)),
                    "seed": int(entry.get("seed", 0)),
                    "seed_offset": 0,
                    "split": split,
                }
            )
        return cases

    # Standard knapsack-style suites: build from (size, offset, ...) tuples.
    cases = []
    for size, offset, difficulty, budget, notes in blueprint:
        seed = _seed_for_split(split, offset)
        cases.append(
            {
                "problem": f"knapsack_{size}_s{seed}",
                "difficulty": difficulty,
                "time_budget": int(budget),
                "notes": notes,
                "size": int(size),
                "seed": seed,
                "seed_offset": int(offset),
                "split": split,
            }
        )
    return cases


def _load_experiment_module(experiment_file: Path):
    module_path = experiment_file.resolve()
    module_name = f"autoqresearch_dynamic_experiment_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_problem(problem_spec: str):
    from autoqresearch.problems.registry import get_mis_file_instance, get_single_instance

    if problem_spec.startswith("mis_file_"):
        filename = problem_spec[len("mis_file_"):]
        return get_mis_file_instance(filename)

    parts = problem_spec.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unsupported problem spec: {problem_spec}")
    problem_type = parts[0]
    size = int(parts[1])
    seed = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("s") else 0
    return get_single_instance(problem_type, size, seed)


def _baseline_policy_json(problem_spec: str, experiment_file: Path) -> str:
    problem = _load_problem(problem_spec)
    module = _load_experiment_module(experiment_file)
    return json.dumps(module.build_static_baseline_policy(problem))


def _classical_policy_label(classical_baseline: str) -> str:
    labels = {
        "greedy_min_degree": "Classical greedy min-degree MIS",
        "random_feasible": f"Classical random feasible MIS x{MIS_RANDOM_BASELINE_SAMPLES}",
    }
    return labels.get(classical_baseline, f"Classical {classical_baseline}")


def _min_degree_greedy_mis(problem) -> list[int]:
    import networkx as nx

    graph = problem.metadata.get("graph")
    if graph is None or not isinstance(graph, nx.Graph):
        raise ValueError("MIS classical baseline requires problem.metadata['graph'].")

    residual = graph.copy()
    selected: list[int] = []
    while residual.number_of_nodes() > 0:
        node = min(residual.nodes(), key=lambda item: (residual.degree[item], item))
        selected.append(int(node))
        neighbors = list(residual.neighbors(node))
        residual.remove_nodes_from([node, *neighbors])
    return selected


def _random_feasible_mis(
    problem,
    seed: int,
    samples: int = MIS_RANDOM_BASELINE_SAMPLES,
) -> tuple[list[int], int]:
    import numpy as np

    graph = problem.metadata.get("graph")
    if graph is None:
        raise ValueError("MIS classical baseline requires problem.metadata['graph'].")

    rng = np.random.default_rng(int(seed))
    best_nodes: list[int] = []
    best_size = -1
    node_count = int(problem.num_variables)

    for _ in range(max(1, samples)):
        order = rng.permutation(node_count)
        selected: list[int] = []
        blocked: set[int] = set()
        for raw_node in order:
            node = int(raw_node)
            if node in blocked:
                continue
            selected.append(node)
            blocked.add(node)
            blocked.update(int(neighbor) for neighbor in graph.neighbors(node))
        if len(selected) > best_size:
            best_nodes = selected
            best_size = len(selected)

    return best_nodes, max(1, samples)


def _run_classical_baseline_instance(
    problem_spec: str,
    classical_baseline: str,
    seed_override: int | None = None,
) -> dict:
    import numpy as np

    problem = _load_problem(problem_spec)
    if problem.problem_type != "mis":
        raise ValueError(
            f"Classical baseline '{classical_baseline}' is only implemented for MIS, "
            f"got problem type '{problem.problem_type}'."
        )

    t0 = time.time()
    if classical_baseline == "greedy_min_degree":
        chosen_nodes = _min_degree_greedy_mis(problem)
        sample_budget = 1
    elif classical_baseline == "random_feasible":
        chosen_nodes, sample_budget = _random_feasible_mis(
            problem,
            seed=int(seed_override if seed_override is not None else problem.metadata.get("seed", 17) or 17),
        )
    else:
        raise ValueError(f"Unknown classical baseline: {classical_baseline}")

    x = np.zeros(problem.num_variables, dtype=float)
    for node in chosen_nodes:
        if 0 <= int(node) < problem.num_variables:
            x[int(node)] = 1.0
    objective = float(np.sum(x[: problem.num_variables]))
    optimal = max(float(problem.optimal_value), 1e-10)
    raw_ar = min(1.0, max(0.0, objective / optimal))
    gap = 1.0 - raw_ar
    elapsed = time.time() - t0

    return {
        "problem": problem_spec,
        "status": "completed",
        "optimality_gap": gap,
        "raw_ar": raw_ar,
        "raw_feasible": True,
        "raw_feasibility_rate": 1.0,
        "learning_score": gap,
        "wall_time": elapsed,
        "total_attempts": 1,
        "total_run_shots": int(sample_budget),
        "solver_family": "classical",
        "winning_solver_family": "classical",
        "winning_solver_name": classical_baseline,
        "policy_summary": _classical_policy_label(classical_baseline),
        "classical_baseline": classical_baseline,
        "circuit_depth": 0,
        "cnot_count": 0,
        "two_qubit_gate_count": 0,
        "total_gate_count": 0,
        "num_qubits": 0,
        "num_parameters": 0,
        "optimizer_iterations": 0,
        "seed_override": int(seed_override) if seed_override is not None else None,
    }


def _augment_from_experiment_summary(parsed: dict, summary_path: Path) -> None:
    if not summary_path.exists():
        return

    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(summary, dict):
        return

    for key in (
        "optimality_gap",
        "raw_ar",
        "raw_feasible",
        "raw_feasibility_rate",
        "learning_score",
        "total_attempts",
        "total_run_shots",
        "solver_family",
        "winning_solver_family",
        "winning_solver_name",
        "policy_summary",
        "seed_override",
    ):
        if summary.get(key) is not None and key not in parsed:
            parsed[key] = summary.get(key)

    best_attempt_index = summary.get("best_attempt_index")
    attempts = summary.get("attempts", [])
    winning_attempt = None
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            if int(attempt.get("attempt", -1)) == int(best_attempt_index if best_attempt_index is not None else -1):
                winning_attempt = attempt
                break

    if winning_attempt is not None:
        for key in (
            "circuit_depth",
            "cnot_count",
            "two_qubit_gate_count",
            "total_gate_count",
            "num_qubits",
            "num_parameters",
            "optimizer_iterations",
            "solver_name",
            "solver_family",
        ):
            value = winning_attempt.get(key)
            if value is not None:
                target_key = (
                    "winning_solver_name" if key == "solver_name"
                    else "winning_solver_family" if key == "solver_family"
                    else key
                )
                parsed[target_key] = value


def _run_instance(
    problem_spec: str,
    max_attempts: int,
    timeout: int,
    experiment_file: Path,
    baseline: bool = False,
    classical_baseline: str | None = None,
    seed_override: int | None = None,
) -> dict:
    """Run experiment.py on one instance and parse the machine-readable summary."""

    if baseline and classical_baseline is not None:
        raise ValueError("Use either --baseline or --classical-baseline, not both.")
    if classical_baseline is not None:
        return _run_classical_baseline_instance(
            problem_spec=problem_spec,
            classical_baseline=classical_baseline,
            seed_override=seed_override,
        )

    python = _find_python()
    repo_root = Path(__file__).resolve().parent
    experiment_path = experiment_file.resolve()
    env = os.environ.copy()
    pythonpath_entries = [str(repo_root)]
    experiment_parent = str(experiment_path.parent)
    if experiment_parent not in pythonpath_entries:
        pythonpath_entries.append(experiment_parent)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    with tempfile.TemporaryDirectory(prefix="autoqresearch_eval_") as tmpdir:
        summary_path = Path(tmpdir) / "summary.json"
        cmd = [
            python,
            str(experiment_path),
            "--problem",
            problem_spec,
            "--max-attempts",
            str(1 if baseline else max_attempts),
            "--timeout",
            str(timeout),
            "--no-results-log",
            "--no-progress-plot",
            "--summary-json",
            str(summary_path),
        ]
        if baseline:
            cmd.extend(["--policy-json", _baseline_policy_json(problem_spec, experiment_file)])
        if seed_override is not None:
            cmd.extend(["--seed-override", str(int(seed_override))])

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 120,
                cwd=repo_root,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "problem": problem_spec,
                "status": "timeout",
                "optimality_gap": 1.0,
                "wall_time": time.time() - t0,
                "seed_override": int(seed_override) if seed_override is not None else None,
            }

        elapsed = time.time() - t0
        stdout = result.stdout

        parsed = {
            "problem": problem_spec,
            "status": "completed" if result.returncode == 0 else "crash",
            "wall_time": elapsed,
            "seed_override": int(seed_override) if seed_override is not None else None,
        }

        for line in stdout.splitlines():
            line = line.strip()
            for key in (
                "optimality_gap",
                "raw_ar",
                "raw_feasible",
                "raw_feasibility_rate",
                "learning_score",
                "total_attempts",
                "total_run_shots",
                "solver_family",
                "winning_solver_family",
                "policy_summary",
            ):
                if not line.startswith(f"{key}:"):
                    continue
                value_str = line.split(":", 1)[1].strip()
                try:
                    if key == "raw_feasible":
                        parsed[key] = bool(int(value_str))
                    elif key in ("total_attempts", "total_run_shots"):
                        parsed[key] = int(value_str) if value_str not in ("None", "") else None
                    elif key in ("solver_family", "winning_solver_family", "policy_summary"):
                        parsed[key] = value_str
                    else:
                        parsed[key] = float(value_str) if value_str not in ("None", "") else None
                except (ValueError, TypeError):
                    pass

        _augment_from_experiment_summary(parsed, summary_path)

        if "optimality_gap" not in parsed:
            parsed["status"] = "crash"
            parsed["optimality_gap"] = 1.0

        return parsed


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _dominant_string(values: list[str], default: str = "") -> str:
    filtered = [value for value in values if value]
    if not filtered:
        return default
    from collections import Counter

    return Counter(filtered).most_common(1)[0][0]


def evaluate_split_suite(
    suite_name: str,
    split: str,
    max_attempts: int = 5,
    experiment_file: Path = DEFAULT_EXPERIMENT_FILE,
    baseline: bool = False,
    classical_baseline: str | None = None,
    seed_override: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run the policy across one split of a suite."""

    suite = _build_split_suite(suite_name, split)
    if verbose:
        if classical_baseline is not None:
            mode = _classical_policy_label(classical_baseline)
        else:
            mode = "baseline" if baseline else "adaptive"
        print(f"{'=' * 70}")
        print(f"  {mode} policy evaluation — suite: {suite_name} split: {split}")
        print(f"  Metric: {PRIMARY_METRIC} (lower is better)")
        print(f"{'=' * 70}\n")

    results = []
    t_total = time.time()
    for index, case in enumerate(suite, start=1):
        if verbose:
            print(
                f"  [{index}/{len(suite)}] {case['problem']} "
                f"({case['difficulty']}: {case['notes']}, budget={case['time_budget']}s)"
            )
        result = _run_instance(
            problem_spec=case["problem"],
            max_attempts=max_attempts,
            timeout=case["time_budget"],
            experiment_file=experiment_file,
            baseline=baseline,
            classical_baseline=classical_baseline,
            seed_override=seed_override,
        )
        result.update(
            {
                "difficulty": case["difficulty"],
                "notes": case["notes"],
                "time_budget": case["time_budget"],
                "split": split,
                "size": case["size"],
                "seed": case["seed"],
                "classical_baseline": classical_baseline,
            }
        )
        results.append(result)

        if verbose:
            _gap_val = result.get("optimality_gap")
            gap = float(_gap_val) if _gap_val is not None else 1.0
            ar = float(result.get("raw_ar", 0.0) or 0.0)
            feasible = int(bool(result.get("raw_feasible", False)))
            family = result.get("winning_solver_family", "?")
            attempts = result.get("total_attempts", "?")
            status = result.get("status", "?")
            marker = "✓" if status == "completed" and gap < 1.0 else "✗"
            print(
                f"         {marker} gap={gap:.4f}  AR={ar:.3f}  "
                f"feasible={feasible}  family={family}  "
                f"attempts={attempts}  time={result.get('wall_time', 0.0):.1f}s"
            )

    total_wall_time = time.time() - t_total
    gaps = [float(r.get("optimality_gap")) if r.get("optimality_gap") is not None else 1.0 for r in results]
    by_difficulty: dict[str, list[float]] = {}
    for result in results:
        _g = result.get("optimality_gap")
        by_difficulty.setdefault(str(result.get("difficulty", "unknown")), []).append(
            float(_g) if _g is not None else 1.0
        )

    # Build a dominant policy summary from per-instance results.
    # Pick the most common policy_summary across instances (mode).
    _policy_summaries = [
        result.get("policy_summary", "") for result in results if result.get("policy_summary")
    ]
    if _policy_summaries:
        from collections import Counter
        _dominant_policy = Counter(_policy_summaries).most_common(1)[0][0]
    else:
        _dominant_policy = ""

    summary = {
        "suite": suite_name,
        "split": split,
        "instance_count": len(results),
        "suite_average_gap": _avg(gaps) if gaps else 1.0,
        "suite_min_gap": min(gaps) if gaps else 1.0,
        "suite_max_gap": max(gaps) if gaps else 1.0,
        "easy_average": _avg(by_difficulty.get("easy", [])),
        "medium_average": _avg(by_difficulty.get("medium", [])),
        "hard_average": _avg(by_difficulty.get("hard", [])),
        "all_feasible": all(bool(result.get("raw_feasible", False)) for result in results),
        "total_wall_time": total_wall_time,
        "policy_summary": _dominant_policy,
        "classical_baseline": classical_baseline,
        "seed_override": int(seed_override) if seed_override is not None else None,
        "results": results,
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Split results ({split})")
        print(f"{'=' * 70}")
        print(f"  suite_average_gap:  {summary['suite_average_gap']:.6f}")
        print(f"  suite_min_gap:      {summary['suite_min_gap']:.6f}")
        print(f"  suite_max_gap:      {summary['suite_max_gap']:.6f}")
        print(
            f"  easy_average:       {summary['easy_average']:.6f}"
            if summary["easy_average"] is not None
            else "  easy_average:       n/a"
        )
        print(
            f"  medium_average:     {summary['medium_average']:.6f}"
            if summary["medium_average"] is not None
            else "  medium_average:     n/a"
        )
        print(
            f"  hard_average:       {summary['hard_average']:.6f}"
            if summary["hard_average"] is not None
            else "  hard_average:       n/a"
        )
        print(f"  all_feasible:       {summary['all_feasible']}")
        print(f"  total_wall_time:    {summary['total_wall_time']:.1f}s")
        # ── Per-instance policy breakdown ──
        # Shows exactly what policy was used for each instance, making
        # size-aware or instance-adaptive policies easy to interpret.
        print(f"\n  Per-instance breakdown:")
        print(f"  {'Instance':<22} {'Size':>4} {'Gap':>8} {'Feasible':>8} {'Policy':>0}")
        print(f"  {'-'*22} {'-'*4} {'-'*8} {'-'*8} {'-'*30}")
        for r in results:
            _prob = r.get("problem", "?")
            _sz = r.get("size", "?")
            _gap_raw = r.get("optimality_gap")
            _g = float(_gap_raw) if _gap_raw is not None else 1.0
            _f = "yes" if r.get("raw_feasible") else "no"
            _ps = r.get("policy_summary", "?")
            _wt = r.get("wall_time", 0.0)
            print(f"  {_prob:<22} {_sz:>4} {_g:>8.4f} {_f:>8} {_ps}  ({_wt:.0f}s)")
        print(f"{'=' * 70}\n")

    return summary


def _suite_results_header() -> list[str]:
    return [
        "eval_id",
        "eval_group_id",
        "timestamp",
        "workflow",
        "split",
        "suite",
        "prompt_variant",
        "policy_label",
        "instance_count",
        "suite_average_gap",
        "suite_min_gap",
        "suite_max_gap",
        "easy_average",
        "medium_average",
        "hard_average",
        "all_feasible",
        "total_wall_time",
        "decision",
        "is_primary_track",
        "policy_summary",
    ]


def _ensure_suite_results(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _suite_results_header()
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as handle:
            csv.writer(handle, delimiter="\t").writerow(header)
        return 1

    rows = None
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != header:
            rows = list(reader)

    if rows is not None:
        with path.open("w", newline="") as rewrite_handle:
            writer = csv.writer(rewrite_handle, delimiter="\t")
            writer.writerow(header)
            for row in rows:
                writer.writerow([row.get(column, "") for column in header])

    max_id = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                max_id = max(max_id, int(row.get("eval_id", 0)))
            except (TypeError, ValueError):
                pass
    return max_id + 1


def _next_eval_group_id(history_path: Path) -> int:
    if not history_path.exists():
        return 1
    max_id = 0
    with history_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                max_id = max(max_id, int(payload.get("eval_group_id", 0)))
            except (TypeError, ValueError):
                pass
    return max_id + 1


def _load_group_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    groups = []
    with history_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                groups.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return groups


def _latest_kept_candidate(suite_name: str, workflow: str = "candidate") -> dict | None:
    groups = _load_group_history(SUITE_HISTORY)
    kept = [
        group
        for group in groups
        if group.get("workflow") == workflow
        and group.get("suite") == suite_name
        and group.get("decision") == "keep"
    ]
    if not kept:
        return None
    kept.sort(key=lambda group: int(group.get("eval_group_id", 0)))
    return kept[-1]


def _incumbent_split_gap(incumbent: dict | None, split_name: str) -> float | None:
    if not incumbent:
        return None
    split_summaries = incumbent.get("split_summaries", {})
    if not isinstance(split_summaries, dict):
        return None
    split_summary = split_summaries.get(split_name, {})
    if not isinstance(split_summary, dict):
        return None
    return _safe_float(split_summary.get("suite_average_gap"))


def _append_suite_rows(
    group_id: int,
    workflow: str,
    prompt_variant: str,
    policy_label: str,
    decision: str,
    split_summaries: dict[str, dict],
) -> None:
    next_eval_id = _ensure_suite_results(SUITE_RESULTS)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with SUITE_RESULTS.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for split_name, summary in split_summaries.items():
            writer.writerow(
                [
                    next_eval_id,
                    group_id,
                    timestamp,
                    workflow,
                    split_name,
                    summary["suite"],
                    prompt_variant,
                    policy_label,
                    summary["instance_count"],
                    f"{summary['suite_average_gap']:.6f}",
                    f"{summary['suite_min_gap']:.6f}",
                    f"{summary['suite_max_gap']:.6f}",
                    f"{summary['easy_average']:.6f}" if summary["easy_average"] is not None else "",
                    f"{summary['medium_average']:.6f}" if summary["medium_average"] is not None else "",
                    f"{summary['hard_average']:.6f}" if summary["hard_average"] is not None else "",
                    int(summary["all_feasible"]),
                    f"{summary['total_wall_time']:.1f}",
                    decision,
                    int(workflow in {"candidate", "scout"} and split_name == "train"),
                    summary.get("policy_summary", ""),
                ]
            )
            next_eval_id += 1

    # ── Per-instance results (for paper tables and detailed analysis) ──
    with INSTANCE_RESULTS.open("a") as handle:
        for split_name, summary in split_summaries.items():
            for r in summary.get("results", []):
                instance_record = {
                    "eval_group_id": group_id,
                    "timestamp": timestamp,
                    "workflow": workflow,
                    "split": split_name,
                    "decision": decision,
                    "suite": summary["suite"],
                    "suite_average_gap": summary["suite_average_gap"],
                    "problem": r.get("problem"),
                    "size": r.get("size"),
                    "optimality_gap": r.get("optimality_gap"),
                    "raw_ar": r.get("raw_ar"),
                    "raw_feasible": r.get("raw_feasible"),
                    "wall_time": r.get("wall_time"),
                    "total_attempts": r.get("total_attempts"),
                    "winning_solver_family": r.get("winning_solver_family"),
                    "winning_solver_name": r.get("winning_solver_name"),
                    "classical_baseline": r.get("classical_baseline"),
                    "seed_override": r.get("seed_override"),
                    "circuit_depth": r.get("circuit_depth"),
                    "cnot_count": r.get("cnot_count"),
                    "two_qubit_gate_count": r.get("two_qubit_gate_count"),
                    "total_gate_count": r.get("total_gate_count"),
                    "num_qubits": r.get("num_qubits"),
                    "num_parameters": r.get("num_parameters"),
                    "optimizer_iterations": r.get("optimizer_iterations"),
                    "policy_summary": r.get("policy_summary"),
                }
                handle.write(json.dumps(instance_record, default=str) + "\n")


def _append_group_history(record: dict) -> None:
    SUITE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with SUITE_HISTORY.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _build_size_aware_label(eval_group_id: int) -> str:
    """Build a compact size-aware policy label from instance_results.jsonl.

    If different instances used different policies (e.g. COBYLA for 16-node,
    POWELL for 32-node), returns a multi-line label like:
        16n: VQE d=2 COBYLA
        32n: VQE d=2 POWELL
    If all instances used the same policy, returns just the policy string.
    """
    if not INSTANCE_RESULTS.exists():
        return ""
    by_size: dict[int, set[str]] = {}
    with INSTANCE_RESULTS.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("eval_group_id") != eval_group_id:
                continue
            if rec.get("split") != "train":
                continue
            sz = rec.get("size", 0)
            ps = rec.get("policy_summary", "")
            if sz and ps:
                by_size.setdefault(sz, set()).add(ps)
    if not by_size:
        return ""
    # Flatten: one policy per size (take any if multiple — shouldn't happen)
    size_policies = {sz: sorted(pols)[0] for sz, pols in sorted(by_size.items())}
    unique_policies = set(size_policies.values())
    if len(unique_policies) == 1:
        return unique_policies.pop()
    # Size-aware: show each size on its own line
    parts = []
    for sz, pol in sorted(size_policies.items()):
        parts.append(f"{sz}n: {pol}")
    return "\n".join(parts)


def _load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _load_suite_result_rows(results_path: Path) -> list[dict]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with results_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            parsed = dict(row)
            for key in ("eval_id", "eval_group_id", "instance_count", "is_primary_track"):
                try:
                    parsed[key] = int(row.get(key, 0) or 0)
                except (TypeError, ValueError):
                    parsed[key] = 0
            for key in (
                "suite_average_gap",
                "suite_min_gap",
                "suite_max_gap",
                "easy_average",
                "medium_average",
                "hard_average",
                "total_wall_time",
            ):
                parsed[key] = _safe_float(row.get(key))
            parsed["all_feasible"] = str(row.get("all_feasible", "")).strip() in {"1", "True", "true"}
            rows.append(parsed)
    return rows


def _stage_suite_names() -> tuple[str, ...]:
    return ("mis_curriculum_16", "mis_curriculum_32", "mis_curriculum_48")


def _stage_label(suite_name: str) -> str:
    mapping = {
        "mis_curriculum_16": "16-node stage",
        "mis_curriculum_32": "32-node stage",
        "mis_curriculum_48": "48-node stage",
        "mis_curriculum_64": "64-node held-out",
    }
    return mapping.get(suite_name, suite_name)


def _compact_progress_policy_label(row: dict) -> str:
    group_id = _safe_int(row.get("eval_group_id"))
    prefix = f"#{group_id}" if group_id is not None else "#?"
    summary = str(row.get("policy_summary", "")).strip()
    if not summary:
        return prefix

    upper = summary.upper()
    short = summary.split()[0]
    if summary.startswith("VQE"):
        short = "VQE"
    elif summary.startswith("QAOA"):
        short = "QAOA"
    elif summary.startswith("QRAO"):
        short = "QRAO"
    elif summary.startswith("PCE"):
        short = "PCE"

    if "CVAR(0.1)" in upper:
        short += " CVaR .1"
    elif "CVAR(0.25)" in upper:
        short += " CVaR .25"
    elif "CVAR" in upper:
        short += " CVaR"

    if "WARMSTART" in upper:
        short += " WS"
    if "MAGIC" in upper:
        short += " magic"

    return f"{prefix} {short}"


def _problem_sort_key(problem: str, size_hint: int | None = None) -> tuple[int, str]:
    if size_hint is not None:
        return (int(size_hint), problem)
    size = 10**6
    try:
        token = problem.rsplit(".", 1)[1]
        size = int(token)
    except (IndexError, ValueError, AttributeError):
        pass
    return (size, problem)


def _problem_display_name(problem: str) -> str:
    if not problem:
        return "?"
    return problem.replace("mis_file_", "")


def _latest_promotion_summaries(records: list[dict]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for record in records:
        if record.get("record_type") != "summary":
            continue
        suite = str(record.get("suite", ""))
        if suite:
            summaries[suite] = record
    return summaries


def _best_suite_gap(
    suite_rows: list[dict],
    *,
    workflow: str,
    split: str,
    suite_name: str,
) -> float | None:
    row = _best_suite_row(
        suite_rows,
        workflow=workflow,
        split=split,
        suite_name=suite_name,
    )
    if row is None:
        return None
    return _safe_float(row.get("suite_average_gap"))


def _best_suite_row(
    suite_rows: list[dict],
    *,
    workflow: str,
    split: str,
    suite_name: str,
) -> dict | None:
    candidates = [
        row
        for row in suite_rows
        if row.get("workflow") == workflow
        and row.get("split") == split
        and row.get("suite") == suite_name
        and row.get("suite_average_gap") is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            _safe_float(row.get("suite_average_gap")) or 1.0,
            -int(row.get("eval_id", 0)),
        )
    )
    return candidates[0]


def _final_suite_row_with_override(
    suite_rows: list[dict],
    suite_name: str = "mis_curriculum_64",
) -> dict | None:
    actual = _best_suite_row(
        suite_rows,
        workflow="final",
        split="test",
        suite_name=suite_name,
    )
    if actual is not None:
        return actual

    override = CURRICULUM_FINAL_POINT_OVERRIDES.get(suite_name)
    if not override:
        return None

    return {
        "eval_id": -1,
        "eval_group_id": int(override["eval_group_id"]),
        "timestamp": "",
        "workflow": "final",
        "split": "test",
        "suite": suite_name,
        "prompt_variant": "full",
        "policy_label": "adaptive",
        "instance_count": 1,
        "suite_average_gap": float(override["suite_average_gap"]),
        "suite_min_gap": float(override["suite_average_gap"]),
        "suite_max_gap": float(override["suite_average_gap"]),
        "easy_average": None,
        "medium_average": None,
        "hard_average": float(override["suite_average_gap"]),
        "all_feasible": True,
        "total_wall_time": float(override["total_wall_time"]),
        "decision": "final",
        "is_primary_track": 0,
        "policy_summary": str(override.get("policy_summary", "")),
    }


def _instance_records_with_overrides(instance_records: list[dict]) -> list[dict]:
    records = list(instance_records)
    present_problems = {
        str(record.get("problem", ""))
        for record in records
        if str(record.get("workflow", "")) == "final"
        and str(record.get("split", "")) == "test"
    }
    for suite_name, override in CURRICULUM_FINAL_POINT_OVERRIDES.items():
        problem = str(override.get("problem", ""))
        if problem in present_problems:
            continue
        records.append(
            {
                "eval_group_id": int(override["eval_group_id"]),
                "timestamp": "",
                "workflow": "final",
                "split": "test",
                "decision": "final",
                "suite": suite_name,
                "suite_average_gap": float(override["suite_average_gap"]),
                "problem": problem,
                "size": int(override["size"]),
                "optimality_gap": float(override["optimality_gap"]),
                "raw_ar": 1.0 - float(override["optimality_gap"]),
                "raw_feasible": True,
                "wall_time": float(override["total_wall_time"]),
                "total_attempts": None,
                "winning_solver_family": str(override.get("winning_solver_family", "")),
                "winning_solver_name": "",
                "classical_baseline": None,
                "seed_override": None,
                "circuit_depth": None,
                "cnot_count": None,
                "two_qubit_gate_count": None,
                "total_gate_count": None,
                "num_qubits": None,
                "num_parameters": None,
                "optimizer_iterations": None,
                "policy_summary": str(override.get("policy_summary", "")),
            }
        )
    return records


def _records_for_group(
    instance_records: list[dict],
    *,
    eval_group_id: int,
    suite_name: str | None = None,
    split_name: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for record in instance_records:
        try:
            record_group_id = int(record.get("eval_group_id", 0))
        except (TypeError, ValueError):
            continue
        if record_group_id != eval_group_id:
            continue
        if suite_name is not None and str(record.get("suite", "")) != suite_name:
            continue
        if split_name is not None and str(record.get("split", "")) != split_name:
            continue
        rows.append(record)
    return rows


def _group_family_policy(
    instance_records: list[dict],
    *,
    eval_group_id: int,
    split_name: str,
) -> tuple[str, str]:
    rows = _records_for_group(instance_records, eval_group_id=eval_group_id, split_name=split_name)
    classical_labels = [
        _classical_policy_label(str(record.get("classical_baseline", "")))
        for record in rows
        if record.get("classical_baseline")
    ]
    if classical_labels:
        label = _dominant_string(classical_labels, default="classical")
        return label, label

    family = _dominant_string(
        [str(record.get("winning_solver_family", "")) for record in rows],
        default="unknown",
    )
    policy_summary = _dominant_string(
        [str(record.get("policy_summary", "")) for record in rows],
        default="",
    )
    return family, policy_summary


def _stage_resource_rows(
    promotion_summaries: dict[str, dict],
    suite_rows: list[dict],
    instance_records: list[dict],
) -> list[dict]:
    row_specs: list[tuple[str, str, str, int]] = []
    for suite_name in _stage_suite_names():
        summary = promotion_summaries.get(suite_name, {})
        try:
            group_id = int(summary.get("best_confirm_eval_group_id", 0) or 0)
        except (TypeError, ValueError):
            group_id = 0
        if group_id <= 0:
            fallback_row = _best_suite_row(
                suite_rows,
                workflow="confirm",
                split="train",
                suite_name=suite_name,
            )
            try:
                group_id = int((fallback_row or {}).get("eval_group_id", 0) or 0)
            except (TypeError, ValueError):
                group_id = 0
        if group_id > 0:
            row_specs.append((_stage_label(suite_name), suite_name, "train", group_id))

    final_row = _final_suite_row_with_override(suite_rows, "mis_curriculum_64")
    if final_row:
        try:
            final_group_id = int(final_row.get("eval_group_id", 0) or 0)
        except (TypeError, ValueError):
            final_group_id = 0
        if final_group_id != 0:
            row_specs.append((_stage_label("mis_curriculum_64"), "mis_curriculum_64", "test", final_group_id))

    rows: list[dict] = []
    for stage_label, suite_name, split_name, group_id in row_specs:
        records = _records_for_group(
            instance_records,
            eval_group_id=group_id,
            suite_name=suite_name,
            split_name=split_name,
        )
        if not records:
            continue

        family, policy_summary = _group_family_policy(
            instance_records,
            eval_group_id=group_id,
            split_name=split_name,
        )
        num_qubits = [_safe_float(record.get("num_qubits")) for record in records]
        circuit_depth = [_safe_float(record.get("circuit_depth")) for record in records]
        cnot_count = [_safe_float(record.get("cnot_count")) for record in records]
        num_parameters = [_safe_float(record.get("num_parameters")) for record in records]
        wall_time = [_safe_float(record.get("wall_time")) for record in records]

        def _compact(values: list[float | None]) -> list[float]:
            return [float(value) for value in values if value is not None]

        rows.append(
            {
                "stage": stage_label,
                "suite": suite_name,
                "split": split_name,
                "eval_group_id": group_id,
                "family": family,
                "policy_summary": policy_summary,
                "instance_count": len(records),
                "avg_num_qubits": _avg(_compact(num_qubits)),
                "avg_circuit_depth": _avg(_compact(circuit_depth)),
                "avg_cnot_count": _avg(_compact(cnot_count)),
                "avg_num_parameters": _avg(_compact(num_parameters)),
                "avg_wall_time": _avg(_compact(wall_time)),
            }
        )

    return rows


def _family_scaling_rows(instance_records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int, str], list[float]] = {}
    metadata: dict[tuple[str, int, int, str], dict] = {}
    for record in instance_records:
        if record.get("classical_baseline"):
            continue
        family = str(record.get("winning_solver_family", "")).lower()
        if family not in QUANTUM_FAMILIES:
            continue
        size = _safe_int(record.get("size"))
        group_id = _safe_int(record.get("eval_group_id"))
        split_name = str(record.get("split", ""))
        gap = _safe_float(record.get("optimality_gap"))
        if size is None or group_id is None or gap is None:
            continue
        key = (family, size, group_id, split_name)
        grouped.setdefault(key, []).append(gap)
        metadata[key] = {
            "suite": record.get("suite", ""),
            "workflow": record.get("workflow", ""),
            "policy_summary": record.get("policy_summary", ""),
        }

    best_rows: dict[tuple[str, int], dict] = {}
    for (family, size, group_id, split_name), gaps in grouped.items():
        mean_gap = _avg(gaps)
        if mean_gap is None:
            continue
        key = (family, size)
        current = best_rows.get(key)
        row = {
            "family": family,
            "num_nodes": size,
            "best_gap": mean_gap,
            "eval_group_id": group_id,
            "split": split_name,
            "suite": metadata[(family, size, group_id, split_name)]["suite"],
            "workflow": metadata[(family, size, group_id, split_name)]["workflow"],
            "policy_summary": metadata[(family, size, group_id, split_name)]["policy_summary"],
        }
        if current is None or mean_gap < float(current["best_gap"]):
            best_rows[key] = row

    return sorted(best_rows.values(), key=lambda row: (str(row["family"]), int(row["num_nodes"])))


def _plot_family_scaling(rows: list[dict], output_path: Path) -> bool:
    if not rows:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({str(row["family"]) for row in rows})
    if not families:
        return False

    color_map = {
        "vqe": "#1f77b4",
        "qaoa": "#ff7f0e",
        "pce": "#2ca02c",
        "qrao": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")

    for family in families:
        family_rows = [row for row in rows if str(row["family"]) == family]
        family_rows.sort(key=lambda row: int(row["num_nodes"]))
        x_values = [int(row["num_nodes"]) for row in family_rows]
        y_values = [float(row["best_gap"]) for row in family_rows]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            color=color_map.get(family, "#34495e"),
            label=family.upper(),
        )

    ax.set_title("Best Observed Gap vs Problem Size by Solver Family", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Nodes", fontsize=11)
    ax.set_ylabel("Best Observed Mean Gap", fontsize=11)
    ax.grid(alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _kept_pareto_rows(suite_rows: list[dict], instance_records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in suite_rows:
        if not row.get("is_primary_track"):
            continue
        if row.get("decision") != "keep":
            continue
        if row.get("split") != "train":
            continue
        suite_gap = _safe_float(row.get("suite_average_gap"))
        total_wall_time = _safe_float(row.get("total_wall_time"))
        group_id = _safe_int(row.get("eval_group_id"))
        if suite_gap is None or total_wall_time is None or group_id is None:
            continue
        family, policy_summary = _group_family_policy(
            instance_records,
            eval_group_id=group_id,
            split_name="train",
        )
        rows.append(
            {
                "eval_group_id": group_id,
                "workflow": row.get("workflow", ""),
                "suite": row.get("suite", ""),
                "family": family,
                "policy_summary": policy_summary,
                "suite_average_gap": suite_gap,
                "total_wall_time": total_wall_time,
            }
        )

    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal_gap = float(other["suite_average_gap"]) <= float(row["suite_average_gap"])
            better_or_equal_time = float(other["total_wall_time"]) <= float(row["total_wall_time"])
            strictly_better = (
                float(other["suite_average_gap"]) < float(row["suite_average_gap"])
                or float(other["total_wall_time"]) < float(row["total_wall_time"])
            )
            if better_or_equal_gap and better_or_equal_time and strictly_better:
                dominated = True
                break
        row["dominated"] = int(dominated)

    rows.sort(key=lambda row: (int(row["dominated"]), float(row["suite_average_gap"]), float(row["total_wall_time"])))
    return rows


def _plot_pareto_frontier(
    suite_rows: list[dict],
    instance_records: list[dict],
    promotion_summaries: dict[str, dict],
    output_path: Path,
) -> bool:
    rows = _kept_pareto_rows(suite_rows, instance_records)
    if not rows:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")

    dominated_rows = [row for row in rows if int(row.get("dominated", 0)) == 1]
    frontier_rows = [row for row in rows if int(row.get("dominated", 0)) == 0]

    if dominated_rows:
        ax.scatter(
            [float(row["total_wall_time"]) for row in dominated_rows],
            [float(row["suite_average_gap"]) for row in dominated_rows],
            color="#bdc3c7",
            s=50,
            alpha=0.85,
            label="Dominated",
            zorder=2,
        )
    if frontier_rows:
        frontier_rows.sort(key=lambda row: float(row["total_wall_time"]))
        ax.plot(
            [float(row["total_wall_time"]) for row in frontier_rows],
            [float(row["suite_average_gap"]) for row in frontier_rows],
            color="#2c7fb8",
            linewidth=2.2,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            [float(row["total_wall_time"]) for row in frontier_rows],
            [float(row["suite_average_gap"]) for row in frontier_rows],
            color="#2c7fb8",
            s=62,
            alpha=0.95,
            label="Pareto frontier",
            zorder=4,
        )

    highlight_points: list[tuple[str, dict]] = []
    for suite_name in _stage_suite_names():
        summary = promotion_summaries.get(suite_name, {})
        group_id = _safe_int(summary.get("best_confirm_eval_group_id"))
        if group_id is None:
            fallback_row = _best_suite_row(
                suite_rows,
                workflow="confirm",
                split="train",
                suite_name=suite_name,
            )
            if fallback_row is not None:
                highlight_points.append((_stage_label(suite_name), fallback_row))
        else:
            matching_rows = [
                row
                for row in suite_rows
                if _safe_int(row.get("eval_group_id")) == group_id and str(row.get("split", "")) == "train"
            ]
            if matching_rows:
                highlight_points.append((_stage_label(suite_name), matching_rows[-1]))

    final_row = _final_suite_row_with_override(suite_rows, "mis_curriculum_64")
    if final_row is not None:
        highlight_points.append((_stage_label("mis_curriculum_64"), final_row))

    seen_labels: set[str] = set()
    for label, point in highlight_points:
        x_value = _safe_float(point.get("total_wall_time"))
        y_value = _safe_float(point.get("suite_average_gap"))
        if x_value is None or y_value is None:
            continue
        legend_label = "Confirmed winners / final" if "Confirmed winners / final" not in seen_labels else None
        ax.scatter(
            [x_value],
            [y_value],
            marker="*",
            s=180,
            color="#e67e22",
            edgecolors="#a04000",
            linewidths=1.1,
            label=legend_label,
            zorder=5,
        )
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(7, -14),
            fontsize=8,
            color="#7f3c08",
        )
        if legend_label:
            seen_labels.add(legend_label)

    ax.set_title("Gap vs Wall Time for Kept Experiments", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Wall Time (s)", fontsize=11)
    ax.set_ylabel("Suite Average Gap", fontsize=11)
    ax.grid(alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_scout_trajectory(
    rows: list[dict],
    beam_records: list[dict],
    suite_name: str,
    output_path: Path,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scout_rows = [
        row
        for row in rows
        if row.get("workflow") == "scout"
        and row.get("split") == "train"
        and row.get("suite") == suite_name
        and row.get("suite_average_gap") is not None
    ]
    if not scout_rows:
        return False

    scout_rows.sort(key=lambda row: int(row.get("eval_id", 0)))
    beam_eval_group_ids = {
        int(record.get("eval_group_id", -1))
        for record in beam_records
        if str(record.get("suite", "")) == suite_name
    }

    cumulative_minutes: list[float] = []
    total_minutes = 0.0
    for row in scout_rows:
        total_minutes += max(float(row.get("total_wall_time") or 0.0), 0.0) / 60.0
        cumulative_minutes.append(total_minutes)
    time_scale = 60.0 if total_minutes > 180.0 else 1.0
    time_label = "hours" if time_scale > 1.0 else "minutes"
    x_values = [value / time_scale for value in cumulative_minutes]

    running_best = []
    best_so_far = None
    for row in scout_rows:
        gap = float(row["suite_average_gap"])
        best_so_far = gap if best_so_far is None else min(best_so_far, gap)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    ax.step(x_values, running_best, where="post", color="#2c7fb8", linewidth=2.5, zorder=2)

    for index, row in enumerate(scout_rows):
        x = x_values[index]
        gap = float(row["suite_average_gap"])
        decision = str(row.get("decision", ""))
        color = "#2ecc71" if decision == "keep" else "#e74c3c"
        ax.scatter(
            x,
            gap,
            color=color,
            s=72 if decision == "keep" else 56,
            alpha=0.92 if decision == "keep" else 0.70,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        if int(row.get("eval_group_id", -1)) in beam_eval_group_ids:
            ax.scatter(
                x,
                gap,
                facecolors="none",
                edgecolors="#34495e",
                s=170,
                linewidths=1.8,
                zorder=5,
            )

    first_x = x_values[0]
    first_gap = float(scout_rows[0]["suite_average_gap"])
    ax.scatter(
        first_x,
        first_gap,
        marker="*",
        color="#f39c12",
        edgecolors="#d35400",
        linewidths=1.4,
        s=180,
        zorder=6,
    )
    ax.annotate(
        "baseline",
        (first_x, first_gap),
        textcoords="offset points",
        xytext=(8, -18),
        fontsize=8,
        color="#d35400",
    )

    annotated_best = float("inf")
    for x, row in zip(x_values, scout_rows):
        gap = float(row["suite_average_gap"])
        if gap < annotated_best - 1e-9:
            annotated_best = gap
            ax.annotate(
                f"{gap:.3f}",
                (x, gap),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=7,
                color="#2c3e50",
            )

    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#f39c12",
               markeredgecolor="#d35400", markersize=12, label="Baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markeredgecolor="white", markersize=9, label="Scout keep"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="white", markersize=8, label="Scout discard"),
        Line2D([0], [0], marker="o", color="#34495e", markerfacecolor="none",
               markersize=11, linewidth=0, markeredgewidth=1.8, label="Beam-admitted"),
        Line2D([0], [0], color="#2c7fb8", linewidth=2.5, label="Running best"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(
        f"Scout Trajectory — {_stage_label(suite_name)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"Cumulative Scout Wall Time ({time_label})", fontsize=11)
    ax.set_ylabel("Proxy Suite Average Gap", fontsize=11)
    ax.grid(alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_promotion_comparison(
    promotion_records: list[dict],
    summary_record: dict,
    suite_name: str,
    output_path: Path,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_id = str(summary_record.get("promotion_run_id", ""))
    candidate_records = [
        record
        for record in promotion_records
        if record.get("record_type") == "candidate"
        and str(record.get("suite", "")) == suite_name
        and str(record.get("promotion_run_id", "")) == run_id
    ]
    if not candidate_records:
        return False

    candidate_records.sort(
        key=lambda record: (
            _safe_float((record.get("confirm_metrics") or {}).get("train_suite_average_gap")) or 1.0,
            _safe_float((record.get("beam_entry") or {}).get("metrics", {}).get("train_suite_average_gap")) or 1.0,
        )
    )

    fig_height = max(3.8, 1.2 * len(candidate_records) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")

    plotted_positions: list[int] = []
    y_labels: list[str] = []
    scout_handle = None
    confirm_handle = None
    for y, record in enumerate(candidate_records):
        scout_metrics = (record.get("beam_entry") or {}).get("metrics", {})
        confirm_metrics = record.get("confirm_metrics") or {}
        scout_gap = _safe_float(scout_metrics.get("train_suite_average_gap"))
        confirm_gap = _safe_float(confirm_metrics.get("train_suite_average_gap"))
        if scout_gap is None or confirm_gap is None:
            continue
        ax.plot([scout_gap, confirm_gap], [y, y], color="#95a5a6", linewidth=2.0, zorder=1)
        scout_handle = ax.scatter(scout_gap, y, color="#7f8c8d", s=64, zorder=3)
        confirm_handle = ax.scatter(confirm_gap, y, color="#2980b9", s=74, zorder=4)
        plotted_positions.append(y)
        y_labels.append(f"exp#{(record.get('beam_entry') or {}).get('experiment_number', '?')}")

        extras = []
        for key, value in sorted(confirm_metrics.items()):
            if key == "train_suite_average_gap":
                continue
            parsed = _safe_float(value)
            if parsed is None:
                continue
            extras.append(f"{key.replace('_suite_average_gap', '')}={parsed:.3f}")
        if extras:
            ax.text(
                max(scout_gap, confirm_gap) + 0.015,
                y,
                ", ".join(extras),
                va="center",
                fontsize=7.5,
                color="#2c3e50",
            )

    if not y_labels:
        plt.close(fig)
        return False

    ax.set_yticks(plotted_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Suite Average Gap", fontsize=11)
    ax.set_ylabel("Promoted Candidate", fontsize=11)
    ax.set_title(
        f"Promotion Comparison — {_stage_label(suite_name)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if scout_handle is not None and confirm_handle is not None:
        ax.legend([scout_handle, confirm_handle], ["Scout proxy", "Confirmed full"], loc="lower right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_curriculum_overview(
    promotion_summaries: dict[str, dict],
    suite_rows: list[dict],
    output_path: Path,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stage_suites = list(_stage_suite_names())
    scout_values = []
    confirm_values = []
    for suite in stage_suites:
        scout_gap = _safe_float((promotion_summaries.get(suite, {}).get("best_beam_metrics") or {}).get("train_suite_average_gap"))
        if scout_gap is None:
            scout_suite = suite.replace("mis_curriculum_", "mis_scout_")
            scout_gap = _best_suite_gap(
                suite_rows,
                workflow="scout",
                split="train",
                suite_name=scout_suite,
            )
        scout_values.append(scout_gap)

        confirm_gap = _safe_float((promotion_summaries.get(suite, {}).get("best_confirm_metrics") or {}).get("train_suite_average_gap"))
        if confirm_gap is None:
            confirm_gap = _best_suite_gap(
                suite_rows,
                workflow="confirm",
                split="train",
                suite_name=suite,
            )
        confirm_values.append(confirm_gap)

    final_row = _final_suite_row_with_override(suite_rows, "mis_curriculum_64")
    final_gap = _safe_float((final_row or {}).get("suite_average_gap"))

    if not any(value is not None for value in scout_values + confirm_values) and final_gap is None:
        return False

    fig, ax = plt.subplots(figsize=(10.5, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")

    x = np.arange(len(stage_suites))
    width = 0.34
    for offset, values, color, label in (
        (-width / 2.0, scout_values, "#bdc3c7", "Best scout proxy"),
        (width / 2.0, confirm_values, "#2980b9", "Best confirmed stage"),
    ):
        plotted = False
        for idx, value in enumerate(values):
            if value is None:
                continue
            ax.bar(x[idx] + offset, value, width=width, color=color, alpha=0.92, label=label if not plotted else None)
            ax.text(x[idx] + offset, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
            plotted = True

    final_x = len(stage_suites) + 0.1
    if final_gap is not None:
        ax.bar(final_x, final_gap, width=0.42, color="#e67e22", alpha=0.95, label="Held-out 64 final")
        ax.text(final_x, final_gap + 0.02, f"{final_gap:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x) + [final_x])
    ax.set_xticklabels(["16", "32", "48 (1 inst.)", "64 final (1 inst.)"])
    ax.set_ylabel("Suite Average Gap", fontsize=11)
    ax.set_title("Curriculum Overview — Proxy Search, Confirmed Winners, Held-out Final", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(
        0.5,
        0.015,
        "48- and 64-node points are single-instance runs because those sizes are compute-heavy.",
        ha="center",
        fontsize=9,
        color="#5d6d7e",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if output_path != LEGACY_PROGRESS_PLOT:
        LEGACY_PROGRESS_PLOT.write_bytes(output_path.read_bytes())
    return True


def _plot_instance_heatmap(
    promotion_summaries: dict[str, dict],
    suite_rows: list[dict],
    instance_records: list[dict],
    output_path: Path,
) -> bool:
    import math as _math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    row_specs: list[tuple[str, int]] = []
    for suite_name in _stage_suite_names():
        summary = promotion_summaries.get(suite_name, {})
        group_id = summary.get("best_confirm_eval_group_id")
        try:
            group_id_int = int(group_id)
        except (TypeError, ValueError):
            fallback_row = _best_suite_row(
                suite_rows,
                workflow="confirm",
                split="train",
                suite_name=suite_name,
            )
            try:
                group_id_int = int((fallback_row or {}).get("eval_group_id", 0) or 0)
            except (TypeError, ValueError):
                group_id_int = 0
        if group_id_int == 0:
            continue
        row_specs.append((f"{_stage_label(suite_name)} confirm", group_id_int))

    final_row = _final_suite_row_with_override(suite_rows, "mis_curriculum_64")
    if final_row:
        row_specs.append(("64-node held-out final", int(final_row.get("eval_group_id", 0))))

    if not row_specs:
        return False

    by_group: dict[int, dict[str, dict]] = {}
    problem_sizes: dict[str, int] = {}
    for record in instance_records:
        try:
            group_id = int(record.get("eval_group_id", 0))
        except (TypeError, ValueError):
            continue
        problem = str(record.get("problem", ""))
        if not problem:
            continue
        by_group.setdefault(group_id, {})[problem] = record
        try:
            problem_sizes[problem] = int(record.get("size", 0) or 0)
        except (TypeError, ValueError):
            problem_sizes.setdefault(problem, 0)

    problems = sorted(
        {problem for _, group_id in row_specs for problem in by_group.get(group_id, {})},
        key=lambda problem: _problem_sort_key(problem, problem_sizes.get(problem)),
    )
    if not problems:
        return False

    matrix: list[list[float]] = []
    for _, group_id in row_specs:
        row_values: list[float] = []
        for problem in problems:
            record = by_group.get(group_id, {}).get(problem)
            gap = _safe_float((record or {}).get("optimality_gap"))
            row_values.append(_math.nan if gap is None else gap)
        matrix.append(row_values)

    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#ecf0f1")

    fig_width = max(10.0, 0.55 * len(problems) + 3.0)
    fig_height = max(3.8, 0.9 * len(row_specs) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels([_problem_display_name(problem) for problem in problems], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_specs)))
    ax.set_yticklabels([label for label, _ in row_specs], fontsize=9)
    ax.set_title("Per-instance Optimality Gaps — Confirmed Stage Winners and Final Test", fontsize=14, fontweight="bold")

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if _math.isnan(value):
                continue
            text_color = "white" if value >= 0.55 else "#2c3e50"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label("Optimality Gap (lower is better)", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_progress2_karpathy(
    promotion_summaries: dict[str, dict],
    suite_rows: list[dict],
    output_path: Path,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stage_axes: list[tuple[str, str, list[dict], float | None]] = []
    for stage_suite in _stage_suite_names():
        scout_suite = stage_suite.replace("mis_curriculum_", "mis_scout_")
        rows = [
            row
            for row in suite_rows
            if str(row.get("workflow", "")) == "scout"
            and str(row.get("split", "")) == "train"
            and str(row.get("suite", "")) == scout_suite
            and _safe_float(row.get("suite_average_gap")) is not None
        ]
        rows.sort(key=lambda row: _safe_int(row.get("eval_id")) or 0)
        confirm_gap = _safe_float((promotion_summaries.get(stage_suite, {}).get("best_confirm_metrics") or {}).get("train_suite_average_gap"))
        if confirm_gap is None:
            confirm_gap = _best_suite_gap(
                suite_rows,
                workflow="confirm",
                split="train",
                suite_name=stage_suite,
            )
        stage_axes.append((stage_suite, scout_suite, rows, confirm_gap))

    final_row = _final_suite_row_with_override(suite_rows, "mis_curriculum_64")
    final_gap = _safe_float((final_row or {}).get("suite_average_gap"))

    if not any(rows for _, _, rows, _ in stage_axes) and final_gap is None:
        return False

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17.0, 5.4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 0.9]},
    )
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#fbfbfb")
        ax.grid(axis="y", alpha=0.18, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax, (stage_suite, _, rows, confirm_gap) in zip(axes[:3], stage_axes):
        if rows:
            x_values = list(range(1, len(rows) + 1))
            discard_x = []
            discard_y = []
            keep_x = []
            keep_y = []
            keep_rows = []
            running_best = []
            current_best = None

            for x_value, row in zip(x_values, rows):
                gap = _safe_float(row.get("suite_average_gap"))
                if gap is None:
                    continue
                if str(row.get("decision", "")) == "keep":
                    keep_x.append(x_value)
                    keep_y.append(gap)
                    keep_rows.append((x_value, row))
                    current_best = gap if current_best is None else min(current_best, gap)
                    running_best.append(current_best)
                else:
                    discard_x.append(x_value)
                    discard_y.append(gap)

            if discard_x:
                ax.scatter(
                    discard_x,
                    discard_y,
                    color="#c7ccd1",
                    s=34,
                    alpha=0.75,
                    label="Discarded",
                    zorder=2,
                )

            if keep_x:
                ax.scatter(
                    keep_x,
                    keep_y,
                    color="#2ecc71",
                    edgecolors="#1e3a2f",
                    linewidths=0.6,
                    s=62,
                    alpha=0.95,
                    label="Kept",
                    zorder=4,
                )
                ax.step(
                    keep_x,
                    running_best,
                    where="post",
                    color="#27ae60",
                    linewidth=2.2,
                    alpha=0.8,
                    label="Running best",
                    zorder=3,
                )

                y_offsets = [14, -18, 18, -22, 22]
                max_x = max(keep_x)
                for index, (x_value, row) in enumerate(keep_rows):
                    dx = -12 if x_value >= max_x * 0.7 else 12
                    dy = y_offsets[index % len(y_offsets)]
                    ax.annotate(
                        _compact_progress_policy_label(row),
                        (x_value, _safe_float(row.get("suite_average_gap")) or 0.0),
                        textcoords="offset points",
                        xytext=(dx, dy),
                        fontsize=7.5,
                        color="#14532d",
                        ha="right" if dx < 0 else "left",
                        va="bottom" if dy > 0 else "top",
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "facecolor": "white",
                            "edgecolor": "#8dd3a8",
                            "linewidth": 0.8,
                            "alpha": 0.94,
                        },
                    )

            baseline_gap = _safe_float(rows[0].get("suite_average_gap"))
            if baseline_gap is not None:
                ax.scatter(
                    [1],
                    [baseline_gap],
                    marker="*",
                    color="#f39c12",
                    edgecolors="#a04000",
                    linewidths=1.0,
                    s=170,
                    zorder=5,
                )

            if confirm_gap is not None:
                confirm_x = len(rows) + 0.9
                ax.axvline(len(rows) + 0.45, color="#d5d8dc", linewidth=1.0, linestyle=":")
                ax.scatter(
                    [confirm_x],
                    [confirm_gap],
                    marker="D",
                    color="#e67e22",
                    edgecolors="#a04000",
                    linewidths=1.0,
                    s=84,
                    zorder=5,
                )
                ax.annotate(
                    f"confirm {confirm_gap:.3f}",
                    (confirm_x, confirm_gap),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color="#7f3c08",
                )
                ax.set_xlim(0.4, len(rows) + 1.35)
                ax.set_xticks(list(range(1, len(rows) + 1)) + [confirm_x])
                ax.set_xticklabels(
                    [str(i) for i in range(1, len(rows) + 1)] + ["C"],
                    fontsize=8,
                )
            else:
                ax.set_xlim(0.4, len(rows) + 0.6)
                ax.set_xticks(list(range(1, len(rows) + 1)))
                ax.set_xticklabels([str(i) for i in range(1, len(rows) + 1)], fontsize=8)
        else:
            ax.set_xticks([])
            ax.text(0.5, 0.5, "No scout history", ha="center", va="center", fontsize=9, color="#7f8c8d", transform=ax.transAxes)

        stage_title = _stage_label(stage_suite).replace(" stage", "")
        ax.set_title(stage_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Scout attempt", fontsize=10)

    final_ax = axes[3]
    if final_gap is not None:
        final_ax.scatter(
            [1],
            [final_gap],
            marker="D",
            color="#e67e22",
            edgecolors="#a04000",
            linewidths=1.0,
            s=92,
            zorder=5,
        )
        final_ax.annotate(
            f"final {final_gap:.3f}",
            (1, final_gap),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#7f3c08",
        )
        final_ax.set_xticks([1])
        final_ax.set_xticklabels(["1tc.64"], fontsize=8)
        final_ax.set_xlim(0.5, 1.5)
    else:
        final_ax.set_xticks([])
        final_ax.text(0.5, 0.5, "No final record", ha="center", va="center", fontsize=9, color="#7f8c8d", transform=final_ax.transAxes)
    final_ax.set_title("64-node final", fontsize=12, fontweight="bold")
    final_ax.set_xlabel("Retained run", fontsize=10)

    axes[0].set_ylabel("Suite Average Gap", fontsize=11)
    axes[0].set_ylim(-0.03, 1.05)

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#f39c12",
               markeredgecolor="#a04000", markersize=11, label="Baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markeredgecolor="#1e3a2f", markersize=8, label="Kept"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c7ccd1",
               markeredgecolor="#c7ccd1", markersize=7, label="Discarded"),
        Line2D([0], [0], color="#27ae60", linewidth=2.2, label="Running best"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e67e22",
               markeredgecolor="#a04000", markersize=8, label="Confirm / final"),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Stage wise trajectories", fontsize=15, fontweight="bold", y=1.06)
    fig.text(
        0.5,
        0.01,
        "Each panel is stage-local. The 48-node confirm and 64-node final are retained single-instance large-instance runs.",
        ha="center",
        fontsize=9,
        color="#5d6d7e",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if output_path != KARPATHY_PROGRESS_PLOT_COPY:
        KARPATHY_PROGRESS_PLOT_COPY.write_bytes(output_path.read_bytes())
    return True


def _update_progress_plot(results_path: Path, output_path: Path) -> list[Path]:
    suite_rows = _load_suite_result_rows(results_path)
    if not suite_rows:
        return []

    generated: list[Path] = []
    beam_records = _load_jsonl_records(BEAM_HISTORY)
    promotion_records = _load_jsonl_records(PROMOTION_LOG)
    promotion_summaries = _latest_promotion_summaries(promotion_records)
    instance_records = _instance_records_with_overrides(_load_jsonl_records(INSTANCE_RESULTS))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for suite_name in _stage_suite_names():
        scout_path = PLOTS_DIR / f"scout_{suite_name}.png"
        if _plot_scout_trajectory(suite_rows, beam_records, suite_name, scout_path):
            generated.append(scout_path)

        summary_record = promotion_summaries.get(suite_name)
        if summary_record:
            promotion_path = PLOTS_DIR / f"promote_{suite_name}.png"
            if _plot_promotion_comparison(promotion_records, summary_record, suite_name, promotion_path):
                generated.append(promotion_path)

    if _plot_curriculum_overview(promotion_summaries, suite_rows, output_path):
        generated.append(output_path)
        if LEGACY_PROGRESS_PLOT.exists():
            generated.append(LEGACY_PROGRESS_PLOT)

    if _plot_progress2_karpathy(promotion_summaries, suite_rows, KARPATHY_PROGRESS_PLOT):
        generated.append(KARPATHY_PROGRESS_PLOT)
        if KARPATHY_PROGRESS_PLOT_COPY.exists():
            generated.append(KARPATHY_PROGRESS_PLOT_COPY)

    heatmap_path = PLOTS_DIR / "instance_heatmap.png"
    if _plot_instance_heatmap(promotion_summaries, suite_rows, instance_records, heatmap_path):
        generated.append(heatmap_path)

    resource_rows = _stage_resource_rows(promotion_summaries, suite_rows, instance_records)
    if resource_rows:
        _write_tsv(RESOURCE_TABLE, resource_rows)
    elif RESOURCE_TABLE.exists():
        RESOURCE_TABLE.unlink()

    scaling_rows = _family_scaling_rows(instance_records)
    if scaling_rows:
        _write_tsv(SCALING_TABLE, scaling_rows)
        if _plot_family_scaling(scaling_rows, SCALING_PLOT):
            generated.append(SCALING_PLOT)
    elif SCALING_TABLE.exists():
        SCALING_TABLE.unlink()

    pareto_rows = _kept_pareto_rows(suite_rows, instance_records)
    if pareto_rows:
        _write_tsv(PARETO_TABLE, pareto_rows)
        if _plot_pareto_frontier(suite_rows, instance_records, promotion_summaries, PARETO_PLOT):
            generated.append(PARETO_PLOT)
    elif PARETO_TABLE.exists():
        PARETO_TABLE.unlink()

    return generated


def _evaluate_plan(
    primary_suite: str,
    guardrails: list[tuple[str, str]],
    *,
    max_attempts: int,
    experiment_file: Path,
    baseline: bool,
    classical_baseline: str | None,
    seed_override: int | None,
    verbose: bool,
) -> tuple[dict, dict[str, dict]]:
    train_summary = evaluate_split_suite(
        suite_name=primary_suite,
        split="train",
        max_attempts=max_attempts,
        experiment_file=experiment_file,
        baseline=baseline,
        classical_baseline=classical_baseline,
        seed_override=seed_override,
        verbose=verbose,
    )
    split_summaries = {"train": train_summary}
    for split_name, suite_name in guardrails:
        split_summaries[split_name] = evaluate_split_suite(
            suite_name=suite_name,
            split=split_name,
            max_attempts=max_attempts,
            experiment_file=experiment_file,
            baseline=baseline,
            classical_baseline=classical_baseline,
            seed_override=seed_override,
            verbose=verbose,
        )
    return train_summary, split_summaries


def _aggregate_robustness_runs(
    suite_name: str,
    seed_runs: list[dict],
) -> tuple[list[dict], list[dict]]:
    by_problem: dict[tuple[str, int], list[dict]] = {}
    suite_gap_values: list[float] = []
    suite_wall_values: list[float] = []
    all_feasible_values: list[float] = []

    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        summary = seed_run["summary"]
        suite_gap = _safe_float(summary.get("suite_average_gap"))
        suite_wall = _safe_float(summary.get("total_wall_time"))
        if suite_gap is not None:
            suite_gap_values.append(suite_gap)
        if suite_wall is not None:
            suite_wall_values.append(suite_wall)
        all_feasible_values.append(1.0 if summary.get("all_feasible") else 0.0)

        for result in summary.get("results", []):
            problem = str(result.get("problem", ""))
            size = _safe_int(result.get("size")) or 0
            if not problem:
                continue
            by_problem.setdefault((problem, size), []).append(
                {
                    "seed": seed,
                    "gap": _safe_float(result.get("optimality_gap")),
                    "wall_time": _safe_float(result.get("wall_time")),
                    "raw_ar": _safe_float(result.get("raw_ar")),
                    "feasible": 1.0 if result.get("raw_feasible") else 0.0,
                    "family": str(result.get("winning_solver_family", "")),
                    "policy_summary": str(result.get("policy_summary", "")),
                }
            )

    instance_rows: list[dict] = []
    for (problem, size), runs in sorted(by_problem.items(), key=lambda item: _problem_sort_key(item[0][0], item[0][1])):
        gap_values = [run["gap"] for run in runs if run["gap"] is not None]
        wall_values = [run["wall_time"] for run in runs if run["wall_time"] is not None]
        ar_values = [run["raw_ar"] for run in runs if run["raw_ar"] is not None]
        feasible_values = [run["feasible"] for run in runs]
        instance_rows.append(
            {
                "suite": suite_name,
                "problem": problem,
                "size": size,
                "seed_count": len(runs),
                "seed_list": ",".join(str(run["seed"]) for run in runs),
                "mean_gap": _avg(gap_values),
                "std_gap": _std(gap_values),
                "mean_wall_time": _avg(wall_values),
                "std_wall_time": _std(wall_values),
                "mean_raw_ar": _avg(ar_values),
                "std_raw_ar": _std(ar_values),
                "feasible_rate": _avg(feasible_values),
                "dominant_solver_family": _dominant_string([run["family"] for run in runs], default=""),
                "dominant_policy_summary": _dominant_string([run["policy_summary"] for run in runs], default=""),
            }
        )

    suite_rows = [
        {
            "suite": suite_name,
            "seed_count": len(seed_runs),
            "seed_list": ",".join(str(int(seed_run["seed"])) for seed_run in seed_runs),
            "mean_suite_average_gap": _avg(suite_gap_values),
            "std_suite_average_gap": _std(suite_gap_values),
            "mean_total_wall_time": _avg(suite_wall_values),
            "std_total_wall_time": _std(suite_wall_values),
            "all_feasible_rate": _avg(all_feasible_values),
        }
    ]
    return instance_rows, suite_rows


def _append_robustness_record(record: dict) -> None:
    ROBUSTNESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ROBUSTNESS_LOG.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def evaluate_seed_robustness(
    suite_name: str,
    seed_values: list[int],
    max_attempts: int,
    experiment_file: Path,
    baseline: bool,
    classical_baseline: str | None,
    verbose: bool,
    no_artifacts: bool,
) -> dict:
    seed_runs: list[dict] = []
    for seed_value in seed_values:
        summary = evaluate_split_suite(
            suite_name=suite_name,
            split="test",
            max_attempts=max_attempts,
            experiment_file=experiment_file,
            baseline=baseline,
            classical_baseline=classical_baseline,
            seed_override=seed_value,
            verbose=verbose,
        )
        seed_runs.append({"seed": int(seed_value), "summary": summary})

    instance_rows, suite_rows = _aggregate_robustness_runs(suite_name, seed_runs)
    group_id = _next_eval_group_id(SUITE_HISTORY)
    policy_label = (
        _classical_policy_label(classical_baseline)
        if classical_baseline is not None
        else ("baseline" if baseline else "adaptive")
    )
    record = {
        "eval_group_id": group_id,
        "timestamp": datetime.now().isoformat(),
        "workflow": "robustness",
        "suite": suite_name,
        "policy_label": policy_label,
        "baseline": baseline,
        "classical_baseline": classical_baseline,
        "seed_values": [int(seed) for seed in seed_values],
        "experiment_file": str(experiment_file),
        "summary": suite_rows[0] if suite_rows else {},
        "per_seed_results": seed_runs,
        "per_instance_summary": instance_rows,
    }

    if not no_artifacts:
        _append_group_history(record)
        _append_robustness_record(record)
        _write_tsv(ROBUSTNESS_INSTANCE_TABLE, instance_rows)
        _write_tsv(ROBUSTNESS_SUITE_TABLE, suite_rows)

    if verbose and suite_rows:
        suite_summary = suite_rows[0]
        mean_gap = suite_summary.get("mean_suite_average_gap")
        std_gap = suite_summary.get("std_suite_average_gap")
        if mean_gap is not None and std_gap is not None:
            print(f"\n{'=' * 70}")
            print("  Robustness summary")
            print(f"{'=' * 70}")
            print(f"  suite:                       {suite_name}")
            print(f"  policy_label:                {policy_label}")
            print(f"  seeds:                       {suite_summary['seed_list']}")
            print(f"  suite_average_gap:           {float(mean_gap):.6f} ± {float(std_gap):.6f}")
            print(f"{'=' * 70}\n")

    return record


def evaluate_workflow(
    suite_name: str,
    workflow: str,
    prompt_variant: str,
    max_attempts: int,
    experiment_file: Path,
    baseline: bool,
    split: str | None = None,
    verbose: bool = True,
    no_plot: bool = False,
    no_dev: bool = False,
    no_artifacts: bool = False,
    classical_baseline: str | None = None,
    seed_override: int | None = None,
    seed_values: list[int] | None = None,
) -> dict:
    """Run a workflow, log split rows, and return the grouped summary."""

    if baseline and classical_baseline is not None:
        raise ValueError("--baseline and --classical-baseline are mutually exclusive.")
    if classical_baseline is not None and workflow not in {"split", "final", "robustness"}:
        raise ValueError(
            "--classical-baseline is only supported for split, final, and robustness workflows."
        )
    if workflow == "robustness":
        return evaluate_seed_robustness(
            suite_name=suite_name,
            seed_values=seed_values or list(DEFAULT_ROBUSTNESS_SEEDS),
            max_attempts=max_attempts,
            experiment_file=experiment_file,
            baseline=baseline,
            classical_baseline=classical_baseline,
            verbose=verbose,
            no_artifacts=no_artifacts,
        )

    if workflow == "candidate":
        incumbent = _latest_kept_candidate(suite_name, workflow="candidate")
        curriculum_plan = CURRICULUM_CANDIDATE_PLANS.get(suite_name)
        incumbent_train = _incumbent_split_gap(incumbent, "train")
        incumbent_dev = _incumbent_split_gap(incumbent, "dev")

        if curriculum_plan is not None:
            train_summary, split_summaries = _evaluate_plan(
                primary_suite=curriculum_plan["primary_suite"],
                guardrails=curriculum_plan["guardrails"],
                max_attempts=max_attempts,
                experiment_file=experiment_file,
                baseline=baseline,
                classical_baseline=classical_baseline,
                seed_override=seed_override,
                verbose=verbose,
            )
            dev_summary = None
            guardrail_ok = True

            for split_name, _ in curriculum_plan["guardrails"]:
                replay_summary = split_summaries[split_name]
                incumbent_replay_gap = _incumbent_split_gap(incumbent, split_name)
                guardrail_ok = guardrail_ok and passes_dev_guardrail(
                    replay_summary["suite_average_gap"],
                    incumbent_replay_gap,
                    tolerance=DEV_REGRESSION_TOLERANCE,
                )

            candidate_keep = True if incumbent is None else (
                is_strict_improvement(
                    train_summary["suite_average_gap"],
                    incumbent_train,
                )
                and guardrail_ok
            )
            decision = "keep" if candidate_keep else "discard"
        else:
            train_summary = evaluate_split_suite(
                suite_name=suite_name,
                split="train",
                max_attempts=max_attempts,
                experiment_file=experiment_file,
                baseline=baseline,
                classical_baseline=classical_baseline,
                seed_override=seed_override,
                verbose=verbose,
            )

            if no_dev:
                # Single-instance mode: no dev guardrail, just check train improvement.
                dev_summary = None
                if incumbent is None:
                    candidate_keep = True
                else:
                    candidate_keep = (
                        train_summary["suite_average_gap"] < incumbent_train
                        if incumbent_train is not None
                        else True
                    )
            else:
                dev_summary = evaluate_split_suite(
                    suite_name=suite_name,
                    split="dev",
                    max_attempts=max_attempts,
                    experiment_file=experiment_file,
                    baseline=baseline,
                    classical_baseline=classical_baseline,
                    seed_override=seed_override,
                    verbose=verbose,
                )
                candidate_keep = True if incumbent is None else accept_candidate(
                    train_summary["suite_average_gap"],
                    dev_summary["suite_average_gap"],
                    incumbent_train,
                    incumbent_dev,
                )

            decision = "keep" if candidate_keep else "discard"
            split_summaries = {"train": train_summary}
            if dev_summary is not None:
                split_summaries["dev"] = dev_summary
    elif workflow == "scout":
        plan = CURRICULUM_SCOUT_PLANS.get(suite_name)
        if plan is None:
            raise ValueError(
                f"Unknown scout plan for suite: {suite_name}. "
                f"Choose from: {sorted(CURRICULUM_SCOUT_PLANS)}"
            )
        incumbent = _latest_kept_candidate(suite_name, workflow="scout")
        incumbent_train = _incumbent_split_gap(incumbent, "train")
        incumbent_dev = None
        train_summary, split_summaries = _evaluate_plan(
            primary_suite=plan["primary_suite"],
            guardrails=plan["guardrails"],
            max_attempts=max_attempts,
            experiment_file=experiment_file,
            baseline=baseline,
            classical_baseline=classical_baseline,
            seed_override=seed_override,
            verbose=verbose,
        )
        guardrail_ok = True
        for split_name, _ in plan["guardrails"]:
            incumbent_replay_gap = _incumbent_split_gap(incumbent, split_name)
            guardrail_ok = guardrail_ok and passes_dev_guardrail(
                split_summaries[split_name]["suite_average_gap"],
                incumbent_replay_gap,
                tolerance=DEV_REGRESSION_TOLERANCE,
            )
        candidate_keep = True if incumbent is None else (
            is_strict_improvement(
                train_summary["suite_average_gap"],
                incumbent_train,
            )
            and guardrail_ok
        )
        dev_summary = None
        decision = "keep" if candidate_keep else "discard"
    elif workflow == "confirm":
        plan = CURRICULUM_CONFIRM_PLANS.get(suite_name)
        if plan is None:
            raise ValueError(
                f"Unknown confirm plan for suite: {suite_name}. "
                f"Choose from: {sorted(CURRICULUM_CONFIRM_PLANS)}"
            )
        train_summary, split_summaries = _evaluate_plan(
            primary_suite=plan["primary_suite"],
            guardrails=plan["guardrails"],
            max_attempts=max_attempts,
            experiment_file=experiment_file,
            baseline=baseline,
            classical_baseline=classical_baseline,
            seed_override=seed_override,
            verbose=verbose,
        )
        incumbent_train = None
        incumbent_dev = None
        dev_summary = None
        decision = "confirm"
    elif workflow == "final":
        split_summaries = {
            "test": evaluate_split_suite(
                suite_name=suite_name,
                split="test",
                max_attempts=max_attempts,
                experiment_file=experiment_file,
                baseline=baseline,
                classical_baseline=classical_baseline,
                seed_override=seed_override,
                verbose=verbose,
            )
        }
        incumbent_train = None
        incumbent_dev = None
        decision = "final"
    elif workflow == "split":
        if split is None:
            raise ValueError("--split is required when --workflow split is selected.")
        split_summaries = {
            split: evaluate_split_suite(
                suite_name=suite_name,
                split=split,
                max_attempts=max_attempts,
                experiment_file=experiment_file,
                baseline=baseline,
                classical_baseline=classical_baseline,
                seed_override=seed_override,
                verbose=verbose,
            )
        }
        incumbent_train = None
        incumbent_dev = None
        decision = f"split:{split}"
    else:
        raise ValueError(f"Unknown workflow: {workflow}")

    group_id = _next_eval_group_id(SUITE_HISTORY)
    if classical_baseline is not None:
        policy_label = _classical_policy_label(classical_baseline)
    else:
        policy_label = "baseline" if baseline else "adaptive"
    if not no_artifacts:
        _append_suite_rows(
            group_id=group_id,
            workflow=workflow,
            prompt_variant=prompt_variant,
            policy_label=policy_label,
            decision=decision,
            split_summaries=split_summaries,
        )

    record = {
        "eval_group_id": group_id,
        "timestamp": datetime.now().isoformat(),
        "workflow": workflow,
        "suite": suite_name,
        "prompt_variant": prompt_variant,
        "policy_label": policy_label,
        "experiment_file": str(experiment_file),
        "baseline": baseline,
        "classical_baseline": classical_baseline,
        "seed_override": int(seed_override) if seed_override is not None else None,
        "decision": decision,
        "dev_regression_tolerance": DEV_REGRESSION_TOLERANCE,
        "primary_metric": PRIMARY_METRIC,
        "is_primary_track": workflow in {"candidate", "scout"},
        "incumbent_train_suite_average_gap": incumbent_train,
        "incumbent_dev_suite_average_gap": incumbent_dev,
        "split_summaries": {
            split_name: {
                key: value
                for key, value in summary.items()
                if key != "results"
            }
            for split_name, summary in split_summaries.items()
        },
        "per_split_results": {
            split_name: summary["results"] for split_name, summary in split_summaries.items()
        },
    }
    if workflow in {"candidate", "scout", "confirm"}:
        record["train_suite_average_gap"] = train_summary["suite_average_gap"]
        record["dev_suite_average_gap"] = (
            dev_summary["suite_average_gap"] if dev_summary is not None else None
        )
        record["candidate_accept"] = int(decision == "keep") if workflow != "confirm" else None
        for split_name, summary in split_summaries.items():
            record[f"{split_name}_suite_average_gap"] = summary["suite_average_gap"]
    elif workflow == "final":
        record["test_suite_average_gap"] = split_summaries["test"]["suite_average_gap"]
    elif split is not None:
        record[f"{split}_suite_average_gap"] = split_summaries[split]["suite_average_gap"]

    if not no_artifacts:
        _append_group_history(record)

    if not no_artifacts and not no_plot:
        try:
            generated_plots = _update_progress_plot(SUITE_RESULTS, SUITE_PROGRESS_PLOT)
            if verbose:
                if generated_plots:
                    print(f"  plots_dir: {PLOTS_DIR}")
                    print(f"  overview_plot: {SUITE_PROGRESS_PLOT}")
                    print(f"  generated_plot_count: {len(generated_plots)}")
                else:
                    print("  plots_dir: n/a (insufficient data yet)")
        except Exception as exc:
            if verbose:
                print(f"  plots_dir: unavailable ({exc})")

    if verbose:
        print(f"\n{'=' * 70}")
        print("  Workflow summary")
        print(f"{'=' * 70}")
        print(f"  workflow:                    {workflow}")
        print(f"  policy_label:                {policy_label}")
        if workflow in {"candidate", "scout", "confirm"}:
            print(f"  train_suite_average_gap:     {train_summary['suite_average_gap']:.6f}")
            if dev_summary is not None:
                print(f"  dev_suite_average_gap:       {dev_summary['suite_average_gap']:.6f}")
            elif no_dev and suite_name not in CURRICULUM_CANDIDATE_PLANS:
                print("  dev_suite_average_gap:       n/a (--no-dev)")
            for split_name, summary in split_summaries.items():
                if split_name == "train" or split_name == "dev":
                    continue
                print(f"  {split_name}_suite_average_gap: {summary['suite_average_gap']:.6f}")
            if workflow in {"candidate", "scout"}:
                incumbent = _latest_kept_candidate(suite_name, workflow=workflow)
                print(
                    "  incumbent_train_suite_average_gap: "
                    f"{incumbent_train:.6f}" if incumbent_train is not None else "  incumbent_train_suite_average_gap: n/a"
                )
                if dev_summary is not None or (not no_dev and suite_name not in CURRICULUM_CANDIDATE_PLANS):
                    print(
                        "  incumbent_dev_suite_average_gap:   "
                        f"{incumbent_dev:.6f}" if incumbent_dev is not None else "  incumbent_dev_suite_average_gap:   n/a"
                    )
                for split_name in split_summaries:
                    if split_name == "train" or split_name == "dev":
                        continue
                    incumbent_gap = _incumbent_split_gap(incumbent, split_name)
                    print(
                        f"  incumbent_{split_name}_suite_average_gap: "
                        f"{incumbent_gap:.6f}" if incumbent_gap is not None else f"  incumbent_{split_name}_suite_average_gap: n/a"
                    )
                print(f"  candidate_decision:          {decision}")
                print(f"  candidate_accept:            {int(decision == 'keep')}")
            else:
                print("  candidate_decision:          confirm")
        elif workflow == "final":
            print(f"  test_suite_average_gap:      {split_summaries['test']['suite_average_gap']:.6f}")
        elif split is not None:
            print(f"  {split}_suite_average_gap:      {split_summaries[split]['suite_average_gap']:.6f}")
        print(f"  eval_group_id:               {group_id}")
        print(f"{'=' * 70}\n")

    return record


def _parse_seed_list(seed_list_raw: str | None) -> list[int]:
    if seed_list_raw is None or not str(seed_list_raw).strip():
        return list(DEFAULT_ROBUSTNESS_SEEDS)
    values: list[int] = []
    for token in str(seed_list_raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or list(DEFAULT_ROBUSTNESS_SEEDS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate experiment.py policy across fixed suite workflows",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="quick",
        choices=sorted(SUITES),
        help="Problem suite scale to evaluate",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="candidate",
        choices=("candidate", "scout", "confirm", "final", "split", "robustness"),
        help="candidate/full confirm, scout=cheap proxy, confirm=full report, final=held-out test, split=one explicit split, robustness=multi-seed held-out validation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=("train", "dev", "test"),
        help="Explicit split to evaluate when --workflow split is used",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max attempts per instance",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="full",
        help="Label for which prompt variant produced this policy",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the immutable conservative VQE baseline through experiment.py",
    )
    parser.add_argument(
        "--classical-baseline",
        type=str,
        default=None,
        choices=("greedy_min_degree", "random_feasible"),
        help="Evaluate a classical MIS baseline instead of experiment.py",
    )
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help="Override the solver/backend seed for a direct evaluation run",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help="Comma-separated seeds for the robustness workflow (default: 17,23,29,31,37)",
    )
    parser.add_argument(
        "--experiment-file",
        type=Path,
        default=DEFAULT_EXPERIMENT_FILE,
        help="Policy file to evaluate (default: experiment.py)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not regenerate the suite-level progress plot",
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="Skip dev evaluation in candidate workflow (single-instance mode)",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not write suite results, instance results, history, or plots",
    )
    args = parser.parse_args()

    evaluate_workflow(
        suite_name=args.suite,
        workflow=args.workflow,
        prompt_variant=args.prompt_variant,
        max_attempts=args.max_attempts,
        experiment_file=args.experiment_file,
        baseline=args.baseline,
        split=args.split,
        no_plot=args.no_plot,
        no_dev=args.no_dev,
        no_artifacts=args.no_artifacts,
        classical_baseline=args.classical_baseline,
        seed_override=args.seed_override,
        seed_values=_parse_seed_list(args.seed_list) if args.workflow == "robustness" else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
