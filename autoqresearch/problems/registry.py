"""
Problem registry with train/dev/test splits.

Fixed seeds ensure reproducibility. The agent searches on TRAIN instances,
validates on DEV, and final reported numbers use held-out TEST instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .base import ProblemInstance
from .maxcut import MaxCutGenerator
from .mis import MISGenerator, load_mis_from_dimacs
from .mdkp import MDKPGenerator
from .knapsack import KnapsackGenerator
from .cvrp import CVRPGenerator, CVRP_INSTANCE_DIR, load_cvrp_from_file


@dataclass
class BenchmarkSplit:
    """A train/dev/test split for one problem class + size."""
    train: list[ProblemInstance]
    dev: list[ProblemInstance]
    test: list[ProblemInstance]


# ── Fixed seed ranges for reproducible splits ──────────────────────
#    Train: seeds 0-4     (agent experiments against these)
#    Dev:   seeds 100-104  (agent validates; can peek)
#    Test:  seeds 200-204  (held-out; reported in paper)

GENERATORS = {
    "maxcut": MaxCutGenerator(degree=3, weighted=False),
    "mis": MISGenerator(edge_probability=0.3),
    "mdkp": MDKPGenerator(num_constraints=5, tightness=0.5),
    "knapsack": KnapsackGenerator(tightness=0.6),
    "cvrp": CVRPGenerator(capacity_method="hard_slack", seed_method="depot_farthest"),
}

# Problem sizes for the core paper
CORE_SIZES = {
    "maxcut": [8, 10, 12, 14, 16],
    "mis": [8, 10, 12, 14],
    "mdkp": [10, 15],   # items; QUBO vars will be larger due to slacks
    "knapsack": [5, 8, 12],  # items; QUBO vars larger due to slacks
    "cvrp": [8, 10, 12],  # customers; GAP/TSP QUBOs are staged
}

# Smaller sizes for quick agent iteration
QUICK_SIZES = {
    "maxcut": [8, 10],
    "mis": [8, 10],
    "mdkp": [10],
    "knapsack": [5, 8],
    "cvrp": [8],
}


def generate_split(
    problem_type: str,
    size: int,
    instances_per_split: int = 5,
) -> BenchmarkSplit:
    """Generate train/dev/test split for a single problem type + size."""
    gen = GENERATORS[problem_type]
    return BenchmarkSplit(
        train=[gen.generate(size, seed=k) for k in range(instances_per_split)],
        dev=[gen.generate(size, seed=100 + k) for k in range(instances_per_split)],
        test=[gen.generate(size, seed=200 + k) for k in range(instances_per_split)],
    )


def generate_all_splits(
    suite: Literal["core", "quick"] = "quick",
    instances_per_split: int = 5,
) -> dict[str, dict[int, BenchmarkSplit]]:
    """
    Generate all benchmark splits.

    Returns:
        {problem_type: {size: BenchmarkSplit}}
    """
    sizes = CORE_SIZES if suite == "core" else QUICK_SIZES
    splits = {}
    for ptype, size_list in sizes.items():
        splits[ptype] = {}
        for size in size_list:
            splits[ptype][size] = generate_split(ptype, size, instances_per_split)
    return splits


# ── DIMACS file-based MIS instance directory ────────────────────────
MIS_INSTANCE_DIR = Path(__file__).resolve().parent.parent.parent / "individual" / "mis"

# Cache to avoid re-reading and re-solving the same file.
_MIS_FILE_CACHE: dict[str, ProblemInstance] = {}
_CVRP_FILE_CACHE: dict[str, ProblemInstance] = {}


def get_mis_file_instance(filename: str, penalty: float | None = None) -> ProblemInstance:
    """Load an MIS instance from individual/mis/<filename>.

    ``filename`` is the stem without directory, e.g. ``"1tc.32"``.
    The actual file is ``individual/mis/1tc.32.txt``.

    ``penalty`` is passed to ``QuadraticProgramToQubo``. If ``None``
    (default), the converter computes an appropriate penalty
    automatically from the problem structure.
    """
    cache_key = f"{filename}_p{penalty}"
    if cache_key in _MIS_FILE_CACHE:
        return _MIS_FILE_CACHE[cache_key]

    fpath = MIS_INSTANCE_DIR / f"{filename}.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"MIS instance file not found: {fpath}")

    instance = load_mis_from_dimacs(fpath, penalty=penalty)
    _MIS_FILE_CACHE[cache_key] = instance
    return instance


def get_cvrp_file_instance(
    filename: str,
    *,
    capacity_method: str = "hard_slack",
    seed_method: str = "depot_farthest",
    gap_penalty: float | None = None,
    taylor_alpha: float = 10.0,
    tilted_kappa: float = 5.0,
    tilted_s_frac: float = 0.10,
    tilted_s_min: float = 1.0,
) -> ProblemInstance:
    """Load a CVRP instance from ``individual/cvrp/<filename>.vrp``."""
    stem = filename[:-4] if filename.endswith(".vrp") else filename
    cache_key = (
        f"{stem}_m{capacity_method}_s{seed_method}_p{gap_penalty}_"
        f"ta{taylor_alpha}_tk{tilted_kappa}_tsf{tilted_s_frac}_tsm{tilted_s_min}"
    )
    if cache_key in _CVRP_FILE_CACHE:
        return _CVRP_FILE_CACHE[cache_key]

    fpath = CVRP_INSTANCE_DIR / f"{stem}.vrp"
    if not fpath.exists():
        raise FileNotFoundError(f"CVRP instance file not found: {fpath}")

    instance = load_cvrp_from_file(
        fpath,
        capacity_method=capacity_method,
        seed_method=seed_method,
        gap_penalty=gap_penalty,
        taylor_alpha=taylor_alpha,
        tilted_kappa=tilted_kappa,
        tilted_s_frac=tilted_s_frac,
        tilted_s_min=tilted_s_min,
    )
    _CVRP_FILE_CACHE[cache_key] = instance
    return instance


def get_cvrp_instance(
    size: int,
    seed: int = 0,
    *,
    capacity_method: str = "hard_slack",
    seed_method: str = "depot_farthest",
    gap_penalty: float | None = None,
    taylor_alpha: float = 10.0,
    tilted_kappa: float = 5.0,
    tilted_s_frac: float = 0.10,
    tilted_s_min: float = 1.0,
) -> ProblemInstance:
    """Generate a synthetic CVRP instance with selectable construction knobs."""
    from .cvrp import (
        CVRP_INSTANCE_DIR,
        default_vehicle_count_for_size,
        generate_synthetic_cvrp_instance,
        make_cvrp_problem,
        read_cvrp_instance,
    )

    vehicles = default_vehicle_count_for_size(int(size))
    instance_file = CVRP_INSTANCE_DIR / f"Synth-n{int(size) + 1}-k{vehicles}-s{int(seed)}.vrp"
    if instance_file.exists():
        instance = read_cvrp_instance(instance_file)
    else:
        instance = generate_synthetic_cvrp_instance(int(size), int(seed), num_vehicles=int(vehicles))
    return make_cvrp_problem(
        instance,
        num_vehicles=int(vehicles),
        seed_method=seed_method,
        seed_random_state=int(seed) + 7,
        capacity_method=capacity_method,
        gap_penalty=gap_penalty,
        taylor_alpha=taylor_alpha,
        tilted_kappa=tilted_kappa,
        tilted_s_frac=tilted_s_frac,
        tilted_s_min=tilted_s_min,
    )


def get_single_instance(
    problem_type: str,
    size: int,
    seed: int = 0,
) -> ProblemInstance:
    """Get a single problem instance (convenience function).

    For file-based MIS instances, use ``problem_type="mis_file"``
    and encode the filename in ``size`` (passed as 0) and ``seed`` (unused).
    Instead, use :func:`get_mis_file_instance` directly.
    """
    return GENERATORS[problem_type].generate(size, seed)
