"""Base solver interfaces and shared result helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit

from ..problems.base import ProblemInstance


TWO_QUBIT_EQUIVALENTS = {"cx", "cz", "ecr", "cnot"}


@dataclass
class SolverResult:
    """Output of a single solver run."""

    best_bitstring: np.ndarray
    best_objective: float
    is_feasible: bool

    counts: dict[str, int]
    num_shots: int

    circuit_depth: int = 0
    cnot_count: int = 0
    two_qubit_gate_count: int = 0
    total_gate_count: int = 0
    gate_counts: dict[str, int] = field(default_factory=dict)
    num_qubits: int = 0
    num_parameters: int = 0

    optimizer_iterations: int = 0
    final_cost: float = 0.0
    wall_time_seconds: float = 0.0

    parameter_values: Optional[np.ndarray] = None
    convergence_history: list[float] = field(default_factory=list)
    solver_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def cvar_value(self) -> float:
        """Compute CVaR at alpha=0.25 from the observed objective values."""

        return compute_cvar_from_counts(
            self.counts,
            self.metadata.get("objective_lookup", {}),
            0.25,
            maximize=self.metadata.get("objective_sense") == "MAXIMIZE",
        )


class BaseSolver(ABC):
    """Abstract base class for quantum optimization solvers."""

    name: str = "base"

    @abstractmethod
    def solve(
        self,
        problem: ProblemInstance,
        policy: dict,
        backend,
        shots: int = 1000,
    ) -> SolverResult:
        """Run the solver on a problem instance."""


def extract_best_solution(
    counts: dict[str, int],
    problem: ProblemInstance,
    prefer_feasible: bool = True,
) -> tuple[np.ndarray, float, bool]:
    """Extract the best solution from measurement counts."""

    maximize = problem.qubo.objective.sense.name == "MAXIMIZE"
    best_val = float("-inf") if maximize else float("inf")
    best_x = None
    best_feasible = False

    for bitstring_str in counts:
        x = np.array([int(bit) for bit in bitstring_str[::-1]], dtype=float)

        if len(x) < problem.num_variables:
            x = np.pad(x, (0, problem.num_variables - len(x)))
        elif len(x) > problem.num_variables:
            x = x[: problem.num_variables]

        val = problem.objective_value(x)
        feasible = problem.is_feasible(x)
        is_better = val > best_val if maximize else val < best_val

        if prefer_feasible:
            if feasible and not best_feasible:
                best_val = val
                best_x = x
                best_feasible = True
            elif feasible == best_feasible and is_better:
                best_val = val
                best_x = x
                best_feasible = feasible
        elif is_better:
            best_val = val
            best_x = x
            best_feasible = feasible

    if best_x is None:
        best_x = np.zeros(problem.num_variables)
        best_val = problem.objective_value(best_x)
        best_feasible = problem.is_feasible(best_x)

    return best_x, float(best_val), bool(best_feasible)


def compute_cvar_from_counts(
    counts: dict[str, int],
    objective_lookup: dict[str, float],
    alpha: float = 0.25,
    maximize: bool = False,
) -> float:
    """Compute CVaR_alpha from sampled objective values."""

    if not counts:
        return 0.0

    values_counts = []
    total = 0
    for bitstring, count in counts.items():
        if bitstring not in objective_lookup:
            continue
        values_counts.append((objective_lookup[bitstring], count))
        total += count

    if not values_counts or total == 0:
        return 0.0

    values_counts.sort(key=lambda pair: pair[0], reverse=bool(maximize))
    cutoff = max(1, int(np.ceil(alpha * total)))
    taken = 0
    cvar_sum = 0.0
    for value, count in values_counts:
        take = min(count, cutoff - taken)
        cvar_sum += value * take
        taken += take
        if taken >= cutoff:
            break

    return cvar_sum / max(taken, 1)


def count_gates(circuit: QuantumCircuit) -> tuple[int, int]:
    """Backward-compatible helper returning depth and CX-like gates."""

    resources = summarize_circuit_resources(circuit)
    return resources["depth"], resources["cnot_count"]


def summarize_circuit_resources(circuit: QuantumCircuit) -> dict[str, int | dict[str, int]]:
    """Summarize depth and gate counts for reporting."""

    gate_counts = {
        str(name): int(count)
        for name, count in circuit.count_ops().items()
        if name not in {"measure", "barrier"}
    }
    cnot_count = sum(gate_counts.get(name, 0) for name in TWO_QUBIT_EQUIVALENTS)
    two_qubit_gate_count = sum(
        1
        for instruction in circuit.data
        if instruction.operation.name not in {"measure", "barrier"}
        and instruction.operation.num_qubits == 2
    )
    total_gate_count = sum(gate_counts.values())
    depth = circuit.depth(
        filter_function=lambda instruction: instruction.operation.name not in {"measure", "barrier"}
    )

    return {
        "depth": int(depth or 0),
        "cnot_count": int(cnot_count),
        "two_qubit_gate_count": int(two_qubit_gate_count),
        "total_gate_count": int(total_gate_count),
        "gate_counts": gate_counts,
    }
