"""
Base classes for combinatorial optimization problem instances.

A ProblemInstance bundles everything needed to evaluate a quantum solver:
the QUBO matrix, the Ising Hamiltonian, the known-optimal (or best-known)
solution value, and metadata for reporting.

DO NOT MODIFY — this is part of the fixed evaluator infrastructure.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


@dataclass(frozen=True)
class ProblemInstance:
    """Immutable description of one combinatorial optimization instance."""

    name: str                           # e.g. "maxcut_12_seed42"
    problem_type: str                   # "maxcut", "mis", "mdkp"
    num_variables: int                  # binary variables in QUBO
    qubo: QuadraticProgram              # Qiskit QuadraticProgram (QUBO form)
    optimal_value: float                # best-known objective (from classical solver)
    optimal_solution: Optional[np.ndarray] = None  # best-known bitstring
    metadata: dict = field(default_factory=dict)    # graph, constraints, etc.

    @property
    def num_qubits_direct(self) -> int:
        """Qubits needed for a 1-to-1 encoding (no compression)."""
        return self.num_variables

    def objective_value(self, bitstring: np.ndarray) -> float:
        """Evaluate the QUBO objective for a given bitstring."""
        # Use QuadraticProgram's built-in evaluation
        try:
            result = self.qubo.objective.evaluate(bitstring)
            return result
        except Exception:
            # Fallback: manual QUBO evaluation
            Q = self.qubo_matrix
            x = np.asarray(bitstring, dtype=float)
            return float(x @ Q @ x)

    @property
    def qubo_matrix(self) -> np.ndarray:
        """Extract the QUBO matrix Q such that f(x) = x^T Q x + const."""
        n = self.num_variables
        Q = np.zeros((n, n))
        obj = self.qubo.objective
        # Linear terms on diagonal
        for idx, coeff in obj.linear.to_dict().items():
            Q[idx, idx] += coeff
        # Quadratic terms
        for (i, j), coeff in obj.quadratic.to_dict().items():
            if i == j:
                Q[i, i] += coeff
            else:
                Q[i, j] += coeff / 2
                Q[j, i] += coeff / 2
        return Q

    def is_feasible(self, bitstring: np.ndarray) -> bool:
        """Check if a solution satisfies all constraints of the original problem."""
        return self.qubo.is_feasible(bitstring)


class ProblemGenerator(ABC):
    """Abstract base class for problem instance generators."""

    @abstractmethod
    def generate(self, size: int, seed: int) -> ProblemInstance:
        """Generate a single problem instance of the given size."""
        ...

    def generate_suite(
        self,
        sizes: list[int],
        seeds_per_size: int = 5,
        base_seed: int = 0,
    ) -> list[ProblemInstance]:
        """Generate a suite of instances across sizes and seeds."""
        instances = []
        for size in sizes:
            for k in range(seeds_per_size):
                seed = base_seed + size * 1000 + k
                instances.append(self.generate(size, seed))
        return instances


def solve_brute_force(qp: QuadraticProgram) -> tuple[float, np.ndarray]:
    """Brute-force solve a quadratic program by enumerating all bitstrings."""

    n = qp.get_num_vars()
    if n > 20:
        raise ValueError(f"Brute force not practical for n={n} > 20")

    maximize = qp.objective.sense.name == "MAXIMIZE"
    best_val = float("-inf") if maximize else float("inf")
    best_x = None
    obj = qp.objective

    for k in range(2 ** n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=float)
        val = obj.evaluate(x)
        if (maximize and val > best_val) or ((not maximize) and val < best_val):
            best_val = val
            best_x = x.copy()

    return best_val, best_x
