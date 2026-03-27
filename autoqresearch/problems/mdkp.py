"""
Multi-Dimensional Knapsack Problem (MDKP) generator.

MDKP:
    maximize   sum_j  v_j * x_j
    subject to sum_j  w_ij * x_j <= c_i   for all i in [m] constraints
    x_j in {0, 1}

QUBO form (via penalty method):
    minimize  -sum_j v_j x_j
              + sum_i penalty_i * (sum_j w_ij x_j + sum_k s_ik 2^k - c_i)^2

This is a heavily constrained COP with inequality structure.
Tests QRAO/PCE compression and Lagrangian methods.

DO NOT MODIFY.
"""

from __future__ import annotations

import numpy as np

from qiskit_optimization import QuadraticProgram

from .base import ProblemInstance, ProblemGenerator, solve_brute_force


class MDKPGenerator(ProblemGenerator):
    """Generate MDKP instances."""

    def __init__(self, num_constraints: int = 5, tightness: float = 0.5):
        """
        Args:
            num_constraints: number of knapsack constraints (m)
            tightness: capacity = tightness * sum(weights_per_constraint)
        """
        self.num_constraints = num_constraints
        self.tightness = tightness

    def generate(self, size: int, seed: int) -> ProblemInstance:
        """
        Generate an MDKP instance.

        Args:
            size: number of items (n)
            seed: random seed
        """
        rng = np.random.RandomState(seed)
        m = self.num_constraints
        n = size

        # Generate random instance
        values = rng.randint(1, 50, size=n).astype(float)
        weights = rng.randint(1, 30, size=(m, n)).astype(float)
        capacities = np.floor(
            self.tightness * weights.sum(axis=1)
        ).astype(float)

        # Build ILP as QuadraticProgram
        qp = QuadraticProgram("mdkp")
        for j in range(n):
            qp.binary_var(f"x{j}")

        # Objective: maximize sum(v_j * x_j) -> minimize -sum(v_j * x_j)
        qp.minimize(linear={f"x{j}": -values[j] for j in range(n)})

        # Constraints: sum(w_ij * x_j) <= c_i
        for i_c in range(m):
            linear_constraint = {f"x{j}": float(weights[i_c, j]) for j in range(n)}
            qp.linear_constraint(
                linear=linear_constraint,
                sense="<=",
                rhs=float(capacities[i_c]),
                name=f"capacity_{i_c}",
            )

        # Convert to QUBO (adds slack variables and penalties)
        from qiskit_optimization.converters import QuadraticProgramToQubo
        converter = QuadraticProgramToQubo(penalty=10.0)
        qubo = converter.convert(qp)

        # Solve exactly for small instances
        if n <= 15:
            opt_val = self._exact_solve(values, weights, capacities, n, m)
        else:
            opt_val = self._greedy_solve(values, weights, capacities, n, m)

        return ProblemInstance(
            name=f"mdkp_{m}x{n}_s{seed}",
            problem_type="mdkp",
            num_variables=qubo.get_num_vars(),
            qubo=qubo,
            optimal_value=opt_val,
            optimal_solution=None,
            metadata={
                "num_items": n,
                "num_constraints": m,
                "values": values,
                "weights": weights,
                "capacities": capacities,
                "tightness": self.tightness,
                "seed": seed,
                "original_qp": qp,
                "converter": converter,
            },
        )

    def _exact_solve(self, values, weights, capacities, n, m) -> float:
        """Brute-force exact solve for small instances."""
        best = 0.0
        for k in range(2 ** n):
            x = np.array([(k >> j) & 1 for j in range(n)], dtype=float)
            # Check feasibility
            feasible = True
            for i_c in range(m):
                if np.dot(weights[i_c], x) > capacities[i_c]:
                    feasible = False
                    break
            if feasible:
                val = np.dot(values, x)
                best = max(best, val)
        return best

    def _greedy_solve(self, values, weights, capacities, n, m) -> float:
        """Greedy heuristic for larger instances (lower bound)."""
        # Value-to-weight ratio heuristic
        avg_weight = weights.mean(axis=0)
        avg_weight[avg_weight == 0] = 1e-10
        ratios = values / avg_weight
        order = np.argsort(-ratios)

        x = np.zeros(n)
        remaining = capacities.copy()
        for j in order:
            if np.all(weights[:, j] <= remaining):
                x[j] = 1
                remaining -= weights[:, j]
        return float(np.dot(values, x))
