"""
Single-constraint 0-1 Knapsack Problem generator.

Knapsack:
    maximize   sum_j  v_j * x_j
    subject to sum_j  w_j * x_j <= C
    x_j in {0, 1}

QUBO form (via penalty method):
    minimize  -sum_j v_j x_j + penalty * (sum_j w_j x_j + sum_k s_k 2^k - C)^2

This creates feasibility pressure on the quantum solver: raw solutions
may violate the capacity constraint, requiring the solver to learn to
produce feasible outputs directly.

Classical reference values are computed exactly with dynamic programming, so
reported optimality gaps are measured against the true knapsack optimum rather
than a heuristic lower bound.
"""

from __future__ import annotations

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from .base import ProblemInstance, ProblemGenerator


class KnapsackGenerator(ProblemGenerator):
    """Generate single-constraint 0-1 knapsack instances."""

    def __init__(self, tightness: float = 0.6):
        """
        Args:
            tightness: capacity = tightness * sum(weights).
                       Lower = tighter constraint = harder feasibility.
        """
        self.tightness = tightness

    def generate(self, size: int, seed: int) -> ProblemInstance:
        """
        Generate a knapsack instance.

        Args:
            size: number of items (n)
            seed: random seed
        """
        rng = np.random.RandomState(seed)
        n = size

        values = rng.randint(10, 50, size=n).astype(float)
        weights = rng.randint(1, 10, size=n).astype(float)
        capacity = float(int(self.tightness * np.sum(weights)))

        # Build QP
        qp = QuadraticProgram("knapsack")
        for j in range(n):
            qp.binary_var(f"x{j}")
        qp.maximize(linear={f"x{j}": float(values[j]) for j in range(n)})
        qp.linear_constraint(
            linear={f"x{j}": float(weights[j]) for j in range(n)},
            sense="<=",
            rhs=capacity,
            name="capacity",
        )

        # Convert to QUBO (adds slack variables and penalties)
        converter = QuadraticProgramToQubo(penalty=10.0)
        qubo = converter.convert(qp)

        # Exact solve for all generated knapsack instances via pseudopolynomial DP.
        opt_val, opt_x = self._exact_solve(values, weights, capacity, n)

        return ProblemInstance(
            name=f"knapsack_{n}_s{seed}",
            problem_type="knapsack",
            num_variables=qubo.get_num_vars(),
            qubo=qubo,
            optimal_value=opt_val,
            optimal_solution=opt_x,
            metadata={
                "num_items": n,
                "values": values,
                "weights": weights,
                "capacity": capacity,
                "tightness": self.tightness,
                "seed": seed,
                "optimal_reference": "exact_dynamic_programming",
                "original_qp": qp,
                "converter": converter,
            },
        )

    def _exact_solve(
        self, values, weights, capacity, n
    ) -> tuple[float, np.ndarray]:
        """Exact 0-1 knapsack solve via dynamic programming with reconstruction."""
        int_weights = [int(round(weight)) for weight in weights]
        int_capacity = int(round(capacity))

        dp = np.zeros((n + 1, int_capacity + 1), dtype=float)
        take = np.zeros((n, int_capacity + 1), dtype=bool)

        for item_idx in range(n):
            weight = int_weights[item_idx]
            value = float(values[item_idx])
            for remaining in range(int_capacity + 1):
                best_without = dp[item_idx, remaining]
                best_with = float("-inf")
                if weight <= remaining:
                    best_with = dp[item_idx, remaining - weight] + value
                if best_with > best_without + 1e-12:
                    dp[item_idx + 1, remaining] = best_with
                    take[item_idx, remaining] = True
                else:
                    dp[item_idx + 1, remaining] = best_without

        best_x = np.zeros(n, dtype=float)
        remaining = int_capacity
        for item_idx in range(n - 1, -1, -1):
            if take[item_idx, remaining]:
                best_x[item_idx] = 1.0
                remaining -= int_weights[item_idx]

        return float(dp[n, int_capacity]), best_x
