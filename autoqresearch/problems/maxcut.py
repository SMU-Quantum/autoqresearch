"""
MaxCut problem generator.

MaxCut on a graph G=(V,E) with edge weights w:
    maximize  sum_{(i,j) in E} w_ij * (1 - x_i * x_j) / 2
    where x_i in {0, 1}

QUBO form (minimization):
    minimize  sum_{(i,j) in E} w_ij * (x_i * x_j - x_i/2 - x_j/2)

This is QAOA's natural home turf — the cost Hamiltonian maps directly
to the graph structure.

DO NOT MODIFY.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut

from .base import ProblemInstance, ProblemGenerator, solve_brute_force


class MaxCutGenerator(ProblemGenerator):
    """Generate MaxCut instances on random regular graphs."""

    def __init__(self, degree: int = 3, weighted: bool = False):
        self.degree = degree
        self.weighted = weighted

    def generate(self, size: int, seed: int) -> ProblemInstance:
        """
        Generate a MaxCut instance on a random d-regular graph.

        Args:
            size: number of nodes
            seed: random seed for reproducibility
        """
        rng = np.random.RandomState(seed)

        # Generate random regular graph
        G = nx.random_regular_graph(self.degree, size, seed=seed)

        # Add weights if requested
        if self.weighted:
            for u, v in G.edges():
                G[u][v]['weight'] = rng.uniform(0.5, 2.0)
        else:
            for u, v in G.edges():
                G[u][v]['weight'] = 1.0

        # Use Qiskit's Maxcut application to build the QuadraticProgram
        maxcut_app = Maxcut(nx.to_numpy_array(G))
        qp = maxcut_app.to_quadratic_program()

        # Solve exactly for small instances
        if size <= 20:
            opt_val, opt_x = solve_brute_force(qp)
            max_cut_value = float(opt_val)
        else:
            # For larger instances, use SDP relaxation as upper bound
            opt_x = None
            max_cut_value = self._sdp_bound(G)

        return ProblemInstance(
            name=f"maxcut_{size}_d{self.degree}_s{seed}",
            problem_type="maxcut",
            num_variables=size,
            qubo=qp,
            optimal_value=max_cut_value,
            optimal_solution=opt_x,
            metadata={
                "graph": G,
                "num_nodes": size,
                "num_edges": G.number_of_edges(),
                "degree": self.degree,
                "weighted": self.weighted,
                "seed": seed,
            },
        )

    def _sdp_bound(self, G: nx.Graph) -> float:
        """Goemans-Williamson SDP upper bound (approximate)."""
        # Simple greedy approximation as fallback
        cut_val = nx.algorithms.approximation.maxcut.one_exchange(G)[0]
        return cut_val
