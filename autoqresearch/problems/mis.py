"""
Maximum Independent Set (MIS) problem generator.

MIS on graph G=(V,E):
    maximize  sum_i x_i
    subject to  x_i + x_j <= 1  for all (i,j) in E
    x_i in {0, 1}

The constrained formulation is converted to QUBO via
``QuadraticProgramToQubo`` which automatically determines an
appropriate penalty weight for the edge constraints.

This is a constrained graph problem — tests penalty-weight tuning
and mixer selection (XY-mixer preserves feasibility).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import networkx as nx

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo

from .base import ProblemInstance, ProblemGenerator, solve_brute_force


# ── DIMACS file-based MIS loader ─────────────────────────────────────


def _read_dimacs_graph(
    file_path: str | Path,
) -> tuple[int, list[tuple[int, int]], dict[str, str]]:
    """Read a DIMACS edge-list file and return ``(num_nodes, edges, comments)``.

    Format:
        p edge <num_nodes> <num_edges>
        e <u> <v>           (1-based indices)
    """
    edges = []
    num_nodes = 0
    comments: dict[str, str] = {}
    with open(file_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "c" and len(parts) >= 3:
                comments[parts[1]] = " ".join(parts[2:])
            elif parts[0] == "p":
                num_nodes = int(parts[2])
            elif parts[0] == "e":
                u = int(parts[1]) - 1  # convert to 0-based
                v = int(parts[2]) - 1
                edges.append((u, v))
    return num_nodes, edges, comments


def _solve_mis_classically(G: nx.Graph) -> int:
    """Solve MIS exactly using docplex/CPLEX if available, else brute-force/approximation."""
    n = G.number_of_nodes()

    # Try brute-force for small instances
    if n <= 20:
        best = 0
        for k in range(2 ** n):
            x = [(k >> i) & 1 for i in range(n)]
            feasible = all(x[u] == 0 or x[v] == 0 for u, v in G.edges())
            if feasible:
                best = max(best, sum(x))
        return best

    # Try docplex/CPLEX for exact solution
    try:
        from docplex.mp.model import Model

        model = Model(name="MIS")
        xvars = model.binary_var_list(n, name="x")
        model.maximize(model.sum(xvars[i] for i in range(n)))
        for u, v in G.edges():
            model.add_constraint(xvars[u] + xvars[v] <= 1)
        solution = model.solve()
        if solution:
            return int(solution.objective_value)
    except ImportError:
        pass

    # Fallback: networkx greedy approximation (lower bound)
    return len(nx.algorithms.approximation.maximum_independent_set(G))


def _build_mis_constrained_qp(
    num_nodes: int, edges: list[tuple[int, int]]
) -> QuadraticProgram:
    """Build a proper constrained QuadraticProgram for MIS.

    maximize  sum_i x_i
    subject to  x_i + x_j <= 1  for every edge (i,j)
    x_i in {0, 1}

    This matches the formulation in the MIS VQE notebook.
    """
    model = Model(name="Maximum Independent Set")
    x = model.binary_var_list(num_nodes, name="x")
    model.maximize(model.sum(x[i] for i in range(num_nodes)))
    for u, v in edges:
        model.add_constraint(x[u] + x[v] <= 1, f"edge_{u}_{v}")
    return from_docplex_mp(model)


def load_mis_from_dimacs(
    file_path: str | Path,
    penalty: float | None = None,
    instance_name: str | None = None,
) -> ProblemInstance:
    """Load an MIS instance from a DIMACS edge-list file.

    Parameters
    ----------
    file_path : path to .txt file in DIMACS edge format
    penalty : penalty weight passed to ``QuadraticProgramToQubo``.
              If ``None`` (default), the converter computes an
              appropriate penalty automatically.
    instance_name : optional human-readable name (defaults to stem of filename)

    Returns
    -------
    ProblemInstance with the QUBO, graph, and optimal MIS size.
    """
    file_path = Path(file_path)
    num_nodes, edges, dimacs_comments = _read_dimacs_graph(file_path)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    name = instance_name or f"mis_file_{file_path.stem}"

    # Build the constrained QP (maximize sum x_i, s.t. x_i+x_j<=1)
    original_qp = _build_mis_constrained_qp(num_nodes, edges)

    # Convert to QUBO with proper penalty
    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(original_qp)
    actual_penalty = converter.penalty

    planted_mis_size = dimacs_comments.get("planted_mis_size")
    if planted_mis_size is not None:
        optimal_mis_size = int(planted_mis_size)
    else:
        optimal_mis_size = _solve_mis_classically(G)

    return ProblemInstance(
        name=name,
        problem_type="mis",
        num_variables=num_nodes,
        qubo=qubo,
        optimal_value=optimal_mis_size,
        optimal_solution=None,
        metadata={
            "graph": G,
            "num_nodes": num_nodes,
            "num_edges": G.number_of_edges(),
            "penalty": actual_penalty,
            "source_file": str(file_path),
            "dimacs_comments": dimacs_comments,
            "num_items": num_nodes,
            "original_qp": original_qp,
            "converter": converter,
        },
    )


class MISGenerator(ProblemGenerator):
    """Generate MIS instances on random Erdos-Renyi graphs."""

    def __init__(self, edge_probability: float = 0.3, penalty: float | None = None):
        self.edge_probability = edge_probability
        self.penalty = penalty  # None → auto-penalty from QuadraticProgramToQubo

    def generate(self, size: int, seed: int) -> ProblemInstance:
        G = nx.erdos_renyi_graph(size, self.edge_probability, seed=seed)

        # Build constrained QP and convert to QUBO with proper penalty
        edges = list(G.edges())
        original_qp = _build_mis_constrained_qp(size, edges)
        converter = QuadraticProgramToQubo(penalty=self.penalty)
        qubo = converter.convert(original_qp)

        # Solve exactly for small instances
        if size <= 20:
            mis_size = self._exact_mis_size(G)
            opt_x = None
        else:
            opt_x = None
            mis_size = _solve_mis_classically(G)

        return ProblemInstance(
            name=f"mis_{size}_p{self.edge_probability}_s{seed}",
            problem_type="mis",
            num_variables=size,
            qubo=qubo,
            optimal_value=mis_size,
            optimal_solution=opt_x,
            metadata={
                "graph": G,
                "num_nodes": size,
                "num_edges": G.number_of_edges(),
                "edge_probability": self.edge_probability,
                "penalty": converter.penalty,
                "seed": seed,
                "original_qp": original_qp,
                "converter": converter,
            },
        )

    def _exact_mis_size(self, G: nx.Graph) -> int:
        """Brute-force exact MIS (small instances only)."""
        n = G.number_of_nodes()
        best = 0
        for k in range(2 ** n):
            x = [(k >> i) & 1 for i in range(n)]
            feasible = True
            for u, v in G.edges():
                if x[u] == 1 and x[v] == 1:
                    feasible = False
                    break
            if feasible:
                best = max(best, sum(x))
        return best
