"""LEGACY / NOT USED FOR KNAPSACK POLICY OBJECTIVE: multi-objective evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..problems.base import ProblemInstance
from ..solvers.base import SolverResult


@dataclass
class EvaluationResult:
    """Complete evaluation of a solver run."""

    approximation_ratio: float
    is_feasible: bool
    feasibility_rate: float

    circuit_depth: int
    cnot_count: int
    two_qubit_gate_count: int
    total_gate_count: int
    gate_counts: dict[str, int] = field(default_factory=dict)
    num_qubits: int = 0

    composite_score: float = 0.0
    wall_time_seconds: float = 0.0
    solver_name: str = ""
    problem_name: str = ""

    def to_tsv_row(self) -> str:
        """Format as a TSV row for results.tsv."""

        return "\t".join(
            [
                self.problem_name,
                self.solver_name,
                f"{self.composite_score:.6f}",
                f"{self.approximation_ratio:.4f}",
                str(int(self.is_feasible)),
                f"{self.feasibility_rate:.4f}",
                str(self.circuit_depth),
                str(self.two_qubit_gate_count),
                str(self.total_gate_count),
                str(self.num_qubits),
                f"{self.wall_time_seconds:.1f}",
            ]
        )

    @staticmethod
    def tsv_header() -> str:
        return "\t".join(
            [
                "problem",
                "solver",
                "composite_score",
                "approx_ratio",
                "feasible",
                "feasibility_rate",
                "depth",
                "two_qubit_gates",
                "total_gates",
                "qubits",
                "wall_time_s",
            ]
        )


class Evaluator:
    """Evaluator that penalizes two-qubit gate count and circuit depth."""

    def __init__(
        self,
        w1: float = 1.0,
        w2: float = 0.05,
        w3: float = 0.05,
        max_depth: int = 500,
        max_two_qubit_gates: int = 200,
    ):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.max_depth = max_depth
        self.max_two_qubit_gates = max_two_qubit_gates

    def evaluate(
        self,
        solver_result: SolverResult,
        problem: ProblemInstance,
    ) -> EvaluationResult:
        """Evaluate a solver result against a problem instance."""

        if problem.optimal_value == 0:
            approx_ratio = 1.0 if solver_result.best_objective == 0 else 0.0
        elif problem.problem_type == "maxcut":
            found_value = self._decode_objective(solver_result, problem)
            approx_ratio = min(1.0, max(0.0, found_value / problem.optimal_value))
        else:
            approx_ratio = min(
                1.0,
                max(0.0, problem.optimal_value / (abs(solver_result.best_objective) + 1e-10)),
            )

        is_feasible = bool(solver_result.is_feasible)
        feasibility_rate = self._compute_feasibility_rate(solver_result, problem)
        two_qubit_gate_count = int(
            solver_result.two_qubit_gate_count or solver_result.cnot_count
        )
        depth = int(solver_result.circuit_depth)

        composite_score = (
            self.w1 * approx_ratio * (1.0 if is_feasible else 0.0)
            - self.w2 * min(1.0, depth / self.max_depth)
            - self.w3 * min(1.0, two_qubit_gate_count / self.max_two_qubit_gates)
        )

        return EvaluationResult(
            approximation_ratio=approx_ratio,
            is_feasible=is_feasible,
            feasibility_rate=feasibility_rate,
            circuit_depth=depth,
            cnot_count=int(solver_result.cnot_count),
            two_qubit_gate_count=two_qubit_gate_count,
            total_gate_count=int(solver_result.total_gate_count),
            gate_counts=dict(solver_result.gate_counts),
            num_qubits=int(solver_result.num_qubits),
            composite_score=float(composite_score),
            wall_time_seconds=float(solver_result.wall_time_seconds),
            solver_name=solver_result.solver_name,
            problem_name=problem.name,
        )

    def _decode_objective(self, result: SolverResult, problem: ProblemInstance) -> float:
        """Decode the objective value back to the original problem's scale."""

        x = result.best_bitstring
        graph = problem.metadata.get("graph")
        if graph is None:
            return 0.0

        return float(
            sum(
                graph[u][v].get("weight", 1.0)
                for u, v in graph.edges()
                if int(x[u]) != int(x[v])
            )
        )

    def _compute_feasibility_rate(self, result: SolverResult, problem: ProblemInstance) -> float:
        """Compute fraction of measured samples that are feasible."""

        if not result.counts:
            return 0.0

        total = 0
        feasible = 0
        for bitstring, count in result.counts.items():
            x = np.array([int(bit) for bit in bitstring[::-1]], dtype=float)
            if len(x) < problem.num_variables:
                x = np.pad(x, (0, problem.num_variables - len(x)))
            elif len(x) > problem.num_variables:
                x = x[: problem.num_variables]
            total += count
            if problem.is_feasible(x):
                feasible += count

        return feasible / max(total, 1)
