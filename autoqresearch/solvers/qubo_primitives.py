"""QUBO-based solver primitives for constrained optimization problems.

Provides QAOA, VQE, and PCE solvers that operate on the Ising Hamiltonian
derived from a QUBO formulation (knapsack, MIS, etc.). Tracks convergence
history for normalized convergence statistics (improvement, stagnation,
final_cost).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import (
    QAOAAnsatz,
    PauliEvolutionGate,
    efficient_su2,
    pauli_two_design,
    real_amplitudes,
)
from qiskit.primitives import BackendSamplerV2
from qiskit.quantum_info import SparsePauliOp

from ..backends.factory import BackendBundle
from ..problems.base import ProblemInstance
from .base import SolverResult, summarize_circuit_resources


# ─── Feasibility & objective helpers ─────────────────────────────


def check_qubo_feasibility(qubo_x: np.ndarray, problem: ProblemInstance) -> bool:
    """Check if a QUBO-space solution satisfies problem constraints."""
    converter = problem.metadata.get("converter")
    original_qp = problem.metadata.get("original_qp")
    if converter is not None and original_qp is not None:
        original_x = converter.interpret(qubo_x)
        return original_qp.is_feasible(original_x)
    return problem.is_feasible(qubo_x)


def qubo_objective_value(qubo_x: np.ndarray, problem: ProblemInstance) -> float:
    """Compute objective value (original space) from a QUBO solution."""
    converter = problem.metadata.get("converter")
    original_qp = problem.metadata.get("original_qp")
    if converter is not None and original_qp is not None:
        original_x = converter.interpret(qubo_x)
        return float(original_qp.objective.evaluate(original_x))
    return 0.0


# ─── MIS-specific feasibility & objective helpers ────────────────


def check_mis_feasibility(x: np.ndarray, problem: ProblemInstance) -> bool:
    """Check if a bitstring is a feasible independent set.

    A solution is feasible iff:
    1. At least one node is selected (empty set is trivially "feasible"
       but meaningless — we reject it).
    2. No two adjacent nodes are both selected.
    """
    graph = problem.metadata.get("graph")
    if graph is None:
        return False
    n_vars = problem.num_variables
    # Reject the empty set — selecting 0 nodes is not a meaningful solution
    if sum(int(x[i]) for i in range(min(len(x), n_vars))) == 0:
        return False
    for u, v in graph.edges():
        if u < n_vars and v < n_vars:
            if int(x[u]) == 1 and int(x[v]) == 1:
                return False
    return True


def mis_objective_value(x: np.ndarray, problem: ProblemInstance) -> float:
    """Compute MIS size (number of selected nodes) for a feasible solution.

    Returns the count of selected nodes if feasible, else 0.
    """
    if check_mis_feasibility(x, problem):
        n_vars = problem.num_variables
        return float(sum(int(x[i]) for i in range(min(len(x), n_vars))))
    return 0.0


def _bitstring_to_array(bitstring_str: str, n_vars: int) -> np.ndarray:
    """Convert a Qiskit measurement bitstring to a node-indexed array.

    Qiskit returns MSB-first strings; reversing gives x[0]=qubit 0=node 0.
    """
    x = np.array([int(bit) for bit in bitstring_str[::-1]], dtype=float)
    if len(x) < n_vars:
        x = np.pad(x, (0, n_vars - len(x)))
    elif len(x) > n_vars:
        x = x[:n_vars]
    return x


def extract_mis_solution(
    counts: dict[str, int], problem: ProblemInstance
) -> tuple[np.ndarray, float, bool]:
    """Extract the MIS solution from measurement counts.

    Uses the **most-probable bitstring** — the single bitstring with
    the highest measurement count.  This is the standard approach
    (matching the reference notebook): the result reflects what the
    quantum circuit actually concentrates probability on.

    No cherry-picking or fallback to secondary bitstrings.

    **Concentration guard**: if the top bitstring's count is not
    statistically significant above the expected maximum of a uniform
    distribution, the circuit is producing noise.  In that case we
    return the all-zeros bitstring (MIS=0, infeasible) so the gap
    honestly reports 1.0.

    The expected maximum count of ``n`` uniform draws into ``k`` bins is
    approximately ``n/k + sqrt(2 * (n/k) * ln(k))``.  We require the
    top count to exceed **twice** this expected maximum to trust it.
    """
    n_vars = problem.num_variables

    if not counts:
        return np.zeros(n_vars), 0.0, False

    total_shots = sum(counts.values())
    n_unique = len(counts)

    # Pick the single most-probable bitstring
    top_bs = max(counts, key=counts.get)
    top_count = counts[top_bs]

    # Concentration guard: compare against expected max of uniform multinomial.
    if n_unique > 1:
        import math
        mu = total_shots / n_unique  # expected count per bin
        # Expected max order statistic ≈ mu + sqrt(2 * mu * ln(k))
        expected_max = mu + math.sqrt(2.0 * mu * math.log(n_unique))
        threshold = 2.0 * expected_max  # 2× the expected max for safety
        if top_count <= threshold:
            # Circuit output is indistinguishable from uniform noise.
            return np.zeros(n_vars), 0.0, False

    x = _bitstring_to_array(top_bs, n_vars)
    feasible = check_mis_feasibility(x, problem)
    selected = int(sum(x[:n_vars]))

    if feasible:
        return x, float(selected), True
    return x, float(selected), False


def compute_mis_feasibility_rate(counts: dict[str, int], problem: ProblemInstance) -> float:
    """Fraction of measured samples that are feasible independent sets."""
    total = 0
    feasible = 0
    n_vars = problem.num_variables
    for bitstring, count in counts.items():
        x = _bitstring_to_array(bitstring, n_vars)
        total += count
        if check_mis_feasibility(x, problem):
            feasible += count
    return feasible / max(total, 1)


def compute_mis_best_feasible_ar(counts: dict[str, int], problem: ProblemInstance) -> float:
    """Approximation ratio of the most-probable bitstring (if feasible).

    Matches the most-probable extraction logic — only checks the single
    top bitstring by count.
    """
    if not counts:
        return 0.0
    n_vars = problem.num_variables
    top_bs = max(counts, key=counts.get)
    x = _bitstring_to_array(top_bs, n_vars)
    if check_mis_feasibility(x, problem):
        mis_size = float(sum(int(x[i]) for i in range(n_vars)))
        ar = mis_size / problem.optimal_value if problem.optimal_value > 0 else 0.0
        return min(1.0, ar)
    return 0.0


# ─── Generic dispatchers (route by problem type) ────────────────


def check_feasibility(x: np.ndarray, problem: ProblemInstance) -> bool:
    """Check feasibility, dispatching by problem type."""
    if problem.problem_type == "mis":
        return check_mis_feasibility(x, problem)
    return check_qubo_feasibility(x, problem)


def objective_value(x: np.ndarray, problem: ProblemInstance) -> float:
    """Compute objective value, dispatching by problem type."""
    if problem.problem_type == "mis":
        return mis_objective_value(x, problem)
    return qubo_objective_value(x, problem)


def extract_solution(
    counts: dict[str, int], problem: ProblemInstance
) -> tuple[np.ndarray, float, bool]:
    """Extract the best solution from measurement counts, dispatching by problem type."""
    if problem.problem_type == "mis":
        return extract_mis_solution(counts, problem)
    return extract_qubo_solution(counts, problem)


def compute_feasibility_rate(counts: dict[str, int], problem: ProblemInstance) -> float:
    """Fraction of measured samples satisfying constraints (dispatches by problem type)."""
    if problem.problem_type == "mis":
        return compute_mis_feasibility_rate(counts, problem)
    # Knapsack path
    total = 0
    feasible = 0
    n_qubo = problem.num_variables
    for bitstring, count in counts.items():
        x = np.array([int(bit) for bit in bitstring[::-1]], dtype=float)
        if len(x) < n_qubo:
            x = np.pad(x, (0, n_qubo - len(x)))
        elif len(x) > n_qubo:
            x = x[:n_qubo]
        total += count
        if check_qubo_feasibility(x, problem):
            feasible += count
    return feasible / max(total, 1)


def compute_best_feasible_ar(counts: dict[str, int], problem: ProblemInstance) -> float:
    """Best approximation ratio among feasible samples (dispatches by problem type)."""
    if problem.problem_type == "mis":
        return compute_mis_best_feasible_ar(counts, problem)
    # Knapsack path
    n_qubo = problem.num_variables
    best_ar = 0.0
    for bitstring in counts:
        x = np.array([int(bit) for bit in bitstring[::-1]], dtype=float)
        if len(x) < n_qubo:
            x = np.pad(x, (0, n_qubo - len(x)))
        elif len(x) > n_qubo:
            x = x[:n_qubo]
        if check_qubo_feasibility(x, problem):
            obj = qubo_objective_value(x, problem)
            ar = obj / problem.optimal_value if problem.optimal_value > 0 else 0.0
            best_ar = max(best_ar, min(1.0, ar))
    return best_ar


def extract_qubo_solution(
    counts: dict[str, int], problem: ProblemInstance
) -> tuple[np.ndarray, float, bool]:
    """Extract the best QUBO solution from measurement counts.

    Prefers feasible solutions. Among feasible, picks highest objective.
    If no feasible solution exists, picks the highest-objective infeasible one.
    """
    n_qubo = problem.num_variables
    best_feasible_val = float("-inf")
    best_feasible_x = None
    best_any_val = float("-inf")
    best_any_x = None

    for bitstring_str in counts:
        x = np.array([int(bit) for bit in bitstring_str[::-1]], dtype=float)
        if len(x) < n_qubo:
            x = np.pad(x, (0, n_qubo - len(x)))
        elif len(x) > n_qubo:
            x = x[:n_qubo]
        obj_val = qubo_objective_value(x, problem)
        feasible = check_qubo_feasibility(x, problem)

        if feasible and obj_val > best_feasible_val:
            best_feasible_val = obj_val
            best_feasible_x = x.copy()
        if obj_val > best_any_val:
            best_any_val = obj_val
            best_any_x = x.copy()

    if best_feasible_x is not None:
        return best_feasible_x, best_feasible_val, True
    if best_any_x is not None:
        return best_any_x, best_any_val, False
    return np.zeros(n_qubo), 0.0, False


def fixed_repair(qubo_x: np.ndarray, problem: ProblemInstance) -> tuple[np.ndarray, bool]:
    """Greedy repair: drop items by worst value/weight ratio until feasible.

    Returns:
        (repaired_original_x, repair_changed)
    """
    converter = problem.metadata.get("converter")
    if converter is not None:
        original_x = np.array(converter.interpret(qubo_x), dtype=float)
    else:
        original_x = qubo_x.copy()

    values = problem.metadata["values"]
    weights = problem.metadata["weights"]
    capacity = problem.metadata["capacity"]
    n_items = problem.metadata["num_items"]

    x = original_x[:n_items].copy()

    if np.dot(weights, x) <= capacity:
        return x, False

    selected = np.where(x > 0.5)[0]
    if len(selected) == 0:
        return x, False

    vw_ratios = values[selected] / (weights[selected] + 1e-10)
    drop_order = selected[np.argsort(vw_ratios)]

    for idx in drop_order:
        x[idx] = 0.0
        if np.dot(weights, x) <= capacity:
            break

    return x, True


def _qubo_pce_graph(problem: ProblemInstance) -> nx.Graph:
    """Build the weighted MaxCut graph used by the notebook PCE reduction."""

    linear = np.asarray(problem.qubo.objective.linear.to_array(), dtype=float)
    quadratic = np.asarray(problem.qubo.objective.quadratic.to_array(), dtype=float)
    quad = np.asarray(quadratic, dtype=float).copy()
    np.fill_diagonal(quad, np.diag(quad) + linear)

    num_variables = quad.shape[0]
    weights = np.zeros((num_variables + 1, num_variables + 1), dtype=float)

    row_sums = np.sum(quad, axis=1)
    col_sums = np.sum(quad, axis=0)
    for index in range(num_variables):
        weight = float(row_sums[index] + col_sums[index])
        weights[0, index + 1] = weight
        weights[index + 1, 0] = weight

    for left in range(num_variables):
        for right in range(left + 1, num_variables):
            weight = float(quad[left, right] + quad[right, left])
            weights[left + 1, right + 1] = weight
            weights[right + 1, left + 1] = weight

    graph = nx.Graph()
    graph.add_nodes_from(range(num_variables + 1))
    for left in range(num_variables + 1):
        for right in range(left + 1, num_variables + 1):
            weight = float(weights[left, right])
            if abs(weight) > 1e-12:
                graph.add_edge(left, right, weight=weight)
    return graph


def _qubo_pce_cut_value(graph: nx.Graph, bits: np.ndarray) -> float:
    """Evaluate the weighted cut induced by a partition bitstring."""

    total = 0.0
    for left, right, data in graph.edges(data=True):
        if int(bits[left]) != int(bits[right]):
            total += float(data.get("weight", 1.0))
    return total


def _qubo_pce_local_search(graph: nx.Graph, bits: np.ndarray) -> tuple[np.ndarray, float]:
    """Single-node flip local search mirroring the notebook workflow."""

    best_bits = np.asarray(bits, dtype=float).copy()
    best_cut = _qubo_pce_cut_value(graph, best_bits)

    improved = True
    while improved:
        improved = False
        for node in range(len(best_bits)):
            candidate = best_bits.copy()
            candidate[node] = 1.0 - candidate[node]
            candidate_cut = _qubo_pce_cut_value(graph, candidate)
            if candidate_cut > best_cut + 1e-12:
                best_bits = candidate
                best_cut = candidate_cut
                improved = True
    return best_bits, float(best_cut)


def _pce_regularization_scale_from_graph(graph: nx.Graph) -> float:
    """Match the regularization scale used by the MaxCut PCE solver."""

    total_weight = sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True))
    mst = nx.minimum_spanning_tree(graph, weight="weight")
    mst_weight = sum(float(data.get("weight", 1.0)) for _, _, data in mst.edges(data=True))
    return total_weight / 2.0 + mst_weight / 4.0


# ─── Circuit construction ────────────────────────────────────────


def build_qaoa_ansatz(qubitOp: SparsePauliOp, reps: int) -> QuantumCircuit:
    """Build a QAOA circuit for a general Ising Hamiltonian."""
    num_qubits = qubitOp.num_qubits

    gamma = ParameterVector("γ", reps)
    beta = ParameterVector("β", reps)

    mixer_paulis = []
    for i in range(num_qubits):
        label = ["I"] * num_qubits
        label[i] = "X"
        mixer_paulis.append("".join(label))
    mixer_op = SparsePauliOp(mixer_paulis)

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for p in range(reps):
        cost_gate = PauliEvolutionGate(qubitOp, time=gamma[p])
        qc.append(cost_gate, range(num_qubits))
        mixer_gate = PauliEvolutionGate(mixer_op, time=beta[p])
        qc.append(mixer_gate, range(num_qubits))

    return qc


def build_qubo_warmstart_qaoa_ansatz(
    problem: ProblemInstance,
    qubitOp: SparsePauliOp,
    policy: dict,
    rng: np.random.Generator,
) -> tuple[QuantumCircuit, np.ndarray]:
    """Build the warm-start QAOA circuit and the induced classical warm bits."""

    from .maxcut_primitives import solve_relaxed_problem

    num_qubits = qubitOp.num_qubits
    reps = int(policy.get("reps", 2))
    epsilon = float(policy.get("ws_epsilon", 0.25))
    source = str(policy.get("ws_source", "relaxation")).lower()

    relaxed = None
    if source in {"lp", "relaxation", "relaxed", "slsqp"}:
        relaxed = solve_relaxed_problem(problem)
    elif source != "random":
        raise ValueError(
            f"Unsupported warm-start source for knapsack: {source}. "
            "Use 'relaxation' or 'random'."
        )

    if relaxed is None:
        warm_bits = rng.integers(0, 2, size=num_qubits).astype(float)
        c_stars = epsilon + (1.0 - 2.0 * epsilon) * warm_bits
    else:
        relaxed = np.asarray(relaxed, dtype=float)
        if len(relaxed) < num_qubits:
            relaxed = np.pad(relaxed, (0, num_qubits - len(relaxed)))
        elif len(relaxed) > num_qubits:
            relaxed = relaxed[:num_qubits]
        warm_bits = (relaxed >= 0.5).astype(float)
        c_stars = np.clip(relaxed, epsilon, 1.0 - epsilon)

    thetas = 2.0 * np.arcsin(np.sqrt(c_stars))
    init_state = QuantumCircuit(num_qubits)
    mixer_operator = QuantumCircuit(num_qubits)
    beta = Parameter("beta")
    for index, theta in enumerate(thetas):
        init_state.ry(theta, index)
        mixer_operator.ry(-theta, index)
        mixer_operator.rz(-2.0 * beta, index)
        mixer_operator.ry(theta, index)

    return (
        QAOAAnsatz(
            cost_operator=qubitOp,
            reps=reps,
            initial_state=init_state,
            mixer_operator=mixer_operator,
        ),
        warm_bits,
    )


def build_variational_ansatz(
    num_qubits: int,
    ansatz_type: str,
    reps: int,
    entanglement: str = "linear",
    custom_ansatz_fn=None,
) -> QuantumCircuit:
    """Build one of the supported variational ansatz families.

    Supported ansatz types:
        - ``efficient_su2``: Ry-Rz layers with entanglement
        - ``real_amplitudes``: Ry layers with entanglement
        - ``pauli_two_design``: Random Pauli rotation layers
        - ``brickwork``: Alternating brick-layer entanglement
        - ``custom``: Use ``custom_ansatz_fn(num_qubits, reps, entanglement)``
          to build an arbitrary QuantumCircuit. The agent can define this
          function in ``experiment.py`` and pass it through the policy dict.
    """

    lowered = str(ansatz_type).lower()
    if lowered == "efficient_su2":
        return efficient_su2(num_qubits, reps=reps, entanglement=entanglement)
    if lowered == "real_amplitudes":
        return real_amplitudes(num_qubits, reps=reps, entanglement=entanglement)
    if lowered in {"pauli_two_design", "paulitwodesign"}:
        return pauli_two_design(num_qubits=num_qubits, reps=reps)
    if lowered == "brickwork":
        from .maxcut_primitives import build_brickwork_ansatz

        return build_brickwork_ansatz(reps, num_qubits)
    if lowered == "custom":
        if custom_ansatz_fn is None:
            raise ValueError(
                "ansatz_type='custom' requires a custom_ansatz_fn(num_qubits, reps, entanglement) "
                "callable in the policy dict."
            )
        circuit = custom_ansatz_fn(num_qubits, reps, entanglement)
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"custom_ansatz_fn must return a QuantumCircuit, got {type(circuit)}"
            )
        # ── Validate custom ansatz structure ──────────────────────────
        # A variational ansatz must:
        #   1. Have trainable parameters (Qiskit Parameter objects)
        #   2. Use only allowed gates: parameterized rotations (rx, ry, rz),
        #      Hadamard (h), and entangling gates (cx/cnot, cz).
        #      Fixed Pauli gates (x, y, z) are NOT allowed — they just
        #      prepare classical states and bypass the variational search.
        ALLOWED_GATES = {"rx", "ry", "rz", "h", "cx", "cnot", "cz", "barrier", "measure"}
        bad_gates = []
        for instruction in circuit.data:
            gate_name = instruction.operation.name.lower()
            if gate_name not in ALLOWED_GATES:
                bad_gates.append(gate_name)
        if bad_gates:
            unique_bad = sorted(set(bad_gates))
            raise ValueError(
                f"custom_ansatz_fn uses disallowed gate(s): {unique_bad}. "
                f"A variational ansatz may only use: {sorted(ALLOWED_GATES - {'barrier', 'measure'})}. "
                f"Use Rx/Ry/Rz with Parameter objects for rotations (not fixed X/Y/Z gates), "
                f"H for superposition, and CX/CZ for entanglement."
            )
        if circuit.num_parameters == 0:
            raise ValueError(
                "custom_ansatz_fn returned a circuit with 0 trainable parameters. "
                "A variational ansatz must contain parameterized rotation gates "
                "(Rx, Ry, Rz with Parameter objects) so the classical "
                "optimizer has something to optimize."
            )
        return circuit
    raise ValueError(f"Unsupported ansatz type: {ansatz_type}")


def build_vqe_ansatz(num_qubits: int, policy: dict) -> QuantumCircuit:
    """Build a VQE ansatz for a general problem.

    When ``policy["ansatz_type"]`` is ``"custom"``, the policy must also
    contain a ``custom_ansatz_fn`` key whose value is a callable
    ``(num_qubits, reps, entanglement) -> QuantumCircuit``.
    """
    ansatz_type = policy.get("ansatz_type", "efficient_su2")
    vqe_reps = int(policy.get("vqe_reps", 2))
    entanglement = policy.get("entanglement", "linear")
    custom_fn = policy.get("custom_ansatz_fn", None)

    return build_variational_ansatz(
        num_qubits, ansatz_type, vqe_reps, entanglement,
        custom_ansatz_fn=custom_fn,
    )


# ─── Cost function ───────────────────────────────────────────────


class EstimatorCostFunction:
    """Estimator-based cost function for VQE (matching reference notebook).

    Uses ``BackendEstimatorV2`` to compute ⟨ψ|H|ψ⟩ — the standard VQE
    approach.  Tracks convergence history.
    """

    def __init__(
        self,
        qubitOp: SparsePauliOp,
        offset: float,
        estimator,
        ansatz: QuantumCircuit,
    ):
        self.qubitOp = qubitOp
        self.offset = offset
        self.estimator = estimator
        self.ansatz = ansatz
        self.history: list[float] = []
        self.iterations: int = 0

    def __call__(self, params: np.ndarray) -> float:
        pub = (self.ansatz, [self.qubitOp], [params])
        result = self.estimator.run(pubs=[pub]).result()
        energy = float(result[0].data.evs[0])
        self.history.append(energy)
        self.iterations += 1
        return energy


class SamplerCostFunction:
    """Sampler-based cost function for QAOA/VQE on QUBOs.

    Tracks convergence history. Supports standard expectation and CVaR.
    Use this when CVaR or sample-level statistics are needed.
    """

    def __init__(
        self,
        qubitOp: SparsePauliOp,
        offset: float,
        sampler,
        ansatz: QuantumCircuit,
        shots: int,
        variant: str = "standard",
        alpha: float = 0.25,
    ):
        self.qubitOp = qubitOp
        self.offset = offset
        self.sampler = sampler
        self.ansatz = ansatz
        self.shots = shots
        self.variant = variant
        self.alpha = alpha
        self.history: list[float] = []
        self.iterations: int = 0

        # Pre-compute Pauli info for fast bitstring evaluation
        self._paulis = qubitOp.paulis
        self._coeffs = qubitOp.coeffs

    def _eval_bitstring(self, bitstring: str) -> float:
        """Evaluate Ising energy for a bitstring."""
        spins = np.array([(-1) ** int(b) for b in bitstring[::-1]])
        value = 0.0
        for pauli, coeff in zip(self._paulis, self._coeffs):
            z_indices = np.where(pauli.z)[0]
            if len(z_indices) == 0:
                value += coeff.real
            else:
                value += coeff.real * np.prod(spins[z_indices])
        return value + self.offset

    def __call__(self, params: np.ndarray) -> float:
        assigned = self.ansatz.assign_parameters(params)
        meas_circuit = assigned.copy()
        if not any(inst.operation.name == "measure" for inst in meas_circuit.data):
            meas_circuit.measure_all()

        job = self.sampler.run([meas_circuit])
        result = job.result()
        try:
            counts = result[0].data.meas.get_counts()
        except AttributeError:
            counts = result[0].data.c.get_counts()

        total = sum(counts.values())
        bitstrings = list(counts.keys())
        probs = [counts[b] / total for b in bitstrings]
        values = [self._eval_bitstring(b) for b in bitstrings]

        if self.variant == "cvar":
            cost = self._compute_cvar(probs, values)
        else:
            cost = sum(p * v for p, v in zip(probs, values))

        self.history.append(float(cost))
        self.iterations += 1
        return cost

    def _compute_cvar(self, probs: list[float], values: list[float]) -> float:
        """CVaR: average over the alpha-tail of lowest values (for minimization)."""
        sorted_pairs = sorted(zip(values, probs))
        cumulative = 0.0
        cvar = 0.0
        for val, prob in sorted_pairs:
            if cumulative + prob > self.alpha:
                remaining = self.alpha - cumulative
                cvar += val * remaining
                cumulative = self.alpha
                break
            cvar += val * prob
            cumulative += prob
        return cvar / max(self.alpha, 1e-10)


# ─── Optimizer helpers ───────────────────────────────────────────


VALID_OPTIMIZERS = {
    "cobyla", "powell", "nelder-mead", "l-bfgs-b", "slsqp", "bfgs", "cg",
}

QISKIT_OPTIMIZERS = {"spsa", "adam"}


def _run_scipy_optimizer(cost_fn, initial_params, method, maxiter, tol):
    """Run a scipy optimizer."""
    method_lower = method.lower().replace("_", "-")
    if method_lower == "nelder-mead":
        method_lower = "Nelder-Mead"
    elif method_lower == "l-bfgs-b":
        method_lower = "L-BFGS-B"
    result = minimize(
        cost_fn,
        initial_params,
        method=method_lower,
        options={"maxiter": maxiter},
        tol=tol,
    )
    return result.x, float(result.fun)


def _run_qiskit_optimizer(cost_fn, initial_params, method, maxiter, lr):
    """Run a Qiskit optimizer (SPSA, ADAM)."""
    from qiskit_algorithms.optimizers import SPSA, ADAM

    if method.lower() == "spsa":
        opt = SPSA(maxiter=maxiter, learning_rate=lr, perturbation=lr * 0.5)
    else:
        opt = ADAM(maxiter=maxiter, lr=lr)

    result = opt.minimize(cost_fn, initial_params)
    return result.x, float(result.fun)


# ─── Main solver functions ───────────────────────────────────────


def solve_qubo_qaoa(
    problem: ProblemInstance,
    policy: dict,
    backend: BackendBundle,
) -> SolverResult:
    """Solve a QUBO problem (knapsack, MIS, etc.) using QAOA.

    Two-phase approach (matching the reference notebooks):
      1. **Optimise**: use ``BackendEstimatorV2`` to compute ⟨ψ|H|ψ⟩.
         If the policy selects ``measurement_mode=cvar``, a sampler-based
         CVaR cost function is used instead.
      2. **Sample**: run the optimised circuit through ``BackendSamplerV2``
         with ``sampler_shots`` and extract the most-probable bitstring.
    """
    from .maxcut_primitives import (
        build_multiangle_qaoa_circuit,
        generic_initial_point,
        normalize_measurement_policy,
        qaoa_initial_point,
        solver_mode_label,
    )

    qubitOp, offset = problem.qubo.to_ising()
    num_qubits = qubitOp.num_qubits

    reps = int(policy.get("reps", 2))
    variant, measurement_mode, cvar_alpha = normalize_measurement_policy(policy)
    optimizer_method = str(policy.get("optimizer_method", "COBYLA"))
    maxiter = int(policy.get("optimizer_maxiter", 150))
    opt_tol = float(policy.get("optimizer_tol", 1e-3))
    lr = float(policy.get("learning_rate", 0.05))
    est_shots = int(policy.get("estimator_shots", backend.shots))
    samp_shots = int(policy.get("sampler_shots", backend.sampler_shots))
    rng = np.random.default_rng(int(policy.get("seed", 42)))

    warm_bits = None
    if variant == "warmstart":
        ansatz, warm_bits = build_qubo_warmstart_qaoa_ansatz(
            problem,
            qubitOp,
            policy,
            rng,
        )
    elif variant == "multiangle":
        ansatz = build_multiangle_qaoa_circuit(
            problem,
            qubitOp,
            reps,
            str(policy.get("ma_tying", "none")).lower(),
        )
    elif variant == "standard":
        ansatz = build_qaoa_ansatz(qubitOp, reps)
    else:
        raise ValueError(
            f"Unsupported QAOA variant: {variant}. "
            "Use standard, warmstart, or multiangle."
        )

    # Build and decompose ansatz
    decomposed = ansatz.decompose(reps=2)
    num_params = decomposed.num_parameters

    # ── Phase 1: cost function ────────────────────────────────────
    if measurement_mode == "cvar":
        # CVaR needs per-bitstring energies → sampler-based
        opt_sampler = BackendSamplerV2(
            backend=backend.backend,
            options={"default_shots": est_shots},
        )
        cost_fn = SamplerCostFunction(
            qubitOp, offset, opt_sampler, decomposed, est_shots,
            variant="cvar", alpha=cvar_alpha,
        )
    else:
        # Standard expectation: use BackendEstimatorV2 (matches notebook)
        from qiskit.primitives import BackendEstimatorV2
        estimator = BackendEstimatorV2(backend=backend.backend)
        cost_fn = EstimatorCostFunction(qubitOp, offset, estimator, decomposed)

    # Initialize parameters
    if variant == "multiangle":
        initial_params = generic_initial_point(
            num_params,
            str(policy.get("initialization", "random")),
            rng,
        )
    else:
        initial_params = qaoa_initial_point(
            decomposed,
            reps,
            str(policy.get("initialization", "random")),
            rng,
            policy.get("initial_gamma"),
            policy.get("initial_beta"),
        )

    # Optimize
    method_lower = optimizer_method.lower()
    if method_lower in QISKIT_OPTIMIZERS:
        best_params, best_cost = _run_qiskit_optimizer(
            cost_fn, initial_params, method_lower, maxiter, lr
        )
    elif method_lower.replace("_", "-") in VALID_OPTIMIZERS or method_lower.replace("-", "_") in {"nelder_mead", "l_bfgs_b"}:
        best_params, best_cost = _run_scipy_optimizer(
            cost_fn, initial_params, optimizer_method, maxiter, opt_tol
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_method}")

    # ── Phase 2: final sampling ───────────────────────────────────
    final_sampler = BackendSamplerV2(
        backend=backend.backend,
        options={"default_shots": samp_shots},
    )
    final_circuit = decomposed.assign_parameters(best_params)
    final_circuit.measure_all()
    job = final_sampler.run([final_circuit])
    sample_result = job.result()
    try:
        final_counts = sample_result[0].data.meas.get_counts()
    except AttributeError:
        final_counts = sample_result[0].data.c.get_counts()

    # Extract solution (dispatches by problem type: MIS or knapsack)
    best_x, best_obj, is_feasible = extract_solution(final_counts, problem)

    # Circuit resources
    resources = summarize_circuit_resources(decomposed)

    return SolverResult(
        best_bitstring=best_x,
        best_objective=best_obj,
        is_feasible=is_feasible,
        counts=final_counts,
        num_shots=samp_shots,
        circuit_depth=resources["depth"],
        cnot_count=resources["cnot_count"],
        two_qubit_gate_count=resources["two_qubit_gate_count"],
        total_gate_count=resources["total_gate_count"],
        gate_counts=resources["gate_counts"],
        num_qubits=num_qubits,
        num_parameters=num_params,
        optimizer_iterations=cost_fn.iterations,
        final_cost=best_cost,
        convergence_history=cost_fn.history,
        solver_name=f"qaoa_{solver_mode_label(variant, measurement_mode)}",
        metadata={
            "variant": variant,
            "measurement_mode": measurement_mode,
            "reps": reps,
            "cvar_alpha": cvar_alpha if measurement_mode == "cvar" else None,
            "ws_source": policy.get("ws_source") if variant == "warmstart" else None,
            "ws_epsilon": policy.get("ws_epsilon") if variant == "warmstart" else None,
            "warm_start_bitstring": warm_bits.astype(int).tolist() if warm_bits is not None else None,
            "ma_tying": policy.get("ma_tying") if variant == "multiangle" else None,
        },
    )


def solve_qubo_vqe(
    problem: ProblemInstance,
    policy: dict,
    backend: BackendBundle,
) -> SolverResult:
    """Solve a QUBO problem (knapsack, MIS, etc.) using VQE.

    Two-phase approach (matching the reference notebook):
      1. **Optimise**: use ``BackendEstimatorV2`` to compute ⟨ψ|H|ψ⟩ and
         drive a classical optimiser (COBYLA by default).  If the policy
         selects ``measurement_mode=cvar``, a sampler-based CVaR cost
         function is used instead.
      2. **Sample**: run the optimised circuit through ``BackendSamplerV2``
         with ``sampler_shots`` and extract the most-probable bitstring.
    """
    from .maxcut_primitives import normalize_measurement_policy, solver_mode_label

    qubitOp, offset = problem.qubo.to_ising()
    num_qubits = qubitOp.num_qubits

    variant, measurement_mode, cvar_alpha = normalize_measurement_policy(policy)
    optimizer_method = str(policy.get("optimizer_method", "COBYLA"))
    maxiter = int(policy.get("optimizer_maxiter", 200))
    opt_tol = float(policy.get("optimizer_tol", 1e-3))
    lr = float(policy.get("learning_rate", 0.05))
    est_shots = int(policy.get("estimator_shots", backend.shots))
    samp_shots = int(policy.get("sampler_shots", backend.sampler_shots))

    # Build ansatz
    ansatz = build_vqe_ansatz(num_qubits, policy)
    decomposed = ansatz.decompose(reps=2)
    num_params = decomposed.num_parameters

    # ── Phase 1: cost function ────────────────────────────────────
    if measurement_mode == "cvar":
        # CVaR needs per-bitstring energies → sampler-based
        opt_sampler = BackendSamplerV2(
            backend=backend.backend,
            options={"default_shots": est_shots},
        )
        cost_fn = SamplerCostFunction(
            qubitOp, offset, opt_sampler, decomposed, est_shots,
            variant="cvar", alpha=cvar_alpha,
        )
    else:
        # Standard expectation: use BackendEstimatorV2 (matches notebook)
        from qiskit.primitives import BackendEstimatorV2
        estimator = BackendEstimatorV2(backend=backend.backend)
        cost_fn = EstimatorCostFunction(qubitOp, offset, estimator, decomposed)

    # Initialize parameters
    rng = np.random.default_rng(int(policy.get("seed", 42)))
    initial_params = rng.uniform(0, 2 * np.pi, num_params)

    # Optimize
    method_lower = optimizer_method.lower()
    if method_lower in QISKIT_OPTIMIZERS:
        best_params, best_cost = _run_qiskit_optimizer(
            cost_fn, initial_params, method_lower, maxiter, lr
        )
    elif method_lower.replace("_", "-") in VALID_OPTIMIZERS or method_lower.replace("-", "_") in {"nelder_mead", "l_bfgs_b"}:
        best_params, best_cost = _run_scipy_optimizer(
            cost_fn, initial_params, optimizer_method, maxiter, opt_tol
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_method}")

    # ── Phase 2: final sampling ───────────────────────────────────
    final_sampler = BackendSamplerV2(
        backend=backend.backend,
        options={"default_shots": samp_shots},
    )
    final_circuit = decomposed.assign_parameters(best_params)
    final_circuit.measure_all()
    job = final_sampler.run([final_circuit])
    sample_result = job.result()
    try:
        final_counts = sample_result[0].data.meas.get_counts()
    except AttributeError:
        final_counts = sample_result[0].data.c.get_counts()

    # Extract solution (dispatches by problem type: MIS or knapsack)
    best_x, best_obj, is_feasible = extract_solution(final_counts, problem)

    # Circuit resources
    resources = summarize_circuit_resources(decomposed)

    return SolverResult(
        best_bitstring=best_x,
        best_objective=best_obj,
        is_feasible=is_feasible,
        counts=final_counts,
        num_shots=samp_shots,
        circuit_depth=resources["depth"],
        cnot_count=resources["cnot_count"],
        two_qubit_gate_count=resources["two_qubit_gate_count"],
        total_gate_count=resources["total_gate_count"],
        gate_counts=resources["gate_counts"],
        num_qubits=num_qubits,
        num_parameters=num_params,
        optimizer_iterations=cost_fn.iterations,
        final_cost=best_cost,
        convergence_history=cost_fn.history,
        solver_name=f"vqe_{solver_mode_label(variant, measurement_mode)}",
        metadata={
            "variant": variant,
            "measurement_mode": measurement_mode,
            "ansatz_type": policy.get("ansatz_type", "efficient_su2"),
            "vqe_reps": policy.get("vqe_reps", 2),
            "cvar_alpha": cvar_alpha if measurement_mode == "cvar" else None,
        },
    )


def solve_qubo_pce(
    problem: ProblemInstance,
    policy: dict,
    backend,
) -> SolverResult:
    """Solve a QUBO problem by reducing to weighted MaxCut and running PCE."""

    from .maxcut_primitives import (
        align_observable,
        compute_cvar_from_counts,
        ensure_backend_bundle,
        generic_initial_point,
        get_estimator_shots,
        get_optimizer_maxiter,
        get_optimizer_method,
        get_optimizer_tol,
        get_sampler_shots,
        minimize_objective,
        normalize_measurement_policy,
        pce_find_n,
        pce_generate_pauli_strings,
        rescale_counts_to_shots,
        rng_from_policy,
        solver_mode_label,
        transpile_circuit,
    )

    sampler_shots = get_sampler_shots(policy, 1000)
    bundle = ensure_backend_bundle(backend, get_estimator_shots(policy, 1000), sampler_shots)
    rng = rng_from_policy(policy)
    _, measurement_mode, cvar_alpha = normalize_measurement_policy(policy)

    k = int(policy.get("pce_k", policy.get("k", 2)))
    if k not in {2, 3}:
        raise ValueError("PCE is restricted to k=2 or k=3 for this project.")

    graph = _qubo_pce_graph(problem)
    num_graph_nodes = int(graph.number_of_nodes())
    num_qubits = pce_find_n(num_graph_nodes, k)
    pauli_strings = pce_generate_pauli_strings(num_qubits, num_graph_nodes, k)
    observables = [SparsePauliOp.from_list([(string, 1.0)]) for string in pauli_strings]

    depth = int(policy.get("pce_depth", policy.get("depth", 10)))
    ansatz_type = str(policy.get("ansatz_type", "brickwork"))
    entanglement = str(policy.get("entanglement", "linear"))
    circuit = build_variational_ansatz(num_qubits, ansatz_type, depth, entanglement)
    compiled = transpile_circuit(bundle, circuit)
    aligned_observables = [align_observable(compiled, observable) for observable in observables]

    alpha_value = policy.get("pce_alpha")
    alpha = float(1.5 * num_qubits if alpha_value is None else alpha_value)
    beta = float(policy.get("pce_beta", 0.5))
    nu = _pce_regularization_scale_from_graph(graph)
    optimizer_method = get_optimizer_method(policy, default="COBYLA")
    optimizer_tol = get_optimizer_tol(policy)
    optimizer_maxiter = get_optimizer_maxiter(policy, default=100)
    learning_rate = float(policy.get("learning_rate", 0.05))

    def pce_loss(params: np.ndarray) -> float:
        result = bundle.estimator.run([(compiled, aligned_observables, params)]).result()[0]
        expectations = np.asarray(result.data.evs, dtype=float)
        transformed = np.tanh(alpha * expectations)

        loss = 0.0
        for left, right, data in graph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            loss += weight * transformed[left] * transformed[right]

        regularization = beta * nu * float((np.mean(transformed**2)) ** 2)
        return float(loss + regularization)

    x0 = generic_initial_point(compiled.num_parameters, policy.get("initialization", "random"), rng)
    outcome = minimize_objective(
        pce_loss,
        x0,
        optimizer_method,
        optimizer_maxiter,
        optimizer_tol,
        learning_rate,
    )

    result = bundle.estimator.run([(compiled, aligned_observables, outcome.params)]).result()[0]
    expectations = np.asarray(result.data.evs, dtype=float)
    graph_bits = (expectations >= 0.0).astype(float)
    initial_graph_bits = graph_bits.copy()
    initial_cut_value = _qubo_pce_cut_value(graph, graph_bits)
    sample_budget = None
    if measurement_mode == "cvar":
        sample_budget = max(
            1,
            int(policy.get("cvar_decode_samples", min(256, sampler_shots))),
        )
        graph_probs = np.clip((expectations + 1.0) / 2.0, 0.0, 1.0)
        raw_counts: dict[str, int] = {}
        best_sampled_bits = graph_bits.copy()
        best_sampled_cut = float("-inf")
        for _ in range(sample_budget):
            sampled_bits = (rng.random(num_graph_nodes) < graph_probs).astype(float)
            sampled_cut = _qubo_pce_cut_value(graph, sampled_bits)
            if bool(policy.get("pce_local_search", False)):
                sampled_bits, sampled_cut = _qubo_pce_local_search(graph, sampled_bits)
            if sampled_cut > best_sampled_cut:
                best_sampled_bits = sampled_bits.copy()
                best_sampled_cut = sampled_cut
            qubo_bits = np.asarray(sampled_bits[1:], dtype=float)
            bitstring = "".join(str(int(bit)) for bit in qubo_bits[::-1])
            raw_counts[bitstring] = raw_counts.get(bitstring, 0) + 1
        counts = rescale_counts_to_shots(raw_counts, sampler_shots)
        graph_bits = best_sampled_bits
        final_cut_value = float(best_sampled_cut)
    else:
        if bool(policy.get("pce_local_search", False)):
            graph_bits, final_cut_value = _qubo_pce_local_search(graph, graph_bits)
        else:
            final_cut_value = initial_cut_value

        # The notebook path treats node 0 as the added auxiliary node and drops it.
        qubo_bits = np.asarray(graph_bits[1:], dtype=float)
        counts = {
            "".join(str(int(bit)) for bit in qubo_bits[::-1]): sampler_shots,
        }

    best_x, best_obj, is_feasible = extract_solution(counts, problem)
    resources = summarize_circuit_resources(compiled)
    final_cost = (
        float(compute_cvar_from_counts(problem, counts, cvar_alpha))
        if measurement_mode == "cvar"
        else float(outcome.value)
    )

    return SolverResult(
        best_bitstring=best_x,
        best_objective=best_obj,
        is_feasible=is_feasible,
        counts=counts,
        num_shots=int(sampler_shots),
        circuit_depth=int(resources["depth"]),
        cnot_count=int(resources["cnot_count"]),
        two_qubit_gate_count=int(resources["two_qubit_gate_count"]),
        total_gate_count=int(resources["total_gate_count"]),
        gate_counts=dict(resources["gate_counts"]),
        num_qubits=compiled.num_qubits,
        num_parameters=compiled.num_parameters,
        optimizer_iterations=int(outcome.iterations),
        final_cost=final_cost,
        parameter_values=np.asarray(outcome.params, dtype=float),
        convergence_history=[float(value) for value in outcome.history],
        solver_name=f"pce_{solver_mode_label(f'k{k}', measurement_mode)}",
        metadata={
            "policy": dict(policy),
            "encoded_qubits": int(num_qubits),
            "pce_graph_nodes": int(num_graph_nodes),
            "compression_ratio": float(num_qubits / max(1, problem.num_variables)),
            "ansatz_type": ansatz_type,
            "pce_alpha": alpha,
            "pce_beta": beta,
            "pce_nu": nu,
            "measurement_mode": measurement_mode,
            "cvar_alpha": cvar_alpha if measurement_mode == "cvar" else None,
            "cvar_decode_samples": sample_budget,
            "pre_local_search_graph_bits": initial_graph_bits.astype(int).tolist(),
            "post_local_search_graph_bits": graph_bits.astype(int).tolist(),
            "pre_local_search_cut_value": float(initial_cut_value),
            "post_local_search_cut_value": float(final_cut_value),
        },
    )
