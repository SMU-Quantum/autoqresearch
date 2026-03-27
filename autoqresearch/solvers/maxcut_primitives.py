"""MaxCut-first primitive workflows shared by the solver wrappers."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import (
    ExcitationPreserving,
    PauliEvolutionGate,
    QAOAAnsatz,
    RXXGate,
    RXGate,
    RYYGate,
    RZGate,
    RZZGate,
    TwoLocal,
    efficient_su2,
    pauli_two_design,
    real_amplitudes,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import ADAM, COBYLA as QiskitCOBYLA, SPSA
from qiskit_optimization.algorithms import SlsqpOptimizer
from qiskit_optimization.algorithms.qrao import (
    MagicRounding,
    QuantumRandomAccessEncoding,
    QuantumRandomAccessOptimizer,
    SemideterministicRounding,
)
from qiskit_optimization.minimum_eigensolvers import VQE as OptimizationVQE
from qiskit_optimization.problems.variable import VarType

from ..backends.factory import BackendBundle, BackendConfig, create_execution_context
from ..problems.base import ProblemInstance
from .base import SolverResult, extract_best_solution, summarize_circuit_resources


@dataclass
class OptimizationOutcome:
    """Normalized optimizer output."""

    params: np.ndarray
    value: float
    iterations: int
    history: list[float]


class SamplerAdapter:
    """Adapter to satisfy MagicRounding's expectation of default_shots."""

    def __init__(self, sampler, default_shots: int):
        self._sampler = sampler
        self.default_shots = int(default_shots)

    def run(self, circuits, shots=None):
        return self._sampler.run(circuits, shots=shots)


def ensure_backend_bundle(
    backend,
    shots: int,
    sampler_shots: int | None = None,
) -> BackendBundle:
    """Normalize solver input into a BackendBundle."""

    if isinstance(backend, BackendBundle):
        return backend
    if hasattr(backend, "estimator") and hasattr(backend, "sampler"):
        return backend
    return create_execution_context(
        BackendConfig(
            mode="ideal_mps",
            shots=int(shots),
            sampler_shots=int(sampler_shots or shots),
        )
    )


def require_maxcut(problem: ProblemInstance) -> None:
    """Explicitly enforce the current project scope."""

    if problem.problem_type != "maxcut":
        raise NotImplementedError(
            "The solver package is now wired for MaxCut-first experiments. "
            "Translate other QUBOs to MaxCut before using these solvers."
        )


def rng_from_policy(policy: dict) -> np.random.Generator:
    """Build a deterministic RNG when the policy provides a seed."""

    return np.random.default_rng(policy.get("seed"))


def get_qaoa_reps(policy: dict) -> int:
    """Support both the old `p` key and the newer `reps` key."""

    return int(policy.get("reps", policy.get("p", 1)))


def get_optimizer_method(policy: dict, default: str = "COBYLA") -> str:
    """Support both the old and the new optimizer keys."""

    return str(policy.get("optimizer_method", policy.get("optimizer", default)))


def get_optimizer_maxiter(policy: dict, default: int = 200) -> int:
    """Support both the old and the new max-iteration keys."""

    return int(policy.get("optimizer_maxiter", policy.get("maxiter", default)))


def get_optimizer_tol(policy: dict, default: float = 1e-3) -> float:
    """Return optimizer tolerance."""

    return float(policy.get("optimizer_tol", policy.get("tol", default)))


def get_estimator_shots(policy: dict, default: int) -> int:
    """Estimator shots used by stochastic objectives."""

    return int(policy.get("estimator_shots", policy.get("shots", default)))


def get_sampler_shots(policy: dict, default: int) -> int:
    """Final sampling shots."""

    return int(policy.get("sampler_shots", policy.get("shots", default)))


def get_num_restarts(policy: dict) -> int:
    """Number of optimizer restarts."""

    return max(1, int(policy.get("num_restarts", 1)))


def transpile_circuit(bundle: BackendBundle, circuit: QuantumCircuit) -> QuantumCircuit:
    """Return the circuit unchanged.

    The project now runs all solver families on the same non-transpiled path so
    backend-level preprocessing does not advantage one family over another.
    """

    return circuit


def align_observable(circuit: QuantumCircuit, observable: SparsePauliOp) -> SparsePauliOp:
    """Apply the transpiled layout to an observable when one exists."""

    layout = getattr(circuit, "layout", None)
    if layout is None:
        return observable
    return observable.apply_layout(layout)


def sample_counts(
    bundle: BackendBundle,
    circuit: QuantumCircuit,
    params: np.ndarray | None,
    shots: int,
) -> dict[str, int]:
    """Bind parameters, measure the circuit, and return raw counts."""

    assigned = circuit.assign_parameters(params) if params is not None else circuit.copy()
    measured = assigned.copy()
    measured.measure_all()
    result = bundle.sampler.run([measured], shots=int(shots)).result()[0]
    return dict(result.data.meas.get_counts())


def counts_objective_lookup(problem: ProblemInstance, counts: dict[str, int]) -> dict[str, float]:
    """Cache the QUBO objective for each measured bitstring."""

    lookup: dict[str, float] = {}
    for bitstring in counts:
        x = np.array([int(bit) for bit in bitstring[::-1]], dtype=float)
        if len(x) < problem.num_variables:
            x = np.pad(x, (0, problem.num_variables - len(x)))
        elif len(x) > problem.num_variables:
            x = x[: problem.num_variables]
        lookup[bitstring] = float(problem.objective_value(x))
    return lookup


def compute_cvar_from_counts(
    problem: ProblemInstance,
    counts: dict[str, int],
    alpha: float,
) -> float:
    """Compute a minimization-compatible CVaR objective from sampled values."""

    if not counts:
        return float("inf")

    lookup = counts_objective_lookup(problem, counts)
    maximize = problem.qubo.objective.sense.name == "MAXIMIZE"
    pairs = sorted(
        (
            ((-lookup[bitstring]) if maximize else lookup[bitstring], count)
            for bitstring, count in counts.items()
        ),
        key=lambda item: item[0],
    )
    total = sum(count for _, count in pairs)
    cutoff = max(1, int(math.ceil(alpha * total)))

    taken = 0
    total_value = 0.0
    for value, count in pairs:
        used = min(count, cutoff - taken)
        total_value += value * used
        taken += used
        if taken >= cutoff:
            break

    return total_value / max(taken, 1)


def normalize_measurement_policy(
    policy: dict,
    default_variant: str = "standard",
) -> tuple[str, str, float]:
    """Separate structural variant choice from CVaR measurement mode."""

    variant = str(policy.get("variant", default_variant)).lower()
    measurement_mode = str(
        policy.get(
            "measurement_mode",
            policy.get("objective_mode", "cvar" if variant == "cvar" else "expectation"),
        )
    ).lower()
    if bool(policy.get("use_cvar", False)):
        measurement_mode = "cvar"
    if variant == "cvar":
        variant = default_variant
        measurement_mode = "cvar"

    if measurement_mode in {"expectation", "standard", "mean"}:
        measurement_mode = "expectation"
    elif measurement_mode != "cvar":
        raise ValueError(
            f"Unsupported measurement mode: {measurement_mode}. Use expectation or cvar."
        )

    return variant, measurement_mode, float(policy.get("cvar_alpha", 0.25))


def solver_mode_label(variant: str, measurement_mode: str) -> str:
    normalized_variant = str(variant).lower()
    if str(measurement_mode).lower() == "cvar":
        return "cvar" if normalized_variant == "standard" else f"cvar_{normalized_variant}"
    return normalized_variant


def solve_relaxed_problem(problem: ProblemInstance) -> np.ndarray | None:
    """Solve the continuous relaxation when possible."""

    relaxed_problem = copy.deepcopy(problem.qubo)
    for variable in relaxed_problem.variables:
        variable.vartype = VarType.CONTINUOUS

    try:
        result = SlsqpOptimizer().solve(relaxed_problem)
        return np.asarray(result.x, dtype=float)
    except Exception:
        return None


def heuristic_maxcut_bitstring(
    problem: ProblemInstance,
    source: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a classical MaxCut bitstring for warm starts."""

    n = problem.num_variables
    graph = problem.metadata.get("graph")
    if graph is None:
        return rng.integers(0, 2, size=n).astype(float)

    if source == "random":
        return rng.integers(0, 2, size=n).astype(float)

    if source == "lp":
        relaxed = solve_relaxed_problem(problem)
        if relaxed is not None and not np.allclose(relaxed, relaxed[0]):
            return (relaxed >= 0.5).astype(float)

    if source == "sdp":
        try:
            from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer

            result = GoemansWilliamsonOptimizer(num_cuts=1).solve(problem.qubo)
            return np.asarray(result.x, dtype=float)
        except Exception:
            pass

    try:
        _, partition = nx.algorithms.approximation.maxcut.one_exchange(
            graph,
            seed=int(rng.integers(0, 1 << 31)),
        )
        bits = np.zeros(n, dtype=float)
        chosen = partition[0] if len(partition) >= 1 else set()
        for node in chosen:
            bits[int(node)] = 1.0
        return bits
    except Exception:
        return rng.integers(0, 2, size=n).astype(float)


def make_warm_start_state(
    problem: ProblemInstance,
    source: str,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[QuantumCircuit, QuantumCircuit, np.ndarray]:
    """Create the warm-start initial state and custom mixer."""

    warm_bits = heuristic_maxcut_bitstring(problem, source, rng)
    if source == "lp":
        relaxed = solve_relaxed_problem(problem)
        if relaxed is not None:
            c_stars = np.clip(relaxed, epsilon, 1.0 - epsilon)
        else:
            c_stars = np.clip(epsilon + (1.0 - 2.0 * epsilon) * warm_bits, epsilon, 1.0 - epsilon)
    else:
        c_stars = np.clip(epsilon + (1.0 - 2.0 * epsilon) * warm_bits, epsilon, 1.0 - epsilon)

    thetas = [2.0 * np.arcsin(np.sqrt(value)) for value in c_stars]
    init_state = QuantumCircuit(problem.num_variables)
    mixer = QuantumCircuit(problem.num_variables)
    beta = Parameter("beta")
    for idx, theta in enumerate(thetas):
        init_state.ry(theta, idx)
        mixer.ry(-theta, idx)
        mixer.rz(-2.0 * beta, idx)
        mixer.ry(theta, idx)
    return init_state, mixer, warm_bits


def extract_index(parameter_name: str) -> int:
    """Extract the integer suffix from a parameter name like beta[2]."""

    start = parameter_name.find("[")
    end = parameter_name.find("]")
    if start < 0 or end < 0:
        return 0
    return int(parameter_name[start + 1 : end])


def pack_qaoa_parameters(
    circuit: QuantumCircuit,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> np.ndarray:
    """Pack beta/gamma arrays into the circuit's parameter order."""

    values = []
    for parameter in circuit.parameters:
        name = parameter.name.lower()
        idx = extract_index(parameter.name)
        if "beta" in name or "β" in parameter.name:
            values.append(float(betas[idx]))
        elif "gamma" in name or "γ" in parameter.name:
            values.append(float(gammas[idx]))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


def unpack_qaoa_parameters(
    circuit: QuantumCircuit,
    parameter_values: np.ndarray,
    reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-layer beta and gamma values from a parameter vector."""

    betas = np.zeros(reps, dtype=float)
    gammas = np.zeros(reps, dtype=float)
    for parameter, value in zip(circuit.parameters, parameter_values):
        idx = extract_index(parameter.name)
        name = parameter.name.lower()
        if "beta" in name or "β" in parameter.name:
            betas[idx] = float(value)
        elif "gamma" in name or "γ" in parameter.name:
            gammas[idx] = float(value)
    return betas, gammas


def qaoa_initial_point(
    circuit: QuantumCircuit,
    reps: int,
    strategy: str,
    rng: np.random.Generator,
    initial_gamma: float | None = None,
    initial_beta: float | None = None,
) -> np.ndarray:
    """Create an initial parameter vector for QAOA circuits."""

    strategy = str(strategy).lower()
    if strategy == "zeros":
        gammas = np.zeros(reps)
        betas = np.zeros(reps)
    elif strategy == "pi_over_2":
        gammas = np.full(reps, np.pi)
        betas = np.full(reps, np.pi / 2.0)
    elif strategy == "tqa":
        gammas = np.linspace(0.1, np.pi / 2.0, reps)
        betas = np.linspace(np.pi / 2.0, 0.1, reps)
    elif strategy == "interp":
        gammas = np.linspace(0.2, 0.8, reps) * np.pi / 2.0
        betas = np.linspace(0.8, 0.2, reps) * np.pi / 2.0
    else:
        gammas = rng.uniform(0.0, 2.0 * np.pi, size=reps)
        betas = rng.uniform(0.0, np.pi, size=reps)

    if initial_gamma is not None:
        gammas = np.full(reps, float(initial_gamma))
    if initial_beta is not None:
        betas = np.full(reps, float(initial_beta))
    return pack_qaoa_parameters(circuit, betas, gammas)


def generic_initial_point(
    num_params: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create generic variational initial points."""

    lowered = str(strategy).lower()
    if lowered == "zeros":
        return np.zeros(num_params, dtype=float)
    if lowered == "small_random":
        return rng.uniform(-0.1, 0.1, size=num_params)
    return rng.uniform(0.0, 2.0 * np.pi, size=num_params)


def minimize_objective(
    objective,
    x0: np.ndarray,
    method: str,
    maxiter: int,
    tol: float,
    learning_rate: float,
) -> OptimizationOutcome:
    """Optimize a black-box objective and normalize the result."""

    history: list[float] = []

    def tracked(params):
        value = float(objective(np.asarray(params, dtype=float)))
        history.append(value)
        return value

    normalized = method.replace("-", "_").upper()
    if normalized in {"COBYLA", "POWELL", "SLSQP", "L_BFGS_B", "NELDER_MEAD", "BFGS", "CG"}:
        scipy_method = {
            "COBYLA": "COBYLA",
            "POWELL": "Powell",
            "SLSQP": "SLSQP",
            "L_BFGS_B": "L-BFGS-B",
            "NELDER_MEAD": "Nelder-Mead",
            "BFGS": "BFGS",
            "CG": "CG",
        }[normalized]
        result = minimize(
            tracked,
            x0,
            method=scipy_method,
            tol=tol,
            options={"maxiter": int(maxiter)},
        )
        iterations = int(getattr(result, "nfev", 0) or getattr(result, "nit", 0))
        return OptimizationOutcome(np.asarray(result.x, dtype=float), float(result.fun), iterations, history)

    if normalized == "SPSA":
        optimizer = SPSA(maxiter=int(maxiter))
        result = optimizer.minimize(fun=tracked, x0=x0)
        iterations = int(getattr(result, "nfev", 0) or maxiter)
        return OptimizationOutcome(np.asarray(result.x, dtype=float), float(result.fun), iterations, history)

    if normalized == "ADAM":
        optimizer = ADAM(maxiter=int(maxiter), lr=float(learning_rate))
        result = optimizer.minimize(fun=tracked, x0=x0)
        iterations = int(getattr(result, "nfev", 0) or maxiter)
        return OptimizationOutcome(np.asarray(result.x, dtype=float), float(result.fun), iterations, history)

    raise ValueError(f"Unsupported optimizer method: {method}")


def build_xy_mixer(num_qubits: int) -> QuantumCircuit:
    """Create an XY mixer circuit."""

    beta = Parameter("beta")
    mixer = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits - 1):
        mixer.append(RXXGate(2.0 * beta), [qubit, qubit + 1])
        mixer.append(RYYGate(2.0 * beta), [qubit, qubit + 1])
    return mixer


def generate_sum_x_pauli_str(length: int) -> list[str]:
    """Generate the single-qubit X terms used for manual QAOA."""

    output = []
    for index in range(length):
        paulis = ["I"] * length
        paulis[index] = "X"
        output.append("".join(paulis))
    return output


def build_qaoa_circuit(
    problem: ProblemInstance,
    qubit_op: SparsePauliOp,
    policy: dict,
    init_state: QuantumCircuit | None = None,
    mixer_operator: QuantumCircuit | None = None,
) -> QuantumCircuit:
    """Build the standard QAOA circuit for the current policy."""

    reps = get_qaoa_reps(policy)
    circuit_type = str(policy.get("circuit_type", "qaoa_ansatz")).lower()
    mixer_type = str(policy.get("mixer", "x")).lower()

    if circuit_type == "pauli_evolution" and mixer_type == "x" and init_state is None and mixer_operator is None:
        gamma = ParameterVector("gamma", reps)
        beta = ParameterVector("beta", reps)
        mixer_ham = SparsePauliOp(generate_sum_x_pauli_str(qubit_op.num_qubits))
        circuit = QuantumCircuit(qubit_op.num_qubits)
        circuit.h(range(qubit_op.num_qubits))
        for layer in range(reps):
            circuit.append(PauliEvolutionGate(qubit_op, time=gamma[layer]), qargs=range(qubit_op.num_qubits))
            circuit.append(PauliEvolutionGate(mixer_ham, time=beta[layer]), qargs=range(qubit_op.num_qubits))
    else:
        mixer = mixer_operator
        if mixer is None and mixer_type == "xy":
            mixer = build_xy_mixer(qubit_op.num_qubits)
        circuit = QAOAAnsatz(
            cost_operator=qubit_op,
            reps=reps,
            initial_state=init_state,
            mixer_operator=mixer,
        )

    decompose_reps = int(policy.get("decompose_reps", 2))
    return circuit.decompose(reps=decompose_reps)


def build_multiangle_qaoa_circuit(
    problem: ProblemInstance,
    qubit_op: SparsePauliOp,
    reps: int,
    tying: str,
) -> QuantumCircuit:
    """Build the MA-QAOA circuit using per-term or tied parameters."""

    graph = problem.metadata.get("graph")
    degrees = dict(graph.degree()) if graph is not None else {}
    colors = nx.coloring.greedy_color(graph, strategy="largest_first") if graph is not None else {}

    cache: dict[tuple, Parameter] = {}

    def grouped_parameter(kind: str, layer: int, qubits: tuple[int, ...]) -> Parameter:
        normalized_tying = str(tying).lower()
        if normalized_tying == "full":
            key = (kind, layer, "all")
        elif normalized_tying in {"partial", "degree"} and graph is not None:
            if kind == "mixer":
                key = (kind, layer, degrees.get(qubits[0], 0))
            else:
                key = (kind, layer, tuple(sorted(degrees.get(qubit, 0) for qubit in qubits)))
        elif normalized_tying == "partition" and graph is not None:
            if kind == "mixer":
                key = (kind, layer, colors.get(qubits[0], 0))
            else:
                key = (kind, layer, tuple(sorted(colors.get(qubit, 0) for qubit in qubits)))
        else:
            key = (kind, layer, qubits)

        if key not in cache:
            cache[key] = Parameter(f"{kind}_{layer}_{len(cache)}")
        return cache[key]

    circuit = QuantumCircuit(qubit_op.num_qubits)
    circuit.h(range(qubit_op.num_qubits))

    for layer in range(reps):
        for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
            indices = [idx for idx, value in enumerate(reversed(str(pauli))) if value == "Z"]
            if len(indices) == 1:
                theta = grouped_parameter("cost", layer, (indices[0],))
                circuit.append(RZGate(2.0 * coeff.real * theta), [indices[0]])
            elif len(indices) == 2:
                theta = grouped_parameter("cost", layer, tuple(indices))
                circuit.append(RZZGate(2.0 * coeff.real * theta), indices)

        for qubit in range(qubit_op.num_qubits):
            theta = grouped_parameter("mixer", layer, (qubit,))
            circuit.append(RXGate(2.0 * theta), [qubit])

    return circuit


def build_vqe_ansatz(
    num_qubits: int,
    ansatz_type: str,
    entanglement: str,
    reps: int,
    rotation_blocks: list[str] | None,
) -> QuantumCircuit:
    """Construct the VQE ansatz requested by the policy."""

    lowered = ansatz_type.lower()
    if lowered == "efficient_su2":
        return efficient_su2(num_qubits, reps=reps, entanglement=entanglement).decompose()
    if lowered == "real_amplitudes":
        return real_amplitudes(num_qubits, reps=reps, entanglement=entanglement).decompose()
    if lowered in {"pauli_two_design", "paulitwodesign"}:
        return pauli_two_design(num_qubits=num_qubits, reps=reps).decompose()
    if lowered == "brickwork":
        return build_brickwork_ansatz(reps, num_qubits)
    if lowered == "two_local":
        return TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks or ["ry", "rz"],
            entanglement_blocks="cx",
            entanglement=entanglement,
            reps=reps,
        ).decompose()
    if lowered == "excitation_preserving":
        return ExcitationPreserving(
            num_qubits=num_qubits,
            entanglement=entanglement,
            reps=reps,
        ).decompose()
    raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def maximize_cut_local_search(problem: ProblemInstance, bitstring: np.ndarray) -> np.ndarray:
    """Greedy one-bit local search on cut value."""

    graph = problem.metadata.get("graph")
    if graph is None:
        return bitstring

    current = bitstring.astype(float).copy()
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_index = None
        current_value = cut_value(problem, current)
        for index in range(len(current)):
            trial = current.copy()
            trial[index] = 1.0 - trial[index]
            gain = cut_value(problem, trial) - current_value
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is not None:
            current[best_index] = 1.0 - current[best_index]
            improved = True
    return current


def cut_value(problem: ProblemInstance, bitstring: np.ndarray) -> float:
    """Evaluate the MaxCut objective on the original graph scale."""

    graph = problem.metadata.get("graph")
    if graph is None:
        return 0.0

    return float(
        sum(
            graph[u][v].get("weight", 1.0)
            for u, v in graph.edges()
            if int(bitstring[u]) != int(bitstring[v])
        )
    )


def assemble_solver_result(
    problem: ProblemInstance,
    solver_name: str,
    counts: dict[str, int],
    num_shots: int,
    circuit: QuantumCircuit,
    best_params: np.ndarray | None,
    convergence_history: list[float],
    final_cost: float,
    optimizer_iterations: int,
    extra_metadata: dict | None = None,
) -> SolverResult:
    """Build the common SolverResult object."""

    best_x, best_obj, is_feasible = extract_best_solution(counts, problem)
    resources = summarize_circuit_resources(circuit)
    metadata = {
        "objective_lookup": counts_objective_lookup(problem, counts),
        "objective_sense": problem.qubo.objective.sense.name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    policy = metadata.get("policy", {}) if isinstance(metadata.get("policy", {}), dict) else {}
    apply_final_local_search = False
    if problem.problem_type == "maxcut":
        if "final_local_search" in policy:
            apply_final_local_search = bool(policy.get("final_local_search"))
        elif "pce_local_search" in policy:
            apply_final_local_search = bool(policy.get("pce_local_search"))
        else:
            apply_final_local_search = False

    original_best_x = best_x.copy()
    original_best_obj = float(best_obj)
    local_search_improved = False
    if apply_final_local_search:
        original_cut = cut_value(problem, best_x)
        refined_x = maximize_cut_local_search(problem, best_x)
        refined_cut = cut_value(problem, refined_x)
        refined_obj = float(problem.objective_value(refined_x))
        if refined_cut > original_cut + 1e-12:
            best_x = refined_x
            best_obj = refined_obj
            is_feasible = bool(problem.is_feasible(refined_x))
            local_search_improved = True

    metadata["final_local_search_applied"] = bool(apply_final_local_search)
    metadata["final_local_search_improved"] = bool(local_search_improved)
    if apply_final_local_search:
        metadata["pre_local_search_bitstring"] = original_best_x.astype(int).tolist()
        metadata["pre_local_search_objective"] = original_best_obj
        metadata["pre_local_search_cut_value"] = float(original_cut)
        metadata["post_local_search_cut_value"] = float(cut_value(problem, best_x))

    return SolverResult(
        best_bitstring=best_x,
        best_objective=best_obj,
        is_feasible=is_feasible,
        counts=counts,
        num_shots=int(num_shots),
        circuit_depth=int(resources["depth"]),
        cnot_count=int(resources["cnot_count"]),
        two_qubit_gate_count=int(resources["two_qubit_gate_count"]),
        total_gate_count=int(resources["total_gate_count"]),
        gate_counts=dict(resources["gate_counts"]),
        num_qubits=circuit.num_qubits,
        num_parameters=circuit.num_parameters,
        optimizer_iterations=int(optimizer_iterations),
        final_cost=float(final_cost),
        parameter_values=None if best_params is None else np.asarray(best_params, dtype=float),
        convergence_history=[float(value) for value in convergence_history],
        solver_name=solver_name,
        metadata=metadata,
    )


def solve_qaoa_variant(problem: ProblemInstance, policy: dict, backend, variant: str) -> SolverResult:
    """Solve MaxCut with the requested QAOA-family variant."""

    require_maxcut(problem)
    sampler_shots = get_sampler_shots(policy, 1000)
    bundle = ensure_backend_bundle(backend, get_estimator_shots(policy, 1000), sampler_shots)
    rng = rng_from_policy(policy)
    qubit_op, offset = problem.qubo.to_ising()
    reps = get_qaoa_reps(policy)
    learning_rate = float(policy.get("learning_rate", 0.05))
    optimizer_method = get_optimizer_method(policy)
    optimizer_tol = get_optimizer_tol(policy)
    optimizer_maxiter = get_optimizer_maxiter(policy)
    variant, measurement_mode, alpha = normalize_measurement_policy(
        {**policy, "variant": variant}
    )

    warm_bits = None
    init_state = None
    mixer_operator = None
    if variant == "warmstart":
        epsilon = float(policy.get("ws_epsilon", 0.25))
        source = str(policy.get("ws_source", "greedy")).lower()
        init_state, mixer_operator, warm_bits = make_warm_start_state(problem, source, epsilon, rng)

    standard_seed = None
    if variant == "multiangle":
        circuit = build_multiangle_qaoa_circuit(
            problem,
            qubit_op,
            reps,
            str(policy.get("ma_tying", "none")).lower(),
        )
        if policy.get("ma_transfer_from_standard"):
            seed_policy = {
                **policy,
                "variant": "standard",
                "measurement_mode": measurement_mode,
                "num_restarts": 1,
            }
            seed_result = solve_qaoa_variant(problem, seed_policy, backend, "standard")
            if seed_result.parameter_values is not None:
                beta_seed, gamma_seed = unpack_qaoa_parameters(
                    build_qaoa_circuit(problem, qubit_op, policy),
                    seed_result.parameter_values,
                    reps,
                )
                initial = []
                for parameter in circuit.parameters:
                    if parameter.name.startswith("mixer_"):
                        layer = int(parameter.name.split("_")[1])
                        initial.append(beta_seed[min(layer, len(beta_seed) - 1)])
                    else:
                        layer = int(parameter.name.split("_")[1])
                        initial.append(gamma_seed[min(layer, len(gamma_seed) - 1)])
                standard_seed = np.asarray(initial, dtype=float)
    else:
        circuit = build_qaoa_circuit(
            problem,
            qubit_op,
            policy,
            init_state=init_state,
            mixer_operator=mixer_operator,
        )

    compiled = transpile_circuit(bundle, circuit)
    observable = align_observable(compiled, qubit_op)

    def estimator_objective(params: np.ndarray) -> float:
        result = bundle.estimator.run([(compiled, observable, params)]).result()[0]
        return float(np.asarray(result.data.evs).item()) + float(offset)

    estimator_shots = get_estimator_shots(policy, bundle.shots)

    def cvar_objective(params: np.ndarray, current_alpha: float) -> float:
        counts = sample_counts(bundle, compiled, params, estimator_shots)
        return compute_cvar_from_counts(problem, counts, current_alpha)

    best_outcome = None
    for restart in range(get_num_restarts(policy)):
        if variant == "multiangle":
            if restart == 0 and standard_seed is not None:
                x0 = standard_seed
            else:
                x0 = generic_initial_point(
                    compiled.num_parameters,
                    policy.get("initialization", "random"),
                    rng,
                )
        else:
            x0 = qaoa_initial_point(
                compiled,
                reps,
                str(policy.get("initialization", "random")),
                rng,
                policy.get("initial_gamma"),
                policy.get("initial_beta"),
            )
            if restart > 0:
                x0 = qaoa_initial_point(compiled, reps, "random", rng, None, None)

        if measurement_mode == "cvar":
            schedule = str(policy.get("alpha_schedule", "fixed")).lower()
            alphas = [alpha]
            if schedule == "anneal" and alpha < 1.0:
                alphas = []
                for value in (1.0, max(alpha, 0.75), max(alpha, 0.5), alpha):
                    if value not in alphas:
                        alphas.append(value)

            history: list[float] = []
            params = x0
            final_value = float("inf")
            total_iterations = 0
            for current_alpha in alphas:
                outcome = minimize_objective(
                    lambda current_params: cvar_objective(current_params, current_alpha),
                    params,
                    optimizer_method,
                    optimizer_maxiter,
                    optimizer_tol,
                    learning_rate,
                )
                params = outcome.params
                final_value = outcome.value
                total_iterations += outcome.iterations
                history.extend(outcome.history)
            outcome = OptimizationOutcome(params=params, value=final_value, iterations=total_iterations, history=history)
        else:
            outcome = minimize_objective(
                estimator_objective,
                x0,
                optimizer_method,
                optimizer_maxiter,
                optimizer_tol,
                learning_rate,
            )

        if best_outcome is None or outcome.value < best_outcome.value:
            best_outcome = outcome

    counts = sample_counts(bundle, compiled, best_outcome.params, sampler_shots)
    metadata = {
        "policy": dict(policy),
        "offset": float(offset),
        "variant": variant,
        "measurement_mode": measurement_mode,
        "cvar_alpha": alpha if measurement_mode == "cvar" else None,
    }
    if warm_bits is not None:
        metadata["warm_start_bitstring"] = warm_bits.tolist()

    return assemble_solver_result(
        problem,
        f"qaoa_{solver_mode_label(variant, measurement_mode)}",
        counts,
        sampler_shots,
        compiled,
        best_outcome.params,
        best_outcome.history,
        best_outcome.value,
        best_outcome.iterations,
        metadata,
    )


def solve_vqe_variant(problem: ProblemInstance, policy: dict, backend, variant: str) -> SolverResult:
    """Solve MaxCut with the requested VQE-family variant."""

    require_maxcut(problem)
    sampler_shots = get_sampler_shots(policy, 1000)
    bundle = ensure_backend_bundle(backend, get_estimator_shots(policy, 1000), sampler_shots)
    rng = rng_from_policy(policy)
    qubit_op, offset = problem.qubo.to_ising()

    ansatz_type = str(policy.get("ansatz_type", policy.get("ansatz", "efficient_su2")))
    entanglement = str(policy.get("entanglement", "linear"))
    reps = int(policy.get("vqe_reps", policy.get("reps", 2)))
    rotation_blocks = policy.get("rotation_blocks")

    circuit = build_vqe_ansatz(qubit_op.num_qubits, ansatz_type, entanglement, reps, rotation_blocks)
    compiled = transpile_circuit(bundle, circuit)
    observable = align_observable(compiled, qubit_op)

    learning_rate = float(policy.get("learning_rate", 0.05))
    optimizer_method = get_optimizer_method(policy)
    optimizer_tol = get_optimizer_tol(policy)
    optimizer_maxiter = get_optimizer_maxiter(policy)
    estimator_shots = get_estimator_shots(policy, bundle.shots)
    variant, measurement_mode, alpha = normalize_measurement_policy(
        {**policy, "variant": variant}
    )

    def estimator_objective(params: np.ndarray) -> float:
        result = bundle.estimator.run([(compiled, observable, params)]).result()[0]
        return float(np.asarray(result.data.evs).item()) + float(offset)

    def cvar_objective(params: np.ndarray) -> float:
        counts = sample_counts(bundle, compiled, params, estimator_shots)
        return compute_cvar_from_counts(problem, counts, alpha)

    best_outcome = None
    for restart in range(get_num_restarts(policy)):
        init_strategy = str(policy.get("initialization", "random"))
        if restart > 0:
            init_strategy = "random"
        x0 = generic_initial_point(compiled.num_parameters, init_strategy, rng)
        outcome = minimize_objective(
            cvar_objective if measurement_mode == "cvar" else estimator_objective,
            x0,
            optimizer_method,
            optimizer_maxiter,
            optimizer_tol,
            learning_rate,
        )
        if best_outcome is None or outcome.value < best_outcome.value:
            best_outcome = outcome

    counts = sample_counts(bundle, compiled, best_outcome.params, sampler_shots)
    return assemble_solver_result(
        problem,
        f"vqe_{solver_mode_label(variant, measurement_mode)}",
        counts,
        sampler_shots,
        compiled,
        best_outcome.params,
        best_outcome.history,
        best_outcome.value,
        best_outcome.iterations,
        {
            "policy": dict(policy),
            "offset": float(offset),
            "variant": variant,
            "measurement_mode": measurement_mode,
            "cvar_alpha": alpha if measurement_mode == "cvar" else None,
        },
    )


def build_qiskit_optimizer(name: str, maxiter: int, learning_rate: float):
    """Construct the optimizer objects expected by qiskit-optimization VQE."""

    normalized = name.replace("-", "_").upper()
    if normalized == "COBYLA":
        return QiskitCOBYLA(maxiter=int(maxiter))
    if normalized == "SPSA":
        return SPSA(maxiter=int(maxiter))
    if normalized == "ADAM":
        return ADAM(maxiter=int(maxiter), lr=float(learning_rate))
    raise ValueError(f"Unsupported optimizer method: {name}")


def counts_from_solution_samples(samples, shots: int) -> dict[str, int]:
    """Convert Qiskit optimization SolutionSample objects into count dictionaries."""

    if not samples:
        return {}

    counts: dict[str, int] = {}
    for sample in samples:
        probability = float(getattr(sample, "probability", 0.0))
        count = max(0, int(round(probability * shots)))
        bitstring = "".join(str(int(bit)) for bit in np.asarray(sample.x, dtype=int)[::-1])
        counts[bitstring] = counts.get(bitstring, 0) + count

    total = sum(counts.values())
    if total == 0:
        first = "".join(str(int(bit)) for bit in np.asarray(samples[0].x, dtype=int)[::-1])
        counts[first] = shots
        return counts

    if total != shots:
        best_key = max(counts, key=counts.get)
        counts[best_key] += shots - total
    return counts


def rescale_counts_to_shots(counts: dict[str, int], shots: int) -> dict[str, int]:
    """Rescale a histogram to match a target shot budget."""

    target = int(shots)
    if target <= 0 or not counts:
        return {}

    total = sum(max(0, int(count)) for count in counts.values())
    if total <= 0:
        return {}

    scaled: dict[str, int] = {}
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    assigned = 0
    for index, (bitstring, count) in enumerate(items):
        if index == len(items) - 1:
            scaled_count = target - assigned
        else:
            scaled_count = int(round(target * max(0, int(count)) / total))
            assigned += scaled_count
        if scaled_count > 0:
            scaled[bitstring] = scaled_count

    diff = target - sum(scaled.values())
    if diff != 0 and scaled:
        best_key = max(scaled, key=scaled.get)
        scaled[best_key] += diff
    return {bitstring: count for bitstring, count in scaled.items() if count > 0}


def solve_qrao(problem: ProblemInstance, policy: dict, backend) -> SolverResult:
    """Solve a QUBO problem with QRAO."""
    sampler_shots = get_sampler_shots(policy, 1000)
    bundle = ensure_backend_bundle(backend, get_estimator_shots(policy, 1000), sampler_shots)
    _, measurement_mode, alpha = normalize_measurement_policy(policy)

    qrac_type = int(policy.get("qrao_max_vars_per_qubit", policy.get("qrac_type", 3)))
    if qrac_type not in {1, 2, 3}:
        raise ValueError("QRAO only supports qrao_max_vars_per_qubit 1, 2, or 3.")

    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=qrac_type)
    encoding.encode(problem.qubo)

    ansatz_type = str(policy.get("ansatz_type", policy.get("ansatz", "real_amplitudes")))
    entanglement = str(policy.get("entanglement", "linear"))
    reps = int(policy.get("vqe_reps", policy.get("reps", 2)))
    rotation_blocks = policy.get("rotation_blocks")
    ansatz = build_vqe_ansatz(encoding.num_qubits, ansatz_type, entanglement, reps, rotation_blocks)
    compiled_ansatz = transpile_circuit(bundle, ansatz)

    learning_rate = float(policy.get("learning_rate", 0.05))
    optimizer = build_qiskit_optimizer(get_optimizer_method(policy), get_optimizer_maxiter(policy), learning_rate)
    min_eigen_solver = OptimizationVQE(
        ansatz=compiled_ansatz,
        optimizer=optimizer,
        estimator=bundle.estimator,
        pass_manager=bundle.pass_manager,
    )

    rounding = str(policy.get("rounding", "semideterministic")).lower()
    if rounding == "magic":
        adapter = SamplerAdapter(bundle.sampler, sampler_shots)
        rounding_scheme = MagicRounding(sampler=adapter, pass_manager=bundle.pass_manager)
    else:
        rounding_scheme = SemideterministicRounding()

    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=min_eigen_solver,
        max_vars_per_qubit=qrac_type,
        rounding_scheme=rounding_scheme,
    )
    result = qrao.solve(problem.qubo)

    counts = counts_from_solution_samples(getattr(result, "samples", []), sampler_shots)
    final_cost = (
        float(compute_cvar_from_counts(problem, counts, alpha))
        if measurement_mode == "cvar"
        else float(result.fval)
    )
    metadata = {
        "policy": dict(policy),
        "compression_ratio": float(encoding.compression_ratio),
        "encoded_qubits": int(encoding.num_qubits),
        "original_variables": int(encoding.num_vars),
        "measurement_mode": measurement_mode,
        "cvar_alpha": alpha if measurement_mode == "cvar" else None,
    }

    return assemble_solver_result(
        problem,
        f"qrao_{solver_mode_label(f'{qrac_type}_{rounding}', measurement_mode)}",
        counts,
        sampler_shots,
        compiled_ansatz,
        None,
        [],
        final_cost,
        0,
        metadata,
    )


def pce_find_n(m: int, k: int) -> int:
    """Find the smallest number of qubits supporting m k-body correlators."""

    n = 1
    while 3 * math.comb(n, k) < m:
        n += 1
    return n


def pce_generate_pauli_strings(n: int, m: int, k: int) -> list[str]:
    """Generate the PCE correlators used to encode MaxCut variables."""

    if k > n:
        raise ValueError("k cannot be greater than n.")
    max_strings = 3 * math.comb(n, k)
    if m > max_strings:
        raise ValueError(f"Cannot encode {m} variables with n={n}, k={k}. Maximum is {max_strings}.")

    pauli_strings: list[str] = []
    for pauli in ("X", "Y", "Z"):
        for positions in combinations(range(n), k):
            string = ["I"] * n
            for index in positions:
                string[index] = pauli
            pauli_strings.append("".join(string))
    return pauli_strings[:m]


def pce_regularization_scale(problem: ProblemInstance) -> float:
    """Return the regularization scale nu from the PCE paper."""

    graph = problem.metadata.get("graph")
    if graph is None:
        return 0.0

    weighted = any("weight" in data for _, _, data in graph.edges(data=True))
    if weighted:
        total_weight = sum(data.get("weight", 1.0) for _, _, data in graph.edges(data=True))
        mst = nx.minimum_spanning_tree(graph, weight="weight")
        mst_weight = sum(data.get("weight", 1.0) for _, _, data in mst.edges(data=True))
        return total_weight / 2.0 + mst_weight / 4.0

    return graph.number_of_edges() / 2.0 + (graph.number_of_nodes() - 1) / 4.0


def build_brickwork_ansatz(depth: int, num_qubits: int) -> QuantumCircuit:
    """Build the BrickWork ansatz used by the working PCE implementation."""

    circuit = QuantumCircuit(num_qubits)
    phi = [Parameter(f"phi_{index}") for index in range(2 * depth * num_qubits)]
    single_params = np.asarray(phi[: depth * num_qubits], dtype=object).reshape(depth, num_qubits)
    entangler_params = np.asarray(phi[depth * num_qubits :], dtype=object).reshape(depth, num_qubits)

    for layer in range(depth):
        for qubit in range(num_qubits):
            parameter = single_params[layer][qubit]
            if layer % 3 == 1:
                circuit.ry(parameter, qubit)
            elif layer % 3 == 2:
                circuit.rx(parameter, qubit)
            else:
                circuit.rz(parameter, qubit)

        for qubit in range(0, num_qubits - 1, 2):
            circuit.append(RXXGate(entangler_params[layer][qubit]), [qubit, qubit + 1])

        for qubit in range(num_qubits):
            parameter = single_params[layer][qubit]
            if layer % 3 == 1:
                circuit.rz(parameter, qubit)
            elif layer % 3 == 2:
                circuit.ry(parameter, qubit)
            else:
                circuit.rx(parameter, qubit)

        for qubit in range(1, num_qubits - 1, 2):
            circuit.append(RXXGate(entangler_params[layer][qubit]), [qubit, qubit + 1])

    return circuit


def solve_pce(problem: ProblemInstance, policy: dict, backend) -> SolverResult:
    """Solve MaxCut with Pauli Correlation Encoding."""

    require_maxcut(problem)
    sampler_shots = get_sampler_shots(policy, 1000)
    bundle = ensure_backend_bundle(backend, get_estimator_shots(policy, 1000), sampler_shots)
    rng = rng_from_policy(policy)
    _, measurement_mode, cvar_alpha = normalize_measurement_policy(policy)

    k = int(policy.get("pce_k", policy.get("k", 2)))
    if k not in {2, 3}:
        raise ValueError("PCE is restricted to k=2 or k=3 for this project.")

    num_variables = problem.num_variables
    num_qubits = pce_find_n(num_variables, k)
    pauli_strings = pce_generate_pauli_strings(num_qubits, num_variables, k)
    observables = [SparsePauliOp.from_list([(string, 1.0)]) for string in pauli_strings]

    depth = int(policy.get("pce_depth", policy.get("depth", 10)))
    ansatz_type = str(policy.get("ansatz_type", "brickwork"))
    entanglement = str(policy.get("entanglement", "linear"))
    circuit = build_vqe_ansatz(num_qubits, ansatz_type, entanglement, depth, None)
    compiled = transpile_circuit(bundle, circuit)
    aligned_observables = [align_observable(compiled, observable) for observable in observables]

    alpha_value = policy.get("pce_alpha")
    alpha = float(1.5 * num_qubits if alpha_value is None else alpha_value)
    beta = float(policy.get("pce_beta", 0.5))
    nu = pce_regularization_scale(problem)
    optimizer_method = get_optimizer_method(policy, default="COBYLA")
    optimizer_tol = get_optimizer_tol(policy)
    optimizer_maxiter = get_optimizer_maxiter(policy, default=100)
    learning_rate = float(policy.get("learning_rate", 0.05))

    graph = problem.metadata["graph"]

    def pce_loss(params: np.ndarray) -> float:
        result = bundle.estimator.run([(compiled, aligned_observables, params)]).result()[0]
        expectations = np.asarray(result.data.evs, dtype=float)
        transformed = np.tanh(alpha * expectations)

        loss = 0.0
        for left, right, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
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
    if measurement_mode == "cvar":
        sample_budget = max(
            1,
            int(policy.get("cvar_decode_samples", min(256, sampler_shots))),
        )
        bit_probs = np.clip((expectations + 1.0) / 2.0, 0.0, 1.0)
        raw_counts: dict[str, int] = {}
        for _ in range(sample_budget):
            sampled_bits = (rng.random(num_variables) < bit_probs).astype(float)
            bitstring = "".join(str(int(bit)) for bit in sampled_bits[::-1])
            raw_counts[bitstring] = raw_counts.get(bitstring, 0) + 1
        counts = rescale_counts_to_shots(raw_counts, sampler_shots)
        final_cost = float(compute_cvar_from_counts(problem, counts, cvar_alpha))
    else:
        best_bits = (expectations >= 0.0).astype(float)
        bitstring = "".join(str(int(bit)) for bit in best_bits[::-1])
        counts = {bitstring: sampler_shots}
        final_cost = float(outcome.value)

    metadata = {
        "policy": dict(policy),
        "encoded_qubits": int(num_qubits),
        "original_variables": int(num_variables),
        "compression_ratio": float(num_qubits / num_variables),
        "ansatz_type": ansatz_type,
        "pce_alpha": alpha,
        "pce_beta": beta,
        "pce_nu": nu,
        "measurement_mode": measurement_mode,
        "cvar_alpha": cvar_alpha if measurement_mode == "cvar" else None,
        "cvar_decode_samples": (
            int(policy.get("cvar_decode_samples", min(256, sampler_shots)))
            if measurement_mode == "cvar"
            else None
        ),
    }

    return assemble_solver_result(
        problem,
        f"pce_{solver_mode_label(f'k{k}', measurement_mode)}",
        counts,
        sampler_shots,
        compiled,
        outcome.params,
        outcome.history,
        final_cost,
        outcome.iterations,
        metadata,
    )
