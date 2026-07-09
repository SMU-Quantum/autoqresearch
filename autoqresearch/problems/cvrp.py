"""CVRP instances and Fisher-Jaikumar GAP/TSP QUBO builders."""

from __future__ import annotations

import math
import re
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from .base import ProblemGenerator, ProblemInstance


CVRP_INSTANCE_DIR = Path(__file__).resolve().parent.parent.parent / "individual" / "cvrp"


def _capacity(instance: dict[str, Any]) -> int:
    return int(float(instance["metadata"]["CAPACITY"]))


def _name(instance: dict[str, Any]) -> str:
    path = instance.get("path")
    fallback = path.stem if isinstance(path, Path) else "cvrp"
    return str(instance.get("metadata", {}).get("NAME", fallback))


def _build_explicit_distance_matrix(
    metadata: dict[str, Any],
    edge_weights: list[int],
) -> list[list[int]]:
    dimension = int(metadata["DIMENSION"])
    edge_format = str(metadata.get("EDGE_WEIGHT_FORMAT", "")).strip()
    matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]

    if edge_format == "LOWER_ROW":
        expected = dimension * (dimension - 1) // 2
        if len(edge_weights) != expected:
            raise ValueError(f"LOWER_ROW expected {expected} weights, found {len(edge_weights)}")
        index = 0
        for row in range(1, dimension):
            for col in range(row):
                weight = int(edge_weights[index])
                index += 1
                matrix[row][col] = weight
                matrix[col][row] = weight
        return matrix

    if edge_format == "FULL_MATRIX":
        expected = dimension * dimension
        if len(edge_weights) != expected:
            raise ValueError(f"FULL_MATRIX expected {expected} weights, found {len(edge_weights)}")
        for row in range(dimension):
            for col in range(dimension):
                matrix[row][col] = int(edge_weights[row * dimension + col])
        return matrix

    raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_format}")


def read_cvrp_instance(file_path: str | Path) -> dict[str, Any]:
    """Read a CVRPLIB-style ``.vrp`` file."""

    path = Path(file_path)
    metadata: dict[str, Any] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depots: list[int] = []
    edge_weights: list[int] = []
    section = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "EOF":
            break
        if line in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}:
            section = line
            continue

        if section == "NODE_COORD_SECTION":
            node, x_coord, y_coord = line.split()[:3]
            coords[int(node)] = (float(x_coord), float(y_coord))
            continue

        if section == "EDGE_WEIGHT_SECTION":
            edge_weights.extend(int(value) for value in line.split())
            continue

        if section == "DEMAND_SECTION":
            node, demand = line.split()[:2]
            demands[int(node)] = int(float(demand))
            continue

        if section == "DEPOT_SECTION":
            node = int(line.split()[0])
            if node != -1:
                depots.append(node)
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')

    if not depots:
        raise ValueError(f"No depot found in {path}")

    dimension = int(metadata["DIMENSION"])
    distance_matrix = None
    if edge_weights:
        distance_matrix = _build_explicit_distance_matrix(metadata, edge_weights)
        nodes = list(range(1, dimension + 1))
    else:
        nodes = sorted(coords)
        if len(nodes) != dimension:
            raise ValueError(f"Expected {dimension} coordinate nodes, found {len(nodes)}")

    if set(demands) != set(nodes):
        raise ValueError("Demand section and node set have different node ids.")

    depot = depots[0]
    return {
        "path": path,
        "metadata": metadata,
        "coords": coords,
        "demands": demands,
        "depots": depots,
        "depot": depot,
        "nodes": nodes,
        "customers": [node for node in nodes if node != depot],
        "distance_matrix": distance_matrix,
    }


def parse_vehicle_count(instance: dict[str, Any], default: int = 1) -> int:
    match = re.search(r"-k(\d+)", _name(instance))
    return int(match.group(1)) if match else int(default)


def cvrp_distance(instance: dict[str, Any], left: int, right: int) -> int:
    if instance.get("distance_matrix") is not None:
        return int(instance["distance_matrix"][left - 1][right - 1])

    ax, ay = instance["coords"][left]
    bx, by = instance["coords"][right]
    return int(round(math.hypot(ax - bx, ay - by)))


def route_cost(instance: dict[str, Any], route: list[int] | tuple[int, ...]) -> float:
    if not route:
        return 0.0
    depot = int(instance["depot"])
    cost = float(cvrp_distance(instance, depot, int(route[0])))
    cost += sum(float(cvrp_distance(instance, int(left), int(right))) for left, right in zip(route, route[1:]))
    cost += float(cvrp_distance(instance, int(route[-1]), depot))
    return cost


def default_vehicle_count_for_size(num_customers: int) -> int:
    if num_customers in {9, 12}:
        return 3
    return 2 if num_customers <= 10 else 3


def generate_synthetic_cvrp_instance(
    num_customers: int,
    seed: int,
    num_vehicles: int | None = None,
    capacity_tightness: float = 0.86,
) -> dict[str, Any]:
    """Generate a small Euclidean CVRP with tight, ambiguous assignments."""

    if num_customers < 4:
        raise ValueError("CVRP synthetic instances need at least 4 customers.")
    vehicles = int(num_vehicles or default_vehicle_count_for_size(num_customers))
    rng = np.random.default_rng(int(seed))

    depot = 1
    nodes = list(range(1, num_customers + 2))
    customers = nodes[1:]
    coords: dict[int, tuple[float, float]] = {depot: (50.0, 50.0)}

    # Place customers around angular sector boundaries with small jitter. This
    # makes seed-based GAP assignment nontrivial without requiring large n.
    base_rotation = rng.uniform(0.0, 2.0 * math.pi / max(vehicles, 1))
    for idx, customer in enumerate(customers):
        boundary_shift = 0.45 * (2.0 * math.pi / vehicles)
        angle = (
            base_rotation
            + idx * 2.0 * math.pi / num_customers
            + ((-1) ** idx) * boundary_shift / max(num_customers, 1)
            + rng.normal(0.0, 0.07)
        )
        radius = 18.0 + 8.0 * (idx % 3) + rng.normal(0.0, 1.5)
        coords[customer] = (
            float(50.0 + radius * math.cos(angle)),
            float(50.0 + radius * math.sin(angle)),
        )

    raw_demands = rng.integers(2, 7, size=num_customers).astype(int)
    if num_customers >= 6:
        raw_demands[1::4] += 2
        raw_demands[2::5] += 1
    total_demand = int(raw_demands.sum())
    capacity = int(math.ceil(total_demand / (vehicles * capacity_tightness)))
    capacity = max(capacity, int(raw_demands.max()))

    demands = {depot: 0}
    for customer, demand in zip(customers, raw_demands):
        demands[customer] = int(demand)

    metadata = {
        "NAME": f"Synth-n{num_customers + 1}-k{vehicles}-s{seed}",
        "COMMENT": "Synthetic small hard CVRP instance for quantum simulation",
        "TYPE": "CVRP",
        "DIMENSION": str(num_customers + 1),
        "EDGE_WEIGHT_TYPE": "EUC_2D",
        "CAPACITY": str(capacity),
    }
    return {
        "path": Path(f"synthetic/Synth-n{num_customers + 1}-k{vehicles}-s{seed}.vrp"),
        "metadata": metadata,
        "coords": coords,
        "demands": demands,
        "depots": [depot],
        "depot": depot,
        "nodes": nodes,
        "customers": customers,
        "distance_matrix": None,
    }


def _customer_radius(instance: dict[str, Any], customer: int) -> float:
    return float(cvrp_distance(instance, int(instance["depot"]), int(customer)))


def _customer_angle(instance: dict[str, Any], customer: int) -> float:
    depot = int(instance["depot"])
    dx = instance["coords"][customer][0] - instance["coords"][depot][0]
    dy = instance["coords"][customer][1] - instance["coords"][depot][1]
    return math.atan2(dy, dx) % (2.0 * math.pi)


def _angular_distance(left: float, right: float) -> float:
    diff = abs((left - right) % (2.0 * math.pi))
    return min(diff, 2.0 * math.pi - diff)


def choose_farthest_first_seeds(instance: dict[str, Any], num_seeds: int) -> list[int]:
    customers = list(instance["customers"])
    depot = int(instance["depot"])
    if num_seeds > len(customers):
        raise ValueError("Cannot choose more seeds than customers.")

    seeds = [max(customers, key=lambda customer: (cvrp_distance(instance, depot, customer), -customer))]
    while len(seeds) < num_seeds:
        remaining = [customer for customer in customers if customer not in seeds]
        next_seed = max(
            remaining,
            key=lambda customer: (
                min(cvrp_distance(instance, customer, seed) for seed in seeds),
                cvrp_distance(instance, depot, customer),
                -customer,
            ),
        )
        seeds.append(int(next_seed))
    return seeds


def choose_depot_farthest_seeds(instance: dict[str, Any], num_seeds: int) -> list[int]:
    return [
        int(customer)
        for customer in sorted(
            instance["customers"],
            key=lambda customer: (_customer_radius(instance, customer), -customer),
            reverse=True,
        )[:num_seeds]
    ]


def choose_largest_demand_seeds(instance: dict[str, Any], num_seeds: int) -> list[int]:
    return [
        int(customer)
        for customer in sorted(
            instance["customers"],
            key=lambda customer: (
                instance["demands"][customer],
                _customer_radius(instance, customer),
                -customer,
            ),
            reverse=True,
        )[:num_seeds]
    ]


def choose_angle_spread_seeds(instance: dict[str, Any], num_seeds: int) -> list[int]:
    if not instance.get("coords"):
        raise ValueError("angle_spread requires NODE_COORD_SECTION coordinates.")

    customers = list(instance["customers"])
    anchor = _customer_angle(
        instance,
        max(customers, key=lambda customer: (_customer_radius(instance, customer), -customer)),
    )
    targets = [(anchor + step * 2.0 * math.pi / num_seeds) % (2.0 * math.pi) for step in range(num_seeds)]
    seeds: list[int] = []
    for target in targets:
        remaining = [customer for customer in customers if customer not in seeds]
        seed = min(
            remaining,
            key=lambda customer: (
                _angular_distance(_customer_angle(instance, customer), target),
                -_customer_radius(instance, customer),
                customer,
            ),
        )
        seeds.append(int(seed))
    return seeds


def choose_sweep_sector_seeds(instance: dict[str, Any], num_seeds: int) -> list[int]:
    if not instance.get("coords"):
        raise ValueError("sweep_sector requires NODE_COORD_SECTION coordinates.")

    customers = list(instance["customers"])
    anchor = _customer_angle(
        instance,
        max(customers, key=lambda customer: (_customer_radius(instance, customer), -customer)),
    )
    sector_width = 2.0 * math.pi / num_seeds
    seeds: list[int] = []
    for sector in range(num_seeds):
        start = (anchor + sector * sector_width) % (2.0 * math.pi)
        sector_customers = []
        for customer in customers:
            shifted_angle = (_customer_angle(instance, customer) - start) % (2.0 * math.pi)
            if shifted_angle < sector_width:
                sector_customers.append(customer)
        remaining = [customer for customer in sector_customers if customer not in seeds]
        if remaining:
            seeds.append(int(max(remaining, key=lambda customer: (_customer_radius(instance, customer), -customer))))

    while len(seeds) < num_seeds:
        remaining = [customer for customer in customers if customer not in seeds]
        seeds.append(int(max(remaining, key=lambda customer: (_customer_radius(instance, customer), -customer))))
    return seeds[:num_seeds]


def choose_random_seeds(instance: dict[str, Any], num_seeds: int, random_state: int = 7) -> list[int]:
    rng = np.random.default_rng(int(random_state))
    return [int(value) for value in rng.choice(instance["customers"], size=num_seeds, replace=False)]


SEED_METHODS = {
    "angle_spread": choose_angle_spread_seeds,
    "sweep_sector": choose_sweep_sector_seeds,
    "farthest_first": choose_farthest_first_seeds,
    "depot_farthest": choose_depot_farthest_seeds,
    "largest_demand": choose_largest_demand_seeds,
    "random": choose_random_seeds,
}


def choose_seed_customers(
    instance: dict[str, Any],
    num_seeds: int,
    method: str = "angle_spread",
    random_state: int = 7,
) -> tuple[list[int], str]:
    if method not in SEED_METHODS:
        raise ValueError(f"Unknown seed method {method!r}. Choose from {sorted(SEED_METHODS)}")
    try:
        if method == "random":
            return SEED_METHODS[method](instance, num_seeds, random_state=random_state), method
        return SEED_METHODS[method](instance, num_seeds), method
    except ValueError:
        if method in {"angle_spread", "sweep_sector"} and not instance.get("coords"):
            fallback = "farthest_first"
            return SEED_METHODS[fallback](instance, num_seeds), fallback
        raise


def assignment_variable_name(customer: int, vehicle: int) -> str:
    return f"z_{customer}_{vehicle}"


def fisher_jaikumar_assignment_cost(
    instance: dict[str, Any],
    customer: int,
    seed: int,
) -> float:
    depot = int(instance["depot"])
    return float(
        cvrp_distance(instance, depot, customer)
        + cvrp_distance(instance, customer, seed)
        - cvrp_distance(instance, depot, seed)
    )


def build_fisher_jaikumar_gap_qp(
    instance: dict[str, Any],
    seeds: list[int],
) -> QuadraticProgram:
    customers = list(instance["customers"])
    num_vehicles = len(seeds)
    capacity = _capacity(instance)
    qp = QuadraticProgram("cvrp_fisher_jaikumar_gap")

    for customer in customers:
        for vehicle in range(num_vehicles):
            qp.binary_var(name=assignment_variable_name(customer, vehicle))

    linear = {
        assignment_variable_name(customer, vehicle): fisher_jaikumar_assignment_cost(instance, customer, seed)
        for customer in customers
        for vehicle, seed in enumerate(seeds)
    }
    qp.minimize(linear=linear)

    for customer in customers:
        qp.linear_constraint(
            linear={assignment_variable_name(customer, vehicle): 1 for vehicle in range(num_vehicles)},
            sense="==",
            rhs=1,
            name=f"assign_customer_{customer}",
        )

    for vehicle in range(num_vehicles):
        qp.linear_constraint(
            linear={
                assignment_variable_name(customer, vehicle): float(instance["demands"][customer])
                for customer in customers
            },
            sense="<=",
            rhs=capacity,
            name=f"capacity_vehicle_{vehicle}",
        )

    for vehicle, seed in enumerate(seeds):
        qp.linear_constraint(
            linear={assignment_variable_name(seed, vehicle): 1},
            sense="==",
            rhs=1,
            name=f"fix_seed_{seed}_vehicle_{vehicle}",
        )
    return qp


def _gap_base_objective(
    instance: dict[str, Any],
    seeds: list[int],
) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    linear = {}
    for customer in instance["customers"]:
        for vehicle, seed in enumerate(seeds):
            linear[assignment_variable_name(customer, vehicle)] = fisher_jaikumar_assignment_cost(
                instance,
                customer,
                seed,
            )
    return 0.0, linear, {}


def _add_assignment_and_seed_constraints(
    qp: QuadraticProgram,
    instance: dict[str, Any],
    seeds: list[int],
) -> None:
    num_vehicles = len(seeds)
    for customer in instance["customers"]:
        qp.linear_constraint(
            linear={assignment_variable_name(customer, vehicle): 1 for vehicle in range(num_vehicles)},
            sense="==",
            rhs=1,
            name=f"assign_customer_{customer}",
        )

    for vehicle, seed in enumerate(seeds):
        qp.linear_constraint(
            linear={assignment_variable_name(seed, vehicle): 1},
            sense="==",
            rhs=1,
            name=f"fix_seed_{seed}_vehicle_{vehicle}",
        )


def _add_taylor_capacity_penalty(
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
    instance: dict[str, Any],
    vehicle: int,
    alpha: float,
) -> float:
    capacity = _capacity(instance)
    constant = alpha * (1.0 - capacity + 0.5 * capacity * capacity)

    for customer in instance["customers"]:
        variable = assignment_variable_name(customer, vehicle)
        demand = float(instance["demands"][customer])
        linear[variable] = linear.get(variable, 0.0) + alpha * (
            (1.0 - capacity) * demand + 0.5 * demand * demand
        )

    for left_index, left in enumerate(instance["customers"]):
        for right in instance["customers"][left_index + 1:]:
            left_var = assignment_variable_name(left, vehicle)
            right_var = assignment_variable_name(right, vehicle)
            value = alpha * float(instance["demands"][left]) * float(instance["demands"][right])
            quadratic[(left_var, right_var)] = quadratic.get((left_var, right_var), 0.0) + value
    return float(constant)


def _tilted_s(capacity: int, s_frac: float = 0.10, s_min: float = 1.0) -> float:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return float(max(s_min, int(s_frac * capacity)))


def _tilted_rho(
    instance: dict[str, Any],
    seeds: list[int],
    kappa: float = 5.0,
    eps: float = 1e-12,
) -> float:
    min_demand = min(float(instance["demands"][customer]) for customer in instance["customers"])
    if min_demand <= 0:
        raise ValueError("Customer demands must be positive for rho scaling")

    cost_scale = max(
        abs(float(fisher_jaikumar_assignment_cost(instance, customer, seed)))
        for customer in instance["customers"]
        for seed in seeds
    )
    return float((kappa * max(cost_scale, 1.0)) / (min_demand * min_demand + eps))


def _add_tilted_capacity_penalty(
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
    instance: dict[str, Any],
    vehicle: int,
    rho: float,
    s: float,
) -> float:
    capacity = _capacity(instance)
    constant = rho * (capacity * capacity - s * capacity)
    linear_factor = rho * (s - 2.0 * capacity)

    for customer in instance["customers"]:
        variable = assignment_variable_name(customer, vehicle)
        demand = float(instance["demands"][customer])
        linear[variable] = linear.get(variable, 0.0) + linear_factor * demand + rho * demand * demand

    for left_index, left in enumerate(instance["customers"]):
        for right in instance["customers"][left_index + 1:]:
            left_var = assignment_variable_name(left, vehicle)
            right_var = assignment_variable_name(right, vehicle)
            value = 2.0 * rho * float(instance["demands"][left]) * float(instance["demands"][right])
            quadratic[(left_var, right_var)] = quadratic.get((left_var, right_var), 0.0) + value
    return float(constant)


def build_fisher_jaikumar_gap_taylor_penalty_qp(
    instance: dict[str, Any],
    seeds: list[int],
    alpha: float = 10.0,
) -> QuadraticProgram:
    qp = QuadraticProgram("cvrp_fisher_jaikumar_gap_taylor_penalty")
    for customer in instance["customers"]:
        for vehicle in range(len(seeds)):
            qp.binary_var(name=assignment_variable_name(customer, vehicle))

    constant, linear, quadratic = _gap_base_objective(instance, seeds)
    for vehicle in range(len(seeds)):
        constant += _add_taylor_capacity_penalty(linear, quadratic, instance, vehicle, alpha=alpha)

    qp.minimize(constant=constant, linear=linear, quadratic=quadratic)
    _add_assignment_and_seed_constraints(qp, instance, seeds)
    return qp


def build_fisher_jaikumar_gap_tilted_penalty_qp(
    instance: dict[str, Any],
    seeds: list[int],
    kappa: float = 5.0,
    s_frac: float = 0.10,
    s_min: float = 1.0,
    rho: float | None = None,
    s: float | None = None,
) -> tuple[QuadraticProgram, float, float]:
    qp = QuadraticProgram("cvrp_fisher_jaikumar_gap_tilted_penalty")
    for customer in instance["customers"]:
        for vehicle in range(len(seeds)):
            qp.binary_var(name=assignment_variable_name(customer, vehicle))

    capacity = _capacity(instance)
    s_used = float(s) if s is not None else _tilted_s(capacity, s_frac=s_frac, s_min=s_min)
    rho_used = float(rho) if rho is not None else _tilted_rho(instance, seeds, kappa=kappa)

    constant, linear, quadratic = _gap_base_objective(instance, seeds)
    for vehicle in range(len(seeds)):
        constant += _add_tilted_capacity_penalty(
            linear,
            quadratic,
            instance,
            vehicle,
            rho=rho_used,
            s=s_used,
        )

    qp.minimize(constant=constant, linear=linear, quadratic=quadratic)
    _add_assignment_and_seed_constraints(qp, instance, seeds)
    return qp, rho_used, s_used


def build_fisher_jaikumar_gap_qubo_model(
    instance: dict[str, Any],
    seeds: list[int],
    capacity_method: str = "tilted",
    gap_penalty: float | None = None,
    taylor_alpha: float = 10.0,
    tilted_kappa: float = 5.0,
    tilted_s_frac: float = 0.10,
    tilted_s_min: float = 1.0,
    tilted_rho: float | None = None,
    tilted_s: float | None = None,
) -> dict[str, Any]:
    method = capacity_method.lower()
    penalty_parameters: dict[str, float | None] = {}

    if method == "hard_slack":
        qp = build_fisher_jaikumar_gap_qp(instance, seeds)
    elif method == "taylor":
        qp = build_fisher_jaikumar_gap_taylor_penalty_qp(instance, seeds, alpha=taylor_alpha)
        penalty_parameters["taylor_alpha"] = float(taylor_alpha)
    elif method == "tilted":
        qp, rho_used, s_used = build_fisher_jaikumar_gap_tilted_penalty_qp(
            instance,
            seeds,
            kappa=tilted_kappa,
            s_frac=tilted_s_frac,
            s_min=tilted_s_min,
            rho=tilted_rho,
            s=tilted_s,
        )
        penalty_parameters.update({"tilted_rho": rho_used, "tilted_s": s_used, "tilted_kappa": tilted_kappa})
    else:
        raise ValueError("capacity_method must be one of hard_slack, taylor, tilted")

    converter = QuadraticProgramToQubo(penalty=gap_penalty)
    qubo = converter.convert(qp)
    return {
        "method": method,
        "qp": qp,
        "qubo": qubo,
        "converter": converter,
        "converter_penalty": converter.penalty,
        "penalty_parameters": penalty_parameters,
    }


def decode_gap_original_assignment(
    original_gap_vector: np.ndarray,
    gap_qp: QuadraticProgram,
    customers: list[int],
    num_vehicles: int,
) -> list[list[int]]:
    values_by_name = {
        variable.name: int(round(float(original_gap_vector[index])))
        for index, variable in enumerate(gap_qp.variables)
    }
    clusters = [[] for _ in range(num_vehicles)]
    for customer in customers:
        assigned = [
            vehicle
            for vehicle in range(num_vehicles)
            if values_by_name.get(assignment_variable_name(customer, vehicle), 0) == 1
        ]
        if len(assigned) != 1:
            raise ValueError(f"Customer {customer} assigned to {assigned}; expected exactly one vehicle.")
        clusters[assigned[0]].append(int(customer))
    return clusters


def decode_gap_raw_assignment(
    original_gap_vector: np.ndarray,
    gap_qp: QuadraticProgram,
    customers: list[int],
    num_vehicles: int,
) -> dict[int, list[int]]:
    """Decode a GAP vector into a raw customer→vehicles mapping (may be invalid)."""
    values_by_name = {
        variable.name: int(round(float(original_gap_vector[index])))
        for index, variable in enumerate(gap_qp.variables)
    }
    assignment: dict[int, list[int]] = {}
    for customer in customers:
        assigned = [
            vehicle
            for vehicle in range(num_vehicles)
            if values_by_name.get(assignment_variable_name(customer, vehicle), 0) == 1
        ]
        assignment[customer] = assigned
    return assignment


def repair_gap_assignment(
    instance: dict[str, Any],
    seeds: list[int],
    raw_assignment: dict[int, list[int]],
) -> list[list[int]] | None:
    """Repair a possibly-invalid GAP assignment into feasible clusters.

    Fixes:
    1. Multi-assigned customers: keep cheapest assignment.
    2. Unassigned customers: assign to cheapest feasible vehicle.
    3. Capacity violations: iteratively move excess customers to feasible vehicles.

    Returns ``None`` if repair fails (no feasible capacity resolution).
    """
    num_vehicles = len(seeds)
    capacity = _capacity(instance)
    customers = list(instance["customers"])

    # Step 1: build initial assignment — resolve multi/un-assigned customers
    customer_vehicle: dict[int, int] = {}

    # Seeds are always fixed to their own vehicle
    for vehicle, seed in enumerate(seeds):
        customer_vehicle[seed] = vehicle

    # Multi-assigned: keep cheapest by Fisher-Jaikumar cost
    for customer in customers:
        if customer in customer_vehicle:
            continue
        assigned = raw_assignment.get(customer, [])
        if len(assigned) == 1:
            customer_vehicle[customer] = assigned[0]
        elif len(assigned) > 1:
            customer_vehicle[customer] = min(
                assigned,
                key=lambda v: fisher_jaikumar_assignment_cost(instance, customer, seeds[v]),
            )
        # unassigned handled below

    # Unassigned: assign to cheapest vehicle with remaining capacity
    unassigned = [c for c in customers if c not in customer_vehicle]
    for customer in sorted(unassigned, key=lambda c: instance["demands"][c], reverse=True):
        loads = [0] * num_vehicles
        for c, v in customer_vehicle.items():
            loads[v] += instance["demands"][c]
        feasible = [v for v in range(num_vehicles) if loads[v] + instance["demands"][customer] <= capacity]
        candidates = feasible if feasible else list(range(num_vehicles))
        customer_vehicle[customer] = min(
            candidates,
            key=lambda v: fisher_jaikumar_assignment_cost(instance, customer, seeds[v]),
        )

    # Step 2: fix capacity violations by moving excess customers
    for _ in range(len(customers) * num_vehicles):
        clusters = [[] for _ in range(num_vehicles)]
        for c in customers:
            clusters[customer_vehicle[c]].append(c)
        loads = cluster_loads(instance, clusters)
        overloaded = [v for v in range(num_vehicles) if loads[v] > capacity]
        if not overloaded:
            return clusters

        vehicle = overloaded[0]
        movable = [c for c in clusters[vehicle] if c not in seeds]
        if not movable:
            return None

        # Try direct move first
        best_move = None
        best_cost = float("inf")
        for customer in movable:
            for target_v in range(num_vehicles):
                if target_v == vehicle:
                    continue
                target_load = loads[target_v] + instance["demands"][customer]
                if target_load > capacity:
                    continue
                cost = fisher_jaikumar_assignment_cost(instance, customer, seeds[target_v])
                if cost < best_cost:
                    best_cost = cost
                    best_move = ("move", customer, target_v)

        # If no direct move works, try swap: exchange a high-demand customer
        # from the overloaded vehicle with a lower-demand one from another
        if best_move is None:
            best_swap_delta = float("-inf")
            for cust_out in movable:
                d_out = instance["demands"][cust_out]
                for target_v in range(num_vehicles):
                    if target_v == vehicle:
                        continue
                    target_movable = [c for c in clusters[target_v] if c not in seeds]
                    for cust_in in target_movable:
                        d_in = instance["demands"][cust_in]
                        if d_in >= d_out:
                            continue  # swap must reduce overload
                        new_src_load = loads[vehicle] - d_out + d_in
                        new_tgt_load = loads[target_v] - d_in + d_out
                        if new_src_load <= capacity and new_tgt_load <= capacity:
                            delta = (d_out - d_in)  # bigger delta = more relief
                            if delta > best_swap_delta:
                                best_swap_delta = delta
                                best_move = ("swap", cust_out, target_v, cust_in, vehicle)

        if best_move is None:
            # Last resort: force-move highest-demand to least loaded
            customer = max(movable, key=lambda c: instance["demands"][c])
            target_v = min(
                (v for v in range(num_vehicles) if v != vehicle),
                key=lambda v: loads[v],
            )
            customer_vehicle[customer] = target_v
        elif best_move[0] == "move":
            customer_vehicle[best_move[1]] = best_move[2]
        elif best_move[0] == "swap":
            _, cust_out, target_v, cust_in, src_v = best_move
            customer_vehicle[cust_out] = target_v
            customer_vehicle[cust_in] = src_v

    return None


def repair_cvrp_from_qubo(
    qubo_x: np.ndarray,
    problem: "ProblemInstance",
) -> tuple[list[list[int]] | None, float]:
    """Attempt to repair a QUBO bitstring into a feasible CVRP clustering.

    Returns (clusters, routed_cost) or (None, inf) if repair fails.
    """
    try:
        converter = problem.metadata["converter"]
        original_qp = problem.metadata["original_qp"]
        original_x = np.asarray(converter.interpret(np.asarray(qubo_x, dtype=float)), dtype=float)
        customers = list(problem.metadata["customers"])
        num_vehicles = int(problem.metadata["num_vehicles"])
        instance = problem.metadata["instance"]
        seeds = list(problem.metadata["seeds"])

        raw = decode_gap_raw_assignment(original_x, original_qp, customers, num_vehicles)
        clusters = repair_gap_assignment(instance, seeds, raw)
        if clusters is None:
            return None, float("inf")
        if not clusters_capacity_feasible(instance, clusters):
            return None, float("inf")
        route_solutions = solve_cvrp_routes_classically(instance, clusters)
        cost = float(sum(s["cost"] for s in route_solutions))
        return clusters, cost
    except Exception:
        return None, float("inf")


def clusters_to_gap_qubo_bitstring(
    problem: "ProblemInstance",
    clusters: list[list[int]],
) -> np.ndarray:
    """Encode repaired/full CVRP clusters as a GAP QUBO-space bitstring."""
    customers = list(problem.metadata["customers"])
    num_vehicles = int(problem.metadata["num_vehicles"])
    assignment: dict[int, int] = {}

    for vehicle, cluster in enumerate(clusters):
        if vehicle >= num_vehicles:
            raise ValueError(f"Cluster index {vehicle} exceeds vehicle count {num_vehicles}.")
        for customer in cluster:
            customer = int(customer)
            if customer in assignment:
                raise ValueError(f"Customer {customer} appears in multiple clusters.")
            assignment[customer] = vehicle

    missing = [customer for customer in customers if customer not in assignment]
    if missing:
        raise ValueError(f"Missing customers in repaired clusters: {missing}.")

    selected_variables = {
        assignment_variable_name(customer, assignment[customer])
        for customer in customers
    }
    x = np.zeros(problem.num_variables, dtype=float)
    for index, variable in enumerate(problem.qubo.variables):
        if variable.name in selected_variables:
            x[index] = 1.0
    return x


def decode_gap_qubo_solution(qubo_x: np.ndarray, problem: ProblemInstance) -> list[list[int]]:
    converter = problem.metadata["converter"]
    original_qp = problem.metadata["original_qp"]
    original_x = np.asarray(converter.interpret(np.asarray(qubo_x, dtype=float)), dtype=float)
    return decode_gap_original_assignment(
        original_x,
        original_qp,
        list(problem.metadata["customers"]),
        int(problem.metadata["num_vehicles"]),
    )


def cluster_loads(instance: dict[str, Any], clusters: list[list[int]]) -> list[int]:
    return [int(sum(instance["demands"][customer] for customer in cluster)) for cluster in clusters]


def clusters_capacity_feasible(instance: dict[str, Any], clusters: list[list[int]]) -> bool:
    capacity = _capacity(instance)
    return all(load <= capacity for load in cluster_loads(instance, clusters))


def route_stage_qubit_counts(clusters: list[list[int]]) -> dict[str, Any]:
    per_route = [len(cluster) ** 2 for cluster in clusters]
    return {
        "per_route": per_route,
        "sequential_qubits": max(per_route) if per_route else 0,
        "combined_qubits": sum(per_route),
    }


def build_route_second_tsp_qp(
    instance: dict[str, Any],
    cluster: list[int],
    name: str,
) -> QuadraticProgram:
    depot = int(instance["depot"])
    n = len(cluster)
    if n == 0:
        raise ValueError("Cannot build a TSP QP for an empty cluster.")

    qp = QuadraticProgram(name=name)
    for customer in cluster:
        for position in range(n):
            qp.binary_var(name=f"y_{customer}_{position}")

    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}

    for customer in cluster:
        linear[f"y_{customer}_0"] = linear.get(f"y_{customer}_0", 0.0) + cvrp_distance(
            instance,
            depot,
            customer,
        )
        linear[f"y_{customer}_{n - 1}"] = linear.get(f"y_{customer}_{n - 1}", 0.0) + cvrp_distance(
            instance,
            customer,
            depot,
        )

    for position in range(n - 1):
        for left in cluster:
            for right in cluster:
                if left == right:
                    continue
                key = (f"y_{left}_{position}", f"y_{right}_{position + 1}")
                quadratic[key] = quadratic.get(key, 0.0) + cvrp_distance(instance, left, right)

    qp.minimize(linear=linear, quadratic=quadratic)

    for customer in cluster:
        qp.linear_constraint(
            linear={f"y_{customer}_{position}": 1 for position in range(n)},
            sense="==",
            rhs=1,
            name=f"visit_customer_{customer}",
        )

    for position in range(n):
        qp.linear_constraint(
            linear={f"y_{customer}_{position}": 1 for customer in cluster},
            sense="==",
            rhs=1,
            name=f"one_customer_at_position_{position}",
        )

    return qp


def solve_route_tsp_classically(instance: dict[str, Any], cluster: list[int]) -> dict[str, Any]:
    if not cluster:
        return {"solver": "classical_exact", "route": [], "cost": 0.0}

    best_route: list[int] | None = None
    best_cost = math.inf
    for candidate in permutations(cluster):
        candidate_route = [int(node) for node in candidate]
        candidate_cost = route_cost(instance, candidate_route)
        if candidate_cost < best_cost:
            best_route = candidate_route
            best_cost = candidate_cost
    return {"solver": "classical_exact", "route": best_route or [], "cost": float(best_cost)}


def decode_tsp_original_assignment(
    original_tsp_vector: np.ndarray,
    tsp_qp: QuadraticProgram,
    cluster: list[int],
) -> list[int]:
    values_by_name = {
        variable.name: int(round(float(original_tsp_vector[index])))
        for index, variable in enumerate(tsp_qp.variables)
    }
    route: list[int] = []
    for position in range(len(cluster)):
        assigned = [
            customer
            for customer in cluster
            if values_by_name.get(f"y_{customer}_{position}", 0) == 1
        ]
        if len(assigned) != 1:
            raise ValueError(f"Position {position} assigned to {assigned}; expected exactly one customer.")
        route.append(int(assigned[0]))
    if sorted(route) != sorted(cluster):
        raise ValueError(f"Decoded route {route} does not visit exactly cluster {cluster}.")
    return route


def decode_tsp_qubo_solution(qubo_x: np.ndarray, problem: ProblemInstance) -> list[int]:
    converter = problem.metadata["converter"]
    original_qp = problem.metadata["original_qp"]
    original_x = np.asarray(converter.interpret(np.asarray(qubo_x, dtype=float)), dtype=float)
    return decode_tsp_original_assignment(
        original_x,
        original_qp,
        list(problem.metadata["cluster"]),
    )


def build_route_tsp_problem(
    instance: dict[str, Any],
    cluster: list[int],
    route_index: int,
    tsp_penalty: float | None = None,
) -> ProblemInstance:
    qp = build_route_second_tsp_qp(instance, cluster, name=f"cvrp_route_{route_index}_tsp")
    converter = QuadraticProgramToQubo(penalty=tsp_penalty)
    qubo = converter.convert(qp)
    classical = solve_route_tsp_classically(instance, cluster)
    return ProblemInstance(
        name=f"{_name(instance)}_route_{route_index}_tsp",
        problem_type="cvrp_tsp",
        num_variables=qubo.get_num_vars(),
        qubo=qubo,
        optimal_value=float(classical["cost"]),
        optimal_solution=None,
        metadata={
            "instance": instance,
            "cluster": list(cluster),
            "route_index": int(route_index),
            "original_qp": qp,
            "converter": converter,
            "converter_penalty": converter.penalty,
            "classical_solution": classical,
        },
    )


def cvrp_gap_feasible(qubo_x: np.ndarray, problem: ProblemInstance) -> bool:
    try:
        converter = problem.metadata["converter"]
        original_qp = problem.metadata["original_qp"]
        original_x = np.asarray(converter.interpret(np.asarray(qubo_x, dtype=float)), dtype=float)
        if not original_qp.is_feasible(original_x):
            return False
        clusters = decode_gap_original_assignment(
            original_x,
            original_qp,
            list(problem.metadata["customers"]),
            int(problem.metadata["num_vehicles"]),
        )
        return clusters_capacity_feasible(problem.metadata["instance"], clusters)
    except Exception:
        return False


def cvrp_routed_cost(qubo_x: np.ndarray, problem: ProblemInstance) -> float:
    if not cvrp_gap_feasible(qubo_x, problem):
        return float("inf")
    clusters = decode_gap_qubo_solution(qubo_x, problem)
    key = tuple(tuple(cluster) for cluster in clusters)
    cache = problem.metadata.setdefault("_route_cost_cache", {})
    if key not in cache:
        route_solutions = [
            solve_route_tsp_classically(problem.metadata["instance"], list(cluster))
            for cluster in clusters
        ]
        cache[key] = float(sum(solution["cost"] for solution in route_solutions))
    return float(cache[key])


def solve_cvrp_routes_classically(
    instance: dict[str, Any],
    clusters: list[list[int]],
) -> list[dict[str, Any]]:
    records = []
    for route_index, cluster in enumerate(clusters):
        solution = solve_route_tsp_classically(instance, list(cluster))
        records.append(
            {
                "route_index": int(route_index),
                "customers": list(cluster),
                "load": int(sum(instance["demands"][customer] for customer in cluster)),
                **solution,
            }
        )
    return records


def solve_cvrp_with_gurobi(
    instance: dict[str, Any],
    num_vehicles: int,
    output: bool = False,
    max_dfj_customers: int = 14,
) -> dict[str, Any] | None:
    """Solve CVRP exactly with a compact DFJ model for small instances."""

    if len(instance["customers"]) > max_dfj_customers:
        return None

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    nodes = list(instance["nodes"])
    customers = list(instance["customers"])
    depot = int(instance["depot"])
    capacity = _capacity(instance)
    vehicles = range(int(num_vehicles))
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    model = gp.Model(f"{_name(instance)}_gurobi_cvrp")
    model.Params.OutputFlag = 1 if output else 0

    x = model.addVars([(k, i, j) for k in vehicles for i, j in arcs], vtype=GRB.BINARY, name="x")
    y = model.addVars([(k, i) for k in vehicles for i in customers], vtype=GRB.BINARY, name="y")
    route_used = model.addVars(list(vehicles), vtype=GRB.BINARY, name="route_used")

    model.setObjective(
        gp.quicksum(cvrp_distance(instance, i, j) * x[k, i, j] for k in vehicles for i, j in arcs),
        GRB.MINIMIZE,
    )

    for customer in customers:
        model.addConstr(gp.quicksum(y[k, customer] for k in vehicles) == 1, name=f"visit_{customer}")

    for k in vehicles:
        model.addConstr(
            gp.quicksum(instance["demands"][customer] * y[k, customer] for customer in customers) <= capacity,
            name=f"capacity_{k}",
        )
        model.addConstr(
            gp.quicksum(x[k, depot, customer] for customer in customers) == route_used[k],
            name=f"depot_depart_{k}",
        )
        model.addConstr(
            gp.quicksum(x[k, customer, depot] for customer in customers) == route_used[k],
            name=f"depot_return_{k}",
        )
        model.addConstr(
            gp.quicksum(y[k, customer] for customer in customers) <= len(customers) * route_used[k],
            name=f"use_route_if_customers_{k}",
        )

        for customer in customers:
            model.addConstr(
                gp.quicksum(x[k, customer, j] for j in nodes if j != customer) == y[k, customer],
                name=f"outflow_{k}_{customer}",
            )
            model.addConstr(
                gp.quicksum(x[k, i, customer] for i in nodes if i != customer) == y[k, customer],
                name=f"inflow_{k}_{customer}",
            )

        for subset_size in range(2, len(customers) + 1):
            for subset in combinations(customers, subset_size):
                model.addConstr(
                    gp.quicksum(x[k, i, j] for i in subset for j in subset if i != j) <= subset_size - 1,
                    name=f"dfj_{k}_{'_'.join(map(str, subset))}",
                )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None

    routes = []
    for k in vehicles:
        if route_used[k].X < 0.5:
            continue
        route = []
        current = depot
        visited = set()
        while True:
            next_nodes = [j for j in nodes if j != current and x[k, current, j].X > 0.5]
            if not next_nodes:
                raise ValueError(f"Could not continue Gurobi route for vehicle {k} from node {current}.")
            next_node = int(next_nodes[0])
            if next_node == depot:
                break
            if next_node in visited:
                raise ValueError(f"Gurobi route for vehicle {k} revisited customer {next_node}.")
            route.append(next_node)
            visited.add(next_node)
            current = next_node

        routes.append(
            {
                "vehicle": int(k),
                "route": route,
                "load": int(sum(instance["demands"][customer] for customer in route)),
                "cost": route_cost(instance, route),
            }
        )

    return {
        "status": int(model.Status),
        "objective": float(model.ObjVal),
        "routes": routes,
        "solver": "gurobi",
    }


def solve_cvrp_by_exact_enumeration(
    instance: dict[str, Any],
    num_vehicles: int,
    max_customers: int = 10,
) -> dict[str, Any]:
    customers = list(instance["customers"])
    if len(customers) > max_customers:
        raise ValueError(f"Exact CVRP enumeration is capped at {max_customers} customers.")

    capacity = _capacity(instance)
    best_solution: dict[str, Any] | None = None
    for assignment in product(range(int(num_vehicles)), repeat=len(customers)):
        clusters = [[] for _ in range(int(num_vehicles))]
        for customer, vehicle in zip(customers, assignment):
            clusters[int(vehicle)].append(int(customer))
        if not clusters_capacity_feasible(instance, clusters):
            continue
        route_solutions = solve_cvrp_routes_classically(instance, clusters)
        objective = float(sum(solution["cost"] for solution in route_solutions))
        if best_solution is None or objective < float(best_solution["objective"]):
            best_solution = {
                "status": "OPTIMAL_BRUTE_FORCE",
                "objective": objective,
                "solver": "exact_enumeration",
                "routes": [
                    {
                        "vehicle": route["route_index"],
                        "route": route["route"],
                        "load": route["load"],
                        "cost": route["cost"],
                    }
                    for route in route_solutions
                    if route["route"]
                ],
            }

    if best_solution is None:
        raise ValueError("No feasible exact-enumeration CVRP solution found.")
    return best_solution


def solve_cvrp_reference(
    instance: dict[str, Any],
    num_vehicles: int,
    gurobi_output: bool = False,
) -> dict[str, Any]:
    gurobi_solution = solve_cvrp_with_gurobi(instance, num_vehicles, output=gurobi_output)
    if gurobi_solution is not None:
        return gurobi_solution
    try:
        return solve_cvrp_by_exact_enumeration(instance, num_vehicles)
    except ValueError:
        seeds, _ = choose_seed_customers(instance, num_vehicles, method="farthest_first")
        clusters = [[seed] for seed in seeds]
        remaining_capacity = [
            _capacity(instance) - int(instance["demands"][seed])
            for seed in seeds
        ]
        unassigned = [customer for customer in instance["customers"] if customer not in seeds]
        for customer in sorted(
            unassigned,
            key=lambda item: (instance["demands"][item], _customer_radius(instance, item)),
            reverse=True,
        ):
            feasible_vehicles = [
                vehicle
                for vehicle, remaining in enumerate(remaining_capacity)
                if int(instance["demands"][customer]) <= remaining
            ]
            candidates = feasible_vehicles or list(range(int(num_vehicles)))
            vehicle = min(
                candidates,
                key=lambda item: fisher_jaikumar_assignment_cost(instance, customer, seeds[item]),
            )
            clusters[vehicle].append(int(customer))
            remaining_capacity[vehicle] -= int(instance["demands"][customer])
        routes = solve_cvrp_routes_classically(instance, clusters)
        return {
            "status": "HEURISTIC",
            "objective": float(sum(route["cost"] for route in routes)),
            "solver": "seed_heuristic",
            "routes": routes,
        }


def solve_gap_greedy(
    instance: dict[str, Any],
    seeds: list[int],
) -> list[list[int]]:
    """Capacity-respecting greedy GAP assignment using Fisher-Jaikumar costs."""
    num_vehicles = len(seeds)
    capacity = _capacity(instance)
    clusters: list[list[int]] = [[] for _ in range(num_vehicles)]
    remaining_capacity = [capacity] * num_vehicles

    # Fix seeds to their vehicles
    for vehicle, seed in enumerate(seeds):
        clusters[vehicle].append(seed)
        remaining_capacity[vehicle] -= instance["demands"][seed]

    # Assign remaining customers by cheapest feasible Fisher-Jaikumar cost
    unassigned = [c for c in instance["customers"] if c not in seeds]
    for customer in sorted(
        unassigned,
        key=lambda c: (instance["demands"][c], _customer_radius(instance, c)),
        reverse=True,
    ):
        feasible = [
            v for v in range(num_vehicles)
            if remaining_capacity[v] >= instance["demands"][customer]
        ]
        candidates = feasible if feasible else list(range(num_vehicles))
        vehicle = min(
            candidates,
            key=lambda v: fisher_jaikumar_assignment_cost(instance, customer, seeds[v]),
        )
        clusters[vehicle].append(customer)
        remaining_capacity[vehicle] -= instance["demands"][customer]
    return clusters


def identify_ambiguous_customers(
    instance: dict[str, Any],
    seeds: list[int],
    classical_assignment: list[list[int]],
    threshold: float = 0.3,
) -> tuple[list[int], dict[int, int]]:
    """Identify customers whose vehicle assignment is ambiguous.

    A customer is "ambiguous" if its second-best Fisher-Jaikumar cost is within
    ``threshold`` fraction of its best cost.  Returns (ambiguous_customers,
    fixed_assignments) where fixed_assignments maps customer→vehicle for the
    clearly-assigned customers.
    """
    num_vehicles = len(seeds)
    fixed: dict[int, int] = {}
    ambiguous: list[int] = []

    # Seeds are always fixed
    for vehicle, seed in enumerate(seeds):
        fixed[seed] = vehicle

    customer_to_vehicle = {}
    for vehicle, cluster in enumerate(classical_assignment):
        for customer in cluster:
            customer_to_vehicle[customer] = vehicle

    for customer in instance["customers"]:
        if customer in fixed:
            continue
        costs = sorted(
            fisher_jaikumar_assignment_cost(instance, customer, seeds[v])
            for v in range(num_vehicles)
        )
        if len(costs) < 2:
            fixed[customer] = customer_to_vehicle.get(customer, 0)
            continue
        best, second = costs[0], costs[1]
        gap = (second - best) / max(abs(best), 1e-10)
        if gap < threshold:
            ambiguous.append(customer)
        else:
            fixed[customer] = customer_to_vehicle.get(customer, 0)

    return ambiguous, fixed


def build_reduced_gap_qubo(
    instance: dict[str, Any],
    seeds: list[int],
    fixed_assignments: dict[int, int],
    ambiguous_customers: list[int],
    capacity_method: str = "tilted",
    gap_penalty: float | None = None,
    taylor_alpha: float = 10.0,
    tilted_kappa: float = 5.0,
    tilted_s_frac: float = 0.10,
    tilted_s_min: float = 1.0,
) -> dict[str, Any]:
    """Build a reduced GAP QUBO over only the ambiguous customers.

    Fixed customers have their demands subtracted from vehicle capacities.
    """
    num_vehicles = len(seeds)
    # Compute reduced capacity per vehicle
    reduced_capacity = [_capacity(instance)] * num_vehicles
    for customer, vehicle in fixed_assignments.items():
        reduced_capacity[vehicle] -= instance["demands"][customer]

    # Build a modified instance with reduced capacity for the sub-problem
    reduced_instance = dict(instance)
    reduced_instance = {
        **instance,
        "customers": ambiguous_customers,
        "metadata": {
            **instance.get("metadata", {}),
            "CAPACITY": str(max(reduced_capacity)),
        },
    }

    # Build a fresh QP manually for the reduced sub-problem
    qp = QuadraticProgram("cvrp_reduced_gap")
    for customer in ambiguous_customers:
        for vehicle in range(num_vehicles):
            qp.binary_var(name=assignment_variable_name(customer, vehicle))

    # Objective: Fisher-Jaikumar assignment costs
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    constant = 0.0

    for customer in ambiguous_customers:
        for vehicle, seed in enumerate(seeds):
            var = assignment_variable_name(customer, vehicle)
            linear[var] = linear.get(var, 0.0) + fisher_jaikumar_assignment_cost(
                instance, customer, seed
            )

    # Add capacity penalties with reduced capacities
    method = capacity_method.lower()
    if method == "tilted":
        rho = _tilted_rho(instance, seeds, kappa=tilted_kappa)
        for vehicle in range(num_vehicles):
            cap = reduced_capacity[vehicle]
            s_val = _tilted_s(max(1, int(cap)), s_frac=tilted_s_frac, s_min=tilted_s_min)
            constant += rho * (cap * cap - s_val * cap)
            linear_factor = rho * (s_val - 2.0 * cap)
            for customer in ambiguous_customers:
                var = assignment_variable_name(customer, vehicle)
                demand = float(instance["demands"][customer])
                linear[var] = linear.get(var, 0.0) + linear_factor * demand + rho * demand * demand
            for li, left in enumerate(ambiguous_customers):
                for right in ambiguous_customers[li + 1:]:
                    lv = assignment_variable_name(left, vehicle)
                    rv = assignment_variable_name(right, vehicle)
                    val = 2.0 * rho * float(instance["demands"][left]) * float(instance["demands"][right])
                    quadratic[(lv, rv)] = quadratic.get((lv, rv), 0.0) + val
    elif method == "taylor":
        alpha = taylor_alpha
        for vehicle in range(num_vehicles):
            cap = reduced_capacity[vehicle]
            constant += alpha * (1.0 - cap + 0.5 * cap * cap)
            for customer in ambiguous_customers:
                var = assignment_variable_name(customer, vehicle)
                demand = float(instance["demands"][customer])
                linear[var] = linear.get(var, 0.0) + alpha * (
                    (1.0 - cap) * demand + 0.5 * demand * demand
                )
            for li, left in enumerate(ambiguous_customers):
                for right in ambiguous_customers[li + 1:]:
                    lv = assignment_variable_name(left, vehicle)
                    rv = assignment_variable_name(right, vehicle)
                    val = alpha * float(instance["demands"][left]) * float(instance["demands"][right])
                    quadratic[(lv, rv)] = quadratic.get((lv, rv), 0.0) + val

    qp.minimize(constant=constant, linear=linear, quadratic=quadratic)

    # Assignment constraints: each ambiguous customer to exactly one vehicle
    for customer in ambiguous_customers:
        qp.linear_constraint(
            linear={assignment_variable_name(customer, v): 1 for v in range(num_vehicles)},
            sense="==",
            rhs=1,
            name=f"assign_customer_{customer}",
        )

    # Seed constraints: only if seed is in ambiguous set
    for vehicle, seed in enumerate(seeds):
        if seed in ambiguous_customers:
            qp.linear_constraint(
                linear={assignment_variable_name(seed, vehicle): 1},
                sense="==",
                rhs=1,
                name=f"fix_seed_{seed}_vehicle_{vehicle}",
            )

    converter = QuadraticProgramToQubo(penalty=gap_penalty)
    qubo = converter.convert(qp)
    return {
        "method": method,
        "qp": qp,
        "qubo": qubo,
        "converter": converter,
        "converter_penalty": converter.penalty,
        "reduced_capacity": reduced_capacity,
        "fixed_assignments": dict(fixed_assignments),
        "ambiguous_customers": list(ambiguous_customers),
    }


def make_cvrp_problem(
    instance: dict[str, Any],
    num_vehicles: int | None = None,
    seed_method: str = "angle_spread",
    seed_random_state: int = 7,
    capacity_method: str = "tilted",
    gap_penalty: float | None = None,
    taylor_alpha: float = 10.0,
    tilted_kappa: float = 5.0,
    tilted_s_frac: float = 0.10,
    tilted_s_min: float = 1.0,
    tilted_rho: float | None = None,
    tilted_s: float | None = None,
    reference_solution: dict[str, Any] | None = None,
    gurobi_output: bool = False,
) -> ProblemInstance:
    vehicles = int(num_vehicles or parse_vehicle_count(instance, default=default_vehicle_count_for_size(len(instance["customers"]))))
    seeds, seed_method_used = choose_seed_customers(
        instance,
        vehicles,
        method=seed_method,
        random_state=seed_random_state,
    )
    gap_model = build_fisher_jaikumar_gap_qubo_model(
        instance,
        seeds,
        capacity_method=capacity_method,
        gap_penalty=gap_penalty,
        taylor_alpha=taylor_alpha,
        tilted_kappa=tilted_kappa,
        tilted_s_frac=tilted_s_frac,
        tilted_s_min=tilted_s_min,
        tilted_rho=tilted_rho,
        tilted_s=tilted_s,
    )
    reference = reference_solution or solve_cvrp_reference(instance, vehicles, gurobi_output=gurobi_output)
    customers = list(instance["customers"])
    return ProblemInstance(
        name=f"cvrp_{_name(instance)}_{gap_model['method']}_{seed_method_used}",
        problem_type="cvrp",
        num_variables=gap_model["qubo"].get_num_vars(),
        qubo=gap_model["qubo"],
        optimal_value=float(reference["objective"]),
        optimal_solution=None,
        metadata={
            "instance": instance,
            "num_customers": len(customers),
            "num_vehicles": vehicles,
            "customers": customers,
            "capacity": _capacity(instance),
            "total_demand": int(sum(instance["demands"][customer] for customer in customers)),
            "seeds": seeds,
            "seed_method": seed_method_used,
            "requested_seed_method": seed_method,
            "capacity_method": gap_model["method"],
            "original_qp": gap_model["qp"],
            "converter": gap_model["converter"],
            "converter_penalty": gap_model["converter_penalty"],
            "penalty_parameters": gap_model["penalty_parameters"],
            "reference_solution": reference,
            "optimal_reference": reference.get("solver", "unknown"),
            "source_file": str(instance.get("path", "")),
        },
    )


def load_cvrp_from_file(file_path: str | Path, **kwargs: Any) -> ProblemInstance:
    return make_cvrp_problem(read_cvrp_instance(file_path), **kwargs)


class CVRPGenerator(ProblemGenerator):
    """Generate synthetic small CVRP GAP QUBO instances."""

    def __init__(
        self,
        capacity_method: str = "hard_slack",
        seed_method: str = "depot_farthest",
        capacity_tightness: float = 0.86,
    ):
        self.capacity_method = capacity_method
        self.seed_method = seed_method
        self.capacity_tightness = capacity_tightness

    def generate(self, size: int, seed: int) -> ProblemInstance:
        vehicles = default_vehicle_count_for_size(int(size))
        instance_file = CVRP_INSTANCE_DIR / f"Synth-n{int(size) + 1}-k{vehicles}-s{int(seed)}.vrp"
        if instance_file.exists():
            instance = read_cvrp_instance(instance_file)
        else:
            instance = generate_synthetic_cvrp_instance(
                int(size),
                int(seed),
                num_vehicles=vehicles,
                capacity_tightness=self.capacity_tightness,
            )
        return make_cvrp_problem(
            instance,
            num_vehicles=vehicles,
            seed_method=self.seed_method,
            seed_random_state=int(seed) + 7,
            capacity_method=self.capacity_method,
        )


# ─── Route comparison plotting ─────────────────────────────────────


def plot_route_comparison(
    instance: dict[str, Any],
    classical_routes: list[dict[str, Any]],
    quantum_routes: list[dict[str, Any]],
    classical_cost: float,
    quantum_cost: float,
    output_path: str | Path,
    title_suffix: str = "",
) -> None:
    """Plot side-by-side classical vs quantum route comparison.

    Each panel shows customer locations, depot, and vehicle routes as colored
    lines.  Customer demands are annotated.  The title shows the total cost.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return

    ROUTE_COLORS = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    def _get_coords(inst):
        """Get or synthesize 2D coordinates for plotting."""
        if inst.get("coords"):
            return dict(inst["coords"])
        # No coordinates — use MDS from distance matrix
        nodes = list(inst["nodes"])
        n = len(nodes)
        dist_matrix = np.zeros((n, n))
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                dist_matrix[i][j] = cvrp_distance(inst, ni, nj)
        # Classical MDS embedding
        D2 = dist_matrix ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2 @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1][:2]
        coords_arr = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
        return {node: (float(coords_arr[i, 0]), float(coords_arr[i, 1]))
                for i, node in enumerate(nodes)}

    synth_coords = _get_coords(instance)

    def _draw_routes(ax, inst, routes, cost_val, panel_title):
        depot = int(inst["depot"])
        coords = synth_coords
        capacity = _capacity(inst)
        dx, dy = coords[depot]

        # Draw routes
        for ridx, route_info in enumerate(routes):
            route = route_info.get("route", [])
            if not route:
                continue
            color = ROUTE_COLORS[ridx % len(ROUTE_COLORS)]
            load = route_info.get("load", sum(inst["demands"][c] for c in route))
            rcost = route_info.get("cost", route_cost(inst, route))

            # depot -> first customer
            fx, fy = coords[route[0]]
            ax.annotate(
                "", xy=(fx, fy), xytext=(dx, dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
            )
            # customer -> customer
            for i in range(len(route) - 1):
                cx, cy = coords[route[i]]
                nx, ny = coords[route[i + 1]]
                ax.annotate(
                    "", xy=(nx, ny), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                )
            # last customer -> depot
            lx, ly = coords[route[-1]]
            ax.annotate(
                "", xy=(dx, dy), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls="--"),
            )

        # Draw customers
        for customer in inst["customers"]:
            cx, cy = coords[customer]
            ax.plot(cx, cy, "o", color="#333333", markersize=8, zorder=5)
            demand = inst["demands"][customer]
            ax.annotate(
                f"{customer}\n(d={demand})",
                (cx, cy),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=6,
                color="#555555",
            )

        # Draw depot
        ax.plot(dx, dy, "*", color="gold", markersize=18, zorder=6,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate("Depot", (dx, dy), textcoords="offset points",
                    xytext=(8, -12), fontsize=7, fontweight="bold")

        # Legend
        legend_handles = []
        for ridx, route_info in enumerate(routes):
            route = route_info.get("route", [])
            if not route:
                continue
            color = ROUTE_COLORS[ridx % len(ROUTE_COLORS)]
            load = route_info.get("load", sum(inst["demands"][c] for c in route))
            rcost = route_info.get("cost", route_cost(inst, route))
            legend_handles.append(
                Line2D([0], [0], color=color, lw=2,
                       label=f"V{ridx}: load={load}/{capacity}, cost={rcost:.0f}")
            )
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=6, loc="lower left",
                      framealpha=0.8)

        ax.set_title(f"{panel_title}\nTotal cost: {cost_val:.1f}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    name = _name(instance)
    _draw_routes(ax1, instance, classical_routes, classical_cost, "Classical Optimal")
    _draw_routes(ax2, instance, quantum_routes, quantum_cost, "Quantum/Hybrid")
    fig.suptitle(f"Route Comparison: {name}{title_suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_summary(
    results: list[dict[str, Any]],
    output_path: str | Path,
    reference_gap: float | None = None,
) -> None:
    """Bar chart of quantum gap across benchmark instances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [r["name"] for r in results]
    gaps = [r["gap"] for r in results]
    colors = ["#4daf4a" if g < 0.15 else "#ff7f00" if g < 0.3 else "#e41a1c" for g in gaps]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 6))
    bars = ax.bar(range(len(names)), gaps, color=colors, edgecolor="black", linewidth=0.5)

    if reference_gap is not None:
        ax.axhline(y=reference_gap, color="#377eb8", linestyle="--", linewidth=2,
                   label=f"E-n13-k4 gap ({reference_gap:.3f})")
        ax.legend(fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Optimality Gap", fontsize=11)
    ax.set_title("XSH-n20 Benchmark: Quantum/Hybrid Gap per Instance", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, max(gaps) * 1.15 if gaps else 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{gap:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
