#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
from docplex.mp.model import Model


@dataclass(frozen=True)
class VariantSpec:
    filename: str
    source_template: str
    target_mis: int
    add_edges: tuple[tuple[int, int], ...] = ()
    remove_edges: tuple[tuple[int, int], ...] = ()
    induced_prefix_nodes: int | None = None


CONFIGS = (
    VariantSpec("p1tc.16.txt", "1tc.16.txt", 8, add_edges=((1, 3),)),
    VariantSpec("p2tc.16.txt", "1tc.16.txt", 8, remove_edges=((2, 3),)),
    VariantSpec("p3tc.16.txt", "1tc.16.txt", 7, add_edges=((1, 2),)),
    VariantSpec("p4tc.16.txt", "1tc.16.txt", 6, add_edges=((1, 2), (15, 16))),
    VariantSpec("p1tc.32.txt", "1tc.32.txt", 13, remove_edges=((2, 3),)),
    VariantSpec("p2tc.32.txt", "1tc.32.txt", 12, add_edges=((1, 2),)),
    VariantSpec("p3tc.32.txt", "1tc.32.txt", 11, add_edges=((1, 2), (1, 3))),
    VariantSpec(
        "p4tc.32.txt",
        "1tc.32.txt",
        10,
        add_edges=((1, 2), (1, 3), (30, 32), (31, 32)),
    ),
    VariantSpec("p5tc.32.txt", "1tc.32.txt", 14, remove_edges=((2, 3), (4, 7))),
    VariantSpec("p6tc.32.txt", "1tc.32.txt", 13, remove_edges=((30, 31),)),
    VariantSpec("p7tc.32.txt", "1tc.32.txt", 11, add_edges=((30, 32), (31, 32))),
    VariantSpec(
        "p8tc.32.txt",
        "1tc.32.txt",
        10,
        add_edges=((1, 2), (3, 4), (17, 18), (30, 32), (31, 32)),
    ),
    VariantSpec(
        "p1tc.48.txt",
        "1tc.64.txt",
        15,
        add_edges=((15, 16),),
        induced_prefix_nodes=48,
    ),
    VariantSpec(
        "p1et.48.txt",
        "1et.64.txt",
        14,
        remove_edges=((4, 7),),
        induced_prefix_nodes=48,
    ),
    VariantSpec(
        "p1dc.48.txt",
        "1dc.64.txt",
        8,
        remove_edges=((1, 2),),
        induced_prefix_nodes=48,
    ),
    VariantSpec("p1tc.64.txt", "1tc.64.txt", 20, add_edges=((1, 2),)),
    VariantSpec("p2tc.64.txt", "1tc.64.txt", 19, add_edges=((13, 15), (34, 36))),
    VariantSpec("p1et.64.txt", "1et.64.txt", 18, add_edges=((1, 2),)),
    VariantSpec(
        "p2et.64.txt",
        "1et.64.txt",
        17,
        add_edges=((29, 32), (48, 50), (56, 57)),
    ),
    VariantSpec("p1dc.64.txt", "1dc.64.txt", 10, remove_edges=((1, 2),)),
    VariantSpec("p2dc.64.txt", "1dc.64.txt", 11, remove_edges=((1, 2), (2, 4))),
)


def load_dimacs_graph(path: Path) -> nx.Graph:
    graph = nx.Graph()
    num_nodes: int | None = None
    with path.open("r") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if not parts:
                continue
            if parts[0] == "p":
                num_nodes = int(parts[2])
                graph.add_nodes_from(range(1, num_nodes + 1))
            elif parts[0] == "e":
                u, v = sorted((int(parts[1]), int(parts[2])))
                graph.add_edge(u, v)
    if num_nodes is None:
        raise ValueError(f"Missing DIMACS problem line in {path}")
    return graph


def induced_prefix_subgraph(graph: nx.Graph, num_nodes: int) -> nx.Graph:
    nodes = list(range(1, num_nodes + 1))
    return graph.subgraph(nodes).copy()


def exact_mis_size(graph: nx.Graph) -> int:
    num_nodes = graph.number_of_nodes()
    if num_nodes <= 20:
        best = 0
        for mask in range(1 << num_nodes):
            chosen = [((mask >> offset) & 1) for offset in range(num_nodes)]
            feasible = True
            for u, v in graph.edges():
                if chosen[u - 1] and chosen[v - 1]:
                    feasible = False
                    break
            if feasible:
                best = max(best, sum(chosen))
        return best

    model = Model(name="MIS")
    model.context.solver.log_output = False
    variables = {node: model.binary_var(name=f"x_{node}") for node in graph.nodes()}
    model.maximize(model.sum(variables.values()))
    for u, v in graph.edges():
        model.add_constraint(variables[u] + variables[v] <= 1)
    solution = model.solve(log_output=False)
    if solution is None:
        raise RuntimeError("Exact MIS validation failed: no CPLEX solution returned.")
    return int(round(solution.objective_value))


def edge_span_stats(graph: nx.Graph) -> tuple[float, int, int]:
    spans = sorted(abs(v - u) for u, v in graph.edges())
    if not spans:
        return 0.0, 0, 0
    avg_span = sum(spans) / len(spans)
    p90_index = max(int(0.9 * len(spans)) - 1, 0)
    return avg_span, spans[p90_index], spans[-1]


def format_edge_comment(edges: tuple[tuple[int, int], ...]) -> str:
    if not edges:
        return "none"
    return " ".join(f"{u}-{v}" for u, v in edges)


def build_variant(base_dir: Path, spec: VariantSpec) -> tuple[nx.Graph, int]:
    source_path = base_dir / spec.source_template
    base_graph = load_dimacs_graph(source_path)
    if spec.induced_prefix_nodes is not None:
        graph = induced_prefix_subgraph(base_graph, spec.induced_prefix_nodes)
    else:
        graph = base_graph.copy()

    for edge in spec.remove_edges:
        u, v = sorted(edge)
        if not graph.has_edge(u, v):
            raise ValueError(f"{spec.filename}: cannot remove missing edge {(u, v)}")
        graph.remove_edge(u, v)

    for edge in spec.add_edges:
        u, v = sorted(edge)
        if graph.has_edge(u, v):
            raise ValueError(f"{spec.filename}: cannot add existing edge {(u, v)}")
        graph.add_edge(u, v)

    actual_mis = exact_mis_size(graph)
    if actual_mis != spec.target_mis:
        raise ValueError(
            f"{spec.filename}: expected MIS {spec.target_mis}, validated {actual_mis}"
        )
    return graph, actual_mis


def write_dimacs(path: Path, graph: nx.Graph, spec: VariantSpec, actual_mis: int) -> None:
    avg_span, p90_span, max_span = edge_span_stats(graph)
    construction = "template_local_edits"
    if spec.induced_prefix_nodes is not None:
        construction = "template_prefix_subgraph_local_edits"

    lines = [
        f"c planted_mis_size {actual_mis}",
        f"c construction {construction}",
        f"c source_template {spec.source_template}",
    ]
    if spec.induced_prefix_nodes is not None:
        lines.append(f"c induced_prefix_nodes {spec.induced_prefix_nodes}")
    lines.extend(
        [
            f"c add_edges {format_edge_comment(spec.add_edges)}",
            f"c remove_edges {format_edge_comment(spec.remove_edges)}",
            f"c avg_edge_span {avg_span:.2f}",
            f"c p90_edge_span {p90_span}",
            f"c max_edge_span {max_span}",
            f"p edge {graph.number_of_nodes()} {graph.number_of_edges()}",
        ]
    )
    for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges()):
        lines.append(f"e {u} {v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for spec in CONFIGS:
        graph, actual_mis = build_variant(out_dir, spec)
        out_path = out_dir / spec.filename
        write_dimacs(out_path, graph, spec, actual_mis)
        avg_span, p90_span, max_span = edge_span_stats(graph)
        print(
            f"Generated {spec.filename}: "
            f"n={graph.number_of_nodes()} "
            f"m={graph.number_of_edges()} "
            f"MIS={actual_mis} "
            f"avg_span={avg_span:.2f} "
            f"p90={p90_span} "
            f"max_span={max_span}"
        )


if __name__ == "__main__":
    main()
