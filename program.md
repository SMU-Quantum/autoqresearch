# AutoQResearch CVRP Program

You are improving a quantum algorithm policy for Capacitated Vehicle Routing Problem (CVRP).

The CVRP workflow uses the Fisher-Jaikumar cluster-first, route-second decomposition:

1. Formulate clustering as a Generalized Assignment Problem (GAP) QUBO.
2. Decode customer clusters from the GAP solution.
3. Formulate each cluster route as a TSP QUBO.
4. Score the final routed CVRP cost against the Gurobi reference optimum.

Both GAP and TSP stages are QUBOs, so both stages can use the same solver families used for MIS: VQE, CVaR VQE, QAOA, QAOA variants, QRAO, and PCE. The route stage also has a `classical` exact TSP option for cheaper comparisons and sanity checks.

**Recent improvement**: For larger instances (>30 QUBO variables), a **hybrid** solver family is now available. It performs a classical greedy GAP assignment, identifies ambiguous customers (those near sector boundaries), and solves only the ambiguous subset quantumly via a reduced QUBO. A **feasibility repair** post-processor is also active: when VQE produces near-feasible bitstrings, they are automatically repaired into feasible solutions by fixing assignment and capacity violations.

## Non-negotiable Rules

1. Do not modify MIS artifacts. MIS results live under `mis_results/`.
2. CVRP agent-loop artifacts must stay under `cvrp_results/`.
3. Edit only `experiment.py` for policy changes and `cvrp_results/agent_journal.md` for the research log.
4. Keep classical local search disabled: `pce_local_search: False` and `final_local_search: False`.
5. Do not tune `seed` for better luck. Keep `seed=17` unless doing a post-lock robustness check.
6. Run the baseline scout before the first policy edit.
7. Run at least 15 agent iterations before declaring a final policy.
8. Do not use held-out benchmark instances for keep/discard search. E-n13 is the held-out final benchmark.

## Required Baseline

The checked-in CVRP baseline is:

```python
{
    "solver_family": "vqe",
    "gap_solver_family": "vqe",
    "route_solver_family": "classical",
    "variant": "standard",
    "measurement_mode": "expectation",
    "ansatz_type": "efficient_su2",
    "vqe_reps": 1,
    "entanglement": "linear",
    "optimizer_method": "COBYLA",
    "optimizer_maxiter": 150,
    "seed": 17,
    "cvrp_seed_method": "depot_farthest",
    "cvrp_gap_penalty_method": "hard_slack",
    "pce_local_search": False,
    "final_local_search": False,
}
```

This means the GAP clustering QUBO is solved by VQE with EfficientSU2 depth 1. Routes default to exact classical TSP so the first search isolates GAP clustering quality. For larger instances (>30 QUBO variables), the policy auto-selects `hybrid` mode which uses classical greedy + quantum refinement on ambiguous customers.

## Objective

The metric is `suite_average_gap`, lower is better.

For CVRP, the gap is computed from full routed CVRP cost:

```text
gap = 1 - gurobi_optimal_cost / candidate_routed_cost
```

An infeasible GAP assignment, infeasible route decode, crash, or timeout is scored as `1.0`.

## Solver Degrees Of Freedom

Use evidence from feasibility rate, routed cost, concentration, wall time, and qubit counts to choose experiments. Do not mechanically grid search.

Common quantum knobs:

```python
{
    "solver_family": "vqe" | "qaoa" | "pce" | "qrao" | "hybrid",
    "gap_solver_family": "vqe" | "qaoa" | "pce" | "qrao" | "hybrid",
    "route_solver_family": "classical" | "vqe" | "qaoa" | "pce" | "qrao",
    "measurement_mode": "expectation" | "cvar",
    "cvar_alpha": 0.25,
    "optimizer_method": "COBYLA" | "SPSA" | "Powell" | "Nelder-Mead",
    "optimizer_maxiter": 150,
    "optimizer_tol": 1e-3,
    "learning_rate": 0.05,
    "estimator_shots": 1024,
    "sampler_shots": 1024,
    "seed": 17,
    "penalty": None,
    "pce_local_search": False,
    "final_local_search": False,
}
```

VQE and CVaR VQE:

```python
{
    "ansatz_type": "efficient_su2" | "real_amplitudes" | "pauli_two_design" | "brickwork" | "custom",
    "vqe_reps": 1,
    "entanglement": "linear" | "circular" | "full" | "sca",
    "measurement_mode": "expectation" | "cvar",
    "cvar_alpha": 0.25,
}
```

QAOA:

```python
{
    "variant": "standard" | "warmstart" | "multiangle",
    "reps": 1,
    "measurement_mode": "expectation" | "cvar",
    "cvar_alpha": 0.25,
    "ws_epsilon": 0.25,
    "ws_source": "relaxation",
    "ma_tying": "none" | "partial" | "full",
}
```

QRAO:

```python
{
    "qrao_max_vars_per_qubit": 1 | 2 | 3,
    "rounding": "semideterministic" | "magic",
    "ansatz_type": "efficient_su2" | "real_amplitudes" | "pauli_two_design" | "brickwork",
    "vqe_reps": 1,
    "entanglement": "linear" | "circular" | "full" | "sca",
}
```

PCE:

```python
{
    "pce_k": 2,
    "pce_depth": 10,
    "pce_alpha": None,
    "pce_beta": 0.5,
    "ansatz_type": "efficient_su2" | "real_amplitudes" | "pauli_two_design" | "brickwork",
    "measurement_mode": "expectation" | "cvar",
}
```

Hybrid (classical greedy + quantum sub-problem):

```python
{
    "gap_solver_family": "hybrid",
    "hybrid_sub_family": "vqe",
    "hybrid_ambiguity_threshold": 0.5,
    "variant": "standard",
    "ansatz_type": "efficient_su2",
    "vqe_reps": 1,
    "measurement_mode": "expectation",
    "cvar_alpha": 0.25,
}
```

CVRP-specific knobs:

```python
{
    "cvrp_seed_method": "depot_farthest" | "angle_spread" | "sweep_sector" | "farthest_first" | "largest_demand" | "random",
    "cvrp_gap_penalty_method": "hard_slack" | "taylor" | "tilted",
    "gap_solver_family": "vqe" | "qaoa" | "pce" | "qrao" | "hybrid",
    "route_solver_family": "classical" | "vqe" | "qaoa" | "pce" | "qrao",
    "route_quantum_qubit_threshold": 16,
    "route_quantum_fallback": True,
    "route_tsp_penalty": None,
    "taylor_alpha": 10.0,
    "tilted_kappa": 5.0,
    "tilted_s_frac": 0.10,
    "tilted_s_min": 1.0,
}
```

Route-stage policy keys can override the GAP-stage policy by using the `route_` prefix, for example `route_ansatz_type`, `route_vqe_reps`, `route_reps`, `route_measurement_mode`, `route_cvar_alpha`, `route_pce_k`, `route_pce_depth`, `route_qrao_max_vars_per_qubit`, and `route_rounding`.

## Instances

The CVRP workflow instances are physically stored in `individual/cvrp/`:

### Training / Curriculum Instances

```text
cvrp_8_s0  -> Synth-n9-k2-s0.vrp     (8 customers, 2 vehicles)
cvrp_8_s1  -> Synth-n9-k2-s1.vrp     (8 customers, 2 vehicles)
cvrp_8_s2  -> Synth-n9-k2-s2.vrp     (8 customers, 2 vehicles)
cvrp_9_s0  -> Synth-n10-k3-s0.vrp    (9 customers, 3 vehicles)
cvrp_10_s0 -> Synth-n11-k2-s0.vrp    (10 customers, 2 vehicles)
cvrp_10_s1 -> Synth-n11-k2-s1.vrp    (10 customers, 2 vehicles)
cvrp_12_s0 -> Synth-n13-k3-s0.vrp    (12 customers, 3 vehicles)
```

### Held-Out Test Instances

```text
final_e13  -> E-n13-k4.vrp           (12 customers, 4 vehicles)
```

Final testing uses only `cvrp_benchmark_e13`: the E-n13-k4 benchmark.

The synthetic instances are small enough for quantum simulation but harder than the toy notebook case. They use tight capacities, ambiguous angular sectors, and clustered coordinates.

## Evaluation Workflow

The agent should execute the following stages in order. Do not skip stages.

### Stage 1: Verify 12-Customer Fix

First, confirm the repair and hybrid improvements work on the previously infeasible 12-customer instances:

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_curriculum_12 --workflow split --split train
```

Confirm `suite_average_gap < 1.0` and all instances are feasible. Record the gap and per-instance results.

### Stage 2: Run E-n13 Benchmark

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_benchmark_e13 --workflow final
```

Record the gap, feasibility, qubits used, and wall time.

### Stage 3: Generate Route Comparison Plots

After all evaluations, generate route comparison plots for every instance. Each plot should show side-by-side:
- **Left panel**: The Gurobi/classical optimal routes (reference solution)
- **Right panel**: The quantum/hybrid routes found by the solver

Both panels should display:
- Customer locations as numbered dots, depot as a star
- Vehicle routes as colored lines (one color per vehicle)
- Customer demands as annotations
- Total routed cost in the title
- Vehicle loads and capacities in a legend

Save plots to `cvrp_results/plots/`:

```text
cvrp_results/plots/route_comparison_Synth-n13-k3-s0.png
cvrp_results/plots/route_comparison_E-n13-k4.png
```

### Stage 4: Write Detailed Results Report

Write a results summary to `cvrp_results/benchmark_report.md` containing:

1. **Executive summary**: One paragraph on what worked and what the overall quality is.
2. **Per-instance results table**: Instance name, customers, vehicles, optimal cost, quantum cost, gap, qubits, ambiguous customers, wall time, feasibility.
3. **Scaling analysis**: How gap and qubits scale through the curriculum and E-n13 benchmark.
4. **Route quality discussion**: Notable patterns in which customers are misassigned vs optimal.
5. **Resource table**: Per-stage circuit metrics (qubits, depth, parameters, CNOT count).

## Suites

Fast probe:

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_scout_8 --workflow split --split train --no-artifacts
```

Baseline scout:

```bash
./.venv/bin/python agent_harness.py --single-run --suite cvrp_curriculum_8 --eval-workflow scout --no-dev --branch codex/cvrp01
```

Stage 1 scout and promotion:

```bash
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_8 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev --branch codex/cvrp01
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_8 --promote-beam --promote-top-k 3 --restore-best --branch codex/cvrp01
```

Stage 2 scout and promotion:

```bash
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_10 --eval-workflow scout --wall-clock-budget 2400 --beam-width 5 --no-dev --branch codex/cvrp01
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_10 --promote-beam --promote-top-k 3 --restore-best --branch codex/cvrp01
```

Stage 3 single 12-customer run:

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_curriculum_12 --workflow split --split train
```

Held-out final (E-n13):

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_benchmark_e13 --workflow final
```

## Plotting And Artifacts

CVRP reuses the MIS suite plotting logic: scout trajectories, promotion comparisons, curriculum overview, progress2 view, instance heatmap, family scaling, and Pareto frontier. The 12-customer stage is plotted as the single explicit split run, not as a scout trajectory. For CVRP suites, the paths are rooted at `cvrp_results/`:

```text
cvrp_results/suite_results.tsv
cvrp_results/suite_history.jsonl
cvrp_results/instance_results.jsonl
cvrp_results/plots/
cvrp_results/progress.png
cvrp_results/progress2.png
cvrp_results/experiment_log.jsonl
cvrp_results/experiment_diffs/
cvrp_results/policy_checkpoints/
cvrp_results/beam_state.json
cvrp_results/beam_history.jsonl
cvrp_results/promotion_log.jsonl
cvrp_results/benchmark_report.md
```

Route comparison plots go in `cvrp_results/plots/route_comparison_*.png`.

## Journal

Maintain `cvrp_results/agent_journal.md`. Before every edit, record:

1. Hypothesis
2. Failure signal
3. Degrees of freedom exercised
4. Change summary

After every evaluation, record:

1. Result: KEEP or DISCARD
2. `suite_average_gap` and per-instance gaps
3. Wall time
4. Feasibility notes
5. Next experiment

Start by running the baseline scout and logging it as iteration 0.
