# AutoQResearch

AutoQResearch is an LLM-guided closed-loop experimentation framework for
adaptive variational quantum optimization. It searches for solver-control
policies rather than one static solver configuration: a policy can react to
feasibility, optimality gap, stagnation, sampling concentration, qubit count,
wall time, and instance scale.

## Publication Status

This repository accompanies the accepted QCE26 paper:

> **AutoQResearch: LLM-Guided Closed-Loop Policy Search for Adaptive
> Variational Quantum Optimization**  
> Monit Sharma and Hoong Chuin Lau  
> QCE26 Technical Paper 238, Quantum-GenAI Co-Design & Discovery (QGDD)
> Technical Papers

The repository contains the framework, benchmark instances, evaluation
protocols, search logs, policy checkpoints, plots, and analysis tables for the
Maximum Independent Set (MIS) and decomposed Capacitated Vehicle Routing Problem
(CVRP) studies described in the paper.

## Repository Map

```text
autoqresearch/        Core package: problems, solvers, backends, metrics
experiment.py         Active adaptive policy surface
evaluate_policy.py    Fixed suite evaluator and artifact generator
agent_harness.py      Scout/keep/revert/promotion harness
program.md            Active CVRP agent instructions
mis_results/          MIS journal, logs, ledgers, and preserved program
cvrp_results/         CVRP journal, logs, ledgers, checkpoints, and plots
individual/mis/       MIS benchmark instances
individual/cvrp/      CVRP benchmark instances
experiment_diffs/     Archived MIS diffs under `mis_diffs/`
plots/plots_mis/      Preserved MIS plot outputs
paper_analysis/       Preserved MIS paper-analysis tables
hardware_runs/        IBM Runtime runners and retained-policy artifacts
studies/              Prompt-ablation manifests and prompts
docs/                 Repository layout and paper notes
```

See [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) for a more detailed
source/artifact map.

## Core Policy Interface

The LLM search is constrained to four policy functions in `experiment.py`:

1. `choose_solver_family(problem)`
2. `build_base_policy(problem, family)`
3. `should_continue(attempt, history, problem, max_attempts)`
4. `adapt_policy(attempt, history, problem, base_policy)`

Those functions define a controller of the form:

```text
state_t -> action_t
```

Actions can change solver family, ansatz, optimizer, CVaR mode, depth/reps,
shots, compression strategy, rounding strategy, route-stage choices, repair
logic, and stopping behavior.

## Solver Space

The framework includes the solver families used in the accepted work:

- VQE and CVaR VQE
- QAOA, warm-start QAOA, and multi-angle QAOA
- PCE through a weighted MaxCut reduction
- QRAO with qubit compression and rounding
- CVRP hybrid decomposition policies for larger GAP QUBOs

The package also retains problem utilities for MaxCut, MIS, MDKP, knapsack, and
CVRP-style QUBO experiments.

## MIS Track

MIS artifacts are preserved under `mis_results/`, with MIS plots under
`plots/plots_mis/`. The preserved MIS agent program is
`mis_results/program_mis.md`.

Representative MIS commands:

```bash
./.venv/bin/python evaluate_policy.py --suite mis_probe_16 --workflow split --split train --no-artifacts
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --promote-beam --promote-top-k 3 --restore-best
./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --no-artifacts
```

## CVRP Track

CVRP is implemented as a Fisher-Jaikumar cluster-first, route-second workflow:

1. Build a Generalized Assignment Problem (GAP) QUBO for customer-to-vehicle
   clustering.
2. Decode customer clusters from the GAP solution.
3. Build one route-second TSP QUBO per decoded cluster.
4. Score the routed CVRP solution against the reference optimum.

The route stage can use the quantum solver families or
`route_solver_family="classical"` for exact classical TSP routing after quantum
GAP clustering. CVRP-specific policy knobs include `gap_solver_family`,
`route_solver_family`, `route_quantum_qubit_threshold`, `route_quantum_fallback`,
`route_tsp_penalty`, `cvrp_seed_method`, and `cvrp_gap_penalty_method`.

CVRP instances live under `individual/cvrp/`:

```text
cvrp_8_s0  -> Synth-n9-k2-s0.vrp
cvrp_8_s1  -> Synth-n9-k2-s1.vrp
cvrp_8_s2  -> Synth-n9-k2-s2.vrp
cvrp_9_s0  -> Synth-n10-k3-s0.vrp
cvrp_10_s0 -> Synth-n11-k2-s0.vrp
cvrp_10_s1 -> Synth-n11-k2-s1.vrp
cvrp_12_s0 -> Synth-n13-k3-s0.vrp
final       -> E-n13-k4.vrp
```

Representative CVRP commands:

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_scout_8 --workflow split --split train --no-artifacts
./.venv/bin/python agent_harness.py --single-run --suite cvrp_curriculum_8 --eval-workflow scout --no-dev
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_8 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_8 --promote-beam --promote-top-k 3 --restore-best
./.venv/bin/python evaluate_policy.py --suite cvrp_benchmark_e13 --workflow final
```

CVRP outputs are routed under `cvrp_results/`, including `suite_results.tsv`,
`instance_results.jsonl`, `experiment_log.jsonl`, `promotion_log.jsonl`,
`policy_checkpoints/`, `paper_analysis/`, and `plots/`.

## Evaluation Methodology

The accepted paper emphasizes staged confirmation:

- **Scout:** cheap proxy evaluation under a fixed workflow
- **Promote:** rerun top beam candidates on the full stage suite
- **Confirm:** select the confirmed winner while replaying earlier-stage
  guardrails
- **Final:** evaluate the locked policy on held-out instances

The primary metric is `suite_average_gap`:

- `0.0` means optimal/reference-matching on every evaluated instance
- `1.0` means failure, timeout, crash, infeasibility, or trivial output
- Lower is better

Resource usage, wall time, feasibility, and concentration are recorded for
analysis, but keep/revert decisions are driven by the fixed metric and guardrail
rules.

## Quick Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Validate the Python stack:

```bash
./.venv/bin/python prepare.py --validate-only
```

## Hardware Runs

Hardware execution support lives under `hardware_runs/`.

Inspect the retained MIS plan without touching IBM Runtime:

```bash
./.venv/bin/python hardware_runs/run_autoq_hardware.py --instance 1tc.32 --plan-only
```

Run or inspect the CVRP E-n13 hardware workflow:

```bash
./.venv/bin/python hardware_runs/run_cvrp_e13_hardware.py --help
```

Use `hardware_runs/ibm_credentials.template.json` as the credential template for
IBM Runtime execution.

## Citation

See [CITATION.cff](CITATION.cff) and [docs/PAPER.md](docs/PAPER.md) for the
current accepted-paper citation note.
