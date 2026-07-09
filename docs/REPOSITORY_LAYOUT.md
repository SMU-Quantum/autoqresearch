# Repository Layout

This document separates stable source entry points from research artifacts. The
layout mirrors the accepted QCE26 paper: MIS and CVRP are separate tracks with
their own logs, checkpoints, plots, and paper-analysis tables.

## Source Code

```text
autoqresearch/
  backends/       Backend factory and runtime adapters
  evaluation/     Shared evaluator helpers
  problems/       Problem definitions and generators, including MIS and CVRP
  solvers/        VQE, QAOA, PCE, QRAO, and MaxCut primitives
  utils/          Metrics and ledger helpers

experiment.py               Active adaptive policy
experiment_baseline.py      Frozen conservative baseline
experiment_handcrafted.py   Handcrafted comparison policy
evaluate_policy.py          Fixed suite evaluator and artifact generator
agent_harness.py            LLM-agent search harness
prepare.py                  Environment validation helper
analysis.py                 Legacy run-analysis helper
study_runner.py             Prompt/study runner
study_analysis.py           Study result analysis
```

## Research Protocol Files

```text
program.md                    Active CVRP agent instructions
mis_results/program_mis.md    Preserved MIS agent instructions
mis_results/agent_journal.md  MIS search journal
cvrp_results/agent_journal.md CVRP search journal
studies/example_manifest.json
studies/prompts/*.md          Prompt ablations
```

## Benchmark Inputs

```text
individual/mis/*.txt          DIMACS-style MIS instances
individual/cvrp/*.vrp         CVRP instances
individual/*.ipynb            Exploratory notebooks
legacy/                       Older examples retained for reference
```

MIS and CVRP suite resolution is implemented in
`autoqresearch/problems/registry.py` and `evaluate_policy.py`.

## Generated Paper Artifacts

These files are tracked because they document the reported QCE26 workflow.

MIS artifacts:

```text
mis_results/experiment_log.jsonl
mis_results/beam_history.jsonl
mis_results/beam_state.json
mis_results/promotion_log.jsonl
mis_results/instance_results.jsonl
mis_results/suite_results.tsv
mis_results/suite_history.jsonl
experiment_diffs/mis_diffs/*.patch
plots/plots_mis/*.png
paper_analysis/*.tsv
```

CVRP artifacts:

```text
cvrp_results/experiment_log.jsonl
cvrp_results/beam_history.jsonl
cvrp_results/beam_state.json
cvrp_results/promotion_log.jsonl
cvrp_results/instance_results.jsonl
cvrp_results/suite_results.tsv
cvrp_results/suite_history.jsonl
cvrp_results/experiment_diffs/*.patch
cvrp_results/policy_checkpoints/
cvrp_results/plots/*.png
cvrp_results/paper_analysis/*.tsv
```

## Hardware Artifacts

```text
hardware_runs/run_autoq_hardware.py
hardware_runs/run_cvrp_e13_hardware.py
hardware_runs/autoq_hardware_backend.py
hardware_runs/static_mis_policies.py
hardware_runs/static_cvrp_policies.py
hardware_runs/ibm_credentials.template.json
hardware_runs/checkpoints_autoq/
hardware_runs/results_hardware/
```

Real IBM credential files are not tracked. The template documents the expected
shape.

## Files That Should Stay Generated

The following should remain untracked unless they are deliberately promoted into
the paper artifact:

```text
__pycache__/
*.pyc
.DS_Store
.venv/
results.tsv
instance_progress.png
hardware_runs/ibm_credentials.json
```

## Reorganization Guidance

Keep MIS artifacts under `mis_results/` and CVRP artifacts under
`cvrp_results/`. If runtime paths change, update `agent_harness.py`,
`evaluate_policy.py`, `program.md`, `mis_results/program_mis.md`, and the docs
in the same change.

