# Repository Layout

This document separates stable source entry points from generated research
artifacts. Paths are intentionally kept close to the QCE26 artifact so logs,
checkpoints, and plots remain reproducible.

## Source Code

```text
autoqresearch/
  backends/       Backend factory and runtime adapters
  evaluation/     Shared evaluator helpers
  problems/       Problem definitions and generators
  solvers/        VQE, QAOA, PCE, QRAO, and MaxCut primitives
  utils/          Metrics and ledger helpers

experiment.py               Current adaptive policy under study
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
program.md                  Agent instructions and reproducibility protocol
agent_journal.md            Human-readable MIS search journal
hardware_run_strategies.md  Hardware-run planning notes
studies/example_manifest.json
studies/prompts/*.md        Prompt ablations
```

## Benchmark Inputs

```text
individual/mis/*.txt        DIMACS-style MIS instances
individual/*.ipynb          Exploratory notebooks
legacy/                     Older examples retained for reference
```

The file-based MIS curriculum is resolved through
`autoqresearch/problems/registry.py`.

## Generated Paper Artifacts

These files are tracked because they document the reported QCE26 workflow:

```text
experiment_log.jsonl        Candidate keep/discard records
beam_history.jsonl          Scout beam admission history
beam_state.json             Last known scout beam state
promotion_log.jsonl         Promoted-candidate confirmation records
instance_results.jsonl      Per-instance run ledger
suite_results.tsv           Suite-level run ledger
experiment_diffs/*.patch    Archived candidate diffs
policy_checkpoints/         Candidate policy snapshots
plots/*.png                 Progress, curriculum, heatmap, and scaling plots
paper_analysis/*.tsv        Paper-facing analysis tables
```

## Hardware Artifacts

```text
hardware_runs/run_autoq_hardware.py
hardware_runs/autoq_hardware_backend.py
hardware_runs/static_mis_policies.py
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
.venv/
results.tsv
instance_progress.png
hardware_runs/ibm_credentials.json
```

## Reorganization Guidance

The top-level JSONL/TSV logs are referenced by the evaluator, harness, README,
and paper text. Keep those paths stable unless you update the code and
documentation in the same change.

