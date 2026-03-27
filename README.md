# AutoQResearch

![include](progress2.png)


AutoQResearch is a small AutoResearch-style repository for discovering a transferable, state-dependent solver-control policy for 0-1 knapsack under resource constraints.

The project is not about tuning one static solver configuration. It is about improving a sequential adaptive policy in `experiment.py` and measuring progress with one fixed suite metric:

- Primary metric: `suite_average_gap`
- Direction: lower is better
- Meaning: `0.0` is optimal on every suite instance, `1.0` is total failure/timeout/crash

## Core Structure

```text
Editable policy surface:
  experiment.py

Fixed evaluation + acceptance path:
  evaluate_policy.py
  agent_harness.py
  program.md

Comparative study layer:
  study_runner.py
  study_analysis.py
```

`experiment.py` exposes exactly four policy functions:

1. `choose_solver_family(problem)`
2. `build_base_policy(problem, family)`
3. `should_continue(attempt, history, problem, max_attempts)`
4. `adapt_policy(attempt, history, problem, base_policy)`

Those functions define a sequential controller over one instance:

```text
state_t -> action_t
```

The state can include feasibility, optimality gap, stagnation, wall time, and instance metadata. The action can change solver family, optimizer, CVaR mode, depth/reps, shots, and stopping behavior.

## What Is Fixed

- `evaluate_policy.py` is the suite-level scorer.
- `suite_average_gap` is the only optimization target for keep/revert.
- Resource usage is logged for secondary frontier analysis, but never enters keep/revert.
- Knapsack `optimality_gap` is computed against an exact dynamic-programming classical optimum.
- Candidate evaluation is split-aware:
  - train drives iteration
  - dev is a guardrail against obvious overfitting
  - test is held out for final reporting
- Baseline comparison uses the same execution engine with a frozen conservative policy.

## Why This Is Not Grid Search

A static parameter set cannot react to failed feasibility, low quality after a feasible attempt, stagnation, or instance size. The controller in `experiment.py` is allowed to make different decisions after each attempt based on what it has observed so far.

The scientific object is the adaptive policy, not a single fixed configuration.

The starting checkpoint in `experiment.py` is intentionally plain:

- initial family: conservative `vqe`
- initial policy: `real_amplitudes`, `vqe_reps=1`, `COBYLA`, linear entanglement
- stop rule: budget-only
- adaptation: none

That is deliberate. The main artifact is the discovery trajectory from this
simple baseline to a stronger adaptive controller.

## Evaluation Workflow

Use the candidate workflow during development:

```bash
./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate
```

This runs:

- train split: primary optimization target
- dev split: guardrail

A candidate is accepted only if:

- train `suite_average_gap` improves strictly
- and dev does not regress by more than `0.02`

For final held-out evaluation:

```bash
./.venv/bin/python evaluate_policy.py --suite standard --workflow final
```

To evaluate the immutable conservative baseline through the same engine:

```bash
./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate --baseline
```

## Baseline Definition

The fixed baseline is:

- solver family: `vqe`
- ansatz: `real_amplitudes`
- `vqe_reps=1`
- entanglement: `linear`
- optimizer: `COBYLA`
- conservative iteration tolerance/shots
- no adaptation

You can run it directly on one instance with:

```bash
./.venv/bin/python experiment_baseline.py --problem knapsack_12_s0
```

## Artifacts

Per-instance diagnostics:

- `results.tsv`
- `instance_progress.png`
- machine-readable per-run JSON summaries from `experiment.py`

Suite-level primary artifacts:

- `suite_results.tsv`
- `suite_history.jsonl`
- `plots/` (stage scout trajectories, promotion comparisons, overview, heatmap)
- `progress.png` (legacy copy of the curriculum overview)
- `experiment_log.jsonl`
- `experiment_diffs/*.patch`

Study-level comparison artifacts:

- `studies/<study_id>/runs.tsv`
- `studies/<study_id>/attempts.jsonl`
- `studies/<study_id>/analysis/*.tsv`
- `studies/<study_id>/analysis/*.png`

`results.tsv` is diagnostic-only and must never be used for keep/revert decisions.
`experiment_log.jsonl` records the full candidate trajectory, including the
proposed `experiment.py` diff for both kept and rejected candidates.
Accepted policy checkpoints also appear as git commits when you run
`agent_harness.py`, and patch files are archived under `experiment_diffs/`.

## Prompt Variants

Files under `studies/prompts/` are ablations on agent guidance, not alternate definitions of the objective. Every prompt variant must target the same suite workflow and the same metric semantics.

## Quick Start

Create the environment if needed:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Validate the stack:

```bash
./.venv/bin/python prepare.py --validate-only
```

Run one adaptive instance diagnostic:

```bash
./.venv/bin/python experiment.py --problem knapsack_12_s0 --max-attempts 3
```

Run the candidate suite:

```bash
./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate
```

Run a comparative study:

```bash
./.venv/bin/python study_runner.py run --manifest studies/example_manifest.json
./.venv/bin/python study_analysis.py --study-dir studies/knapsack_repr_demo
```
