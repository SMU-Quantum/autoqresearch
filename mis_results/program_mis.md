# AutoQResearch — Agent Instructions

You are improving a quantum algorithm policy for Maximum Independent Set (MIS).

Your job is to discover the best algorithm, ansatz, hyperparameters, and adaptive strategy for solving MIS across a diverse set of graph instances. The goal is a policy that **generalises** across graph sizes and structures, not one that overfits to a single instance.

## Non-negotiable rules

1. Edit exactly two files: `experiment.py` (policy code) and `agent_journal.md` (your research log).
2. Only edit the four functions in the policy surface inside `experiment.py`.
3. Everything outside that surface is fixed evaluation infrastructure.
4. **No classical local search.** All solvers must have `pce_local_search: False` and `final_local_search: False`. Classical bit-swap post-processing is disabled so all solver families compete purely on quantum algorithm output.
5. **Evidence-driven exploration mandate.** Use the available degrees of freedom below and choose experiments based on observed solver behavior.
6. **Minimum iterations.** You must run at least **15 agent iterations** before declaring your final policy.
7. `optimality_gap` is the metric. `0.0` = found the maximum independent set. `1.0` = total failure (infeasible or trivial).
8. MIS is a constrained problem — even finding a feasible solution (no adjacent nodes both selected) is non-trivial at scale.
9. **Do not abort runs for taking too long.** Let every solver run to completion. Record the wall time. The tradeoff between solution quality and compute time is a finding — not a reason to skip an experiment.

## Available Degrees Of Freedom

You are not required to mechanically exhaust a checklist. Instead, use the search space below to design experiments that respond to observed failure modes in the logs:

- poor concentration: low `top1_prob`, flat `top10`, concentration guard firing
- low feasibility: low `raw_feasibility_rate`, infeasible top samples
- optimizer stagnation: high `convergence_stagnation`, weak cost improvement
- runtime blowups: strong quality but unacceptable wall time
- size-specific failure: works on 16/32 nodes but not on larger graphs

### Solver families
- **VQE** — variational eigensolver with parameterised ansatz
- **QAOA** — quantum approximate optimisation algorithm
- **PCE** — Pauli Correlation Encoding (QUBO → weighted MaxCut reduction)
- **QRAO** — Quantum Random Access Optimisation (qubit compression + rounding)

### VQE degrees of freedom
- `ansatz_type`: `real_amplitudes`, `efficient_su2`, `pauli_two_design`, `brickwork`, `custom`
- `vqe_reps`
- `entanglement`
- `measurement_mode`: `expectation` or `cvar`
- `cvar_alpha`
- `optimizer_method`, `optimizer_maxiter`, `optimizer_tol`, `learning_rate`
- `estimator_shots`, `sampler_shots`
- `penalty`

### QAOA degrees of freedom
- `variant`: `standard`, `warmstart`, `multiangle`
- `reps`
- warm-start knobs: `ws_epsilon`, `ws_source`
- multi-angle knobs: `ma_tying`
- `measurement_mode`: `expectation` or `cvar`
- `cvar_alpha`
- `optimizer_method`, `optimizer_maxiter`, `optimizer_tol`, `learning_rate`
- `estimator_shots`, `sampler_shots`
- `penalty`

### PCE degrees of freedom
- `pce_k`
- `pce_depth`
- `pce_alpha`
- `pce_beta`
- `ansatz_type`
- `measurement_mode`: `expectation` or `cvar`
- `cvar_alpha`
- `optimizer_method`, `optimizer_maxiter`, `optimizer_tol`, `learning_rate`
- `estimator_shots`, `sampler_shots`
- `penalty`

### QRAO degrees of freedom
- `qrao_max_vars_per_qubit`
- `rounding`
- `ansatz_type`
- `vqe_reps`
- `entanglement`
- `measurement_mode`: `expectation` or `cvar`
- `cvar_alpha`
- `optimizer_method`, `optimizer_maxiter`, `optimizer_tol`, `learning_rate`
- `estimator_shots`, `sampler_shots`
- `penalty`

### Coverage guidance
- Touch all solver families unless early evidence clearly rules one out.
- Prefer deepening promising branches over spending budget on already dominated ones.
- If standard ansatz families plateau, repeatedly show poor concentration, or fail to generalise, explicitly consider a `custom` ansatz via `custom_ansatz_fn`.
- Use custom ansatz search deliberately: justify what structural weakness in the current ansatz you are trying to fix.

### Seed policy (important)
Seeds simulate the randomness of real quantum hardware. **Do NOT use seed as a tuning parameter.** Changing the seed to get a better result is not a valid improvement — it's luck that won't transfer to real hardware. Seeds are only for:
- **Verification**: if you get a surprisingly good result, re-run with 2-3 different seeds to confirm it's robust, not a lucky draw.
- **Post-lock robustness validation**: once the final policy is frozen, it is valid to run a fixed 5-seed robustness report on the held-out 64-node suite. That is evidence collection, not search.
- **Reproducibility**: keep seed=17 as the default so results are deterministic across runs.

## Training and evaluation suites

### `mis_probe_16` — quick sanity check (~30s)
Single 16-node instance (MIS=8). Use for fast testing before a full eval. `agent_harness.py` does not define a scout plan for this suite, so use a direct split evaluation or a direct `experiment.py` run instead. This path does not write to the log:
```bash
./.venv/bin/python evaluate_policy.py --suite mis_probe_16 --workflow split --split train --no-artifacts
```

### `mis_curriculum_16` — curriculum stage 1
5× 16-node instances for confirmation. Use the scout workflow first; it evaluates a cheaper 2-instance proxy and maintains a beam of promising candidates.
```bash
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --promote-beam --promote-top-k 3 --restore-best
```

### `mis_curriculum_32` — curriculum stage 2
5× 32-node instances for confirmation. The scout workflow uses a cheaper 2-instance 32-node proxy and replays a cheap 16-node proxy as a guardrail.
```bash
./.venv/bin/python agent_harness.py --suite mis_curriculum_32 --eval-workflow scout --wall-clock-budget 2400 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite mis_curriculum_32 --promote-beam --promote-top-k 3 --restore-best
```

### `mis_curriculum_48` — curriculum stage 3
Single retained 48-node sparse instance for confirmation. The scout workflow tracks that retained sparse target and replays cheap 32-node and 16-node proxies as guardrails.

```bash
./.venv/bin/python agent_harness.py --suite mis_curriculum_48 --eval-workflow scout --wall-clock-budget 3000 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite mis_curriculum_48 --promote-beam --promote-top-k 3 --restore-best
```

### `mis_curriculum_64` — held-out final evaluation
Single retained 64-node sparse held-out instance. **Do not use this suite for KEEP/DISCARD.** Run only after locking your final 48-stage policy:

```bash
./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --no-artifacts
```

Legacy note: `mis_train` is deprecated and removed from the public suite surface. Use the curriculum suites above instead.

## What "better" means

There are now two levels of evidence:

- **Scout level**: cheap proxy search under a fixed wall-clock budget. The scout incumbent is updated only if the proxy primary metric improves strictly and replay proxies do not regress badly.
- **Confirm level**: the top beam candidates are rerun on the full stage suite plus full replay guardrails. Only confirmed results count as stage winners in the journal and final summary.

The replay tolerance is fixed: earlier-stage `suite_average_gap` must not worsen by more than `0.02` versus the incumbent for that stage or workflow.

## Recommended workflow

1. **Probe first**: test your idea on `mis_probe_16` through `evaluate_policy.py --workflow split --split train --no-artifacts` (~30s).
2. **Scout stage 1**: search on `mis_curriculum_16` using `--eval-workflow scout`, a fixed wall-clock budget, and beam width 5.
3. **Promote stage 1**: run `--promote-beam` on `mis_curriculum_16`, confirm the top 3 beam candidates on the full 16-node stage, and restore the best confirmed snapshot.
4. **Scout stage 2**: carry that confirmed winner forward unchanged, seed `mis_curriculum_32`, then search with `--eval-workflow scout`.
5. **Promote stage 2**: confirm the top 32-node beam candidates on the full 32-node stage and restore the best confirmed snapshot.
6. **Scout stage 3**: repeat on `mis_curriculum_48`.
7. **Promote stage 3**: confirm the top 48-node beam candidates on the full 48-node stage and restore the best confirmed snapshot.
8. **Final**: evaluate the locked policy on `mis_curriculum_64` once, through the true `final` workflow.

## The editable policy surface

The checked-in starting policy is the required baseline: **VQE with `real_amplitudes` and depth (`vqe_reps`) = 1**. Other solver families remain available inside `build_base_policy()`, but they should only be activated deliberately by editing `choose_solver_family()` in response to observed evidence. There must be no baked-in warm-start QAOA default.

### Available policy keys

**Common to all families:**
```python
{
    "solver_family": "vqe" | "qaoa" | "pce" | "qrao",
    "optimizer_method": "COBYLA",     # also: "SPSA", "Powell", "Nelder-Mead"
    "optimizer_maxiter": 150,
    "optimizer_tol": 1e-3,
    "learning_rate": 0.05,            # for SPSA
    "estimator_shots": 1024,          # powers of 2 only
    "sampler_shots": 1024,            # powers of 2 only
    "seed": 17,                        # DO NOT tune — see seed policy below
    "measurement_mode": "expectation", # or "cvar"
    "cvar_alpha": 0.25,               # only used when measurement_mode="cvar"
    "penalty": None,                  # None=auto, or float to override QUBO penalty
    "pce_local_search": False,        # must stay False
    "final_local_search": False,      # must stay False
}
```

**VQE-specific:**
```python
{
    "variant": "standard",
    "ansatz_type": "real_amplitudes",  # also: "efficient_su2", "pauli_two_design", "brickwork", "custom"
    "vqe_reps": 1,                    # circuit depth (1, 2, 3, ...)
    "entanglement": "linear",         # also: "circular", "full", "sca"
    "custom_ansatz_fn": None,         # callable(num_qubits, reps, entanglement) -> QuantumCircuit
}
```

**QAOA-specific:**
```python
{
    "variant": "standard",            # also: "warmstart", "multiangle"
    "reps": 1,                        # QAOA layers (1, 2, 3, ...)
    # Warm-start options:
    "ws_epsilon": 0.25,               # relaxation mixing parameter
    "ws_source": "relaxation",        # warm-start source
    # Multi-angle options:
    "ma_tying": "none",               # "none", "partial", "full"
}
```

**PCE-specific:**
```python
{
    "pce_k": 2,                       # partition size
    "pce_depth": 10,                  # PCE circuit depth
    "pce_alpha": None,                # regularisation
    "pce_beta": 0.5,                  # loss function parameter
}
```

**QRAO-specific:**
```python
{
    "qrao_max_vars_per_qubit": 3,     # compression ratio (1, 2, or 3)
    "rounding": "semideterministic",  # also: "magic"
}
```

## Solver implementation details

Read these files to understand what each solver does:

- `autoqresearch/solvers/qubo_primitives.py` — VQE and QAOA solvers for QUBO problems (MIS, knapsack). Contains `solve_qubo_vqe()`, `solve_qubo_qaoa()`, `solve_qubo_pce()`, warm-start QAOA, multi-angle QAOA, all ansatz builders, CVaR cost function, and solution extraction with concentration guard.
- `autoqresearch/solvers/pce_solver.py` — PCE solver wrapper. Routes MIS to `solve_qubo_pce()` which converts QUBO → weighted MaxCut.
- `autoqresearch/solvers/qrao_solver.py` — QRAO solver. Uses 3:1 qubit compression with semideterministic or magic rounding.
- `autoqresearch/solvers/maxcut_primitives.py` — MaxCut-native solvers (for reference, not directly used for MIS).

## Sampling concentration signal

Each attempt prints `top1_prob` and a `top10` summary showing the 10 most-probable bitstrings with their counts, number of selected nodes, and feasibility (F=feasible, X=infeasible).

- If `top1_prob < 0.01` and all top counts are 1-3, the circuit produces **uniform noise** — the concentration guard rejects this and returns gap=1.0.
- If `top1_prob > 0.05` with a clear winner, the circuit has learned something meaningful.
- Use this signal to decide: increase depth? switch ansatz? increase shots? switch family?

## Agent journal

Maintain `agent_journal.md`. Before every edit, log:
1. **Hypothesis**: what you're trying and why
2. **Failure signal**: what observed issue motivated this experiment (for example low `top1_prob`, low feasibility rate, stagnation, or excessive runtime)
3. **Degrees of freedom exercised**: which solver knobs or families you are changing
4. **Change summary**: one-line description

After each evaluation, log:
5. **Result**: KEEP/DISCARD, `suite_average_gap`, and per-instance gaps
6. **Wall time**: total evaluation time
7. **Analysis**: what you learned, what to try next

## Autonomous loop

### Before your first edit

1. Read solver files to understand what is available.
2. **Run the baseline scout.** Do NOT edit `experiment.py` first. Run the unmodified policy on the first curriculum stage:
   ```bash
   ./.venv/bin/python agent_harness.py --single-run --suite mis_curriculum_16 --eval-workflow scout --no-dev
   ```
   The harness will automatically record this as experiment #0 (baseline) with the checked-in VQE RealAmplitudes depth=1 policy. This seeds the 16-node scout incumbent.
3. Log the baseline scout `suite_average_gap` as entry #0 in your journal.
4. **Plan your exploration.** Write an initial roadmap in your journal using the available degrees of freedom above. This roadmap is provisional: refine it as evidence comes in. Do not pre-commit to exhaustive coverage of weak branches.

### Each iteration

1. Log hypothesis in the journal, including the observed failure mode that motivated the experiment.
2. Edit policy functions in `experiment.py`.
3. Optionally probe on `mis_probe_16` via `evaluate_policy.py --workflow split --split train --no-artifacts`.
4. During search, run the stage-appropriate scout eval:
   - stage 1: `--suite mis_curriculum_16 --eval-workflow scout --no-dev`
   - stage 2: `--suite mis_curriculum_32 --eval-workflow scout --no-dev`
   - stage 3: `--suite mis_curriculum_48 --eval-workflow scout --no-dev`
5. Log scout result and analysis.
6. When the wall-clock budget expires or progress plateaus, run `--promote-beam` for that stage.
7. Log the confirmed stage winner separately from the scout history.
8. Repeat.

### Stopping — ONLY after validation

**DO NOT STOP EARLY.** You must complete the full research protocol:

1. Run at least 15 training iterations across scout stages (more is better).
2. Document that the explored families/knobs were chosen from evidence, and note which families were ruled out early and why.
3. For each stage, distinguish clearly in the journal between:
   - scout leader(s) on the cheap proxy
   - promoted beam candidates
   - confirmed stage winner on the full suite
4. Before moving from 16 → 32 or 32 → 48, first restore the best confirmed snapshot from the previous stage. Then seed the next scout stage with that policy unchanged.
5. Run the held-out final suite on the final best policy:
   ```bash
   ./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --no-artifacts
   ```
6. Run post-lock paper validation on the held-out suite:
   ```bash
   ./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow robustness --seed-list 17,23,29,31,37 --no-artifacts
   ./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --classical-baseline greedy_min_degree --no-artifacts
   ./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --classical-baseline random_feasible --seed-override 23 --no-artifacts
   ```
   When artifacts are enabled, the evaluator will also emit stage-winner resource tables plus family-scaling and Pareto-frontier outputs.
7. Write a final summary in `agent_journal.md` covering:
   - Best policy found and its confirmed `suite_average_gap`
   - The confirmed gaps at each curriculum stage (16, 32, 48)
   - Which families/variants worked best and which failed
   - Which hyperparameters mattered most
   - Key insights and surprising findings
   - Validation result on the unseen retained 64-node instance
8. **Only then** are you done.

If you are running low on context or time, do NOT silently stop. Instead: run the validation, write the summary, and finish cleanly. Never leave the run in an incomplete state.

### Rules

- One conceptual change per iteration.
- Do not repeat failed strategies.
- Touch all solver families unless early evidence rules one out. If you rule one out, say why in the journal.
- Every experiment must be justified by an observed failure mode or a concrete transfer hypothesis.
- Record wall time for every run. If a method is slow but effective, note the tradeoff.
- **Curriculum-aware, size-aware policies are encouraged**: use `problem.num_variables` to set different hyperparameters for 16-node vs 32-node vs 48-node regimes.

## Readable files

- `experiment.py` — your code
- `agent_journal.md` — your research notebook
- `experiment_log.jsonl` — evaluation history (KEEP/DISCARD decisions)
- `beam_state.json` — active scout beam per curriculum stage
- `beam_history.jsonl` — beam entries that were admitted during scout search
- `promotion_log.jsonl` — confirmation runs for promoted beam candidates
- `instance_results.jsonl` — per-instance results with gap, policy, wall time for each graph
- `experiment_diffs/*.patch` — archived diffs
- `policy_checkpoints/` — snapshot copies of promoted scout candidates
- `suite_results.tsv` — evaluation ledger (suite-level averages)
- `plots/` — stage-specific scout trajectories, promotion comparisons, overview, and heatmap
- `paper_analysis/` — robustness summaries, stage-winner resource tables, scaling tables, and Pareto tables
- `progress.png` — legacy copy of the curriculum overview plot
- `autoqresearch/solvers/` — solver implementations
- `autoqresearch/problems/mis.py` — MIS problem definition
- `individual/mis/` — benchmark instance files
