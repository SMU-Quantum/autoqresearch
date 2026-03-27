# Custom Hardware Runner

This runner keeps all hardware-specific work under [hardware_runs](/Users/monitsharma/SMU-Quantum/autoqresearch/hardware_runs).

Default behavior now is:
- one fixed retained winner per retained MIS instance
- one solve per instance
- no adaptive fallback chain on hardware unless you explicitly opt back in

Files:

- [run_autoq_hardware.py](/Users/monitsharma/SMU-Quantum/autoqresearch/hardware_runs/run_autoq_hardware.py): standalone CLI for MIS-on-IBM runs
- [autoq_hardware_backend.py](/Users/monitsharma/SMU-Quantum/autoqresearch/hardware_runs/autoq_hardware_backend.py): IBM Runtime adapters, credential rotation, and benchmark-style backend bundle creation
- [ibm_credentials.template.json](/Users/monitsharma/SMU-Quantum/autoqresearch/hardware_runs/ibm_credentials.template.json): credential pool template

What it does:

- imports the solver stack from [experiment.py](/Users/monitsharma/SMU-Quantum/autoqresearch/experiment.py)
- runs the real `autoqresearch` VQE, QAOA, and QRAO solvers against IBM Runtime
- checks IBM runtime before every primitive submission and rotates to the next account when remaining runtime falls below `--ibm-min-runtime-seconds`
- defaults to static retained winners:
  - `1tc.16 -> QAOA warmstart CVaR`
  - `1tc.32 -> QRAO 3:1 magic`
  - `p1tc.48 -> QRAO 2:1 semideterministic`
  - `1tc.64 -> QRAO 2:1 semideterministic`
- writes benchmark-style artifacts:
  - `run.log`
  - `summary.json`
  - per-instance `qubo.lp`
  - per-instance `trace.jsonl`
  - per-instance `best_counts.json`
  - per-instance `winning_policy.json`
  - per-instance `result.json`

Example commands:

```bash
./.venv/bin/python hardware_runs/run_autoq_hardware.py \
  --retained-only \
  --ibm-credentials-json hardware_runs/ibm_credentials.json
```

```bash
./.venv/bin/python hardware_runs/run_autoq_hardware.py \
  --instance 1tc.16 \
  --instance 1tc.32 \
  --instance p1tc.48 \
  --instance 1tc.64 \
  --ibm-credentials-json hardware_runs/ibm_credentials.json \
  --ibm-min-runtime-seconds 60 \
  --timeout-sec 43200
```

```bash
./.venv/bin/python hardware_runs/run_autoq_hardware.py \
  --instance 1tc.32 \
  --plan-only
```

Notes:

- `--plan-only` resolves the instances and prints the starting policy without touching IBM Runtime.
- The default runner mode is the adaptive multi-attempt controller from `experiment.py`.
- `--static-retained` replays one fixed retained-instance winner per instance if you specifically want that benchmark.
- Checkpoints live under `hardware_runs/checkpoints_autoq/mis.json` by default.
- Output runs land under `hardware_runs/results_hardware/mis/`.
