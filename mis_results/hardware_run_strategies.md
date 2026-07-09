# Hardware Run Strategies

This note lists the retained instance-level strategies from the reduced MIS curriculum so they can be replayed on actual quantum hardware.

Scope:
- Includes the retained 16-node curriculum instances, retained 32-node curriculum instances, the retained 48-node sparse instance, and the retained 64-node sparse held-out instance.
- Uses the evidence preserved in [agent_journal.md](/Users/monitsharma/SMU-Quantum/autoqresearch/agent_journal.md), [instance_results.jsonl](/Users/monitsharma/SMU-Quantum/autoqresearch/instance_results.jsonl), and the current locked controller in [experiment.py](/Users/monitsharma/SMU-Quantum/autoqresearch/experiment.py).

## Global Defaults

These settings are common unless a row below overrides them:

```json
{
  "optimizer_method": "COBYLA",
  "optimizer_maxiter": 150,
  "optimizer_tol": 1e-3,
  "learning_rate": 0.05,
  "entanglement": "linear",
  "estimator_shots": 1024,
  "sampler_shots": 1024,
  "seed": 17,
  "penalty": null,
  "pce_local_search": false,
  "final_local_search": false
}
```

Recommended command skeleton:

```bash
./.venv/bin/python experiment.py \
  --problem <mis_instance_name> \
  --backend <hardware_backend_mode> \
  --max-attempts <n> \
  --summary-json runs/<tag>_summary.json \
  --attempts-jsonl runs/<tag>_attempts.jsonl \
  --winning-policy-json runs/<tag>_winning_policy.json \
  --no-results-log \
  --no-progress-plot
```

If you want to pin a fixed one-shot policy instead of using the adaptive controller in [experiment.py](/Users/monitsharma/SMU-Quantum/autoqresearch/experiment.py), pass it with `--policy-json '<json>'`.

## 16-Node Instances

Best retained 16-node family strategy:
- Solver: `QAOA warmstart`
- Key knobs: `reps=1`, `measurement_mode=cvar`, `cvar_alpha=0.25`, `ws_epsilon=0.25`, `ws_source=relaxation`
- Attempts: `1`
- Evidence: direct full-stage confirm `eval_group_id=10`

Fixed policy JSON:

```json
{
  "solver_family": "qaoa",
  "variant": "warmstart",
  "reps": 1,
  "ws_epsilon": 0.25,
  "ws_source": "relaxation",
  "measurement_mode": "cvar",
  "cvar_alpha": 0.25,
  "optimizer_method": "COBYLA",
  "optimizer_maxiter": 150,
  "optimizer_tol": 1e-3,
  "learning_rate": 0.05,
  "entanglement": "linear",
  "estimator_shots": 1024,
  "sampler_shots": 1024,
  "seed": 17,
  "penalty": null,
  "pce_local_search": false,
  "final_local_search": false
}
```

Per-instance retained 16-node results:

| Instance | Recommended strategy | Observed gap |
| --- | --- | --- |
| `mis_file_1tc.16` | QAOA warmstart CVaR(0.25), depth 1 | `0.0000` |
| `mis_file_p1tc.16` | QAOA warmstart CVaR(0.25), depth 1 | `0.0000` |
| `mis_file_p2tc.16` | QAOA warmstart CVaR(0.25), depth 1 | `0.1250` |
| `mis_file_p3tc.16` | QAOA warmstart CVaR(0.25), depth 1 | `0.0000` |
| `mis_file_p4tc.16` | QAOA warmstart CVaR(0.25), depth 1 | `0.0000` |

## 32-Node Instances

Best retained 32-node family strategy:
- Solver: `QRAO`
- Key knobs: `qrao_max_vars_per_qubit=3`, `qrac_type=3`, `rounding=magic`, `ansatz_type=real_amplitudes`, `vqe_reps=1`
- Measurement mode: `expectation`
- Attempts: `5` with the same static policy
- Evidence: full 32-node confirm winner `eval_group_id=19`

Fixed policy JSON:

```json
{
  "solver_family": "qrao",
  "qrao_max_vars_per_qubit": 3,
  "qrac_type": 3,
  "rounding": "magic",
  "ansatz_type": "real_amplitudes",
  "vqe_reps": 1,
  "measurement_mode": "expectation",
  "optimizer_method": "COBYLA",
  "optimizer_maxiter": 150,
  "optimizer_tol": 1e-3,
  "learning_rate": 0.05,
  "entanglement": "linear",
  "estimator_shots": 1024,
  "sampler_shots": 1024,
  "seed": 17,
  "penalty": null,
  "pce_local_search": false,
  "final_local_search": false
}
```

Per-instance retained 32-node results:

| Instance | Recommended strategy | Observed gap |
| --- | --- | --- |
| `mis_file_1tc.32` | QRAO 3:1 magic, RealAmplitudes d=1 | `0.2500` |
| `mis_file_p1tc.32` | QRAO 3:1 magic, RealAmplitudes d=1 | `0.1538` |
| `mis_file_p3tc.32` | QRAO 3:1 magic, RealAmplitudes d=1 | `0.2727` |
| `mis_file_p5tc.32` | QRAO 3:1 magic, RealAmplitudes d=1 | `0.3571` |
| `mis_file_p8tc.32` | QRAO 3:1 magic, RealAmplitudes d=1 | `0.2000` |

## 48-Node Retained Sparse Instance

Best retained 48-node strategy is not a single static policy. It is the large-instance sparse controller now encoded in [experiment.py](/Users/monitsharma/SMU-Quantum/autoqresearch/experiment.py):

Attempt schedule:
1. Attempt 0: `QRAO 3:1 semideterministic`, `RealAmplitudes`, `vqe_reps=1`
2. Attempt 1: if attempt 0 is infeasible or `gap > 0.4`, switch to `QRAO 2:1 semideterministic`
3. Attempt 2: keep the `QRAO 2:1 semideterministic` branch if another retry is allowed

Observed evidence:
- Best direct reduced sparse-only result in the journal: `mis_file_p1tc.48=0.3333`
- Best retained confirm result in filtered logs: `mis_file_p1tc.48=0.5333` at `eval_group_id=28`

Hardware recommendation:
- Use the adaptive controller already in [experiment.py](/Users/monitsharma/SMU-Quantum/autoqresearch/experiment.py)
- Run with `--max-attempts 3`
- Do not pin a static `3:1`-only policy for this instance

Fixed attempt policies if you want to orchestrate manually:

Attempt 0 policy:

```json
{
  "solver_family": "qrao",
  "qrao_max_vars_per_qubit": 3,
  "qrac_type": 3,
  "rounding": "semideterministic",
  "ansatz_type": "real_amplitudes",
  "vqe_reps": 1,
  "measurement_mode": "expectation",
  "optimizer_method": "COBYLA",
  "optimizer_maxiter": 150,
  "optimizer_tol": 1e-3,
  "learning_rate": 0.05,
  "entanglement": "linear",
  "estimator_shots": 1024,
  "sampler_shots": 1024,
  "seed": 17,
  "penalty": null,
  "pce_local_search": false,
  "final_local_search": false
}
```

Fallback policy for attempts 1-2:

```json
{
  "solver_family": "qrao",
  "qrao_max_vars_per_qubit": 2,
  "qrac_type": 2,
  "rounding": "semideterministic",
  "ansatz_type": "real_amplitudes",
  "vqe_reps": 1,
  "measurement_mode": "expectation",
  "optimizer_method": "COBYLA",
  "optimizer_maxiter": 150,
  "optimizer_tol": 1e-3,
  "learning_rate": 0.05,
  "entanglement": "linear",
  "estimator_shots": 1024,
  "sampler_shots": 1024,
  "seed": 17,
  "penalty": null,
  "pce_local_search": false,
  "final_local_search": false
}
```

Retained 48-node instance:

| Instance | Recommended strategy | Observed result |
| --- | --- | --- |
| `mis_file_p1tc.48` | Adaptive `QRAO 3:1 semideterministic -> QRAO 2:1 semideterministic`, max 3 attempts | Best journal result `gap=0.3333`; best retained confirm `gap=0.5333` |

## 64-Node Retained Sparse Held-Out Instance

Use the same large-instance sparse controller as the retained 48-node case:
- Attempt 0: `QRAO 3:1 semideterministic`
- Attempt 1+: fallback to `QRAO 2:1 semideterministic`
- Run with `--max-attempts 3`

Observed retained 64-node result:
- `mis_file_1tc.64=0.5500`, feasible, `784.8s`

Important note:
- That `0.5500` result is preserved in the journal, but the later rerun-only 64 final row was removed from the filtered JSON/TSV history. So the recommendation to use the same sparse `>32` controller is partly an inference from the retained controller and neighboring 48-node behavior, not from a surviving per-attempt 64-node JSON trace.

Retained 64-node instance:

| Instance | Recommended strategy | Observed result |
| --- | --- | --- |
| `mis_file_1tc.64` | Adaptive `QRAO 3:1 semideterministic -> QRAO 2:1 semideterministic`, max 3 attempts | Journal result `gap=0.5500` |

## Practical Recommendation

For hardware replay, the simplest split is:
- 16-node jobs: use the fixed QAOA warm-start JSON above with `--max-attempts 1`
- 32-node jobs: use the fixed QRAO 3:1 magic JSON above with `--max-attempts 5`
- 48/64-node sparse jobs: use the current adaptive controller in [experiment.py](/Users/monitsharma/SMU-Quantum/autoqresearch/experiment.py) with `--max-attempts 3`