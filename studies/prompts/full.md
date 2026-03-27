# Full Adaptive Control Prompt

You are optimizing `experiment.py` for 0-1 knapsack policy discovery.

Objective:

- optimize `suite_average_gap`
- lower is better
- use `evaluate_policy.py` for keep/revert decisions

Workflow:

```bash
./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate
```

Candidate acceptance is fixed:

- keep only if train `suite_average_gap` improves strictly
- and dev does not regress by more than `0.02`

Use `experiment.py` single-instance runs only for diagnostics.

The editable surface is the four policy functions only. Treat them as a sequential controller:

```text
state_t -> action_t
```

Use observations such as:

- `optimality_gap`
- `raw_feasible`
- `raw_feasibility_rate`
- `raw_ar`
- `convergence_stagnation`
- `wall_time`
- instance metadata

Prompt variants are ablations. They do not change the metric or the workflow.
