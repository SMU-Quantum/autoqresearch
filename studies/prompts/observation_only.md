# Observation Only Prompt

You are optimizing `experiment.py` for 0-1 knapsack.

Metric:

- `suite_average_gap`
- lower is better

Workflow:

```bash
./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate
```

Acceptance rule:

- keep only if train improves strictly
- and dev does not regress by more than `0.02`

Attempt observations available in `history`:

- `learning_score`
- `optimality_gap`
- `raw_feasible`
- `raw_feasibility_rate`
- `raw_ar`
- `convergence_improvement`
- `convergence_stagnation`
- `final_cost`
- `policy_used`
- `wall_time`

Edit only the four policy functions. Use single-instance runs only for diagnostics.
