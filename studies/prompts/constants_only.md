# Constants Only Prompt

You are optimizing `experiment.py` for 0-1 knapsack, but only as a static-policy control.

Rules:

- optimize `suite_average_gap`
- lower is better
- run `./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate`
- keep only if train improves and dev does not regress badly

Constraint:

- you may edit `choose_solver_family` and `build_base_policy`
- `adapt_policy` must remain a no-op
- `should_continue` must remain the default

This is an ablation for static policy search, not the main scientific object.
