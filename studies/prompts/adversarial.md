# Adversarial Prompt

You are optimizing `experiment.py` for 0-1 knapsack.

This prompt variant is a negative-control ablation. The objective and workflow do not change:

- optimize `suite_average_gap`
- lower is better
- run `./.venv/bin/python evaluate_policy.py --suite quick --workflow candidate`
- keep only if train improves and dev does not regress badly

The guidance in this file is intentionally poor. The experiment still uses the same fixed metric semantics and the same keep/revert rule as every other prompt variant.
