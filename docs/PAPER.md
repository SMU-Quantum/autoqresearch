# Paper Artifact Notes

## Accepted Paper

**Title:** AutoQResearch: LLM-Guided Closed-Loop Policy Search for Adaptive
Variational Quantum Optimization

**Authors:** Monit Sharma and Hoong Chuin Lau

**Venue/status:** Accepted as QCE26 Technical Paper 238 in the
Quantum-GenAI Co-Design & Discovery (QGDD) Technical Papers track.

## Artifact Scope

This repository is the public artifact for the accepted work. It contains:

- the AutoQResearch Python framework
- the constrained editable policy surface used by the LLM agent
- fixed MIS and CVRP evaluation harnesses with scout/promote/confirm protocols
- solver implementations for VQE, QAOA, PCE, and QRAO
- MIS and CVRP benchmark instances
- archived policy diffs and promoted policy checkpoints
- generated plots and tables used for paper-facing analysis
- hardware-run scripts, retained policies, and result logs

MIS artifacts live under `mis_results/` and `plots/plots_mis/`. CVRP artifacts
live under `cvrp_results/`.

## Reproduction Entry Points

Validate the environment:

```bash
./.venv/bin/python prepare.py --validate-only
```

MIS representative workflow:

```bash
./.venv/bin/python evaluate_policy.py --suite mis_probe_16 --workflow split --split train --no-artifacts
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --promote-beam --promote-top-k 3 --restore-best
./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --no-artifacts
```

CVRP representative workflow:

```bash
./.venv/bin/python evaluate_policy.py --suite cvrp_scout_8 --workflow split --split train --no-artifacts
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_8 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite cvrp_curriculum_8 --promote-beam --promote-top-k 3 --restore-best
./.venv/bin/python evaluate_policy.py --suite cvrp_benchmark_e13 --workflow final
```

The exact historical results are captured in the tracked ledgers and logs:

```text
mis_results/
cvrp_results/
experiment_diffs/mis_diffs/
plots/plots_mis/
paper_analysis/
hardware_runs/results_hardware/
```

## Citation

Until the official proceedings citation and DOI are available, cite the paper
as:

```bibtex
@inproceedings{sharma2026autoqresearch,
  title     = {AutoQResearch: LLM-Guided Closed-Loop Policy Search for Adaptive Variational Quantum Optimization},
  author    = {Sharma, Monit and Lau, Hoong Chuin},
  booktitle = {QCE26 Quantum-GenAI Co-Design and Discovery (QGDD) Technical Papers},
  note      = {Technical Paper 238, accepted},
  year      = {2026}
}
```

