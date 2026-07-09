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
- the fixed MIS evaluation harness and scout/promote/confirm protocol
- solver implementations for VQE, QAOA, PCE, and QRAO
- MIS benchmark instances used by the staged curriculum
- archived policy diffs and promoted policy checkpoints
- generated plots and tables used for paper-facing analysis
- hardware-run scripts, retained MIS policies, and result logs

The accepted paper also discusses a decomposed CVRP workflow. This repository
keeps the reusable framework and paper-facing MIS artifact in the foreground;
CVRP-specific material should be added under a dedicated source and artifact
directory if it is released here later.

## Main Reproduction Entry Points

Use these commands for the MIS artifact:

```bash
./.venv/bin/python prepare.py --validate-only
./.venv/bin/python evaluate_policy.py --suite mis_probe_16 --workflow split --split train --no-artifacts
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --eval-workflow scout --wall-clock-budget 1800 --beam-width 5 --no-dev
./.venv/bin/python agent_harness.py --suite mis_curriculum_16 --promote-beam --promote-top-k 3 --restore-best
./.venv/bin/python evaluate_policy.py --suite mis_curriculum_64 --workflow final --no-artifacts
```

The exact historical results are captured in the tracked ledgers and logs:

```text
agent_journal.md
experiment_log.jsonl
beam_history.jsonl
promotion_log.jsonl
instance_results.jsonl
suite_results.tsv
paper_analysis/
plots/
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

