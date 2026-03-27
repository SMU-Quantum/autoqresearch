"""
Experiment ledger: results.tsv tracking and git commit/rollback helpers.

LEGACY / NOT USED FOR KNAPSACK POLICY OBJECTIVE.

The ledger is a TSV file that records every experiment. It is NOT
committed to git — only experiment.py changes are tracked.

DO NOT MODIFY.
"""

from __future__ import annotations

import os
import csv
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..evaluation.evaluator import EvaluationResult


class ExperimentLedger:
    """Manages the results.tsv experiment log."""

    def __init__(self, results_path: str = "results.tsv"):
        self.path = Path(results_path)
        self._ensure_exists()

    def _ensure_exists(self):
        """Create results.tsv with header if it doesn't exist."""
        if not self.path.exists():
            with open(self.path, 'w') as f:
                f.write(
                    "experiment_id\ttimestamp\t"
                    + EvaluationResult.tsv_header()
                    + "\tstatus\tdescription\n"
                )

    def record(
        self,
        eval_result: EvaluationResult,
        experiment_id: int,
        status: str = "discard",
        description: str = "",
    ):
        """Append an experiment result to the ledger."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, 'a') as f:
            row = (
                f"{experiment_id}\t{timestamp}\t{eval_result.to_tsv_row()}\t"
                f"{status}\t{description}\n"
            )
            f.write(row)

    def get_best_score(self) -> float:
        """Read the best composite score from the ledger."""
        if not self.path.exists():
            return float('-inf')

        best = float('-inf')
        with open(self.path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                try:
                    score = float(row.get('composite_score', '-inf'))
                    best = max(best, score)
                except (ValueError, TypeError):
                    continue
        return best

    def get_num_experiments(self) -> int:
        """Count completed experiments."""
        if not self.path.exists():
            return 0
        with open(self.path, 'r') as f:
            return sum(1 for _ in f) - 1  # subtract header

    def get_last_n_results(self, n: int = 5) -> list[dict]:
        """Get the last N experiment results as dicts."""
        if not self.path.exists():
            return []
        with open(self.path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
        return rows[-n:]


class GitManager:
    """Manages git commit/rollback for the experiment loop."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, cwd=self.repo_path,
            capture_output=True, text=True, timeout=30,
        )

    def setup_branch(self, tag: str) -> str:
        """Create and checkout an experiment branch."""
        branch = f"autoqresearch/{tag}"
        # Check if branch exists
        result = self._run(["git", "branch", "--list", branch])
        if branch not in result.stdout:
            self._run(["git", "checkout", "-b", branch])
        else:
            self._run(["git", "checkout", branch])
        return branch

    def commit(self, message: str):
        """Stage experiment.py and commit."""
        self._run(["git", "add", "experiment.py"])
        self._run(["git", "commit", "-m", message])

    def rollback(self):
        """Discard changes to experiment.py (revert to last commit)."""
        self._run(["git", "checkout", "--", "experiment.py"])

    def get_last_commit_hash(self) -> str:
        result = self._run(["git", "rev-parse", "--short", "HEAD"])
        return result.stdout.strip()

    def get_diff(self) -> str:
        """Get the diff of experiment.py since last commit."""
        result = self._run(["git", "diff", "experiment.py"])
        return result.stdout
