#!/usr/bin/env python3
"""
AutoQResearch agent harness — scout / beam / confirm outer loop.

This harness evaluates the current policy through ``evaluate_policy.py`` using
either the cheap-proxy scout workflow or the full candidate workflow. It
assumes ``experiment.py`` starts from the static conservative VQE checkpoint
and records the discovery trajectory in three places:

  - accepted policies become git commits
  - every candidate evaluation is appended to ``experiment_log.jsonl`` with the
    proposed ``experiment.py`` diff preserved even if the candidate is reverted
  - scout candidates that enter the top-K beam are snapshotted under
    ``policy_checkpoints/`` and tracked in ``beam_state.json``

The scout decision rule is:

  1. run cheap proxy evaluation under a fixed wall-clock budget
  2. read train_suite_average_gap and replay guardrail gaps
  3. keep only if proxy train improves and replay does not regress badly
  4. otherwise revert experiment.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EVALUATION_FILE = Path("evaluate_policy.py")
VENV_PYTHON = Path(".venv/bin/python")
EXPERIMENT_FILE = Path("experiment.py")
EXPERIMENT_LOG = Path("experiment_log.jsonl")
DIFF_ARCHIVE_DIR = Path("experiment_diffs")
POLICY_CHECKPOINT_DIR = Path("policy_checkpoints")
BEAM_STATE_FILE = Path("beam_state.json")
BEAM_LOG = Path("beam_history.jsonl")
PROMOTION_LOG = Path("promotion_log.jsonl")


def _find_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _git_has_changes() -> bool:
    result = _git("diff", "--name-only", str(EXPERIMENT_FILE), check=False)
    return bool(result.stdout.strip())


def _git_commit(message: str) -> bool:
    try:
        _git("add", str(EXPERIMENT_FILE))
        _git("commit", "-m", message)
        return True
    except subprocess.CalledProcessError:
        return False


def _git_revert() -> bool:
    try:
        _git("checkout", "--", str(EXPERIMENT_FILE))
        return True
    except subprocess.CalledProcessError:
        return False


def _git_current_branch() -> str:
    result = _git("branch", "--show-current", check=False)
    return result.stdout.strip()


def _git_head() -> str | None:
    result = _git("rev-parse", "HEAD", check=False)
    head = result.stdout.strip()
    return head or None


def _git_diff_text() -> str:
    result = _git("diff", "--", str(EXPERIMENT_FILE), check=False)
    return result.stdout


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _archive_diff(experiment_number: int, diff_text: str) -> str | None:
    if not diff_text.strip():
        return None
    DIFF_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = DIFF_ARCHIVE_DIR / f"experiment_{experiment_number:04d}.patch"
    path.write_text(diff_text)
    return str(path)


def _ensure_branch(branch: str) -> None:
    current = _git_current_branch()
    if current == branch:
        return
    result = _git("checkout", branch, check=False)
    if result.returncode != 0:
        _git("checkout", "-b", branch)


def _parse_evaluation_stdout(stdout: str) -> dict:
    parsed: dict[str, object] = {}
    int_keys = {"candidate_accept", "eval_group_id"}
    str_keys = {"candidate_decision", "policy_label", "workflow"}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value_str = line.split(":", 1)
        key = key.strip()
        value_str = value_str.strip()
        if key.endswith("_suite_average_gap") or key.startswith("incumbent_"):
            try:
                parsed[key] = float(value_str) if value_str not in ("None", "", "n/a") else None
            except (TypeError, ValueError):
                parsed[key] = None
        elif key in int_keys:
            try:
                parsed[key] = int(value_str) if value_str not in ("None", "") else None
            except (TypeError, ValueError):
                parsed[key] = None
        elif key in str_keys:
            parsed[key] = value_str
    return parsed


def _run_policy_evaluation(
    suite: str,
    workflow: str,
    max_attempts: int,
    timeout: int,
    prompt_variant: str,
    experiment_file: Path = EXPERIMENT_FILE,
    no_dev: bool = False,
    baseline: bool = False,
    no_artifacts: bool = False,
) -> dict:
    python = _find_python()
    cmd = [
        python,
        str(EVALUATION_FILE),
        "--suite",
        suite,
        "--workflow",
        workflow,
        "--max-attempts",
        str(max_attempts),
        "--prompt-variant",
        prompt_variant,
        "--experiment-file",
        str(experiment_file),
    ]
    if baseline:
        cmd.append("--baseline")
    if no_dev:
        cmd.append("--no-dev")
    if no_artifacts:
        cmd.append("--no-artifacts")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    stdout = result.stdout
    stderr = result.stderr

    parsed = {
        "workflow": workflow,
        "returncode": result.returncode,
        "wall_time": elapsed,
        "stdout_tail": stdout[-2000:] if len(stdout) > 2000 else stdout,
        "stderr_tail": stderr[-1000:] if len(stderr) > 1000 else stderr,
    }
    parsed.update(_parse_evaluation_stdout(stdout))

    requires_train_gap = workflow in {"candidate", "scout", "confirm"}
    if result.returncode != 0 or (requires_train_gap and "train_suite_average_gap" not in parsed):
        parsed["status"] = "crash"
    else:
        parsed["status"] = "completed"
    return parsed


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _load_beam_state() -> dict[str, list[dict]]:
    if not BEAM_STATE_FILE.exists():
        return {}
    try:
        payload = json.loads(BEAM_STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_beam_state(state: dict[str, list[dict]]) -> None:
    BEAM_STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def _suite_gap_metrics(record: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in record.items():
        if not key.endswith("_suite_average_gap"):
            continue
        if key.startswith("incumbent_"):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        metrics[key] = parsed
    return metrics


def _beam_rank_tuple(metrics: dict[str, float], wall_time: float) -> tuple[float, float, float]:
    train_gap = float(metrics.get("train_suite_average_gap", 1.0))
    guardrail_values = [
        float(value)
        for key, value in metrics.items()
        if key != "train_suite_average_gap"
    ]
    guardrail_mean = (
        sum(guardrail_values) / len(guardrail_values)
        if guardrail_values
        else 0.0
    )
    return (train_gap, guardrail_mean, float(wall_time))


def _maybe_add_to_beam(
    suite: str,
    workflow: str,
    experiment_number: int,
    prompt_variant: str,
    result: dict,
    beam_width: int,
    dry_run: bool,
) -> dict | None:
    if workflow != "scout" or result.get("status") != "completed":
        return None

    metrics = _suite_gap_metrics(result)
    if "train_suite_average_gap" not in metrics:
        return None

    source_text = EXPERIMENT_FILE.read_text()
    source_sha256 = _sha256_text(source_text)
    state = _load_beam_state()
    entries = [entry for entry in state.get(suite, []) if isinstance(entry, dict)]
    if any(entry.get("source_sha256") == source_sha256 for entry in entries):
        return None

    rank_tuple = _beam_rank_tuple(metrics, float(result.get("wall_time", 0.0) or 0.0))
    if len(entries) >= beam_width:
        worst_rank = max(
            _beam_rank_tuple(entry.get("metrics", {}), float(entry.get("wall_time", 0.0) or 0.0))
            for entry in entries
        )
        if rank_tuple >= worst_rank:
            return None

    snapshot_dir = POLICY_CHECKPOINT_DIR / suite
    snapshot_path = snapshot_dir / f"{workflow}_{experiment_number:04d}_{source_sha256[:8]}.py"
    if not dry_run:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(source_text)

    entry = {
        "suite": suite,
        "workflow": workflow,
        "experiment_number": experiment_number,
        "timestamp": datetime.now().isoformat(),
        "prompt_variant": prompt_variant,
        "source_sha256": source_sha256,
        "snapshot_path": str(snapshot_path),
        "metrics": metrics,
        "wall_time": float(result.get("wall_time", 0.0) or 0.0),
        "rank_tuple": list(rank_tuple),
        "candidate_decision": result.get("candidate_decision"),
        "eval_group_id": result.get("eval_group_id"),
    }
    entries.append(entry)
    entries.sort(
        key=lambda item: _beam_rank_tuple(
            item.get("metrics", {}),
            float(item.get("wall_time", 0.0) or 0.0),
        )
    )
    state[suite] = entries[:beam_width]
    if not dry_run:
        _save_beam_state(state)
        _append_jsonl(BEAM_LOG, entry)
    return entry


def _append_log(record: dict) -> None:
    _append_jsonl(EXPERIMENT_LOG, record)


def _read_best_candidate(
    suite: str | None = None,
    workflow: str | None = None,
) -> dict | None:
    if not EXPERIMENT_LOG.exists():
        return None
    best = None
    with EXPERIMENT_LOG.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("decision") != "keep":
                continue
            if suite is not None and record.get("suite") != suite:
                continue
            if workflow is not None and record.get("workflow") != workflow:
                continue
            best = {
                "train_suite_average_gap": record.get("train_suite_average_gap"),
                "dev_suite_average_gap": record.get("dev_suite_average_gap"),
            }
    return best


def _active_beam_entries(suite: str) -> list[dict]:
    state = _load_beam_state()
    entries = [entry for entry in state.get(suite, []) if isinstance(entry, dict)]
    entries.sort(
        key=lambda item: _beam_rank_tuple(
            item.get("metrics", {}),
            float(item.get("wall_time", 0.0) or 0.0),
        )
    )
    return entries


def _next_experiment_number() -> int:
    if not EXPERIMENT_LOG.exists():
        return 0
    count = 0
    with EXPERIMENT_LOG.open("r") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _run_beam_promotion(
    suite: str,
    max_attempts: int,
    timeout: int,
    prompt_variant: str,
    top_k: int,
    dry_run: bool,
    restore_best: bool,
) -> int:
    beam = _active_beam_entries(suite)
    if not beam:
        print(f"No beam entries found for suite '{suite}'.")
        return 1

    selected = beam[:max(1, top_k)]
    print(f"\n{'=' * 70}")
    print(f"  PROMOTION RUN  |  suite={suite}  |  beam_candidates={len(selected)}")
    print(f"{'=' * 70}\n")

    promotion_run_id = f"{suite}-{int(time.time() * 1000)}"
    confirmed: list[dict] = []
    for rank, entry in enumerate(selected, start=1):
        snapshot_path = Path(str(entry.get("snapshot_path", "")))
        if not snapshot_path.exists():
            print(f"  [{rank}] missing snapshot: {snapshot_path}")
            continue
        print(
            f"  [{rank}/{len(selected)}] confirming exp#{entry.get('experiment_number')} "
            f"train_gap={entry.get('metrics', {}).get('train_suite_average_gap')} "
            f"snapshot={snapshot_path}"
        )
        result = _run_policy_evaluation(
            suite=suite,
            workflow="confirm",
            max_attempts=max_attempts,
            timeout=timeout,
            prompt_variant=prompt_variant,
            experiment_file=snapshot_path,
            no_artifacts=dry_run,
        )
        metrics = _suite_gap_metrics(result)
        confirm_record = {
            "record_type": "candidate",
            "promotion_run_id": promotion_run_id,
            "suite": suite,
            "timestamp": datetime.now().isoformat(),
            "prompt_variant": prompt_variant,
            "beam_entry": entry,
            "confirm_result": result,
            "confirm_metrics": metrics,
            "confirm_rank_tuple": list(
                _beam_rank_tuple(metrics, float(result.get("wall_time", 0.0) or 0.0))
            ) if metrics else None,
        }
        confirmed.append(confirm_record)
        if not dry_run:
            _append_jsonl(PROMOTION_LOG, confirm_record)

    successful = [record for record in confirmed if record.get("confirm_metrics")]
    if not successful:
        print("No successful confirm evaluations completed.")
        return 1

    successful.sort(
        key=lambda item: _beam_rank_tuple(
            item.get("confirm_metrics", {}),
            float(item.get("confirm_result", {}).get("wall_time", 0.0) or 0.0),
        )
    )
    best = successful[0]
    best_snapshot = Path(best["beam_entry"]["snapshot_path"])
    best_metrics = best["confirm_metrics"]
    print(f"\nBest confirmed snapshot: {best_snapshot}")
    print(f"  train_suite_average_gap: {best_metrics.get('train_suite_average_gap')}")
    for key, value in sorted(best_metrics.items()):
        if key == "train_suite_average_gap":
            continue
        print(f"  {key}: {value}")

    if restore_best and not dry_run:
        EXPERIMENT_FILE.write_text(best_snapshot.read_text())
        print(f"\nRestored best confirmed snapshot into {EXPERIMENT_FILE}")

    summary_record = {
        "record_type": "summary",
        "promotion_run_id": promotion_run_id,
        "suite": suite,
        "timestamp": datetime.now().isoformat(),
        "prompt_variant": prompt_variant,
        "confirmed_count": len(successful),
        "best_snapshot_path": str(best_snapshot),
        "best_confirm_eval_group_id": best.get("confirm_result", {}).get("eval_group_id"),
        "best_confirm_metrics": best_metrics,
        "best_beam_metrics": best.get("beam_entry", {}).get("metrics", {}),
        "restored_best": bool(restore_best and not dry_run),
        "restored_to": str(EXPERIMENT_FILE) if restore_best and not dry_run else None,
    }
    if not dry_run:
        _append_jsonl(PROMOTION_LOG, summary_record)

    return 0


def run_single(
    suite: str,
    workflow: str,
    max_attempts: int,
    timeout: int,
    prompt_variant: str,
    experiment_number: int,
    best_candidate: dict | None,
    dry_run: bool,
    beam_width: int,
    no_dev: bool = False,
) -> dict:
    best_train = None if best_candidate is None else best_candidate.get("train_suite_average_gap")
    best_dev = None if best_candidate is None else best_candidate.get("dev_suite_average_gap")
    proposed_source = EXPERIMENT_FILE.read_text()
    proposed_diff = _git_diff_text()
    proposed_diff_path = None if dry_run else _archive_diff(experiment_number, proposed_diff)
    head_before = None if dry_run else _git_head()

    print(f"\n{'=' * 70}")
    print(
        f"  Experiment #{experiment_number}  |  suite={suite}  |  workflow={workflow}  |  "
        f"best_train={best_train}  |  best_dev={best_dev}"
    )
    print(f"{'=' * 70}\n")

    result = _run_policy_evaluation(
        suite=suite,
        workflow=workflow,
        max_attempts=max_attempts,
        timeout=timeout,
        prompt_variant=prompt_variant,
        no_dev=no_dev,
        no_artifacts=dry_run,
    )
    acceptance_rule = (
        "proxy_primary_with_replay_guardrails"
        if workflow == "scout"
        else "train_primary_with_replay_guardrails"
        if workflow == "candidate" and suite.startswith("mis_curriculum_")
        else "train_primary_with_dev_guardrail"
    )

    record = {
        "experiment_number": experiment_number,
        "timestamp": datetime.now().isoformat(),
        "suite": suite,
        "workflow": workflow,
        "prompt_variant": prompt_variant,
        "acceptance_rule": acceptance_rule,
        "git_head_before": head_before,
        "experiment_file": str(EXPERIMENT_FILE),
        "proposed_source_sha256": _sha256_text(proposed_source),
        "proposed_diff": proposed_diff or None,
        "proposed_diff_path": proposed_diff_path,
        "proposed_diff_line_count": len(proposed_diff.splitlines()) if proposed_diff else 0,
        **result,
    }
    beam_entry = _maybe_add_to_beam(
        suite=suite,
        workflow=workflow,
        experiment_number=experiment_number,
        prompt_variant=prompt_variant,
        result=result,
        beam_width=beam_width,
        dry_run=dry_run,
    )
    if beam_entry is not None:
        record["beam_entry"] = beam_entry

    if result["status"] == "crash":
        print("\n  CRASH during suite evaluation. Reverting.\n")
        record["decision"] = "revert"
        if not dry_run:
            _git_revert()
            record["git_head_after"] = _git_head()
    elif result.get("candidate_decision") == "keep":
        train_gap = result.get("train_suite_average_gap")
        dev_gap = result.get("dev_suite_average_gap")
        _train_str = f"{train_gap:.6f}" if train_gap is not None else "n/a"
        _dev_str = f"{dev_gap:.6f}" if dev_gap is not None else "n/a"
        print(f"\n  KEEP: train_gap={_train_str}  dev_gap={_dev_str}\n")
        record["decision"] = "keep"
        if not dry_run and _git_has_changes():
            message = (
                f"experiment {experiment_number}: "
                f"train_gap {_train_str} "
                f"dev_gap {_dev_str} "
                f"suite={suite} "
                f"workflow={workflow}"
            )
            record["commit_message"] = message
            if _git_commit(message):
                record["git_head_after"] = _git_head()
        elif not dry_run:
            record["git_head_after"] = _git_head()
    else:
        train_gap = result.get("train_suite_average_gap")
        dev_gap = result.get("dev_suite_average_gap")
        print(
            "\n  DISCARD: "
            f"train_gap={train_gap}  "
            f"dev_gap={dev_gap}  "
            f"decision={result.get('candidate_decision')}. Reverting.\n"
        )
        record["decision"] = "revert"
        if not dry_run:
            _git_revert()
            record["git_head_after"] = _git_head()

    if not dry_run:
        _append_log(record)
    else:
        record["decision"] = "dry_run"
        print("  (dry-run: no log entry written, no git changes)\n")
    return record


def run_loop(
    suite: str,
    workflow: str,
    max_experiments: int,
    max_attempts: int,
    timeout: int,
    prompt_variant: str,
    branch: str,
    dry_run: bool,
    beam_width: int,
    wall_clock_budget: int | None = None,
    no_dev: bool = False,
) -> None:
    if not dry_run:
        _ensure_branch(branch)

    best_candidate = _read_best_candidate(suite=suite, workflow=workflow)
    print(f"Starting experiment loop on branch '{branch}'")
    print(f"Best kept candidate: {best_candidate}")
    print(f"Suite: {suite}")
    print(f"Workflow: {workflow}")
    print(f"Prompt variant: {prompt_variant}")
    print(f"Max experiments: {max_experiments}")
    if wall_clock_budget is not None:
        print(f"Wall-clock budget: {wall_clock_budget}s")

    t_loop_start = time.time()
    next_experiment_number = _next_experiment_number()
    for offset in range(max_experiments):
        if wall_clock_budget is not None and (time.time() - t_loop_start) >= wall_clock_budget:
            print("\nWall-clock budget exhausted. Stopping search loop.")
            break
        record = run_single(
            suite=suite,
            workflow=workflow,
            max_attempts=max_attempts,
            timeout=timeout,
            prompt_variant=prompt_variant,
            experiment_number=next_experiment_number + offset,
            best_candidate=best_candidate,
            dry_run=dry_run,
            beam_width=beam_width,
            no_dev=no_dev,
        )
        if record.get("decision") == "keep":
            best_candidate = {
                "train_suite_average_gap": record.get("train_suite_average_gap"),
                "dev_suite_average_gap": record.get("dev_suite_average_gap"),
            }

    print(f"\nLoop complete. Best kept candidate: {best_candidate}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoQResearch agent harness — guarded keep/revert on suite_average_gap",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="mis_curriculum_16",
        choices=(
            "single_mis32",
            "generalize_mis",
            "mis_probe_16",
            "mis_validate_64",
            "mis_curriculum_16",
            "mis_curriculum_32",
            "mis_curriculum_48",
            "mis_curriculum_64",
            "single20",
            "quick",
            "standard",
            "full",
            "generalize",
        ),
        help="Suite for candidate evaluation (default: mis_curriculum_16 = curriculum stage 1)",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="full",
        help="Prompt variant label recorded in evaluation logs",
    )
    parser.add_argument(
        "--eval-workflow",
        type=str,
        default="scout",
        choices=("scout", "candidate"),
        help="Search workflow for iterative runs (default: scout)",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=100,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max attempts per instance within experiment.py",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per candidate suite evaluation in seconds",
    )
    parser.add_argument(
        "--wall-clock-budget",
        type=int,
        default=None,
        help="Optional wall-clock budget in seconds for the search loop",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Maximum number of scout candidates to retain in the beam",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="codex/mis01",
        help="Git branch for this experiment series",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run one candidate suite evaluation and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not git commit/revert",
    )
    parser.add_argument(
        "--promote-beam",
        action="store_true",
        help="Run confirm evaluation on the current scout beam for this suite",
    )
    parser.add_argument(
        "--promote-top-k",
        type=int,
        default=3,
        help="How many beam entries to confirm during promotion",
    )
    parser.add_argument(
        "--restore-best",
        action="store_true",
        help="After promotion, restore the best confirmed snapshot into experiment.py",
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="Skip dev evaluation (single-instance mode, no dev guardrail)",
    )
    args = parser.parse_args()

    if args.promote_beam:
        return _run_beam_promotion(
            suite=args.suite,
            max_attempts=args.max_attempts,
            timeout=args.timeout,
            prompt_variant=args.prompt_variant,
            top_k=args.promote_top_k,
            dry_run=args.dry_run,
            restore_best=args.restore_best,
        )

    if args.single_run:
        # ── Auto-baseline: if no log exists, run the static conservative baseline as #0 ──
        if not EXPERIMENT_LOG.exists() and not args.dry_run:
            print("\n" + "=" * 70)
            print("  BASELINE RUN (experiment #0) — static conservative baseline")
            print("  This establishes the starting point for improvement.")
            print("=" * 70 + "\n")
            baseline_result = _run_policy_evaluation(
                suite=args.suite,
                workflow=args.eval_workflow,
                max_attempts=args.max_attempts,
                timeout=args.timeout,
                prompt_variant=args.prompt_variant,
                no_dev=args.no_dev,
                baseline=True,
            )
            baseline_record = {
                "experiment_number": 0,
                "timestamp": datetime.now().isoformat(),
                "suite": args.suite,
                "workflow": args.eval_workflow,
                "prompt_variant": args.prompt_variant,
                "acceptance_rule": "baseline",
                "git_head_before": _git_head(),
                "experiment_file": str(EXPERIMENT_FILE),
                "proposed_source_sha256": _sha256_text(EXPERIMENT_FILE.read_text()),
                "proposed_diff": None,
                "proposed_diff_path": None,
                "proposed_diff_line_count": 0,
                "is_baseline": True,
                **baseline_result,
            }
            beam_entry = _maybe_add_to_beam(
                suite=args.suite,
                workflow=args.eval_workflow,
                experiment_number=0,
                prompt_variant=args.prompt_variant,
                result=baseline_result,
                beam_width=args.beam_width,
                dry_run=args.dry_run,
            )
            if beam_entry is not None:
                baseline_record["beam_entry"] = beam_entry
            baseline_record["decision"] = "keep"
            _append_log(baseline_record)
            train_gap = baseline_result.get("train_suite_average_gap")
            _tg = f"{train_gap:.6f}" if train_gap is not None else "n/a"
            print(f"\n  BASELINE recorded: suite_average_gap={_tg}")
            print("  This is now the incumbent for KEEP/DISCARD decisions.\n")
            return 0

        best_candidate = _read_best_candidate(suite=args.suite, workflow=args.eval_workflow)
        experiment_number = _next_experiment_number()
        record = run_single(
            suite=args.suite,
            workflow=args.eval_workflow,
            max_attempts=args.max_attempts,
            timeout=args.timeout,
            prompt_variant=args.prompt_variant,
            experiment_number=experiment_number,
            best_candidate=best_candidate,
            dry_run=args.dry_run,
            beam_width=args.beam_width,
            no_dev=args.no_dev,
        )
        return 0 if record.get("status") != "crash" else 1

    run_loop(
        suite=args.suite,
        workflow=args.eval_workflow,
        max_experiments=args.max_experiments,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        prompt_variant=args.prompt_variant,
        branch=args.branch,
        dry_run=args.dry_run,
        beam_width=args.beam_width,
        wall_clock_budget=args.wall_clock_budget,
        no_dev=args.no_dev,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
