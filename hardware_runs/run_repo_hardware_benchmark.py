#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _arg_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _has_placeholder_credentials(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    placeholders = (
        "YOUR_TOKEN_",
        "YOUR_CRN_",
        "YOUR_INSTANCE_",
    )
    return any(marker in text for marker in placeholders)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    repo_root = script_dir / "quantum-optimization-benchmarks"
    upstream_entrypoint = repo_root / "research_benchmark" / "run_hardware_benchmark.py"
    default_python = workspace_root / ".venv" / "bin" / "python"
    python_bin = Path(os.environ.get("PYTHON_BIN", str(default_python))).expanduser()

    passthrough = sys.argv[1:]
    if not passthrough or "--help-wrapper" in passthrough:
        print(
            "Usage:\n"
            "  hardware_runs/run_repo_hardware_benchmark.py "
            "--problem <problem> --method <method> [upstream args...]\n\n"
            "Wrapper behavior:\n"
            "  - uses the cloned benchmark repo under hardware_runs/quantum-optimization-benchmarks\n"
            "  - writes outputs to hardware_runs/results_hardware\n"
            "  - writes checkpoints to hardware_runs/checkpoints/<method>\n"
            "  - defaults to hardware_runs/ibm_credentials.template.json\n"
            "  - forces --ibm-min-runtime-seconds 60\n\n"
            "Override the Python interpreter with PYTHON_BIN=/path/to/python.\n"
        )
        return 0

    if not upstream_entrypoint.is_file():
        print(f"Missing upstream entrypoint: {upstream_entrypoint}", file=sys.stderr)
        return 2
    if not python_bin.is_file():
        print(f"Python interpreter not found: {python_bin}", file=sys.stderr)
        return 2

    method = _arg_value(passthrough, "--method")
    if not method:
        print("Missing required upstream flag: --method", file=sys.stderr)
        return 2

    creds_value = _arg_value(passthrough, "--ibm-credentials-json")
    if creds_value is None:
        creds_path = script_dir / "ibm_credentials.template.json"
        passthrough.extend(["--ibm-credentials-json", str(creds_path)])
    else:
        creds_path = Path(creds_value).expanduser().resolve()

    if creds_path.is_file() and _has_placeholder_credentials(creds_path):
        print(
            f"Credential file still contains placeholders: {creds_path}\n"
            "Populate it with real IBM token/CRN pairs before running.",
            file=sys.stderr,
        )
        return 2

    output_root = script_dir / "results_hardware"
    checkpoint_dir = script_dir / "checkpoints" / method
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    forwarded = [
        str(python_bin),
        str(upstream_entrypoint),
        *passthrough,
        "--project-root",
        str(repo_root),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--ibm-min-runtime-seconds",
        os.environ.get("IBM_MIN_RUNTIME_SECONDS", "60"),
    ]

    print("Wrapper command:")
    print(" ".join(forwarded))
    print()
    return subprocess.call(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
