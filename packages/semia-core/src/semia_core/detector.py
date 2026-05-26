# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 RiemaLabs
"""Detector dispatcher for Semia SDL rules.

Two backends are supported, in priority order:

1. Soufflé, when its binary is available (configured by ``SEMIA_SOUFFLE_BIN``
   or found on ``PATH``). The core package never downloads or bundles Soufflé.
2. A built-in pure-Python Datalog evaluator (``semia_core.datalog_eval``) that
   covers the surface used by the SDL rules. This is the default when Soufflé
   is not present so users can run audits without an external install.

Override the backend selection with ``SEMIA_DETECTOR_BACKEND``:

- ``auto`` (default): Soufflé if present, else built-in.
- ``souffle``: Soufflé only; report ``unavailable`` if missing.
- ``builtin``: built-in evaluator only.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

from .artifacts import DetectorResult, Finding
from .datalog_eval import EvalResult, ParseError, run_evaluator
from .datalog_eval.engine import EvalError


def find_souffle_binary() -> str | None:
    """Return the configured/available Souffle binary, if any."""

    configured = os.environ.get("SEMIA_SOUFFLE_BIN")
    if configured:
        path = Path(configured).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which("souffle")


def run_detector(
    facts_path: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: float | None = 120,
) -> DetectorResult:
    """Run the SDL detector over an assembled facts/rules file."""

    backend = os.environ.get("SEMIA_DETECTOR_BACKEND", "auto").strip().lower() or "auto"
    if backend not in {"auto", "souffle", "builtin"}:
        return DetectorResult(
            status="failed",
            backend="none",
            message=f"unknown SEMIA_DETECTOR_BACKEND={backend!r}",
        )

    if backend in ("auto", "souffle"):
        souffle_bin = find_souffle_binary()
        if souffle_bin is not None:
            return _run_souffle(souffle_bin, facts_path, output_dir, timeout_seconds)
        if backend == "souffle":
            return DetectorResult(
                status="unavailable",
                backend="souffle",
                message="Souffle binary not found; set SEMIA_SOUFFLE_BIN or install souffle on PATH.",
            )

    return _run_builtin(facts_path, output_dir, timeout_seconds)


def _run_souffle(
    souffle_bin: str,
    facts_path: str | Path,
    output_dir: str | Path,
    timeout_seconds: float | None,
) -> DetectorResult:
    facts = Path(facts_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [souffle_bin, str(facts), "-D", str(out)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=str(facts.parent),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DetectorResult(status="failed", backend="souffle", message=str(exc), output_dir=out)

    if result.returncode != 0:
        return DetectorResult(
            status="failed",
            backend="souffle",
            stdout=result.stdout,
            stderr=result.stderr,
            message=f"Souffle exited with {result.returncode}",
            output_dir=out,
        )

    return DetectorResult(
        status="ok",
        backend="souffle",
        findings=tuple(_read_findings(out)),
        stdout=result.stdout,
        stderr=result.stderr,
        output_dir=out,
    )


def _run_builtin(
    facts_path: str | Path,
    output_dir: str | Path,
    timeout_seconds: float | None,
) -> DetectorResult:
    facts = Path(facts_path)
    out = Path(output_dir)
    try:
        result: EvalResult = run_evaluator(facts, out, timeout_seconds=timeout_seconds)
    except (ParseError, EvalError, FileNotFoundError, OSError) as exc:
        return DetectorResult(
            status="failed",
            backend="builtin",
            message=f"builtin evaluator: {exc}",
            output_dir=out,
        )

    return DetectorResult(
        status="ok",
        backend="builtin",
        findings=tuple(_read_findings(out)),
        message=f"built-in evaluator ran {len(result.output_files)} output relation(s)",
        output_dir=out,
    )


def _read_findings(output_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for csv_path in sorted(output_dir.glob("label_*.csv")):
        label = csv_path.stem
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                findings.append(Finding(label=label, fields=tuple(row)))
    return findings
