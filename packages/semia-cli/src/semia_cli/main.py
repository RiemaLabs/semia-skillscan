# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 RiemaLabs
"""Argparse entry point for the Semia CLI MVP."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from . import core_adapter, llm_adapter
from .core_adapter import CoreApiError
from .llm_adapter import LlmSynthesisError, SynthesisSettings
from .synthesis_patch import apply_incremental_patch, parse_incremental_diff

SYNTHESIZED_FACTS = "synthesized_facts.dl"
# Mirrors of the synthesis-loop defaults exposed for `semia synthesis-status` so
# the reported stop criteria match what `synthesize_facts` actually uses. Keep
# in sync with llm_config.DEFAULT_SYNTHESIS_CEILING / plateau defaults.
SYNTHESIS_PLATEAU_CEILING = 0.9
SYNTHESIS_PLATEAU_PATIENCE_DEFAULT = 3
SYNTHESIS_PLATEAU_MIN_IMPROVEMENT_DEFAULT = 0.01


def _get_version() -> str:
    for distribution_name in ("semia-audit", "semia"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.1.2+unknown"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stdout = getattr(args, "_stdout", sys.stdout)
    stderr = getattr(args, "_stderr", sys.stderr)

    try:
        args.handler(args, stdout)
    except CoreApiError as exc:
        print(f"semia: {exc}", file=stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"semia: {exc}", file=stderr)
        return 2
    except LlmSynthesisError as exc:
        print(f"semia: {exc}", file=stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semia",
        description="Semia Skill Behavior Mapping audit CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="prepare a Semia run directory from a skill source",
    )
    prepare_parser.add_argument("skill_path", type=Path)
    prepare_parser.add_argument(
        "--out",
        dest="run_dir",
        type=Path,
        help="run directory (default: .semia/runs/<skill-slug>, where slug is "
        "the skill directory name or the skill file stem)",
    )
    prepare_parser.set_defaults(handler=_prepare)

    synthesize_parser = subparsers.add_parser(
        "synthesize",
        help="build and validate the skill behavior map",
    )
    synthesize_parser.add_argument("run_dir", type=Path)
    synthesize_parser.add_argument("--facts", dest="facts_path", type=Path)
    synthesize_parser.add_argument(
        "--apply-patch",
        dest="patch_path",
        type=Path,
        help="apply an incremental Datalog patch (REPLACE/REMOVE/add) to "
        "synthesized_facts.dl deterministically, then validate. No LLM call.",
    )
    _add_llm_options(synthesize_parser)
    _add_host_metadata_options(synthesize_parser)
    _add_taint_threshold_option(synthesize_parser)
    synthesize_parser.set_defaults(handler=_synthesize)

    status_parser = subparsers.add_parser(
        "synthesis-status",
        help="report scoring, plateau, and next-step diagnostics for an existing run "
        "(read-only, no LLM call)",
    )
    status_parser.add_argument("run_dir", type=Path)
    status_parser.set_defaults(handler=_synthesis_status)

    detect_parser = subparsers.add_parser(
        "detect",
        help="run deterministic Semia detectors for a prepared run",
    )
    detect_parser.add_argument("run_dir", type=Path)
    detect_parser.set_defaults(handler=_detect)

    report_parser = subparsers.add_parser(
        "report",
        help="render a Semia audit report",
    )
    report_parser.add_argument("run_dir", type=Path)
    report_parser.add_argument("--format", choices=("md", "json", "sarif"), required=True)
    report_parser.set_defaults(handler=_report)

    scan_parser = subparsers.add_parser(
        "scan",
        help="prepare, synthesize, detect, and render a report",
    )
    scan_parser.add_argument("skill_path", type=Path)
    scan_parser.add_argument(
        "--out",
        dest="run_dir",
        type=Path,
        help="run directory (default: .semia/runs/<skill-slug>, where slug is "
        "the skill directory name or the skill file stem)",
    )
    scan_parser.add_argument(
        "--facts",
        dest="facts_path",
        type=Path,
        help="existing synthesized facts to copy into the run before detect/report",
    )
    _add_llm_options(scan_parser)
    _add_host_metadata_options(scan_parser)
    _add_taint_threshold_option(scan_parser)
    scan_parser.add_argument(
        "--offline-baseline",
        action="store_true",
        help="use a conservative non-LLM fallback instead of calling synthesize",
    )
    scan_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after prepare and print synthesis guidance",
    )
    scan_parser.add_argument(
        "--no-recommendation",
        action="store_true",
        help="skip the final LLM recommendation step (saves one LLM call)",
    )
    scan_parser.set_defaults(handler=_scan)

    repair_parser = subparsers.add_parser(
        "repair",
        help="generate SKILL.md patches to fix detected violations",
    )
    repair_parser.add_argument(
        "target",
        type=Path,
        help="either a skill directory (runs scan first) or an existing run directory "
        "(with --from-scan)",
    )
    repair_parser.add_argument(
        "--from-scan",
        action="store_true",
        help="treat target as an existing run directory (skip scan)",
    )
    repair_parser.add_argument(
        "--trace-only",
        action="store_true",
        help="trace findings to source but do not generate patches",
    )
    _add_llm_options(repair_parser)
    repair_parser.set_defaults(handler=_repair)

    return parser


def _prepare(args: argparse.Namespace, stdout: TextIO) -> None:
    skill_path = _existing_path(args.skill_path, "skill_path")
    run_dir = _resolve_run_dir(args.run_dir, skill_path, stdout)
    result = core_adapter.prepare(skill_path, run_dir)
    _print_result(stdout, result, fallback=f"Prepared Semia run at {run_dir}")


def _synthesize(args: argparse.Namespace, stdout: TextIO) -> None:
    run_dir = _existing_path(args.run_dir, "run_dir")
    target = run_dir / SYNTHESIZED_FACTS
    patch_path = getattr(args, "patch_path", None)
    if patch_path is not None:
        if args.facts_path is not None:
            raise CoreApiError("--apply-patch and --facts are mutually exclusive")
        if not target.exists():
            raise FileNotFoundError(
                f"cannot apply patch; {target} does not exist yet. "
                "Write an initial synthesized_facts.dl first."
            )
        resolved_patch = _existing_path(patch_path, "patch_path")
        diff = parse_incremental_diff(resolved_patch.read_text(encoding="utf-8"))
        if diff is None:
            raise CoreApiError(
                f"patch at {resolved_patch} contains no REPLACE/REMOVE directives or "
                "additions; refusing to apply a no-op patch"
            )
        current = target.read_text(encoding="utf-8")
        _atomic_write_text(target, apply_incremental_patch(current, diff))
        print(f"Applied incremental patch from {resolved_patch} to {target}", file=stdout)
        validation_path = target
    elif args.facts_path is not None:
        facts_path = _existing_path(args.facts_path, "facts_path")
        if facts_path != target:
            shutil.copyfile(facts_path, target)
        validation_path = target
    else:
        stderr = getattr(args, "_stderr", sys.stderr)
        synth_kwargs: dict[str, Any] = {
            "provider": args.provider,
            "model": args.model,
            "base_url": getattr(args, "base_url", None),
            "validator": core_adapter.check,
        }
        progress = _make_progress_callback(stderr)
        if progress is not None:
            synth_kwargs["on_progress"] = progress
        result = llm_adapter.synthesize_facts(run_dir, **synth_kwargs)
        _print_result(stdout, result, fallback=f"Synthesized behavior map for {run_dir}")
        validation_path = target
    result = core_adapter.check(
        run_dir,
        validation_path,
        host_session_id=getattr(args, "host_session_id", None),
        host_model=getattr(args, "host_model", None),
        evidence_taint_threshold=getattr(args, "evidence_taint_threshold", None),
    )
    _print_result(stdout, result, fallback=f"Synthesized behavior map for {run_dir}")


def _detect(args: argparse.Namespace, stdout: TextIO) -> None:
    run_dir = _existing_path(args.run_dir, "run_dir")
    result = core_adapter.detect(run_dir)
    _print_result(stdout, result, fallback=f"Ran detectors for {run_dir}")


def _report(args: argparse.Namespace, stdout: TextIO) -> None:
    run_dir = _existing_path(args.run_dir, "run_dir")
    result = core_adapter.report(run_dir, args.format)
    _print_result(stdout, result, fallback=f"Rendered {args.format} report for {run_dir}")


def _scan(args: argparse.Namespace, stdout: TextIO) -> None:
    skill_path = _existing_path(args.skill_path, "skill_path")
    run_dir = _resolve_run_dir(args.run_dir, skill_path, stdout)
    result = core_adapter.prepare(skill_path, run_dir)
    _print_result(stdout, result, fallback=f"Prepared Semia run at {run_dir}")
    if args.prepare_only:
        print("", file=stdout)
        print(
            "Next step: use your current agent session to synthesize the behavior map.", file=stdout
        )
        print(f"Write the synthesized facts into: {run_dir / SYNTHESIZED_FACTS}", file=stdout)
        print(f"Then run: semia synthesize {run_dir}", file=stdout)
        print(f"Then run: semia detect {run_dir}", file=stdout)
        print(f"Then run: semia report {run_dir} --format md", file=stdout)
        return
    if args.facts_path is not None:
        facts_path = _existing_path(args.facts_path, "facts_path")
        target = run_dir / SYNTHESIZED_FACTS
        if facts_path != target:
            shutil.copyfile(facts_path, target)
        print("", file=stdout)
        print(f"Copied synthesized facts into: {target}", file=stdout)
    elif not (run_dir / SYNTHESIZED_FACTS).exists():
        print("", file=stdout)
        if args.offline_baseline:
            print(
                "No synthesized facts supplied; using a conservative offline baseline map.",
                file=stdout,
            )
            _print_result(
                stdout,
                core_adapter.extract_baseline(run_dir),
                fallback=f"Wrote baseline behavior map for {run_dir}",
            )
        else:
            provider = llm_adapter.default_provider(args.provider)
            model = llm_adapter.default_model(args.model, provider)
            base_url = llm_adapter.default_base_url(getattr(args, "base_url", None), provider)
            print(
                f"No synthesized facts supplied; running synthesize with provider `{provider}`.",
                file=stdout,
            )
            if model:
                print(f"Using model `{model}`.", file=stdout)
            else:
                print("Using the provider's configured default model.", file=stdout)
            if base_url:
                print(f"Using base URL `{base_url}`.", file=stdout)
            stderr = getattr(args, "_stderr", sys.stderr)
            synth_kwargs: dict[str, Any] = {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "validator": core_adapter.check,
            }
            progress = _make_progress_callback(stderr)
            if progress is not None:
                synth_kwargs["on_progress"] = progress
            _print_result(
                stdout,
                llm_adapter.synthesize_facts(run_dir, **synth_kwargs),
                fallback=f"Synthesized behavior map for {run_dir}",
            )

    _print_result(
        stdout,
        core_adapter.check(
            run_dir,
            run_dir / SYNTHESIZED_FACTS,
            host_session_id=getattr(args, "host_session_id", None),
            host_model=getattr(args, "host_model", None),
            evidence_taint_threshold=getattr(args, "evidence_taint_threshold", None),
        ),
        fallback=f"Validated synthesized facts for {run_dir}",
    )
    _print_result(stdout, core_adapter.detect(run_dir), fallback=f"Ran detectors for {run_dir}")
    report = core_adapter.report(run_dir, "md")
    if isinstance(report, str):
        print(report, file=stdout)
    else:
        _print_result(stdout, report, fallback=f"Rendered report for {run_dir}")
    if not args.no_recommendation and not args.offline_baseline:
        _run_recommendation(args, run_dir, stdout)


def _run_recommendation(args: argparse.Namespace, run_dir: Path, stdout: TextIO) -> None:
    """Final LLM pass: produce a plain-English verdict from skill + report.

    Failures are reported on stderr but never abort the scan — the
    deterministic findings already on disk are the source of truth; the
    recommendation is a convenience layer on top.
    """

    from . import recommendation

    stderr = getattr(args, "_stderr", sys.stderr)
    try:
        result = recommendation.recommend(
            run_dir,
            provider=args.provider,
            model=args.model,
            base_url=getattr(args, "base_url", None),
        )
    except (LlmSynthesisError, FileNotFoundError, CoreApiError) as exc:
        print(f"semia: recommendation skipped — {exc}", file=stderr)
        return
    print("", file=stdout)
    print("## Recommendation\n", file=stdout)
    try:
        print(Path(result["recommendation"]).read_text(encoding="utf-8"), file=stdout)
    except OSError:
        _print_result(stdout, result, fallback=f"Wrote recommendation for {run_dir}")


def _repair(args: argparse.Namespace, stdout: TextIO) -> None:
    from . import repair as repair_mod

    target = _existing_path(args.target, "target")

    if args.from_scan:
        # Target is an existing run directory
        run_dir = target
    else:
        # Target is a skill directory — run scan first
        skill_path = target
        run_dir = _resolve_run_dir(None, skill_path, stdout)
        print(f"\nScanning {skill_path}...", file=stdout)
        result = core_adapter.prepare(skill_path, run_dir)
        _print_result(stdout, result, fallback=f"Prepared at {run_dir}")

        # Synthesize
        if not (run_dir / SYNTHESIZED_FACTS).exists():
            stderr = getattr(args, "_stderr", sys.stderr)
            synth_kwargs: dict[str, Any] = {
                "provider": args.provider,
                "model": args.model,
                "base_url": getattr(args, "base_url", None),
                "validator": core_adapter.check,
            }
            progress = _make_progress_callback(stderr)
            if progress is not None:
                synth_kwargs["on_progress"] = progress
            _print_result(
                stdout,
                llm_adapter.synthesize_facts(run_dir, **synth_kwargs),
                fallback=f"Synthesized for {run_dir}",
            )

        # Check + detect
        _print_result(
            stdout,
            core_adapter.check(run_dir, run_dir / SYNTHESIZED_FACTS),
            fallback=f"Validated for {run_dir}",
        )
        _print_result(stdout, core_adapter.detect(run_dir), fallback=f"Detected for {run_dir}")

    _write_repair_report(run_dir, stdout)

    # Run repair
    print("", file=stdout)
    result = repair_mod.repair(
        run_dir,
        provider=args.provider,
        model=args.model,
        base_url=getattr(args, "base_url", None),
        trace_only=args.trace_only,
        stdout=stdout,
    )
    _print_result(stdout, result, fallback="Repair complete")


def _write_repair_report(run_dir: Path, stdout: TextIO) -> None:
    core_adapter.report(run_dir, "md")
    print(f"Report: {run_dir / 'report.md'}", file=stdout)


def _existing_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_run_dir(run_dir: Path | None, skill_path: Path, stdout: TextIO) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    slug = skill_path.name if skill_path.is_dir() else skill_path.stem
    default = (Path(".semia/runs") / slug).resolve()
    print(f"Using default run dir: {default}", file=stdout)
    return default


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a temp file + ``os.replace``.

    Mirrors :func:`semia_cli.synthesis_loop._atomic_write` so a crash mid-write
    cannot leave ``synthesized_facts.dl`` truncated when ``--apply-patch`` is
    used.
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=("responses", "anthropic", "codex", "claude", "openai"),
        help="LLM provider for synthesize; one of: responses (OpenAI Responses "
        "API, default), anthropic (Anthropic Messages API), codex (codex CLI), "
        "claude (Claude Code CLI). `openai` is an alias for `responses`. "
        "Default: SEMIA_LLM_PROVIDER or responses.",
    )
    parser.add_argument(
        "--model",
        help="model name passed to the provider (free-form); default: "
        "SEMIA_LLM_MODEL, or gpt-5.5 (responses), claude-opus-4-7 "
        "(anthropic / claude), provider default (codex)",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        help="HTTP base URL for the responses or anthropic provider; default: "
        "OPENAI_BASE_URL (responses) or ANTHROPIC_BASE_URL (anthropic); "
        "ignored for codex / claude.",
    )


def _add_host_metadata_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host-session-id",
        dest="host_session_id",
        help="record the calling agent session id in run_manifest.json (plugin "
        "mode reproducibility)",
    )
    parser.add_argument(
        "--host-model",
        dest="host_model",
        help="record the calling agent's model id in run_manifest.json (plugin "
        "mode reproducibility)",
    )


def _add_taint_threshold_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evidence-taint-threshold",
        dest="evidence_taint_threshold",
        type=float,
        help="reject the program when evidence_match_rate < THRESHOLD (in [0, 1]). "
        "Defaults to SEMIA_EVIDENCE_TAINT_THRESHOLD env or 0.1. "
        "Recommended 0.5 for plugin mode where host-session synthesis is harder to bound.",
    )


def _synthesis_status(args: argparse.Namespace, stdout: TextIO) -> None:
    run_dir = _existing_path(args.run_dir, "run_dir")
    check_payload = _read_json_optional(run_dir / "synthesis_check.json")
    alignment_payload = _read_json_optional(run_dir / "synthesis_evidence_alignment.json")
    metadata_payload = _read_json_optional(run_dir / "synthesis_metadata.json")
    manifest_payload = _read_json_optional(run_dir / "run_manifest.json")

    if check_payload is None and alignment_payload is None:
        print(
            f"No synthesis artifacts found in {run_dir}. Run `semia synthesize` first.",
            file=stdout,
        )
        return

    # Settings come from synthesis_metadata.json when synthesize ran via the LLM
    # loop (standalone CLI); plugin-mode runs skip the loop so metadata may be
    # absent, in which case fall back to live SynthesisSettings (env + defaults).
    # Reading from llm_config rather than hard-coding ensures status never
    # diverges from what synthesize actually used.
    effective = _effective_synthesis_settings(metadata_payload)

    match_rate = (alignment_payload or {}).get("evidence_match_rate")
    support = (check_payload or {}).get("evidence_support_coverage")
    reference = (alignment_payload or {}).get("reference_unit_coverage")
    grounding = (alignment_payload or {}).get("grounding_score")
    program_valid = (check_payload or {}).get("program_valid")
    taint_threshold = (check_payload or {}).get("evidence_taint_threshold")

    score = _composite_synthesis_score(match_rate, support, reference, effective["score_weights"])
    ceiling = effective["ceiling"]
    suggestions = _synthesis_suggestions(check_payload, alignment_payload, score, ceiling)

    payload = {
        "run_dir": str(run_dir),
        "program_valid": program_valid,
        "scores": {
            "composite": score,
            "evidence_match_rate": match_rate,
            "evidence_support_coverage": support,
            "reference_unit_coverage": reference,
            "grounding_score": grounding,
            "weights": list(effective["score_weights"]),
        },
        "stop_criteria": {
            "ceiling_score": ceiling,
            "plateau_patience_iterations": effective["plateau_patience"],
            "plateau_min_improvement": effective["plateau_min_improvement"],
            "ceiling_reached": score is not None and score >= ceiling,
        },
        "evidence_taint_threshold": taint_threshold,
        "synthesis_metadata": _synthesis_metadata_summary(metadata_payload),
        "host_synthesis": (manifest_payload or {}).get("host_synthesis"),
        "synthesized_facts_sha256": (manifest_payload or {}).get("synthesized_facts_sha256"),
        "prepared_skill_sha256": (manifest_payload or {}).get("prepared_skill_sha256"),
        "suggestions": suggestions,
    }
    _print_result(stdout, payload, fallback=f"Synthesis status for {run_dir}")


def _effective_synthesis_settings(
    metadata_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve stop-criteria and scoring settings to display.

    Prefers values frozen in ``synthesis_metadata.json`` so historical runs
    report what they actually used. Falls back to the live
    ``SynthesisSettings`` (env-driven) when metadata is absent or partial —
    e.g. plugin-mode runs that go through ``synthesize --facts`` and never
    invoke the LLM loop.
    """

    live = SynthesisSettings.from_env()
    meta = metadata_payload or {}
    weights_raw = meta.get("score_weights")
    if isinstance(weights_raw, list | tuple) and len(weights_raw) == 3:
        try:
            weights: tuple[float, float, float] = (
                float(weights_raw[0]),
                float(weights_raw[1]),
                float(weights_raw[2]),
            )
        except (TypeError, ValueError):
            weights = live.score_weights
    else:
        weights = live.score_weights
    return {
        "ceiling": float(meta.get("ceiling", live.ceiling)),
        "plateau_patience": int(meta.get("plateau_patience", live.plateau_patience)),
        "plateau_min_improvement": float(
            meta.get("plateau_min_improvement", live.plateau_min_improvement)
        ),
        "score_weights": weights,
    }


def _composite_synthesis_score(
    match_rate: float | None,
    support: float | None,
    reference: float | None,
    weights: tuple[float, float, float],
) -> float | None:
    if match_rate is None and support is None and reference is None:
        return None
    w_match, w_support, w_reference = weights
    return (
        w_match * float(match_rate or 0.0)
        + w_support * float(support or 0.0)
        + w_reference * float(reference or 0.0)
    )


def _synthesis_suggestions(
    check_payload: dict[str, Any] | None,
    alignment_payload: dict[str, Any] | None,
    score: float | None,
    ceiling: float,
) -> list[str]:
    suggestions: list[str] = []
    if check_payload is None:
        suggestions.append("Run `semia synthesize` to produce check artifacts.")
        return suggestions
    if not check_payload.get("program_valid"):
        suggestions.append("Fix structural errors listed in synthesis_check.json before scoring.")
        return suggestions
    if score is not None and score >= ceiling:
        suggestions.append(f"Composite score {score:.3f} ≥ ceiling {ceiling}; ship.")
        return suggestions
    match_rate = (alignment_payload or {}).get("evidence_match_rate")
    if match_rate is not None and float(match_rate) < 0.6:
        suggestions.append(
            "Low evidence_match_rate: rewrite *_evidence_text(...) quotes to match "
            "actual phrases in prepared_skill.md (no paraphrasing)."
        )
    reference = (alignment_payload or {}).get("reference_unit_coverage")
    if reference is not None and float(reference) < 0.4:
        suggestions.append(
            "Low reference_unit_coverage: cover more of prepared_skill.md by adding "
            "evidence-text facts for under-covered actions/calls."
        )
    if check_payload.get("warnings"):
        suggestions.append(
            f"{len(check_payload['warnings'])} warning(s) in synthesis_check.json — "
            "address EVD011/EVD012 to lift fact_support_coverage."
        )
    if not suggestions:
        suggestions.append("No targeted suggestion; iterate or stop.")
    return suggestions


def _synthesis_metadata_summary(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    iterations = metadata.get("iterations") or []
    return {
        "selected_iteration": metadata.get("selected_iteration"),
        "stop_reason": metadata.get("stop_reason"),
        "completed": metadata.get("completed"),
        "iterations_run": len(iterations),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
    }


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _print_result(stdout: TextIO, result: Any, fallback: str) -> None:
    if result is None:
        print(fallback, file=stdout)
        return
    if isinstance(result, str):
        print(result, file=stdout)
        return
    if isinstance(result, bytes):
        print(result.decode("utf-8"), file=stdout)
        return
    if _wants_human_output(stdout):
        summary = _summarize_result(result)
        if summary is not None:
            print(summary, file=stdout)
            return
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), file=stdout)


def _wants_human_output(stream: TextIO) -> bool:
    """Pretty-print to TTYs unless the caller forces JSON via ``SEMIA_JSON=1``."""
    if os.environ.get("SEMIA_JSON") == "1":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _summarize_result(result: Any) -> str | None:
    """Compact one-line summary for known result shapes; ``None`` falls back to JSON."""
    if not isinstance(result, dict):
        return None
    status = result.get("status")
    if status == "prepared":
        units = result.get("semantic_units")
        run_dir = result.get("run_dir")
        run_str = f" → {_short_path(run_dir)}" if run_dir else ""
        unit_str = f"{units} semantic unit(s)" if units is not None else "skill"
        return f"Prepared {unit_str}{run_str}"
    if status == "synthesized":
        score = result.get("score")
        iters = result.get("iterations")
        stop = result.get("stop_reason")
        provider = result.get("provider")
        model = result.get("model")
        score_s = f"{score:.3f}" if isinstance(score, int | float) else "n/a"
        bits = [f"score {score_s}"]
        if iters is not None:
            bits.append(f"{iters} iter")
        if stop:
            bits.append(f"stop={stop}")
        if provider:
            bits.append(f"provider={provider}")
        if model:
            bits.append(f"model={model}")
        return "Synthesized: " + ", ".join(bits)
    if status == "baseline_synthesized":
        mode = result.get("mode") or "baseline"
        return f"Wrote baseline behavior map (mode={mode})"
    if status in {"checked", "check_failed"}:
        valid = "valid" if result.get("program_valid") else "INVALID"
        errors = result.get("errors", 0)
        warnings = result.get("warnings", 0)
        match = result.get("evidence_match_rate")
        match_s = f"{match:.3f}" if isinstance(match, int | float) else "n/a"
        return f"Checked: {valid}, errors={errors}, warnings={warnings}, evidence_match={match_s}"
    if status == "detected":
        findings = result.get("findings")
        backend = result.get("backend")
        suffix = f" (backend={backend})" if backend else ""
        if findings is None:
            return f"Detected{suffix}"
        return f"Detected: {findings} finding(s){suffix}"
    if status == "aligned":
        match = result.get("evidence_match_rate")
        match_s = f"{match:.3f}" if isinstance(match, int | float) else "n/a"
        return f"Aligned: evidence_match_rate={match_s}"
    return None


def _short_path(value: Any) -> str:
    """Render ``value`` relative to CWD when possible; absolute otherwise."""
    text = str(value)
    try:
        return str(Path(text).relative_to(Path.cwd()))
    except (ValueError, TypeError, OSError):
        return text


def _make_progress_callback(stderr: TextIO) -> Callable[[dict[str, Any]], None] | None:
    """Return a stderr progress writer, or ``None`` to skip progress entirely.

    Disabled when ``SEMIA_QUIET=1`` or ``SEMIA_PROGRESS=0``. Forced on when
    ``SEMIA_PROGRESS=1``. Otherwise enabled iff stderr is a TTY — mirrors what
    a human running ``semia`` interactively would expect, while leaving piped
    or test invocations silent.
    """
    if os.environ.get("SEMIA_QUIET") == "1":
        return None
    forced = os.environ.get("SEMIA_PROGRESS")
    if forced == "0":
        return None
    if forced != "1" and not getattr(stderr, "isatty", lambda: False)():
        return None

    def emit(event: dict[str, Any]) -> None:
        line = _format_progress_event(event)
        if line is None:
            return
        print(line, file=stderr, flush=True)

    return emit


def _format_progress_event(event: dict[str, Any]) -> str | None:
    kind = event.get("event")
    if kind == "started":
        max_iters = event.get("max_iterations")
        provider = event.get("provider")
        model = event.get("model")
        bits = []
        if max_iters is not None:
            bits.append(f"up to {max_iters} iter")
        if provider:
            bits.append(f"provider={provider}")
        if model:
            bits.append(f"model={model}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        return f"  synthesizing{suffix}…"
    if kind == "iteration":
        iter_n = event.get("iteration")
        if not event.get("valid"):
            return f"  iter {iter_n}: invalid candidate, retrying…"
        score = event.get("score")
        score_s = f"{score:.3f}" if isinstance(score, int | float) else "n/a"
        if event.get("accepted"):
            delta = event.get("delta")
            delta_s = f" ({delta:+.3f})" if isinstance(delta, int | float) and delta else ""
            best = event.get("best_score")
            best_s = f", best {best:.3f}" if isinstance(best, int | float) and best != score else ""
            stop = event.get("stop_reason")
            stop_s = f" — {stop}" if stop else ""
            return f"  iter {iter_n}: accepted, score {score_s}{delta_s}{best_s}{stop_s}"
        best = event.get("best_score")
        best_s = f", best {best:.3f}" if isinstance(best, int | float) else ""
        return f"  iter {iter_n}: rejected, score {score_s}{best_s}"
    if kind == "stopped":
        reason = event.get("stop_reason") or "done"
        best = event.get("best_score")
        best_s = f", best {best:.3f}" if isinstance(best, int | float) else ""
        iters = event.get("iterations")
        iters_s = f", {iters} iter total" if iters is not None else ""
        return f"  stopped: {reason}{best_s}{iters_s}"
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value
