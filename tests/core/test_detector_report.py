# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 RiemaLabs
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/semia-core/src"))

from semia_core import detector as detector_module
from semia_core.artifacts import (
    AuditReport,
    CheckResult,
    DetectorResult,
    EvidenceAlignmentResult,
    Finding,
)
from semia_core.checker import check_program
from semia_core.detector import run_detector
from semia_core.pipeline import (
    ARTIFACT_DETECTION_RESULT,
    ARTIFACT_PREPARE_UNITS,
    ARTIFACT_REPORT_JSON,
    ARTIFACT_SYNTHESIS_ALIGNMENT,
    ARTIFACT_SYNTHESIS_CHECK,
    check,
    detect,
    prepare,
    report,
)
from semia_core.report import render_markdown_report


class DetectorReportTests(unittest.TestCase):
    def test_detector_reports_unavailable_when_souffle_backend_forced(self) -> None:
        previous_bin = os.environ.get("SEMIA_SOUFFLE_BIN")
        previous_backend = os.environ.get("SEMIA_DETECTOR_BACKEND")
        os.environ["SEMIA_SOUFFLE_BIN"] = "/definitely/not/souffle"
        os.environ["SEMIA_DETECTOR_BACKEND"] = "souffle"
        try:
            with tempfile.TemporaryDirectory() as td:
                facts = Path(td) / "facts.dl"
                facts.write_text('skill("demo").\n', encoding="utf-8")
                result = run_detector(facts, Path(td) / "out")
        finally:
            for key, prev in (
                ("SEMIA_SOUFFLE_BIN", previous_bin),
                ("SEMIA_DETECTOR_BACKEND", previous_backend),
            ):
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.backend, "souffle")

    def test_detector_falls_back_to_builtin_when_souffle_missing(self) -> None:
        previous_bin = os.environ.get("SEMIA_SOUFFLE_BIN")
        previous_backend = os.environ.get("SEMIA_DETECTOR_BACKEND")
        os.environ["SEMIA_SOUFFLE_BIN"] = "/definitely/not/souffle"
        os.environ["SEMIA_DETECTOR_BACKEND"] = "auto"
        try:
            with tempfile.TemporaryDirectory() as td:
                facts = Path(td) / "facts.dl"
                facts.write_text('skill("demo").\n', encoding="utf-8")
                result = run_detector(facts, Path(td) / "out")
        finally:
            for key, prev in (
                ("SEMIA_SOUFFLE_BIN", previous_bin),
                ("SEMIA_DETECTOR_BACKEND", previous_backend),
            ):
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "builtin")

    def test_builtin_detector_passes_timeout_to_evaluator(self) -> None:
        previous_backend = os.environ.get("SEMIA_DETECTOR_BACKEND")
        os.environ["SEMIA_DETECTOR_BACKEND"] = "builtin"
        try:
            with tempfile.TemporaryDirectory() as td:
                facts = Path(td) / "facts.dl"
                facts.write_text('skill("demo").\n', encoding="utf-8")
                output_dir = Path(td) / "out"
                with mock.patch.object(
                    detector_module,
                    "run_evaluator",
                    return_value=detector_module.EvalResult(),
                ) as evaluator:
                    result = run_detector(facts, output_dir, timeout_seconds=7)
        finally:
            if previous_backend is None:
                os.environ.pop("SEMIA_DETECTOR_BACKEND", None)
            else:
                os.environ["SEMIA_DETECTOR_BACKEND"] = previous_backend

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "builtin")
        evaluator.assert_called_once_with(facts, output_dir, timeout_seconds=7)

    def test_render_markdown_report(self) -> None:
        check = check_program('skill("demo").\n')
        markdown = render_markdown_report(
            AuditReport(title="Semia", source_id="demo", check_result=check)
        )

        # Per the trimmed-MD contract, structural-check / evidence-grounding /
        # diagnostics no longer render to Markdown — they live in JSON/SARIF.
        self.assertIn("# Semia", markdown)
        self.assertNotIn("Structural Check", markdown)
        self.assertNotIn("Evidence Grounding", markdown)
        self.assertNotIn("Quality Diagnostics", markdown)

    @unittest.skipUnless(shutil.which("souffle"), "souffle not installed")
    def test_detector_rules_compile_and_emit_findings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skill_dir = root / "skill"
            run_dir = root / "run"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "# Demo\n\n- Run a command from downloaded content.\n",
                encoding="utf-8",
            )
            prepare(skill_dir, out_dir=run_dir)
            (run_dir / "synthesized_facts.dl").write_text(
                "\n".join(
                    [
                        '#include "rules/sdl/skill_dl_static_analysis.dl"',
                        'skill("demo").',
                        'skill_evidence_text("demo", "Demo").',
                        'action("act", "demo").',
                        'action_evidence_text("act", "Run a command from downloaded content").',
                        'value("v_payload", "act", "untrusted").',
                        'value_evidence_text("v_payload", "act", "downloaded content").',
                        'call("c_exec", "act").',
                        'call_evidence_text("c_exec", "Run a command").',
                        'call_effect("c_exec", "proc_exec").',
                        'call_effect_evidence_text("c_exec", "proc_exec", "Run a command").',
                        'call_input("c_exec", "v_payload").',
                        'call_input_evidence_text("c_exec", "v_payload", "downloaded content").',
                        'call_region_untrusted("c_exec").',
                        'call_region_untrusted_evidence_text("c_exec", "downloaded content").',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            checked = check(run_dir)
            result = detect(run_dir)

        self.assertEqual(checked["errors"], 0)
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["findings"], 1)

    def test_report_preserves_unmatched_evidence_after_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "prepare_units.json").write_text(
                json.dumps({"source_id": "demo", "units": []}),
                encoding="utf-8",
            )
            alignment_payload = {
                "evidence_match_rate": 0.5,
                "reference_unit_coverage": 0.25,
                "grounding_score": 0.4,
                "alignments": [
                    {
                        "relation": "skill_evidence_text",
                        "args": ["demo", "Some unmatched quote"],
                        "line": 7,
                        "evidence_text": "Some unmatched quote",
                        "evidence_id": None,
                        "unit_id": None,
                        "score": 0.1,
                        "matched": False,
                    },
                    {
                        "relation": "action_evidence_text",
                        "args": ["act_x", "Matched line"],
                        "line": 9,
                        "evidence_text": "Matched line",
                        "evidence_id": "su_1",
                        "unit_id": 1,
                        "score": 0.95,
                        "matched": True,
                    },
                ],
                "normalized_facts": [],
            }
            (run_dir / ARTIFACT_SYNTHESIS_ALIGNMENT).write_text(
                json.dumps(alignment_payload), encoding="utf-8"
            )

            report(run_dir, format="json")
            self.assertTrue((run_dir / ARTIFACT_REPORT_JSON).exists())

            # Markdown no longer carries evidence-grounding details (that data
            # is preserved in report.json and the SARIF), but the file must
            # still be produced.
            markdown = report(run_dir, format="md")
            self.assertNotIn("Unmatched evidence:", markdown)
            self.assertNotIn("Evidence Grounding", markdown)

    def test_souffle_subprocess_runs_in_facts_parent_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            facts = run_dir / "detection_input.dl"
            facts.write_text('skill("demo").\n', encoding="utf-8")
            output_dir = run_dir / "out"

            captured: dict[str, object] = {}

            class _FakeCompleted:
                returncode = 0
                stdout = ""
                stderr = ""

            def _fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                captured["cwd"] = kwargs.get("cwd")
                captured["timeout"] = kwargs.get("timeout")
                return _FakeCompleted()

            with (
                mock.patch.object(
                    detector_module, "find_souffle_binary", return_value="/fake/souffle"
                ),
                mock.patch.object(detector_module.subprocess, "run", side_effect=_fake_run),
            ):
                previous_backend = os.environ.get("SEMIA_DETECTOR_BACKEND")
                os.environ["SEMIA_DETECTOR_BACKEND"] = "souffle"
                try:
                    result = run_detector(facts, output_dir, timeout_seconds=7)
                finally:
                    if previous_backend is None:
                        os.environ.pop("SEMIA_DETECTOR_BACKEND", None)
                    else:
                        os.environ["SEMIA_DETECTOR_BACKEND"] = previous_backend

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.backend, "souffle")
            self.assertEqual(captured["cwd"], str(facts.parent))
            self.assertEqual(captured["timeout"], 7)


class RenderMarkdownReportTests(unittest.TestCase):
    def test_title_and_source_render_at_top(self) -> None:
        markdown = render_markdown_report(AuditReport(title="X", source_id="y"))
        self.assertTrue(markdown.startswith("# X\n\nSource: `y`\n"))

    def test_empty_report_renders_only_header(self) -> None:
        markdown = render_markdown_report(AuditReport(title="X", source_id="y"))
        self.assertEqual(markdown, "# X\n\nSource: `y`\n")

    def test_check_result_does_not_render_in_markdown(self) -> None:
        """Structural-check details belong in report.json / report.sarif.json
        only — the Markdown is reserved for detector findings + evidence."""

        cr = CheckResult(issues=(), program_valid=True, evidence_support_coverage=1.0)
        markdown = render_markdown_report(AuditReport(title="X", source_id="y", check_result=cr))
        self.assertNotIn("Structural Check", markdown)
        self.assertNotIn("Errors:", markdown)
        self.assertNotIn("Evidence support coverage", markdown)

    def test_evidence_result_does_not_render_in_markdown(self) -> None:
        """Evidence-grounding metrics belong in JSON only."""

        er = EvidenceAlignmentResult(
            alignments=(),
            normalized_facts=(),
            evidence_match_rate=1.0,
            reference_unit_coverage=1.0,
            grounding_score=1.0,
        )
        markdown = render_markdown_report(AuditReport(title="X", source_id="y", evidence_result=er))
        self.assertNotIn("Evidence Grounding", markdown)
        self.assertNotIn("Unmatched evidence", markdown)

    def test_detector_section_with_no_findings_omits_finding_lines(self) -> None:
        dr = DetectorResult(status="ok", findings=())
        markdown = render_markdown_report(AuditReport(title="X", source_id="y", detector_result=dr))
        self.assertIn("- Findings: 0", markdown)
        self.assertIn("- Status: `ok`", markdown)
        finding_lines = [ln for ln in markdown.splitlines() if ln.startswith("- `")]
        self.assertEqual(finding_lines, [])

    def test_detector_section_findings_render_label_and_fields(self) -> None:
        findings = (
            Finding(label="EscapingSecret", fields=("v_token", "c_send"), severity="error"),
            Finding(label="NoFields", fields=(), severity="warning"),
        )
        dr = DetectorResult(status="ok", findings=findings, message="hello")
        markdown = render_markdown_report(AuditReport(title="X", source_id="y", detector_result=dr))
        self.assertIn("- Message: hello", markdown)
        self.assertIn("- `EscapingSecret`: `v_token`, `c_send`", markdown)
        self.assertIn("- `NoFields`", markdown)

    def test_notes_section_renders_bullets_when_non_empty(self) -> None:
        markdown = render_markdown_report(
            AuditReport(title="X", source_id="y", notes=("first", "second"))
        )
        self.assertIn("## Notes", markdown)
        self.assertIn("- first", markdown)
        self.assertIn("- second", markdown)

    def test_diagnostics_does_not_render_in_markdown(self) -> None:
        """ssa_input_availability and friends belong in JSON only."""

        markdown = render_markdown_report(
            AuditReport(title="X", source_id="y", diagnostics={"ssa_input_availability": 0.5})
        )
        self.assertNotIn("Quality Diagnostics", markdown)
        self.assertNotIn("SSA input availability", markdown)

    def test_detector_findings_inline_evidence_quotes_from_atom_map(self) -> None:
        """When an atom referenced by a finding has *_evidence_text rows in
        the synthesized facts, the report inlines those quotes as a sub-bullet
        so the reader sees the original skill source that triggered each finding."""

        dr = DetectorResult(
            status="ok",
            findings=(
                Finding(
                    label="DangerousExec",
                    fields=("act_run", "c_eval"),
                    severity="error",
                ),
            ),
        )
        evidence = {
            "c_eval": ("browser-use eval to extract email info",),
            "act_run": ("使用 Python 生成邮件总结",),
            "v_unused": ("never referenced",),
        }
        markdown = render_markdown_report(
            AuditReport(title="X", source_id="y", detector_result=dr),
            evidence_by_atom=evidence,
        )
        self.assertIn("- `DangerousExec`: `act_run`, `c_eval`", markdown)
        self.assertIn("- `c_eval` evidence: 'browser-use eval to extract email info'", markdown)
        self.assertIn("- `act_run` evidence: '使用 Python 生成邮件总结'", markdown)
        # Atoms not referenced by any finding stay out of the report.
        self.assertNotIn("never referenced", markdown)


class SarifReportTests(unittest.TestCase):
    def _write_run(
        self, root: Path, findings: list[dict], *, check_payload: dict | None = None
    ) -> None:
        (root / ARTIFACT_PREPARE_UNITS).write_text(
            json.dumps({"source_id": "demo", "units": []}), encoding="utf-8"
        )
        (root / ARTIFACT_DETECTION_RESULT).write_text(
            json.dumps({"status": "ok", "findings": findings}), encoding="utf-8"
        )
        if check_payload is not None:
            (root / ARTIFACT_SYNTHESIS_CHECK).write_text(
                json.dumps(check_payload), encoding="utf-8"
            )

    def test_sarif_has_schema_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(
                root, [{"label": "L", "fields": [], "severity": "warning", "message": "m"}]
            )
            payload = report(root, format="sarif")
        self.assertEqual(payload["version"], "2.1.0")
        self.assertIn("sarif-2.1.0", payload["$schema"])
        self.assertEqual(payload["runs"][0]["tool"]["driver"]["name"], "Semia")

    def test_sarif_dedupes_rules_by_label(self) -> None:
        findings = [
            {"label": "Dup", "fields": ["a"], "severity": "warning", "message": "m"},
            {"label": "Dup", "fields": ["b"], "severity": "warning", "message": "m"},
            {"label": "Dup", "fields": ["c"], "severity": "warning", "message": "m"},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, findings)
            payload = report(root, format="sarif")
        rules = payload["runs"][0]["tool"]["driver"]["rules"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(len(payload["runs"][0]["results"]), 3)

    def test_sarif_level_mapping_for_each_severity(self) -> None:
        findings = [
            {"label": "Err", "fields": [], "severity": "error", "message": "m"},
            {"label": "Info", "fields": [], "severity": "info", "message": "m"},
            {"label": "Note", "fields": [], "severity": "note", "message": "m"},
            {"label": "Warn", "fields": [], "severity": "warning", "message": "m"},
            {"label": "Other", "fields": [], "severity": "gibberish", "message": "m"},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, findings)
            payload = report(root, format="sarif")
        levels = {r["ruleId"]: r["level"] for r in payload["runs"][0]["results"]}
        self.assertEqual(
            levels,
            {"Err": "error", "Info": "note", "Note": "note", "Warn": "warning", "Other": "warning"},
        )

    def test_sarif_result_message_falls_back_to_label(self) -> None:
        findings = [{"label": "OnlyLabel", "fields": [], "severity": "warning", "message": ""}]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, findings)
            payload = report(root, format="sarif")
        self.assertEqual(payload["runs"][0]["results"][0]["message"]["text"], "OnlyLabel")

    def test_sarif_result_message_uses_fields_when_message_empty(self) -> None:
        findings = [
            {"label": "L", "fields": ["alpha", "beta"], "severity": "warning", "message": ""}
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, findings)
            payload = report(root, format="sarif")
        self.assertEqual(payload["runs"][0]["results"][0]["message"]["text"], "alpha, beta")

    def test_sarif_driver_properties_carry_ssa_input_availability(self) -> None:
        findings = [{"label": "L", "fields": [], "severity": "warning", "message": "m"}]
        check_payload = {
            "program_valid": True,
            "evidence_support_coverage": 1.0,
            "errors": [],
            "warnings": [],
            "ssa_input_availability": 0.75,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, findings, check_payload=check_payload)
            payload = report(root, format="sarif")
        props = payload["runs"][0]["tool"]["driver"].get("properties")
        self.assertEqual(props, {"ssa_input_availability": 0.75})


if __name__ == "__main__":
    unittest.main()
