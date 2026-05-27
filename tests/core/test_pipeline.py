# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 RiemaLabs
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/semia-core/src"))

from semia_core.pipeline import (
    ARTIFACT_DETECTION_FINDINGS,
    ARTIFACT_DETECTION_INPUT,
    ARTIFACT_DETECTION_RESULT,
    ARTIFACT_MANIFEST,
    ARTIFACT_PREPARE_METADATA,
    ARTIFACT_PREPARE_UNITS,
    ARTIFACT_PREPARE_UNITS_DL,
    ARTIFACT_PREPARED_SKILL,
    ARTIFACT_REPORT_JSON,
    ARTIFACT_REPORT_MD,
    ARTIFACT_REPORT_SARIF,
    ARTIFACT_SYNTHESIS_ALIGNMENT,
    ARTIFACT_SYNTHESIS_CHECK,
    ARTIFACT_SYNTHESIS_NORMALIZED,
    ARTIFACT_SYNTHESIS_PROMPT,
    ARTIFACT_SYNTHESIZED_FACTS,
    align_evidence,
    check,
    check_facts,
    detect,
    extract_baseline,
    prepare,
    render_report,
    report,
)

_SKILL_MD = "# Demo Skill\n\n- Read a local file.\n- Send no network traffic.\n"

_WORKED_EXAMPLE_SKILL_MD = "# Demo Skill\n\nthe only action: read a file.\n"

_VALID_V2_FACTS = """#include "rules/sdl/skill_dl_static_analysis.dl"
skill("demo").
skill_evidence_text("demo", "Demo Skill").
action("act", "demo").
action_evidence_text("act", "the only action").
call("c", "act").
call_effect("c", "fs_read").
call_evidence_text("c", "read a file").
call_effect_evidence_text("c", "fs_read", "read a file").
"""


def _write_skill_dir(root: Path, content: str = _SKILL_MD, *, with_helper: bool = False) -> Path:
    (root / "SKILL.md").write_text(content, encoding="utf-8")
    if with_helper:
        (root / "helper.py").write_text("print('hello')\n", encoding="utf-8")
    return root


class PrepareTests(unittest.TestCase):
    def test_prepare_writes_all_five_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            result = prepare(skill_dir, run_dir=run_dir)

            self.assertEqual(result["status"], "prepared")
            for name in (
                ARTIFACT_PREPARED_SKILL,
                ARTIFACT_PREPARE_METADATA,
                ARTIFACT_PREPARE_UNITS,
                ARTIFACT_SYNTHESIS_PROMPT,
                ARTIFACT_MANIFEST,
            ):
                self.assertTrue((run_dir / name).exists(), f"missing {name}")

    def test_prepare_out_dir_alias_equivalent_to_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            a = Path(td) / "run_a"
            b = Path(td) / "run_b"
            res_a = prepare(skill_dir, run_dir=a)
            res_b = prepare(skill_dir, out_dir=b)

            self.assertEqual(res_a["source_id"], res_b["source_id"])
            self.assertEqual(
                (a / ARTIFACT_PREPARED_SKILL).read_text(encoding="utf-8"),
                (b / ARTIFACT_PREPARED_SKILL).read_text(encoding="utf-8"),
            )

    def test_prepare_handles_single_file_skill(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_path = Path(td) / "SKILL.md"
            skill_path.write_text(_SKILL_MD, encoding="utf-8")
            run_dir = Path(td) / "run"
            result = prepare(skill_path, run_dir=run_dir)

            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["source_id"], "SKILL")
            self.assertGreater(result["semantic_units"], 0)
            self.assertTrue((run_dir / ARTIFACT_PREPARED_SKILL).exists())

    def test_prepare_manifest_has_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)

            manifest = json.loads((run_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_contract"], "semia-run-v1")
            self.assertEqual(manifest["stage"], "prepared")
            self.assertIn("source_id", manifest)
            self.assertIn("source_hash", manifest)


class ExtractBaselineTests(unittest.TestCase):
    def test_extract_baseline_writes_synthesized_facts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)

            result = extract_baseline(run_dir)
            facts_text = (run_dir / ARTIFACT_SYNTHESIZED_FACTS).read_text(encoding="utf-8")

            self.assertIn("skill(", facts_text)
            self.assertIn("skill_evidence_text(", facts_text)
            self.assertEqual(result["status"], "baseline_synthesized")

    def test_extract_baseline_sets_manifest_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)

            manifest = json.loads((run_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["synthesis_mode"], "conservative_baseline")
            self.assertIn("synthesis_written_at", manifest)

    def test_extract_baseline_emits_doc_claims_when_text_supports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(
                skill_dir, content="# Demo\n\n- read only operation.\n- no network usage here.\n"
            )
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)

            facts_text = (run_dir / ARTIFACT_SYNTHESIZED_FACTS).read_text(encoding="utf-8")
            self.assertIn("skill_doc_claim(", facts_text)

    def test_extract_baseline_writes_include_directive(self) -> None:
        """The baseline output must include the SDL rules so strict check passes."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            skill_dir = run_dir / "skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "# Demo\n\n- Read a local file.\n", encoding="utf-8"
            )
            prepare(skill_dir, out_dir=run_dir)
            extract_baseline(run_dir)
            facts_text = (run_dir / "synthesized_facts.dl").read_text(encoding="utf-8")
            self.assertTrue(
                facts_text.startswith('#include "rules/sdl/skill_dl_static_analysis.dl"')
            )

    def test_baseline_pipeline_passes_check(self) -> None:
        """End-to-end: prepare -> extract_baseline -> check_facts -> program_valid=True."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            skill_dir = run_dir / "skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "# Demo\n\n- Read a local file.\n", encoding="utf-8"
            )
            prepare(skill_dir, out_dir=run_dir)
            extract_baseline(run_dir)
            result = check_facts(run_dir)
            self.assertEqual(result["status"], "checked", result)
            self.assertTrue(result["program_valid"])


class CheckFactsTests(unittest.TestCase):
    def _prepare(self, td: Path, content: str = _SKILL_MD) -> Path:
        skill_dir = td / "skill"
        skill_dir.mkdir()
        _write_skill_dir(skill_dir, content=content)
        run_dir = td / "run"
        prepare(skill_dir, run_dir=run_dir)
        return run_dir

    def test_check_facts_defaults_to_synthesized_facts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepare(Path(td))
            extract_baseline(run_dir)
            result = check_facts(run_dir)

            self.assertIn("ssa_input_availability", result)
            self.assertTrue((run_dir / ARTIFACT_SYNTHESIS_CHECK).exists())
            self.assertTrue((run_dir / ARTIFACT_SYNTHESIS_ALIGNMENT).exists())
            self.assertTrue((run_dir / ARTIFACT_SYNTHESIS_NORMALIZED).exists())

    def test_check_facts_accepts_explicit_facts_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepare(Path(td), content=_WORKED_EXAMPLE_SKILL_MD)
            facts_path = Path(td) / "external_facts.dl"
            facts_path.write_text(_VALID_V2_FACTS, encoding="utf-8")
            result = check_facts(run_dir, facts_path=facts_path)

            self.assertTrue(result["program_valid"], result)

    def test_check_facts_reports_sdl004_when_include_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepare(Path(td))
            (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(
                'skill("demo").\nskill_evidence_text("demo", "Demo Skill").\n',
                encoding="utf-8",
            )
            result = check_facts(run_dir)
            payload = json.loads((run_dir / ARTIFACT_SYNTHESIS_CHECK).read_text(encoding="utf-8"))

            self.assertFalse(result["program_valid"])
            codes = {issue["code"] for issue in payload["errors"]}
            self.assertIn("SDL004", codes)

    def test_check_facts_check_payload_includes_ssa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepare(Path(td))
            extract_baseline(run_dir)
            check_facts(run_dir)

            payload = json.loads((run_dir / ARTIFACT_SYNTHESIS_CHECK).read_text(encoding="utf-8"))
            self.assertIn("ssa_input_availability", payload)
            self.assertIsInstance(payload["ssa_input_availability"], (int, float))


class CheckAliasTests(unittest.TestCase):
    def test_check_alias_returns_same_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)
            a = check(run_dir, facts_path=None)
            b = check_facts(run_dir)

            self.assertEqual(set(a.keys()), set(b.keys()))
            self.assertEqual(a["program_valid"], b["program_valid"])
            self.assertEqual(a["status"], b["status"])


class AlignEvidenceTests(unittest.TestCase):
    def test_align_evidence_overwrites_alignment_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)
            check_facts(run_dir)
            (run_dir / ARTIFACT_SYNTHESIS_ALIGNMENT).write_text("OVERWRITE_ME", encoding="utf-8")

            result = align_evidence(run_dir, facts_path=None)
            payload = json.loads(
                (run_dir / ARTIFACT_SYNTHESIS_ALIGNMENT).read_text(encoding="utf-8")
            )

            self.assertEqual(result["status"], "aligned")
            self.assertIn("alignments", payload)
            self.assertIn("evidence_match_rate", payload)

    def test_align_evidence_returns_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)
            result = align_evidence(run_dir)

            self.assertGreaterEqual(result["evidence_match_rate"], 0.0)
            self.assertGreaterEqual(result["reference_unit_coverage"], 0.0)


class DetectTests(unittest.TestCase):
    def _pipeline_through_check(
        self, td: Path, content: str = _WORKED_EXAMPLE_SKILL_MD, facts: str = _VALID_V2_FACTS
    ) -> Path:
        skill_dir = td / "skill"
        skill_dir.mkdir()
        _write_skill_dir(skill_dir, content=content)
        run_dir = td / "run"
        prepare(skill_dir, run_dir=run_dir)
        (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(facts, encoding="utf-8")
        check_facts(run_dir)
        return run_dir

    def test_detect_writes_input_findings_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_through_check(Path(td))
            detect(run_dir)

            self.assertTrue((run_dir / ARTIFACT_DETECTION_INPUT).exists())
            self.assertTrue((run_dir / ARTIFACT_DETECTION_FINDINGS).exists())
            self.assertTrue((run_dir / ARTIFACT_DETECTION_RESULT).exists())

    def test_detect_copies_rules_to_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_through_check(Path(td))
            detect(run_dir)

            rules_dir = run_dir / "rules" / "sdl"
            self.assertTrue((rules_dir / "skill_dl_static_analysis.dl").exists())
            self.assertTrue((rules_dir / "skill_description_lang.dl").exists())

    def test_detect_returns_status_from_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_through_check(Path(td))
            result = detect(run_dir)

            self.assertIn(result["status"], {"ok", "unavailable", "failed"})
            self.assertIn("backend", result)
            self.assertIn("findings", result)

    def test_detect_raises_when_normalized_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)

            with self.assertRaises(FileNotFoundError):
                detect(run_dir)


class ReportTests(unittest.TestCase):
    def _pipeline_full(self, td: Path) -> Path:
        skill_dir = td / "skill"
        skill_dir.mkdir()
        _write_skill_dir(skill_dir, content=_WORKED_EXAMPLE_SKILL_MD)
        run_dir = td / "run"
        prepare(skill_dir, run_dir=run_dir)
        (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(_VALID_V2_FACTS, encoding="utf-8")
        check_facts(run_dir)
        detect(run_dir)
        return run_dir

    def test_report_md_returns_string_and_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_full(Path(td))
            result = report(run_dir, format="md")

            self.assertIsInstance(result, str)
            self.assertIn("Semia Report", result)
            self.assertTrue((run_dir / ARTIFACT_REPORT_MD).exists())

    def test_report_json_returns_dict_and_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_full(Path(td))
            result = report(run_dir, format="json")

            self.assertIsInstance(result, dict)
            self.assertIn("title", result)
            self.assertTrue((run_dir / ARTIFACT_REPORT_JSON).exists())

    def test_report_sarif_writes_2_1_0_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_full(Path(td))
            result = report(run_dir, format="sarif")
            on_disk = json.loads((run_dir / ARTIFACT_REPORT_SARIF).read_text(encoding="utf-8"))

            self.assertEqual(result["version"], "2.1.0")
            self.assertEqual(on_disk["version"], "2.1.0")
            self.assertIn("runs", result)
            self.assertEqual(result["runs"][0]["tool"]["driver"]["name"], "Semia")

    def test_report_sarif_includes_rule_per_finding_label(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_full(Path(td))
            detector_payload = json.loads(
                (run_dir / ARTIFACT_DETECTION_RESULT).read_text(encoding="utf-8")
            )
            detector_payload["findings"] = [
                {"label": "rule_a", "fields": ["x"], "severity": "warning", "message": "msg a"},
                {"label": "rule_a", "fields": ["y"], "severity": "warning", "message": "msg a"},
                {"label": "rule_b", "fields": ["z"], "severity": "error", "message": "msg b"},
            ]
            (run_dir / ARTIFACT_DETECTION_RESULT).write_text(
                json.dumps(detector_payload), encoding="utf-8"
            )
            result = report(run_dir, format="sarif")

            rule_ids = {rule["id"] for rule in result["runs"][0]["tool"]["driver"]["rules"]}
            self.assertEqual(rule_ids, {"rule_a", "rule_b"})
            self.assertEqual(len(result["runs"][0]["results"]), 3)

    def test_report_raises_on_unsupported_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._pipeline_full(Path(td))
            with self.assertRaises(ValueError):
                report(run_dir, format="xml")

    def test_report_json_md_roundtrip_preserves_normalized_facts(self) -> None:
        """After report(json), report(md) should still have normalized facts available."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "prepare_units.json").write_text(
                json.dumps({"source_id": "demo", "units": []}), encoding="utf-8"
            )
            alignment_payload = {
                "evidence_match_rate": 1.0,
                "reference_unit_coverage": 1.0,
                "grounding_score": 1.0,
                "alignments": [],
                "normalized_facts": [
                    'skill_evidence("demo", "su_0").',
                    'call_effect_evidence("c", "fs_read", "su_5").',
                ],
            }
            (run_dir / "synthesis_evidence_alignment.json").write_text(
                json.dumps(alignment_payload), encoding="utf-8"
            )
            payload = report(run_dir, format="json")
            markdown = report(run_dir, format="md")
            self.assertIsInstance(markdown, str)
            self.assertIn("normalized_facts", payload.get("evidence", {}))
            self.assertEqual(len(payload["evidence"]["normalized_facts"]), 2)


class RenderReportAliasTests(unittest.TestCase):
    def test_render_report_is_alias_of_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir, content=_WORKED_EXAMPLE_SKILL_MD)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(_VALID_V2_FACTS, encoding="utf-8")
            check_facts(run_dir)
            detect(run_dir)

            a = render_report(run_dir, format="md")
            b = report(run_dir, format="md")
            self.assertEqual(a, b)


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_baseline_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir, with_helper=True)
            run_dir = Path(td) / "run"

            prep = prepare(skill_dir, run_dir=run_dir)
            base = extract_baseline(run_dir)
            chk = check_facts(run_dir)
            det = detect(run_dir)
            md = report(run_dir, format="md")

            self.assertEqual(prep["status"], "prepared")
            self.assertEqual(base["status"], "baseline_synthesized")
            self.assertIn(chk["status"], {"checked", "check_failed"})
            self.assertIn(det["status"], {"ok", "unavailable", "failed"})

            manifest = json.loads((run_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            for key in ("prepared_at", "synthesis_written_at", "checked_at", "detected_at"):
                self.assertIn(key, manifest, f"manifest missing {key}")
            self.assertIn("Semia Report", md)

    def test_valid_v2_facts_program_valid_with_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir, content=_WORKED_EXAMPLE_SKILL_MD)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(_VALID_V2_FACTS, encoding="utf-8")

            result = check_facts(run_dir)
            self.assertTrue(result["program_valid"], result)
            self.assertGreater(result["evidence_support_coverage"], 0.5)

    def test_report_md_idempotent_on_same_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir, content=_WORKED_EXAMPLE_SKILL_MD)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(_VALID_V2_FACTS, encoding="utf-8")
            check_facts(run_dir)
            detect(run_dir)

            first = report(run_dir, format="md")
            second = report(run_dir, format="md")
            self.assertEqual(first, second)


class HostileFenceAndProvenanceTests(unittest.TestCase):
    def test_prepare_emits_hostile_input_nonce_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            result = prepare(skill_dir, run_dir=run_dir)

            nonce = result["hostile_input_nonce"]
            self.assertIsInstance(nonce, str)
            self.assertGreater(len(nonce), 8)

            meta = json.loads((run_dir / ARTIFACT_PREPARE_METADATA).read_text(encoding="utf-8"))
            self.assertEqual(meta["hostile_input_nonce"], nonce)
            self.assertEqual(len(meta["prepared_skill_sha256"]), 64)

            manifest = json.loads((run_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["hostile_input_nonce"], nonce)
            self.assertEqual(manifest["prepared_skill_sha256"], meta["prepared_skill_sha256"])

            prompt = (run_dir / ARTIFACT_SYNTHESIS_PROMPT).read_text(encoding="utf-8")
            self.assertIn(f"<<<SEMIA_HOSTILE_INPUT id={nonce}>>>", prompt)
            self.assertIn(f"<<<SEMIA_END id={nonce}>>>", prompt)

    def test_prepare_nonce_changes_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            a = Path(td) / "run_a"
            b = Path(td) / "run_b"
            nonce_a = prepare(skill_dir, run_dir=a)["hostile_input_nonce"]
            nonce_b = prepare(skill_dir, run_dir=b)["hostile_input_nonce"]
            self.assertNotEqual(nonce_a, nonce_b)

    def test_prepared_skill_sha256_is_deterministic(self) -> None:
        """Same input must produce byte-identical artifacts across runs.

        Load-bearing for reproducibility: ``run_manifest.json`` claims a
        ``(source, model, session)`` tuple pinned a particular report, but the
        source claim is only meaningful if the inlined skill, the SHA, and
        the inventory hash are all stable across runs.

        Covers three independent stability surfaces:
        1. ``prepared_skill.md`` body is byte-identical (deterministic
           directory inlining, deterministic file ordering).
        2. ``prepared_skill_sha256`` in ``prepare_metadata.json`` and
           ``run_manifest.json`` agrees with (1) and across runs.
        3. ``bundle.source.source_hash`` (the inventory-content hash) is
           stable too, so downstream callers using either hash agree.
        """
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir, with_helper=True)
            a = Path(td) / "run_a"
            b = Path(td) / "run_b"
            prepare(skill_dir, run_dir=a)
            prepare(skill_dir, run_dir=b)

            body_a = (a / ARTIFACT_PREPARED_SKILL).read_bytes()
            body_b = (b / ARTIFACT_PREPARED_SKILL).read_bytes()
            self.assertEqual(body_a, body_b, "prepared_skill.md must be byte-identical")

            meta_a = json.loads((a / ARTIFACT_PREPARE_METADATA).read_text(encoding="utf-8"))
            meta_b = json.loads((b / ARTIFACT_PREPARE_METADATA).read_text(encoding="utf-8"))
            self.assertEqual(meta_a["prepared_skill_sha256"], meta_b["prepared_skill_sha256"])
            # SHA must match what we can re-derive from the file body — guards
            # against the metadata field decoupling from the actual artifact.
            import hashlib as _hashlib

            self.assertEqual(
                meta_a["prepared_skill_sha256"],
                _hashlib.sha256(body_a).hexdigest(),
            )

            manifest_a = json.loads((a / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            manifest_b = json.loads((b / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_a["prepared_skill_sha256"],
                manifest_b["prepared_skill_sha256"],
            )
            # source_hash is the inventory-content hash (different from
            # prepared_skill_sha256, which hashes the inlined output). Must
            # also be stable.
            self.assertEqual(manifest_a["source_hash"], manifest_b["source_hash"])

    def test_check_facts_records_synthesized_facts_sha_and_host(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)
            result = check_facts(
                run_dir,
                host_session_id="sess-xyz",
                host_model="claude-opus-4-7",
            )

            self.assertEqual(len(result["synthesized_facts_sha256"]), 64)
            manifest = json.loads((run_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["host_synthesis"]["session_id"], "sess-xyz")
            self.assertEqual(manifest["host_synthesis"]["model"], "claude-opus-4-7")
            self.assertIn("recorded_at", manifest["host_synthesis"])
            self.assertEqual(
                manifest["synthesized_facts_sha256"],
                result["synthesized_facts_sha256"],
            )

    def test_evidence_taint_threshold_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)
            result = check_facts(run_dir)
            self.assertEqual(result["evidence_taint_threshold"], 0.1)

    def test_evidence_taint_threshold_blocks_hallucinated_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)

            facts = (run_dir / ARTIFACT_SYNTHESIZED_FACTS).read_text(encoding="utf-8")
            facts += (
                'skill_doc_claim("sk", "no_network").\n'
                'skill_doc_claim_evidence_text("sk", "no_network", '
                '"ZZZ_NOT_PRESENT_IN_PREPARED_SKILL_AT_ALL_xqxqxq").\n'
            )
            (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(facts, encoding="utf-8")

            result = check_facts(run_dir, evidence_taint_threshold=0.95)
            self.assertFalse(result["program_valid"])
            payload = json.loads((run_dir / ARTIFACT_SYNTHESIS_CHECK).read_text(encoding="utf-8"))
            codes = {issue["code"] for issue in payload["errors"]}
            self.assertIn("EVD020", codes)

    def test_evidence_taint_threshold_reads_env(self) -> None:
        import os as _os

        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            _write_skill_dir(skill_dir)
            run_dir = Path(td) / "run"
            prepare(skill_dir, run_dir=run_dir)
            extract_baseline(run_dir)
            prev = _os.environ.get("SEMIA_EVIDENCE_TAINT_THRESHOLD")
            _os.environ["SEMIA_EVIDENCE_TAINT_THRESHOLD"] = "0.95"
            try:
                facts = (run_dir / ARTIFACT_SYNTHESIZED_FACTS).read_text(encoding="utf-8")
                facts += (
                    'skill_doc_claim("sk", "no_network").\n'
                    'skill_doc_claim_evidence_text("sk", "no_network", '
                    '"ZZZ_NOT_PRESENT_xqxqxq").\n'
                )
                (run_dir / ARTIFACT_SYNTHESIZED_FACTS).write_text(facts, encoding="utf-8")
                result = check_facts(run_dir)
                self.assertFalse(result["program_valid"])
                self.assertEqual(result["evidence_taint_threshold"], 0.95)
            finally:
                if prev is None:
                    _os.environ.pop("SEMIA_EVIDENCE_TAINT_THRESHOLD", None)
                else:
                    _os.environ["SEMIA_EVIDENCE_TAINT_THRESHOLD"] = prev


class PrepareUnitsDlTests(unittest.TestCase):
    def _prepared(self, td: Path) -> Path:
        skill_dir = td / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "# Demo\n\n- Read a file.\n- Do not call the network.\n", encoding="utf-8"
        )
        run_dir = td / "run"
        prepare(skill_dir, out_dir=run_dir)
        return run_dir

    def test_prepare_writes_units_dl_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepared(Path(td))
            self.assertTrue((run_dir / ARTIFACT_PREPARE_UNITS_DL).exists())

    def test_prepare_units_dl_starts_with_include(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepared(Path(td))
            text = (run_dir / ARTIFACT_PREPARE_UNITS_DL).read_text(encoding="utf-8")
            first_non_blank = next(line for line in text.splitlines() if line.strip())
            self.assertEqual(first_non_blank, '#include "rules/sdl/skill_description_lang.dl"')

    def test_prepare_units_dl_emits_one_evidence_unit_per_semantic_unit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepared(Path(td))
            dl_text = (run_dir / ARTIFACT_PREPARE_UNITS_DL).read_text(encoding="utf-8")
            json_payload = json.loads(
                (run_dir / ARTIFACT_PREPARE_UNITS).read_text(encoding="utf-8")
            )
            count = sum(1 for line in dl_text.splitlines() if line.startswith("evidence_unit("))
            self.assertEqual(count, json_payload["total_units"])

    def test_prepare_units_dl_includes_type_and_location(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._prepared(Path(td))
            dl_text = (run_dir / ARTIFACT_PREPARE_UNITS_DL).read_text(encoding="utf-8")
            json_payload = json.loads(
                (run_dir / ARTIFACT_PREPARE_UNITS).read_text(encoding="utf-8")
            )
            first = json_payload["units"][0]
            ev = first["evidence_id"]
            self.assertIn(f'evidence_unit("{ev}", {first["id"]}).', dl_text)
            self.assertIn(f'evidence_unit_type("{ev}",', dl_text)
            self.assertIn(
                f'evidence_unit_location("{ev}", "{first["source_file"]}", '
                f"{first['line_start']}, {first['line_end']}).",
                dl_text,
            )

    def test_prepare_units_dl_parses_with_facts_parser(self) -> None:
        from semia_core.facts import parse_facts

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            skill_dir = Path(td) / "skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Demo\n\n- Read a file.\n", encoding="utf-8")
            prepare(skill_dir, out_dir=run_dir)
            text = (run_dir / "prepare_units.dl").read_text(encoding="utf-8")
            program = parse_facts(text)
            self.assertEqual(
                len(program.unknown_facts), 0, [f.render() for f in program.unknown_facts]
            )
            relations = {f.relation for f in program.evidence_unit_facts}
            self.assertIn("evidence_unit", relations)
            self.assertIn("evidence_unit_type", relations)
            self.assertIn("evidence_unit_location", relations)


if __name__ == "__main__":
    unittest.main()
