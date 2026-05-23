# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 RiemaLabs
"""Tests for the built-in pure-Python Datalog evaluator."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/semia-core/src"))

from semia_core.datalog_eval import ParseError, parse_dl_file, run_evaluator
from semia_core.datalog_eval.engine import EvalError, evaluate
from semia_core.datalog_eval.parser import Program

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = REPO_ROOT / "packages/semia-core/src/semia_core/rules/sdl"

# Scenarios with hand-checked expected findings, used both for the built-in
# evaluator and (when Soufflé is present) cross-validation. Each scenario
# targets a specific rule head; assertion tests live in SDLRulesEndToEndTests.
SCENARIOS: tuple[tuple[str, str], ...] = (
    (
        "minimal_proc_exec_untrusted",
        """
        skill("demo").
        action("act", "demo").
        value("v_payload", "act", "untrusted").
        call("c_exec", "act").
        call_effect("c_exec", "proc_exec").
        call_input("c_exec", "v_payload").
        call_region_untrusted("c_exec").
        """,
    ),
    (
        "rich_recursion_contains_disjunction",
        """
        skill("sk").
        skill_doc_claim("sk", "read_only").
        action("a1", "sk").
        action_param("a1", "api_key", "v_param").
        action_trigger("a1", "on_install").

        value("v_param", "a1", "param").
        value("v_inter", "a1", "untrusted").
        value("v_target", "a1", "untrusted").
        value("v_secret_local", "a1", "secret").

        call("c_fetch", "a1").
        call_effect("c_fetch", "net_read").
        call_region_untrusted("c_fetch").
        call_output("c_fetch", "v_inter").

        call("c_run", "a1").
        call_effect("c_run", "proc_exec").
        call_input("c_run", "v_target").
        call_unconditional("c_fetch", "c_run").
        call_code("c_run", "obfuscated").

        call("c_egress", "a1").
        call_effect("c_egress", "net_write").
        call_region_untrusted("c_egress").
        call_input("c_egress", "v_secret_local").

        call("c_creds", "a1").
        call_effect("c_creds", "fs_read").
        call_region_secret("c_creds").
        """,
    ),
    (
        # Exercises !gate_strong and !gated_action denials. The action carries
        # a human_approval gate, so even with a proc_exec on attacker-derived
        # input and a secret-bearing egress, the gated findings stay silent;
        # only label_dangerous_execution_primitives (no gate denial) fires.
        "gated_strong_suppresses_high_priv_findings",
        """
        skill("g").
        action("a", "g").
        action_gate("a", "human_approval").
        action_param("a", "user_input", "v_in").

        value("v_in", "a", "param").
        value("v_secret", "a", "secret").

        call("c_exec", "a").
        call_effect("c_exec", "proc_exec").
        call_input("c_exec", "v_in").

        call("c_egress", "a").
        call_effect("c_egress", "net_write").
        call_region_untrusted("c_egress").
        call_input("c_egress", "v_secret").
        """,
    ),
    (
        # Exercises the cross-action LLM-mediated data_flows over-approximation
        # (v_confused_deputy). a_ext's net_read output and incoming webhook
        # parameter must reach a_priv's proc_exec input via the cross-action
        # rule — they have no direct call_action / call_action_arg edge.
        "cross_action_confused_deputy",
        """
        skill("cd").

        action("a_ext", "cd").
        action_trigger("a_ext", "external").
        action_param("a_ext", "webhook_payload", "v_in_ext").

        value("v_in_ext", "a_ext", "param").
        value("v_out_ext", "a_ext", "derived").

        call("c_recv", "a_ext").
        call_effect("c_recv", "net_read").
        call_region_untrusted("c_recv").
        call_input("c_recv", "v_in_ext").
        call_output("c_recv", "v_out_ext").

        action("a_priv", "cd").
        action_param("a_priv", "cmd", "v_in_priv").

        value("v_in_priv", "a_priv", "param").

        call("c_run", "a_priv").
        call_effect("c_run", "proc_exec").
        call_input("c_run", "v_in_priv").
        """,
    ),
    (
        # Exercises the secret + untrusted co-ingress rule on
        # label_sensitive_local_resource_overreach. The fixture deliberately
        # disables the egress and scope variants (stdout effect = no
        # untrusted_egress_call; value_secret_allowed_action present), so any
        # row on the secret co-ingress head proves the migrated rule fired.
        "secret_untrusted_co_ingress",
        """
        skill("co").
        action("a", "co").
        action_param("a", "query", "v_query").
        action_param("a", "api_key", "v_secret").

        value("v_query", "a", "param").
        value("v_secret", "a", "secret").

        value_secret_allowed_action("v_secret", "a").

        call("c_local", "a").
        call_effect("c_local", "stdout").
        call_input("c_local", "v_query").
        call_input("c_local", "v_secret").
        """,
    ),
    (
        # Exercises label_dormant_malicious_payload's dead_obfuscated /
        # dead_encoded_binary sub-patterns. c_a and c_b sit in a mutual cycle,
        # so neither is a root_call and neither is reachable_call. The cycle
        # is the only structural shape that produces unreachable calls under
        # the current schema.
        "dead_obfuscated_payload",
        """
        skill("dp").
        action("a", "dp").

        call("c_a", "a").
        call_effect("c_a", "stdout").
        call_code("c_a", "obfuscated").

        call("c_b", "a").
        call_effect("c_b", "stdout").
        call_code("c_b", "encoded_binary").

        call_unconditional("c_a", "c_b").
        call_unconditional("c_b", "c_a").
        """,
    ),
    (
        # Exercises label_hardcoded_c2_communication (the previously
        # zero-coverage label) and, as a bonus, the call_code
        # "unresolved_target" head of label_unverifiable_dependency_source.
        "hardcoded_c2_unresolved_target",
        """
        skill("c2").
        action("a", "c2").

        value("v_target", "a", "literal").

        call("c_beacon", "a").
        call_effect("c_beacon", "net_read").
        call_region("c_beacon", "v_target").
        call_region_untrusted("c_beacon").
        call_code("c_beacon", "unresolved_target").
        """,
    ),
    (
        # Exercises the sensitive_local variants of three labels:
        #   - label_implicit_egress_channels (sensitive head)
        #   - label_shadow_credentials (sensitive_local head, via fs_list +
        #     region_sensitive + skill_has_untrusted_egress)
        #   - label_sensitive_local_resource_overreach heads 2/4/6 (egress,
        #     scope, co-ingress) for the sensitive variant
        # v_sens_other is allow-listed so the co-ingress row on c_local is
        # uniquely attributable to head 6 — scope (head 4) is suppressed for
        # that value, and c_local has stdout effect so head 2 doesn't apply.
        "sensitive_local_pipeline",
        """
        skill("sl").
        action("a", "sl").
        action_param("a", "destination", "v_dest").

        value("v_dest", "a", "param").
        value("v_sens", "a", "sensitive_local").
        value("v_sens_other", "a", "sensitive_local").

        value_sensitive_allowed_action("v_sens_other", "a").

        call("c_recon", "a").
        call_effect("c_recon", "fs_list").
        call_region_sensitive("c_recon").

        call("c_egress", "a").
        call_effect("c_egress", "net_write").
        call_region_untrusted("c_egress").
        call_input("c_egress", "v_sens").

        call("c_local", "a").
        call_effect("c_local", "stdout").
        call_input("c_local", "v_sens_other").
        call_input("c_local", "v_dest").
        """,
    ),
    (
        # Exercises four label_dangerous_execution_primitives heads that the
        # earlier fixtures don't hit: code_eval in an untrusted region (via
        # derived taint from a param input), proc_exec + encoded_binary,
        # code_eval + encoded_binary, and code_eval + obfuscated.
        "code_eval_and_encoded_payloads",
        """
        skill("ce").
        action("a", "ce").
        action_param("a", "code", "v_code").

        value("v_code", "a", "param").

        call("c_eval_untrusted", "a").
        call_effect("c_eval_untrusted", "code_eval").
        call_input("c_eval_untrusted", "v_code").

        call("c_proc_enc", "a").
        call_effect("c_proc_enc", "proc_exec").
        call_code("c_proc_enc", "encoded_binary").

        call("c_eval_enc", "a").
        call_effect("c_eval_enc", "code_eval").
        call_code("c_eval_enc", "encoded_binary").

        call("c_eval_obf", "a").
        call_effect("c_eval_obf", "code_eval").
        call_code("c_eval_obf", "obfuscated").
        """,
    ),
    (
        # Exercises label_unverifiable_dependency_source's agent_call → exec
        # head, and label_dormant_malicious_payload's remote_killswitch
        # (taint-based: a high-priv call gated by a value whose source is
        # marked untrusted in its kind declaration).
        "agent_dropper_and_killswitch",
        """
        skill("ak").
        action("a", "ak").

        value("v_agent_out", "a", "derived").
        value("v_cond", "a", "untrusted").

        call("c_agent", "a").
        call_effect("c_agent", "agent_call").
        call_region_untrusted("c_agent").
        call_output("c_agent", "v_agent_out").

        call("c_run", "a").
        call_effect("c_run", "proc_exec").
        call_input("c_run", "v_agent_out").
        call_unconditional("c_agent", "c_run").

        call("c_check", "a").
        call_effect("c_check", "fs_read").
        call_output("c_check", "v_cond").

        call("c_chain", "a").
        call_effect("c_chain", "chain_write").
        call_conditional("c_check", "c_chain", "v_cond").
        """,
    ),
    (
        # Verifies the !value(sink, _, "literal") denial on
        # label_unsanitized_context_ingestion. The untrusted param v_in
        # reaches both v_const (literal) and v_var (derived) by flowing
        # through c_load's outputs, which two downstream proc_exec calls
        # then consume separately. Without the denial both proc_exec calls
        # would yield UCI rows; the denial drops only the literal-sink one.
        # label_dangerous_execution_primitives has no such denial and fires
        # on both, so the suppression is observable as rule-scoped.
        "literal_sink_denial_in_indirect_injection",
        """
        skill("lit").
        action("a", "lit").
        action_param("a", "input", "v_in").

        value("v_in", "a", "param").
        value("v_const", "a", "literal").
        value("v_var", "a", "derived").

        call("c_load", "a").
        call_effect("c_load", "fs_read").
        call_input("c_load", "v_in").
        call_output("c_load", "v_const").
        call_output("c_load", "v_var").

        call("c_exec_const", "a").
        call_effect("c_exec_const", "proc_exec").
        call_input("c_exec_const", "v_const").
        call_unconditional("c_load", "c_exec_const").

        call("c_exec_var", "a").
        call_effect("c_exec_var", "proc_exec").
        call_input("c_exec_var", "v_var").
        call_unconditional("c_load", "c_exec_var").
        """,
    ),
)


def _stage_run_dir(tmp_root: Path, scenario_facts: str) -> Path:
    rules_dst = tmp_root / "rules" / "sdl"
    rules_dst.mkdir(parents=True)
    for name in ("skill_description_lang.dl", "skill_dl_static_analysis.dl"):
        (rules_dst / name).write_text(
            (RULES_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    facts_path = tmp_root / "detection_input.dl"
    header = '#include "rules/sdl/skill_dl_static_analysis.dl"\n\n'
    facts_path.write_text(header + scenario_facts.strip() + "\n", encoding="utf-8")
    return facts_path


def _read_relation_csvs(out_dir: Path) -> dict[str, list[tuple[str, ...]]]:
    relations: dict[str, list[tuple[str, ...]]] = {}
    for csv_path in sorted(out_dir.glob("*.csv")):
        text = csv_path.read_text(encoding="utf-8").strip()
        rows = [tuple(line.split("\t")) for line in text.split("\n")] if text else []
        relations[csv_path.stem] = sorted(rows)
    return relations


class ParserTests(unittest.TestCase):
    def test_rejects_unsupported_directive(self) -> None:
        with self.assertRaises(ParseError):
            parse_dl_file(self._write_temp(".aggregate foo\n"))

    def test_parses_decl_output_facts_and_rules(self) -> None:
        path = self._write_temp(
            """
            .decl edge(a: symbol, b: symbol)
            .decl path(a: symbol, b: symbol)
            .output path

            edge("x", "y").
            edge("y", "z").
            path(a, b) :- edge(a, b).
            path(a, c) :- edge(a, b), path(b, c).
            """
        )
        program = parse_dl_file(path)
        self.assertIn("edge", program.decls)
        self.assertIn("path", program.outputs)
        self.assertEqual(program.facts["edge"], {("x", "y"), ("y", "z")})
        self.assertEqual(len(program.rules), 2)

    def test_disjunction_expands_to_multiple_rules(self) -> None:
        path = self._write_temp(
            """
            .decl r(x: symbol)
            .output r
            r(x) :- s(x), (t(x); u(x)).
            """
        )
        program = parse_dl_file(path)
        self.assertEqual(len(program.rules), 2)

    def test_anonymous_variable_names_reset_per_parse(self) -> None:
        path = self._write_temp(
            """
            .decl edge(a: symbol, b: symbol)
            .decl root(a: symbol)
            .output root
            edge("a", "b").
            root(x) :- edge(x, _), !edge(_, x).
            """
        )

        def _anon_names(program: Program) -> list[str]:
            names: list[str] = []
            for rule in program.rules:
                for atom in rule.body:
                    for term in atom.args:
                        if term.is_var and term.value.startswith("_anon_"):
                            names.append(term.value)
            return names

        first = parse_dl_file(path)
        second = parse_dl_file(path)
        first_names = _anon_names(first)
        second_names = _anon_names(second)
        self.assertEqual(first_names, second_names)
        self.assertTrue(first_names, "expected anon variables to be present")
        self.assertEqual(min(first_names), "_anon_1")

    def _write_temp(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "p.dl"
        path.write_text(body, encoding="utf-8")
        self.addCleanup(shutil.rmtree, tmp)
        return path


class StratificationTests(unittest.TestCase):
    def test_negative_cycle_rejected(self) -> None:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(
            """
            .decl a(x: symbol)
            .decl b(x: symbol)
            a(x) :- !b(x), seed(x).
            b(x) :- !a(x), seed(x).
            seed("z").
            """,
            encoding="utf-8",
        )
        program = parse_dl_file(path)
        with self.assertRaises(EvalError):
            evaluate(program)

    def test_deep_predicate_chain_does_not_recurse(self) -> None:
        chain_len = 2000
        src = (
            "\n".join(
                f'.decl p_{i}(x: symbol)\np_{i}("x") :- p_{i + 1}("x").' for i in range(chain_len)
            )
            + f'\n.decl p_{chain_len}(x: symbol)\n.output p_0\np_{chain_len}("x").\n'
        )
        path = Path(tempfile.mkdtemp()) / "chain.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(src, encoding="utf-8")

        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(900)
        self.addCleanup(sys.setrecursionlimit, original_limit)

        program = parse_dl_file(path)
        relations = evaluate(program)
        self.assertEqual(relations["p_0"], {("x",)})
        self.assertEqual(relations[f"p_{chain_len}"], {("x",)})


class EvaluatorTests(unittest.TestCase):
    def test_evaluate_times_out_when_deadline_already_expired(self) -> None:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(
            """
            .decl seed(x: symbol)
            seed("x").
            """,
            encoding="utf-8",
        )
        program = parse_dl_file(path)

        with self.assertRaisesRegex(EvalError, "timed out"):
            evaluate(program, timeout_seconds=0)

    def test_evaluate_rejects_excessive_derived_tuples(self) -> None:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(
            """
            .decl seed(x: symbol)
            .decl derived(x: symbol)
            seed("a").
            seed("b").
            derived(x) :- seed(x).
            """,
            encoding="utf-8",
        )
        program = parse_dl_file(path)

        with self.assertRaisesRegex(EvalError, "more than 3 tuples"):
            evaluate(program, max_derived_tuples=3)

    def test_transitive_closure(self) -> None:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(
            """
            .decl edge(a: symbol, b: symbol)
            .decl path(a: symbol, b: symbol)
            .output path
            edge("x", "y").
            edge("y", "z").
            edge("z", "w").
            path(a, b) :- edge(a, b).
            path(a, c) :- edge(a, b), path(b, c).
            """,
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory() as out:
            run_evaluator(path, out)
            csv_text = (Path(out) / "path.csv").read_text(encoding="utf-8").strip()
            rows = sorted(csv_text.split("\n"))
        self.assertEqual(
            rows,
            sorted(["x\ty", "y\tz", "z\tw", "x\tz", "y\tw", "x\tw"]),
        )

    def test_anonymous_wildcard_in_negation(self) -> None:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(
            """
            .decl edge(a: symbol, b: symbol)
            .decl root(a: symbol)
            .output root
            edge("a", "b").
            edge("b", "c").
            root(x) :- edge(x, _), !edge(_, x).
            """,
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory() as out:
            run_evaluator(path, out)
            text = (Path(out) / "root.csv").read_text(encoding="utf-8").strip()
        self.assertEqual(text, "a")

    def test_contains_builtin(self) -> None:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(
            """
            .decl param(name: symbol)
            .decl secret_param(name: symbol)
            .output secret_param
            param("user_token").
            param("display_name").
            secret_param(n) :- param(n), (contains("token", n); contains("password", n)).
            """,
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory() as out:
            run_evaluator(path, out)
            text = (Path(out) / "secret_param.csv").read_text(encoding="utf-8").strip()
        self.assertEqual(text, "user_token")


class SDLRulesEndToEndTests(unittest.TestCase):
    def test_minimal_scenario_produces_expected_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[0][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        self.assertEqual(
            relations["label_dangerous_execution_primitives"],
            [("demo", "act", "c_exec")],
        )
        self.assertEqual(
            relations["label_ungated_irreversible_operation"],
            [("demo", "act", "c_exec")],
        )
        self.assertEqual(
            relations["label_unsanitized_context_ingestion"],
            [("demo", "act", "c_exec", "v_payload")],
        )

    def test_rich_scenario_produces_expected_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[1][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        self.assertEqual(
            relations["label_behavior_claim_contradiction"],
            [("sk", "a1", "c_egress", "read_only", "net_write")],
        )
        self.assertEqual(
            relations["label_obfuscation"],
            [("sk", "a1", "c_run", "obfuscated")],
        )
        self.assertEqual(len(relations["label_implicit_egress_channels"]), 2)
        self.assertEqual(len(relations["label_unsanitized_context_ingestion"]), 3)
        self.assertEqual(len(relations["label_unverifiable_dependency_source"]), 1)
        # c_creds (fs_read + region_secret) + c_egress (untrusted egress) ⇒
        # the migrated skill_has_untrusted_egress precondition is satisfied.
        self.assertEqual(
            relations["label_shadow_credentials"],
            [("sk", "a1", "c_creds", "secret")],
        )

    def test_gated_strong_suppresses_high_priv_findings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[2][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        # human_approval gate triggers all three gate-aware denials.
        self.assertEqual(relations.get("label_unsanitized_context_ingestion", []), [])
        self.assertEqual(relations.get("label_implicit_egress_channels", []), [])
        self.assertEqual(relations.get("label_ungated_irreversible_operation", []), [])
        # Rules without a gate denial still fire — proves the suppression is
        # surgical, not blanket.
        self.assertIn(
            ("g", "a", "c_exec"),
            relations.get("label_dangerous_execution_primitives", []),
        )

    def test_cross_action_confused_deputy_routes_taint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[3][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        rows = relations.get("label_unsanitized_context_ingestion", [])
        # The cross-action LLM-mediated data_flows rule lets v_out_ext (the
        # net_read output of a_ext) and v_in_ext (its webhook param) reach
        # c_run's proc_exec input in a_priv. There is no call_action /
        # call_action_arg edge between the two actions; only the migrated
        # over-approximation can connect them.
        self.assertIn(("cd", "a_priv", "c_run", "v_out_ext"), rows)
        self.assertIn(("cd", "a_priv", "c_run", "v_in_ext"), rows)

    def test_secret_untrusted_co_ingress_fires(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[4][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        rows = relations.get("label_sensitive_local_resource_overreach", [])
        # stdout effect kills the egress variant; value_secret_allowed_action
        # kills the scope variant. The remaining co-ingress head is the only
        # rule that can produce this row.
        self.assertIn(("co", "a", "c_local", "secret", "v_secret"), rows)

    def test_dead_obfuscated_payload_fires(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[5][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        rows = relations.get("label_dormant_malicious_payload", [])
        self.assertIn(("dp", "a", "c_a", "dead_obfuscated"), rows)
        self.assertIn(("dp", "a", "c_b", "dead_encoded_binary"), rows)

    def test_hardcoded_c2_unresolved_target_fires(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[6][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        self.assertIn(
            ("c2", "a", "c_beacon", "v_target"),
            relations.get("label_hardcoded_c2_communication", []),
        )
        # Bonus: same fixture exercises the unresolved_target head of
        # label_unverifiable_dependency_source.
        self.assertIn(
            ("c2", "a", "c_beacon", "c_beacon"),
            relations.get("label_unverifiable_dependency_source", []),
        )

    def test_sensitive_local_pipeline_fires_all_sensitive_heads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[7][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        # IEC sensitive head
        self.assertIn(
            ("sl", "a", "c_egress", "sensitive_local", "v_sens"),
            relations.get("label_implicit_egress_channels", []),
        )
        # SC sensitive head — fs_list in sensitive region + skill_has_untrusted_egress
        self.assertIn(
            ("sl", "a", "c_recon", "sensitive_local"),
            relations.get("label_shadow_credentials", []),
        )
        slo_rows = relations.get("label_sensitive_local_resource_overreach", [])
        # SLO egress + sensitive (head 2) — same row as scope (head 4)
        self.assertIn(("sl", "a", "c_egress", "sensitive_local", "v_sens"), slo_rows)
        # SLO co-ingress sensitive (head 6) — uniquely attributable: v_sens_other
        # is allow-listed (scope head suppressed for it) and c_local has stdout
        # effect (egress head doesn't apply).
        self.assertIn(("sl", "a", "c_local", "sensitive_local", "v_sens_other"), slo_rows)

    def test_code_eval_and_encoded_payloads_fire(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[8][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        rows = relations.get("label_dangerous_execution_primitives", [])
        # Each of the four remaining DEP heads gets its own call so the rows
        # are individually attributable.
        self.assertIn(("ce", "a", "c_eval_untrusted"), rows)
        self.assertIn(("ce", "a", "c_proc_enc"), rows)
        self.assertIn(("ce", "a", "c_eval_enc"), rows)
        self.assertIn(("ce", "a", "c_eval_obf"), rows)

    def test_agent_dropper_and_killswitch_fire(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[9][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        # UDS rule 3: agent_call output reaches proc_exec input
        self.assertIn(
            ("ak", "a", "c_run", "c_agent"),
            relations.get("label_unverifiable_dependency_source", []),
        )
        # DMP remote_killswitch: high-priv call gated by an untrusted-kind value
        self.assertIn(
            ("ak", "a", "c_chain", "remote_killswitch"),
            relations.get("label_dormant_malicious_payload", []),
        )

    def test_literal_sink_denial_silences_uci_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            facts = _stage_run_dir(Path(td), SCENARIOS[10][1])
            out = Path(td) / "out"
            run_evaluator(facts, out)
            relations = _read_relation_csvs(out)
        # UCI fires only on the non-literal sink. Without the !value literal
        # denial, c_exec_const would also produce a row.
        self.assertEqual(
            relations.get("label_unsanitized_context_ingestion", []),
            [("lit", "a", "c_exec_var", "v_in")],
        )
        # DEP has no literal denial, so it still fires on both proc_exec calls.
        dep = relations.get("label_dangerous_execution_primitives", [])
        self.assertIn(("lit", "a", "c_exec_const"), dep)
        self.assertIn(("lit", "a", "c_exec_var"), dep)

    @unittest.skipUnless(shutil.which("souffle"), "souffle not installed")
    def test_builtin_matches_souffle_on_all_scenarios(self) -> None:
        for name, body in SCENARIOS:
            with self.subTest(scenario=name):
                with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
                    fa = _stage_run_dir(Path(td_a), body)
                    fb = _stage_run_dir(Path(td_b), body)
                    out_a = Path(td_a) / "out"
                    out_b = Path(td_b) / "out"
                    out_b.mkdir()
                    run_evaluator(fa, out_a)
                    subprocess.run(
                        [shutil.which("souffle"), str(fb), "-D", str(out_b)],
                        check=True,
                        capture_output=True,
                    )
                    builtin = _read_relation_csvs(out_a)
                    souffle = _read_relation_csvs(out_b)
                self.assertEqual(
                    set(builtin.keys()),
                    set(souffle.keys()),
                    msg=f"output relation set differs for scenario {name}",
                )
                for key in builtin:
                    self.assertEqual(
                        builtin[key],
                        souffle[key],
                        msg=f"relation {key} differs for scenario {name}",
                    )


class EqBuiltinTests(unittest.TestCase):
    """Cover the variable-binding branches of `_match_builtin` for `eq`/`neq`.

    The bound==bound and disequality-with-mismatch branches are exercised
    indirectly by the SDL rule scenarios; the variable-binding paths only
    fire when one or both operands of `=` are unbound at the point the
    builtin is reached, which the rule corpus does not happen to trigger.
    """

    def _run_program(self, dl_text: str) -> dict[str, set[str]]:
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(dl_text, encoding="utf-8")
        with tempfile.TemporaryDirectory() as out:
            run_evaluator(path, out)
            out_dir = Path(out)
            return {
                csv.stem: {line for line in csv.read_text(encoding="utf-8").splitlines() if line}
                for csv in out_dir.glob("*.csv")
            }

    def test_eq_binds_unbound_left_var_to_constant_right(self) -> None:
        # `x = "v"` in body: left is unbound variable, right is a constant.
        # Exercises engine.py 196-202 (left unbound, right bound).
        program = """
        .decl out(v: symbol)
        .output out
        out(x) :- x = "literal".
        """
        result = self._run_program(program)
        self.assertEqual(result, {"out": {"literal"}})

    def test_eq_binds_unbound_right_var_to_constant_left(self) -> None:
        # `"v" = x`: symmetric to the above, exercises engine.py 203-209.
        program = """
        .decl out(v: symbol)
        .output out
        out(x) :- "literal" = x.
        """
        result = self._run_program(program)
        self.assertEqual(result, {"out": {"literal"}})

    def test_eq_raises_when_both_sides_unbound(self) -> None:
        # Both `y` and `z` are body-only and unbound when the builtin runs;
        # neither can be bound from the other. Exercises engine.py 210.
        from semia_core.datalog_eval.engine import EvalError

        program = """
        .decl src(v: symbol)
        .decl out(v: symbol)
        .output out
        src("a").
        out(x) :- src(x), y = z.
        """
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(program, encoding="utf-8")
        with tempfile.TemporaryDirectory() as out_dir, self.assertRaises(EvalError):
            run_evaluator(path, out_dir)

    def test_neq_raises_when_either_side_unbound(self) -> None:
        # Disequality requires both arguments bound — exercises engine.py 212.
        from semia_core.datalog_eval.engine import EvalError

        program = """
        .decl src(v: symbol)
        .decl out(v: symbol)
        .output out
        src("a").
        out(x) :- src(x), y != "z".
        """
        path = Path(tempfile.mkdtemp()) / "p.dl"
        self.addCleanup(shutil.rmtree, path.parent)
        path.write_text(program, encoding="utf-8")
        with tempfile.TemporaryDirectory() as out_dir, self.assertRaises(EvalError):
            run_evaluator(path, out_dir)


class ParserErrorRecoveryTests(unittest.TestCase):
    """Exercise the parser's lexical edge cases: string escapes, block
    comments, malformed atoms, and unsupported constructs.

    These paths are reached only by unusual or hostile input, so the SDL
    corpus does not cover them. They matter because the parser is the entry
    point for untrusted facts coming back from synthesis.
    """

    def test_parse_dl_text_entrypoint_returns_program(self) -> None:
        # Exercises parser.py 86-89 (the text variant of parse_dl_file).
        from semia_core.datalog_eval.parser import parse_dl_text

        program = parse_dl_text(
            """
            .decl src(x: symbol)
            .output src
            src("a").
            """
        )
        self.assertIn("src", program.facts)
        self.assertEqual(program.facts["src"], {("a",)})

    def test_block_comments_are_stripped(self) -> None:
        # Exercises parser.py 191-197 (block comment handler).
        from semia_core.datalog_eval.parser import parse_dl_text

        program = parse_dl_text(
            """
            .decl src(x: symbol)
            .output src
            /* this fact is commented out
               and spans multiple lines.
            src("ignored").
            */
            src("kept").
            """
        )
        self.assertEqual(program.facts["src"], {("kept",)})

    def test_strip_comments_preserves_escaped_quote_in_string(self) -> None:
        # Exercises parser.py 173-175 (escape inside quoted string while
        # stripping comments — `\\` must not terminate the string).
        from semia_core.datalog_eval.parser import parse_dl_text

        program = parse_dl_text(
            r"""
            .decl src(x: symbol)
            .output src
            src("a\"b").
            """
        )
        self.assertEqual(program.facts["src"], {('a"b',)})

    def test_unescape_string_handles_all_known_escapes(self) -> None:
        # Exercises parser.py 468-471 (escape mapping: \n, \t, \r, \", \\,
        # and unknown sequences which fall through to the literal char).
        from semia_core.datalog_eval.parser import parse_dl_text

        program = parse_dl_text(
            r"""
            .decl src(x: symbol)
            .output src
            src("nl\nhere").
            src("tab\there").
            src("cr\rhere").
            src("bs\\here").
            src("unknown\xhere").
            """
        )
        self.assertEqual(
            program.facts["src"],
            {
                ("nl\nhere",),
                ("tab\there",),
                ("cr\rhere",),
                ("bs\\here",),
                ("unknownxhere",),
            },
        )

    def test_malformed_atom_raises_parse_error(self) -> None:
        # Exercises parser.py 437 (open_idx <= 0 or close mismatch).
        from semia_core.datalog_eval.parser import parse_dl_text

        with self.assertRaises(ParseError):
            parse_dl_text('.decl src(x: symbol)\nsrc("a"\n')  # missing close paren

    def test_invalid_relation_name_raises(self) -> None:
        # Exercises parser.py 440 (relation name doesn't match identifier).
        from semia_core.datalog_eval.parser import parse_dl_text

        with self.assertRaises(ParseError):
            parse_dl_text('.decl src(x: symbol)\n123bad("a").\n')

    def test_unrecognized_term_raises(self) -> None:
        # Exercises parser.py 458 (term is not _, string, int, or ident).
        from semia_core.datalog_eval.parser import parse_dl_text

        with self.assertRaises(ParseError):
            parse_dl_text(".decl src(x: symbol)\nsrc(@@@bad).\n")

    def test_empty_term_raises(self) -> None:
        # Exercises parser.py 449 (empty argument).
        from semia_core.datalog_eval.parser import parse_dl_text

        with self.assertRaises(ParseError):
            parse_dl_text('.decl src(x: symbol, y: symbol)\nsrc("a", ).\n')

    def test_explicit_negation_on_equality_rejected(self) -> None:
        # Exercises parser.py 419 (negation on builtins not allowed).
        from semia_core.datalog_eval.parser import parse_dl_text

        with self.assertRaises(ParseError):
            parse_dl_text(
                """
                .decl src(x: symbol)
                .decl out(x: symbol)
                .output out
                src("a").
                out(x) :- src(x), !(x = "a").
                """
            )


if __name__ == "__main__":
    unittest.main()
