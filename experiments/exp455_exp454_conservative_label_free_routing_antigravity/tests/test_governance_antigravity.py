from __future__ import annotations

import html.parser
import importlib.util
import json
from pathlib import Path
import re
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP455_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP455_RUNBOOK_antigravity_KO.html",
    "docs/EXP455_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP455_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp455_antigravity.html",
    "docs/codemap/codemap_exp455_antigravity.json",
    "docs/codemap/codemap_exp455_antigravity.lock",
)


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_terminal_governance_is_failed_and_non_promoted_antigravity(self) -> None:
        """Break caught: terminal failure is hidden or represented as a promoted result."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertEqual(path.suffix, ".html")
            parser = HtmlParserAntigravity()
            parser.feed(path.read_text(encoding="utf-8"))
            parser.close()
        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for value in ("official_evaluation_failed_immutable_no_retry", "1회, exit 1", "0회", "결과 폴더</th><td>없음", "성능 결과</th><td>계산되지 않음"):
            self.assertIn(value, dashboard)

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        exp455_state = state["exp455"]
        exp455_catalog = next(item for item in catalog["experiments"] if item["id"] == "Exp455")
        exp455_champion = next(item for item in champion["history"] if item["experiment"] == "Exp455")
        for record in (exp455_state, exp455_catalog, exp455_champion):
            self.assertEqual(record["kind"], "conservative_label_free_routing")
            self.assertEqual(record["status"], "official_evaluation_failed_immutable_no_retry")
            self.assertEqual(record["implementation_status"], "verified_implementation_official_evaluation_failed")
            self.assertEqual(record["official_lifecycle"]["evaluate_invocations"], 1)
            self.assertEqual(record["official_lifecycle"]["evaluate_exit_code"], 1)
            self.assertEqual(record["official_lifecycle"]["inspect_invocations"], 0)
            self.assertEqual(record["official_lifecycle"]["label_access_count"], 0)
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
        self.assertEqual(champion["pending_candidate"], state["pending_candidate"])

    def test_task_six_terminal_progress_and_map_boundaries_antigravity(self) -> None:
        """Break caught: Task 6 hides the failed official outcome or invents a result."""
        plan = (ROOT_ANTIGRAVITY / "docs/superpowers/plans/2026-08-17-exp455-conservative-label-free-routing-antigravity.html").read_text(encoding="utf-8")
        task_one = re.search(r"<section>\s*<h2><input\b[^>]*checked[^>]*>Task 1:.*?</section>", plan, flags=re.DOTALL)
        self.assertIsNotNone(task_one)
        task_three = re.search(r"<section>\s*<h2><input\b[^>]*checked[^>]*>Task 3:.*?</section>", plan, flags=re.DOTALL)
        self.assertIsNotNone(task_three)
        task_four = re.search(r"<section>\s*<h2><input\b[^>]*checked[^>]*>Task 4:.*?</section>", plan, flags=re.DOTALL)
        self.assertIsNotNone(task_four)
        task_five = re.search(r"<section>\s*<h2><input\b[^>]*checked[^>]*>Task 5:.*?</section>", plan, flags=re.DOTALL)
        self.assertIsNotNone(task_five)
        task_six = re.search(r"<section>\s*<h2><input\b[^>]*checked[^>]*>Task 6:.*?</section>", plan, flags=re.DOTALL)
        self.assertIsNotNone(task_six)
        assert task_six is not None
        self.assertIn("features must match authority targets in order", task_six.group(0))
        self.assertIn('<input checked disabled type="checkbox">공식 결과가 있을 때 Codex가 8개 fold', task_six.group(0))
        self.assertIn('fresh Terra High read-only review <code>CLEAN</code>', task_six.group(0))

        spec = importlib.util.spec_from_file_location("exp455_map_builder_antigravity", ROOT_ANTIGRAVITY / "scratch/build_codemap_exp455_antigravity.py")
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        nodes = {node["id"]: node for node in graph["nodes"]}
        for node_id in ("config", "contracts", "authority", "features", "router", "evaluation", "artifacts", "replay", "dashboard_export"):
            self.assertIn(node_id, nodes)
        self.assertNotIn("status", nodes["router"])
        self.assertNotIn("status", nodes["evaluation"])
        builder.validate_map_antigravity(graph, lock, rendered)
