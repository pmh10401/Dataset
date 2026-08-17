from __future__ import annotations

import html.parser
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
BASE_COMMIT_ANTIGRAVITY = "24722fabe8ae9f0c89ee4f9c28cb4f3c68f78b7b"
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP456_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP456_RUNBOOK_antigravity_KO.html",
    "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP456_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp456_antigravity.html",
    "docs/codemap/codemap_exp456_antigravity.json",
    "docs/codemap/codemap_exp456_antigravity.lock",
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


def record_at_base_antigravity(
    relative: str, experiment: str, collection: str | None = None, identifier: str = "experiment"
) -> object:
    text = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT_ANTIGRAVITY}:{relative}"], cwd=ROOT_ANTIGRAVITY, text=True
    )
    data = json.loads(text)
    if collection is None:
        return data[experiment.lower()]
    return next(item for item in data[collection] if item[identifier] == experiment)


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_planned_governance_dashboard_and_map_antigravity(self) -> None:
        """Break caught: Exp456 starts without planned-only truth or invents performance/source code."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY + (TRIPLET_ANTIGRAVITY[0],):
            parser = HtmlParserAntigravity()
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
            parser.close()

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        exp456_state = state["exp456"]
        exp456_catalog = next(item for item in catalog["experiments"] if item["id"] == "Exp456")
        exp456_champion = next(item for item in champion["history"] if item["experiment"] == "Exp456")
        for record in (exp456_state, exp456_catalog, exp456_champion):
            self.assertEqual(record["kind"], "exp455_dataset_key_order_repair")
            self.assertEqual(record["status"], "planned_awaiting_implementation")
            self.assertEqual(record["label_query_count"], 0)
            self.assertFalse(record["official_evaluate_started"])
            self.assertIsNone(record["evaluation_sha256"])
            self.assertFalse(record["result_root_created"])
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])

        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for value in (
            "planned_awaiting_implementation",
            "Task 2/5",
            "0회",
            "not_available",
            "result_root_created=false",
            "Task 3",
        ):
            self.assertIn(value, dashboard)

        spec = importlib.util.spec_from_file_location(
            "exp456_map_builder_antigravity", ROOT_ANTIGRAVITY / "scratch/build_codemap_exp456_antigravity.py"
        )
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(nodes["future_evaluation"]["status"], "future_unknown")
        self.assertEqual(nodes["future_evaluation"]["evidence"], [])
        self.assertTrue(graph["unknown_relationships"])
        builder.validate_map_antigravity(graph, lock, rendered)

    def test_dashboard_exposes_only_approved_fields_and_preflight_lists_all_coverage_antigravity(self) -> None:
        """Break caught: the dashboard leaks a non-approved field or preflight omits Exp455 evaluation coverage."""
        dashboard = (ROOT_ANTIGRAVITY / "docs/EXP456_DASHBOARD_antigravity.html").read_text(encoding="utf-8")
        self.assertNotIn("TEST label 조회", dashboard)
        report = (ROOT_ANTIGRAVITY / "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html").read_text(encoding="utf-8")
        for coverage_test in (
            "test_evaluation_antigravity.py",
            "test_replay_antigravity.py",
            "test_artifacts_antigravity.py",
            "test_dashboard_export_antigravity.py",
        ):
            self.assertIn(coverage_test, report)

    def test_reports_bind_to_checked_in_map_source_parent_antigravity(self) -> None:
        """Break caught: a report states a different map source-parent than the checked-in map triplet."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        assert isinstance(graph, dict) and isinstance(lock, dict)
        source_parent = graph["source_parent_commit"]
        self.assertEqual(source_parent, lock["source_parent_commit"])
        self.assertEqual(source_parent, "f244659c85c286a6d1ed087cbb3380cff3aaaee1")
        implementation_report = (ROOT_ANTIGRAVITY / "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html").read_text(encoding="utf-8")
        self.assertIn("checked-in map source-parent commit SHA는 <code>f244659c85c286a6d1ed087cbb3380cff3aaaee1</code>", implementation_report)
        self.assertIn("<code>77bb8b7de74de8a1a1a1a4ebf3b784cfdd681d17</code>은 Exp455 byte guard reference일 뿐 map binding이 아니다.", implementation_report)
        task_report = (ROOT_ANTIGRAVITY / ".superpowers/sdd/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity.html/task-1-report-antigravity.html").read_text(encoding="utf-8")
        self.assertIn("source-parent <code>17c6ffa100f70368f0c86efaa4a5f04890e4f12d</code>", task_report)
        self.assertIn("<code>24722fabe8ae9f0c89ee4f9c28cb4f3c68f78b7b</code> is the Task 1 implementation base and Exp455 deep-equality reference, not the map source-parent binding.", task_report)

    def test_exp455_and_champion_pending_are_preserved_antigravity(self) -> None:
        """Break caught: registering Exp456 changes Exp455 terminal evidence or champion/pending truth."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        self.assertEqual(state["exp455"], record_at_base_antigravity("RESEARCH_STATE.json", "Exp455"))
        self.assertEqual(
            next(item for item in catalog["experiments"] if item["id"] == "Exp455"),
            record_at_base_antigravity("EXPERIMENT_CATALOG.json", "Exp455", "experiments", "id"),
        )
        self.assertEqual(
            next(item for item in champion["history"] if item["experiment"] == "Exp455"),
            record_at_base_antigravity("CHAMPION_LEDGER.json", "Exp455", "history"),
        )
        baseline_state = json.loads(
            subprocess.check_output(
                ["git", "show", f"{BASE_COMMIT_ANTIGRAVITY}:RESEARCH_STATE.json"], cwd=ROOT_ANTIGRAVITY, text=True
            )
        )
        baseline_champion = json.loads(
            subprocess.check_output(
                ["git", "show", f"{BASE_COMMIT_ANTIGRAVITY}:CHAMPION_LEDGER.json"], cwd=ROOT_ANTIGRAVITY, text=True
            )
        )
        self.assertEqual(state["pending_candidate"], baseline_state["pending_candidate"])
        self.assertEqual(champion["pending_candidate"], baseline_champion["pending_candidate"])


if __name__ == "__main__":
    unittest.main()
