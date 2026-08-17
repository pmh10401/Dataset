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
    def test_terminal_governance_dashboard_and_map_antigravity(self) -> None:
        """Break caught: Task 5 terminal truth, aggregate dashboard, or map drifts."""
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
        expected_hashes = {
            "evaluation_sha256": "6117a5f2ef28a3d543efcf4b499dd715508a4345a5e92679e1c14f0b5e1087cc",
            "decision_sha256": "ea49b140475d4c9d8245a85ec0479a60f1ba6c46e8763b0c35877ec16ca6dc01",
            "artifact_map_sha256": "3c1321f74016d363899b41357609edca62ecf410d71708a9c90a6b803a4445e0",
            "dashboard_snapshot_sha256": "7a9196efdc8764b53a5810f169bcfd7b12d6bfab1bf383ff024421b791fea407",
        }
        expected_comparisons = [
            {"name": "always_baseline", "fn": 837, "fp": 66, "f1": 0.5742574257425742},
            {"name": "family_wide_local_gap", "fn": 835, "fp": 63, "f1": 0.5764150943396227},
            {"name": "oracle_per_dataset", "fn": 828, "fp": 64, "f1": 0.5808270676691729},
            {"name": "lodo_router", "fn": 843, "fp": 65, "f1": 0.5704824976348155},
        ]
        for record in (exp456_state, exp456_catalog, exp456_champion):
            self.assertEqual(record["kind"], "exp455_dataset_key_order_repair")
            self.assertEqual(record["status"], "official_evaluation_inspected_complete_not_qualified")
            self.assertEqual(record["label_query_count"], 0)
            self.assertTrue(record["official_evaluate_started"])
            self.assertTrue(record["official_inspect_started"])
            self.assertTrue(record["result_root_created"])
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertEqual({key: record[key] for key in expected_hashes}, expected_hashes)
            self.assertEqual(record["comparisons"], expected_comparisons)
            self.assertEqual(record["family_fn_delta"], {"ECG5000": 6, "ChlorineConcentration": 0})
            lifecycle = record["official_lifecycle"]
            self.assertEqual(lifecycle["evaluate_invocations"], 1)
            self.assertEqual(lifecycle["evaluate_exit_code"], 0)
            self.assertEqual(lifecycle["inspect_invocations"], 1)
            self.assertEqual(lifecycle["inspect_exit_code"], 0)
            self.assertEqual(lifecycle["label_query_count"], 0)
            self.assertTrue(lifecycle["aggregate_json_equal"])
            self.assertEqual(lifecycle["completed_folds"], 8)
            self.assertFalse(lifecycle["passed"])
            self.assertEqual(lifecycle["status"], "routing_candidate_not_qualified_for_external_validation")

        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for value in (
            "official_evaluation_inspected_complete_not_qualified",
            "Task 5/5 complete",
            "1회, exit 0",
            "routing_candidate_not_qualified_for_external_validation",
            "passed=false",
            "ECG5000</th><td>+6",
            "ChlorineConcentration</th><td>0",
        ):
            self.assertIn(value, dashboard)
        self.assertNotIn("not_available", dashboard)
        for comparison in expected_comparisons:
            for value in comparison.values():
                self.assertIn(str(value), dashboard)

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
        self.assertNotIn("status", nodes["evaluation"])
        self.assertTrue(nodes["evaluation"]["evidence"])
        self.assertIn("cli", nodes)
        self.assertNotIn("future_cli", nodes)
        self.assertFalse(graph["unknown_relationships"])
        self.assertEqual(
            builder.FUTURE_RELATIONSHIPS_ANTIGRAVITY,
            "Task 5 records the immutable official result; no Exp456 source relationship remains future or unknown.",
        )
        self.assertNotIn("Task 1", json.dumps(graph, ensure_ascii=False))
        self.assertIn("Task 5", nodes["dashboard"]["role"])
        self.assertIn("Task 5", nodes["governance_regression"]["role"])
        self.assertIn("Task 5", nodes["map_builder"]["role"])
        builder.validate_map_antigravity(graph, lock, rendered)

    def test_dashboard_exposes_only_approved_fields_and_reconciles_terminal_report_antigravity(self) -> None:
        """Break caught: the dashboard leaks a non-approved field or diverges from terminal result report."""
        dashboard = (ROOT_ANTIGRAVITY / "docs/EXP456_DASHBOARD_antigravity.html").read_text(encoding="utf-8")
        self.assertNotIn("TEST label 조회", dashboard)
        self.assertIn("raw TEST label이나 개별 index를 표시하지 않는다.", dashboard)
        self.assertNotIn("candidate_confusion", dashboard)
        report = (ROOT_ANTIGRAVITY / "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html").read_text(encoding="utf-8")
        for terminal_value in (
            "official_evaluation_inspected_complete_not_qualified",
            "routing_candidate_not_qualified_for_external_validation",
            "passed=false",
            "completed folds 8",
            "3c1321f74016d363899b41357609edca62ecf410d71708a9c90a6b803a4445e0",
            "6117a5f2ef28a3d543efcf4b499dd715508a4345a5e92679e1c14f0b5e1087cc",
            "ea49b140475d4c9d8245a85ec0479a60f1ba6c46e8763b0c35877ec16ca6dc01",
            "7a9196efdc8764b53a5810f169bcfd7b12d6bfab1bf383ff024421b791fea407",
            "ECG5000 <strong>+6</strong>",
            "ChlorineConcentration <strong>0</strong>",
        ):
            self.assertIn(terminal_value, report)
        for dashboard_value in (
            "837", "66", "0.5742574257425742", "835", "63", "0.5764150943396227",
            "828", "64", "0.5808270676691729", "843", "65", "0.5704824976348155",
        ):
            self.assertIn(dashboard_value, dashboard)
            self.assertIn(dashboard_value, report)

    def test_reports_bind_to_checked_in_map_source_parent_antigravity(self) -> None:
        """Break caught: a report states a different map source-parent than the checked-in map triplet."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        assert isinstance(graph, dict) and isinstance(lock, dict)
        source_parent = graph["source_parent_commit"]
        self.assertEqual(source_parent, lock["source_parent_commit"])
        map_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
        direct_parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
        map_only_paths = tuple(sorted(subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", map_commit],
            cwd=ROOT_ANTIGRAVITY,
            text=True,
        ).splitlines()))
        self.assertEqual(map_only_paths, tuple(sorted(TRIPLET_ANTIGRAVITY)))
        self.assertEqual(source_parent, direct_parent)
        implementation_report = (ROOT_ANTIGRAVITY / "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html").read_text(encoding="utf-8")
        self.assertIn("최종 checked-in map-only child는 바로 앞의 clean functional closeout commit을 source-parent로 bind한다.", implementation_report)
        self.assertIn("reviewed candidate <code>a45ff019e0736bbf895bb698b8eccfcbc69cd190</code>는 official execution input이며 map source-parent가 아니다.", implementation_report)
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
