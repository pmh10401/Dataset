from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PACKAGE_ANTIGRAVITY = "experiments/exp465_ecg200_c8_external_validation_antigravity"
DESIGN_ANTIGRAVITY = "docs/superpowers/specs/2026-08-22-exp465-ecg200-c8-external-validation-antigravity-design.html"
PLAN_ANTIGRAVITY = "docs/superpowers/plans/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html"
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP465_DASHBOARD_antigravity.html",
    "docs/EXP465_PREREGISTRATION_antigravity.html",
    "docs/EXP465_RUNBOOK_antigravity.html",
    "docs/EXP465_IMPLEMENTATION_REPORT_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp465_antigravity.html",
    "docs/codemap/codemap_exp465_antigravity.json",
    "docs/codemap/codemap_exp465_antigravity.lock",
)
MAP_BUILDER_ANTIGRAVITY = "scratch/build_codemap_exp465_antigravity.py"
OFFICIAL_ROOT_TEXT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp465_ecg200_c8_external_validation_antigravity_20260822"
DESIGN_SHA256_ANTIGRAVITY = "a5f5f43e1787ad2942c883761ee95478e8cf5fcdcc958722a22fe0e1ad98d904"


def strict_json_antigravity(path: Path) -> Any:
    def reject_duplicates_antigravity(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


class _HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class CheckboxParserAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.checkbox_count = 0
        self.checked_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "input":
            return
        attr_map = dict(attrs)
        if attr_map.get("type") != "checkbox":
            return
        self.checkbox_count += 1
        if "checked" in attr_map:
            self.checked_count += 1


def load_builder_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / MAP_BUILDER_ANTIGRAVITY
    if not path.is_file():
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp465_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp465 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GovernanceAntigravityTest(unittest.TestCase):
    maxDiff = None

    def test_task1_truth_documents_ledgers_and_readonly_codemap_antigravity(self) -> None:
        """Break caught: Task 1 contract surface, ledger truth, checkbox state, or codemap drifts."""
        required_files = (
            DESIGN_ANTIGRAVITY,
            PLAN_ANTIGRAVITY,
            f"{PACKAGE_ANTIGRAVITY}/config_antigravity.py",
            f"{PACKAGE_ANTIGRAVITY}/contracts_antigravity.py",
            *DOCUMENTS_ANTIGRAVITY,
            *TRIPLET_ANTIGRAVITY,
            MAP_BUILDER_ANTIGRAVITY,
        )
        missing = [relative for relative in required_files if not (ROOT_ANTIGRAVITY / relative).is_file()]
        self.assertEqual(missing, [], f"missing Exp465 Task 1 artifacts: {missing}")
        for relative in required_files:
            path = Path(relative)
            if path.suffix in {".py", ".html", ".json", ".lock"}:
                self.assertIn("antigravity", path.name, relative)

        self.assertEqual(
            hashlib.sha256((ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes()).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )
        official_root = Path(OFFICIAL_ROOT_TEXT_ANTIGRAVITY)
        self.assertFalse(official_root.exists(), OFFICIAL_ROOT_TEXT_ANTIGRAVITY)
        self.assertFalse(official_root.is_symlink(), OFFICIAL_ROOT_TEXT_ANTIGRAVITY)

        parser = _HtmlParserAntigravity()
        for relative in DOCUMENTS_ANTIGRAVITY:
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
        parser.close()

        dashboard_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        for phrase in (
            "Task 1 계약·문서·코드맵 구현",
            "독립 재검토도 PASS",
            "8 tests GREEN × 2회",
            "code-map <code>--validate-only</code> 통과",
            "official evaluate / inspect <code>0 / 0</code>",
            "private run <code>0</code>",
            "label query <code>0</code>",
            "모든 결과 hash <code>null</code>",
        ):
            self.assertIn(phrase, dashboard_text)

        preregistration_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]).read_text(encoding="utf-8")
        for phrase in (
            "implementation_plan_written_awaiting_user_review",
            "808875aeac1c358a90c630fe369327c4",
            '("C0","C6","C8")',
            "≥ 0.85",
            "≥ 0.90",
            "정확히 <code>0</code>개",
            "Wilson 상한은 설명값이다",
            "PyTorch·MLX·MPS",
        ):
            self.assertIn(phrase, preregistration_text)

        runbook_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(encoding="utf-8")
        for phrase in (
            OFFICIAL_ROOT_TEXT_ANTIGRAVITY,
            "must stay absent until separate approval",
            "Do not open <code>ECG200_TEST.ts</code>",
            "--validate-only",
            "읽기 전용",
        ):
            self.assertIn(phrase, runbook_text)

        report_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for phrase in (
            "task_1_review_pass_complete",
            "RED",
            "두 번 모두 통과했다",
            "<code>git diff --check</code>",
            "canonical CPU fallback",
        ):
            self.assertIn(phrase, report_text)

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        exp465_state = state["exp465"]
        catalog_matches = [row for row in catalog["experiments"] if row.get("id") == "Exp465"]
        champion_matches = [row for row in champion["history"] if row.get("experiment") == "Exp465"]
        self.assertEqual(len(catalog_matches), 1)
        self.assertEqual(len(champion_matches), 1)
        catalog_record = catalog_matches[0]
        champion_record = champion_matches[0]
        for record in (exp465_state, catalog_record, champion_record):
            with self.subTest(record=record["experiment"]):
                self.assertEqual(record["experiment"], "Exp465")
                self.assertEqual(record["kind"], "ecg200_c8_external_train_qualification")
                self.assertEqual(record["status"], "implementation_plan_written_awaiting_user_review")
                self.assertEqual(record["implementation_status"], "task_1_contract_documents_and_codemap_complete")
                self.assertEqual(record["run_identity"], "exp465-ecg200-c8-external-validation-antigravity-20260822")
                self.assertEqual(record["run_root"], OFFICIAL_ROOT_TEXT_ANTIGRAVITY)
                self.assertEqual(record["package"], PACKAGE_ANTIGRAVITY)
                self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
                self.assertFalse(record["test_labels_opened"])
                self.assertEqual(record["label_query_count"], 0)
                self.assertFalse(record["routing_allowed"])
                self.assertFalse(record["promotion_allowed"])
                self.assertFalse(record["pending_candidate_changed"])
                self.assertFalse(record["scientific_winner_selected"])
                self.assertFalse(record["official_root_present"])
                self.assertIsNone(record["artifact_map_sha256"])
        scientific_contract = exp465_state["scientific_contract"]
        self.assertEqual(scientific_contract["train_shape"], [100, 96])
        self.assertEqual(scientific_contract["normal_label_plus_one_count"], 69)
        self.assertEqual(scientific_contract["method_order"], ["C0", "C6", "C8"])
        self.assertEqual(scientific_contract["primary_candidate"], "C8")
        self.assertEqual(scientific_contract["seed_pair"], [20260721, 20261721])
        self.assertEqual(scientific_contract["fold_count"], 5)
        self.assertEqual(scientific_contract["onset"], 56)
        self.assertEqual(scientific_contract["cosine_min"], 0.95)
        self.assertEqual(scientific_contract["per_type_auc_min"], 0.85)
        self.assertEqual(scientific_contract["mean_auc_min"], 0.9)
        self.assertEqual(scientific_contract["required_stability_records"], 10)
        self.assertEqual(exp465_state["official_evaluate_count"], 0)
        self.assertEqual(exp465_state["official_inspect_count"], 0)
        self.assertEqual(exp465_state["private_run_count"], 0)
        self.assertTrue(scientific_contract["seed_normal_confirmation_count_must_remain_zero"])
        self.assertTrue(scientific_contract["c0_gradual_and_period_non_degradation_required"])

        plan_text = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        checkbox_parser = CheckboxParserAntigravity()
        checkbox_parser.feed(plan_text)
        self.assertEqual(checkbox_parser.checkbox_count, 42)
        self.assertEqual(checkbox_parser.checked_count, 7)
        task1_marker = '<h2><label><input type="checkbox" checked disabled> Task 1 —'
        task2_marker = '<h2><label><input type="checkbox" disabled> Task 2 —'
        task1_section = plan_text[plan_text.index(task1_marker):plan_text.index(task2_marker)]
        self.assertEqual(task1_section.count('type="checkbox" checked disabled'), 7)
        self.assertEqual(task1_section.count('type="checkbox" disabled'), 0)

        builder = load_builder_antigravity()
        source_parent = subprocess.check_output([
            "git", "rev-parse", "HEAD^",
        ], cwd=ROOT_ANTIGRAVITY, text=True).strip()
        builder.build_triplet_antigravity(source_parent, validate_only=True)
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        self.assertEqual(graph["schema"], "exp465_task1_contract_codemap_v1")
        self.assertEqual(lock["schema"], "exp465_task1_contract_codemap_lock_v1")
        self.assertEqual(graph["source_parent_commit"], source_parent)
        self.assertEqual(lock["source_parent_commit"], source_parent)
        self.assertEqual(graph["unknown_relationships"], [])
        node_ids = {node["id"] for node in graph["nodes"]}
        expected_nodes = {
            "approved_design",
            "implementation_plan",
            "task1_report",
            "dashboard",
            "preregistration",
            "runbook",
            "implementation_report",
            "research_state",
            "experiment_catalog",
            "champion_ledger",
            "task1_config",
            "task1_contracts",
            "task1_config_test",
            "task1_contracts_test",
            "task1_governance_test",
            "map_builder",
        }
        self.assertEqual(node_ids, expected_nodes)
        self.assertEqual(lock["fingerprint_count"], len(lock["tracked_fingerprints"]))
        self.assertEqual(lock["represented_paths"], sorted(TRIPLET_ANTIGRAVITY))


if __name__ == "__main__":
    unittest.main()
