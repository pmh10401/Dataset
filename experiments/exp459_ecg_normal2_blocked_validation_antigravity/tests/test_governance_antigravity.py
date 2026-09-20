"""Governance regression for Exp459 Task 1 planned-not-started truth."""

from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
import copy
import re
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity.html"
)
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity-design.html"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP459_PREREGISTRATION_antigravity.html",
    "docs/EXP459_RUNBOOK_antigravity.html",
    "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP459_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp459_antigravity.html",
    "docs/codemap/codemap_exp459_antigravity.json",
    "docs/codemap/codemap_exp459_antigravity.lock",
)
EXP458_IMMUTABLE_TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp458_antigravity.html": (
        "7fa2cf91c4442aff6c1adad906b1387290594c008435426444431ba0988503cc"
    ),
    "docs/codemap/codemap_exp458_antigravity.json": (
        "9e2c8e4866e4b30c29b39e4a62e1c7f1af5500e7f16976ed76edf8df5b90eed7"
    ),
    "docs/codemap/codemap_exp458_antigravity.lock": (
        "a56d24fb607bd2cd20422c23384df3f33ebd9394a0dc014e29b5b19881f17edc"
    ),
}
EXP458_IMMUTABLE_MAP_SOURCE_PARENT_ANTIGRAVITY = (
    "e6d9e6e1f4febc5c8e8cd549c2ee20d5b7eccde6"
)
DESIGN_SHA256_ANTIGRAVITY = (
    "47d530a0cbe7f7cdb0b85f2c0c546b7ede158d8a85824eba8c8caece16b2e275"
)
PLAN_SHA256_ANTIGRAVITY = (
    "db1aaf8f959a55c8ee7758edb61b95e6f7bf8d9ee8ea821ca39733213692d994"
)
RUN_IDENTITY_ANTIGRAVITY = (
    "exp459-ecg-normal2-blocked-validation-antigravity-20260818"
)
IMPLEMENTATION_STATUS_ANTIGRAVITY = (
    "task_6_official_unstable_preserved"
)
STATUS_ANTIGRAVITY = "official_evaluation_inspected_complete_unstable_no_promotion"
AUTHORITY_DIGEST_ANTIGRAVITY = (
    "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee"
)
AUTHORITY_MODULE_ANTIGRAVITY = (
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/"
    "frozen_validation_authority_antigravity.py"
)
PRIVATE_VALIDATION_ANTIGRAVITY = {
    "run_count": 2,
    "status": "unstable",
    "passed": False,
    "canonical_digest": AUTHORITY_DIGEST_ANTIGRAVITY,
    "fold_plan_rows": 5,
    "fold_evidence_rows": 10,
    "seed_summary_rows": 2,
    "leaf_count": 7,
    "temporary_roots_cleaned": True,
    "official_root_absent": True,
    "failed_folds": [
        {
            "seed": 20260721,
            "fold_index": 3,
            "period_change_auc": 0.8208339620743423,
            "pooled_auc": 0.9160442130677531,
            "normal_wilson_upper": 0.013439124039399194,
        },
        {
            "seed": 20261721,
            "fold_index": 4,
            "period_change_auc": 0.8004124541019063,
            "pooled_auc": 0.9241518283788541,
            "normal_wilson_upper": 0.013439124039399194,
        },
    ],
}
OFFICIAL_ARTIFACT_MAP_SHA256_ANTIGRAVITY = (
    "a30fbe7ed34f88c1ab9b7fd40963edebbf1a4127d8d64c99141224cb2dbb106d"
)
OFFICIAL_DASHBOARD_SHA256_ANTIGRAVITY = (
    "c1dee85d9a78651df3366f1a5c24ec57d16cd0581e1b1f8fbe4098746c5ef350"
)
OFFICIAL_DECISION_SHA256_ANTIGRAVITY = (
    "527318cacdbc63905ab58237d793803eb286192456eb007362e6bed11c8f8716"
)
OFFICIAL_EVALUATION_SHA256_ANTIGRAVITY = (
    "a1b47f5595477b85bd078bb26dde3917c97996a48ca0d6d3818fae3f7f8c2364"
)
OFFICIAL_REVIEWED_COMMIT_ANTIGRAVITY = (
    "05bca7e3b03e14ddc9200e24174606bcb951827e"
)
OFFICIAL_RESULT_ANTIGRAVITY = {
    "status": "unstable",
    "passed": False,
    "fold_plan_rows": 5,
    "fold_records": 10,
    "seed_summaries": 2,
    "failed_fold_count": 2,
    "failed_folds": PRIVATE_VALIDATION_ANTIGRAVITY["failed_folds"],
}
OFFICIAL_LEAF_SHA256_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json": OFFICIAL_ARTIFACT_MAP_SHA256_ANTIGRAVITY,
    "context_antigravity.json": "757f1af091ce7efba11d9dfb4ab6fc9ad5b53a6527a0da102033ed5a71b556be",
    "dashboard_snapshot_antigravity.json": OFFICIAL_DASHBOARD_SHA256_ANTIGRAVITY,
    "decision_antigravity.json": OFFICIAL_DECISION_SHA256_ANTIGRAVITY,
    "fold_evidence_antigravity.jsonl": OFFICIAL_EVALUATION_SHA256_ANTIGRAVITY,
    "fold_plan_antigravity.jsonl": "7e2b60c31ff493077fbe6e501f25704d5b527b3cea8496e8b265e66b71191985",
    "seed_summary_antigravity.jsonl": "f530e4c411085f3406be896f1802357b5ce67a53b5e1a4f753ac02fdf2b061d7",
}
SCIENTIFIC_LIMITATION_DISCLOSURE_ANTIGRAVITY = (
    "각 fold에서는 정상 점수 282개 또는 283개를 자기 자신과 비교해 p값을 계산한다",
    "가능한 가장 작은 p값은",
    "정상 경보 수가 항상 0",
    "Wilson 값은 fold별 오경보 차이를 구분하지 못한다",
    "seed 20260721의 fold 3과 seed 20261721의 fold 4",
)
RUN_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp459_ecg_normal2_blocked_validation_antigravity_20260818"
)
PACKAGE_ANTIGRAVITY = "experiments/exp459_ecg_normal2_blocked_validation_antigravity"
PENDING_CANDIDATE_ANTIGRAVITY = {
    "evaluation": None,
    "experiment": None,
    "prediction_sha256": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
}
CHAMPION_SNAPSHOT_ANTIGRAVITY = {
    "champion_status": "no_new_champion_verified_in_current_loop",
    "pending_candidate": PENDING_CANDIDATE_ANTIGRAVITY,
    "verified_references": [
        {
            "dataset_balanced_macro": {
                "f1": 0.6089825086931719,
                "false_positive_rate": 0.002196457972318159,
                "precision": 0.8262807700918903,
                "recall": 0.591448835046688,
            },
            "dataset_count": 1117,
            "evaluation_artifact": (
                "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit/"
                "exp416_confusion_matrix.json"
            ),
            "micro": {
                "f1": 0.6241851368970013,
                "false_positive_rate": 0.0015782729630902897,
                "fn": 2003,
                "fp": 303,
                "precision": 0.8633904418394951,
                "recall": 0.4887697805002552,
                "tn": 191679,
                "tp": 1915,
            },
            "name": "Exp404 Hard prediction audited by Exp416",
            "operational_policy_changed": False,
            "prediction_coordinate_count": 2218,
            "prediction_dataset_count": 888,
            "prediction_sha256": (
                "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
            ),
            "role": "current_verified_full_population_reference_not_mandatory_code_base",
            "test_labels_loaded_after_prediction_freeze": True,
        }
    ],
}


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates_antigravity,
    )


def map_builder_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp459_antigravity.py"
    self = path.is_file()
    if not self:
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp459_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp459 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlTextExtractorAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def unresolved_evidence_antigravity(graph: dict[str, object]) -> list[dict[str, str]]:
    unresolved: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for collection in ("nodes", "edges"):
        for item in graph[collection]:
            for evidence in item["evidence"] if collection == "nodes" else [item["evidence"]]:
                key = (evidence["path"], evidence["symbol"])
                if key in seen:
                    continue
                seen.add(key)
                path = ROOT_ANTIGRAVITY / evidence["path"]
                if not path.is_file():
                    unresolved.append({"path": evidence["path"], "symbol": evidence["symbol"]})
                    continue
                text = path.read_text(encoding="utf-8")
                if path.suffix == ".html":
                    parser = HtmlTextExtractorAntigravity()
                    parser.feed(text)
                    haystack = " ".join(parser.parts)
                else:
                    haystack = text
                if evidence["symbol"] not in haystack:
                    unresolved.append({"path": evidence["path"], "symbol": evidence["symbol"]})
    return unresolved


class GovernanceAntigravityTest(unittest.TestCase):
    def test_task1_documents_triplet_and_checkboxes_antigravity(self) -> None:
        """Break caught: Task 2 claims progress without HTML/docs/map truth."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            with self.subTest(relative=relative):
                path = ROOT_ANTIGRAVITY / relative
                self.assertTrue(path.is_file(), relative)
                self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertEqual(path.suffix, ".html")
            self.assertFalse(path.with_suffix(".md").exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", text.lower())
            self.assertIn("Exp459", text)

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        for task_number in range(1, 7):
            section = re.search(
                rf"<h2>Task {task_number}:.*?(?=<h2>Task {task_number + 1}:|<h2>전체 검증 명령 모음)",
                plan,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(section)
            assert section is not None
            tags = re.findall(r"<input\b[^>]*>", section.group(0), flags=re.IGNORECASE)
            checked = [tag for tag in tags if re.search(r"\bchecked(?:\s|=|>)", tag)]
            self.assertEqual(len(checked), len(tags))
        status_rows = re.findall(r"<li><label><input\b[^>]*>(Task \d 완료)</label></li>", plan)
        self.assertEqual(status_rows, [f"Task {n} 완료" for n in range(1, 7)])
        initial_checked = re.findall(
            r"<li><label><input\b[^>]*checked[^>]*>(Task \d 완료)</label></li>",
            plan,
            flags=re.IGNORECASE,
        )
        self.assertEqual(
            initial_checked,
            [
                "Task 1 완료",
                "Task 2 완료",
                "Task 3 완료",
                "Task 4 완료",
                "Task 5 완료",
                "Task 6 완료",
            ],
        )

    def test_task5_authority_sequence_and_import_surface_are_unambiguous_antigravity(self) -> None:
        """Break caught: Task 5 can create or use authority before exact private equality/review."""
        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        task5 = re.search(
            r"<h2>Task 5:.*?(?=<h2>Task 6:)",
            plan,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(task5)
        assert task5 is not None
        sequence = [
            "private qualification/evaluation 두 번을 public CLI 없이 temp root로 실행",
            "canonical 5 fold plan / 10 fold evidence / 2 seed summaries / decision input / digest exact-equal",
            "experiments/exp459_ecg_normal2_blocked_validation_antigravity/frozen_validation_authority_antigravity.py",
            "CLI evaluate와 bare inspect는 모두 version-controlled authority module을 require",
            "bare inspect import boundary: authority + replay only; sources/SQLite/Hydra 금지",
            "authority commit을 full tests 후 fresh final review",
            "SHIP 후에만 Task 6 approval",
        ]
        previous = -1
        for phrase in sequence:
            with self.subTest(phrase=phrase):
                position = task5.group(0).find(phrase, previous + 1)
                self.assertGreater(position, previous)
                previous = position
        authority_path = (
            ROOT_ANTIGRAVITY
            / PACKAGE_ANTIGRAVITY
            / "frozen_validation_authority_antigravity.py"
        )
        self.assertTrue(authority_path.is_file())

        document_expectations = {
            "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html": (
                "Task 6 official lifecycle",
                OFFICIAL_REVIEWED_COMMIT_ANTIGRAVITY,
                AUTHORITY_DIGEST_ANTIGRAVITY,
                "unstable",
            ),
            "docs/EXP459_RUNBOOK_antigravity.html": (
                "Task 6 official command evidence",
                "evaluate/inspect payloads exact-equal",
                "Do not rerun, retry, tune thresholds",
                OFFICIAL_ARTIFACT_MAP_SHA256_ANTIGRAVITY,
            ),
            "docs/EXP459_DASHBOARD_antigravity.html": (
                "Task 6/6",
                "official root present",
                "운영 승격 근거로 사용하지 않는다",
                AUTHORITY_DIGEST_ANTIGRAVITY,
            ),
        }
        for relative, phrases in document_expectations.items():
            text = (ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8")
            for phrase in phrases:
                with self.subTest(relative=relative, phrase=phrase):
                    self.assertIn(phrase, text)

    def test_task5_scientific_limitation_disclosure_is_visible_antigravity(self) -> None:
        """Break caught: reports overstate normal-alert/Wilson evidence as discriminative."""
        for relative in (*DOCUMENTS_ANTIGRAVITY, PLAN_ANTIGRAVITY):
            text = (ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8")
            for phrase in SCIENTIFIC_LIMITATION_DISCLOSURE_ANTIGRAVITY:
                with self.subTest(relative=relative, phrase=phrase):
                    self.assertIn(phrase, text)

    def test_exp459_task6_truth_and_non_promotion_antigravity(self) -> None:
        """Break caught: Exp459 official truth changes champion or loses frozen hashes."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict)
        assert isinstance(catalog, dict)
        assert isinstance(champion, dict)

        exp459_state = state["exp459"]
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp459")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp459")
        self.assertEqual(sum(item["id"] == "Exp459" for item in catalog["experiments"]), 1)
        self.assertEqual(sum(item["experiment"] == "Exp459" for item in champion["history"]), 1)

        for record in (exp459_state, catalog_entry, champion_entry):
            with self.subTest(record=record["experiment"]):
                self.assertEqual(record["experiment"], "Exp459")
                self.assertEqual(record["kind"], "ecg_normal2_blocked_validation")
                self.assertEqual(record["status"], STATUS_ANTIGRAVITY)
                self.assertEqual(record["implementation_status"], IMPLEMENTATION_STATUS_ANTIGRAVITY)
                self.assertEqual(record["frozen_validation_authority"], AUTHORITY_MODULE_ANTIGRAVITY)
                self.assertEqual(record["frozen_validation_authority_digest"], AUTHORITY_DIGEST_ANTIGRAVITY)
                self.assertEqual(record["private_validation"], PRIVATE_VALIDATION_ANTIGRAVITY)
                self.assertEqual(record["run_identity"], RUN_IDENTITY_ANTIGRAVITY)
                self.assertEqual(record["run_root"], str(RUN_ROOT_ANTIGRAVITY))
                self.assertEqual(record["package"], PACKAGE_ANTIGRAVITY)
                self.assertEqual(record["approved_design"], DESIGN_ANTIGRAVITY)
                self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
                self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
                self.assertEqual(record["approved_plan_sha256"], PLAN_SHA256_ANTIGRAVITY)
                self.assertTrue(record["official_evaluate_started"])
                self.assertTrue(record["official_inspect_started"])
                self.assertEqual(record["official_evaluate_invocations"], 1)
                self.assertEqual(record["official_inspect_invocations"], 1)
                self.assertTrue(record["result_root_created"])
                self.assertEqual(record["official_result"], OFFICIAL_RESULT_ANTIGRAVITY)
                self.assertIsNone(record["prediction_sha256"])
                self.assertEqual(
                    record["evaluation_sha256"],
                    OFFICIAL_EVALUATION_SHA256_ANTIGRAVITY,
                )
                self.assertEqual(
                    record["decision_sha256"],
                    OFFICIAL_DECISION_SHA256_ANTIGRAVITY,
                )
                self.assertEqual(
                    record["artifact_map_sha256"],
                    OFFICIAL_ARTIFACT_MAP_SHA256_ANTIGRAVITY,
                )
                self.assertEqual(
                    record["dashboard_snapshot_sha256"],
                    OFFICIAL_DASHBOARD_SHA256_ANTIGRAVITY,
                )
                self.assertEqual(record["leaf_sha256"], OFFICIAL_LEAF_SHA256_ANTIGRAVITY)
                lifecycle = record["official_lifecycle"]
                self.assertEqual(lifecycle["reviewed_commit"], OFFICIAL_REVIEWED_COMMIT_ANTIGRAVITY)
                self.assertEqual(lifecycle["evaluate_invocations"], 1)
                self.assertEqual(lifecycle["evaluate_exit_code"], 0)
                self.assertIs(type(lifecycle["evaluate_pid"]), int)
                self.assertGreater(lifecycle["evaluate_pid"], 0)
                self.assertEqual(lifecycle["inspect_invocations"], 1)
                self.assertEqual(lifecycle["inspect_exit_code"], 0)
                self.assertIs(type(lifecycle["inspect_pid"]), int)
                self.assertGreater(lifecycle["inspect_pid"], 0)
                self.assertTrue(lifecycle["aggregate_json_equal"])
                self.assertTrue(lifecycle["semantic_replay_equal"])
                self.assertTrue(lifecycle["authority_payload_equal"])
                self.assertFalse(lifecycle["passed"])
                self.assertEqual(lifecycle["status"], "unstable")
                self.assertEqual(lifecycle["label_query_count"], 0)
                self.assertTrue(lifecycle["result_root_created"])
                self.assertEqual(lifecycle["result_root_leaf_count"], 7)
                self.assertTrue(lifecycle["result_root_is_non_symlink_directory"])
                self.assertEqual(
                    lifecycle["artifact_map_non_map_leaf_recomputation"],
                    "exact_match",
                )
                self.assertEqual(lifecycle["authority_digest"], AUTHORITY_DIGEST_ANTIGRAVITY)
                self.assertEqual(
                    lifecycle["evaluate_stdout_sha256"],
                    OFFICIAL_DASHBOARD_SHA256_ANTIGRAVITY,
                )
                self.assertEqual(
                    lifecycle["inspect_stdout_sha256"],
                    OFFICIAL_DASHBOARD_SHA256_ANTIGRAVITY,
                )
                self.assertEqual(
                    lifecycle["failed_folds"],
                    PRIVATE_VALIDATION_ANTIGRAVITY["failed_folds"],
                )
                self.assertEqual(record["label_query_count"], 0)
                self.assertFalse(record["test_labels_opened"])
                self.assertFalse(record["routing_allowed"])
                self.assertFalse(record["promotion_allowed"])
                self.assertFalse(record["pending_candidate_changed"])
                self.assertFalse(record["scientific_winner_selected"])
                self.assertEqual(
                    record["predecessor"],
                    {
                        "experiment": "Exp458",
                        "reviewed_commit": "fc530cd2b392394aa7ffbd1408bc06d5b1b7c806",
                        "artifact_map_sha256": (
                            "7ef36f232db5d2f2396f3dcdeda15af2e93c34e18a6e05dc411e37fe855737f9"
                        ),
                    },
                )
                self.assertEqual(
                    record["scientific_contract"],
                    {
                        "target_dataset": "ECG5000_normal_2",
                        "hydra_seeds": [20260721, 20261721],
                        "fold_sizes": [283, 283, 283, 282, 282],
                        "fold_count": 5,
                        "primary_types": [
                            "level_shift",
                            "gradual_drift",
                            "amplitude_scaling",
                            "period_change",
                        ],
                        "per_type_auc_min": 0.85,
                        "pooled_auc_min": 0.90,
                        "normal_wilson_upper_max": 0.05,
                        "shuffle_allowed": False,
                        "final_decision_rule": "all_10_fold_seed_records_must_pass_for_stable",
                    },
                )

        self.assertEqual(state["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {
                "champion_status": champion["champion_status"],
                "pending_candidate": champion["pending_candidate"],
                "verified_references": champion["verified_references"],
            },
            CHAMPION_SNAPSHOT_ANTIGRAVITY,
        )
        self.assertEqual(
            hashlib.sha256((ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes()).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )
        self.assertTrue(RUN_ROOT_ANTIGRAVITY.is_dir())
        self.assertFalse(RUN_ROOT_ANTIGRAVITY.is_symlink())
        self.assertEqual(
            {entry.name for entry in RUN_ROOT_ANTIGRAVITY.iterdir()},
            set(OFFICIAL_LEAF_SHA256_ANTIGRAVITY),
        )
        self.assertEqual(
            {
                leaf_name: hashlib.sha256((RUN_ROOT_ANTIGRAVITY / leaf_name).read_bytes()).hexdigest()
                for leaf_name in OFFICIAL_LEAF_SHA256_ANTIGRAVITY
            },
            OFFICIAL_LEAF_SHA256_ANTIGRAVITY,
        )

    def test_exp459_map_is_bounded_and_validates_antigravity(self) -> None:
        """Break caught: Task 1 map drifts from HTML/JSON/lock parity or loses governance coverage."""
        for relative in TRIPLET_ANTIGRAVITY:
            self.assertTrue((ROOT_ANTIGRAVITY / relative).is_file(), relative)
        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        self.assertLessEqual(len(graph["nodes"]), 38)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertIn("governance_regression", node_by_id)
        self.assertIn("task2_config", node_by_id)
        self.assertIn("task2_contracts", node_by_id)
        self.assertIn("task2_blocked_folds", node_by_id)
        self.assertIn("task3_sources", node_by_id)
        self.assertIn("task3_qualification", node_by_id)
        self.assertIn("task4_evaluation", node_by_id)
        self.assertIn("task4_artifacts", node_by_id)
        self.assertIn("task4_replay", node_by_id)
        self.assertIn("task4_dashboard_export", node_by_id)
        self.assertIn("task5_cli", node_by_id)
        self.assertIn("task5_frozen_authority", node_by_id)
        self.assertIn("task5_launcher", node_by_id)
        self.assertIn("task5_cli_tests", node_by_id)
        self.assertIn("research_state", node_by_id)
        self.assertIn("experiment_catalog", node_by_id)
        self.assertIn("champion_ledger", node_by_id)
        self.assertTrue(
            any(
                edge["from"] == "governance_regression"
                and edge["to"] == "research_state"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "governance_regression"
                and edge["to"] == "champion_ledger"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task2_contract_tests"
                and edge["to"] == "task2_contracts"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task3_source_tests"
                and edge["to"] == "task3_sources"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task3_qualification_tests"
                and edge["to"] == "task3_qualification"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_evaluation_tests"
                and edge["to"] == "task4_evaluation"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_artifact_tests"
                and edge["to"] == "task4_artifacts"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_replay_tests"
                and edge["to"] == "task4_replay"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_dashboard_tests"
                and edge["to"] == "task4_dashboard_export"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task5_cli_tests"
                and edge["to"] == "task5_cli"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task5_cli_tests"
                and edge["to"] == "task5_frozen_authority"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task5_cli_tests"
                and edge["to"] == "task5_launcher"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertEqual(graph["unknown_relationships"], [])
        builder.validate_map_antigravity(graph, lock, rendered)

    def test_exp459_map_child_preserves_exp458_triplet_antigravity(self) -> None:
        """Break caught: Exp459 map-only child rewrites the prior Exp458 map triplet."""
        actual = {
            relative: hashlib.sha256((ROOT_ANTIGRAVITY / relative).read_bytes()).hexdigest()
            for relative in EXP458_IMMUTABLE_TRIPLET_ANTIGRAVITY
        }
        self.assertEqual(actual, EXP458_IMMUTABLE_TRIPLET_ANTIGRAVITY)
        graph = strict_json_antigravity(
            ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp458_antigravity.json"
        )
        lock = strict_json_antigravity(
            ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp458_antigravity.lock"
        )
        self.assertEqual(
            graph["source_parent_commit"],
            EXP458_IMMUTABLE_MAP_SOURCE_PARENT_ANTIGRAVITY,
        )
        self.assertEqual(
            lock["source_parent_commit"],
            EXP458_IMMUTABLE_MAP_SOURCE_PARENT_ANTIGRAVITY,
        )
        self.assertNotEqual(
            EXP458_IMMUTABLE_MAP_SOURCE_PARENT_ANTIGRAVITY,
            "57199eb1b66b56e30b8cb4ed091bf21fe2b0c27e",
        )

    def test_exp459_map_evidence_symbols_resolve_in_cited_sources_antigravity(self) -> None:
        """Break caught: map evidence cites HTML or Python symbols that do not resolve in the cited file."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        self.assertEqual(unresolved_evidence_antigravity(graph), [])

    def test_exp459_validator_rejects_unresolved_evidence_symbol_antigravity(self) -> None:
        """Break caught: validator accepts an evidence symbol that does not resolve in its cited source."""
        builder = map_builder_antigravity()
        source_commit = builder.subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_ANTIGRAVITY,
            text=True,
        ).strip()
        graph, lock, rendered = builder.build_map_antigravity(
            source_commit,
            "2026-08-18T00:00:00Z",
        )
        graph = copy.deepcopy(graph)
        graph["edges"][0]["evidence"]["symbol"] = "missing_symbol_antigravity"
        rendered = builder.html_antigravity(graph)
        with self.assertRaisesRegex(ValueError, "unresolved evidence"):
            builder.validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    unittest.main()
