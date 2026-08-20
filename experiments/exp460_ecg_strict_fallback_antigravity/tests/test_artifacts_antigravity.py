from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def authority_antigravity():
    from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
        capture_exp459_fallback_authority_antigravity,
    )

    return capture_exp459_fallback_authority_antigravity()


def result_antigravity():
    from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
        compare_fallback_rules_antigravity,
    )

    return compare_fallback_rules_antigravity(authority_antigravity())


def canonical_json_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def expected_comparison_rows_antigravity() -> list[dict[str, object]]:
    return [
        {
            "f1": 0.5742574257425742,
            "fn": 837,
            "fp": 66,
            "route": "baseline",
            "rule": "always_baseline",
            "tn": 70788,
            "tp": 609,
        },
        {
            "f1": 0.5796148426491311,
            "fn": 829,
            "fp": 66,
            "route": "candidate",
            "rule": "always_candidate",
            "tn": 70788,
            "tp": 617,
        },
        {
            "f1": 0.5796148426491311,
            "fn": 829,
            "fp": 66,
            "route": "candidate",
            "rule": "majority_8_of_10",
            "tn": 70788,
            "tp": 617,
        },
        {
            "f1": 0.5742574257425742,
            "fn": 837,
            "fp": 66,
            "route": "baseline",
            "rule": "strict_10_of_10",
            "tn": 70788,
            "tp": 609,
        },
    ]


def expected_failed_folds_antigravity() -> list[dict[str, object]]:
    return [
        {
            "period_change_auc": 0.8208339620743423,
            "fold_index": 3,
            "seed": 20260721,
        },
        {
            "period_change_auc": 0.8004124541019063,
            "fold_index": 4,
            "seed": 20261721,
        },
    ]


def expected_decision_antigravity() -> dict[str, object]:
    return {
        "all_records_passed": False,
        "joint_success": False,
        "passed_record_count": 8,
        "strict_route": "baseline",
        "strict_status": "fallback_triggered_performance_not_improved",
        "worst_period_change_auc": 0.8004124541019063,
    }


def expected_context_antigravity() -> dict[str, object]:
    return {
        "dataset": "ECG5000_normal_2",
        "experiment_id": "Exp460",
        "exp459_decision_sha256": "527318cacdbc63905ab58237d793803eb286192456eb007362e6bed11c8f8716",
        "exp459_frozen_authority_digest": "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        "exp459_official_artifact_map_sha256": "a30fbe7ed34f88c1ab9b7fd40963edebbf1a4127d8d64c99141224cb2dbb106d",
        "label_query_count": 0,
        "official_decision_rule": "strict_10_of_10_only",
        "predecessor_experiment_id": "Exp459",
        "predecessor_kind": "ecg_normal2_blocked_validation",
        "strict_failure_route": "baseline",
    }


def expected_dashboard_antigravity() -> dict[str, object]:
    return {
        "comparison_rows": expected_comparison_rows_antigravity(),
        "dataset": "ECG5000_normal_2",
        "experiment_id": "Exp460",
        "failed_folds": expected_failed_folds_antigravity(),
        "implementation_official": False,
        "joint_success": False,
        "label_query_count": 0,
        "official_evaluate_invocations": 0,
        "official_inspect_invocations": 0,
        "predecessor_authority_digest": "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        "predecessor_experiment_id": "Exp459",
        "progress": "Task 4/6",
        "status": "fallback_triggered_performance_not_improved",
        "strict_route": "baseline",
        "strict_status": "fallback_triggered_performance_not_improved",
        "worst_period_change_auc": 0.8004124541019063,
    }


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publishes_exact_canonical_six_leaf_inventory_create_only_antigravity(self) -> None:
        """Break caught: publication drifts from the six-leaf contract or overwrites an existing root."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            _LEAVES_ANTIGRAVITY,
            publish_result_antigravity,
        )

        expected = (
            "context_antigravity.json",
            "rule_comparisons_antigravity.jsonl",
            "failed_folds_antigravity.jsonl",
            "decision_antigravity.json",
            "dashboard_snapshot_antigravity.json",
            "artifact_sha256_antigravity.json",
        )
        self.assertEqual(_LEAVES_ANTIGRAVITY, expected)
        authority = authority_antigravity()
        result = result_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact_root_antigravity"
            published = publish_result_antigravity(root, authority, result)
            self.assertEqual(tuple(name for name, _digest in published.leaf_sha256), expected)
            self.assertEqual({entry.name for entry in root.iterdir()}, set(expected))
            self.assertTrue(all((root / name).read_bytes().endswith(b"\n") for name in expected))

            comparison_rows = expected_comparison_rows_antigravity()
            failed_folds = expected_failed_folds_antigravity()
            decision = expected_decision_antigravity()
            context = expected_context_antigravity()
            dashboard = expected_dashboard_antigravity()
            expected_bytes = {
                "context_antigravity.json": canonical_json_antigravity(context),
                "rule_comparisons_antigravity.jsonl": b"".join(
                    canonical_json_antigravity(row) for row in comparison_rows
                ),
                "failed_folds_antigravity.jsonl": b"".join(
                    canonical_json_antigravity(row) for row in failed_folds
                ),
                "decision_antigravity.json": canonical_json_antigravity(decision),
                "dashboard_snapshot_antigravity.json": canonical_json_antigravity(dashboard),
            }
            expected_artifact_map = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in expected_bytes.items()
            }
            expected_bytes["artifact_sha256_antigravity.json"] = canonical_json_antigravity(
                expected_artifact_map
            )
            self.assertEqual(
                {
                    name: (root / name).read_bytes()
                    for name in expected
                },
                expected_bytes,
            )

            artifact_map = json.loads((root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(tuple(artifact_map), tuple(sorted(expected[:-1])))
            self.assertNotIn("artifact_sha256_antigravity.json", artifact_map)

            with self.assertRaises(ValueError):
                publish_result_antigravity(root, authority, result)

    def test_rejects_missing_or_wrong_inputs_and_unsafe_root_antigravity(self) -> None:
        """Break caught: publication trusts missing authority, inconsistent results, or unsafe destinations."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )

        authority = authority_antigravity()
        result = result_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target_antigravity"
            target.mkdir()
            link = Path(temporary) / "link_antigravity"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                publish_result_antigravity(link, authority, result)

        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            publish_result_antigravity(Path(temporary) / "artifact_root_antigravity", object(), result)

        forged = result_antigravity()
        object.__setattr__(forged, "joint_success", True)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            publish_result_antigravity(Path(temporary) / "artifact_root_antigravity", authority, forged)


if __name__ == "__main__":
    unittest.main()
