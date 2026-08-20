from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp461_ecg_period_signature_qualification_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    publication_inputs_antigravity,
)


def canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def rehash_antigravity(root: Path) -> None:
    artifact_map = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in LEAVES_ANTIGRAVITY[:-1]
    }
    (root / "artifact_sha256_antigravity.json").write_bytes(canonical_antigravity(artifact_map))


def rewrite_dependent_leaves_antigravity(root: Path, rows: list[dict[str, object]]) -> None:
    from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
        _jsonl_bytes_antigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.dashboard_export_antigravity import (
        export_dashboard_antigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity import (
        evaluate_qualification_antigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.qualification_antigravity import (
        MethodFoldMetricAntigravity,
        MethodQualificationAntigravity,
        summarize_methods_antigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
        capture_exp460_period_authority_antigravity,
    )

    metric_rows = tuple(
        MethodFoldMetricAntigravity(
            method=row["method"],
            seed=row["seed"],
            fold_index=row["fold_index"],
            normal_trials=row["normal_trials"],
            normal_alerts=row["normal_alerts"],
            normal_wilson_upper=row["normal_wilson_upper"],
            per_type_auc=row["per_type_auc"],
            pooled_auc=row["pooled_auc"],
            scores_finite=row["scores_finite"],
            passed=row["passed"],
            failure_reasons=tuple(row["failure_reasons"]),
            **(
                {
                    "normal_scores": tuple(row["normal_scores"]),
                    "primary_scores": {
                        name: tuple(values)
                        for name, values in row["primary_scores"].items()
                    },
                }
                if "normal_scores" in row
                else {}
            ),
        )
        for row in rows
    )
    qualification = MethodQualificationAntigravity(
        dataset="ECG5000_normal_2",
        method_order=(
            "hydra_local_gap",
            "period_signature_local_gap",
            "hydra_period_rank_max",
        ),
        primary_candidate="hydra_period_rank_max",
        fold_metrics=metric_rows,
        summaries=summarize_methods_antigravity(metric_rows),
        label_query_count=0,
    )
    authority = capture_exp460_period_authority_antigravity()
    decision = evaluate_qualification_antigravity(authority, qualification)
    summaries = summarize_methods_antigravity(metric_rows)

    (root / "method_fold_metrics_antigravity.jsonl").write_bytes(
        _jsonl_bytes_antigravity(tuple(rows))
    )
    (root / "failed_folds_antigravity.jsonl").write_bytes(
        _jsonl_bytes_antigravity(tuple(row for row in rows if not row["passed"]))
    )
    (root / "method_summary_antigravity.json").write_bytes(
        canonical_antigravity(
            {
                "methods": [
                    {
                        "method": summary.method,
                        "fold_count": summary.fold_count,
                        "passed_count": summary.passed_count,
                        "failed_count": summary.failed_count,
                        "min_period_change_auc": summary.min_period_change_auc,
                        "min_pooled_auc": summary.min_pooled_auc,
                        "max_normal_wilson_upper": summary.max_normal_wilson_upper,
                        "all_scores_finite": summary.all_scores_finite,
                        "passed": summary.passed,
                    }
                    for summary in summaries
                ]
            }
        )
    )
    (root / "decision_antigravity.json").write_bytes(
        canonical_antigravity(
            {
                "status": decision.status,
                "primary_candidate": decision.primary_candidate,
                "eligible_for_routing_research": decision.eligible_for_routing_research,
                "primary_passed_count": decision.primary_passed_count,
                "primary_required_count": decision.primary_required_count,
                "failure_reasons": list(decision.failure_reasons),
                "routing_applied": decision.routing_applied,
                "test_labels_opened": decision.test_labels_opened,
                "label_query_count": decision.label_query_count,
            }
        )
    )
    (root / "dashboard_snapshot_antigravity.json").write_bytes(
        canonical_antigravity(export_dashboard_antigravity(authority, qualification, decision))
    )
    rehash_antigravity(root)


class ReplayAntigravityTest(unittest.TestCase):
    def test_inspect_recomputes_qualification_decision_dashboard_and_hashes_antigravity(self) -> None:
        """Break caught: replay trusts stored summaries, decisions, dashboards, or hash maps."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp461_task4_antigravity"
            publication = publish_result_antigravity(root, authority, qualification, decision)
            replay = inspect_replay_at_antigravity(root)

            self.assertEqual(replay.qualification, qualification)
            self.assertEqual(replay.decision, decision)
            self.assertEqual(replay.artifact_map_sha256, publication.artifact_map_sha256)

    def test_replay_rejects_inventory_canonical_and_rehashed_semantic_tamper_antigravity(self) -> None:
        """Break caught: replay accepts missing leaves, noncanonical JSON, or rehashed result tampering."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        def mutate_json(root: Path, name: str, mutate) -> None:
            path = root / name
            value = json.loads(path.read_text(encoding="utf-8"))
            mutate(value)
            path.write_bytes(canonical_antigravity(value))

        def mutate_jsonl(root: Path, name: str, mutate) -> None:
            path = root / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            mutate(rows)
            path.write_bytes(b"".join(canonical_antigravity(row) for row in rows))

        mutations = (
            lambda root: (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8"),
            lambda root: (root / "method_summary_antigravity.json").write_text('{"summary": true}\n', encoding="utf-8"),
            lambda root: mutate_jsonl(
                root,
                "method_fold_metrics_antigravity.jsonl",
                lambda rows: rows[0].__setitem__("method", "hydra_period_rank_max"),
            ),
            lambda root: mutate_jsonl(
                root,
                "method_fold_metrics_antigravity.jsonl",
                lambda rows: rows[-1]["per_type_auc"].__setitem__("period_change", 0.5),
            ),
            lambda root: mutate_json(
                root,
                "decision_antigravity.json",
                lambda value: value.__setitem__("eligible_for_routing_research", False),
            ),
            lambda root: mutate_json(
                root,
                "dashboard_snapshot_antigravity.json",
                lambda value: value.__setitem__("primary_status", "tampered"),
            ),
        )
        authority, qualification, decision = publication_inputs_antigravity()
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "exp461_task4_antigravity"
                publish_result_antigravity(root, authority, qualification, decision)
                mutate(root)
                if (root / "artifact_sha256_antigravity.json").exists() and not (root / "extra_antigravity.json").exists():
                    rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root)

    def test_replay_rejects_coherent_rehashed_primary_auc_tamper_antigravity(self) -> None:
        """Break caught: replay accepts edited metric AUCs with regenerated summary/decision/dashboard/hash leaves."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp461_task4_antigravity"
            publish_result_antigravity(root, authority, qualification, decision)
            rows = [
                json.loads(line)
                for line in (root / "method_fold_metrics_antigravity.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for row in rows:
                if (
                    row["method"] == "hydra_period_rank_max"
                    and row["seed"] == 20260721
                    and row["fold_index"] == 3
                ):
                    row["per_type_auc"]["period_change"] = 0.99
                    row["pooled_auc"] = sum(row["per_type_auc"].values()) / len(row["per_type_auc"])
                    row["passed"] = True
                    row["failure_reasons"] = []
                    break
            rewrite_dependent_leaves_antigravity(root, rows)

            with self.assertRaises(ValueError):
                inspect_replay_at_antigravity(root)

    def test_replay_rejects_coherent_rehashed_pass_reason_forgery_antigravity(self) -> None:
        """Break caught: replay accepts below-threshold AUC with forged passed=True and empty reasons."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp461_task4_antigravity"
            publish_result_antigravity(root, authority, qualification, decision)
            rows = [
                json.loads(line)
                for line in (root / "method_fold_metrics_antigravity.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for row in rows:
                if (
                    row["method"] == "hydra_period_rank_max"
                    and row["seed"] == 20261721
                    and row["fold_index"] == 4
                ):
                    row["per_type_auc"]["period_change"] = 0.80
                    row["pooled_auc"] = sum(row["per_type_auc"].values()) / len(row["per_type_auc"])
                    if "primary_scores" in row:
                        row["primary_scores"]["period_change"] = row["normal_scores"]
                    row["passed"] = True
                    row["failure_reasons"] = []
                    break
            rewrite_dependent_leaves_antigravity(root, rows)

            with self.assertRaises(ValueError):
                inspect_replay_at_antigravity(root)


if __name__ == "__main__":
    unittest.main()
