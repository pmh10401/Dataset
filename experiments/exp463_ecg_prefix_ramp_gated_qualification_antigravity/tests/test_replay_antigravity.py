from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    publication_inputs_antigravity,
)


def canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def rehash_antigravity(root: Path) -> None:
    artifact_map = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in LEAVES_ANTIGRAVITY[:-1]
        if (root / name).is_file()
    }
    (root / LEAVES_ANTIGRAVITY[-1]).write_bytes(canonical_antigravity(artifact_map))


class ReplayAntigravityTest(unittest.TestCase):
    def test_replay_recomputes_metrics_summaries_decision_and_dashboard_source_free_antigravity(self) -> None:
        """Break caught: replay trusts stored aggregate leaves or score booleans."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp463_task4_antigravity"
            publication = publish_result_antigravity(root, authority, qualification, decision)
            replay = inspect_replay_at_antigravity(root)
            self.assertEqual(replay.qualification, qualification)
            self.assertEqual(replay.decision, decision)
            self.assertEqual(replay.artifact_map_sha256, publication.artifact_map_sha256)

    def test_replay_rejects_rehashed_tamper_matrix_antigravity(self) -> None:
        """Break caught: a changed payload is accepted merely because the map was rehashed."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()

        def rows_antigravity(root: Path) -> list[dict[str, object]]:
            return [
                json.loads(line)
                for line in (root / "method_fold_metrics_antigravity.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        def write_rows_antigravity(root: Path, rows: list[dict[str, object]]) -> None:
            (root / "method_fold_metrics_antigravity.jsonl").write_bytes(
                b"".join(canonical_antigravity(row) for row in rows)
            )

        def tamper_score_evidence_antigravity(root: Path) -> None:
            rows = rows_antigravity(root)
            rows[0]["normal_scores"][0] += 0.125  # type: ignore[index,operator]
            write_rows_antigravity(root, rows)
            rehash_antigravity(root)

        def tamper_numeric_metric_antigravity(root: Path) -> None:
            rows = rows_antigravity(root)
            row = rows[0]
            per_type = dict(row["per_type_auc"])  # type: ignore[arg-type]
            per_type["level_shift"] -= 0.001
            row["per_type_auc"] = per_type
            row["pooled_auc"] = sum(per_type.values()) / len(per_type)
            write_rows_antigravity(root, rows)
            rehash_antigravity(root)

        def tamper_summary_antigravity(root: Path) -> None:
            path = root / "method_summary_antigravity.jsonl"
            summaries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            summaries[0]["min_pooled_auc"] -= 0.001
            path.write_bytes(b"".join(canonical_antigravity(summary) for summary in summaries))
            rehash_antigravity(root)

        def tamper_decision_antigravity(root: Path) -> None:
            path = root / "decision_antigravity.json"
            decision_value = json.loads(path.read_text(encoding="utf-8"))
            decision_value["status"] += "-tampered"
            path.write_bytes(canonical_antigravity(decision_value))
            rehash_antigravity(root)

        def tamper_dashboard_antigravity(root: Path) -> None:
            path = root / "dashboard_snapshot_antigravity.json"
            dashboard = json.loads(path.read_text(encoding="utf-8"))
            dashboard["failed_fold_count"] += 1
            path.write_bytes(canonical_antigravity(dashboard))
            rehash_antigravity(root)

        def tamper_missing_leaf_antigravity(root: Path) -> None:
            (root / "dashboard_snapshot_antigravity.json").unlink()
            rehash_antigravity(root)

        def tamper_extra_leaf_antigravity(root: Path) -> None:
            (root / "extra_antigravity.json").write_bytes(canonical_antigravity({}))
            rehash_antigravity(root)

        def tamper_reordered_rows_antigravity(root: Path) -> None:
            rows = rows_antigravity(root)
            rows[0], rows[1] = rows[1], rows[0]
            write_rows_antigravity(root, rows)
            rehash_antigravity(root)

        def tamper_bool_antigravity(root: Path) -> None:
            rows = rows_antigravity(root)
            rows[0]["passed"] = not rows[0]["passed"]
            write_rows_antigravity(root, rows)
            rehash_antigravity(root)

        def tamper_string_bool_antigravity(root: Path) -> None:
            rows = rows_antigravity(root)
            rows[0]["scores_finite"] = "true"
            write_rows_antigravity(root, rows)
            rehash_antigravity(root)

        def tamper_failed_fold_projection_antigravity(root: Path) -> None:
            rows = rows_antigravity(root)
            (root / "failed_folds_antigravity.jsonl").write_bytes(
                canonical_antigravity(rows[0])
            )
            rehash_antigravity(root)

        mutations = (
            ("score_evidence", tamper_score_evidence_antigravity),
            ("numeric_metric", tamper_numeric_metric_antigravity),
            ("summary", tamper_summary_antigravity),
            ("decision", tamper_decision_antigravity),
            ("dashboard", tamper_dashboard_antigravity),
            ("missing_leaf", tamper_missing_leaf_antigravity),
            ("extra_leaf", tamper_extra_leaf_antigravity),
            ("reordered_rows", tamper_reordered_rows_antigravity),
            ("bool", tamper_bool_antigravity),
            ("string_bool", tamper_string_bool_antigravity),
            ("failed_fold_projection", tamper_failed_fold_projection_antigravity),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, mutate in mutations:
                with self.subTest(mutation=name):
                    root = Path(temporary) / f"exp463_task4_{name}_antigravity"
                    publish_result_antigravity(root, authority, qualification, decision)
                    mutate(root)
                    with self.assertRaises(ValueError):
                        inspect_replay_at_antigravity(root)


if __name__ == "__main__":
    unittest.main()
