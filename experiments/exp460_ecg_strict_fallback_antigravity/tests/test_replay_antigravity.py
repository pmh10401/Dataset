from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp460_ecg_strict_fallback_antigravity.tests.test_artifacts_antigravity import (
    authority_antigravity,
    result_antigravity,
)


def canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def rehash_antigravity(root: Path) -> None:
    names = (
        "context_antigravity.json",
        "rule_comparisons_antigravity.jsonl",
        "failed_folds_antigravity.jsonl",
        "decision_antigravity.json",
        "dashboard_snapshot_antigravity.json",
    )
    artifact_map = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    (root / "artifact_sha256_antigravity.json").write_bytes(canonical_antigravity(artifact_map))


class ReplayAntigravityTest(unittest.TestCase):
    def test_source_free_replay_recomputes_rule_rows_failed_folds_decision_and_dashboard_antigravity(self) -> None:
        """Break caught: replay trusts stored leaves instead of rebuilding the result from authority semantics."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority = authority_antigravity()
        result = result_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact_root_antigravity"
            publish_result_antigravity(root, authority, result)
            replayed = inspect_replay_at_antigravity(root, authority)
            self.assertEqual(
                [(row.rule, row.route, row.tp, row.fp, row.fn, row.tn, row.f1) for row in replayed.rule_results],
                [(row.rule, row.route, row.tp, row.fp, row.fn, row.tn, row.f1) for row in result.rule_results],
            )
            self.assertEqual(replayed.strict_route, "baseline")
            self.assertEqual(
                replayed.strict_status,
                "fallback_triggered_performance_not_improved",
            )
            self.assertFalse(replayed.joint_success)
            self.assertEqual(replayed.passed_record_count, 8)
            self.assertEqual(replayed.worst_period_change_auc, 0.8004124541019063)

    def test_rejects_inventory_hash_canonical_duplicate_bool_and_reordered_rows_antigravity(self) -> None:
        """Break caught: byte validation misses extra/missing leaves, duplicate keys, bool-int drift, or row reorder."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        mutations = (
            lambda root: (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8"),
            lambda root: (root / "failed_folds_antigravity.jsonl").unlink(),
            lambda root: (root / "decision_antigravity.json").write_text(
                '{"joint_success":false,"joint_success":true,"strict_route":"baseline","strict_status":"fallback_triggered_performance_not_improved","all_records_passed":false,"passed_record_count":8,"worst_period_change_auc":0.8004124541019063}\n',
                encoding="utf-8",
            ),
            lambda root: (root / "decision_antigravity.json").write_text(
                '{"all_records_passed":0,"joint_success":false,"passed_record_count":8,"strict_route":"baseline","strict_status":"fallback_triggered_performance_not_improved","worst_period_change_auc":0.8004124541019063}\n',
                encoding="utf-8",
            ),
            lambda root: self._reverse_jsonl_antigravity(root, "rule_comparisons_antigravity.jsonl"),
            lambda root: self._reverse_jsonl_antigravity(root, "failed_folds_antigravity.jsonl"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact_root_antigravity"
                publish_result_antigravity(root, authority_antigravity(), result_antigravity())
                mutate(root)
                if (root / "artifact_sha256_antigravity.json").exists() and not (root / "extra_antigravity.json").exists():
                    try:
                        rehash_antigravity(root)
                    except FileNotFoundError:
                        pass
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root, authority_antigravity())

    def test_rejects_rehashed_whitespace_and_key_order_mutations_antigravity(self) -> None:
        """Break caught: serializer and replay drift together so noncanonical whitespace or key order still pass."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        mutations = (
            lambda root: self._rewrite_json_with_whitespace_antigravity(root, "context_antigravity.json"),
            lambda root: self._rewrite_json_with_key_order_antigravity(
                root,
                "decision_antigravity.json",
                (
                    "worst_period_change_auc",
                    "strict_status",
                    "strict_route",
                    "passed_record_count",
                    "joint_success",
                    "all_records_passed",
                ),
            ),
            lambda root: self._rewrite_first_jsonl_row_with_key_order_antigravity(
                root,
                "rule_comparisons_antigravity.jsonl",
                ("tp", "tn", "rule", "route", "fp", "fn", "f1"),
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact_root_antigravity"
                publish_result_antigravity(root, authority_antigravity(), result_antigravity())
                mutate(root)
                rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root, authority_antigravity())

    def test_rejects_rehashed_semantic_tamper_across_result_leaves_antigravity(self) -> None:
        """Break caught: replay accepts rehashed route, confusion, F1, status, failed-fold, dashboard, or map tamper."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        mutations = (
            lambda root: self._mutate_jsonl_antigravity(root, "rule_comparisons_antigravity.jsonl", lambda rows: rows[0].__setitem__("route", "candidate")),
            lambda root: self._mutate_jsonl_antigravity(root, "rule_comparisons_antigravity.jsonl", lambda rows: rows[3].__setitem__("fn", 829)),
            lambda root: self._mutate_jsonl_antigravity(root, "rule_comparisons_antigravity.jsonl", lambda rows: rows[1].__setitem__("f1", 0.0)),
            lambda root: self._mutate_jsonl_antigravity(root, "failed_folds_antigravity.jsonl", lambda rows: rows[0].__setitem__("period_change_auc", 0.85)),
            lambda root: self._mutate_json_antigravity(root, "decision_antigravity.json", lambda value: value.__setitem__("strict_status", "candidate_retained_joint_success")),
            lambda root: self._mutate_json_antigravity(root, "decision_antigravity.json", lambda value: value.__setitem__("joint_success", True)),
            lambda root: self._mutate_json_antigravity(root, "dashboard_snapshot_antigravity.json", lambda value: value.__setitem__("label_query_count", 1)),
            lambda root: self._mutate_json_antigravity(root, "artifact_sha256_antigravity.json", lambda value: value.__setitem__("context_antigravity.json", "0" * 64)),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact_root_antigravity"
                publish_result_antigravity(root, authority_antigravity(), result_antigravity())
                mutate(root)
                if index != len(mutations) - 1:
                    rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root, authority_antigravity())

    def test_rejects_coordinated_dashboard_decision_and_map_tamper_with_original_authority_antigravity(self) -> None:
        """Break caught: replay accepts a self-consistent rehashed semantic rewrite."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact_root_antigravity"
            publish_result_antigravity(root, authority_antigravity(), result_antigravity())
            self._mutate_jsonl_antigravity(
                root,
                "rule_comparisons_antigravity.jsonl",
                lambda rows: (
                    rows[2].__setitem__("route", "baseline"),
                    rows[2].__setitem__("tp", 609),
                    rows[2].__setitem__("fn", 837),
                    rows[2].__setitem__("f1", 0.5742574257425742),
                ),
            )
            self._mutate_json_antigravity(
                root,
                "decision_antigravity.json",
                lambda value: value.__setitem__("passed_record_count", 7),
            )
            self._mutate_json_antigravity(
                root,
                "dashboard_snapshot_antigravity.json",
                lambda value: (
                    value.__setitem__("worst_period_change_auc", 0.85),
                    value.__setitem__("strict_route", "candidate"),
                    value.__setitem__("strict_status", "candidate_retained_joint_success"),
                ),
            )
            rehash_antigravity(root)
            with self.assertRaises(ValueError):
                inspect_replay_at_antigravity(root, authority_antigravity())

    @staticmethod
    def _reverse_jsonl_antigravity(root: Path, name: str) -> None:
        path = root / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_bytes(b"".join(canonical_antigravity(row) for row in reversed(rows)))

    @staticmethod
    def _mutate_json_antigravity(root: Path, name: str, mutate) -> None:
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_bytes(canonical_antigravity(value))

    @staticmethod
    def _mutate_jsonl_antigravity(root: Path, name: str, mutate) -> None:
        path = root / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        path.write_bytes(b"".join(canonical_antigravity(row) for row in rows))

    @staticmethod
    def _rewrite_json_with_whitespace_antigravity(root: Path, name: str) -> None:
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rewrite_json_with_key_order_antigravity(
        root: Path,
        name: str,
        key_order: tuple[str, ...],
    ) -> None:
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value[key], ensure_ascii=False)}"
                for key in key_order
            )
            + "}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rewrite_first_jsonl_row_with_key_order_antigravity(
        root: Path,
        name: str,
        key_order: tuple[str, ...],
    ) -> None:
        path = root / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        first = rows[0]
        first_line = (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(first[key], ensure_ascii=False)}"
                for key in key_order
            )
            + "}\n"
        ).encode("utf-8")
        remaining = b"".join(canonical_antigravity(row) for row in rows[1:])
        path.write_bytes(first_line + remaining)


if __name__ == "__main__":
    unittest.main()
