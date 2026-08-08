"""Behavior contracts for Exp441's label-free frozen-source replay."""
from __future__ import annotations

import ast
from concurrent.futures import Future
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    from experiments.exp441_hard_empty_single_shot_recovery_antigravity import sources_antigravity as sources
except ImportError:
    sources = None


def _candidate(dataset: str, index: int) -> dict[str, object]:
    return {
        "dataset": dataset, "instance_index": index, "raw_length": 5,
        "depths": [1, 2], "agreement_count": 2,
        "second_excess_ratio": 2.0, "second_empirical_rank": 0.5,
        "depth_scores": [],
    }


class _CompletionOrderExecutorAntigravity:
    """A real future-shaped test double that completes submitted work in reverse."""

    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers
        self.tasks: list[tuple[object, tuple[object, ...]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, function, *args):
        self.tasks.append((function, args))
        future = Future()
        future.set_result(function(*args))
        return future


class FrozenSourcesAntigravityTests(unittest.TestCase):
    """Each test names a verifier branch whose accidental removal is a bug."""

    def setUp(self) -> None:
        self.assertIsNotNone(sources, "the frozen source verifier must exist")

    def test_hard_coordinate_identity_is_compact_sorted_json_array(self) -> None:
        # Break caught: hash coordinate completion order instead of frozen identity order.
        self.assertEqual(
            sources.hard_coordinate_identity_sha256_antigravity({("B", 2), ("A", 7)}),
            sha256(b'[["A",7],["B",2]]').hexdigest(),
        )

    def test_prediction_allowlist_rejects_evaluation_fixture(self) -> None:
        # Break caught: widening the label-free source firewall to an evaluation artifact.
        with self.assertRaisesRegex(ValueError, "evaluation source is forbidden"):
            sources._validate_allowed_prediction_path_antigravity(
                Path("exp440/evaluation_antigravity/evaluation_antigravity.json")
            )

    def test_parse_task_rejects_digest_dataset_and_index_contract_breaks(self) -> None:
        # Break caught: trusting a record whose digest, identity, or TEST index is malformed.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record_antigravity.json"
            payload = {
                "dataset": "A", "status": "scored", "provisional_candidates": [_candidate("A", 3)],
                "abstention_indices": [], "excluded_train_rows": 0, "test_row_count": 3,
            }
            path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            digest = sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "outside TEST identity"):
                sources._parse_record_task_antigravity((str(path), "A", digest, 0))
            with self.assertRaisesRegex(ValueError, "record digest"):
                sources._parse_record_task_antigravity((str(path), "A", "0" * 64, 0))
            with self.assertRaisesRegex(ValueError, "record dataset"):
                sources._parse_record_task_antigravity((str(path), "B", digest, 0))

    def test_parse_task_rejects_unknown_payload_and_boolean_test_count(self) -> None:
        # Break caught: permissive parsing that accepts unpinned evidence fields or bool counts.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record_antigravity.json"
            payload = {
                "dataset": "A", "status": "scored", "provisional_candidates": [],
                "abstention_indices": [], "excluded_train_rows": 0, "test_row_count": True,
                "unexpected": False,
            }
            path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            task = (str(path), "A", sha256(path.read_bytes()).hexdigest(), 0)
            with self.assertRaisesRegex(ValueError, "record payload"):
                sources._parse_record_task_antigravity(task)

    def test_ordered_worker_parse_commits_context_order_and_fixed_eleven_workers(self) -> None:
        # Break caught: exposing worker completion order, changing the reviewed worker count, or leaking task data.
        with tempfile.TemporaryDirectory() as directory:
            tasks = []
            for sequence, dataset in enumerate(("A", "B")):
                path = Path(directory) / f"{sequence}_record_antigravity.json"
                payload = {
                    "dataset": dataset, "status": "scored", "provisional_candidates": [],
                    "abstention_indices": [], "excluded_train_rows": 0, "test_row_count": 1,
                }
                path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
                tasks.append((str(path), dataset, sha256(path.read_bytes()).hexdigest(), sequence))
            events: list[object] = []
            executor = _CompletionOrderExecutorAntigravity(max_workers=11)
            with patch.object(sources, "ProcessPoolExecutor", return_value=executor):
                records = sources._parse_records_fixed_workers_antigravity(tuple(tasks), events.append)
            self.assertEqual(tuple(record.dataset for record in records), ("A", "B"))
            self.assertEqual(executor.max_workers, 11)
            self.assertTrue(all(len(args[0]) == 4 for _, args in executor.tasks))
            self.assertEqual([event["dataset"] for event in events], ["A", "B"])

    def test_public_verifier_routes_only_to_fixed_worker_runtime(self) -> None:
        # Break caught: production reaching the isolated sequential reference implementation.
        with patch.object(sources, "_verify_frozen_sources_with_observer_antigravity", return_value="snapshot") as runtime, \
                patch.object(sources, "_verify_frozen_sources_single_worker_reference_antigravity") as reference:
            self.assertEqual(sources.verify_frozen_sources_antigravity(), "snapshot")
        runtime.assert_called_once()
        reference.assert_not_called()

    def test_source_module_has_no_sqlite_or_forbidden_exp440_evaluation_literals(self) -> None:
        # Break caught: a source-verifier dependency that can open labels or evaluation evidence.
        text = Path(sources.__file__).read_text()
        tree = ast.parse(text)
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("evaluation" + "_antigravity", text)

    def test_real_replay_is_fully_label_free_and_has_frozen_population_counts(self) -> None:
        # Break caught: a replay that silently changes source identity, counts, or touches labels.
        snapshot = sources.verify_frozen_sources_antigravity()
        self.assertEqual(len(snapshot.context_datasets), 1117)
        self.assertEqual(len(snapshot.hard_coordinates), 2218)
        self.assertEqual(len(snapshot.provisional_candidates), 840)
        self.assertEqual(snapshot.label_access_count, 0)
        self.assertEqual(snapshot.hard_coordinate_identity_sha256,
                         "858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322")


if __name__ == "__main__":
    unittest.main()
