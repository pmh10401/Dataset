"""Behavior contracts for Exp441's label-free frozen-source replay."""
from __future__ import annotations

import ast
import copy
from concurrent.futures import Future
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import tempfile
import threading
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


class _DelayedExecutorAntigravity:
    """Future-shaped worker pool whose completion order is deliberately unrelated to context."""

    def __init__(self, *, max_workers: int, delays: dict[str, float]) -> None:
        self.max_workers = max_workers
        self.delays = delays
        self.futures: list[Future] = []
        self.tasks: list[tuple[object, tuple[object, ...]]] = []
        self.threads: list[threading.Thread] = []
        self.completed_datasets: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        for thread in self.threads:
            thread.join()

    def submit(self, function, *args):
        future = Future()
        self.futures.append(future)
        self.tasks.append((function, args))

        def complete() -> None:
            time.sleep(self.delays[args[0][1]])
            try:
                result = function(*args)
            except BaseException as error:
                self.completed_datasets.append(args[0][1])
                future.set_exception(error)
            else:
                self.completed_datasets.append(args[0][1])
                future.set_result(result)

        thread = threading.Thread(target=complete)
        self.threads.append(thread)
        thread.start()
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

    def test_parse_task_rejects_boolean_test_count_without_unknown_key(self) -> None:
        # Break caught: accepting bool as a TEST row count after the schema itself is otherwise exact.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record_antigravity.json"
            payload = {
                "dataset": "A", "status": "scored", "provisional_candidates": [],
                "abstention_indices": [], "excluded_train_rows": 0, "test_row_count": True,
            }
            path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            with self.assertRaisesRegex(ValueError, "record payload"):
                sources._parse_record_task_antigravity((str(path), "A", sha256(path.read_bytes()).hexdigest(), 0))

    def test_checkpoint_root_inventory_requires_exact_regular_files_and_real_records_directory(self) -> None:
        # Break caught: accepting a checkpoint root with hidden artifacts or a non-directory records entry.
        required = ("context_antigravity.json", "state_antigravity.json", "retained_candidates_antigravity.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint_antigravity"
            root.mkdir()
            for name in required:
                (root / name).write_bytes(b"{}\n")
            (root / "records_antigravity").mkdir()
            self.assertEqual(sources._validate_checkpoint_root_inventory_antigravity(root), root / "records_antigravity")
            (root / "extra_antigravity.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(ValueError, "checkpoint inventory"):
                sources._validate_checkpoint_root_inventory_antigravity(root)
        for kind in ("directory", "symlink", "fifo"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoint_antigravity"
                root.mkdir()
                for name in required:
                    (root / name).write_bytes(b"{}\n")
                records = root / "records_antigravity"
                if kind == "directory":
                    records.write_bytes(b"not a directory")
                elif kind == "symlink":
                    target = root / "real_records_antigravity"
                    target.mkdir()
                    records.symlink_to(target, target_is_directory=True)
                else:
                    os.mkfifo(records)
                with self.assertRaisesRegex(ValueError, "checkpoint inventory"):
                    sources._validate_checkpoint_root_inventory_antigravity(root)

    def test_named_publication_inventory_rejects_extra_and_nonregular_members(self) -> None:
        # Break caught: accepting a ten-file publication directory with a substituted or special-file artifact.
        names = ("a_antigravity.json", "b_antigravity.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "publication_antigravity"
            root.mkdir()
            for name in names:
                (root / name).write_bytes(b"{}\n")
            sources._validate_named_regular_inventory_antigravity(root, names, "publication")
            (root / "extra_antigravity.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(ValueError, "publication inventory"):
                sources._validate_named_regular_inventory_antigravity(root, names, "publication")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "publication_antigravity"
            root.mkdir()
            (root / names[0]).write_bytes(b"{}\n")
            os.mkfifo(root / names[1])
            with self.assertRaisesRegex(ValueError, "publication inventory"):
                sources._validate_named_regular_inventory_antigravity(root, names, "publication")

    def test_all_frozen_file_hash_boundaries_reject_wrong_expected_digest(self) -> None:
        # Break caught: any checkpoint, publication, or Hard source being read without its frozen SHA gate.
        paths = (
            sources.FROZEN_CONFIG.checkpoint_context_path,
            sources.FROZEN_CONFIG.checkpoint_state_path,
            sources.FROZEN_CONFIG.checkpoint_retained_inventory_path,
            str(Path(sources.FROZEN_CONFIG.publication_path) / "prediction_antigravity.jsonl"),
            str(Path(sources.FROZEN_CONFIG.publication_path) / "artifact_sha256_antigravity.json"),
            sources.FROZEN_CONFIG.hard_prediction_path,
            sources.FROZEN_CONFIG.hard_manifest_path,
        )
        for path in paths:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                sources._read_pinned_allowed_antigravity(Path(path), "0" * 64)

    def test_hard_manifest_requires_exact_schema_and_coordinate_lane_agreement(self) -> None:
        # Break caught: accepting a stale, partial, or internally inconsistent Hard manifest.
        manifest_path = Path(sources.FROZEN_CONFIG.hard_manifest_path)
        prediction_path = Path(sources.FROZEN_CONFIG.hard_prediction_path)
        manifest = json.loads(manifest_path.read_text())
        coordinates = sources._hard_coordinates_antigravity(prediction_path.read_bytes(), manifest)
        self.assertEqual(
            sources._validate_hard_manifest_antigravity(manifest, coordinates),
            None,
        )
        missing = copy.deepcopy(manifest)
        missing.pop("rows")
        extra = copy.deepcopy(manifest)
        extra["unexpected"] = False
        wrong_type = copy.deepcopy(manifest)
        wrong_type["datasets"] = True
        wrong_lane = copy.deepcopy(manifest)
        lane_dataset = next(dataset for dataset, values in wrong_lane["frozen_hard_lanes"].items() if values)
        wrong_lane["frozen_hard_lanes"][lane_dataset] = []
        for invalid in (missing, extra, wrong_type, wrong_lane):
            with self.assertRaisesRegex(ValueError, "Hard manifest"):
                sources._validate_hard_manifest_antigravity(invalid, coordinates)

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
            self.assertEqual([(event["test_row_count"], event["raw_length"]) for event in events], [(1, None), (1, None)])
            self.assertTrue(all(os.environ[name] == "1" for name in sources._NUMERICAL_THREAD_ENV_ANTIGRAVITY))

    def test_parent_accepts_context_prefix_after_later_c_completes_and_fails_first(self) -> None:
        # Break caught: treating completion order as commit order, or hiding A/B after C has already failed.
        with tempfile.TemporaryDirectory() as directory:
            tasks = []
            for sequence, dataset in enumerate(("A", "B", "C")):
                path = Path(directory) / f"{sequence}_record_antigravity.json"
                payload = {
                    "dataset": dataset, "status": "scored", "provisional_candidates": [],
                    "abstention_indices": [], "excluded_train_rows": 0, "test_row_count": 1,
                }
                if dataset == "C":
                    payload["unexpected"] = False
                path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
                tasks.append((str(path), dataset, sha256(path.read_bytes()).hexdigest(), sequence))
            executor = _DelayedExecutorAntigravity(max_workers=11, delays={"A": 0.20, "B": 0.30, "C": 0.01})
            events: list[Mapping[str, object]] = []
            c_failed_before_a_accept: list[bool] = []

            def observe(event: Mapping[str, object]) -> None:
                events.append(event)
                if event["sequence"] == 0:
                    c_failed_before_a_accept.append(executor.futures[2].done())

            with patch.object(sources, "ProcessPoolExecutor", return_value=executor):
                with self.assertRaisesRegex(ValueError, "record payload"):
                    sources._parse_records_fixed_workers_antigravity(tuple(tasks), observe)
            self.assertEqual([event["dataset"] for event in events], ["A", "B"])
            self.assertEqual(executor.max_workers, 11)
            self.assertEqual(len(executor.tasks), 3)
            self.assertEqual(executor.completed_datasets[0], "C")
            self.assertEqual(c_failed_before_a_accept, [True])
            self.assertTrue(all(type(event).__name__ == "mappingproxy" for event in events))

    def test_reverification_rejects_a_source_mutated_after_its_initial_hash(self) -> None:
        # Break caught: returning a snapshot after a pinned source changes during the replay window.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_antigravity.json"
            path.write_bytes(b"first\n")
            expected = sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b"second\n")
            with patch.object(sources, "_validate_allowed_prediction_path_antigravity", return_value=path):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    sources._reverify_paths_antigravity({path: expected})

    def test_full_public_entrance_rejects_mutation_after_initial_pin_before_final_reverify(self) -> None:
        # Break caught: a full verifier returning a snapshot after an initially pinned source is later replaced.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pinned_source_antigravity.json"
            path.write_bytes(b"first\n")
            expected = sha256(path.read_bytes()).hexdigest()
            phase: list[str] = []
            sentinel = object()

            def load_prefix():
                payload, observed = sources._read_pinned_regular_antigravity(path)
                self.assertEqual((payload, observed), (b"first\n", expected))
                phase.append("initial-pin")
                return (), {}, frozenset(), {}, {path: expected}

            def parse_records(_tasks, _observer):
                self.assertEqual(phase, ["initial-pin"])
                path.write_bytes(b"replaced\n")
                phase.append("mutated-before-final-reverify")
                return ()

            with patch.object(sources, "_load_prefix_antigravity", side_effect=load_prefix), \
                    patch.object(sources, "_record_tasks_antigravity", return_value=()), \
                    patch.object(sources, "_parse_records_fixed_workers_antigravity", side_effect=parse_records), \
                    patch.object(sources, "_snapshot_from_records_antigravity", return_value=sentinel), \
                    patch.object(sources, "_validate_allowed_prediction_path_antigravity", return_value=path):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    sources.verify_frozen_sources_antigravity()
            self.assertEqual(phase, ["initial-pin", "mutated-before-final-reverify"])
            self.assertEqual(sources.FrozenSourceSnapshotAntigravity.__dataclass_fields__["label_access_count"].default, 0)

    def test_public_and_observer_routes_enter_fixed_worker_parser_never_reference(self) -> None:
        # Break caught: either production entrance bypassing the fixed 11-worker parser or using the test-only reference.
        sentinel = object()
        with patch.object(sources, "_verify_frozen_sources_with_observer_antigravity", return_value=sentinel) as runtime, \
                patch.object(sources, "_verify_frozen_sources_single_worker_reference_antigravity") as reference:
            self.assertIs(sources.verify_frozen_sources_antigravity(), sentinel)
        runtime.assert_called_once()
        reference.assert_not_called()
        prefix = ((), {}, frozenset(), {}, {})
        events: list[Mapping[str, object]] = []
        with patch.object(sources, "_load_prefix_antigravity", return_value=prefix), \
                patch.object(sources, "_record_tasks_antigravity", return_value=()), \
                patch.object(sources, "_parse_records_fixed_workers_antigravity", return_value=()) as parser, \
                patch.object(sources, "_snapshot_from_records_antigravity", return_value=sentinel), \
                patch.object(sources, "_reverify_paths_antigravity"), \
                patch.object(sources, "_verify_frozen_sources_single_worker_reference_antigravity") as reference:
            self.assertIs(sources._verify_frozen_sources_with_observer_antigravity(events.append), sentinel)
        parser.assert_called_once()
        reference.assert_not_called()

    def test_both_entrances_reach_same_fixed_eleven_executor_without_overrides(self) -> None:
        # Break caught: public and observer routes diverging in worker topology or exposing an override surface.
        class EmptyExecutor:
            def __init__(self, *, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def submit(self, *_args: object):
                raise AssertionError("empty controlled task set must not submit")

        prefix = ((), {}, frozenset(), {}, {})
        sentinel = object()
        executors: list[EmptyExecutor] = []

        def executor_factory(*, max_workers: int) -> EmptyExecutor:
            instance = EmptyExecutor(max_workers=max_workers)
            executors.append(instance)
            return instance

        with patch.object(sources, "_load_prefix_antigravity", return_value=prefix), \
                patch.object(sources, "_record_tasks_antigravity", return_value=()), \
                patch.object(sources, "_snapshot_from_records_antigravity", return_value=sentinel), \
                patch.object(sources, "_reverify_paths_antigravity"), \
                patch.object(sources, "ProcessPoolExecutor", side_effect=executor_factory), \
                patch.object(sources, "_verify_frozen_sources_single_worker_reference_antigravity") as reference:
            self.assertIs(sources.verify_frozen_sources_antigravity(), sentinel)
            observer_events: list[Mapping[str, object]] = []
            self.assertIs(sources._verify_frozen_sources_with_observer_antigravity(observer_events.append), sentinel)
        self.assertEqual([item.max_workers for item in executors], [11, 11])
        self.assertEqual(observer_events, [])
        self.assertEqual(tuple(inspect.signature(sources.verify_frozen_sources_antigravity).parameters), ())
        self.assertEqual(tuple(inspect.signature(sources._verify_frozen_sources_with_observer_antigravity).parameters), ("observer",))
        reference.assert_not_called()

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

    def test_only_test_or_final_audit_can_reference_sequential_reference_helper(self) -> None:
        # Break caught: a production entrypoint gaining a route to the sequential parity implementation.
        package = Path(sources.__file__).parent
        reference = "_verify_frozen_sources_single_worker_reference_antigravity"

        def forbidden_uses(tree: ast.AST, *, allow_private_definition: bool = False) -> list[str]:
            hits: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == reference:
                    if not allow_private_definition:
                        hits.append("definition")
                elif isinstance(node, ast.Name) and node.id == reference:
                    hits.append("name")
                elif isinstance(node, ast.Attribute) and node.attr == reference:
                    hits.append("attribute")
                elif isinstance(node, ast.ImportFrom) and any(alias.name == reference for alias in node.names):
                    hits.append("from-import")
                elif isinstance(node, ast.Import) and any(alias.name == reference or alias.asname == reference for alias in node.names):
                    hits.append("import")
            return hits

        self.assertEqual(forbidden_uses(ast.parse(f"from module import {reference} as alias")), ["from-import"])
        self.assertEqual(forbidden_uses(ast.parse(f"module.{reference}()")), ["attribute"])
        self.assertEqual(forbidden_uses(ast.parse(f"{reference}()")), ["name"])
        self.assertEqual(forbidden_uses(ast.parse(f"def {reference}():\n    pass"), allow_private_definition=True), [])
        self.assertIn("name", forbidden_uses(ast.parse(f"def {reference}():\n    pass\n{reference}()"), allow_private_definition=True))
        self.assertEqual(forbidden_uses(ast.parse(f"def {reference}():\n    pass\nmodule.{reference}()"), allow_private_definition=True), ["attribute"])
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text())
            self.assertEqual(forbidden_uses(tree, allow_private_definition=path.name == "sources_antigravity.py"), [], path.name)

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
