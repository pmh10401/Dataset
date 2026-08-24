"""Synthetic behavior tests for resumable Exp471 score preparation."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    score_cache_antigravity as score_cache,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    METHOD_ORDER_ANTIGRAVITY,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.source_authority_antigravity import (
    Exp471SourceSessionAntigravity,
)


_DIGEST_A_ANTIGRAVITY = "a" * 64
_DIGEST_B_ANTIGRAVITY = "b" * 64
_DATASETS_ANTIGRAVITY = ("dataset_alpha", "dataset_beta", "dataset_gamma")


def _authority_antigravity() -> Exp471SourceAuthorityAntigravity:
    source_files = {"synthetic_antigravity.py": _DIGEST_A_ANTIGRAVITY}
    tree = sha256(canonical_json_bytes_exp471_antigravity(source_files)).hexdigest()
    return Exp471SourceAuthorityAntigravity(
        source_commit=score_cache.EXP467_SOURCE_COMMIT_ANTIGRAVITY,
        source_context_sha256=score_cache.EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
        inventory_sha256=_DIGEST_A_ANTIGRAVITY,
        catalog_sha256=_DIGEST_A_ANTIGRAVITY,
        database_sha256=_DIGEST_A_ANTIGRAVITY,
        source_file_sha256=source_files,
        source_tree_sha256=tree,
        condition_manifest_sha256={
            condition: sha256(condition.encode("ascii")).hexdigest()
            for condition in METHOD_ORDER_ANTIGRAVITY[:4]
        },
        total_successes=score_cache.EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
        imported_total=score_cache.EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
        native_total=score_cache.EXP467_NATIVE_TOTAL_ANTIGRAVITY,
        authority_sha256=_DIGEST_B_ANTIGRAVITY,
    )


def _source_bundle_antigravity() -> dict[str, object]:
    conditions: dict[str, object] = {}
    baselines: dict[str, object] = {}
    for method_index, method in enumerate(METHOD_ORDER_ANTIGRAVITY):
        source_sha256 = sha256(f"source:{method}".encode("utf-8")).hexdigest()
        if method_index < 4:
            conditions[method] = {
                "source_manifest_sha256": source_sha256,
                "scores": {
                    dataset: [
                        float(method_index + dataset_index) + 0.125,
                        float(method_index + dataset_index) + 0.875,
                    ]
                    for dataset_index, dataset in enumerate(_DATASETS_ANTIGRAVITY)
                },
            }
        else:
            baselines[method] = {
                "source_manifest_sha256": source_sha256,
                "scores": {
                    dataset: {
                        "train_oof_scores": [0.0, 1.0, 2.0, 3.0],
                        "test_scores": [
                            float(dataset_index) + 0.5,
                            float(dataset_index) + 2.5,
                        ],
                    }
                    for dataset_index, dataset in enumerate(_DATASETS_ANTIGRAVITY)
                },
            }
    return {
        "schema": "exp471-verified-score-input-antigravity.v1",
        "dataset_order": list(_DATASETS_ANTIGRAVITY),
        "conditions": conditions,
        "baselines": baselines,
    }


def _session_antigravity(
    reader,
    *,
    label_callback=lambda: (_ for _ in ()).throw(
        AssertionError("TEST labels must remain unreachable")
    ),
) -> Exp471SourceSessionAntigravity:
    return Exp471SourceSessionAntigravity(
        authority=_authority_antigravity(),
        verification=object(),
        context=object(),
        catalog=object(),
        migration_manifest=object(),
        verifier=score_cache,
        score_reader_antigravity=reader,
        label_callback_antigravity=label_callback,
    )


def _tree_bytes_antigravity(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ScoreCacheTestsAntigravity(unittest.TestCase):
    def _issued_patches_antigravity(self, root: Path):
        return (
            mock.patch.object(
                score_cache,
                "_source_session_is_issued_antigravity",
                return_value=True,
            ),
            mock.patch.object(
                score_cache,
                "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY",
                root,
            ),
        )

    def test_prepares_exact_complete_order_and_raw_micro_contract(self) -> None:
        """Wrong ordering, delayed durability, or raw/micro mixing must fail."""
        label_calls: list[object] = []
        with tempfile.TemporaryDirectory(prefix="exp471_prepare_antigravity_") as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(
                _source_bundle_antigravity,
                label_callback=lambda: label_calls.append(object()),
            )
            events: list[dict[str, object]] = []

            def record_durable_event(event: dict[str, object]) -> None:
                if event["status"] == "durable":
                    reopened = score_cache.read_score_shard_antigravity(
                        root, str(event["method"]), str(event["dataset"])
                    )
                    self.assertEqual(
                        (reopened.method, reopened.dataset),
                        (event["method"], event["dataset"]),
                    )
                events.append(event)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=2, progress=record_durable_event
                )
                raw_condition = score_cache.read_score_shard_antigravity(
                    root, "A", "dataset_alpha"
                )
                raw_baseline = score_cache.read_score_shard_antigravity(
                    root, "knn_distance", "dataset_alpha"
                )

            self.assertEqual(manifest.total_shards, 30)
            self.assertEqual(tuple(manifest.method_order), METHOD_ORDER_ANTIGRAVITY)
            self.assertEqual(tuple(manifest.dataset_order), _DATASETS_ANTIGRAVITY)
            self.assertEqual(events[-1]["durable_completed"], 30)
            self.assertEqual(
                [(event["method"], event["dataset"]) for event in events if event["status"] == "durable"],
                [
                    (method, dataset)
                    for method in METHOD_ORDER_ANTIGRAVITY
                    for dataset in _DATASETS_ANTIGRAVITY
                ],
            )
            self.assertEqual(raw_condition.raw_scores, (0.125, 0.875))
            self.assertEqual(raw_condition.raw_scores, raw_condition.micro_scores)
            self.assertEqual(raw_baseline.raw_scores, (0.5, 2.5))
            self.assertEqual(
                raw_baseline.micro_scores, (0.19999999999999996, 0.6)
            )
            self.assertEqual(label_calls, [])

    def test_resume_omits_verified_durable_identity_from_worker_schedule(self) -> None:
        """A durable verified shard must not be recomputed after interruption."""
        with tempfile.TemporaryDirectory(prefix="exp471_resume_antigravity_") as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    score_cache.prepare_score_cache_antigravity(
                        session,
                        workers=1,
                        progress=lambda event: (
                            (_ for _ in ()).throw(RuntimeError("synthetic interruption"))
                            if event["status"] == "durable"
                            else None
                        ),
                    )
            first = score_cache.read_score_shard_antigravity(
                root, "A", "dataset_alpha"
            )
            calls: list[tuple[str, str]] = []
            original = score_cache._prepare_task_antigravity

            def recording(task):
                calls.append((task.method, task.dataset))
                return original(task)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache, "_prepare_task_antigravity", recording
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=1
                )
            reopened = score_cache.read_score_shard_antigravity(
                root, "A", "dataset_alpha"
            )
            self.assertEqual(manifest.total_shards, 30)
            self.assertNotIn(("A", "dataset_alpha"), calls)
            self.assertEqual(first, reopened)

    def test_failure_rows_are_unique_and_retry_waits_for_complete_first_wave(self) -> None:
        """Duplicate retry rows or an early retry must fail deterministic recovery."""
        with tempfile.TemporaryDirectory(prefix="exp471_retry_antigravity_") as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            attempts: dict[tuple[str, str], int] = {}
            original = score_cache._prepare_task_antigravity

            def flaky(task):
                identity = (task.method, task.dataset)
                attempts[identity] = attempts.get(identity, 0) + 1
                if identity == ("A", "dataset_alpha") and attempts[identity] == 1:
                    raise RuntimeError("one synthetic failure")
                return original(task)

            events: list[dict[str, object]] = []
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache, "_prepare_task_antigravity", flaky
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=1, progress=events.append
                )

            failures = [event for event in events if event["status"] == "failed"]
            retry_positions = [
                index for index, event in enumerate(events) if event["wave"] == "retry"
            ]
            first_wave_positions = [
                index for index, event in enumerate(events) if event["wave"] == "first"
            ]
            self.assertEqual(
                [(row["method"], row["dataset"], row["attempt"]) for row in failures],
                [("A", "dataset_alpha", 1)],
            )
            self.assertEqual(len({(row["method"], row["dataset"]) for row in failures}), 1)
            self.assertGreater(min(retry_positions), max(first_wave_positions))
            self.assertEqual(attempts[("A", "dataset_alpha")], 2)
            self.assertEqual(manifest.total_shards, 30)

    def test_second_failure_leaves_no_terminal_manifest(self) -> None:
        """A twice-failed identity must prevent terminal manifest publication."""
        with tempfile.TemporaryDirectory(prefix="exp471_failed_antigravity_") as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            original = score_cache._prepare_task_antigravity

            def broken(task):
                if (task.method, task.dataset) == ("A", "dataset_alpha"):
                    raise RuntimeError("persistent synthetic failure")
                return original(task)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache, "_prepare_task_antigravity", broken
            ):
                with self.assertRaisesRegex(RuntimeError, "persistent synthetic failure"):
                    score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertFalse((root / "score_cache_manifest_antigravity.json").exists())
            status = score_cache.status_score_cache_antigravity(root)
            self.assertFalse(status["complete"])
            self.assertEqual(status["durable_completed"], 29)

    def test_rejects_unissued_session_before_score_or_label_capabilities(self) -> None:
        """A forged session must not reach either source or label capability."""
        score_calls: list[object] = []
        label_calls: list[object] = []
        session = _session_antigravity(
            lambda: score_calls.append(object()),
            label_callback=lambda: label_calls.append(object()),
        )
        with self.assertRaisesRegex(ValueError, "issued"):
            score_cache.prepare_score_cache_antigravity(session, workers=1)
        self.assertEqual((score_calls, label_calls), ([], []))

    def test_rejects_nonfinite_input_before_any_worker_runs(self) -> None:
        """NaN or infinity must be rejected in the verified parent session."""
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory(
                prefix="exp471_finite_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "score_cache_antigravity"
                bundle = _source_bundle_antigravity()
                bundle["baselines"]["knn_distance"]["scores"]["dataset_alpha"][
                    "test_scores"
                ][0] = invalid
                session = _session_antigravity(lambda: bundle)
                worker_calls: list[object] = []
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root, mock.patch.object(
                    score_cache,
                    "_prepare_task_antigravity",
                    side_effect=lambda task: worker_calls.append(task),
                ):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        score_cache.prepare_score_cache_antigravity(session, workers=1)
                self.assertEqual(worker_calls, [])

    def test_one_and_two_worker_requests_produce_identical_complete_bytes(self) -> None:
        """Worker selection must never alter canonical shard or manifest bytes."""
        with tempfile.TemporaryDirectory(prefix="exp471_parity_antigravity_") as temporary:
            base = Path(temporary).resolve()
            roots = (
                base / "one_antigravity" / "score_cache_antigravity",
                base / "two_antigravity" / "score_cache_antigravity",
            )
            session = _session_antigravity(_source_bundle_antigravity)
            for workers, root in zip((1, 2), roots, strict=True):
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root:
                    score_cache.prepare_score_cache_antigravity(
                        session, workers=workers
                    )
            self.assertEqual(_tree_bytes_antigravity(roots[0]), _tree_bytes_antigravity(roots[1]))

    def test_open_complete_binds_manifest_to_issued_session_authority(self) -> None:
        """A complete cache from a different authority must fail closed."""
        with tempfile.TemporaryDirectory(prefix="exp471_open_antigravity_") as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)
                changed = replace(
                    session,
                    authority=replace(session.authority, authority_sha256="f" * 64),
                )
                with self.assertRaisesRegex(ValueError, "authority"):
                    score_cache.open_complete_score_cache_antigravity(changed)

    def test_benchmark_uses_measured_clock_and_requires_parity_and_speed_gate(self) -> None:
        """Parallel selection must require byte parity and at least 1.1x speed."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )[:3]
        with mock.patch.object(
            score_cache,
            "_monotonic_antigravity",
            side_effect=(10.0, 12.19, 20.0, 22.0),
        ):
            slow = score_cache.benchmark_score_preparation_antigravity(
                tasks, parallel_workers=2
            )
        self.assertEqual(slow["single_worker_seconds"], 2.19)
        self.assertEqual(slow["parallel_seconds"], 2.0)
        self.assertTrue(slow["byte_parity"])
        self.assertFalse(slow["parallel_accepted"])
        self.assertEqual(slow["selected_workers"], 1)
        self.assertIn("1.1x", slow["fallback_reason"])
        self.assertEqual(slow["inner_numerical_threads"], 1)
        self.assertEqual(slow["acceleration"]["selected_backend"], "numba_cpu")

    def test_public_synthetic_worker_benchmark(self) -> None:
        """Public literals exercise measured one-worker and CPU-minus-one provenance."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )
        result = score_cache.benchmark_score_preparation_antigravity(
            tasks, parallel_workers=min(11, score_cache.CPU_WORKERS_ANTIGRAVITY)
        )
        self.assertGreaterEqual(result["detected_logical_cpus"], 1)
        self.assertGreaterEqual(result["requested_parallel_workers"], 1)
        self.assertIn(result["selected_workers"], (1, result["requested_parallel_workers"]))
        self.assertGreaterEqual(result["single_worker_seconds"], 0.0)
        self.assertGreaterEqual(result["parallel_seconds"], 0.0)
        self.assertTrue(result["byte_parity"])
        if result["selected_workers"] != 1:
            self.assertGreaterEqual(result["speedup"], 1.1)

    def test_zero_elapsed_benchmark_remains_finite_and_falls_back(self) -> None:
        """A zero-resolution fake clock must not emit non-finite provenance."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )[:2]
        with mock.patch.object(
            score_cache,
            "_monotonic_antigravity",
            side_effect=(10.0, 11.0, 20.0, 20.0),
        ):
            result = score_cache.benchmark_score_preparation_antigravity(
                tasks, parallel_workers=2
            )
        self.assertTrue(math.isfinite(result["speedup"]))
        self.assertFalse(result["parallel_accepted"])
        canonical_json_bytes_exp471_antigravity(result)

    def test_cpu_parallel_requests_cannot_exceed_cpu_minus_one_policy(self) -> None:
        """An explicit request must not bypass the workspace CPU worker ceiling."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )[:2]
        with self.assertRaisesRegex(ValueError, "CPU worker policy"):
            score_cache.benchmark_score_preparation_antigravity(
                tasks,
                parallel_workers=score_cache.CPU_WORKERS_ANTIGRAVITY + 1,
            )
        reader_calls: list[object] = []
        session = _session_antigravity(
            lambda: reader_calls.append(_source_bundle_antigravity())
        )
        with mock.patch.object(
            score_cache,
            "_source_session_is_issued_antigravity",
            return_value=True,
        ), self.assertRaisesRegex(ValueError, "CPU worker policy"):
            score_cache.prepare_score_cache_antigravity(
                session, workers=score_cache.CPU_WORKERS_ANTIGRAVITY + 1
            )
        self.assertEqual(reader_calls, [])

    def test_parallel_smoke_failure_records_reason_and_keeps_cpu_fallback(self) -> None:
        """A broken process pool must not displace the canonical CPU path."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )[:2]
        with mock.patch.object(
            score_cache,
            "_shard_bytes_antigravity",
            side_effect=((b"same",), RuntimeError("synthetic pool failure")),
        ), mock.patch.object(
            score_cache,
            "_monotonic_antigravity",
            side_effect=(10.0, 11.0, 20.0, 20.5),
        ):
            result = score_cache.benchmark_score_preparation_antigravity(
                tasks, parallel_workers=2
            )
        self.assertEqual(result["selected_workers"], 1)
        self.assertFalse(result["parallel_accepted"])
        self.assertFalse(result["byte_parity"])
        self.assertEqual(result["parallel_error_type"], "RuntimeError")
        self.assertIn("synthetic pool failure", result["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
