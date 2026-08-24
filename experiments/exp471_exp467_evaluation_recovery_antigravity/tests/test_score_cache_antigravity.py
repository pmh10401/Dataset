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
    authority = _authority_antigravity()
    coordinates = {
        dataset: {
            "count": 2,
            "indices": [0, 1],
            "coordinates_sha256": sha256(
                canonical_json_bytes_exp471_antigravity(
                    {"dataset": dataset, "count": 2, "indices": [0, 1]}
                )
            ).hexdigest(),
        }
        for dataset in _DATASETS_ANTIGRAVITY
    }
    coordinates_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(coordinates)
    ).hexdigest()
    conditions: dict[str, object] = {}
    baselines: dict[str, object] = {}
    baseline_authority_sha256: dict[str, str] = {}
    for method_index, method in enumerate(METHOD_ORDER_ANTIGRAVITY):
        source_sha256 = sha256(f"source:{method}".encode("utf-8")).hexdigest()
        if method_index < 4:
            conditions[method] = {
                "source_manifest_sha256": authority.condition_manifest_sha256[
                    method
                ],
                "scores": {
                    dataset: {
                        "test_indices": [0, 1],
                        "test_scores": [
                            float(method_index + dataset_index) + 0.125,
                            float(method_index + dataset_index) + 0.875,
                        ],
                    }
                    for dataset_index, dataset in enumerate(_DATASETS_ANTIGRAVITY)
                },
            }
        else:
            baseline_bare = {
                "schema": "exp471-verified-baseline-authority-antigravity.v1",
                "method": method,
                "database_sha256": authority.database_sha256,
                "dataset_order": list(_DATASETS_ANTIGRAVITY),
                "test_coordinates_sha256": coordinates_sha256,
                "source_manifest_sha256": source_sha256,
            }
            baseline_digest = sha256(
                canonical_json_bytes_exp471_antigravity(baseline_bare)
            ).hexdigest()
            baseline_authority_sha256[method] = baseline_digest
            baselines[method] = {
                "authority": {
                    **baseline_bare,
                    "authority_sha256": baseline_digest,
                },
                "scores": {
                    dataset: {
                        "train_oof_scores": [0.0, 1.0, 2.0, 3.0],
                        "test_indices": [0, 1],
                        "test_scores": [
                            float(dataset_index) + 0.5,
                            float(dataset_index) + 2.5,
                        ],
                    }
                    for dataset_index, dataset in enumerate(_DATASETS_ANTIGRAVITY)
                },
            }
    return {
        "schema": "exp471-verified-score-input-antigravity.v2",
        "database_sha256": authority.database_sha256,
        "dataset_order": list(_DATASETS_ANTIGRAVITY),
        "test_coordinates": coordinates,
        "baseline_authority_sha256": baseline_authority_sha256,
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


def _cache_bytes_antigravity(root: Path) -> dict[str, bytes]:
    return {
        relative: payload
        for relative, payload in _tree_bytes_antigravity(root).items()
        if relative == "score_cache_manifest_antigravity.json"
        or relative.startswith("score_shards_antigravity/")
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

    def _assert_input_rejected_before_compute_antigravity(
        self, bundle: dict[str, object], pattern: str
    ) -> None:
        label_calls: list[object] = []
        worker_calls: list[object] = []
        calibration_calls: list[object] = []
        original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

        def record_calibration(reference, query):
            calibration_calls.append((reference, query))
            return original_calibration(reference, query)

        with tempfile.TemporaryDirectory(
            prefix="exp471_invalid_input_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(
                lambda: bundle,
                label_callback=lambda: label_calls.append(object()),
            )
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_prepare_task_antigravity",
                side_effect=lambda task: worker_calls.append(task),
            ), mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                record_calibration,
            ), self.assertRaisesRegex(ValueError, pattern):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
        self.assertEqual(worker_calls, [])
        self.assertEqual(calibration_calls, [])
        self.assertEqual(label_calls, [])

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
        """Durable identities must skip source preparation and calibration."""
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
                            and (event["method"], event["dataset"])
                            == ("knn_distance", "dataset_alpha")
                            else None
                        ),
                    )
            first = score_cache.read_score_shard_antigravity(
                root, "A", "dataset_alpha"
            )
            worker_calls: list[tuple[str, str]] = []
            source_calls: list[tuple[str, str]] = []
            calibration_calls: list[object] = []
            original_worker = score_cache._prepare_task_antigravity
            original_source = score_cache._prepare_score_identity_antigravity
            original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

            def record_worker(task):
                worker_calls.append((task.method, task.dataset))
                return original_worker(task)

            def record_source(metadata, method, dataset):
                source_calls.append((method, dataset))
                return original_source(metadata, method, dataset)

            def record_calibration(reference, query):
                calibration_calls.append((reference.size, query.size))
                return original_calibration(reference, query)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache, "_prepare_task_antigravity", record_worker
            ), mock.patch.object(
                score_cache, "_prepare_score_identity_antigravity", record_source
            ), mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                record_calibration,
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=1
                )
            reopened = score_cache.read_score_shard_antigravity(
                root, "A", "dataset_alpha"
            )
            self.assertEqual(manifest.total_shards, 30)
            self.assertEqual(len(source_calls), 17)
            self.assertEqual(len(worker_calls), 17)
            self.assertEqual(len(calibration_calls), 17)
            self.assertNotIn(("A", "dataset_alpha"), source_calls)
            self.assertNotIn(("knn_distance", "dataset_alpha"), source_calls)
            self.assertEqual(first, reopened)

    def test_complete_reopen_skips_all_score_preparation_and_calibration(self) -> None:
        """A complete verified cache must reopen without rebuilding any score."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_complete_resume_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                first = score_cache.prepare_score_cache_antigravity(
                    session, workers=1
                )

            source_calls: list[object] = []
            calibration_calls: list[object] = []
            original_source = score_cache._prepare_score_identity_antigravity
            original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

            def record_source(metadata, method, dataset):
                source_calls.append((method, dataset))
                return original_source(metadata, method, dataset)

            def record_calibration(reference, query):
                calibration_calls.append((reference, query))
                return original_calibration(reference, query)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_prepare_score_identity_antigravity",
                record_source,
            ), mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                record_calibration,
            ):
                second = score_cache.prepare_score_cache_antigravity(
                    session, workers=1
                )
                opened = score_cache.open_complete_score_cache_antigravity(
                    session
                )
            self.assertEqual((first, second, opened), (first, first, first))
            self.assertEqual(source_calls, [])
            self.assertEqual(calibration_calls, [])

    def test_score_bundle_authority_database_and_coordinates_fail_closed(self) -> None:
        """Unbound source identities or TEST coordinates must never reach compute."""
        wrong_condition = _source_bundle_antigravity()
        wrong_condition["conditions"]["A"]["source_manifest_sha256"] = "f" * 64
        self._assert_input_rejected_before_compute_antigravity(
            wrong_condition, "condition manifest authority"
        )

        wrong_baseline_authority = _source_bundle_antigravity()
        wrong_baseline_authority["baseline_authority_sha256"]["knn_distance"] = (
            "f" * 64
        )
        self._assert_input_rejected_before_compute_antigravity(
            wrong_baseline_authority, "baseline authority"
        )

        wrong_database = _source_bundle_antigravity()
        wrong_database["baselines"]["knn_distance"]["authority"][
            "database_sha256"
        ] = "f" * 64
        self._assert_input_rejected_before_compute_antigravity(
            wrong_database, "database identity"
        )

        coordinate_mutations = (
            [0],
            [0, 1, 2],
            [1, 0],
        )
        for indices in coordinate_mutations:
            with self.subTest(indices=indices):
                wrong_coordinates = _source_bundle_antigravity()
                wrong_coordinates["baselines"]["knn_distance"]["scores"][
                    "dataset_alpha"
                ]["test_indices"] = indices
                self._assert_input_rejected_before_compute_antigravity(
                    wrong_coordinates, "TEST coordinates"
                )

        wrong_condition_order = _source_bundle_antigravity()
        wrong_condition_order["conditions"]["A"]["scores"]["dataset_alpha"][
            "test_indices"
        ] = [1, 0]
        self._assert_input_rejected_before_compute_antigravity(
            wrong_condition_order, "TEST coordinates"
        )

    def test_wrong_shape_and_unequal_test_count_fail_before_compute(self) -> None:
        """All ten methods must share one finite 1-D TEST row shape per dataset."""
        wrong_shape = _source_bundle_antigravity()
        wrong_shape["conditions"]["A"]["scores"]["dataset_alpha"][
            "test_scores"
        ] = [[0.125, 0.875]]
        self._assert_input_rejected_before_compute_antigravity(
            wrong_shape, "finite float64 vector"
        )

        wrong_count = _source_bundle_antigravity()
        wrong_count["baselines"]["knn_distance"]["scores"]["dataset_alpha"][
            "test_scores"
        ] = [0.5]
        self._assert_input_rejected_before_compute_antigravity(
            wrong_count, "TEST count"
        )

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
        """An accepted two-worker wave must match canonical one-worker bytes."""
        with tempfile.TemporaryDirectory(prefix="exp471_parity_antigravity_") as temporary:
            base = Path(temporary).resolve()
            roots = (
                base / "one_antigravity" / "score_cache_antigravity",
                base / "two_antigravity" / "score_cache_antigravity",
            )
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(roots[0])
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)

            worker_counts: list[int] = []
            original = score_cache._run_tasks_antigravity

            def record_wave(tasks, workers):
                worker_counts.append(workers)
                return original(tasks, workers)

            issued, cache_root = self._issued_patches_antigravity(roots[1])
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_monotonic_antigravity",
                side_effect=(10.0, 12.2, 20.0, 21.0),
            ), mock.patch.object(
                score_cache, "_run_tasks_antigravity", record_wave
            ):
                score_cache.prepare_score_cache_antigravity(session, workers=2)
            self.assertGreaterEqual(worker_counts.count(2), 2)
            self.assertEqual(
                _cache_bytes_antigravity(roots[0]),
                _cache_bytes_antigravity(roots[1]),
            )
            status = score_cache.status_score_cache_antigravity(roots[1])
            decision = status["benchmark_decision"]
            self.assertEqual(decision["selected_workers"], 2)
            self.assertTrue(decision["parallel_accepted"])
            self.assertEqual(decision["speedup_threshold"], 1.1)
            self.assertTrue(
                (roots[1] / "score_preparation_benchmark_antigravity.json").is_file()
            )
            canonical_json_bytes_exp471_antigravity(status)

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

    def test_open_complete_reconstructs_expected_cache_context(self) -> None:
        """A self-consistent but different baseline authority must not reopen cache."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_open_context_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)

            changed_bundle = _source_bundle_antigravity()
            baseline_authority = changed_bundle["baselines"]["knn_distance"][
                "authority"
            ]
            baseline_authority["source_manifest_sha256"] = "f" * 64
            bare = {
                key: value
                for key, value in baseline_authority.items()
                if key != "authority_sha256"
            }
            changed_digest = sha256(
                canonical_json_bytes_exp471_antigravity(bare)
            ).hexdigest()
            baseline_authority["authority_sha256"] = changed_digest
            changed_bundle["baseline_authority_sha256"][
                "knn_distance"
            ] = changed_digest
            changed_session = _session_antigravity(lambda: changed_bundle)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, self.assertRaisesRegex(ValueError, "context"):
                score_cache.open_complete_score_cache_antigravity(changed_session)

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
        self.assertEqual(slow["speedup_threshold"], 1.1)
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

    def test_benchmark_acceptance_and_parity_failure_are_injected_and_finite(self) -> None:
        """Injected measurements cover accepted and byte-mismatch decisions."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )[:2]
        with mock.patch.object(
            score_cache,
            "_shard_bytes_antigravity",
            side_effect=((b"same",), (b"same",)),
        ), mock.patch.object(
            score_cache,
            "_monotonic_antigravity",
            side_effect=(10.0, 12.2, 20.0, 21.0),
        ):
            accepted = score_cache.benchmark_score_preparation_antigravity(
                tasks, parallel_workers=2
            )
        self.assertTrue(accepted["parallel_accepted"])
        self.assertEqual(accepted["selected_workers"], 2)
        self.assertEqual(accepted["speedup_threshold"], 1.1)
        canonical_json_bytes_exp471_antigravity(accepted)

        with mock.patch.object(
            score_cache,
            "_shard_bytes_antigravity",
            side_effect=((b"one",), (b"different",)),
        ), mock.patch.object(
            score_cache,
            "_monotonic_antigravity",
            side_effect=(10.0, 12.2, 20.0, 21.0),
        ):
            mismatch = score_cache.benchmark_score_preparation_antigravity(
                tasks, parallel_workers=2
            )
        self.assertFalse(mismatch["byte_parity"])
        self.assertFalse(mismatch["parallel_accepted"])
        self.assertEqual(mismatch["selected_workers"], 1)
        canonical_json_bytes_exp471_antigravity(mismatch)

    def test_actual_parallel_wave_failure_falls_back_and_persists_decision(self) -> None:
        """Accepted smoke plus actual pool failure must finish on deterministic CPU."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_actual_pool_fallback_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            original = score_cache._run_tasks_antigravity
            parallel_calls = 0
            worker_counts: list[int] = []

            def fail_actual_wave(tasks, workers):
                nonlocal parallel_calls
                worker_counts.append(workers)
                if workers == 2:
                    parallel_calls += 1
                    if parallel_calls == 2:
                        raise RuntimeError("synthetic actual pool failure")
                return original(tasks, workers)

            events: list[dict[str, object]] = []
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_monotonic_antigravity",
                side_effect=(10.0, 12.2, 20.0, 21.0),
            ), mock.patch.object(
                score_cache, "_run_tasks_antigravity", fail_actual_wave
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=2, progress=events.append
                )
            self.assertEqual(manifest.total_shards, 30)
            self.assertEqual(worker_counts, [1, 2, 2, 1])
            self.assertEqual(
                [(event["method"], event["dataset"]) for event in events],
                [
                    (method, dataset)
                    for method in METHOD_ORDER_ANTIGRAVITY
                    for dataset in _DATASETS_ANTIGRAVITY
                ],
            )
            self.assertTrue(all(event["status"] == "durable" for event in events))
            decision = score_cache.status_score_cache_antigravity(root)[
                "benchmark_decision"
            ]
            self.assertEqual(decision["selected_workers"], 1)
            self.assertFalse(decision["parallel_accepted"])
            self.assertEqual(decision["parallel_error_type"], "RuntimeError")
            self.assertIn("actual first wave", decision["fallback_reason"])

    def test_retry_parallel_infrastructure_failure_uses_same_cpu_fallback(self) -> None:
        """A retry pool failure must not become a second score-task failure."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_retry_pool_fallback_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            original = score_cache._run_tasks_antigravity
            calls: list[int] = []
            actual_wave_seen = False

            def fail_retry_wave(tasks, workers):
                nonlocal actual_wave_seen
                calls.append(workers)
                outcomes = original(tasks, workers)
                if workers == 2 and len(tasks) == 30 and calls.count(2) == 2:
                    actual_wave_seen = True
                    first = outcomes[0]
                    return (
                        score_cache._TaskOutcomeAntigravity(
                            first.task,
                            None,
                            "ValueError",
                            "synthetic score failure",
                        ),
                        *outcomes[1:],
                    )
                if workers == 2 and actual_wave_seen and len(tasks) == 1:
                    raise RuntimeError("synthetic retry pool failure")
                return outcomes

            events: list[dict[str, object]] = []
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_monotonic_antigravity",
                side_effect=(10.0, 12.2, 20.0, 21.0),
            ), mock.patch.object(
                score_cache, "_run_tasks_antigravity", fail_retry_wave
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=2, progress=events.append
                )
            self.assertEqual(manifest.total_shards, 30)
            self.assertEqual(calls, [1, 2, 2, 2, 1])
            failures = [event for event in events if event["status"] == "failed"]
            self.assertEqual(
                [(event["method"], event["dataset"]) for event in failures],
                [("A", "dataset_alpha")],
            )
            decision = score_cache.status_score_cache_antigravity(root)[
                "benchmark_decision"
            ]
            self.assertEqual(decision["selected_workers"], 1)
            self.assertIn("actual retry wave", decision["fallback_reason"])

    def test_complete_status_requires_context_bound_create_only_decision(self) -> None:
        """Complete status must fail when its decision is missing or replaceable."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_status_binding_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            status = score_cache.status_score_cache_antigravity(root)
            decision = status["benchmark_decision"]
            score_cache._publish_benchmark_decision_antigravity(root, decision)
            changed = dict(decision)
            changed["fallback_reason"] = "different literal decision"
            changed = score_cache._finalize_benchmark_decision_antigravity(changed)
            with self.assertRaisesRegex(ValueError, "existing"):
                score_cache._publish_benchmark_decision_antigravity(root, changed)

            benchmark_path = root / "score_preparation_benchmark_antigravity.json"
            benchmark_path.chmod(0o600)
            benchmark_path.unlink()
            with self.assertRaisesRegex(ValueError, "benchmark decision is missing"):
                score_cache.status_score_cache_antigravity(root)

    def test_extreme_clock_measurements_keep_benchmark_json_finite(self) -> None:
        """Overflowing synthetic speedup must conservatively remain finite."""
        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            _authority_antigravity(), _source_bundle_antigravity()
        )[:2]
        with mock.patch.object(
            score_cache,
            "_shard_bytes_antigravity",
            side_effect=((b"same",), (b"same",)),
        ), mock.patch.object(
            score_cache,
            "_monotonic_antigravity",
            side_effect=(0.0, 1.0e308, 0.0, 5.0e-324),
        ):
            result = score_cache.benchmark_score_preparation_antigravity(
                tasks, parallel_workers=2
            )
        self.assertTrue(math.isfinite(result["speedup"]))
        canonical_json_bytes_exp471_antigravity(result)


if __name__ == "__main__":
    unittest.main()
