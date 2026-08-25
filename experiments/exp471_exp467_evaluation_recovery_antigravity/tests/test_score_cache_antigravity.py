"""Synthetic behavior tests for resumable Exp471 score preparation."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
import os
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
    Exp471ScoreCacheManifestAntigravity,
    Exp471ScoreShardAntigravity,
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.source_authority_antigravity import (
    Exp471SourceSessionAntigravity,
)


_DIGEST_A_ANTIGRAVITY = "a" * 64
_DIGEST_B_ANTIGRAVITY = "b" * 64
_DATASETS_ANTIGRAVITY = ("dataset_alpha", "dataset_beta", "dataset_gamma")
_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY = {
    "knn_distance": "classical",
    "lof_novelty": "classical",
    "ocsvm_rbf": "classical",
    "isolation_forest": "classical",
    "deep_svdd_1d": "deep_svdd",
    "drocc_1d": "drocc",
}
_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY = {
    root: {
        "manifest_sha256": sha256(
            f"synthetic manifest:{root}".encode("utf-8")
        ).hexdigest(),
        "run_config_sha256": sha256(
            f"synthetic run config:{root}".encode("utf-8")
        ).hexdigest(),
    }
    for root in ("classical", "deep_svdd", "drocc")
}


def _test_coordinates_antigravity() -> dict[str, object]:
    return {
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


def _coordinate_inventory_sha256_antigravity(
    coordinates: dict[str, object],
) -> str:
    record = {
        "schema": "exp471-test-coordinate-inventory-antigravity.v1",
        "dataset_count": len(coordinates),
        "datasets": [
            {
                "dataset": dataset,
                "test_count": coordinates[dataset]["count"],
                "test_indices": list(coordinates[dataset]["indices"]),
            }
            for dataset in coordinates
        ],
    }
    return sha256(canonical_json_bytes_exp471_antigravity(record)).hexdigest()


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
    coordinates = _test_coordinates_antigravity()
    coordinates_sha256 = _coordinate_inventory_sha256_antigravity(coordinates)
    conditions: dict[str, object] = {}
    baselines: dict[str, object] = {}
    baseline_authority_sha256: dict[str, str] = {}
    for method_index, method in enumerate(METHOD_ORDER_ANTIGRAVITY):
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
            root = _BASELINE_ROOT_BY_METHOD_ANTIGRAVITY[method]
            root_authority = _BASELINE_ROOT_AUTHORITY_ANTIGRAVITY[root]
            baseline_bare = {
                "schema": "exp471-verified-baseline-authority-antigravity.v1",
                "method": method,
                "database_sha256": authority.database_sha256,
                "dataset_order": list(_DATASETS_ANTIGRAVITY),
                "test_coordinates_sha256": coordinates_sha256,
                "source_manifest_sha256": root_authority["manifest_sha256"],
                "run_config_sha256": root_authority["run_config_sha256"],
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
                        "status": "scored",
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
        "schema": "exp471-verified-score-input-antigravity.v3",
        "database_sha256": authority.database_sha256,
        "dataset_order": list(_DATASETS_ANTIGRAVITY),
        "test_coordinates": coordinates,
        "baseline_authority_sha256": baseline_authority_sha256,
        "coverage": {
            "total_identities": 30,
            "scored_identities": 30,
            "abstained_identities": 0,
            "failed_identities": 0,
            "condition_scored_identities": 12,
            "baseline_scored_identities": 18,
            "baseline_abstained_identities": 0,
        },
        "conditions": conditions,
        "baselines": baselines,
    }


def _mixed_source_bundle_antigravity() -> dict[str, object]:
    bundle = _source_bundle_antigravity()
    bundle["baselines"]["deep_svdd_1d"]["scores"]["dataset_alpha"] = {
        "status": "abstained",
        "test_indices": [0, 1],
        "reason": "insufficient_train_rows",
    }
    bundle["coverage"] = {
        "total_identities": 30,
        "scored_identities": 29,
        "abstained_identities": 1,
        "failed_identities": 0,
        "condition_scored_identities": 12,
        "baseline_scored_identities": 17,
        "baseline_abstained_identities": 1,
    }
    return bundle


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


def _rehashed_shard_antigravity(
    shard: Exp471ScoreShardAntigravity, **changes: object
) -> Exp471ScoreShardAntigravity:
    payload = {
        "schema": "exp471-score-cache-shard-antigravity.v2",
        "source_authority_sha256": shard.source_authority_sha256,
        "cache_context_sha256": shard.cache_context_sha256,
        "method": shard.method,
        "dataset": shard.dataset,
        "status": shard.status,
        "test_count": shard.test_count,
        "test_indices": list(shard.test_indices),
        "raw_scores": list(shard.raw_scores),
        "micro_scores": list(shard.micro_scores),
        "abstention_reason": shard.abstention_reason,
        "source_manifest_sha256": shard.source_manifest_sha256,
    }
    payload.update(changes)
    payload_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(payload)
    ).hexdigest()
    envelope_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(
            {**payload, "payload_sha256": payload_sha256}
        )
    ).hexdigest()
    return Exp471ScoreShardAntigravity(
        **{key: value for key, value in payload.items() if key != "schema"},
        payload_sha256=payload_sha256,
        envelope_sha256=envelope_sha256,
    )


def _rewrite_complete_manifest_antigravity(
    root: Path,
    manifest: Exp471ScoreCacheManifestAntigravity,
    changed_shard: Exp471ScoreShardAntigravity,
) -> None:
    shard_digests = dict(manifest.shard_sha256)
    shard_digests[f"{changed_shard.method}:{changed_shard.dataset}"] = (
        changed_shard.envelope_sha256
    )
    status_counts = {"scored": 0, "abstained": 0}
    for method in manifest.method_order:
        for dataset in manifest.dataset_order:
            shard = score_cache.read_score_shard_antigravity(root, method, dataset)
            status_counts[shard.status] += 1
    record = {
        "schema": "exp471-score-cache-manifest-antigravity.v2",
        "source_authority_sha256": manifest.source_authority_sha256,
        "cache_context_sha256": manifest.cache_context_sha256,
        "method_order": list(manifest.method_order),
        "dataset_order": list(manifest.dataset_order),
        "shard_sha256": shard_digests,
        "total_shards": manifest.total_shards,
        "scored_shards": status_counts["scored"],
        "abstained_shards": status_counts["abstained"],
        "failed_shards": 0,
    }
    rewritten = Exp471ScoreCacheManifestAntigravity(
        **{key: value for key, value in record.items() if key != "schema"},
        manifest_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(record)
        ).hexdigest(),
    )
    path = root / "score_cache_manifest_antigravity.json"
    path.write_bytes(
        score_cache._artifacts_antigravity._canonical_manifest_bytes_antigravity(
            rewritten
        )
    )


def _insert_extra_shard_leaf_antigravity(root: Path) -> Path:
    source = root / score_cache.score_shard_relative_path_antigravity(
        "A", "dataset_alpha"
    )
    extra = source.parent / "foreign_identity_antigravity.json"
    extra.write_bytes(source.read_bytes())
    return extra


def _remove_complete_manifest_antigravity(root: Path) -> None:
    manifest = root / "score_cache_manifest_antigravity.json"
    manifest.chmod(0o600)
    manifest.unlink()


def _rewrite_receipt_antigravity(
    path: Path, **changes: object
) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(changes)
    bare = {key: value for key, value in record.items() if key != "receipt_sha256"}
    record["receipt_sha256"] = sha256(
        canonical_json_bytes_exp471_antigravity(bare)
    ).hexdigest()
    path.chmod(0o600)
    path.write_bytes(canonical_json_bytes_exp471_antigravity(record))


def _manifest_recoveries_antigravity(
    root: Path,
) -> tuple[tuple[Path, dict[str, object]], ...]:
    recoveries: list[tuple[Path, dict[str, object]]] = []
    for directory in sorted(
        root.glob("score_cache_manifest_recovery_*_antigravity")
    ):
        metadata = (
            directory / "score_cache_manifest_recovery_metadata_antigravity.json"
        )
        raw = metadata.read_bytes()
        record = json.loads(raw.decode("utf-8"))
        if raw != canonical_json_bytes_exp471_antigravity(record):
            raise AssertionError("manifest recovery metadata is noncanonical")
        recoveries.append((directory, record))
    return tuple(recoveries)


class ScoreCacheTestsAntigravity(unittest.TestCase):
    def setUp(self) -> None:
        coordinates_sha256 = _coordinate_inventory_sha256_antigravity(
            _test_coordinates_antigravity()
        )
        patches = (
            mock.patch.object(
                score_cache,
                "EXP471_BASELINE_DATABASE_SHA256_ANTIGRAVITY",
                _DIGEST_A_ANTIGRAVITY,
                create=True,
            ),
            mock.patch.object(
                score_cache,
                "EXP471_BASELINE_DATASET_COUNT_ANTIGRAVITY",
                len(_DATASETS_ANTIGRAVITY),
                create=True,
            ),
            mock.patch.object(
                score_cache,
                "EXP471_TEST_COORDINATE_INVENTORY_SHA256_ANTIGRAVITY",
                coordinates_sha256,
                create=True,
            ),
            mock.patch.object(
                score_cache,
                "EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY",
                _BASELINE_ROOT_BY_METHOD_ANTIGRAVITY,
                create=True,
            ),
            mock.patch.object(
                score_cache,
                "EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY",
                _BASELINE_ROOT_AUTHORITY_ANTIGRAVITY,
                create=True,
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

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

    def test_typed_abstention_is_parent_published_and_never_scheduled(self) -> None:
        """An abstention entering calibration or a worker is a routing bug."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_typed_abstention_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            scheduled: list[tuple[str, str]] = []
            calibration_calls: list[object] = []
            progress: list[dict[str, object]] = []
            original_run = score_cache._run_tasks_antigravity
            original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

            def record_run(tasks, workers):
                scheduled.extend((task.method, task.dataset) for task in tasks)
                return original_run(tasks, workers)

            def record_calibration(reference, query):
                calibration_calls.append((reference.size, query.size))
                return original_calibration(reference, query)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache, "_run_tasks_antigravity", record_run
            ), mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                record_calibration,
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=1, progress=progress.append
                )

            abstained_identity = ("deep_svdd_1d", "dataset_alpha")
            self.assertNotIn(abstained_identity, scheduled)
            self.assertEqual(len(calibration_calls), 17)
            shard = score_cache.read_score_shard_antigravity(
                root, *abstained_identity
            )
            self.assertEqual(shard.status, "abstained")
            self.assertEqual(shard.test_indices, (0, 1))
            self.assertEqual(shard.raw_scores, ())
            self.assertEqual(shard.micro_scores, ())
            self.assertEqual(shard.abstention_reason, "insufficient_train_rows")
            self.assertEqual(
                (manifest.scored_shards, manifest.abstained_shards, manifest.failed_shards),
                (29, 1, 0),
            )
            abstention_events = [
                event for event in progress if event["status"] == "abstained"
            ]
            self.assertEqual(
                [(event["method"], event["dataset"]) for event in abstention_events],
                [abstained_identity],
            )
            for event in progress:
                self.assertEqual(
                    set(event) - {"error_type", "error_message", "method", "dataset", "wave", "attempt", "status"},
                    {"schema", "durable_completed", "scored", "abstained", "failed", "remaining", "elapsed_seconds", "eta_seconds"},
                )

    def test_parent_abstention_interruption_is_observable_without_benchmark(self) -> None:
        """A parent-only durable shard must not invent measured worker evidence."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_parent_abstention_interrupt_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, self.assertRaisesRegex(
                RuntimeError, "parent interruption"
            ):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=1,
                    progress=lambda event: (
                        (_ for _ in ()).throw(RuntimeError("parent interruption"))
                        if event["status"] == "abstained" else None
                    ),
                )
            status = score_cache.status_score_cache_antigravity(root)
            self.assertFalse(status["complete"])
            self.assertEqual(
                (status["scored"], status["abstained"], status["failed"]),
                (0, 1, 0),
            )
            self.assertIsNone(status["benchmark_decision"])
            self.assertIsNone(status["execution_decision"])

    def test_mixed_interruption_resumes_both_arms_without_republish(self) -> None:
        """A mixed restart must preserve existing scored and abstained bytes."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_mixed_resume_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            interrupted_root = base / "interrupted_antigravity" / "score_cache_antigravity"
            fresh_root = base / "fresh_antigravity" / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(interrupted_root)
            with issued, cache_root, self.assertRaisesRegex(
                RuntimeError, "mixed interruption"
            ):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=1,
                    progress=lambda event: (
                        (_ for _ in ()).throw(RuntimeError("mixed interruption"))
                        if event["status"] == "durable"
                        and (event["method"], event["dataset"])
                        == ("knn_distance", "dataset_alpha")
                        else None
                    ),
                )
            existing = {
                ("deep_svdd_1d", "dataset_alpha"): score_cache.read_score_shard_antigravity(
                    interrupted_root, "deep_svdd_1d", "dataset_alpha"
                ),
                ("A", "dataset_alpha"): score_cache.read_score_shard_antigravity(
                    interrupted_root, "A", "dataset_alpha"
                ),
                ("knn_distance", "dataset_alpha"): score_cache.read_score_shard_antigravity(
                    interrupted_root, "knn_distance", "dataset_alpha"
                ),
            }
            existing_bytes = {
                identity: (interrupted_root / score_cache.score_shard_relative_path_antigravity(*identity)).read_bytes()
                for identity in existing
            }
            published: list[tuple[str, str]] = []
            calibration_calls: list[object] = []
            original_publish = score_cache.publish_score_shard_antigravity
            original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

            def record_publish(root, shard):
                published.append((shard.method, shard.dataset))
                return original_publish(root, shard)

            def record_calibration(reference, query):
                calibration_calls.append((reference.size, query.size))
                return original_calibration(reference, query)

            issued, cache_root = self._issued_patches_antigravity(interrupted_root)
            with issued, cache_root, mock.patch.object(
                score_cache, "publish_score_shard_antigravity", record_publish
            ), mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                record_calibration,
            ):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertTrue(set(existing).isdisjoint(published))
            self.assertEqual(len(calibration_calls), 16)
            for identity, payload in existing_bytes.items():
                self.assertEqual(
                    (interrupted_root / score_cache.score_shard_relative_path_antigravity(*identity)).read_bytes(),
                    payload,
                )

            issued, cache_root = self._issued_patches_antigravity(fresh_root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertEqual(
                _cache_bytes_antigravity(interrupted_root),
                _cache_bytes_antigravity(fresh_root),
            )

    def test_scored_receipt_is_durable_before_publish_interruption(self) -> None:
        """A durable baseline shard must already have external micro authority."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_scored_receipt_interrupt_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            receipt_preceded_shard: list[bool] = []
            original_publish = score_cache.publish_score_shard_antigravity

            def publish_after_receipt(cache_root, shard):
                if (shard.method, shard.dataset) == (
                    "knn_distance",
                    "dataset_alpha",
                ):
                    receipt = (
                        root
                        / "score_expectations_antigravity"
                        / score_cache.score_shard_relative_path_antigravity(
                            shard.method, shard.dataset
                        ).name
                    )
                    receipt_preceded_shard.append(receipt.is_file())
                return original_publish(cache_root, shard)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "publish_score_shard_antigravity",
                publish_after_receipt,
            ), self.assertRaisesRegex(RuntimeError, "receipt interruption"):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=1,
                    progress=lambda event: (
                        (_ for _ in ()).throw(RuntimeError("receipt interruption"))
                        if event["status"] == "durable"
                        and (event["method"], event["dataset"])
                        == ("knn_distance", "dataset_alpha")
                        else None
                    ),
                )
            self.assertEqual(receipt_preceded_shard, [True])
            receipts = tuple(
                (root / "score_expectations_antigravity").glob("*_antigravity.json")
            )
            self.assertTrue(receipts)
            self.assertTrue(
                score_cache.read_score_shard_antigravity(
                    root, "knn_distance", "dataset_alpha"
                )
            )

    def test_partial_self_rehashed_wrong_baseline_micro_is_rejected(self) -> None:
        """Shard-local hashes cannot authorize changed baseline calibration."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_wrong_micro_resume_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, self.assertRaisesRegex(
                RuntimeError, "wrong micro interruption"
            ):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=1,
                    progress=lambda event: (
                        (_ for _ in ()).throw(
                            RuntimeError("wrong micro interruption")
                        )
                        if event["status"] == "durable"
                        and (event["method"], event["dataset"])
                        == ("knn_distance", "dataset_alpha")
                        else None
                    ),
                )
            original = score_cache.read_score_shard_antigravity(
                root, "knn_distance", "dataset_alpha"
            )
            changed = _rehashed_shard_antigravity(
                original, micro_scores=[0.123, 0.987]
            )
            path = root / score_cache.score_shard_relative_path_antigravity(
                changed.method, changed.dataset
            )
            path.chmod(0o600)
            path.write_bytes(
                score_cache._artifacts_antigravity._canonical_shard_bytes_antigravity(
                    changed
                )
            )
            calibration_calls: list[object] = []
            original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

            def record_calibration(reference, query):
                calibration_calls.append((reference.size, query.size))
                return original_calibration(reference, query)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                record_calibration,
            ), self.assertRaisesRegex(ValueError, "expected scored envelope"):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertEqual(calibration_calls, [])

    def test_complete_fast_path_rebinds_typed_partition_to_v3_metadata(self) -> None:
        """A self-consistent 30/0 cache cannot replace expected 29/1 coverage."""

        for api in ("prepare", "open"):
            with self.subTest(api=api), tempfile.TemporaryDirectory(
                prefix="exp471_complete_partition_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "score_cache_antigravity"
                session = _session_antigravity(_mixed_source_bundle_antigravity)
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root:
                    manifest = score_cache.prepare_score_cache_antigravity(
                        session, workers=1
                    )
                abstained = score_cache.read_score_shard_antigravity(
                    root, "deep_svdd_1d", "dataset_alpha"
                )
                changed = _rehashed_shard_antigravity(
                    abstained,
                    status="scored",
                    raw_scores=[0.2, 0.8],
                    micro_scores=[0.3, 0.9],
                    abstention_reason=None,
                )
                path = root / score_cache.score_shard_relative_path_antigravity(
                    changed.method, changed.dataset
                )
                path.chmod(0o600)
                path.write_bytes(
                    score_cache._artifacts_antigravity._canonical_shard_bytes_antigravity(
                        changed
                    )
                )
                manifest_path = root / "score_cache_manifest_antigravity.json"
                manifest_path.chmod(0o600)
                _rewrite_complete_manifest_antigravity(root, manifest, changed)
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root, self.assertRaisesRegex(
                    ValueError,
                    "verified input|status coverage|existing score shard",
                ):
                    if api == "prepare":
                        score_cache.prepare_score_cache_antigravity(
                            session, workers=1
                        )
                    else:
                        score_cache.open_complete_score_cache_antigravity(
                            session
                        )

    def test_partial_baseline_scored_without_receipt_fails_before_recalibration(self) -> None:
        """A scored shard without its prior create-only receipt is untrusted."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_missing_receipt_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, self.assertRaisesRegex(
                RuntimeError, "missing receipt interruption"
            ):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=1,
                    progress=lambda event: (
                        (_ for _ in ()).throw(
                            RuntimeError("missing receipt interruption")
                        )
                        if event["status"] == "durable"
                        and (event["method"], event["dataset"])
                        == ("knn_distance", "dataset_alpha")
                        else None
                    ),
                )
            receipt_name = score_cache.score_shard_relative_path_antigravity(
                "knn_distance", "dataset_alpha"
            ).name
            receipt = root / "score_expectations_antigravity" / receipt_name
            receipt.chmod(0o600)
            receipt.unlink()
            calibration_calls: list[object] = []
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "upper_tail_pvalues_numba_antigravity",
                side_effect=lambda *args: calibration_calls.append(args),
            ), self.assertRaisesRegex(ValueError, "missing"):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertEqual(calibration_calls, [])

    def test_terminal_and_complete_fast_path_reject_extra_shard_leaf(self) -> None:
        """Declared 30-shard coverage must reject an actual 31st namespace leaf."""

        for complete in (False, True):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory(
                prefix="exp471_extra_leaf_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "score_cache_antigravity"
                session = _session_antigravity(_source_bundle_antigravity)
                issued, cache_root = self._issued_patches_antigravity(root)
                if complete:
                    with issued, cache_root:
                        score_cache.prepare_score_cache_antigravity(
                            session, workers=1
                        )
                else:
                    with issued, cache_root, self.assertRaisesRegex(
                        RuntimeError, "inventory interruption"
                    ):
                        score_cache.prepare_score_cache_antigravity(
                            session,
                            workers=1,
                            progress=lambda event: (
                                (_ for _ in ()).throw(
                                    RuntimeError("inventory interruption")
                                )
                                if event["status"] == "durable"
                                and event["durable_completed"] == 1
                                else None
                            ),
                        )
                source = root / score_cache.score_shard_relative_path_antigravity(
                    "A", "dataset_alpha"
                )
                extra = source.parent / "foreign_identity_antigravity.json"
                extra.write_bytes(source.read_bytes())
                self.assertEqual(len(tuple(source.parent.iterdir())), 31 if complete else 2)
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root, self.assertRaisesRegex(
                    ValueError, "inventory"
                ):
                    score_cache.prepare_score_cache_antigravity(
                        session, workers=1
                    )

    def test_terminal_manifest_link_mutation_rolls_back_false_complete(self) -> None:
        """An extra leaf introduced during manifest link must roll back the marker."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_link_inventory_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            original_link = score_cache._artifacts_antigravity.os.link
            injected = False
            linked_manifest_bytes: list[bytes] = []

            def insert_extra_then_link(source, destination, **kwargs):
                nonlocal injected
                if (
                    destination == "score_cache_manifest_antigravity.json"
                    and not injected
                ):
                    injected = True
                    _insert_extra_shard_leaf_antigravity(root)
                result = original_link(source, destination, **kwargs)
                if destination == "score_cache_manifest_antigravity.json":
                    linked_manifest_bytes.append(
                        (root / destination).read_bytes()
                    )
                return result

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache._artifacts_antigravity.os,
                "link",
                side_effect=insert_extra_then_link,
            ), self.assertRaisesRegex(ValueError, "inventory"):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertTrue(injected)
            self.assertFalse(
                (root / "score_cache_manifest_antigravity.json").exists()
            )
            recoveries = _manifest_recoveries_antigravity(root)
            self.assertEqual(len(recoveries), 1)
            directory, record = recoveries[0]
            recovered = directory / "score_cache_manifest_quarantined_antigravity.json"
            self.assertEqual(recovered.read_bytes(), linked_manifest_bytes[0])
            self.assertEqual(record["recovery_path"], str(recovered))

    def test_complete_prepare_and_open_reject_reopen_inventory_mutation(self) -> None:
        """First/last shard reopen mutations must invalidate complete authority."""

        targets = (
            ("first", "A", "dataset_alpha"),
            ("last", "deep_svdd_1d", "dataset_gamma"),
        )
        for api in ("prepare", "open"):
            for position, method, dataset in targets:
                with self.subTest(
                    api=api, position=position
                ), tempfile.TemporaryDirectory(
                    prefix="exp471_complete_reopen_inventory_antigravity_"
                ) as temporary:
                    root = Path(temporary).resolve() / "score_cache_antigravity"
                    session = _session_antigravity(_source_bundle_antigravity)
                    issued, cache_root = self._issued_patches_antigravity(root)
                    with issued, cache_root:
                        score_cache.prepare_score_cache_antigravity(
                            session, workers=1
                        )
                    target_name = score_cache.score_shard_relative_path_antigravity(
                        method, dataset
                    ).name
                    observed = 0

                    def mutate_during_reopen(event, relative):
                        nonlocal observed
                        if event == "cache:after_leaf_open" and relative == target_name:
                            observed += 1
                            if observed == 3:
                                _insert_extra_shard_leaf_antigravity(root)

                    issued, cache_root = self._issued_patches_antigravity(root)
                    with issued, cache_root, mock.patch.object(
                        score_cache._artifacts_antigravity,
                        "_race_hook_antigravity",
                        mutate_during_reopen,
                    ), self.assertRaisesRegex(ValueError, "inventory"):
                        if api == "prepare":
                            score_cache.prepare_score_cache_antigravity(
                                session, workers=1
                            )
                        else:
                            score_cache.open_complete_score_cache_antigravity(
                                session
                            )
                    self.assertGreaterEqual(observed, 3)
                    self.assertFalse(
                        (root / "score_cache_manifest_antigravity.json").exists()
                    )

    def test_complete_prepare_and_open_quarantine_early_verifier_mutation(self) -> None:
        """Both shared-verifier passes must quarantine false complete markers."""

        targets = (
            ("first", "A", "dataset_alpha"),
            ("last", "deep_svdd_1d", "dataset_gamma"),
        )
        for api in ("prepare", "open"):
            for verifier_pass in (1, 2):
                for position, method, dataset in targets:
                    with self.subTest(
                        api=api,
                        verifier_pass=verifier_pass,
                        position=position,
                    ), tempfile.TemporaryDirectory(
                        prefix="exp471_early_verifier_recovery_antigravity_"
                    ) as temporary:
                        root = Path(temporary).resolve() / "score_cache_antigravity"
                        session = _session_antigravity(_source_bundle_antigravity)
                        issued, cache_root = self._issued_patches_antigravity(root)
                        with issued, cache_root:
                            score_cache.prepare_score_cache_antigravity(
                                session, workers=1
                            )
                        manifest_path = (
                            root / "score_cache_manifest_antigravity.json"
                        )
                        manifest_bytes = manifest_path.read_bytes()
                        target_name = (
                            score_cache.score_shard_relative_path_antigravity(
                                method, dataset
                            ).name
                        )
                        observed = 0

                        def mutate_shared_verifier(event, relative):
                            nonlocal observed
                            if (
                                event == "cache:after_leaf_open"
                                and relative == target_name
                            ):
                                observed += 1
                                if observed == verifier_pass:
                                    _insert_extra_shard_leaf_antigravity(root)

                        issued, cache_root = self._issued_patches_antigravity(root)
                        with issued, cache_root, mock.patch.object(
                            score_cache._artifacts_antigravity,
                            "_race_hook_antigravity",
                            mutate_shared_verifier,
                        ), self.assertRaisesRegex(ValueError, "inventory"):
                            if api == "prepare":
                                score_cache.prepare_score_cache_antigravity(
                                    session, workers=1
                                )
                            else:
                                score_cache.open_complete_score_cache_antigravity(
                                    session
                                )
                        self.assertGreaterEqual(observed, verifier_pass)
                        self.assertFalse(manifest_path.exists())
                        recoveries = _manifest_recoveries_antigravity(root)
                        self.assertEqual(len(recoveries), 1)
                        directory, record = recoveries[0]
                        recovered = (
                            directory
                            / "score_cache_manifest_quarantined_antigravity.json"
                        )
                        self.assertEqual(recovered.read_bytes(), manifest_bytes)
                        self.assertEqual(directory.parent, root)
                        self.assertEqual(record["recovery_path"], str(recovered))
                        self.assertNotIn(directory, tuple((root / "score_shards_antigravity").iterdir()))

    def test_manifest_replacement_during_cleanup_survives_in_recovery(self) -> None:
        """A post-stat replacement must be moved intact, never unlinked."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_replacement_recovery_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            manifest = root / "score_cache_manifest_antigravity.json"
            expected = manifest.stat()
            replacement = root / "unrelated_manifest_replacement_antigravity.json"
            unrelated = b'{"unrelated_antigravity":true}\n'
            replacement.write_bytes(unrelated)
            replacement_identity = replacement.stat()
            target_name = score_cache.score_shard_relative_path_antigravity(
                "A", "dataset_alpha"
            ).name
            target_reads = 0
            replacement_injected = False

            def replace_during_recovery(event, relative):
                nonlocal target_reads, replacement_injected
                if event == "cache:after_leaf_open" and relative == target_name:
                    target_reads += 1
                    if target_reads == 3:
                        _insert_extra_shard_leaf_antigravity(root)
                if event == "cache:before_manifest_quarantine_rename":
                    replacement.replace(manifest)
                    replacement_injected = True

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache._artifacts_antigravity,
                "_race_hook_antigravity",
                replace_during_recovery,
            ), self.assertRaisesRegex(ValueError, "inventory"):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertTrue(replacement_injected)
            self.assertFalse(manifest.exists())
            recoveries = _manifest_recoveries_antigravity(root)
            self.assertEqual(len(recoveries), 1)
            directory, record = recoveries[0]
            recovered = directory / "score_cache_manifest_quarantined_antigravity.json"
            recovered_identity = recovered.stat()
            self.assertEqual(recovered.read_bytes(), unrelated)
            self.assertEqual(
                (recovered_identity.st_dev, recovered_identity.st_ino),
                (replacement_identity.st_dev, replacement_identity.st_ino),
            )
            self.assertEqual(
                record["expected_identity"],
                {"device": expected.st_dev, "inode": expected.st_ino},
            )
            self.assertEqual(
                record["observed_identity"],
                {
                    "device": replacement_identity.st_dev,
                    "inode": replacement_identity.st_ino,
                },
            )

    def test_manifest_recovery_collision_never_overwrites_unrelated_destination(self) -> None:
        """A late recovery leaf must block quarantine without losing either leaf."""

        for kind in ("regular", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(
                prefix=f"exp471_manifest_recovery_{kind}_collision_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "score_cache_antigravity"
                session = _session_antigravity(_source_bundle_antigravity)
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root:
                    score_cache.prepare_score_cache_antigravity(session, workers=1)
                manifest = root / "score_cache_manifest_antigravity.json"
                manifest_bytes = manifest.read_bytes()
                manifest_identity = manifest.stat()
                target_name = score_cache.score_shard_relative_path_antigravity(
                    "A", "dataset_alpha"
                ).name
                target_reads = 0
                injected = False
                destination: Path | None = None
                destination_identity: os.stat_result | None = None
                foreign_target: Path | None = None
                foreign_bytes = b'{"unrelated_antigravity":true}\n'

                def collide_during_recovery(event, relative):
                    nonlocal target_reads, injected, destination
                    nonlocal destination_identity, foreign_target
                    if event == "cache:after_leaf_open" and relative == target_name:
                        target_reads += 1
                        if target_reads == 1:
                            _insert_extra_shard_leaf_antigravity(root)
                    if event == "cache:before_manifest_quarantine_rename":
                        recovery_directories = tuple(
                            root.glob(
                                "score_cache_manifest_recovery_*_antigravity"
                            )
                        )
                        self.assertEqual(len(recovery_directories), 1)
                        destination = (
                            recovery_directories[0]
                            / "score_cache_manifest_quarantined_antigravity.json"
                        )
                        if kind == "regular":
                            destination.write_bytes(foreign_bytes)
                        else:
                            foreign_target = root / "foreign_recovery_target_antigravity.json"
                            foreign_target.write_bytes(foreign_bytes)
                            destination.symlink_to(foreign_target)
                        destination_identity = destination.lstat()
                        injected = True

                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root, mock.patch.object(
                    score_cache._artifacts_antigravity,
                    "_race_hook_antigravity",
                    collide_during_recovery,
                ), self.assertRaises(FileExistsError):
                    score_cache.prepare_score_cache_antigravity(session, workers=1)
                self.assertTrue(injected)
                self.assertIsNotNone(destination)
                self.assertIsNotNone(destination_identity)
                self.assertEqual(manifest.read_bytes(), manifest_bytes)
                self.assertEqual(
                    (manifest.stat().st_dev, manifest.stat().st_ino),
                    (manifest_identity.st_dev, manifest_identity.st_ino),
                )
                self.assertEqual(
                    (destination.lstat().st_dev, destination.lstat().st_ino),
                    (destination_identity.st_dev, destination_identity.st_ino),
                )
                if kind == "regular":
                    self.assertEqual(destination.read_bytes(), foreign_bytes)
                else:
                    self.assertTrue(destination.is_symlink())
                    self.assertEqual(os.readlink(destination), str(foreign_target))
                    self.assertEqual(foreign_target.read_bytes(), foreign_bytes)
                recovery_metadata = (
                    destination.parent
                    / "score_cache_manifest_recovery_metadata_antigravity.json"
                )
                self.assertFalse(recovery_metadata.exists())

    def test_repeated_manifest_recovery_preserves_newer_valid_marker(self) -> None:
        """Repeating one recovery operation must not move a newer manifest."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_recovery_idempotent_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            manifest = root / "score_cache_manifest_antigravity.json"
            manifest_bytes = manifest.read_bytes()
            expected = manifest.stat()
            reason = "repeat recovery test antigravity"
            first = score_cache._artifacts_antigravity._quarantine_cache_manifest_antigravity(
                root,
                reason=reason,
                expected_identity=(expected.st_dev, expected.st_ino),
            )
            self.assertFalse(manifest.exists())
            recoveries = _manifest_recoveries_antigravity(root)
            self.assertEqual(len(recoveries), 1)
            directory, _record = recoveries[0]
            before = {
                path.name: path.read_bytes()
                for path in directory.iterdir()
                if path.is_file()
            }
            manifest.write_bytes(manifest_bytes)
            newer = manifest.stat()
            second = score_cache._artifacts_antigravity._quarantine_cache_manifest_antigravity(
                root,
                reason=reason,
                expected_identity=(expected.st_dev, expected.st_ino),
            )
            after = {
                path.name: path.read_bytes()
                for path in directory.iterdir()
                if path.is_file()
            }
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(
                (manifest.stat().st_dev, manifest.stat().st_ino),
                (newer.st_dev, newer.st_ino),
            )
            self.assertEqual(len(_manifest_recoveries_antigravity(root)), 1)

    def test_successful_complete_paths_create_no_manifest_recovery(self) -> None:
        """Normal publish, prepare reuse, and open create no recovery artifacts."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_no_manifest_recovery_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root:
                score_cache.prepare_score_cache_antigravity(session, workers=1)
                score_cache.prepare_score_cache_antigravity(session, workers=1)
                score_cache.open_complete_score_cache_antigravity(session)
            self.assertEqual(_manifest_recoveries_antigravity(root), ())

    def test_receipt_only_interruption_recomputes_and_reuses_receipt(self) -> None:
        """A receipt durable before its shard safely anchors the resumed recompute."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_receipt_only_resume_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_mixed_source_bundle_antigravity)
            original_publish = score_cache.publish_score_shard_antigravity

            def interrupt_before_target_shard(cache_root, shard):
                if (shard.method, shard.dataset) == (
                    "knn_distance",
                    "dataset_alpha",
                ):
                    raise RuntimeError("receipt-only interruption")
                return original_publish(cache_root, shard)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "publish_score_shard_antigravity",
                interrupt_before_target_shard,
            ), self.assertRaisesRegex(RuntimeError, "receipt-only interruption"):
                score_cache.prepare_score_cache_antigravity(session, workers=1)

            name = score_cache.score_shard_relative_path_antigravity(
                "knn_distance", "dataset_alpha"
            ).name
            receipt = root / "score_expectations_antigravity" / name
            shard_path = root / "score_shards_antigravity" / name
            before = receipt.stat()
            receipt_bytes = receipt.read_bytes()
            self.assertFalse(shard_path.exists())
            recomputed: list[tuple[str, str]] = []
            original_prepare = score_cache._prepare_task_antigravity

            def record_recompute(task):
                recomputed.append((task.method, task.dataset))
                return original_prepare(task)

            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_prepare_task_antigravity",
                record_recompute,
            ):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            after = receipt.stat()
            self.assertIn(("knn_distance", "dataset_alpha"), recomputed)
            self.assertEqual(receipt.read_bytes(), receipt_bytes)
            self.assertEqual(
                (before.st_dev, before.st_ino), (after.st_dev, after.st_ino)
            )
            self.assertEqual(
                score_cache.read_score_shard_antigravity(
                    root, "knn_distance", "dataset_alpha"
                ).status,
                "scored",
            )

    def test_swapped_and_foreign_receipts_fail_before_calibration_or_reuse(self) -> None:
        """Receipt identity, context, and source authority are immutable resume pins."""

        cases = ("identity", "cache_context", "source_authority")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                prefix="exp471_foreign_receipt_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "score_cache_antigravity"
                session = _session_antigravity(_source_bundle_antigravity)
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root:
                    score_cache.prepare_score_cache_antigravity(
                        session, workers=1
                    )
                _remove_complete_manifest_antigravity(root)
                alpha_name = score_cache.score_shard_relative_path_antigravity(
                    "knn_distance", "dataset_alpha"
                ).name
                receipt = root / "score_expectations_antigravity" / alpha_name
                if case == "identity":
                    beta_name = score_cache.score_shard_relative_path_antigravity(
                        "knn_distance", "dataset_beta"
                    ).name
                    foreign = root / "score_expectations_antigravity" / beta_name
                    receipt.chmod(0o600)
                    receipt.write_bytes(foreign.read_bytes())
                elif case == "cache_context":
                    _rewrite_receipt_antigravity(
                        receipt, cache_context_sha256="f" * 64
                    )
                else:
                    _rewrite_receipt_antigravity(
                        receipt, source_authority_sha256="e" * 64
                    )
                calibration_calls: list[object] = []
                issued, cache_root = self._issued_patches_antigravity(root)
                with issued, cache_root, mock.patch.object(
                    score_cache,
                    "upper_tail_pvalues_numba_antigravity",
                    side_effect=lambda *args: calibration_calls.append(args),
                ), self.assertRaisesRegex(ValueError, "expected scored envelope"):
                    score_cache.prepare_score_cache_antigravity(
                        session, workers=1
                    )
                self.assertEqual(calibration_calls, [])

    def test_malformed_abstention_is_terminal_before_calibration_or_workers(self) -> None:
        """A score-bearing abstention must fail parent validation permanently."""

        bundle = _mixed_source_bundle_antigravity()
        bundle["baselines"]["deep_svdd_1d"]["scores"]["dataset_alpha"][
            "test_scores"
        ] = [0.25, 0.75]
        self._assert_input_rejected_before_compute_antigravity(
            bundle, "abstained score-vector schema"
        )

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

    def test_last_shard_interruption_preserves_measured_and_actual_provenance(
        self,
    ) -> None:
        """Resume must reuse pre-shard decisions without invented measurements."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_last_shard_interrupt_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(_source_bundle_antigravity)
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "_monotonic_antigravity",
                side_effect=(10.0, 12.2, 20.0, 21.0),
            ), self.assertRaisesRegex(RuntimeError, "last shard interruption"):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=2,
                    progress=lambda event: (
                        (_ for _ in ()).throw(
                            RuntimeError("last shard interruption")
                        )
                        if event["status"] == "durable"
                        and event["durable_completed"] == 30
                        else None
                    ),
                )

            benchmark_path = root / "score_preparation_benchmark_antigravity.json"
            self.assertTrue(benchmark_path.is_file())
            execution_paths = tuple(
                sorted(root.glob("score_preparation_execution_event_*_antigravity.json"))
            )
            self.assertTrue(execution_paths)
            provenance_bytes = {
                path.name: path.read_bytes()
                for path in (benchmark_path, *execution_paths)
            }
            interrupted = score_cache.status_score_cache_antigravity(root)
            self.assertFalse(interrupted["complete"])
            self.assertEqual(interrupted["durable_completed"], 30)
            self.assertTrue(
                interrupted["benchmark_decision"]["parallel_accepted"]
            )
            self.assertEqual(
                interrupted["benchmark_decision"]["selected_workers"], 2
            )
            self.assertEqual(
                interrupted["execution_decision"]["selected_workers"], 2
            )

            source_calls: list[object] = []
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, mock.patch.object(
                score_cache,
                "benchmark_score_preparation_antigravity",
                side_effect=AssertionError("resume must not remeasure"),
            ), mock.patch.object(
                score_cache,
                "_prepare_score_identity_antigravity",
                side_effect=lambda *args: source_calls.append(args),
            ):
                manifest = score_cache.prepare_score_cache_antigravity(
                    session, workers=1
                )
            self.assertEqual(manifest.total_shards, 30)
            self.assertEqual(source_calls, [])
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in (benchmark_path, *execution_paths)
                },
                provenance_bytes,
            )
            resumed = score_cache.status_score_cache_antigravity(root)
            self.assertTrue(resumed["complete"])
            self.assertEqual(
                resumed["benchmark_decision"], interrupted["benchmark_decision"]
            )
            self.assertEqual(
                resumed["execution_decision"], interrupted["execution_decision"]
            )

    def test_partial_shards_without_measured_provenance_fail_closed(self) -> None:
        """Legacy shards must not cause fabricated zero-time benchmark evidence."""
        label_calls: list[object] = []
        with tempfile.TemporaryDirectory(
            prefix="exp471_missing_provenance_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "score_cache_antigravity"
            session = _session_antigravity(
                _source_bundle_antigravity,
                label_callback=lambda: label_calls.append(object()),
            )
            issued, cache_root = self._issued_patches_antigravity(root)
            with issued, cache_root, self.assertRaisesRegex(
                RuntimeError, "partial interruption"
            ):
                score_cache.prepare_score_cache_antigravity(
                    session,
                    workers=1,
                    progress=lambda event: (
                        (_ for _ in ()).throw(RuntimeError("partial interruption"))
                        if event["status"] == "durable"
                        and event["durable_completed"] == 1
                        else None
                    ),
                )
            benchmark_path = root / "score_preparation_benchmark_antigravity.json"
            if benchmark_path.exists():
                benchmark_path.chmod(0o600)
                benchmark_path.unlink()

            source_calls: list[object] = []
            calibration_calls: list[object] = []
            original_source = score_cache._prepare_score_identity_antigravity
            original_calibration = score_cache.upper_tail_pvalues_numba_antigravity

            def record_source(*args):
                source_calls.append(args)
                return original_source(*args)

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
            ), self.assertRaisesRegex(
                ValueError, "measured benchmark decision is missing"
            ):
                score_cache.prepare_score_cache_antigravity(session, workers=1)
            self.assertEqual(source_calls, [])
            self.assertEqual(calibration_calls, [])
            self.assertEqual(label_calls, [])
            with self.assertRaisesRegex(
                ValueError, "measured benchmark decision is missing"
            ):
                score_cache.status_score_cache_antigravity(root)

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

    def test_all_v3_arms_reject_bool_and_float_test_indices_before_compute(self) -> None:
        """JSON-number equality must not bypass exact integer TEST identity."""

        arms = (
            ("condition", "A", _source_bundle_antigravity),
            ("baseline_scored", "knn_distance", _source_bundle_antigravity),
            ("baseline_abstained", "deep_svdd_1d", _mixed_source_bundle_antigravity),
        )
        for arm, method, factory in arms:
            for invalid in ([False, 1], [0, 1.0]):
                with self.subTest(arm=arm, invalid=invalid):
                    bundle = factory()
                    namespace = "conditions" if arm == "condition" else "baselines"
                    bundle[namespace][method]["scores"]["dataset_alpha"][
                        "test_indices"
                    ] = invalid
                    self._assert_input_rejected_before_compute_antigravity(
                        bundle, "TEST coordinates"
                    )

    def test_all_v3_arms_reject_negative_and_duplicate_test_indices(self) -> None:
        """Every typed arm must reject non-authoritative integer coordinates."""

        arms = (
            ("condition", "A", _source_bundle_antigravity),
            ("baseline_scored", "knn_distance", _source_bundle_antigravity),
            ("baseline_abstained", "deep_svdd_1d", _mixed_source_bundle_antigravity),
        )
        for arm, method, factory in arms:
            for invalid in ([-1, 1], [0, 0]):
                with self.subTest(arm=arm, invalid=invalid):
                    bundle = factory()
                    namespace = "conditions" if arm == "condition" else "baselines"
                    bundle[namespace][method]["scores"]["dataset_alpha"][
                        "test_indices"
                    ] = invalid
                    self._assert_input_rejected_before_compute_antigravity(
                        bundle, "TEST coordinates"
                    )

    def test_self_consistent_wrong_baseline_and_global_coordinates_fail_pins(self) -> None:
        """Rehashed payload-local authority must not displace independent pins."""

        wrong_baseline = _source_bundle_antigravity()
        baseline_authority = wrong_baseline["baselines"]["knn_distance"][
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
        wrong_baseline["baseline_authority_sha256"][
            "knn_distance"
        ] = changed_digest
        self._assert_input_rejected_before_compute_antigravity(
            wrong_baseline, "pinned baseline authority"
        )

        wrong_coordinates = _source_bundle_antigravity()
        coordinates = wrong_coordinates["test_coordinates"]
        coordinates["dataset_alpha"]["indices"] = [1, 0]
        coordinates["dataset_alpha"]["coordinates_sha256"] = sha256(
            canonical_json_bytes_exp471_antigravity(
                {
                    "dataset": "dataset_alpha",
                    "count": 2,
                    "indices": [1, 0],
                }
            )
        ).hexdigest()
        for method in METHOD_ORDER_ANTIGRAVITY[:4]:
            wrong_coordinates["conditions"][method]["scores"][
                "dataset_alpha"
            ]["test_indices"] = [1, 0]
        changed_coordinates_sha256 = _coordinate_inventory_sha256_antigravity(
            coordinates
        )
        for method in METHOD_ORDER_ANTIGRAVITY[4:]:
            source = wrong_coordinates["baselines"][method]
            source["scores"]["dataset_alpha"]["test_indices"] = [1, 0]
            source["authority"][
                "test_coordinates_sha256"
            ] = changed_coordinates_sha256
            bare = {
                key: value
                for key, value in source["authority"].items()
                if key != "authority_sha256"
            }
            changed_digest = sha256(
                canonical_json_bytes_exp471_antigravity(bare)
            ).hexdigest()
            source["authority"]["authority_sha256"] = changed_digest
            wrong_coordinates["baseline_authority_sha256"][method] = changed_digest
        self._assert_input_rejected_before_compute_antigravity(
            wrong_coordinates, "pinned TEST coordinate inventory"
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

    def test_one_and_policy_worker_mixed_requests_produce_identical_complete_bytes(self) -> None:
        """Policy-worker mixed shards must match canonical one-worker bytes."""
        with tempfile.TemporaryDirectory(prefix="exp471_parity_antigravity_") as temporary:
            base = Path(temporary).resolve()
            roots = (
                base / "one_antigravity" / "score_cache_antigravity",
                base / "policy_antigravity" / "score_cache_antigravity",
            )
            session = _session_antigravity(_mixed_source_bundle_antigravity)
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
                score_cache.prepare_score_cache_antigravity(
                    session, workers=score_cache.CPU_WORKERS_ANTIGRAVITY
                )
            self.assertGreaterEqual(
                worker_counts.count(score_cache.CPU_WORKERS_ANTIGRAVITY), 2
            )
            self.assertEqual(
                _cache_bytes_antigravity(roots[0]),
                _cache_bytes_antigravity(roots[1]),
            )
            status = score_cache.status_score_cache_antigravity(roots[1])
            decision = status["benchmark_decision"]
            self.assertEqual(
                decision["selected_workers"], score_cache.CPU_WORKERS_ANTIGRAVITY
            )
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
        """A self-consistent unauthorized baseline must not reopen a cache."""
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
            with issued, cache_root, self.assertRaisesRegex(
                ValueError, "pinned baseline authority"
            ):
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
            status = score_cache.status_score_cache_antigravity(root)
            measured = status["benchmark_decision"]
            self.assertEqual(measured["selected_workers"], 2)
            self.assertTrue(measured["parallel_accepted"])
            self.assertIsNone(measured["parallel_error_type"])
            execution = status["execution_decision"]
            self.assertEqual(execution["selected_workers"], 1)
            self.assertTrue(execution["fallback_occurred"])
            fallback = execution["events"][-1]
            self.assertEqual(fallback["wave"], "first")
            self.assertEqual(fallback["error_type"], "RuntimeError")
            self.assertIn("synthetic actual pool failure", fallback["error_message"])

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
            status = score_cache.status_score_cache_antigravity(root)
            self.assertEqual(status["benchmark_decision"]["selected_workers"], 2)
            self.assertTrue(status["benchmark_decision"]["parallel_accepted"])
            execution = status["execution_decision"]
            self.assertEqual(execution["selected_workers"], 1)
            fallback = execution["events"][-1]
            self.assertEqual(fallback["wave"], "retry")
            self.assertIn("synthetic retry pool failure", fallback["error_message"])

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
