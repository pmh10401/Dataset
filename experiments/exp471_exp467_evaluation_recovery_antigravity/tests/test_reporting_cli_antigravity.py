"""Public-literal lifecycle tests for Exp471 reporting and its closed CLI."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import threading
import unittest
import warnings
from unittest import mock

from scratch import (
    run_exp471_exp467_evaluation_recovery_antigravity as exp471_launcher,
)

from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    cli_antigravity as cli,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    evaluation_antigravity as evaluation,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    baseline_preflight_antigravity as baseline_preflight,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    reporting_antigravity as reporting,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.cli_antigravity import (
    main_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    ACCELERATION_ASSESSMENT_ANTIGRAVITY,
    CPU_WORKERS_ANTIGRAVITY,
    INNER_NUMERICAL_THREADS_ANTIGRAVITY,
    LOGICAL_CPU_COUNT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    canonical_json_bytes_exp471_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_evaluation_antigravity import (
    _LABELS_ANTIGRAVITY,
    _authority_antigravity,
    _evaluation_patches_antigravity,
    _publish_cache_antigravity,
    _session_antigravity,
)


def _worker_provenance_antigravity(
    requested_workers: int = 1, *, single_worker_seconds: float = 0.0
) -> dict[str, object]:
    cache_context = "66854b3754856512a8a99e47582dc2a18000a29f403e1964ab669a87804605fb"
    fallback = (
        "parallel worker request is one; canonical CPU single-worker retained"
        if requested_workers == 1
        else "parallel speedup is below the required 1.1x gate"
    )
    benchmark_bare = {
        "schema": "exp471-score-preparation-benchmark-antigravity.v1",
        "cache_context_sha256": cache_context,
        "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
        "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
        "requested_parallel_workers": requested_workers,
        "selected_workers": 1,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        "single_worker_seconds": single_worker_seconds,
        "parallel_seconds": 0.0,
        "speedup_threshold": 1.1,
        "speedup": 1.0,
        "byte_parity": True,
        "parallel_accepted": False,
        "parallel_error_type": None,
        "parallel_error_message": None,
        "fallback_reason": fallback,
        "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
    }
    benchmark = {
        **benchmark_bare,
        "decision_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(benchmark_bare)
        ).hexdigest(),
    }
    event_bare = {
        "schema": "exp471-score-preparation-execution-event-antigravity.v1",
        "cache_context_sha256": cache_context,
        "benchmark_decision_sha256": benchmark["decision_sha256"],
        "sequence": 0,
        "prior_execution_sha256": None,
        "event_type": "selection",
        "wave": "initial",
        "selected_workers": 1,
        "error_type": None,
        "error_message": None,
        "fallback_reason": "",
    }
    event = {
        **event_bare,
        "execution_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(event_bare)
        ).hexdigest(),
    }
    return {
        "schema": "exp471-worker-provenance-antigravity.v1",
        "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
        "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
        "benchmark_decision": benchmark,
        "execution_decision": {
            "schema": "exp471-score-preparation-execution-antigravity.v1",
            "cache_context_sha256": cache_context,
            "benchmark_decision_sha256": benchmark["decision_sha256"],
            "selected_workers": 1,
            "fallback_occurred": False,
            "events": [event],
        },
    }


def _artifact_antigravity(base: Path):
    cache_root = base / "score_cache_antigravity"
    evaluation_root = base / "evaluation_antigravity"
    authority = _authority_antigravity()
    session = _session_antigravity(authority)
    cache = _publish_cache_antigravity(cache_root, authority)
    worker = _worker_provenance_antigravity()
    (cache_root / "score_preparation_benchmark_antigravity.json").write_bytes(
        canonical_json_bytes_exp471_antigravity(worker["benchmark_decision"])
    )
    for event in worker["execution_decision"]["events"]:
        sequence = event["sequence"]
        (
            cache_root
            / f"score_preparation_execution_event_{sequence:06d}_antigravity.json"
        ).write_bytes(canonical_json_bytes_exp471_antigravity(event))
    with ExitStack() as stack:
        for patcher in _evaluation_patches_antigravity(
            cache_root, evaluation_root
        ):
            stack.enter_context(patcher)
        artifact = evaluation.evaluate_exp471_antigravity(
            session,
            cache,
            label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
        )
    return artifact


def _rehash_report_leaf_antigravity(
    root: Path, relative: str, payload: bytes
) -> None:
    """Rehash one public leaf through both outer authority layers."""

    (root / relative).write_bytes(payload)
    artifact_map_path = root / reporting._ARTIFACT_MAP_NAME_ANTIGRAVITY
    artifact_map = json.loads(artifact_map_path.read_text(encoding="utf-8"))
    artifact_map["files"][relative] = sha256(payload).hexdigest()
    artifact_map_raw = canonical_json_bytes_exp471_antigravity(artifact_map)
    artifact_map_path.write_bytes(artifact_map_raw)
    marker_path = root / reporting._COMPLETE_MARKER_NAME_ANTIGRAVITY
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["report_file_sha256"] = artifact_map["files"]
    marker["report_manifest_sha256"] = sha256(artifact_map_raw).hexdigest()
    marker_path.write_bytes(canonical_json_bytes_exp471_antigravity(marker))


def _independent_report_contender_antigravity(
    artifact,
    report_root: str,
    task5_root: str,
    worker: Mapping[str, object],
    result_queue,
    ready,
    release,
    hold_lock: bool,
) -> None:
    """Run one literal-fixture publisher in a distinct OS process."""

    reporting._TASK5_SCORE_CACHE_ROOT_ANTIGRAVITY = Path(task5_root)

    def renderer(frozen, root):
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("public process release timeout antigravity")
        return reporting.write_exp450_report_antigravity(frozen, root)

    try:
        authority = reporting.write_exp471_report_antigravity(
            artifact,
            Path(report_root),
            worker_provenance=worker,
            renderer=renderer if hold_lock else None,
        )
    except Exception as error:
        result_queue.put(("error", type(error).__name__, str(error)))
    else:
        result_queue.put(("success", authority.report_manifest_sha256))


def _preflight_record_antigravity(
    count: int = 2,
    *,
    status: str = "complete",
    train_length: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "count": count,
        "indices": list(range(count)),
        "status": status,
    }
    if status == "complete":
        train_count = count if train_length is None else train_length
        record["n_train"] = train_count
        record["n_test"] = count
        record["train_oof_scores"] = [float(index) + 0.25 for index in range(train_count)]
        record["test_scores"] = [float(index) + 0.75 for index in range(count)]
    elif status == "abstain":
        record["reason"] = "insufficient_train_rows"
        record["n_train"] = 0 if train_length is None else train_length
        record["n_test"] = count
        record["train_oof_scores"] = []
        record["test_scores"] = []
    return record


def _production_preflight_coordinates_antigravity() -> dict[str, object]:
    """Return the six-method baseline inventory with 6,307 scored entries."""

    abstention_plan = {
        "knn_distance": 5,
        "deep_svdd_1d": 195,
        "drocc_1d": 195,
    }
    coordinates: dict[str, object] = {}
    for method in (
        "knn_distance",
        "lof_novelty",
        "ocsvm_rbf",
        "isolation_forest",
        "deep_svdd_1d",
        "drocc_1d",
    ):
        rows: dict[str, object] = {}
        for index in range(1_117):
            dataset = f"production_dataset_{index:04d}_antigravity"
            limit = abstention_plan.get(method, 0)
            status = "abstain" if index < limit else "complete"
            rows[dataset] = _preflight_record_antigravity(status=status)
        coordinates[method] = rows
    return coordinates


def _preflight_expected_coordinates_antigravity(
    datasets, *, count: int = 2
) -> dict[str, tuple[int, ...]]:
    return {dataset: tuple(range(count)) for dataset in datasets}


def _issued_preflight_context_antigravity(session: object) -> object:
    """Return the already-successful preflight boundary for loader-only tests."""

    return type(
        "IssuedPreflightContextAntigravity",
        (),
        {
            "datasets": tuple(
                entry["dataset"] for entry in session.catalog["datasets"]
            )
        },
    )()


class BaselinePreflightTestsAntigravity(unittest.TestCase):
    """Strict typed-abstain classifier contracts before proposal preprocessing."""

    BASELINE_METHODS_ANTIGRAVITY = (
        "knn_distance",
        "lof_novelty",
        "ocsvm_rbf",
        "isolation_forest",
        "deep_svdd_1d",
        "drocc_1d",
    )
    DATASET_ANTIGRAVITY = "public_literal_dataset"

    def _valid_fixture(
        self,
        datasets: tuple[str, ...] = (DATASET_ANTIGRAVITY,),
        *,
        n_train: int = 3,
        n_test: int = 2,
    ) -> tuple[
        tuple[str, ...],
        dict[str, object],
        dict[str, tuple[int, ...]],
    ]:
        coordinates: dict[str, object] = {}
        for method in self.BASELINE_METHODS_ANTIGRAVITY:
            coordinates[method] = {
                dataset: _preflight_record_antigravity(
                    count=n_test, train_length=n_train
                )
                for dataset in datasets
            }
        expected = _preflight_expected_coordinates_antigravity(
            datasets, count=n_test
        )
        return datasets, coordinates, expected

    def _run_preflight(self, datasets, coordinates, expected_coordinates):
        return baseline_preflight.preflight_frozen_baselines_antigravity(
            datasets=datasets,
            coordinates=coordinates,
            expected_coordinates=expected_coordinates,
        )

    def test_full_six_scored_inventory_preserves_train_and_test_lengths(self) -> None:
        datasets, coordinates, expected = self._valid_fixture(
            n_train=3, n_test=2
        )
        result = self._run_preflight(datasets, coordinates, expected)
        self.assertEqual(
            tuple(method for method, _dataset in result.entries),
            self.BASELINE_METHODS_ANTIGRAVITY,
        )
        for method in self.BASELINE_METHODS_ANTIGRAVITY:
            entry = result.entries[(method, self.DATASET_ANTIGRAVITY)]
            self.assertEqual(entry.method, method)
            self.assertEqual(entry.dataset, self.DATASET_ANTIGRAVITY)
            self.assertEqual(entry.status, "scored")
            self.assertIsNone(entry.abstention_reason)
            self.assertEqual(entry.test_indices, (0, 1))
            self.assertEqual(entry.train_oof_scores, (0.25, 1.25, 2.25))
            self.assertEqual(entry.test_scores, (0.75, 1.75))
        self.assertEqual(result.scored_count, 6)
        self.assertEqual(result.abstained_count, 0)
        self.assertEqual(result.failed_count, 0)

    def test_legal_abstain_is_typed_and_has_no_score_vectors(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["deep_svdd_1d"][self.DATASET_ANTIGRAVITY] = (
            _preflight_record_antigravity(
                status="abstain", train_length=3
            )
        )
        result = self._run_preflight(datasets, coordinates, expected)
        entry = result.entries[("deep_svdd_1d", self.DATASET_ANTIGRAVITY)]
        self.assertEqual(entry.status, "abstained")
        self.assertEqual(entry.abstention_reason, "insufficient_train_rows")
        self.assertEqual(entry.train_oof_scores, ())
        self.assertEqual(entry.test_scores, ())
        self.assertEqual(result.scored_count, 5)
        self.assertEqual(result.abstained_count, 1)
        self.assertEqual(result.failed_count, 0)

    def test_failure_status_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["lof_novelty"][self.DATASET_ANTIGRAVITY]["status"] = (
            "failure"
        )
        with self.assertRaisesRegex(
            ValueError, "source status for lof_novelty/public_literal_dataset"
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_missing_dataset_row_fails_coverage_branch(self) -> None:
        datasets, coordinates, expected = self._valid_fixture(
            (self.DATASET_ANTIGRAVITY, "second_public_literal_dataset")
        )
        del coordinates["ocsvm_rbf"]["second_public_literal_dataset"]
        with self.assertRaisesRegex(ValueError, "coverage for ocsvm_rbf"):
            self._run_preflight(datasets, coordinates, expected)

    def test_extra_dataset_row_fails_coverage_branch(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["ocsvm_rbf"]["extra_public_literal_dataset"] = (
            _preflight_record_antigravity(train_length=3)
        )
        with self.assertRaisesRegex(ValueError, "coverage for ocsvm_rbf"):
            self._run_preflight(datasets, coordinates, expected)

    def test_unknown_abstention_reason_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        record = _preflight_record_antigravity(
            status="abstain", train_length=3
        )
        record["reason"] = "worker_crashed"
        coordinates["isolation_forest"][self.DATASET_ANTIGRAVITY] = record
        with self.assertRaisesRegex(
            ValueError,
            "abstention reason for isolation_forest/public_literal_dataset",
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_nonempty_abstain_list_fails_closed(self) -> None:
        for field in ("train_oof_scores", "test_scores"):
            with self.subTest(field=field):
                datasets, coordinates, expected = self._valid_fixture()
                record = _preflight_record_antigravity(
                    status="abstain", train_length=3
                )
                record[field] = [0.5]
                coordinates["deep_svdd_1d"][self.DATASET_ANTIGRAVITY] = record
                with self.assertRaisesRegex(
                    ValueError,
                    "nonempty abstention scores for deep_svdd_1d/public_literal_dataset",
                ):
                    self._run_preflight(datasets, coordinates, expected)

    def test_missing_abstain_array_fails_closed(self) -> None:
        for field in ("train_oof_scores", "test_scores"):
            with self.subTest(field=field):
                datasets, coordinates, expected = self._valid_fixture()
                record = _preflight_record_antigravity(
                    status="abstain", train_length=3
                )
                del record[field]
                coordinates["deep_svdd_1d"][self.DATASET_ANTIGRAVITY] = record
                with self.assertRaisesRegex(
                    ValueError,
                    "missing explicit abstention arrays for deep_svdd_1d/public_literal_dataset",
                ):
                    self._run_preflight(datasets, coordinates, expected)

    def test_abstain_arrays_reject_non_list_representations(self) -> None:
        invalid_values = (None, "", b"", {}, ())
        for field in ("train_oof_scores", "test_scores"):
            for value in invalid_values:
                with self.subTest(field=field, value_type=type(value).__name__):
                    datasets, coordinates, expected = self._valid_fixture()
                    record = _preflight_record_antigravity(
                        status="abstain", train_length=3
                    )
                    record[field] = value
                    coordinates["knn_distance"][self.DATASET_ANTIGRAVITY] = record
                    with self.assertRaisesRegex(
                        ValueError,
                        "abstention score arrays for knn_distance/public_literal_dataset",
                    ):
                        self._run_preflight(datasets, coordinates, expected)

    def test_nonfinite_scored_vector_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["knn_distance"][self.DATASET_ANTIGRAVITY][
            "test_scores"
        ] = [0.75, float("inf")]
        with self.assertRaisesRegex(ValueError, "baseline score vector"):
            self._run_preflight(datasets, coordinates, expected)

    def test_shard_indices_reject_bool_and_float_values(self) -> None:
        for invalid_index in (True, 1.0):
            with self.subTest(index=invalid_index):
                datasets, coordinates, expected = self._valid_fixture()
                coordinates["drocc_1d"][self.DATASET_ANTIGRAVITY][
                    "indices"
                ] = [0, invalid_index]
                with self.assertRaisesRegex(
                    ValueError,
                    "source TEST coordinates for drocc_1d/public_literal_dataset",
                ):
                    self._run_preflight(datasets, coordinates, expected)

    def test_expected_indices_reject_bool_and_float_values(self) -> None:
        for invalid_index in (True, 1.0):
            with self.subTest(index=invalid_index):
                datasets, coordinates, expected = self._valid_fixture()
                expected[self.DATASET_ANTIGRAVITY] = (0, invalid_index)
                with self.assertRaisesRegex(
                    ValueError, "independent TEST coordinate authority"
                ):
                    self._run_preflight(datasets, coordinates, expected)

    def test_shard_indices_must_match_independent_expected_indices(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["drocc_1d"][self.DATASET_ANTIGRAVITY]["indices"] = [1, 0]
        with self.assertRaisesRegex(
            ValueError,
            "source TEST coordinates for drocc_1d/public_literal_dataset",
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_expected_coordinate_map_missing_dataset_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture(
            (self.DATASET_ANTIGRAVITY, "second_public_literal_dataset")
        )
        del expected["second_public_literal_dataset"]
        with self.assertRaisesRegex(
            ValueError, "independent TEST coordinate order"
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_expected_coordinate_map_extra_dataset_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        expected["extra_public_literal_dataset"] = (0, 1)
        with self.assertRaisesRegex(
            ValueError, "independent TEST coordinate order"
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_expected_coordinate_map_reordered_dataset_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture(
            (self.DATASET_ANTIGRAVITY, "second_public_literal_dataset")
        )
        reordered = {
            dataset: expected[dataset]
            for dataset in reversed(tuple(expected))
        }
        with self.assertRaisesRegex(
            ValueError, "independent TEST coordinate order"
        ):
            self._run_preflight(datasets, coordinates, reordered)

    def test_test_score_length_must_equal_n_test(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["knn_distance"][self.DATASET_ANTIGRAVITY][
            "test_scores"
        ] = [0.75]
        with self.assertRaisesRegex(
            ValueError,
            "score vector lengths for knn_distance/public_literal_dataset",
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_train_score_length_must_equal_n_train(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["knn_distance"][self.DATASET_ANTIGRAVITY][
            "train_oof_scores"
        ] = [0.25, 1.25]
        with self.assertRaisesRegex(
            ValueError,
            "score vector lengths for knn_distance/public_literal_dataset",
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_n_test_and_count_must_match_expected_coordinate_length(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        record = coordinates["knn_distance"][self.DATASET_ANTIGRAVITY]
        record["count"] = 3
        record["n_test"] = 3
        record["test_scores"] = [0.75, 1.75, 2.75]
        with self.assertRaisesRegex(
            ValueError,
            "source row counts for knn_distance/public_literal_dataset",
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_missing_n_test_fails_source_row_count_branch(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        del coordinates["knn_distance"][self.DATASET_ANTIGRAVITY]["n_test"]
        with self.assertRaisesRegex(
            ValueError,
            "source row counts for knn_distance/public_literal_dataset",
        ):
            self._run_preflight(datasets, coordinates, expected)

    def test_duplicate_dataset_in_datasets_tuple_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        duplicated = (datasets[0], datasets[0])
        with self.assertRaisesRegex(ValueError, "dataset order"):
            self._run_preflight(duplicated, coordinates, expected)

    def test_production_shape_reports_exact_counts(self) -> None:
        datasets = tuple(
            f"production_dataset_{index:04d}_antigravity"
            for index in range(1_117)
        )
        result = self._run_preflight(
            datasets,
            _production_preflight_coordinates_antigravity(),
            _preflight_expected_coordinates_antigravity(datasets),
        )
        self.assertEqual(len(result.entries), 6_702)
        self.assertEqual(result.scored_count, 6_307)
        self.assertEqual(result.abstained_count, 395)
        self.assertEqual(result.failed_count, 0)

    def test_missing_baseline_method_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        del coordinates["lof_novelty"]
        with self.assertRaisesRegex(ValueError, "method order"):
            self._run_preflight(datasets, coordinates, expected)

    def test_reordered_baseline_methods_fail_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        reordered = {
            method: coordinates[method]
            for method in reversed(tuple(coordinates))
        }
        with self.assertRaisesRegex(ValueError, "method order"):
            self._run_preflight(datasets, reordered, expected)

    def test_proposal_method_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["aspe_proposal"] = coordinates.pop("drocc_1d")
        with self.assertRaisesRegex(ValueError, "method order"):
            self._run_preflight(datasets, coordinates, expected)

    def test_extra_method_fails_closed(self) -> None:
        datasets, coordinates, expected = self._valid_fixture()
        coordinates["unexpected_baseline"] = {
            self.DATASET_ANTIGRAVITY: _preflight_record_antigravity(
                train_length=3
            )
        }
        with self.assertRaisesRegex(ValueError, "method order"):
            self._run_preflight(datasets, coordinates, expected)


class ProductionScoreBundleTestsAntigravity(unittest.TestCase):
    """Production preflight ordering and verified score-input v3 contracts."""

    DATASETS_ANTIGRAVITY = tuple(
        f"production_dataset_{index:04d}_antigravity" for index in range(1_117)
    )

    @staticmethod
    def _test_identity_antigravity(
        dataset: str, dataset_id: int, *, count: int = 2
    ) -> dict[str, object]:
        indices = list(range(count))
        bare = {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "series_length": 3,
            "split": "TEST",
            "indices": indices,
            "values_blob_sha256": [
                sha256(f"{dataset}:{index}".encode("utf-8")).hexdigest()
                for index in indices
            ],
        }
        return {
            **bare,
            "row_digest": sha256(
                canonical_json_bytes_exp471_antigravity(bare)
            ).hexdigest(),
            "count": count,
        }

    def _condition_manifest_antigravity(
        self, *, drift: bool = False
    ) -> dict[str, object]:
        identities = [
            self._test_identity_antigravity(dataset, index)
            for index, dataset in enumerate(self.DATASETS_ANTIGRAVITY)
        ]
        if drift:
            identities[-1] = self._test_identity_antigravity(
                self.DATASETS_ANTIGRAVITY[-1], len(identities) - 1, count=3
            )
        return {
            "datasets": [
                {"dataset": dataset, "test": identity}
                for dataset, identity in zip(
                    self.DATASETS_ANTIGRAVITY, identities, strict=True
                )
            ],
            "test_identities": identities,
            "score_rows": {
                dataset: [0.125, 0.875]
                for dataset in self.DATASETS_ANTIGRAVITY
            },
        }

    def _session_antigravity(
        self, calls: list[str], *, condition_drift: bool = False
    ) -> object:
        authority = type(
            "AuthorityAntigravity",
            (),
            {
                "database_sha256": "5" * 64,
                "condition_manifest_sha256": {
                    condition: condition.lower() * 64
                    for condition in METHOD_ORDER_ANTIGRAVITY[:4]
                },
            },
        )()
        manifests = {
            condition: self._condition_manifest_antigravity(
                drift=condition_drift and condition == "D"
            )
            for condition in METHOD_ORDER_ANTIGRAVITY[:4]
        }

        def build_exp467_score_manifests_antigravity(*_args, **_kwargs):
            calls.append("build")
            return type(
                "ScoreAuthorityAntigravity", (), {"manifests": manifests}
            )()

        verifier = type(
            "VerifierAntigravity",
            (),
            {
                "build_exp467_score_manifests_antigravity": staticmethod(
                    build_exp467_score_manifests_antigravity
                )
            },
        )()
        session = type("SessionAntigravity", (), {})()
        session.authority = authority
        session.verification = object()
        session.catalog = {
            "database": {},
            "datasets": [
                {"dataset": dataset} for dataset in self.DATASETS_ANTIGRAVITY
            ],
        }
        session.context = type(
            "ContextAntigravity",
            (),
            {"run_root": Path("/tmp/exp471_synthetic_run_antigravity")},
        )()
        session.migration_manifest = object()
        session.verifier = verifier
        return session

    def _baseline_patches_antigravity(
        self,
        temporary: str,
        *,
        calls: list[str],
        root_error: str | None = None,
        mutation: tuple[str, object] | None = None,
    ):
        from experiments.exp450_resampling_image_ablation_antigravity import (
            evaluation_antigravity as frozen_evaluation,
        )

        root_methods = {
            "classical": baseline_preflight.BASELINE_METHODS_ANTIGRAVITY[:4],
            "deep_svdd": ("deep_svdd_1d",),
            "drocc": ("drocc_1d",),
        }
        abstention_plan = {
            "knn_distance": 5,
            "deep_svdd_1d": 195,
            "drocc_1d": 195,
        }
        manifests: dict[str, object] = {}
        roots: dict[str, Path] = {}
        root_authority: dict[str, dict[str, str]] = {}
        identities = [
            self._test_identity_antigravity(dataset, index)
            for index, dataset in enumerate(self.DATASETS_ANTIGRAVITY)
        ]

        class SyntheticManifestAntigravity(dict):
            pass

        for root_index, (root_name, root_method_order) in enumerate(
            root_methods.items()
        ):
            root = Path(temporary) / f"{root_name}_root_antigravity"
            root.mkdir(parents=True)
            manifest_path = root / "frozen_score_manifest.json"
            manifest_path.write_text(
                json.dumps({"root": root_name}, sort_keys=True), encoding="utf-8"
            )
            run_config_sha256 = str(root_index + 1) * 64
            methods: dict[str, dict[str, dict[str, object]]] = {}
            for method in root_method_order:
                rows: dict[str, dict[str, object]] = {}
                abstention_count = abstention_plan.get(method, 0)
                for dataset_index, dataset in enumerate(self.DATASETS_ANTIGRAVITY):
                    status = (
                        "abstain"
                        if dataset_index < abstention_count
                        else "complete"
                    )
                    shard = _preflight_record_antigravity(status=status)
                    shard["test_indices"] = shard.pop("indices")
                    shard.pop("count")
                    rows[dataset] = shard
                methods[method] = rows
            manifest = SyntheticManifestAntigravity(
                database_sha256="5" * 64,
                methods=methods,
                datasets=[
                    {"dataset": dataset, "test": identity}
                    for dataset, identity in zip(
                        self.DATASETS_ANTIGRAVITY, identities, strict=True
                    )
                ],
                test_identities=identities,
            )
            manifest.run_root = root
            manifest.run_config_sha256 = run_config_sha256
            manifests[root_name] = manifest
            roots[root_name] = root
            root_authority[root_name] = {
                "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                "run_config_sha256": run_config_sha256,
            }

        if mutation is not None:
            mutation_name, change = mutation
            if mutation_name == "missing":
                del manifests["classical"]["methods"]["knn_distance"][
                    self.DATASETS_ANTIGRAVITY[5]
                ]
            elif mutation_name == "reordered_methods":
                methods = manifests["classical"]["methods"]
                manifests["classical"]["methods"] = {
                    method: methods[method] for method in reversed(tuple(methods))
                }
            elif mutation_name == "reordered_rows":
                rows = manifests["classical"]["methods"]["knn_distance"]
                manifests["classical"]["methods"]["knn_distance"] = {
                    dataset: rows[dataset] for dataset in reversed(tuple(rows))
                }
            else:
                shard = manifests["classical"]["methods"]["knn_distance"][
                    self.DATASETS_ANTIGRAVITY[5]
                ]
                change(shard)

        def verify_roots(observed_roots, _verifier):
            calls.append("verify_roots")
            self.assertEqual(dict(observed_roots), roots)
            if root_error is not None:
                raise ValueError(root_error)
            return manifests

        original_identity = frozen_evaluation._baseline_identity_and_shards

        def baseline_identity(manifest):
            calls.append("baseline_identity")
            return original_identity(manifest)

        original_preflight = (
            baseline_preflight.preflight_frozen_baselines_antigravity
        )

        def preflight(**kwargs):
            result = original_preflight(**kwargs)
            calls.append("preflight")
            return result

        coordinate_inventory = {
            "schema": "exp471-test-coordinate-inventory-antigravity.v1",
            "dataset_count": len(self.DATASETS_ANTIGRAVITY),
            "datasets": [
                {
                    "dataset": dataset,
                    "test_count": 2,
                    "test_indices": [0, 1],
                }
                for dataset in self.DATASETS_ANTIGRAVITY
            ],
        }
        coordinate_sha256 = sha256(
            canonical_json_bytes_exp471_antigravity(coordinate_inventory)
        ).hexdigest()
        return (
            mock.patch.object(
                frozen_evaluation,
                "BASELINE_ROOTS",
                roots,
            ),
            mock.patch.object(
                frozen_evaluation,
                "_verify_baseline_roots",
                side_effect=verify_roots,
            ),
            mock.patch.object(
                frozen_evaluation,
                "_baseline_identity_and_shards",
                side_effect=baseline_identity,
            ),
            mock.patch.object(
                cli,
                "EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY",
                root_authority,
            ),
            mock.patch.object(
                cli,
                "EXP471_BASELINE_DATABASE_SHA256_ANTIGRAVITY",
                "5" * 64,
                create=True,
            ),
            mock.patch.object(
                cli,
                "EXP471_BASELINE_DATASET_COUNT_ANTIGRAVITY",
                1_117,
                create=True,
            ),
            mock.patch.object(
                cli,
                "EXP471_TEST_COORDINATE_INVENTORY_SHA256_ANTIGRAVITY",
                coordinate_sha256,
                create=True,
            ),
            mock.patch.object(
                baseline_preflight,
                "preflight_frozen_baselines_antigravity",
                side_effect=preflight,
            ),
        )

    def _source_patches_antigravity(self, calls: list[str]):
        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            source_authority_antigravity as source_authority,
        )

        values_api = type(
            "ValuesApiAntigravity",
            (),
            {
                "database_config_antigravity": type(
                    "ConfigAntigravity",
                    (),
                    {
                        "database_path": "/tmp/public_database_antigravity.sqlite",
                        "database_sha256": "5" * 64,
                    },
                )()
            },
        )()

        def require(_session):
            calls.append("require")
            return values_api

        def bind(*_args, **_kwargs):
            calls.append("bind")
            return object()

        def load(*_args, **_kwargs):
            if not calls or calls[-1] != "load":
                calls.append("load")
            return {}

        return (
            mock.patch.object(
                source_authority,
                "_require_captured_values_api_antigravity",
                side_effect=require,
            ),
            mock.patch.object(
                source_authority,
                "_bind_captured_values_database_antigravity",
                side_effect=bind,
            ),
            mock.patch.object(
                source_authority,
                "_load_captured_dataset_values_antigravity",
                side_effect=load,
            ),
        )

    def _production_bundle_antigravity(
        self,
        *,
        calls: list[str],
        condition_drift: bool = False,
        mutation: tuple[str, object] | None = None,
    ):
        session = self._session_antigravity(
            calls, condition_drift=condition_drift
        )
        with tempfile.TemporaryDirectory(
            prefix="exp471_production_preflight_synthetic_antigravity_"
        ) as temporary:
            with ExitStack() as stack:
                for patcher in self._baseline_patches_antigravity(
                    temporary, calls=calls, mutation=mutation
                ):
                    stack.enter_context(patcher)
                for patcher in self._source_patches_antigravity(calls):
                    stack.enter_context(patcher)
                return cli._production_score_bundle_antigravity(session)

    def test_preflight_precedes_captured_loading_and_composition(self) -> None:
        """Verified preflight completes before raw loading and A-D composition."""

        calls: list[str] = []
        self._production_bundle_antigravity(calls=calls)
        self.assertEqual(calls[0], "verify_roots")
        self.assertEqual(calls.count("baseline_identity"), 3)
        self.assertLess(calls.index("preflight"), calls.index("require"))
        self.assertLess(calls.index("require"), calls.index("load"))
        self.assertLess(calls.index("load"), calls.index("build"))

    def test_preflight_context_coordinates_are_deeply_immutable(self) -> None:
        """Later proposal work cannot mutate the verified TEST authority."""

        calls: list[str] = []
        session = self._session_antigravity(calls)
        with tempfile.TemporaryDirectory(
            prefix="exp471_immutable_preflight_context_antigravity_"
        ) as temporary, ExitStack() as stack:
            for patcher in self._baseline_patches_antigravity(
                temporary, calls=calls
            ):
                stack.enter_context(patcher)
            context = cli._preflight_production_baselines_antigravity(
                session.authority, session.catalog
            )
        indices = context.test_coordinates[self.DATASETS_ANTIGRAVITY[0]][
            "indices"
        ]
        with self.assertRaises(TypeError):
            indices[0] = 1

    def test_invalid_root_stops_before_captured_loading_and_composition(self) -> None:
        """A rejected pinned root cannot resolve the captured-values capability."""

        calls: list[str] = []
        session = self._session_antigravity(calls)
        with tempfile.TemporaryDirectory(
            prefix="exp471_production_preflight_order_antigravity_"
        ) as temporary:
            patches = self._baseline_patches_antigravity(
                temporary, calls=calls,
                root_error="pinned baseline root authority differs",
            )
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                with self.assertRaisesRegex(
                    ValueError, "pinned baseline root authority differs"
                ):
                    cli._production_score_bundle_antigravity(session)
        self.assertEqual(calls, ["verify_roots"])

    def test_legal_source_abstention_becomes_v3_arm(self) -> None:
        """One typed abstained arm preserves totals without score-vector keys."""

        calls: list[str] = []
        bundle = self._production_bundle_antigravity(calls=calls)
        self.assertEqual(bundle["coverage"]["baseline_abstained_identities"], 395)
        arm = bundle["baselines"]["deep_svdd_1d"]["scores"][
            self.DATASETS_ANTIGRAVITY[0]
        ]
        self.assertEqual(arm["status"], "abstained")
        self.assertEqual(arm["reason"], "insufficient_train_rows")
        self.assertEqual(arm["test_indices"], [0, 1])
        self.assertNotIn("train_oof_scores", arm)
        self.assertNotIn("test_scores", arm)

    def test_complete_and_abstained_v3_arms_exact_vectors_and_counts(self) -> None:
        """Exact vectors, indices, coverage partition, and schema are emitted."""

        bundle = self._production_bundle_antigravity(calls=[])
        self.assertEqual(bundle["schema"], "exp471-verified-score-input-antigravity.v3")
        self.assertEqual(bundle["coverage"], {
            "total_identities": 11_170,
            "scored_identities": 10_775,
            "abstained_identities": 395,
            "failed_identities": 0,
            "condition_scored_identities": 4_468,
            "baseline_scored_identities": 6_307,
            "baseline_abstained_identities": 395,
        })
        scored = bundle["baselines"]["knn_distance"]["scores"][
            self.DATASETS_ANTIGRAVITY[5]
        ]
        self.assertEqual(scored["status"], "scored")
        self.assertEqual(scored["test_indices"], [0, 1])
        self.assertEqual(scored["train_oof_scores"], [0.25, 1.25])
        self.assertEqual(scored["test_scores"], [0.75, 1.75])
        for method in ("deep_svdd_1d", "drocc_1d"):
            arm = bundle["baselines"][method]["scores"][
                self.DATASETS_ANTIGRAVITY[0]
            ]
            self.assertEqual(arm["status"], "abstained")
            self.assertNotIn("train_oof_scores", arm)
            self.assertNotIn("test_scores", arm)
        self.assertIn(
            "authority_sha256",
            bundle["baselines"]["knn_distance"]["authority"],
        )

    def test_invalid_baseline_variants_stop_before_proposal_work(self) -> None:
        """Missing, failure, unknown reason, coordinates, and count drift fail closed."""

        cases = (
            ("missing", None, "coverage for knn_distance"),
            ("reordered_methods", None, "baseline shard order"),
            ("reordered_rows", None, "baseline shard order"),
            (
                "failure",
                lambda shard: shard.update({"status": "failure"}),
                "source status",
            ),
            (
                "unknown_reason",
                lambda shard: shard.update({"status": "abstain", "reason": "worker_crashed"}),
                "abstention reason",
            ),
            (
                "bad_indices",
                lambda shard: shard.update({"test_indices": [1, 0]}),
                "source TEST coordinates",
            ),
            (
                "count_drift",
                lambda shard: shard.update({"n_test": 3}),
                "source row counts",
            ),
            (
                "nonfinite_scores",
                lambda shard: shard.update({"test_scores": [0.75, float("inf")]}),
                "baseline score vector",
            ),
        )
        for mutation, change, expected_error in cases:
            with self.subTest(mutation=mutation):
                calls: list[str] = []
                with self.assertRaisesRegex(ValueError, expected_error):
                    self._production_bundle_antigravity(
                        calls=calls, mutation=(mutation, change)
                    )
                self.assertNotIn("require", calls)
                self.assertNotIn("load", calls)
                self.assertNotIn("build", calls)

    def test_condition_coordinate_drift_fails_after_build(self) -> None:
        """A-D output is rechecked against baseline TEST authority."""

        calls: list[str] = []
        with self.assertRaisesRegex(
            ValueError, "production condition TEST coordinates"
        ):
            self._production_bundle_antigravity(
                calls=calls, condition_drift=True
            )
        self.assertIn("build", calls)


class ReportingTestsAntigravity(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="exp471_reporting_cli_antigravity_"
        )
        self.base = Path(self.temporary.name).resolve()
        self.artifact_base = self.base / "artifact_antigravity"
        self.artifact = _artifact_antigravity(self.artifact_base)
        self.task5_root = self.artifact_base / "score_cache_antigravity"
        self.task5_root_patch = mock.patch.object(
            reporting,
            "_TASK5_SCORE_CACHE_ROOT_ANTIGRAVITY",
            self.task5_root,
            create=True,
        )
        self.task5_root_patch.start()
        self.report_root = self.base / "report_antigravity"

    def tearDown(self) -> None:
        self.task5_root_patch.stop()
        self.temporary.cleanup()

    def test_two_argument_public_interface_reads_durable_worker_provenance(self) -> None:
        """The specified two-argument API must bind the reopened worker decision."""

        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            score_cache_antigravity,
        )

        status = {
            **_worker_provenance_antigravity(),
            "complete": True,
            "manifest_sha256": self.artifact.cache_manifest_sha256,
            "total_shards": self.artifact.payload["score_cache"]["total_shards"],
        }
        with mock.patch.object(
            score_cache_antigravity,
            "status_score_cache_antigravity",
            return_value=status,
        ) as worker_status:
            authority = reporting.write_exp471_report_antigravity(
                self.artifact, self.report_root
            )
        self.assertRegex(authority.report_manifest_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(worker_status.call_count, 3)

    def test_report_closes_all_ten_methods_axes_tables_and_artifact_map(self) -> None:
        """Missing methods, swapped curve axes, or an unbound leaf must fail."""

        authority = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        reopened = reporting.read_exp471_report_antigravity(
            self.artifact, self.report_root
        )
        self.assertEqual(reopened, authority)
        self.assertEqual(
            set(authority.report_file_sha256),
            {
                path.relative_to(self.report_root).as_posix()
                for path in self.report_root.rglob("*")
                if path.is_file()
                and path.name
                not in {
                    reporting._COMPLETE_MARKER_NAME_ANTIGRAVITY,
                    reporting._ARTIFACT_MAP_NAME_ANTIGRAVITY,
                }
            },
        )
        metrics = json.loads(
            (
                self.report_root
                / "exp450_report_artifacts_antigravity"
                / "exp450_metric_tables_antigravity.json"
            ).read_text(encoding="utf-8")
        )
        curves = json.loads(
            (
                self.report_root
                / "exp450_report_artifacts_antigravity"
                / "exp450_curve_tables_antigravity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(tuple(metrics["method_order"]), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(set(metrics["macro"]), set(METHOD_ORDER_ANTIGRAVITY))
        self.assertEqual(set(metrics["micro"]), set(METHOD_ORDER_ANTIGRAVITY))
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        self.assertEqual(
            {row["method"] for row in curves["roc"]["rows"]},
            set(METHOD_ORDER_ANTIGRAVITY),
        )

    def test_changed_leaf_and_artifact_map_digest_are_rejected(self) -> None:
        """Changed report bytes or a rehashed foreign map cannot retain authority."""

        reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        leaf = (
            self.report_root
            / "exp450_report_artifacts_antigravity"
            / "exp450_metric_tables_antigravity.csv"
        )
        original_leaf = leaf.read_bytes()
        leaf.write_bytes(original_leaf + b"changed-public-literal\n")
        with self.assertRaisesRegex(ValueError, "leaf digest"):
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            )
        leaf.write_bytes(original_leaf)

        artifact_map = self.report_root / reporting._ARTIFACT_MAP_NAME_ANTIGRAVITY
        value = json.loads(artifact_map.read_text(encoding="utf-8"))
        value["files"][next(iter(value["files"]))] = "f" * 64
        artifact_map.write_bytes(canonical_json_bytes_exp471_antigravity(value))
        with self.assertRaisesRegex(ValueError, "artifact map"):
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            )

    def test_exact_reuse_preserves_every_inode_and_skips_renderer(self) -> None:
        """A second exact report call must not write or render any report leaf."""

        render_calls: list[str] = []

        def render(frozen, root):
            render_calls.append("render")
            return reporting.write_exp450_report_antigravity(frozen, root)

        first = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
            renderer=render,
        )
        before = {
            path.relative_to(self.report_root).as_posix(): (
                path.stat().st_ino,
                path.read_bytes(),
            )
            for path in self.report_root.rglob("*")
            if path.is_file()
        }
        second = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
            renderer=render,
        )
        after = {
            path.relative_to(self.report_root).as_posix(): (
                path.stat().st_ino,
                path.read_bytes(),
            )
            for path in self.report_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(render_calls, ["render"])

    def test_markerless_foreign_destination_and_symlinks_fail_closed(self) -> None:
        """A foreign reservation or linked destination must never be reclaimed."""

        self.report_root.mkdir()
        foreign = self.report_root / "foreign_antigravity.txt"
        foreign.write_bytes(b"foreign-public-literal\n")
        before = foreign.read_bytes()
        with self.assertRaisesRegex(ValueError, "markerless|foreign"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
            )
        self.assertEqual(foreign.read_bytes(), before)

        linked = self.base / "linked_report_antigravity"
        linked.symlink_to(self.report_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                linked,
                worker_provenance=_worker_provenance_antigravity(),
            )

    def test_fault_before_rename_is_invisible_and_after_rename_reopens(self) -> None:
        """Only a fully marker-bearing rename may become externally visible."""

        def before(stage: str) -> None:
            if stage == "before_rename":
                raise RuntimeError("public fault before rename")

        with self.assertRaisesRegex(RuntimeError, "before rename"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=before,
            )
        self.assertFalse(os.path.lexists(self.report_root))

        after_calls: list[str] = []

        def after(stage: str) -> None:
            after_calls.append(stage)
            if stage == "after_rename":
                raise RuntimeError("public fault after rename")

        authority = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
            fault=after,
        )
        self.assertIn("after_rename", after_calls)
        self.assertEqual(
            authority,
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            ),
        )

    def test_live_lock_rejects_dead_owner_recovers_and_symlink_fails(self) -> None:
        """PID reuse and unsafe lock records must not authorize lock removal."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        current = reporting._process_start_identity_antigravity(os.getpid())
        active = reporting._lock_record_antigravity(os.getpid(), current)
        lock.write_bytes(canonical_json_bytes_exp471_antigravity(active))
        with self.assertRaisesRegex(ValueError, "active"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
            )
        self.assertEqual(
            lock.read_bytes(), canonical_json_bytes_exp471_antigravity(active)
        )

        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    99_999_999, "dead-public-process-antigravity"
                )
            )
        )
        reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        self.assertFalse(os.path.lexists(lock))

        other_root = self.base / "other_report_antigravity"
        other_lock = reporting._report_lock_path_antigravity(other_root)
        target = self.base / "foreign_lock_antigravity.json"
        target.write_bytes(b"{}\n")
        other_lock.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "lock"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                other_root,
                worker_provenance=_worker_provenance_antigravity(),
            )

    def test_live_pid_with_different_start_identity_recovers_as_pid_reuse(self) -> None:
        """A live reused PID must not preserve a lock from another process lifetime."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        actual = reporting._process_start_identity_antigravity(os.getpid())
        self.assertIsNotNone(actual)
        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    os.getpid(), f"different-{actual}-public-antigravity"
                )
            )
        )
        authority = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        self.assertRegex(authority.report_manifest_sha256, r"^[0-9a-f]{64}$")
        self.assertFalse(os.path.lexists(lock))

    def test_pre_rename_stage_replacement_is_never_published(self) -> None:
        """Replacing the verified staging name must not publish foreign bytes."""

        saved = self.base / "owned_stage_saved_antigravity"
        foreign = b"foreign-stage-public-literal-antigravity\n"

        def swap(stage_name: str) -> None:
            if stage_name != "before_rename":
                return
            stage = next(self.base.glob(".report_antigravity.stage_antigravity_*"))
            stage.rename(saved)
            stage.mkdir()
            (stage / "foreign_antigravity.txt").write_bytes(foreign)

        with self.assertRaisesRegex(ValueError, "staging|identity"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=swap,
            )
        self.assertFalse(os.path.lexists(self.report_root))
        replacement = next(
            self.base.glob(".report_antigravity.stage_antigravity_*")
        )
        self.assertEqual((replacement / "foreign_antigravity.txt").read_bytes(), foreign)
        self.assertTrue(saved.is_dir())

    def test_failed_stage_cleanup_preserves_namespace_replacement(self) -> None:
        """Failure cleanup must never recursively delete a replacement directory."""

        saved = self.base / "owned_failed_stage_saved_antigravity"
        foreign = b"foreign-cleanup-public-literal-antigravity\n"

        def swap_and_fail(stage_name: str) -> None:
            if stage_name != "before_rename":
                return
            stage = next(self.base.glob(".report_antigravity.stage_antigravity_*"))
            stage.rename(saved)
            stage.mkdir()
            (stage / "foreign_antigravity.txt").write_bytes(foreign)
            raise RuntimeError("public stage cleanup fault")

        with self.assertRaisesRegex(RuntimeError, "cleanup fault"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=swap_and_fail,
            )
        replacement = next(
            self.base.glob(".report_antigravity.stage_antigravity_*")
        )
        self.assertEqual((replacement / "foreign_antigravity.txt").read_bytes(), foreign)
        self.assertTrue(saved.is_dir())

    def test_lock_release_revalidates_identity_after_verified_read(self) -> None:
        """Normal release must not unlink a foreign lock pathname replacement."""

        token = reporting._acquire_report_lock_antigravity(self.report_root)
        owned = self.base / "owned_lock_saved_antigravity.json"
        foreign = b"foreign-lock-public-literal-antigravity\n"
        original = reporting._read_lock_antigravity

        def swap_after_read(path):
            result = original(path)
            path.rename(owned)
            path.write_bytes(foreign)
            return result

        with mock.patch.object(
            reporting, "_read_lock_antigravity", side_effect=swap_after_read
        ):
            with self.assertRaisesRegex(ValueError, "lock.*identity|changed"):
                reporting._release_report_lock_antigravity(token)
        self.assertEqual(token.path.read_bytes(), foreign)
        self.assertEqual(owned.read_bytes(), token.payload)

    def test_dead_lock_quarantine_cleanup_revalidates_moved_identity(self) -> None:
        """Dead-lock recovery must preserve a foreign quarantine replacement."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    99_999_999, "dead-public-process-antigravity"
                )
            )
        )
        owned = self.base / "owned_dead_lock_saved_antigravity.json"
        foreign = b"foreign-dead-lock-public-literal-antigravity\n"
        original = reporting._read_lock_antigravity

        def swap_quarantine_after_read(path):
            result = original(path)
            if path.name.endswith(".dead_antigravity"):
                path.rename(owned)
                path.write_bytes(foreign)
            return result

        with mock.patch.object(
            reporting,
            "_read_lock_antigravity",
            side_effect=swap_quarantine_after_read,
        ):
            with self.assertRaisesRegex(ValueError, "dead lock.*identity|changed"):
                reporting._acquire_report_lock_antigravity(self.report_root)
        quarantine = next(self.base.glob(".*.dead_antigravity"))
        self.assertEqual(quarantine.read_bytes(), foreign)
        self.assertTrue(owned.is_file())

    def test_postcheck_stage_substitution_never_reaches_canonical_root(self) -> None:
        """A stage substituted after the last check must remain non-authoritative."""

        saved = self.base / "owned_postcheck_stage_saved_antigravity"
        foreign = b"foreign-postcheck-stage-public-literal-antigravity\n"
        swapped = False

        def swap_after_last_check(scope, parent, name):
            nonlocal swapped
            if scope == "staging before rename" and not swapped:
                swapped = True
                stage = self.base / name
                stage.rename(saved)
                stage.mkdir()
                (stage / "foreign_antigravity.txt").write_bytes(foreign)

        with mock.patch.object(
            reporting,
            "_postcheck_namespace_hook_antigravity",
            side_effect=swap_after_last_check,
        ):
            with self.assertRaisesRegex(ValueError, "staging|identity|claim"):
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    self.report_root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
        self.assertFalse(os.path.lexists(self.report_root))
        self.assertTrue(saved.is_dir())
        self.assertTrue(
            any(
                path.is_file() and path.read_bytes() == foreign
                for path in self.base.rglob("*")
            )
        )

    def test_postcheck_lock_release_substitution_is_claimed_and_preserved(self) -> None:
        """A foreign lock substituted after the last check must not be unlinked."""

        token = reporting._acquire_report_lock_antigravity(self.report_root)
        owned = self.base / "owned_postcheck_lock_saved_antigravity.json"
        foreign = b"foreign-postcheck-lock-public-literal-antigravity\n"
        swapped = False

        def swap_after_last_check(scope, parent, name):
            nonlocal swapped
            if scope == "lock release" and not swapped:
                swapped = True
                token.path.rename(owned)
                token.path.write_bytes(foreign)

        with mock.patch.object(
            reporting,
            "_postcheck_namespace_hook_antigravity",
            side_effect=swap_after_last_check,
        ):
            with self.assertRaisesRegex(ValueError, "lock.*identity|claim"):
                reporting._release_report_lock_antigravity(token)
        self.assertEqual(owned.read_bytes(), token.payload)
        self.assertTrue(
            any(
                path.is_file() and path.read_bytes() == foreign
                for path in self.base.rglob("*")
            )
        )

    def test_postcheck_dead_quarantine_substitution_is_claimed_and_preserved(self) -> None:
        """A foreign dead-lock quarantine replacement must not be unlinked."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    99_999_999, "dead-postcheck-public-process-antigravity"
                )
            )
        )
        value, raw, opened = reporting._read_lock_antigravity(lock)
        parent_descriptor, parent = reporting._open_directory_anchor_antigravity(
            self.base
        )
        owned = self.base / "owned_postcheck_dead_saved_antigravity.json"
        foreign = b"foreign-postcheck-dead-public-literal-antigravity\n"
        swapped = False

        def swap_after_last_check(scope, parent_fd, name):
            nonlocal swapped
            if scope == "dead lock quarantine" and not swapped:
                swapped = True
                quarantine = self.base / name
                quarantine.rename(owned)
                quarantine.write_bytes(foreign)

        try:
            with mock.patch.object(
                reporting,
                "_postcheck_namespace_hook_antigravity",
                side_effect=swap_after_last_check,
            ):
                with self.assertRaisesRegex(ValueError, "dead lock.*identity|claim"):
                    reporting._recover_dead_lock_antigravity(
                        lock,
                        value,
                        raw,
                        opened,
                        parent_descriptor,
                        parent.st_dev,
                        parent.st_ino,
                    )
        finally:
            os.close(parent_descriptor)
        self.assertEqual(owned.read_bytes(), raw)
        self.assertTrue(
            any(
                path.is_file() and path.read_bytes() == foreign
                for path in self.base.rglob("*")
            )
        )

    def test_poststat_recursive_leaf_substitution_is_claimed_and_preserved(self) -> None:
        """Recursive cleanup must never unlink a leaf substituted after stat."""

        token = reporting._acquire_report_lock_antigravity(self.report_root)
        stage = reporting._create_stage_token_antigravity(self.report_root, token)
        leaf = stage.path / "owned_leaf_antigravity.txt"
        owned = self.base / "owned_poststat_leaf_saved_antigravity.txt"
        original_payload = b"owned-poststat-leaf-public-literal-antigravity\n"
        foreign = b"foreign-poststat-leaf-public-literal-antigravity\n"
        leaf.write_bytes(original_payload)
        swapped = False

        def swap_after_stat(scope, parent_descriptor, name):
            nonlocal swapped
            if (
                name == leaf.name
                and parent_descriptor == stage.descriptor
                and scope == "staging regular child"
                and not swapped
            ):
                swapped = True
                leaf.rename(owned)
                leaf.write_bytes(foreign)

        cleanup_error = None
        try:
            with mock.patch.object(
                reporting,
                "_postcheck_namespace_hook_antigravity",
                side_effect=swap_after_stat,
            ):
                try:
                    reporting._cleanup_stage_antigravity(stage, token)
                except ValueError as error:
                    cleanup_error = error
        finally:
            os.close(stage.descriptor)
            reporting._release_report_lock_antigravity(token)
        self.assertIsNotNone(cleanup_error)
        self.assertEqual(owned.read_bytes(), original_payload)
        self.assertTrue(
            any(
                path.is_file() and path.read_bytes() == foreign
                for path in self.base.rglob("*")
            )
        )

    def test_final_delete_claim_substitution_preserves_owned_and_foreign(self) -> None:
        """A claim replaced after its final check must never delete the replacement."""

        token = reporting._acquire_report_lock_antigravity(self.report_root)
        owned = self.base / "owned_final_delete_claim_saved_antigravity.json"
        foreign = b"foreign-final-delete-claim-public-literal-antigravity\n"
        swapped = False

        def swap_after_final_claim_check(scope, parent, name):
            nonlocal swapped
            if scope == "lock release delete claim" and not swapped:
                swapped = True
                claim = self.base / name
                claim.rename(owned)
                claim.write_bytes(foreign)

        with mock.patch.object(
            reporting,
            "_final_claim_namespace_hook_antigravity",
            side_effect=swap_after_final_claim_check,
        ):
            with self.assertRaisesRegex(ValueError, "claim.*identity|retained"):
                reporting._release_report_lock_antigravity(token)
        self.assertEqual(owned.read_bytes(), token.payload)
        self.assertFalse(os.path.lexists(token.path))
        self.assertTrue(
            any(
                path.is_file() and path.read_bytes() == foreign
                for path in self.base.rglob("*")
            )
        )

    def test_final_publication_claim_substitution_evacuates_canonical_foreign(self) -> None:
        """A final claim replacement may be quarantined but never remain canonical."""

        owned = self.base / "owned_final_publish_claim_saved_antigravity"
        foreign = b"foreign-final-publish-claim-public-literal-antigravity\n"
        swapped = False

        def swap_immediately_before_final_rename(scope, parent, source):
            nonlocal swapped
            if scope == "staging publication claim" and not swapped:
                swapped = True
                claim = self.base / source
                claim.rename(owned)
                claim.mkdir()
                (claim / "foreign_antigravity.txt").write_bytes(foreign)

        with mock.patch.object(
            reporting,
            "_final_claim_namespace_hook_antigravity",
            side_effect=swap_immediately_before_final_rename,
        ):
            with self.assertRaisesRegex(ValueError, "published|canonical|identity|leaf"):
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    self.report_root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
        self.assertFalse(os.path.lexists(self.report_root))
        self.assertTrue(
            (owned / reporting._COMPLETE_MARKER_NAME_ANTIGRAVITY).is_file()
        )
        self.assertTrue(
            any(
                path.is_file() and path.read_bytes() == foreign
                for path in self.base.rglob("*")
            )
        )

    def test_parent_replacement_fails_before_publication_or_cleanup(self) -> None:
        """A replaced parent must strand owned objects instead of mutating the new parent."""

        moved_parent = self.base.with_name(f"{self.base.name}_moved_antigravity")

        def restore_parent() -> None:
            if not moved_parent.exists():
                return
            if self.base.exists():
                self.base.rmdir()
            moved_parent.rename(self.base)

        self.addCleanup(restore_parent)

        def replace_parent(stage_name: str) -> None:
            if stage_name != "before_rename":
                return
            self.base.rename(moved_parent)
            self.base.mkdir()

        with self.assertRaisesRegex(ValueError, "parent.*changed|identity"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=replace_parent,
            )
        self.assertEqual(tuple(self.base.iterdir()), ())
        self.assertTrue(moved_parent.is_dir())

    def test_concurrent_identical_publishers_render_once_and_reuse_winner(self) -> None:
        """An identical concurrent loser must reopen the exact winner without rendering."""

        rendering = threading.Event()
        release = threading.Event()
        render_calls: list[str] = []

        def render(frozen, root):
            render_calls.append("render")
            rendering.set()
            self.assertTrue(release.wait(timeout=5))
            return reporting.write_exp450_report_antigravity(frozen, root)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                renderer=render,
            )
            self.assertTrue(rendering.wait(timeout=5))
            second = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                renderer=render,
            )
            release.set()
            first_authority = first.result(timeout=10)
            second_authority = second.result(timeout=10)
        self.assertEqual(first_authority, second_authority)
        self.assertEqual(render_calls, ["render"])

    def test_concurrent_different_provenance_rejects_loser_without_render(self) -> None:
        """A concurrent loser with different provenance must not reuse the winner."""

        rendering = threading.Event()
        release = threading.Event()
        render_calls: list[str] = []

        def render(frozen, root):
            render_calls.append("render")
            rendering.set()
            self.assertTrue(release.wait(timeout=5))
            return reporting.write_exp450_report_antigravity(frozen, root)

        different = _worker_provenance_antigravity(single_worker_seconds=0.125)
        self.assertNotEqual(different, _worker_provenance_antigravity())
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                renderer=render,
            )
            self.assertTrue(rendering.wait(timeout=5))
            second = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=different,
                renderer=render,
            )
            release.set()
            first.result(timeout=10)
            with self.assertRaisesRegex(ValueError, "provenance"):
                second.result(timeout=10)
        self.assertEqual(render_calls, ["render"])

    def test_independent_process_lock_contention_has_one_safe_owner(self) -> None:
        """Two OS processes must leave one valid report and no unsafe lock reuse."""

        context = multiprocessing.get_context("fork")
        results = context.Queue()
        ready = context.Event()
        release = context.Event()
        arguments = (
            self.artifact,
            str(self.report_root),
            str(self.task5_root),
            _worker_provenance_antigravity(),
            results,
            ready,
            release,
        )
        owner = context.Process(
            target=_independent_report_contender_antigravity,
            args=(*arguments, True),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            owner.start()
        self.assertTrue(ready.wait(timeout=10))
        contender = context.Process(
            target=_independent_report_contender_antigravity,
            args=(*arguments, False),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            contender.start()
        contender.join(timeout=10)
        self.assertFalse(contender.is_alive())
        release.set()
        owner.join(timeout=15)
        self.assertFalse(owner.is_alive())
        self.assertEqual((owner.exitcode, contender.exitcode), (0, 0))
        observed = [results.get(timeout=5), results.get(timeout=5)]
        self.assertEqual(sum(row[0] == "success" for row in observed), 1)
        self.assertEqual(sum(row[:2] == ("error", "ValueError") for row in observed), 1)
        self.assertTrue(
            any("active owner" in row[-1] for row in observed if row[0] == "error")
        )
        self.assertFalse(
            os.path.lexists(reporting._report_lock_path_antigravity(self.report_root))
        )
        self.assertRegex(
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            ).report_manifest_sha256,
            r"^[0-9a-f]{64}$",
        )

    def test_rehashed_scientific_leaf_tampering_is_rejected(self) -> None:
        """Every frozen format must remain the exact Task 6 representation."""

        nested = "exp450_report_artifacts_antigravity"

        def json_mutation(relative: str, mutate) -> bytes:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
            mutate(value)
            return canonical_json_bytes_exp471_antigravity(value)

        mutations = {
            f"{nested}/exp450_metric_tables_antigravity.json": lambda root: json_mutation(
                f"{nested}/exp450_metric_tables_antigravity.json",
                lambda value: value["method_order"].pop(),
            ),
            f"{nested}/exp450_curve_tables_antigravity.json": lambda root: json_mutation(
                f"{nested}/exp450_curve_tables_antigravity.json",
                lambda value: value["roc"].update(
                    {"x_name": "true_positive_rate", "y_name": "false_positive_rate"}
                ),
            ),
            f"{nested}/exp450_paired_differences_antigravity.json": lambda root: json_mutation(
                f"{nested}/exp450_paired_differences_antigravity.json",
                lambda value: value["paired_statistics"][0].update(
                    {"point_estimate": 0.125}
                ),
            ),
            f"{nested}/exp450_report_manifest_antigravity.json": lambda root: canonical_json_bytes_exp471_antigravity(
                {
                    **json.loads(
                        (
                            root
                            / nested
                            / "exp450_report_manifest_antigravity.json"
                        ).read_text(encoding="utf-8")
                    ),
                    "exp450_metric_tables_antigravity.json": "f" * 64,
                }
            ),
            f"{nested}/exp450_metric_tables_antigravity.csv": lambda root: (
                root / nested / "exp450_metric_tables_antigravity.csv"
            ).read_bytes()
            + b"foreign,metric,csv,public,literal\n",
            f"{nested}/exp450_curve_tables_antigravity.csv": lambda root: (
                root / nested / "exp450_curve_tables_antigravity.csv"
            ).read_bytes()
            + b"foreign,curve,csv,public,literal\n",
            f"{nested}/exp450_paired_differences_antigravity.csv": lambda root: (
                root / nested / "exp450_paired_differences_antigravity.csv"
            ).read_bytes()
            + b"foreign,paired,csv,public,literal\n",
            f"{nested}/exp450_curves_antigravity.svg": lambda root: (
                root / nested / "exp450_curves_antigravity.svg"
            ).read_bytes().replace(b"roc", b"bad", 1),
            f"{nested}/exp450_curves_antigravity.png": lambda root: (
                root / nested / "exp450_curves_antigravity.png"
            ).read_bytes()[:-1]
            + b"X",
            f"{nested}/exp450_paper_report_antigravity.html": lambda root: (
                root / nested / "exp450_paper_report_antigravity.html"
            ).read_bytes().replace(b"continuous", b"fabricated", 1),
        }
        for index, (relative, mutate) in enumerate(mutations.items()):
            with self.subTest(relative=relative):
                root = self.base / f"semantic_{index}_report_antigravity"
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
                _rehash_report_leaf_antigravity(root, relative, mutate(root))
                with self.assertRaisesRegex(ValueError, "frozen|semantic|leaf"):
                    reporting.read_exp471_report_antigravity(self.artifact, root)

    def test_rehashed_complete_provenance_tampering_is_rejected(self) -> None:
        """Every environment, worker, closure, axis, and digest field is authoritative."""

        mutations = {
            "worker": lambda value: value["worker_provenance"].update(
                {"detected_logical_cpus": 999}
            ),
            "interpreter": lambda value: value["interpreter"].update(
                {"version": "0.0-public-antigravity"}
            ),
            "dependency": lambda value: value["dependencies"].update(
                {"numpy": "0-public-antigravity"}
            ),
            "axes": lambda value: value["curve_axes"]["roc"].update(
                {"x": "true_positive_rate"}
            ),
            "evaluation": lambda value: value["evaluation_implementation"].update(
                {"scientific_contract": "foreign-public-antigravity"}
            ),
            "rendered": lambda value: value["rendered_leaf_sha256"].update(
                {next(iter(value["rendered_leaf_sha256"])): "f" * 64}
            ),
        }
        for index, (name, mutate) in enumerate(mutations.items()):
            with self.subTest(name=name):
                root = self.base / f"provenance_{index}_report_antigravity"
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
                relative = reporting._PROVENANCE_NAME_ANTIGRAVITY
                value = json.loads((root / relative).read_text(encoding="utf-8"))
                mutate(value)
                _rehash_report_leaf_antigravity(
                    root,
                    relative,
                    canonical_json_bytes_exp471_antigravity(value),
                )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    reporting.read_exp471_report_antigravity(self.artifact, root)

    def test_rehashed_valid_worker_history_substitution_is_rejected(self) -> None:
        """A second valid history cannot replace independently durable Task 5 bytes."""

        reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        relative = reporting._PROVENANCE_NAME_ANTIGRAVITY
        provenance = json.loads(
            (self.report_root / relative).read_text(encoding="utf-8")
        )
        substituted = _worker_provenance_antigravity(single_worker_seconds=0.125)
        self.assertNotEqual(
            substituted["benchmark_decision"]["decision_sha256"],
            provenance["worker_provenance"]["benchmark_decision"][
                "decision_sha256"
            ],
        )
        provenance["worker_provenance"] = substituted
        _rehash_report_leaf_antigravity(
            self.report_root,
            relative,
            canonical_json_bytes_exp471_antigravity(provenance),
        )
        with self.assertRaisesRegex(ValueError, "durable.*worker|worker.*authority"):
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            )

    def test_rehashed_method_metric_and_coordinate_mutations_are_rejected(self) -> None:
        """Order, numeric metrics, and paired curve coordinates share one authority."""

        nested = "exp450_report_artifacts_antigravity"
        cases = (
            (
                "reordered-methods",
                f"{nested}/exp450_metric_tables_antigravity.json",
                lambda value: value["method_order"].reverse(),
            ),
            (
                "changed-metric",
                f"{nested}/exp450_metric_tables_antigravity.json",
                lambda value: value["macro"][METHOD_ORDER_ANTIGRAVITY[0]].update(
                    {"average_precision": 0.125}
                ),
            ),
            (
                "swapped-roc-coordinate",
                f"{nested}/exp450_curve_tables_antigravity.json",
                lambda value: value["roc"]["rows"][1].update(
                    {
                        "false_positive_rate": value["roc"]["rows"][1][
                            "true_positive_rate"
                        ],
                        "true_positive_rate": value["roc"]["rows"][1][
                            "false_positive_rate"
                        ],
                    }
                ),
            ),
        )
        for index, (name, relative, mutate) in enumerate(cases):
            with self.subTest(name=name):
                root = self.base / f"scientific_{index}_report_antigravity"
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
                value = json.loads((root / relative).read_text(encoding="utf-8"))
                mutate(value)
                _rehash_report_leaf_antigravity(
                    root,
                    relative,
                    canonical_json_bytes_exp471_antigravity(value),
                )
                with self.assertRaisesRegex(ValueError, "frozen|semantic|leaf"):
                    reporting.read_exp471_report_antigravity(self.artifact, root)

    def test_partial_worker_provenance_is_rejected_before_rendering(self) -> None:
        """A self-described worker mapping cannot replace Task 5 decisions."""

        with self.assertRaisesRegex(ValueError, "worker provenance"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance={
                    "benchmark_decision": {"selected_workers": 1},
                    "execution_decision": {"selected_workers": 1},
                },
            )
        self.assertFalse(os.path.lexists(self.report_root))


class _LifecycleRuntimeAntigravity:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cache_complete = False
        self.evaluation_complete = False
        self.report_complete = False
        self.authority = type(
            "AuthorityAntigravity",
            (),
            {"authority_sha256": "a" * 64, "inventory_sha256": "b" * 64},
        )()
        self.session = type(
            "SessionAntigravity", (), {"authority": self.authority}
        )()
        self.cache = type(
            "CacheAntigravity",
            (),
            {"manifest_sha256": "c" * 64, "total_shards": 10},
        )()
        self.artifact = type(
            "ArtifactAntigravity", (), {"evaluation_sha256": "d" * 64}
        )()
        self.report_authority = type(
            "ReportAuthorityAntigravity",
            (),
            {"report_manifest_sha256": "e" * 64},
        )()

    def verify_source_antigravity(self):
        self.calls.append("verify-source")
        return self.session

    def open_cache_antigravity(self, session):
        self.calls.append("open-cache")
        if not self.cache_complete:
            raise FileNotFoundError("public cache absent")
        return self.cache

    def benchmark_prepare_antigravity(self, session):
        self.calls.append("benchmark-prepare")
        return {"selected_workers": 1, "byte_parity": True}

    def prepare_antigravity(self, session, progress):
        self.calls.append("prepare")
        self.cache_complete = True
        progress(
            {
                "durable_completed": 10,
                "total": 10,
                "failures": 0,
                "remaining": 0,
            }
        )
        return self.cache

    def open_evaluation_antigravity(self, session, cache):
        self.calls.append("open-evaluation")
        if not self.evaluation_complete:
            raise FileNotFoundError("public evaluation absent")
        return self.artifact

    def evaluate_antigravity(self, session, cache):
        self.calls.append("evaluate")
        self.evaluation_complete = True
        return self.artifact

    def report_antigravity(self, artifact):
        self.calls.append("report")
        if not self.report_complete:
            self.calls.append("render")
            self.report_complete = True
        return self.report_authority

    def status_antigravity(self):
        self.calls.append("status")
        return {
            "durable_completed": 10 if self.cache_complete else 0,
            "total": 10,
            "failures": 0,
            "remaining": 0 if self.cache_complete else 10,
            "verified_stage": "report" if self.report_complete else "source",
            "paths": {"root": "/public/literal/exp471_antigravity"},
            "worker_decision": {"selected_workers": 1},
            "latest_progress": None,
            "failure": None,
        }


class CliTestsAntigravity(unittest.TestCase):
    def test_progress_read_rejects_root_swap_after_descriptor_read(self) -> None:
        """A read from a detached root descriptor must never return authority."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_read_root_swap_antigravity_"
        ) as directory:
            root = Path(directory).resolve() / "root_antigravity"
            root.mkdir()
            artifact = root / "cache_antigravity"
            saved = root.with_name("saved_root_antigravity")
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                callback = cli._progress_callback_antigravity(
                    "score-cache", artifact
                )
                callback(
                    {
                        "durable_completed": 1,
                        "total": 11_170,
                        "status": "durable",
                    }
                )
                original_read = cli._read_all_descriptor_antigravity
                swapped = False

                def swap_after_read(descriptor):
                    nonlocal swapped
                    raw = original_read(descriptor)
                    if not swapped:
                        swapped = True
                        root.rename(saved)
                        root.mkdir()
                    return raw

                with mock.patch.object(
                    cli,
                    "_read_all_descriptor_antigravity",
                    side_effect=swap_after_read,
                ):
                    with self.assertRaisesRegex(ValueError, "progress root changed"):
                        cli._latest_progress_antigravity()
            self.assertFalse((root / cli._PROGRESS_NAME_ANTIGRAVITY).exists())
            self.assertTrue((saved / cli._PROGRESS_NAME_ANTIGRAVITY).is_file())

    def test_progress_append_rejects_root_swap_after_last_log_check(self) -> None:
        """Append must fail before writing through a detached root descriptor."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_append_root_swap_antigravity_"
        ) as directory:
            root = Path(directory).resolve() / "root_antigravity"
            root.mkdir()
            artifact = root / "cache_antigravity"
            saved = root.with_name("saved_root_antigravity")
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                callback = cli._progress_callback_antigravity(
                    "score-cache", artifact
                )
                callback(
                    {
                        "durable_completed": 1,
                        "total": 11_170,
                        "status": "durable",
                    }
                )
                original_identity = cli._progress_log_identity_antigravity
                checks = 0

                def swap_after_last_prewrite_check(root_descriptor, descriptor):
                    nonlocal checks
                    result = original_identity(root_descriptor, descriptor)
                    checks += 1
                    if checks == 2:
                        root.rename(saved)
                        root.mkdir()
                    return result

                with mock.patch.object(
                    cli,
                    "_progress_log_identity_antigravity",
                    side_effect=swap_after_last_prewrite_check,
                ):
                    with self.assertRaisesRegex(ValueError, "progress root changed"):
                        callback(
                            {
                                "durable_completed": 2,
                                "total": 11_170,
                                "status": "durable",
                            }
                        )
            self.assertFalse((root / cli._PROGRESS_NAME_ANTIGRAVITY).exists())
            detached_records = (
                saved / cli._PROGRESS_NAME_ANTIGRAVITY
            ).read_bytes().splitlines()
            self.assertEqual(len(detached_records), 1)

    def test_progress_rejects_stage_false_totals_and_artifact_paths(self) -> None:
        """Canonical syntax cannot self-assert a false stage or durable artifact."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_stage_contract_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            path = root / cli._PROGRESS_NAME_ANTIGRAVITY
            artifact = root / "cache_antigravity"
            cases = (
                ("report", 1, 2, root / "foreign_report_antigravity"),
                ("score-cache", 1, 10, root / "foreign_cache_antigravity"),
                ("evaluation", 10, 10, root / "foreign_eval_antigravity.json"),
            )
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                for stage, completed, total, artifact in cases:
                    with self.subTest(stage=stage, artifact=artifact.name):
                        record = {
                            "schema": "exp471-progress-antigravity.v1",
                            "stage": stage,
                            "durable_completed": completed,
                            "total": total,
                            "failures": 0,
                            "attempts": 1,
                            "remaining": total - completed,
                            "elapsed_seconds": 0.0,
                            "eta_seconds": 0.0 if completed == total else None,
                            "timestamp_utc": "2026-08-25T00:00:00+00:00",
                            "artifact_path": str(artifact),
                            "root_path": str(root),
                        }
                        path.write_bytes(
                            canonical_json_bytes_exp471_antigravity(record)
                        )
                        with self.assertRaisesRegex(
                            ValueError, "progress (stage|path|total|terminal)"
                        ):
                            cli._latest_progress_antigravity()

    def test_progress_rejects_exact_terminal_report_without_durable_marker(self) -> None:
        """An exact path and 11/11 count cannot invent terminal authority."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_terminal_marker_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            score = root / "score_cache_antigravity"
            evaluation_root = root / "evaluation_antigravity"
            report_root = root / "report_antigravity"
            record = {
                "schema": "exp471-progress-antigravity.v1",
                "stage": "report",
                "durable_completed": 11,
                "total": 11,
                "failures": 0,
                "attempts": 1,
                "remaining": 0,
                "elapsed_seconds": 0.0,
                "eta_seconds": 0.0,
                "timestamp_utc": "2026-08-25T00:00:00+00:00",
                "artifact_path": str(report_root),
                "root_path": str(root),
            }
            (root / cli._PROGRESS_NAME_ANTIGRAVITY).write_bytes(
                canonical_json_bytes_exp471_antigravity(record)
            )
            with ExitStack() as stack:
                for name, value in (
                    ("EXP471_ROOT_ANTIGRAVITY", root),
                    ("EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", score),
                    ("EXP471_EVALUATION_ROOT_ANTIGRAVITY", evaluation_root),
                    ("EXP471_REPORT_ROOT_ANTIGRAVITY", report_root),
                ):
                    stack.enter_context(mock.patch.object(cli, name, value))
                with self.assertRaisesRegex(
                    ValueError, "progress.*(lifecycle|artifact|marker)"
                ):
                    cli._latest_progress_antigravity()

    def test_production_status_on_fresh_root_is_empty_and_observational(self) -> None:
        """Production status before the first run must create no durable object."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_fresh_status_antigravity_"
        ) as directory:
            root = Path(directory).resolve() / "absent_root_antigravity"
            score = root / "score_cache_antigravity"
            evaluation_root = root / "evaluation_antigravity"
            report_root = root / "report_antigravity"
            stdout, stderr = io.StringIO(), io.StringIO()
            with ExitStack() as stack:
                for name, value in (
                    ("EXP471_ROOT_ANTIGRAVITY", root),
                    ("EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", score),
                    ("EXP471_EVALUATION_ROOT_ANTIGRAVITY", evaluation_root),
                    ("EXP471_REPORT_ROOT_ANTIGRAVITY", report_root),
                ):
                    stack.enter_context(mock.patch.object(cli, name, value))
                result = main_antigravity(
                    ["status"], stdout=stdout, stderr=stderr
                )
            self.assertEqual(result, 0)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["durable_completed"], 0)
            self.assertEqual(payload["latest_progress"], None)
            self.assertIn("preprocessing_progress", payload)
            self.assertEqual(payload.get("preprocessing_progress"), None)
            self.assertEqual(payload["failures"], 0)
            self.assertFalse(os.path.lexists(root))
            self.assertFalse(os.path.lexists(score))
            self.assertFalse(os.path.lexists(evaluation_root))
            self.assertFalse(os.path.lexists(report_root))

    def test_missing_directory_revalidates_opened_prefix_generation(self) -> None:
        """ENOENT must not hide replacement of an already opened root prefix."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_status_missing_prefix_swap_antigravity_"
        ) as directory:
            base = Path(directory).resolve()
            root = base / "root_antigravity"
            saved = base / "saved_root_antigravity"
            root.mkdir()
            old_literal = root / "old_public_literal_antigravity.txt"
            old_literal.write_bytes(b"old-public-literal-antigravity\n")
            new_payload = b"new-public-literal-antigravity\n"
            original_open = cli.os.open
            swapped = False

            def swap_before_missing_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal swapped
                if path == "evaluation_antigravity" and not swapped:
                    swapped = True
                    root.rename(saved)
                    root.mkdir()
                    (root / "new_public_literal_antigravity.txt").write_bytes(
                        new_payload
                    )
                return original_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(
                cli.os, "open", side_effect=swap_before_missing_open
            ):
                with self.assertRaisesRegex(ValueError, "directory path changed"):
                    cli._open_existing_directory_chain_antigravity(
                        root / "evaluation_antigravity"
                    )
            self.assertTrue(swapped)
            self.assertEqual(
                (saved / old_literal.name).read_bytes(),
                b"old-public-literal-antigravity\n",
            )
            self.assertEqual(
                (root / "new_public_literal_antigravity.txt").read_bytes(),
                new_payload,
            )

    def test_production_status_binds_observations_to_one_root_generation(self) -> None:
        """Cache/evaluation/report/progress observations cannot span root generations."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_status_root_generation_antigravity_"
        ) as directory:
            base = Path(directory).resolve()
            root = base / "root_antigravity"
            saved = base / "saved_root_antigravity"
            score = root / "score_cache_antigravity"
            evaluation_root = root / "evaluation_antigravity"
            report_root = root / "report_antigravity"
            root.mkdir()
            old_payload = b"old-status-root-public-literal-antigravity\n"
            new_payload = b"new-status-root-public-literal-antigravity\n"
            (root / "old_public_literal_antigravity.txt").write_bytes(old_payload)
            original_read = cli._read_configured_leaf_antigravity
            swapped = False

            def swap_before_evaluation_observation(configured_root, name):
                nonlocal swapped
                if name == "evaluation_antigravity.json" and not swapped:
                    swapped = True
                    root.rename(saved)
                    root.mkdir()
                    (root / "new_public_literal_antigravity.txt").write_bytes(
                        new_payload
                    )
                return original_read(configured_root, name)

            stdout, stderr = io.StringIO(), io.StringIO()
            with ExitStack() as stack:
                for attribute, value in (
                    ("EXP471_ROOT_ANTIGRAVITY", root),
                    ("EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", score),
                    ("EXP471_EVALUATION_ROOT_ANTIGRAVITY", evaluation_root),
                    ("EXP471_REPORT_ROOT_ANTIGRAVITY", report_root),
                ):
                    stack.enter_context(mock.patch.object(cli, attribute, value))
                stack.enter_context(
                    mock.patch.object(
                        cli,
                        "_read_configured_leaf_antigravity",
                        side_effect=swap_before_evaluation_observation,
                    )
                )
                result = main_antigravity(
                    ["status"], stdout=stdout, stderr=stderr
                )
            self.assertTrue(swapped)
            self.assertEqual(result, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                json.loads(stderr.getvalue())["exception_type"], "ValueError"
            )
            self.assertEqual(
                (saved / "old_public_literal_antigravity.txt").read_bytes(),
                old_payload,
            )
            self.assertEqual(
                (root / "new_public_literal_antigravity.txt").read_bytes(),
                new_payload,
            )

    def test_production_status_rejects_unsafe_paths_observationally(self) -> None:
        """Status must reject linked or mistyped artifact namespaces without writes."""

        cases = (
            ("root-directory-symlink", "root-directory-symlink"),
            ("cache-directory-symlink", "cache-directory-symlink"),
            ("evaluation-directory-symlink", "evaluation-directory-symlink"),
            ("report-directory-symlink", "report-directory-symlink"),
            ("evaluation-leaf-symlink", "evaluation-leaf-symlink"),
            ("report-marker-symlink", "report-marker-symlink"),
            ("progress-leaf-symlink", "progress-leaf-symlink"),
            ("root-file", "root-file"),
            ("cache-root-file", "cache-root-file"),
            ("evaluation-root-file", "evaluation-root-file"),
            ("report-root-file", "report-root-file"),
            ("progress-leaf-directory", "progress-leaf-directory"),
        )
        for index, (name, kind) in enumerate(cases):
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"exp471_status_unsafe_{index}_antigravity_"
            ) as directory:
                base = Path(directory).resolve()
                root = base / "root_antigravity"
                root.mkdir()
                score = root / "score_cache_antigravity"
                evaluation_root = root / "evaluation_antigravity"
                report_root = root / "report_antigravity"
                foreign = base / "foreign_antigravity"
                foreign.mkdir()
                foreign_leaf = foreign / "foreign_literal_antigravity.json"
                foreign_leaf.write_bytes(b"{}\n")
                if kind == "root-directory-symlink":
                    root.rmdir()
                    root.symlink_to(foreign, target_is_directory=True)
                elif kind == "cache-directory-symlink":
                    score.symlink_to(foreign, target_is_directory=True)
                elif kind == "evaluation-directory-symlink":
                    (foreign / "evaluation_antigravity.json").write_bytes(b"{}\n")
                    evaluation_root.symlink_to(foreign, target_is_directory=True)
                elif kind == "report-directory-symlink":
                    (
                        foreign / ".exp471_report_complete_antigravity.json"
                    ).write_bytes(b"{}\n")
                    report_root.symlink_to(foreign, target_is_directory=True)
                elif kind == "evaluation-leaf-symlink":
                    evaluation_root.mkdir()
                    (evaluation_root / "evaluation_antigravity.json").symlink_to(
                        foreign_leaf
                    )
                elif kind == "report-marker-symlink":
                    report_root.mkdir()
                    (
                        report_root / ".exp471_report_complete_antigravity.json"
                    ).symlink_to(foreign_leaf)
                elif kind == "progress-leaf-symlink":
                    (root / cli._PROGRESS_NAME_ANTIGRAVITY).symlink_to(foreign_leaf)
                elif kind == "root-file":
                    root.rmdir()
                    root.write_bytes(b"foreign-public-literal\n")
                elif kind == "cache-root-file":
                    score.write_bytes(b"foreign-public-literal\n")
                elif kind == "evaluation-root-file":
                    evaluation_root.write_bytes(b"foreign-public-literal\n")
                elif kind == "report-root-file":
                    report_root.write_bytes(b"foreign-public-literal\n")
                else:
                    (root / cli._PROGRESS_NAME_ANTIGRAVITY).mkdir()
                before = {
                    path.relative_to(base).as_posix(): (
                        path.lstat().st_mode,
                        path.read_bytes() if path.is_file() and not path.is_symlink() else None,
                    )
                    for path in base.rglob("*")
                }
                stdout, stderr = io.StringIO(), io.StringIO()
                with ExitStack() as stack:
                    for attribute, value in (
                        ("EXP471_ROOT_ANTIGRAVITY", root),
                        ("EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", score),
                        ("EXP471_EVALUATION_ROOT_ANTIGRAVITY", evaluation_root),
                        ("EXP471_REPORT_ROOT_ANTIGRAVITY", report_root),
                    ):
                        stack.enter_context(mock.patch.object(cli, attribute, value))
                    result = main_antigravity(
                        ["status"], stdout=stdout, stderr=stderr
                    )
                after = {
                    path.relative_to(base).as_posix(): (
                        path.lstat().st_mode,
                        path.read_bytes() if path.is_file() and not path.is_symlink() else None,
                    )
                    for path in base.rglob("*")
                }
                self.assertEqual(result, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(
                    json.loads(stderr.getvalue())["exception_type"], "ValueError"
                )
                self.assertEqual(after, before)

    def test_codemap_contains_every_tracked_exp471_operator_entrypoint(self) -> None:
        """A tracked Exp471 launcher must be fingerprinted, mapped, and linked to CLI."""

        repository = Path(__file__).resolve().parents[3]
        generator_path = repository / "scratch/generate_exp471_codemap_antigravity.py"
        generator = runpy.run_path(str(generator_path))
        paths = generator["tracked_existing_paths_antigravity"](repository)
        relative = {
            path.relative_to(repository).as_posix()
            for path in paths
        }
        tracked = subprocess.run(
            (
                "git",
                "-C",
                str(repository),
                "ls-files",
                "scratch/run_exp471*_antigravity.py",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertTrue(tracked)
        self.assertTrue(set(tracked).issubset(relative))
        callers = generator["ast_callers_antigravity"](paths)
        payload = generator["build_codemap_payload_antigravity"](paths, callers)
        launcher = Path(exp471_launcher.__file__).relative_to(repository).as_posix()
        cli_path = (
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "cli_antigravity.py"
        )
        nodes = {node["path"]: node for node in payload["nodes"]}
        self.assertIn(launcher, nodes)
        self.assertTrue(nodes[launcher]["effects"])
        self.assertIn(
            Path(__file__).relative_to(repository).as_posix(),
            nodes[launcher]["tests"],
        )
        self.assertIn(
            {"caller": launcher, "callee": cli_path, "kind": "ast_import"},
            payload["edges"],
        )
        self.assertTrue(
            any(
                flow.get("from") == launcher and flow.get("to") == "paper report manifest"
                for flow in payload["flows"]
            )
        )

    def test_progress_rejects_noncanonical_foreign_and_nonfinite_history(self) -> None:
        """Status must never return malformed history or an arbitrary secret field."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_schema_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            path = root / cli._PROGRESS_NAME_ANTIGRAVITY
            artifact = root / "cache_antigravity"
            invalid = (
                b'{ "schema": "foreign", "secret": "/public/literal" }\n',
                b'{"schema":"foreign","secret":"public-literal-antigravity"}\n',
                b'{"schema":"exp471-progress-antigravity.v1","eta_seconds":NaN}\n',
                b'{"schema":"exp471-progress-antigravity.v1"',
            )
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                for raw in invalid:
                    with self.subTest(raw=raw):
                        path.write_bytes(raw)
                        with self.assertRaisesRegex(
                            ValueError,
                            "progress.*malformed|noncanonical|schema|truncated",
                        ):
                            cli._latest_progress_antigravity()
                path.unlink()
                callback = cli._progress_callback_antigravity(
                    "score-cache", artifact
                )
                callback(
                    {
                        "durable_completed": 1,
                        "total": 11_170,
                        "status": "durable",
                    }
                )
                value = json.loads(path.read_text(encoding="utf-8"))
                value["secret"] = "public-literal-secret-antigravity"
                path.write_bytes(canonical_json_bytes_exp471_antigravity(value))
                with self.assertRaisesRegex(ValueError, "progress.*schema"):
                    cli._latest_progress_antigravity()

    def test_progress_rejects_hardlinks_symlinks_and_path_swap(self) -> None:
        """Append/read must bind the single-link log inode that occupies its name."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_identity_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            path = root / cli._PROGRESS_NAME_ANTIGRAVITY
            target = root / "foreign_progress_target_antigravity.jsonl"
            artifact = root / "cache_antigravity"
            target.write_bytes(b"foreign-public-literal-antigravity\n")
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                os.link(target, path)
                with self.assertRaisesRegex(ValueError, "progress log.*unsafe"):
                    cli._latest_progress_antigravity()
                callback = cli._progress_callback_antigravity(
                    "score-cache", artifact
                )
                with self.assertRaisesRegex(ValueError, "progress log.*unsafe"):
                    callback(
                        {
                            "durable_completed": 1,
                            "total": 11_170,
                            "status": "durable",
                        }
                    )
                path.unlink()
                path.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "progress log.*unsafe"):
                    cli._latest_progress_antigravity()
                path.unlink()

                callback = cli._progress_callback_antigravity(
                    "score-cache", artifact
                )
                original_flock = cli.fcntl.flock
                saved = root / "owned_progress_saved_antigravity.jsonl"
                swapped = False

                def swap_after_open(descriptor, operation):
                    nonlocal swapped
                    result = original_flock(descriptor, operation)
                    if operation == cli.fcntl.LOCK_EX and not swapped:
                        swapped = True
                        path.rename(saved)
                        path.write_bytes(b"foreign-public-literal-antigravity\n")
                    return result

                with mock.patch.object(
                    cli.fcntl, "flock", side_effect=swap_after_open
                ):
                    with self.assertRaisesRegex(ValueError, "progress log.*changed"):
                        callback(
                            {
                                "durable_completed": 1,
                                "total": 11_170,
                                "status": "durable",
                            }
                        )
                self.assertEqual(
                    path.read_bytes(), b"foreign-public-literal-antigravity\n"
                )

    def test_progress_preserves_failure_attempt_history_and_unknown_eta(self) -> None:
        """A new invocation must continue cumulative attempts/failures truthfully."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_history_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            artifact = root / "cache_antigravity"
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                first = cli._progress_callback_antigravity("score-cache", artifact)
                first(
                    {
                        "durable_completed": 0,
                        "total": 11_170,
                        "status": "failed",
                    }
                )
                first_event = cli._latest_progress_antigravity()
                self.assertEqual(first_event["attempts"], 1)
                self.assertEqual(first_event["failures"], 1)
                self.assertIsNone(first_event["eta_seconds"])

                resumed = cli._progress_callback_antigravity("score-cache", artifact)
                resumed(
                    {
                        "durable_completed": 1,
                        "total": 11_170,
                        "status": "durable",
                    }
                )
                latest = cli._latest_progress_antigravity()
                self.assertEqual(latest["attempts"], 2)
                self.assertEqual(latest["failures"], 1)
                self.assertEqual(latest["remaining"], 11_169)
                self.assertNotIn("secret", latest)

    def test_concurrent_progress_appends_are_canonical_and_cumulative(self) -> None:
        """Concurrent callbacks must produce whole ordered records without lost attempts."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_concurrent_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            artifact = root / "cache_antigravity"
            with mock.patch.object(
                cli, "EXP471_ROOT_ANTIGRAVITY", root
            ), mock.patch.object(
                cli, "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY", artifact
            ):
                callbacks = [
                    cli._progress_callback_antigravity("score-cache", artifact)
                    for _ in range(8)
                ]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [
                        pool.submit(
                            callback,
                            {
                                "durable_completed": 1,
                                "total": 11_170,
                                "status": "durable",
                            },
                        )
                        for callback in callbacks
                    ]
                    for future in futures:
                        future.result(timeout=10)
                path = root / cli._PROGRESS_NAME_ANTIGRAVITY
                records = [json.loads(line) for line in path.read_bytes().splitlines()]
                self.assertEqual(len(records), 8)
                self.assertEqual([row["attempts"] for row in records], list(range(1, 9)))
                self.assertTrue(
                    all(
                        canonical_json_bytes_exp471_antigravity(row).rstrip(b"\n")
                        == line
                        for row, line in zip(records, path.read_bytes().splitlines())
                    )
                )

    def test_complete_cache_score_reader_skips_source_score_preparation(self) -> None:
        """Reopening terminal cache must not rebuild Exp467 scores or calibration input."""

        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            cli_antigravity as cli,
        )

        session = object()
        reopened = {"schema": "public-reopened-score-input-antigravity.v1"}
        with mock.patch.object(cli.os.path, "lexists", return_value=True), \
             mock.patch.object(
                 cli,
                 "_reopened_score_bundle_antigravity",
                 return_value=reopened,
                 create=True,
             ) as reopen, \
             mock.patch.object(
                 cli,
                 "_production_score_bundle_antigravity",
                 side_effect=AssertionError("source score preparation repeated"),
             ) as prepare:
            self.assertIs(cli._score_input_antigravity(session), reopened)
        self.assertEqual(reopen.call_count, 1)
        self.assertEqual(prepare.call_count, 0)

    def test_production_scores_use_only_issued_captured_values_api(self) -> None:
        """A live-worktree loader must never enter captured score composition."""

        from paper.equipment_anomaly_detection import contracts_paper, data_paper
        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            source_authority_antigravity as source_authority,
        )
        from experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_source_authority_antigravity import (
            SyntheticSourceAntigravity,
            _synthetic_constants_antigravity,
        )

        with tempfile.TemporaryDirectory(
            prefix="exp471_cli_captured_values_antigravity_"
        ) as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            with _synthetic_constants_antigravity(fixture):
                session = source_authority.open_exp467_source_session_antigravity(
                    fixture.repository, fixture.source_root
                )

            class ForeignLiveConfigAntigravity:
                database_path = str(
                    fixture.root / "foreign_live_database_antigravity.sqlite"
                )
                database_sha256 = session.authority.database_sha256

            def foreign_live_values_antigravity(*_args, **_kwargs):
                raise AssertionError("live values API was imported")

            observed: Exception | None = None
            with mock.patch.object(
                contracts_paper, "FrozenPaperConfig", ForeignLiveConfigAntigravity
            ), mock.patch.object(
                data_paper,
                "bind_preverified_database_read_only_paper",
                foreign_live_values_antigravity,
            ), mock.patch.object(
                data_paper, "load_dataset_values", foreign_live_values_antigravity
            ), mock.patch.object(
                cli,
                "_preflight_production_baselines_antigravity",
                return_value=_issued_preflight_context_antigravity(session),
            ):
                try:
                    cli._production_score_bundle_antigravity(session)
                except Exception as error:
                    observed = error
            self.assertIsInstance(observed, RuntimeError)
            self.assertEqual(str(observed), "captured values reached score closure")
            calls = session.captured_values_api_antigravity.load_dataset_values_antigravity.__globals__[
                "VALUES_CALLS"
            ]
            self.assertEqual([row[0] for row in calls], ["bind", "load"])

    def test_production_scores_reject_replaced_captured_loader_before_call(self) -> None:
        """A foreign callable cannot self-describe its way into score composition."""

        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            source_authority_antigravity as source_authority,
        )
        from experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_source_authority_antigravity import (
            SyntheticSourceAntigravity,
            _synthetic_constants_antigravity,
        )

        with tempfile.TemporaryDirectory(
            prefix="exp471_cli_foreign_loader_antigravity_"
        ) as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            with _synthetic_constants_antigravity(fixture):
                session = source_authority.open_exp467_source_session_antigravity(
                    fixture.repository, fixture.source_root
                )
            values = session.captured_values_api_antigravity
            captured_loader = values.load_dataset_values_antigravity
            foreign_calls: list[str] = []

            def foreign_loader(*args, **kwargs):
                foreign_calls.append("called")
                return captured_loader(*args, **kwargs)

            foreign_loader.__module__ = captured_loader.__module__
            object.__setattr__(
                values, "load_dataset_values_antigravity", foreign_loader
            )
            observed: Exception | None = None
            with mock.patch.object(
                cli,
                "_preflight_production_baselines_antigravity",
                return_value=_issued_preflight_context_antigravity(session),
            ):
                try:
                    cli._production_score_bundle_antigravity(session)
                except Exception as error:
                    observed = error
            self.assertIsInstance(observed, ValueError)
            self.assertRegex(str(observed), "captured values API")
            self.assertEqual(foreign_calls, [])

    def test_production_scores_reject_in_place_captured_execution_mutations(self) -> None:
        """Same-object code, defaults, and helper mutations fail before execution."""

        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            source_authority_antigravity as source_authority,
        )
        from experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_source_authority_antigravity import (
            SyntheticSourceAntigravity,
            _foreign_dataset_helper_antigravity,
            _foreign_dataset_helper_code_antigravity,
            _foreign_loader_code_antigravity,
            _foreign_sql_observer_antigravity,
            _synthetic_constants_antigravity,
        )

        mutations = (
            "loader_code",
            "helper_code",
            "loader_kwdefaults",
            "helper_global",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"exp471_execution_seal_{mutation}_antigravity_"
            ) as temporary:
                fixture = SyntheticSourceAntigravity(temporary)
                with _synthetic_constants_antigravity(fixture):
                    session = source_authority.open_exp467_source_session_antigravity(
                        fixture.repository, fixture.source_root
                    )
                values = session.captured_values_api_antigravity
                loader = values.load_dataset_values_antigravity
                loader_globals = loader.__globals__
                helper = loader_globals["_raw_dataset_antigravity"]
                values_calls = loader_globals["VALUES_CALLS"]
                restore = None
                if mutation == "loader_code":
                    original = loader.__code__
                    loader.__code__ = _foreign_loader_code_antigravity.__code__
                    restore = lambda: setattr(loader, "__code__", original)
                elif mutation == "helper_code":
                    original = helper.__code__
                    helper.__code__ = _foreign_dataset_helper_code_antigravity.__code__
                    restore = lambda: setattr(helper, "__code__", original)
                elif mutation == "loader_kwdefaults":
                    original = dict(loader.__kwdefaults__)
                    loader.__kwdefaults__["sql_observer"] = (
                        _foreign_sql_observer_antigravity
                    )

                    def restore_kwdefaults_antigravity():
                        loader.__kwdefaults__.clear()
                        loader.__kwdefaults__.update(original)

                    restore = restore_kwdefaults_antigravity
                else:
                    original = loader_globals["_raw_dataset_antigravity"]
                    loader_globals["_raw_dataset_antigravity"] = (
                        _foreign_dataset_helper_antigravity
                    )
                    restore = lambda: loader_globals.__setitem__(
                        "_raw_dataset_antigravity", original
                    )
                observed: Exception | None = None
                with mock.patch.object(
                    cli,
                    "_preflight_production_baselines_antigravity",
                    return_value=_issued_preflight_context_antigravity(session),
                ):
                    try:
                        cli._production_score_bundle_antigravity(session)
                    except Exception as error:
                        observed = error
                    finally:
                        restore()
                self.assertIsInstance(observed, ValueError)
                self.assertIn("execution seal", str(observed))
                self.assertEqual(values_calls, [])

    def test_report_lifecycle_order_and_exact_second_call_skip_counts(self) -> None:
        """Resume must not repeat preparation, labels, metrics, or rendering."""

        runtime = _LifecycleRuntimeAntigravity()
        for _ in range(2):
            stdout, stderr = io.StringIO(), io.StringIO()
            self.assertEqual(
                main_antigravity(
                    ["report"],
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                ),
                0,
            )
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue())["command"], "report")
        self.assertEqual(
            runtime.calls,
            [
                "verify-source",
                "open-cache",
                "prepare",
                "open-evaluation",
                "evaluate",
                "report",
                "render",
                "verify-source",
                "open-cache",
                "open-evaluation",
                "report",
            ],
        )
        self.assertEqual(runtime.calls.count("prepare"), 1)
        self.assertEqual(runtime.calls.count("evaluate"), 1)
        self.assertEqual(runtime.calls.count("render"), 1)

    def test_fixed_commands_route_literal_json_and_status_starts_no_work(self) -> None:
        """Status is observational and every accepted command is a fixed literal."""

        for command in (
            "verify-source",
            "benchmark-prepare",
            "prepare",
            "status",
        ):
            runtime = _LifecycleRuntimeAntigravity()
            if command == "prepare":
                runtime.cache_complete = False
            stdout, stderr = io.StringIO(), io.StringIO()
            self.assertEqual(
                main_antigravity(
                    [command],
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                ),
                0,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["command"], command)
            self.assertEqual(stderr.getvalue(), "")
            if command == "status":
                self.assertEqual(runtime.calls, ["status"])

        runtime = _LifecycleRuntimeAntigravity()
        stdout, stderr = io.StringIO(), io.StringIO()
        self.assertEqual(
            main_antigravity(
                ["report", "--report-root", "/foreign"],
                runtime=runtime,
                stdout=stdout,
                stderr=stderr,
            ),
            2,
        )
        self.assertEqual(stdout.getvalue(), "")
        error = json.loads(stderr.getvalue())
        self.assertEqual(error["error"], "invalid_arguments")
        self.assertNotIn("/foreign", stderr.getvalue())
        self.assertEqual(runtime.calls, [])

    def test_runtime_error_is_one_concise_json_without_traceback_or_paths(self) -> None:
        """Normal CLI errors must reveal neither traceback nor sensitive details."""

        runtime = _LifecycleRuntimeAntigravity()
        runtime.verify_source_antigravity = mock.Mock(
            side_effect=ValueError("private /secret/path public literal failure")
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        self.assertEqual(
            main_antigravity(
                ["verify-source"],
                runtime=runtime,
                stdout=stdout,
                stderr=stderr,
            ),
            2,
        )
        self.assertEqual(stdout.getvalue(), "")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        error = json.loads(lines[0])
        self.assertEqual(error["error"], "command_failed")
        self.assertEqual(error["command"], "verify-source")
        self.assertEqual(error["exception_type"], "ValueError")
        self.assertNotIn("secret", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
