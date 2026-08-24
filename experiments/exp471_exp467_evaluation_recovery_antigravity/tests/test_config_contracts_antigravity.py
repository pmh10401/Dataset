"""Behavior tests for the frozen Exp471 configuration and strict contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
import unittest

from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    ACCELERATION_ASSESSMENT_ANTIGRAVITY,
    CPU_WORKERS_ANTIGRAVITY,
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
    EXP467_SOURCE_ROOT_ANTIGRAVITY,
    EXP471_EVALUATION_ROOT_ANTIGRAVITY,
    EXP471_REPORT_ROOT_ANTIGRAVITY,
    EXP471_ROOT_ANTIGRAVITY,
    EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
    INNER_NUMERICAL_THREADS_ANTIGRAVITY,
    LOGICAL_CPU_COUNT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
    OFFICIAL_PYTHON_ANTIGRAVITY,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    Exp471EvaluationArtifactAntigravity,
    Exp471ReportAuthorityAntigravity,
    Exp471ScoreCacheManifestAntigravity,
    Exp471ScoreShardAntigravity,
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from scratch.generate_exp471_codemap_antigravity import (
    REPOSITORY_ROOT,
    ast_callers_antigravity,
    build_codemap_payload_antigravity,
    tracked_existing_paths_antigravity,
)


EXPECTED_METHOD_ORDER_ANTIGRAVITY = (
    "A",
    "B",
    "C",
    "D",
    "knn_distance",
    "lof_novelty",
    "ocsvm_rbf",
    "isolation_forest",
    "deep_svdd_1d",
    "drocc_1d",
)
SOURCE_FILES_ANTIGRAVITY = {
    "module_antigravity.py": "1" * 64,
    "nested/data_antigravity.json": "2" * 64,
}
SOURCE_TREE_SHA256_ANTIGRAVITY = (
    "c2c84b5c21aef4f8d0a64612634204d79ca1dc53343e398861dcbcc48a5fb387"
)


def source_authority_antigravity(
    *,
    source_file_sha256: object = SOURCE_FILES_ANTIGRAVITY,
    source_tree_sha256: str = SOURCE_TREE_SHA256_ANTIGRAVITY,
) -> Exp471SourceAuthorityAntigravity:
    return Exp471SourceAuthorityAntigravity(
        source_commit="1e616731a88da9bc0268823e1d2cef2a53c1054f",
        source_context_sha256=(
            "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af"
        ),
        inventory_sha256="3" * 64,
        catalog_sha256="4" * 64,
        database_sha256="5" * 64,
        source_file_sha256=source_file_sha256,  # type: ignore[arg-type]
        source_tree_sha256=source_tree_sha256,
        condition_manifest_sha256={
            "A": "6" * 64,
            "B": "7" * 64,
            "C": "8" * 64,
            "D": "9" * 64,
        },
        total_successes=4468,
        imported_total=4234,
        native_total=234,
        authority_sha256="a" * 64,
    )


def score_shard_antigravity(
    *,
    method: str = "A",
    dataset: str = "dataset_1",
    test_count: int = 2,
    raw_scores: object = (0.25, 0.75),
    micro_scores: object = (0.5, 1.0),
) -> Exp471ScoreShardAntigravity:
    return Exp471ScoreShardAntigravity(
        source_authority_sha256="a" * 64,
        cache_context_sha256="b" * 64,
        method=method,
        dataset=dataset,
        test_count=test_count,
        raw_scores=raw_scores,  # type: ignore[arg-type]
        micro_scores=micro_scores,  # type: ignore[arg-type]
        source_manifest_sha256="c" * 64,
        payload_sha256="d" * 64,
        envelope_sha256="e" * 64,
    )


def cache_manifest_antigravity(
    *,
    method_order: object = EXPECTED_METHOD_ORDER_ANTIGRAVITY,
    dataset_order: object = ("dataset_1", "dataset_2"),
) -> Exp471ScoreCacheManifestAntigravity:
    shard_sha256 = {
        f"{method}:{dataset}": "f" * 64
        for method in EXPECTED_METHOD_ORDER_ANTIGRAVITY
        for dataset in ("dataset_1", "dataset_2")
    }
    return Exp471ScoreCacheManifestAntigravity(
        source_authority_sha256="a" * 64,
        cache_context_sha256="b" * 64,
        method_order=method_order,  # type: ignore[arg-type]
        dataset_order=dataset_order,  # type: ignore[arg-type]
        shard_sha256=shard_sha256,
        total_shards=20,
        manifest_sha256="c" * 64,
    )


class Exp471ConfigTestsAntigravity(unittest.TestCase):
    def test_fixed_source_identity_roots_and_method_order_are_exact(self) -> None:
        """A changed source or method identity must fail the frozen recovery boundary."""
        self.assertEqual(
            EXP467_SOURCE_COMMIT_ANTIGRAVITY,
            "1e616731a88da9bc0268823e1d2cef2a53c1054f",
        )
        self.assertEqual(
            EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
            "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af",
        )
        self.assertEqual(
            EXP467_SOURCE_ROOT_ANTIGRAVITY,
            Path(
                "/Users/mac/Documents/Dataset/"
                "exp467_numba_calibration_resume_official_antigravity_20260823"
            ),
        )
        self.assertEqual(
            EXP471_ROOT_ANTIGRAVITY.name,
            "exp471_exp467_evaluation_recovery_antigravity_20260824",
        )
        self.assertEqual(tuple(METHOD_ORDER_ANTIGRAVITY), EXPECTED_METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(
            OFFICIAL_PYTHON_ANTIGRAVITY,
            Path(
                "/Users/mac/Documents/Dataset/venvs/"
                "detection-coverage-antigravity-py314/bin/python"
            ),
        )

    def test_worker_policy_and_target_roots_are_absolute_and_contained(self) -> None:
        """Worker drift or an escaped target path must not enter Exp471 provenance."""
        self.assertEqual(LOGICAL_CPU_COUNT_ANTIGRAVITY, os.cpu_count())
        self.assertEqual(CPU_WORKERS_ANTIGRAVITY, max(1, os.cpu_count() - 1))
        self.assertEqual(INNER_NUMERICAL_THREADS_ANTIGRAVITY, 1)
        self.assertTrue(EXP471_ROOT_ANTIGRAVITY.is_absolute())
        self.assertEqual(
            (
                EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
                EXP471_EVALUATION_ROOT_ANTIGRAVITY,
                EXP471_REPORT_ROOT_ANTIGRAVITY,
            ),
            (
                EXP471_ROOT_ANTIGRAVITY / "score_cache_antigravity",
                EXP471_ROOT_ANTIGRAVITY / "evaluation_antigravity",
                EXP471_ROOT_ANTIGRAVITY / "report_antigravity",
            ),
        )
        for child in (
            EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
            EXP471_EVALUATION_ROOT_ANTIGRAVITY,
            EXP471_REPORT_ROOT_ANTIGRAVITY,
        ):
            self.assertTrue(child.is_absolute())
            self.assertEqual(len(child.relative_to(EXP471_ROOT_ANTIGRAVITY).parts), 1)

    def test_acceleration_assessment_records_all_backends_without_promotion(self) -> None:
        """Availability evidence must never imply an unvalidated accelerator run."""
        assessment = ACCELERATION_ASSESSMENT_ANTIGRAVITY
        self.assertIsInstance(assessment, type(MappingProxyType({})))
        self.assertEqual(
            assessment["schema"],
            "exp471-acceleration-assessment-antigravity.v1",
        )
        for key in (
            "pytorch_available",
            "mlx_available",
            "mlx_metal_available",
            "mps_built",
            "mps_available",
        ):
            self.assertIs(type(assessment[key]), bool, key)
        for key in ("pytorch_version", "mlx_version", "mlx_metal_version"):
            self.assertIs(type(assessment[key]), str, key)
            self.assertTrue(assessment[key], key)
        self.assertIs(assessment["tensor_eligible"], False)
        self.assertIs(assessment["exact_mlx_expression_available"], False)
        self.assertEqual(assessment["selected_backend"], "numba_cpu")
        self.assertEqual(assessment["canonical_cpu_fallback"], "numba_float64_cpu")
        self.assertIs(assessment["mps_heavy_process_launched"], False)
        with self.assertRaises(TypeError):
            assessment["selected_backend"] = "mlx"  # type: ignore[index]

    def test_codemap_scope_metadata_tracks_actual_production_nodes(self) -> None:
        """A nonempty indexed production closure must not retain no-modules metadata."""
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        payload = build_codemap_payload_antigravity(
            paths, ast_callers_antigravity(paths)
        )
        scope = payload["scope"]
        production_count = scope["production_node_count"]
        expected_policy = (
            "no_production_modules_yet"
            if production_count == 0
            else "index_tracked_exp471_production_closure"
        )
        self.assertEqual(
            (
                production_count,
                scope["governance_fingerprint_evidence"][
                    "exp471_production_modules_exist"
                ],
                scope["external_closure_policy"],
            ),
            (production_count, production_count > 0, expected_policy),
        )

        initial_paths = tuple(
            path
            for path in paths
            if not path.relative_to(REPOSITORY_ROOT).as_posix().startswith(
                "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            )
        )
        initial_payload = build_codemap_payload_antigravity(
            initial_paths, ast_callers_antigravity(initial_paths)
        )
        initial_scope = initial_payload["scope"]
        self.assertEqual(
            (
                initial_scope["production_node_count"],
                initial_scope["governance_fingerprint_evidence"][
                    "exp471_production_modules_exist"
                ],
                initial_scope["external_closure_policy"],
            ),
            (0, False, "no_production_modules_yet"),
        )


class Exp471CanonicalJsonTestsAntigravity(unittest.TestCase):
    def test_canonical_json_is_deterministic_utf8_and_recursively_plain(self) -> None:
        """Mapping order and immutable containers must yield one literal byte stream."""
        value = MappingProxyType({"z": ("한글", True, None), "a": 1.25})
        self.assertEqual(
            canonical_json_bytes_exp471_antigravity(value),
            b'{"a":1.25,"z":["\xed\x95\x9c\xea\xb8\x80",true,null]}\n',
        )
        self.assertEqual(
            canonical_json_bytes_exp471_antigravity({"a": 1, "b": 2}),
            canonical_json_bytes_exp471_antigravity({"b": 2, "a": 1}),
        )

    def test_canonical_json_rejects_every_nested_nonfinite_float(self) -> None:
        """NaN or infinity at any depth must not reach a generic artifact digest."""
        cases = (
            float("nan"),
            {"nested": [0.0, float("inf")]},
            {"nested": MappingProxyType({"value": float("-inf")})},
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite canonical JSON"):
                    canonical_json_bytes_exp471_antigravity(value)

    def test_canonical_json_rejects_non_string_keys_and_non_json_objects(self) -> None:
        """Implicit key coercion and arbitrary repr serialization must remain impossible."""
        for value in ({1: "value"}, {"value": object()}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "canonical JSON"):
                    canonical_json_bytes_exp471_antigravity(value)


class Exp471ContractSchemaTestsAntigravity(unittest.TestCase):
    def test_dataclass_fields_are_the_exact_frozen_schema(self) -> None:
        """Adding, removing, or reordering an authority field changes persisted bytes."""
        expected = {
            Exp471SourceAuthorityAntigravity: (
                "source_commit",
                "source_context_sha256",
                "inventory_sha256",
                "catalog_sha256",
                "database_sha256",
                "source_file_sha256",
                "source_tree_sha256",
                "condition_manifest_sha256",
                "total_successes",
                "imported_total",
                "native_total",
                "authority_sha256",
            ),
            Exp471ScoreShardAntigravity: (
                "source_authority_sha256",
                "cache_context_sha256",
                "method",
                "dataset",
                "test_count",
                "raw_scores",
                "micro_scores",
                "source_manifest_sha256",
                "payload_sha256",
                "envelope_sha256",
            ),
            Exp471ScoreCacheManifestAntigravity: (
                "source_authority_sha256",
                "cache_context_sha256",
                "method_order",
                "dataset_order",
                "shard_sha256",
                "total_shards",
                "manifest_sha256",
            ),
            Exp471EvaluationArtifactAntigravity: (
                "source_authority_sha256",
                "cache_manifest_sha256",
                "code_closure_sha256",
                "method_order",
                "dataset_order",
                "payload",
                "evaluation_sha256",
            ),
            Exp471ReportAuthorityAntigravity: (
                "source_authority_sha256",
                "cache_manifest_sha256",
                "evaluation_sha256",
                "report_file_sha256",
                "report_manifest_sha256",
            ),
        }
        for contract, names in expected.items():
            with self.subTest(contract=contract.__name__):
                self.assertEqual(tuple(field.name for field in fields(contract)), names)

    def test_source_authority_freezes_complete_contained_source_maps(self) -> None:
        """Caller mutation or an escaped source path must not alter source authority."""
        source_files = dict(SOURCE_FILES_ANTIGRAVITY)
        conditions = {
            "A": "6" * 64,
            "B": "7" * 64,
            "C": "8" * 64,
            "D": "9" * 64,
        }
        authority = Exp471SourceAuthorityAntigravity(
            source_commit="1e616731a88da9bc0268823e1d2cef2a53c1054f",
            source_context_sha256=(
                "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af"
            ),
            inventory_sha256="3" * 64,
            catalog_sha256="4" * 64,
            database_sha256="5" * 64,
            source_file_sha256=source_files,
            source_tree_sha256=SOURCE_TREE_SHA256_ANTIGRAVITY,
            condition_manifest_sha256=conditions,
            total_successes=4468,
            imported_total=4234,
            native_total=234,
            authority_sha256="a" * 64,
        )
        source_files["late_antigravity.py"] = "b" * 64
        conditions["A"] = "c" * 64
        self.assertEqual(dict(authority.source_file_sha256), SOURCE_FILES_ANTIGRAVITY)
        self.assertEqual(authority.condition_manifest_sha256["A"], "6" * 64)
        self.assertIsInstance(authority.source_file_sha256, type(MappingProxyType({})))
        with self.assertRaises(TypeError):
            authority.source_file_sha256["module_antigravity.py"] = "0" * 64  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            authority.total_successes = 0  # type: ignore[misc]

        for path in (
            "/absolute_antigravity.py",
            "../escaped_antigravity.py",
            "nested/../escaped_antigravity.py",
            "nested//empty_antigravity.py",
            "nested\\windows_antigravity.py",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "contained relative path"):
                    source_authority_antigravity(
                        source_file_sha256={path: "1" * 64},
                        source_tree_sha256="0" * 64,
                    )

    def test_source_tree_digest_and_fixed_population_are_enforced(self) -> None:
        """A missing source byte or changed execution-origin count must invalidate authority."""
        authority = source_authority_antigravity()
        self.assertEqual(authority.source_tree_sha256, SOURCE_TREE_SHA256_ANTIGRAVITY)
        with self.assertRaisesRegex(ValueError, "source_tree_sha256"):
            source_authority_antigravity(source_tree_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "source population"):
            replace(authority, total_successes=4467)
        with self.assertRaisesRegex(ValueError, "source population"):
            replace(authority, imported_total=4233, native_total=235)

    def test_all_digest_fields_require_exact_lowercase_sha256(self) -> None:
        """Short, uppercase, or non-hex terminal bindings must fail construction."""
        authority = source_authority_antigravity()
        shard = score_shard_antigravity()
        manifest = cache_manifest_antigravity()
        evaluation = Exp471EvaluationArtifactAntigravity(
            source_authority_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            code_closure_sha256="c" * 64,
            method_order=EXPECTED_METHOD_ORDER_ANTIGRAVITY,
            dataset_order=("dataset_1",),
            payload={"metric": 0.5},
            evaluation_sha256="d" * 64,
        )
        report = Exp471ReportAuthorityAntigravity(
            source_authority_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            evaluation_sha256="d" * 64,
            report_file_sha256={"summary_antigravity.json": "e" * 64},
            report_manifest_sha256="f" * 64,
        )
        mutations = (
            lambda: replace(authority, inventory_sha256="A" * 64),
            lambda: replace(shard, cache_context_sha256="b" * 63),
            lambda: replace(manifest, manifest_sha256="g" * 64),
            lambda: replace(evaluation, code_closure_sha256="C" * 64),
            lambda: replace(report, report_manifest_sha256="f" * 65),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                    mutate()

    def test_score_shard_freezes_finite_vectors_and_exact_row_identity(self) -> None:
        """A mutated, non-finite, or mis-sized score vector must not become a shard."""
        raw_scores = [0.25, 0.75]
        micro_scores = [0.5, 1.0]
        shard = score_shard_antigravity(
            raw_scores=raw_scores,
            micro_scores=micro_scores,
        )
        raw_scores[0] = 999.0
        micro_scores.append(2.0)
        self.assertEqual(shard.raw_scores, (0.25, 0.75))
        self.assertEqual(shard.micro_scores, (0.5, 1.0))
        self.assertEqual(
            json.loads(canonical_json_bytes_exp471_antigravity(shard)),
            {
                "cache_context_sha256": "b" * 64,
                "dataset": "dataset_1",
                "envelope_sha256": "e" * 64,
                "method": "A",
                "micro_scores": [0.5, 1.0],
                "payload_sha256": "d" * 64,
                "raw_scores": [0.25, 0.75],
                "source_authority_sha256": "a" * 64,
                "source_manifest_sha256": "c" * 64,
                "test_count": 2,
            },
        )
        for kwargs, message in (
            ({"method": "unknown"}, "method"),
            ({"dataset": ""}, "dataset"),
            ({"test_count": 1}, "test_count"),
            ({"raw_scores": (0.0, float("inf"))}, "finite"),
            ({"micro_scores": (0.0, float("nan"))}, "finite"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    score_shard_antigravity(**kwargs)

    def test_cache_manifest_rejects_duplicate_method_or_dataset_identities(self) -> None:
        """One logical shard identity must occur exactly once in complete coverage."""
        manifest = cache_manifest_antigravity()
        self.assertEqual(manifest.method_order, EXPECTED_METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(manifest.dataset_order, ("dataset_1", "dataset_2"))
        self.assertEqual(manifest.total_shards, 20)
        self.assertIsInstance(manifest.shard_sha256, type(MappingProxyType({})))
        with self.assertRaises(TypeError):
            manifest.shard_sha256["A:dataset_1"] = "0" * 64  # type: ignore[index]

        duplicate_methods = (
            "A",
            "A",
            "C",
            "D",
            "knn_distance",
            "lof_novelty",
            "ocsvm_rbf",
            "isolation_forest",
            "deep_svdd_1d",
            "drocc_1d",
        )
        with self.assertRaisesRegex(ValueError, "method_order"):
            cache_manifest_antigravity(method_order=duplicate_methods)
        with self.assertRaisesRegex(ValueError, "duplicate dataset"):
            cache_manifest_antigravity(dataset_order=("dataset_1", "dataset_1"))
        with self.assertRaisesRegex(ValueError, "total_shards"):
            replace(manifest, total_shards=19)

    def test_evaluation_payload_and_report_files_are_recursively_immutable(self) -> None:
        """Nested evaluation data and contained report hashes must resist caller mutation."""
        payload = {
            "metrics": {"A": {"ap": 0.5}},
            "curve": [0.0, 1.0],
        }
        evaluation = Exp471EvaluationArtifactAntigravity(
            source_authority_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            code_closure_sha256="c" * 64,
            method_order=list(EXPECTED_METHOD_ORDER_ANTIGRAVITY),
            dataset_order=["dataset_1"],
            payload=payload,
            evaluation_sha256="d" * 64,
        )
        payload["metrics"]["A"]["ap"] = 1.0  # type: ignore[index]
        payload["curve"].append(2.0)  # type: ignore[union-attr]
        self.assertEqual(evaluation.payload["metrics"]["A"]["ap"], 0.5)  # type: ignore[index]
        self.assertEqual(evaluation.payload["curve"], (0.0, 1.0))
        self.assertEqual(evaluation.method_order, EXPECTED_METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(evaluation.dataset_order, ("dataset_1",))
        with self.assertRaises(TypeError):
            evaluation.payload["metric"] = 1.0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(evaluation, payload={"metric": float("inf")})
        with self.assertRaisesRegex(ValueError, "duplicate dataset"):
            replace(evaluation, dataset_order=("dataset_1", "dataset_1"))

        report_files = {
            "tables/summary_antigravity.csv": "e" * 64,
            "figure_antigravity.png": "f" * 64,
        }
        report = Exp471ReportAuthorityAntigravity(
            source_authority_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            evaluation_sha256="d" * 64,
            report_file_sha256=report_files,
            report_manifest_sha256="c" * 64,
        )
        report_files["late_antigravity.txt"] = "0" * 64
        self.assertEqual(
            set(report.report_file_sha256),
            {"tables/summary_antigravity.csv", "figure_antigravity.png"},
        )
        with self.assertRaises(TypeError):
            report.report_file_sha256["figure_antigravity.png"] = "0" * 64  # type: ignore[index]
        for path in (
            "/absolute_antigravity.json",
            "../escaped_antigravity.json",
            "tables/../escaped_antigravity.json",
            "tables//empty_antigravity.json",
            "tables\\windows_antigravity.json",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "contained relative path"):
                    replace(report, report_file_sha256={path: "e" * 64})


if __name__ == "__main__":
    unittest.main()
