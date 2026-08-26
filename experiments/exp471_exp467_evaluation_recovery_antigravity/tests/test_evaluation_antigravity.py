"""Public-literal tests for Exp471 durable continuous-score evaluation."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    METHOD_ORDER,
    curve_tables_exp450_antigravity,
    metric_tables_exp450_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    cache_artifacts_antigravity as cache_artifacts,
    evaluation_antigravity as evaluation,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
    EXP467_NATIVE_TOTAL_ANTIGRAVITY,
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
    EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    Exp471EvaluationArtifactAntigravity,
    Exp471ScoreCacheManifestAntigravity,
    Exp471ScoreShardAntigravity,
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.source_authority_antigravity import (
    Exp471SourceSessionAntigravity,
)


_DATASET_ANTIGRAVITY = "public_literal_dataset"
_LABELS_ANTIGRAVITY = {_DATASET_ANTIGRAVITY: (0, 1)}


def _authority_antigravity() -> Exp471SourceAuthorityAntigravity:
    source_files = {
        "public_literal_source_antigravity.py": sha256(b"public literal source").hexdigest()
    }
    bare = {
        "source_commit": EXP467_SOURCE_COMMIT_ANTIGRAVITY,
        "source_context_sha256": EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
        "inventory_sha256": sha256(b"public literal inventory").hexdigest(),
        "catalog_sha256": sha256(b"public literal catalog").hexdigest(),
        "database_sha256": sha256(b"public literal database").hexdigest(),
        "source_file_sha256": source_files,
        "source_tree_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(source_files)
        ).hexdigest(),
        "condition_manifest_sha256": {
            method: sha256(f"public literal {method}".encode("ascii")).hexdigest()
            for method in METHOD_ORDER_ANTIGRAVITY[:4]
        },
        "total_successes": EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
        "imported_total": EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
        "native_total": EXP467_NATIVE_TOTAL_ANTIGRAVITY,
    }
    return Exp471SourceAuthorityAntigravity(
        **bare,
        authority_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    )


def _different_authority_antigravity() -> Exp471SourceAuthorityAntigravity:
    original = _authority_antigravity()
    bare = {
        "source_commit": original.source_commit,
        "source_context_sha256": original.source_context_sha256,
        "inventory_sha256": sha256(b"different public inventory").hexdigest(),
        "catalog_sha256": original.catalog_sha256,
        "database_sha256": original.database_sha256,
        "source_file_sha256": dict(original.source_file_sha256),
        "source_tree_sha256": original.source_tree_sha256,
        "condition_manifest_sha256": dict(original.condition_manifest_sha256),
        "total_successes": original.total_successes,
        "imported_total": original.imported_total,
        "native_total": original.native_total,
    }
    return Exp471SourceAuthorityAntigravity(
        **bare,
        authority_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    )


def _session_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
) -> Exp471SourceSessionAntigravity:
    return Exp471SourceSessionAntigravity(
        authority=authority,
        verification=object(),
        context=object(),
        catalog=object(),
        migration_manifest=object(),
        verifier=evaluation,
    )


def _score_maps_antigravity() -> tuple[dict[str, object], dict[str, object]]:
    raw: dict[str, object] = {}
    micro: dict[str, object] = {}
    for index, method in enumerate(METHOD_ORDER_ANTIGRAVITY):
        raw[method] = {
            _DATASET_ANTIGRAVITY: (0.05 + index / 100.0, 0.75 + index / 100.0)
        }
        micro[method] = {
            _DATASET_ANTIGRAVITY: (0.10 + index / 100.0, 0.90 + index / 100.0)
        }
    return raw, micro


def _publish_cache_antigravity(
    root: Path,
    authority: Exp471SourceAuthorityAntigravity,
) -> Exp471ScoreCacheManifestAntigravity:
    raw_scores, micro_scores = _score_maps_antigravity()
    shard_digests: dict[str, str] = {}
    cache_context = sha256(b"public literal cache context").hexdigest()
    for method in METHOD_ORDER_ANTIGRAVITY:
        shard = _score_shard_antigravity(
            authority,
            method,
            cache_context=cache_context,
            raw_scores=raw_scores,
            micro_scores=micro_scores,
        )
        cache_artifacts.publish_score_shard_antigravity(root, shard)
        shard_digests[f"{method}:{_DATASET_ANTIGRAVITY}"] = shard.envelope_sha256

    bare = {
        "source_authority_sha256": authority.authority_sha256,
        "cache_context_sha256": cache_context,
        "method_order": list(METHOD_ORDER_ANTIGRAVITY),
        "dataset_order": [_DATASET_ANTIGRAVITY],
        "shard_sha256": shard_digests,
        "total_shards": len(METHOD_ORDER_ANTIGRAVITY),
        "scored_shards": len(METHOD_ORDER_ANTIGRAVITY),
        "abstained_shards": 0,
        "failed_shards": 0,
    }
    digest_bare = {
        "schema": "exp471-score-cache-manifest-antigravity.v2",
        **bare,
    }
    manifest = Exp471ScoreCacheManifestAntigravity(
        **bare,
        manifest_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(digest_bare)
        ).hexdigest(),
    )
    cache_artifacts.publish_cache_manifest_antigravity(root, manifest)
    return manifest


def _score_shard_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    method: str,
    *,
    cache_context: str | None = None,
    raw_scores: dict[str, object] | None = None,
    micro_scores: dict[str, object] | None = None,
) -> Exp471ScoreShardAntigravity:
    if cache_context is None:
        cache_context = sha256(b"public literal cache context").hexdigest()
    if raw_scores is None or micro_scores is None:
        raw_scores, micro_scores = _score_maps_antigravity()
    payload = {
        "source_authority_sha256": authority.authority_sha256,
        "cache_context_sha256": cache_context,
        "method": method,
        "dataset": _DATASET_ANTIGRAVITY,
        "status": "scored",
        "test_count": 2,
        "test_indices": [0, 1],
        "raw_scores": list(raw_scores[method][_DATASET_ANTIGRAVITY]),
        "micro_scores": list(micro_scores[method][_DATASET_ANTIGRAVITY]),
        "abstention_reason": None,
        "source_manifest_sha256": sha256(
            f"public literal source {method}".encode("ascii")
        ).hexdigest(),
    }
    digest_payload = {
        "schema": "exp471-score-cache-shard-antigravity.v2",
        **payload,
    }
    payload_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(digest_payload)
    ).hexdigest()
    envelope = {**digest_payload, "payload_sha256": payload_sha256}
    return Exp471ScoreShardAntigravity(
        **payload,
        payload_sha256=payload_sha256,
        envelope_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(envelope)
        ).hexdigest(),
    )


def _evaluation_patches_antigravity(cache_root: Path, evaluation_root: Path):
    return (
        mock.patch.object(
            evaluation.source_authority_antigravity,
            "_source_session_is_issued_antigravity",
            return_value=True,
        ),
        mock.patch.object(
            evaluation,
            "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY",
            cache_root,
        ),
        mock.patch.object(
            evaluation,
            "EXP471_EVALUATION_ROOT_ANTIGRAVITY",
            evaluation_root,
        ),
        mock.patch.object(
            evaluation.score_cache_antigravity,
            "open_complete_score_cache_antigravity",
            side_effect=lambda _session: cache_artifacts.read_cache_manifest_antigravity(
                cache_root
            ),
        ),
    )


def _roc_coordinates_antigravity(curves: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row["method"],
            row["false_positive_rate"],
            row["true_positive_rate"],
            row["threshold"],
        )
        for row in curves["roc"]["rows"]
    )


def _pr_coordinates_antigravity(curves: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (row["method"], row["recall"], row["precision"])
        for row in curves["precision_recall"]["rows"]
    )


def _digest_recomputed_artifact_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    mutate,
) -> Exp471EvaluationArtifactAntigravity:
    payload = json.loads(
        canonical_json_bytes_exp471_antigravity(artifact.payload).decode("utf-8")
    )
    mutate(payload)
    bare = {
        "source_authority_sha256": artifact.source_authority_sha256,
        "cache_manifest_sha256": artifact.cache_manifest_sha256,
        "code_closure_sha256": artifact.code_closure_sha256,
        "method_order": list(artifact.method_order),
        "dataset_order": list(artifact.dataset_order),
        "payload": payload,
    }
    return Exp471EvaluationArtifactAntigravity(
        **bare,
        evaluation_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    )


def _typed_evidence_antigravity(
    datasets: tuple[str, ...],
    *,
    abstained: set[tuple[str, str]] = frozenset(),
) -> object:
    """Build only synthetic in-memory typed evidence for pure evaluator tests."""

    statuses: dict[str, dict[str, str]] = {}
    reasons: dict[str, dict[str, str | None]] = {}
    raw: dict[str, dict[str, tuple[float, ...]]] = {}
    micro: dict[str, dict[str, tuple[float, ...]]] = {}
    for method_index, method in enumerate(METHOD_ORDER_ANTIGRAVITY):
        statuses[method] = {}
        reasons[method] = {}
        raw[method] = {}
        micro[method] = {}
        for dataset in datasets:
            if (method, dataset) in abstained:
                statuses[method][dataset] = "abstained"
                reasons[method][dataset] = "insufficient_train_rows"
                continue
            statuses[method][dataset] = "scored"
            reasons[method][dataset] = None
            raw[method][dataset] = (
                0.49 + method_index / 1000.0,
                0.50 + method_index / 1000.0,
            )
            micro[method][dataset] = (0.994, 0.995)
    return evaluation._TypedCacheAntigravity(
        cache=SimpleNamespace(dataset_order=datasets),
        raw_scores=raw,
        micro_scores=micro,
        test_counts={dataset: 2 for dataset in datasets},
        test_indices={dataset: (5, 9) for dataset in datasets},
        status=statuses,
        reasons=reasons,
    )


class EvaluationTestsAntigravity(unittest.TestCase):
    def test_explicit_code_closure_is_complete_versioned_and_content_bound(self) -> None:
        """Omitting or changing an Exp471 authority member must change closure."""

        required = {
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "evaluation_antigravity.py",
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "config_antigravity.py",
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "contracts_antigravity.py",
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "source_authority_antigravity.py",
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "cache_artifacts_antigravity.py",
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "score_cache_antigravity.py",
            "experiments/exp450_resampling_image_ablation_antigravity/"
            "evaluation_antigravity.py",
        }
        current = evaluation._code_closure_record_antigravity()
        self.assertEqual(
            current["schema"],
            "exp471-evaluation-code-closure-antigravity.v1",
        )
        self.assertEqual(set(current["file_sha256"]), required)
        self.assertRegex(current["closure_sha256"], r"^[0-9a-f]{64}$")

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_closure_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve()
            for index, relative in enumerate(sorted(required)):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"public closure member {index}\n".encode("ascii"))
            first = evaluation._code_closure_record_antigravity(root)
            changed_path = root / sorted(required)[0]
            changed_path.write_bytes(b"changed public closure member\n")
            second = evaluation._code_closure_record_antigravity(root)
            self.assertNotEqual(
                first["file_sha256"][sorted(required)[0]],
                second["file_sha256"][sorted(required)[0]],
            )
            self.assertNotEqual(first["closure_sha256"], second["closure_sha256"])

    def test_code_closure_missing_and_symlink_members_fail_closed(self) -> None:
        """Closure traversal must require every regular member without following links."""

        required = tuple(evaluation._CODE_CLOSURE_PATHS_ANTIGRAVITY)
        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_closure_paths_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve()
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"public closure member\n")
            missing = root / required[0]
            missing.unlink()
            with self.assertRaises(ValueError):
                evaluation._code_closure_record_antigravity(root)
            missing.write_bytes(b"public closure member\n")
            target = root / required[1]
            target.unlink()
            target.symlink_to(root / required[0])
            with self.assertRaises(ValueError):
                evaluation._code_closure_record_antigravity(root)

            directory_root = root / "directory_case_antigravity"
            foreign_root = root / "foreign_case_antigravity"
            directory_root.mkdir()
            for relative in required:
                path = foreign_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"foreign public closure member\n")
            (directory_root / "experiments").symlink_to(
                foreign_root / "experiments",
                target_is_directory=True,
            )
            with self.assertRaises(ValueError):
                evaluation._code_closure_record_antigravity(directory_root)

    def test_prior_artifact_rejects_changed_current_closure_member(self) -> None:
        """A durable artifact must not reopen under a different Exp471 closure map."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_closure_reopen_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = (
                _evaluation_patches_antigravity(cache_root, evaluation_root)
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )

        changed = json.loads(
            canonical_json_bytes_exp471_antigravity(
                evaluation._code_closure_record_antigravity()
            ).decode("utf-8")
        )
        member = sorted(changed["file_sha256"])[0]
        changed["file_sha256"][member] = "f" * 64
        bare = {
            "schema": changed["schema"],
            "file_sha256": changed["file_sha256"],
        }
        changed["closure_sha256"] = sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest()
        with mock.patch.object(
            evaluation,
            "_code_closure_record_antigravity",
            return_value=changed,
        ), self.assertRaises(ValueError):
            evaluation.reconstruct_frozen_evaluation_exp471_antigravity(artifact)

    def test_digest_recomputed_scientific_tampering_fails_reconstruction(self) -> None:
        """Rehashing malformed metrics or curve geometry must not issue authority."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_semantic_tamper_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = (
                _evaluation_patches_antigravity(cache_root, evaluation_root)
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )

        def swap_dataset_rows(payload):
            rows = payload["continuous_tables"]["scored_support"]["per_dataset"]
            rows[0], rows[1] = rows[1], rows[0]

        def wrong_metric_string(payload):
            payload["metric_tables"]["per_dataset"][0][
                "average_precision"
            ] = "forged"

        def wrong_metric_bool(payload):
            payload["metric_tables"]["per_dataset"][0]["auroc"] = True

        def wrong_metric_range(payload):
            payload["metric_tables"]["per_dataset"][0]["auroc"] = 1.01

        def wrong_macro_relation(payload):
            row = payload["metric_tables"]["macro"]["A"]
            row["average_precision_denominator"] = 0
            row["average_precision_excluded"] = 0

        def reordered_paired_rows(payload):
            rows = payload["metric_tables"]["paired_statistics"]
            rows[0], rows[1] = rows[1], rows[0]

        def malformed_paired_row(payload):
            payload["metric_tables"]["paired_statistics"][0]["wins"] = True

        def roc_out_of_range(payload):
            payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"][0][
                "false_positive_rate"
            ] = 2.0

        def roc_nonmonotonic(payload):
            payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"][1][
                "true_positive_rate"
            ] = 1.0
            payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"][2][
                "true_positive_rate"
            ] = 0.5

        def roc_threshold_order(payload):
            rows = payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"]
            rows[2]["threshold"] = rows[1]["threshold"] + 1.0

        def roc_second_null(payload):
            payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"][1]["threshold"] = None

        def pr_out_of_range(payload):
            payload["curve_tables"]["scored_support"]["curves"]["precision_recall"]["rows"][0][
                "recall"
            ] = -1.0

        def pr_nonmonotonic(payload):
            rows = payload["curve_tables"]["scored_support"]["curves"]["precision_recall"]["rows"]
            rows[1]["recall"] = 0.0
            rows[2]["recall"] = 0.5

        mutations = (
            ("per-dataset row swap", swap_dataset_rows),
            ("ROC range", roc_out_of_range),
            ("ROC monotonicity", roc_nonmonotonic),
            ("ROC threshold order", roc_threshold_order),
            ("ROC second null", roc_second_null),
            ("PR range", pr_out_of_range),
            ("PR monotonicity", pr_nonmonotonic),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                tampered = _digest_recomputed_artifact_antigravity(
                    artifact, mutation
                )
                with self.assertRaises(ValueError):
                    evaluation.reconstruct_frozen_evaluation_exp471_antigravity(
                        tampered
                    )

    def test_real_curves_persist_json_safe_thresholds_without_metric_or_coordinate_drift(
        self,
    ) -> None:
        """Metric substitution or unpaired curve serialization must fail."""

        raw_scores, micro_scores = _score_maps_antigravity()
        raw_metrics = metric_tables_exp450_antigravity(
            _LABELS_ANTIGRAVITY,
            raw_scores,
            micro_scores=micro_scores,
        )
        raw_curves = curve_tables_exp450_antigravity(
            _LABELS_ANTIGRAVITY,
            micro_scores,
        )
        first: dict[str, object] = {}
        for row in raw_curves["roc"]["rows"]:
            first.setdefault(row["method"], row)
        self.assertTrue(
            all(math.isinf(first[method]["threshold"]) for method in METHOD_ORDER)
        )

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_public_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            label_calls: list[tuple[str, ...]] = []

            def load_labels(datasets: tuple[str, ...]):
                label_calls.append(datasets)
                return _LABELS_ANTIGRAVITY

            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                cache_root, evaluation_root
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=load_labels,
                )
                reconstructed = (
                    evaluation.reconstruct_frozen_evaluation_exp471_antigravity(
                        artifact
                    )
                )

            persisted = (
                evaluation_root / "evaluation_antigravity.json"
            ).read_bytes()

        self.assertEqual(label_calls, [("public_literal_dataset",)])
        self.assertNotIn(b"Infinity", persisted)
        json.loads(persisted.decode("utf-8"))
        self.assertEqual(
            artifact.payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"][0]["threshold"],
            None,
        )
        self.assertEqual(raw_metrics, reconstructed.metric_tables)
        self.assertEqual(
            _roc_coordinates_antigravity(raw_curves),
            _roc_coordinates_antigravity(reconstructed.curve_tables),
        )
        self.assertEqual(
            _pr_coordinates_antigravity(raw_curves),
            _pr_coordinates_antigravity(reconstructed.curve_tables),
        )
        self.assertRegex(artifact.evaluation_sha256, r"^[0-9a-f]{64}$")

    def test_missing_incomplete_tampered_and_wrong_authority_cache_never_reach_labels(
        self,
    ) -> None:
        """A nonterminal or foreign cache must not unlock TEST labels."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_cache_gate_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            complete_root = base / "complete_cache_antigravity"
            cache = _publish_cache_antigravity(complete_root, authority)
            label_calls: list[object] = []

            def labels(_datasets):
                label_calls.append(object())
                return _LABELS_ANTIGRAVITY

            for name, cache_root in (
                ("missing", base / "missing_cache_antigravity"),
                ("incomplete", base / "incomplete_cache_antigravity"),
            ):
                if name == "incomplete":
                    cache_artifacts.publish_score_shard_antigravity(
                        cache_root,
                        _score_shard_antigravity(authority, METHOD_ORDER[0]),
                    )
                issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                    cache_root, base / f"{name}_evaluation_antigravity"
                )
                with issued, score_root, artifact_root, cache_gate, self.assertRaises(ValueError):
                    evaluation.evaluate_exp471_antigravity(
                        session, cache, label_loader=labels
                    )

            tampered_relative = cache_artifacts.score_shard_relative_path_antigravity(
                METHOD_ORDER[0], _DATASET_ANTIGRAVITY
            )
            tampered_path = complete_root / tampered_relative
            tampered_path.chmod(0o644)
            tampered_path.write_bytes(tampered_path.read_bytes()[:-1] + b" ")
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                complete_root, base / "tampered_evaluation_antigravity"
            )
            with issued, score_root, artifact_root, cache_gate, self.assertRaises(ValueError):
                evaluation.evaluate_exp471_antigravity(
                    session, cache, label_loader=labels
                )

            second_root = base / "wrong_authority_cache_antigravity"
            second_cache = _publish_cache_antigravity(second_root, authority)
            wrong_session = _session_antigravity(_different_authority_antigravity())
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                second_root, base / "wrong_authority_evaluation_antigravity"
            )
            with issued, score_root, artifact_root, cache_gate, self.assertRaisesRegex(
                ValueError, "authority"
            ):
                evaluation.evaluate_exp471_antigravity(
                    wrong_session, second_cache, label_loader=labels
                )

        self.assertEqual(label_calls, [])

    def test_unissued_session_is_rejected_before_cache_or_label_access(self) -> None:
        """A forged source capability must fail before opening downstream gates."""

        authority = _authority_antigravity()
        session = _session_antigravity(authority)
        cache_calls: list[object] = []
        label_calls: list[object] = []
        with mock.patch.object(
            evaluation.source_authority_antigravity,
            "_source_session_is_issued_antigravity",
            return_value=False,
        ), mock.patch.object(
            evaluation.cache_artifacts_antigravity,
            "read_cache_manifest_antigravity",
            side_effect=lambda _root: cache_calls.append(object()),
        ), self.assertRaisesRegex(ValueError, "issued"):
            evaluation.evaluate_exp471_antigravity(
                session,
                object(),
                label_loader=lambda _datasets: label_calls.append(object()),
            )
        self.assertEqual(cache_calls, [])
        self.assertEqual(label_calls, [])

    def test_score_cache_authority_gate_runs_before_the_single_label_call(self) -> None:
        """Bypassing Task 5's terminal cache authority must fail this test."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_score_gate_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            events: list[str] = []
            issued, score_root, artifact_root, default_cache_gate = _evaluation_patches_antigravity(
                cache_root, evaluation_root
            )
            with issued, score_root, artifact_root, default_cache_gate, mock.patch.object(
                evaluation.score_cache_antigravity,
                "open_complete_score_cache_antigravity",
                side_effect=lambda observed: (
                    events.append("cache"),
                    self.assertIs(observed, session),
                    cache,
                )[2],
            ) as cache_gate:
                evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: (
                        events.append("labels"),
                        _LABELS_ANTIGRAVITY,
                    )[1],
                )
            self.assertEqual(cache_gate.call_args_list, [mock.call(session), mock.call(session)])
            self.assertEqual(events, ["cache", "labels", "cache"])

    def test_typed_failed_cache_never_unlocks_labels(self) -> None:
        """A failed typed-cache partition is terminal before TEST-label access."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_typed_failed_gate_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            object.__setattr__(cache, "scored_shards", len(METHOD_ORDER_ANTIGRAVITY) - 1)
            object.__setattr__(cache, "failed_shards", 1)
            label_calls: list[tuple[str, ...]] = []
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                cache_root, base / "evaluation_antigravity"
            )
            with issued, score_root, artifact_root, cache_gate, mock.patch.object(
                evaluation.score_cache_antigravity,
                "open_complete_score_cache_antigravity",
                return_value=cache,
            ), mock.patch.object(
                evaluation.cache_artifacts_antigravity,
                "read_cache_manifest_antigravity",
                return_value=cache,
            ), self.assertRaisesRegex(ValueError, "failed"):
                evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda datasets: (
                        label_calls.append(datasets),
                        _LABELS_ANTIGRAVITY,
                    )[1],
                )
            self.assertEqual(label_calls, [])

    def test_missing_identity_and_invalid_abstention_never_unlock_labels(self) -> None:
        """The typed-cache gate rejects count and union violations before labels."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_typed_union_gate_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            label_calls: list[tuple[str, ...]] = []
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                cache_root, base / "evaluation_antigravity"
            )
            object.__setattr__(cache, "total_shards", 11_169)
            with issued, score_root, artifact_root, cache_gate, mock.patch.object(
                evaluation.score_cache_antigravity,
                "open_complete_score_cache_antigravity",
                return_value=cache,
            ), mock.patch.object(
                evaluation.cache_artifacts_antigravity,
                "read_cache_manifest_antigravity",
                return_value=cache,
            ), self.assertRaisesRegex(ValueError, "identity count"):
                evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda datasets: (
                        label_calls.append(datasets), _LABELS_ANTIGRAVITY
                    )[1],
                )
            self.assertEqual(label_calls, [])

            cache = _publish_cache_antigravity(base / "invalid_cache_antigravity", authority)
            invalid = _score_shard_antigravity(authority, "knn_distance")
            object.__setattr__(invalid, "status", "abstained")
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                base / "invalid_cache_antigravity", base / "invalid_evaluation_antigravity"
            )
            original_read = cache_artifacts.read_score_shard_antigravity
            with issued, score_root, artifact_root, cache_gate, mock.patch.object(
                evaluation.cache_artifacts_antigravity,
                "read_score_shard_antigravity",
                side_effect=lambda root, method, dataset: (
                    invalid
                    if (method, dataset) == ("knn_distance", _DATASET_ANTIGRAVITY)
                    else original_read(root, method, dataset)
                ),
            ), self.assertRaisesRegex(ValueError, "abstention"):
                evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda datasets: (
                        label_calls.append(datasets), _LABELS_ANTIGRAVITY
                    )[1],
                )
            self.assertEqual(label_calls, [])

    def test_v2_payload_has_exact_top_level_authorities(self) -> None:
        """A complete typed cache opens labels once and publishes only v2 authorities."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_v2_payload_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            label_calls: list[tuple[str, ...]] = []
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                cache_root, base / "evaluation_antigravity"
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda datasets: (
                        label_calls.append(datasets), _LABELS_ANTIGRAVITY
                    )[1],
                )
        self.assertEqual(label_calls, [(_DATASET_ANTIGRAVITY,)])
        self.assertEqual(
            set(artifact.payload),
            {
                "schema", "source_authority", "score_cache", "implementation",
                "coverage", "decision_tables", "continuous_tables", "curve_tables",
                "support_authority",
            },
        )
        self.assertEqual(artifact.payload["schema"], "exp471-evaluation-antigravity.v2")

    def test_durable_reuse_is_byte_exact_and_skips_labels_and_metric_recomputation(
        self,
    ) -> None:
        """An exact terminal artifact must reopen without scientific recompute."""

        durable_bytes: list[bytes] = []
        durable_digests: list[str] = []
        for run in range(2):
            with tempfile.TemporaryDirectory(
                prefix=f"exp471_evaluation_reuse_{run}_antigravity_"
            ) as temporary:
                base = Path(temporary).resolve()
                cache_root = base / "score_cache_antigravity"
                evaluation_root = base / "evaluation_antigravity"
                authority = _authority_antigravity()
                session = _session_antigravity(authority)
                cache = _publish_cache_antigravity(cache_root, authority)
                label_calls: list[tuple[str, ...]] = []
                issued, score_root, artifact_root, cache_gate = (
                    _evaluation_patches_antigravity(cache_root, evaluation_root)
                )
                with issued, score_root, artifact_root, cache_gate:
                    first = evaluation.evaluate_exp471_antigravity(
                        session,
                        cache,
                        label_loader=lambda datasets: (
                            label_calls.append(datasets),
                            _LABELS_ANTIGRAVITY,
                        )[1],
                    )
                    path = evaluation_root / "evaluation_antigravity.json"
                    first_stat = path.stat()
                    with mock.patch.object(
                        evaluation,
                        "metric_tables_exp450_antigravity",
                        side_effect=AssertionError("metrics must not be recomputed"),
                    ), mock.patch.object(
                        evaluation,
                        "curve_tables_exp450_antigravity",
                        side_effect=AssertionError("curves must not be recomputed"),
                    ):
                        second = evaluation.evaluate_exp471_antigravity(
                            session,
                            cache,
                            label_loader=lambda _datasets: (_ for _ in ()).throw(
                                AssertionError("labels must not reopen")
                            ),
                        )
                    second_stat = path.stat()
                    reopened = evaluation.read_evaluation_exp471_antigravity(
                        evaluation_root, session, cache
                    )
                self.assertEqual((first, second, reopened), (first, first, first))
                self.assertEqual(label_calls, [(_DATASET_ANTIGRAVITY,)])
                self.assertEqual(
                    (first_stat.st_dev, first_stat.st_ino),
                    (second_stat.st_dev, second_stat.st_ino),
                )
                durable_bytes.append(path.read_bytes())
                durable_digests.append(first.evaluation_sha256)

        self.assertEqual(durable_bytes[0], durable_bytes[1])
        self.assertEqual(durable_digests[0], durable_digests[1])

    def test_reuse_read_and_reconstruction_skip_scientific_bootstrap(self) -> None:
        """Every durable path must treat persisted bootstrap rows as evidence."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_no_bootstrap_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = (
                _evaluation_patches_antigravity(cache_root, evaluation_root)
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )
                operations = (
                    (
                        "exact durable reuse",
                        lambda: evaluation.evaluate_exp471_antigravity(
                            session,
                            cache,
                            label_loader=lambda _datasets: (_ for _ in ()).throw(
                                AssertionError("labels must not reopen")
                            ),
                        ),
                    ),
                    (
                        "direct read",
                        lambda: evaluation.read_evaluation_exp471_antigravity(
                            evaluation_root, session, cache
                        ),
                    ),
                    (
                        "frozen reconstruction",
                        lambda: evaluation.reconstruct_frozen_evaluation_exp471_antigravity(
                            artifact
                        ),
                    ),
                )
                for name, operation in operations:
                    with self.subTest(name=name), mock.patch.object(
                        evaluation,
                        "_bootstrap_expectations_antigravity",
                        create=True,
                        side_effect=AssertionError(
                            "durable evaluation must not regenerate bootstrap draws"
                        ),
                    ) as bootstrap, mock.patch.object(
                        evaluation.np.random,
                        "default_rng",
                        side_effect=AssertionError(
                            "durable evaluation must not generate bootstrap draws"
                        ),
                    ), mock.patch.object(
                        evaluation.np,
                        "empty",
                        side_effect=AssertionError(
                            "durable evaluation must not rebuild estimate vectors"
                        ),
                    ), mock.patch.object(
                        evaluation.np,
                        "quantile",
                        side_effect=AssertionError(
                            "durable evaluation must not recalculate quantiles"
                        ),
                    ):
                        operation()
                        bootstrap.assert_not_called()

    def test_reuse_and_direct_read_skip_score_preparation(self) -> None:
        """Durable authority checks must not rebuild score mappings."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_no_score_prepare_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = (
                _evaluation_patches_antigravity(cache_root, evaluation_root)
            )
            with issued, score_root, artifact_root, cache_gate:
                evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )
                operations = (
                    lambda: evaluation.evaluate_exp471_antigravity(
                        session,
                        cache,
                        label_loader=lambda _datasets: (_ for _ in ()).throw(
                            AssertionError("labels must not reopen")
                        ),
                    ),
                    lambda: evaluation.read_evaluation_exp471_antigravity(
                        evaluation_root, session, cache
                    ),
                )
                for operation in operations:
                    with mock.patch.object(
                        evaluation,
                        "_verified_cache_scores_antigravity",
                        side_effect=AssertionError(
                            "durable evaluation must not prepare score mappings"
                        ),
                    ) as score_preparation:
                        operation()
                        score_preparation.assert_not_called()

    def test_initial_evaluation_calls_frozen_metric_and_paired_builders_once(
        self,
    ) -> None:
        """Initial labels and scores must enter the unchanged builder once."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_initial_builder_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = (
                _evaluation_patches_antigravity(cache_root, evaluation_root)
            )
            with issued, score_root, artifact_root, cache_gate, mock.patch.object(
                evaluation,
                "metric_tables_exp450_antigravity",
                wraps=evaluation.metric_tables_exp450_antigravity,
            ) as metric_builder, mock.patch.object(
                evaluation.frozen_evaluation_antigravity,
                "_paired_statistics",
                wraps=evaluation.frozen_evaluation_antigravity._paired_statistics,
            ) as paired_builder:
                evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )

        self.assertEqual(metric_builder.call_count, 2)
        self.assertEqual(paired_builder.call_count, 2)

    def test_reconstruction_validates_the_complete_payload_once(self) -> None:
        """Reconstruction must reuse the one validated immutable payload."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_single_validation_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = (
                _evaluation_patches_antigravity(cache_root, evaluation_root)
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )

        with mock.patch.object(
            evaluation,
            "_validated_payload_antigravity",
            wraps=evaluation._validated_payload_antigravity,
        ) as validator:
            evaluation.reconstruct_frozen_evaluation_exp471_antigravity(artifact)
        validator.assert_called_once_with(artifact)

    def test_reopen_rejects_null_outside_each_methods_first_roc_threshold(
        self,
    ) -> None:
        """A moved JSON null must not reconstruct as an unrelated metric value."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_null_gate_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            evaluation_root = base / "evaluation_antigravity"
            mutated_root = base / "mutated_evaluation_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                cache_root, evaluation_root
            )
            with issued, score_root, artifact_root, cache_gate:
                artifact = evaluation.evaluate_exp471_antigravity(
                    session,
                    cache,
                    label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
                )

            roc_rows = artifact.payload["curve_tables"]["scored_support"]["curves"]["roc"]["rows"]
            null_methods = [
                row["method"] for row in roc_rows if row["threshold"] is None
            ]
            self.assertEqual(null_methods, list(METHOD_ORDER_ANTIGRAVITY))

            value = json.loads(
                (evaluation_root / "evaluation_antigravity.json").read_text(
                    encoding="utf-8"
                )
            )
            value["payload"]["curve_tables"]["scored_support"]["curves"]["precision_recall"]["rows"][0][
                "precision"
            ] = None
            bare = {
                key: nested
                for key, nested in value.items()
                if key != "evaluation_sha256"
            }
            value["evaluation_sha256"] = sha256(
                canonical_json_bytes_exp471_antigravity(bare)
            ).hexdigest()
            mutated_root.mkdir()
            (mutated_root / "evaluation_antigravity.json").write_bytes(
                canonical_json_bytes_exp471_antigravity(value)
            )
            issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                cache_root, mutated_root
            )
            with issued, score_root, artifact_root, cache_gate, self.assertRaisesRegex(
                ValueError, "null"
            ):
                evaluation.read_evaluation_exp471_antigravity(
                    mutated_root, session, cache
                )

    def test_label_loader_is_called_once_then_exact_order_and_count_are_enforced(
        self,
    ) -> None:
        """Duplicate access or misaligned literal labels must fail the boundary."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_evaluation_labels_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            cache_root = base / "score_cache_antigravity"
            authority = _authority_antigravity()
            session = _session_antigravity(authority)
            cache = _publish_cache_antigravity(cache_root, authority)
            for name, labels in (
                ("wrong_order", {"another_dataset": (0, 1)}),
                ("wrong_count", {_DATASET_ANTIGRAVITY: (0,)}),
            ):
                calls: list[tuple[str, ...]] = []
                issued, score_root, artifact_root, cache_gate = _evaluation_patches_antigravity(
                    cache_root, base / f"{name}_evaluation_antigravity"
                )
                with issued, score_root, artifact_root, cache_gate, self.assertRaises(ValueError):
                    evaluation.evaluate_exp471_antigravity(
                        session,
                        cache,
                        label_loader=lambda datasets, labels=labels: (
                            calls.append(datasets),
                            labels,
                        )[1],
                    )
                self.assertEqual(calls, [(_DATASET_ANTIGRAVITY,)])

    def test_typed_abstention_has_no_alerts_confusion_and_semantic_nulls(self) -> None:
        """Typed abstention is a full-population no-alert, never a fake score."""

        typed = _typed_evidence_antigravity(
            ("toy_antigravity",),
            abstained={("knn_distance", "toy_antigravity")},
        )
        alerts = evaluation._decision_alerts_antigravity(
            method="knn_distance",
            protocol="matched_budget",
            alpha=None,
            raw_scores=None,
            micro_scores=None,
            test_indices=(5, 9),
            candidate_c_alert_count=1,
        )
        self.assertEqual(alerts, ())
        self.assertEqual(
            evaluation._confusion_row_antigravity((0, 1), (5, 9), alerts),
            {"tp": 0, "fp": 0, "fn": 1, "tn": 1},
        )
        metrics = metric_tables_exp450_antigravity(
            {"toy_antigravity": (0, 1)},
            typed.raw_scores,
            micro_scores=typed.micro_scores,
        )
        record = evaluation._continuous_table_record_antigravity(metrics, typed)
        row = next(
            row for row in record["per_dataset"] if row["method"] == "knn_distance"
        )
        self.assertIsNone(row["average_precision"])
        self.assertEqual(row["average_precision_exclusion"], "insufficient_train_rows")
        self.assertEqual(
            [entry["metric"] for entry in record["semantic_nulls"] if entry["method"] == "knn_distance"],
            ["average_precision", "auroc", "roc", "precision_recall"],
        )

    def test_scored_metrics_and_decision_boundaries_reuse_frozen_semantics(self) -> None:
        """Exp450 metrics, proposal threshold, alphas, and matched budget stay frozen."""

        typed = _typed_evidence_antigravity(("toy_antigravity",))
        labels = {"toy_antigravity": (0, 1)}
        observed = metric_tables_exp450_antigravity(
            labels, typed.raw_scores, micro_scores=typed.micro_scores
        )
        expected = metric_tables_exp450_antigravity(
            labels, typed.raw_scores, micro_scores=typed.micro_scores
        )
        self.assertEqual(observed, expected)
        self.assertEqual(
            evaluation._decision_alerts_antigravity(
                method="C",
                protocol="proposal_threshold",
                alpha=None,
                raw_scores=(0.499999, 0.5),
                micro_scores=(0.0, 0.0),
                test_indices=(5, 9),
                candidate_c_alert_count=0,
            ),
            (9,),
        )
        self.assertEqual(
            evaluation._decision_alerts_antigravity(
                method="knn_distance",
                protocol="conformal",
                alpha=0.005,
                raw_scores=(0.0, 0.0),
                micro_scores=(0.994, 0.995),
                test_indices=(5, 9),
                candidate_c_alert_count=0,
            ),
            (9,),
        )
        self.assertEqual(
            evaluation._decision_alerts_antigravity(
                method="knn_distance",
                protocol="matched_budget",
                alpha=None,
                raw_scores=(0.99, 0.10, 0.80),
                micro_scores=(0.0, 0.0, 0.0),
                test_indices=(7, 3, 4),
                candidate_c_alert_count=2,
            ),
            (4, 7),
        )

    def test_all_ten_support_is_ordered_and_has_a_bound_digest(self) -> None:
        """The frozen production-shaped abstention layout leaves exactly 922 datasets."""

        datasets = tuple(f"production_{index:04d}_antigravity" for index in range(1_117))
        abstained = {
            (method, dataset)
            for method, limit in (("knn_distance", 5), ("deep_svdd_1d", 195), ("drocc_1d", 195))
            for dataset in datasets[:limit]
        }
        typed = _typed_evidence_antigravity(datasets, abstained=abstained)
        support = evaluation._all_ten_common_support_antigravity(typed)
        authority = evaluation._support_authority_antigravity(support)
        self.assertEqual(len(support), 922)
        self.assertEqual(support, datasets[195:])
        self.assertRegex(authority["all_ten_common_support_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            authority["all_ten_common_support_sha256"],
            sha256(canonical_json_bytes_exp471_antigravity(list(support))).hexdigest(),
        )

    def test_curve_support_retains_frozen_roc_and_pr_axis_names(self) -> None:
        """Support-specific curves keep FPR/TPR and recall/precision orientation."""

        typed = _typed_evidence_antigravity(("toy_antigravity",))
        curves = curve_tables_exp450_antigravity(
            {"toy_antigravity": (0, 1)}, typed.micro_scores
        )
        record = evaluation._curve_support_record_antigravity(
            curves, typed, ("toy_antigravity",)
        )
        self.assertEqual(record["curves"]["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(record["curves"]["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(record["curves"]["precision_recall"]["x_name"], "recall")
        self.assertEqual(record["curves"]["precision_recall"]["y_name"], "precision")


if __name__ == "__main__":
    unittest.main()
