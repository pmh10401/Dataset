"""Public-literal tests for Exp471 durable continuous-score evaluation."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
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
    }
    manifest = Exp471ScoreCacheManifestAntigravity(
        **bare,
        manifest_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(bare)
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
        "test_count": 2,
        "raw_scores": list(raw_scores[method][_DATASET_ANTIGRAVITY]),
        "micro_scores": list(micro_scores[method][_DATASET_ANTIGRAVITY]),
        "source_manifest_sha256": sha256(
            f"public literal source {method}".encode("ascii")
        ).hexdigest(),
    }
    payload_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(payload)
    ).hexdigest()
    envelope = {**payload, "payload_sha256": payload_sha256}
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


class EvaluationTestsAntigravity(unittest.TestCase):
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
            artifact.payload["curve_tables"]["roc"]["rows"][0]["threshold"],
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

            roc_rows = artifact.payload["curve_tables"]["roc"]["rows"]
            null_methods = [
                row["method"] for row in roc_rows if row["threshold"] is None
            ]
            self.assertEqual(null_methods, list(METHOD_ORDER_ANTIGRAVITY))

            value = json.loads(
                (evaluation_root / "evaluation_antigravity.json").read_text(
                    encoding="utf-8"
                )
            )
            value["payload"]["curve_tables"]["precision_recall"]["rows"][0][
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


if __name__ == "__main__":
    unittest.main()
