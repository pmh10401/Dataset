from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp403_408.contracts import (
    ALPHAS,
    DatasetInput,
    FeatureBlock,
    LearnedSourceScores,
)
from experiments.exp403_408.feature_registry import FEATURE_SOURCES
from experiments.exp403_408.synthetic_anomaly import MODEL_NAMES
from experiments.exp403_408.tri_view_ssl import RawTriViewScores
from experiments.exp409_prospective_readiness.sources import build_source_batch


def _bundle(n_train: int = 50) -> DatasetInput:
    train = np.arange(n_train * 4, dtype=float).reshape(n_train, 4)
    test = np.arange(20, dtype=float).reshape(5, 4)
    return DatasetInput(
        dataset="sample",
        train_normal=train,
        test_values=test,
        target_length=4,
        hard_indices=frozenset({4}),
    )


def _blocks(n_train: int = 50) -> tuple[FeatureBlock, ...]:
    return tuple(
        FeatureBlock(
            name=source.name,
            group=source.group,
            train_oof=np.linspace(0.0, 1.0, n_train),
            test_values=np.linspace(0.1, 0.9, 5),
            source_hash=f"{index + 1:064x}",
        )
        for index, source in enumerate(FEATURE_SOURCES)
    )


def _score(model_name: str = "") -> LearnedSourceScores:
    values = tuple(0.5 for _ in range(5))
    return LearnedSourceScores(
        scores_by_alpha={str(alpha): values for alpha in ALPHAS},
        raw_test_scores=values,
        available=True,
        model_name=model_name,
    )


class SourceBatchTests(unittest.TestCase):
    @patch(
        "experiments.exp409_prospective_readiness.sources._crossfit_normal_scores"
    )
    @patch("experiments.exp409_prospective_readiness.sources._produce_raw_scores")
    @patch(
        "experiments.exp409_prospective_readiness.sources.score_synthetic_models"
    )
    @patch("experiments.exp409_prospective_readiness.sources.score_tri_view")
    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks"
    )
    def test_records_exact_thirteen_source_statuses_and_hashes(
        self,
        feature_blocks,
        tri_scores,
        synthetic_scores,
        tri_raw,
        synthetic_oof,
    ) -> None:
        feature_blocks.return_value = _blocks()
        tri_scores.return_value = _score()
        tri_raw.return_value = RawTriViewScores(
            train_oof_scores=tuple(np.linspace(0.0, 1.0, 50)),
            test_scores=tuple(0.5 for _ in range(5)),
        )
        synthetic_scores.return_value = tuple(_score(name) for name in MODEL_NAMES)
        synthetic_oof.return_value = np.linspace(0.0, 1.0, 50)

        result = build_source_batch("sample", _bundle())

        expected = (
            {source.name for source in FEATURE_SOURCES}
            | {"tri_view"}
            | {f"synthetic_{name}" for name in MODEL_NAMES}
        )
        self.assertEqual({row.source for row in result.statuses}, expected)
        self.assertEqual(len(result.statuses), 13)
        self.assertEqual(set(result.output_sha256), expected)
        self.assertTrue(
            all(len(digest) == 64 for digest in result.output_sha256.values())
        )
        self.assertEqual(len(result.sources), 13)

    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks",
        return_value=_blocks(49),
    )
    def test_below_fifty_records_four_learned_abstentions(
        self, feature_blocks
    ) -> None:
        result = build_source_batch("sample", _bundle(49))

        learned = {
            row.source: (row.status, row.reason)
            for row in result.statuses
            if row.source == "tri_view" or row.source.startswith("synthetic_")
        }
        self.assertEqual(len(learned), 4)
        self.assertEqual(
            set(learned.values()), {("abstained", "n_train_below_50")}
        )
        self.assertEqual(len(result.sources), 9)

    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks"
    )
    def test_allows_explicit_feature_abstention(self, feature_blocks) -> None:
        blocks = list(_blocks(49))
        blocks[0] = FeatureBlock(
            name=blocks[0].name,
            group=blocks[0].group,
            train_oof=(),
            test_values=(),
            source_hash=blocks[0].source_hash,
            available=False,
            reason="source_abstained",
        )
        feature_blocks.return_value = tuple(blocks)

        result = build_source_batch("sample", _bundle(49))

        status = {row.source: row for row in result.statuses}[blocks[0].name]
        self.assertEqual((status.status, status.reason), ("abstained", "source_abstained"))
        self.assertEqual(len(result.sources), 8)

    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks"
    )
    def test_rejects_unknown_feature_abstention_reason(self, feature_blocks) -> None:
        blocks = list(_blocks(49))
        blocks[0] = FeatureBlock(
            name=blocks[0].name,
            group=blocks[0].group,
            train_oof=(),
            test_values=(),
            source_hash=blocks[0].source_hash,
            available=False,
            reason="mystery",
        )
        feature_blocks.return_value = tuple(blocks)

        with self.assertRaisesRegex(ValueError, "abstention reason"):
            build_source_batch("sample", _bundle(49))

    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks"
    )
    def test_rejects_source_identity_mismatch(self, feature_blocks) -> None:
        blocks = list(_blocks(49))
        blocks[0] = FeatureBlock(
            name="unexpected",
            group=blocks[0].group,
            train_oof=blocks[0].train_oof,
            test_values=blocks[0].test_values,
            source_hash=blocks[0].source_hash,
        )
        feature_blocks.return_value = tuple(blocks)

        with self.assertRaisesRegex(ValueError, "coverage"):
            build_source_batch("sample", _bundle(49))

    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks"
    )
    def test_rejects_train_length_mismatch(self, feature_blocks) -> None:
        blocks = list(_blocks(49))
        blocks[0] = FeatureBlock(
            name=blocks[0].name,
            group=blocks[0].group,
            train_oof=np.arange(48, dtype=float),
            test_values=blocks[0].test_values,
            source_hash=blocks[0].source_hash,
        )
        feature_blocks.return_value = tuple(blocks)

        with self.assertRaisesRegex(ValueError, "scores"):
            build_source_batch("sample", _bundle(49))

    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks"
    )
    def test_rejects_non_finite_content(self, feature_blocks) -> None:
        blocks = list(_blocks(49))
        bad = np.asarray(blocks[0].test_values, dtype=float).copy()
        bad[0] = np.nan
        blocks[0] = FeatureBlock(
            name=blocks[0].name,
            group=blocks[0].group,
            train_oof=blocks[0].train_oof,
            test_values=bad,
            source_hash=blocks[0].source_hash,
        )
        feature_blocks.return_value = tuple(blocks)

        with self.assertRaisesRegex(ValueError, "scores"):
            build_source_batch("sample", _bundle(49))

    @patch(
        "experiments.exp409_prospective_readiness.sources._produce_raw_scores"
    )
    @patch(
        "experiments.exp409_prospective_readiness.sources.score_synthetic_models"
    )
    @patch("experiments.exp409_prospective_readiness.sources.score_tri_view")
    @patch(
        "experiments.exp409_prospective_readiness.sources.load_feature_blocks",
        return_value=_blocks(),
    )
    def test_rejects_unknown_learned_abstention_reason(
        self, feature_blocks, tri_scores, synthetic_scores, tri_raw
    ) -> None:
        tri_scores.return_value = LearnedSourceScores(
            available=False, abstention_reason="unexpected"
        )
        synthetic_scores.return_value = tuple(_score(name) for name in MODEL_NAMES)
        tri_raw.return_value = RawTriViewScores(
            train_oof_scores=tuple(np.linspace(0.0, 1.0, 50)),
            test_scores=tuple(0.5 for _ in range(5)),
        )

        with self.assertRaisesRegex(ValueError, "abstention reason"):
            build_source_batch("sample", _bundle())


if __name__ == "__main__":
    unittest.main()
