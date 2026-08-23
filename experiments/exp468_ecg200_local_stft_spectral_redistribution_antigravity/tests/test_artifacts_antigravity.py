"""Task 5 publication contract tests, written before production modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.artifacts_antigravity import (
    PublishedResultAntigravity,
    publish_result_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.evaluation_antigravity import (
    evaluate_qualification_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity import (
    blocked_fold_indices_antigravity,
    qualify_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity import (
    FoldMethodScoresAntigravity,
    score_fold_methods_antigravity,
)


@lru_cache(maxsize=1)
def records_fixture_antigravity() -> MappingProxyType:
    x = np.arange(140, dtype=np.float64)
    base = 0.01 * x + np.sin(2.0 * np.pi * x / 21.0) + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    train = np.asarray(
        base[None, :] + np.random.default_rng(46850).normal(scale=0.01, size=(69, 140)),
        dtype=np.float64,
        order="C",
    )
    train.setflags(write=False)
    authority = capture_exp465_authority_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for seed in (20260721, 20261721):
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train,
                fit,
                query,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    return MappingProxyType(records)


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publish_writes_exact_canonical_seven_leaf_publication(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualify_antigravity(records)
        decision = evaluate_qualification_antigravity(qualification)
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            published = publish_result_antigravity(
                Path(directory) / "root-a",
                records,
                qualification,
                decision,
            )
            self.assertIs(type(published), PublishedResultAntigravity)
            self.assertEqual(
                tuple(sorted(path.name for path in published.root.iterdir())),
                (
                    "artifact_sha256_antigravity.json",
                    "context_antigravity.json",
                    "dashboard_snapshot_antigravity.json",
                    "decision_antigravity.json",
                    "failed_folds_antigravity.jsonl",
                    "method_fold_metrics_antigravity.jsonl",
                    "method_summary_antigravity.jsonl",
                ),
            )
            self.assertEqual(published.qualification, qualification)
            self.assertEqual(published.decision, decision)
            self.assertEqual(
                len((published.root / "method_fold_metrics_antigravity.jsonl").read_text().splitlines()),
                30,
            )
            self.assertEqual(
                len((published.root / "method_summary_antigravity.jsonl").read_text().splitlines()),
                3,
            )

    def test_same_inputs_publish_byte_identical_roots_and_existing_root_is_rejected(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualify_antigravity(records)
        decision = evaluate_qualification_antigravity(qualification)
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root_a = publish_result_antigravity(Path(directory) / "a", records, qualification, decision).root
            root_b = publish_result_antigravity(Path(directory) / "b", records, qualification, decision).root
            for path_a in root_a.iterdir():
                self.assertEqual(path_a.read_bytes(), (root_b / path_a.name).read_bytes())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_result_antigravity(root_a, records, qualification, decision)


if __name__ == "__main__":
    unittest.main()
