from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _records_antigravity():
    """Build ten in-memory synthetic records; this never opens ECG200 files."""
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.config_antigravity import (
        FROZEN_CONFIG_ANTIGRAVITY,
    )
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
        blocked_fold_indices_antigravity,
    )
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import (
        FoldMethodScoresAntigravity,
        _combine_c11_numba_antigravity,
        _scope_rows_antigravity,
    )
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.spectral_entropy_antigravity import (
        score_spectral_entropy_numba_antigravity,
    )

    config = FROZEN_CONFIG_ANTIGRAVITY
    train = np.random.default_rng(4705).normal(size=(69, 140)).astype(np.float64)
    records = {}
    scopes = ("normal",) + config.change_types
    for seed in config.seeds:
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            scope_rows = _scope_rows_antigravity(train, query)
            all_query = np.concatenate(tuple(scope_rows[name] for name in scopes), axis=0)
            e11 = score_spectral_entropy_numba_antigravity(train[fit], all_query)
            width = query.size
            raw = {name: e11.query_raw_entropy[index * width:(index + 1) * width] for index, name in enumerate(scopes)}
            rank = {name: e11.query_ranks[index * width:(index + 1) * width] for index, name in enumerate(scopes)}
            confirmed = {name: e11.confirmation_mask[index * width:(index + 1) * width] for index, name in enumerate(scopes)}
            c0_normal = np.full(width, 0.5, dtype=np.float64)
            c0_primary = {name: np.full(width, 0.5, dtype=np.float64) for name in config.change_types}
            c11_normal = _combine_c11_numba_antigravity(c0_normal, rank["normal"], confirmed["normal"])
            c11_primary = {
                name: _combine_c11_numba_antigravity(c0_primary[name], rank[name], confirmed[name])
                for name in config.change_types
            }
            records[(seed, fold_index)] = FoldMethodScoresAntigravity(
                seed=seed, fold_index=fold_index, methods=config.method_order,
                original={"C0": c0_normal, "E11": rank["normal"], "C11": c11_normal},
                primary={
                    "C0": c0_primary,
                    "E11": {name: rank[name] for name in config.change_types},
                    "C11": c11_primary,
                },
                fit_raw_entropy=e11.fit_raw_entropy,
                query_raw_entropy=raw,
                fit_normal_max_raw_entropy=e11.fit_normal_max_raw_entropy,
                confirmed=confirmed,
                input_train140=train, input_fit_indices=fit, input_query_indices=query,
                backend="numba",
            )
    return records


def _published_inputs_antigravity():
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.evaluation_antigravity import (
        evaluate_repair_gates_antigravity,
    )
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
        qualify_records_antigravity,
    )

    records = _records_antigravity()
    qualification = qualify_records_antigravity(records)
    return records, qualification, evaluate_repair_gates_antigravity(qualification)


class ArtifactPublicationTestAntigravity(unittest.TestCase):
    def test_publish_writes_exact_leaf_set_hashes_and_reproducible_bytes_antigravity(self) -> None:
        """Break caught: publication emits a variable leaf set or non-deterministic bytes."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.artifacts_antigravity import (
            LEAF_NAMES_ANTIGRAVITY,
            publish_result_antigravity,
        )

        records, qualification, decision = _published_inputs_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first_antigravity"
            second = Path(directory) / "second_antigravity"
            published = publish_result_antigravity(first, records, qualification, decision)
            publish_result_antigravity(second, records, qualification, decision)
            self.assertEqual(tuple(sorted(path.name for path in first.iterdir())), tuple(sorted(LEAF_NAMES_ANTIGRAVITY)))
            self.assertEqual(tuple(published.hashes), LEAF_NAMES_ANTIGRAVITY[1:])
            for name in LEAF_NAMES_ANTIGRAVITY:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertEqual(len(published.hashes), 6)

    def test_publish_rejects_contaminated_root_nonfinite_and_caller_summary_mismatch_antigravity(self) -> None:
        """Break caught: cleanup, NaN, or blindly trusted caller gates enter evidence."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )

        records, qualification, decision = _published_inputs_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root_antigravity"
            root.mkdir()
            (root / "unrelated.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                publish_result_antigravity(root, records, qualification, decision)
            self.assertEqual((root / "unrelated.txt").read_text(encoding="utf-8"), "preserve")
        bad_train = np.array(next(iter(records.values())).input_train140, copy=True)
        bad_train[0, 0] = np.nan
        with self.assertRaises((TypeError, ValueError)):
            dataclasses.replace(next(iter(records.values())), input_train140=bad_train)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "qualification"):
                publish_result_antigravity(
                    Path(directory) / "mismatch_antigravity", records,
                    dataclasses.replace(qualification, finite=False), decision,
                )


if __name__ == "__main__":
    unittest.main()
