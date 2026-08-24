from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
    score_acf_peak_lag_numba_antigravity,
)


def _records_antigravity():
    """Build ten in-memory synthetic records; this never opens ECG200 files."""
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.qualification_antigravity import blocked_fold_indices_antigravity
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.scoring_antigravity import FoldMethodScoresAntigravity, _combine_c12_numba_antigravity, _scope_rows_antigravity

    config = FROZEN_CONFIG_ANTIGRAVITY
    train = np.random.default_rng(4715).normal(size=(69, 140)).astype(np.float64)
    records = {}
    scopes = ("normal",) + config.change_types
    for seed in config.seeds:
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            generated = _scope_rows_antigravity(train, query)
            query_rows = np.concatenate(tuple(generated[name] for name in scopes), axis=0)
            acf = score_acf_peak_lag_numba_antigravity(train[fit], query_rows)
            width = query.size
            split = {}
            values = (acf.query_early_peak_lags, acf.query_late_peak_lags, acf.query_peak_presence, acf.query_shifts, acf.query_raw_shift, acf.query_ranks, acf.confirmation_mask)
            for index, name in enumerate(scopes):
                start = index * width
                stop = (index + 1) * width
                split[name] = tuple(
                    np.array(value[start:stop], copy=True) if value.ndim == 2 else value[start:stop]
                    for value in values
                )
            c0_normal = np.linspace(-1.0, 1.0, width, dtype=np.float64)
            c0_primary = {name: np.linspace(1.0, -1.0, width, dtype=np.float64) for name in config.change_types}
            original = {
                "C0": c0_normal,
                "E12": split["normal"][5],
                "C12": _combine_c12_numba_antigravity(c0_normal, split["normal"][5], split["normal"][6]),
            }
            primary = {
                "C0": c0_primary,
                "E12": {name: split[name][5] for name in config.change_types},
                "C12": {name: _combine_c12_numba_antigravity(c0_primary[name], split[name][5], split[name][6]) for name in config.change_types},
            }
            records[(seed, fold_index)] = FoldMethodScoresAntigravity(
                seed=seed, fold_index=fold_index, methods=config.method_order,
                original=original, primary=primary,
                fit_early_peak_lags=acf.fit_early_peak_lags, fit_late_peak_lags=acf.fit_late_peak_lags,
                fit_peak_presence=acf.fit_peak_presence, fit_shifts=acf.fit_shifts, fit_raw_shift=acf.fit_raw_shift,
                query_early_peak_lags={name: split[name][0] for name in scopes},
                query_late_peak_lags={name: split[name][1] for name in scopes},
                query_peak_presence={name: split[name][2] for name in scopes},
                query_shifts={name: split[name][3] for name in scopes},
                query_raw_shift={name: split[name][4] for name in scopes},
                fit_normal_max_raw_shift=acf.fit_normal_max_raw_shift,
                confirmed={name: split[name][6] for name in scopes},
                input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba",
            )
    return records


def _published_inputs_antigravity():
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.evaluation_antigravity import evaluate_repair_gates_antigravity
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.qualification_antigravity import qualify_records_antigravity

    records = _records_antigravity()
    qualification = qualify_records_antigravity(records)
    return records, qualification, evaluate_repair_gates_antigravity(
        period_fixed=qualification.period_fixed,
        period_broken=qualification.period_broken,
        protected_broken=qualification.protected_broken,
        normal_confirmations=qualification.normal_confirmations,
        fallback_exact=qualification.fallback_exact,
        finite=qualification.finite,
    )


class ArtifactPublicationTestAntigravity(unittest.TestCase):
    def test_publish_writes_exact_leaf_set_and_reproducible_bytes_antigravity(self) -> None:
        """Break caught: publication emits a variable leaf set or nondeterministic bytes."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import LEAF_NAMES_ANTIGRAVITY, publish_result_antigravity

        records, qualification, decision = _published_inputs_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first_antigravity"
            second = Path(directory) / "second_antigravity"
            published = publish_result_antigravity(first, records, qualification, decision)
            publish_result_antigravity(second, records, qualification, decision)
            self.assertEqual(tuple(sorted(path.name for path in first.iterdir())), tuple(sorted(LEAF_NAMES_ANTIGRAVITY)))
            self.assertEqual(tuple(published.hashes), LEAF_NAMES_ANTIGRAVITY[1:])
            self.assertEqual(len(published.hashes), 6)
            for name in LEAF_NAMES_ANTIGRAVITY:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_context_preserves_all_required_scoring_evidence_antigravity(self) -> None:
        """Break caught: canonical context omits peak, shift, raw, rank, or confirmation inputs."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import publish_result_antigravity

        records, qualification, decision = _published_inputs_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)
            payload = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8"))
            record = payload["records"][0]
            keys = (
                "fit_early_peak_lags", "fit_late_peak_lags", "fit_peak_presence", "fit_shifts",
                "fit_raw_shift", "query_early_peak_lags", "query_late_peak_lags",
                "query_peak_presence", "query_shifts", "query_raw_shift", "confirmed",
                "c0_normal", "c0_primary", "fit_indices", "query_indices",
            )
            for key in keys:
                self.assertIn(key, record)

    def test_publish_rejects_dirty_root_and_caller_gate_mismatch_antigravity(self) -> None:
        """Break caught: cleanup or blindly trusted caller gates enter publication evidence."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import publish_result_antigravity

        records, qualification, decision = _published_inputs_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root_antigravity"
            root.mkdir()
            (root / "unrelated.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                publish_result_antigravity(root, records, qualification, decision)
            self.assertTrue((root / "unrelated.txt").exists())
        forged = dataclasses.replace(qualification, finite=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch_antigravity"
            with self.assertRaisesRegex(ValueError, "qualification"):
                publish_result_antigravity(path, records, forged, decision)


if __name__ == "__main__":
    unittest.main()
