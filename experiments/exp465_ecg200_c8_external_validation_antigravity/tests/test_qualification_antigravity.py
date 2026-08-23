from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from functools import lru_cache
import unittest
import numpy as np
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import centered_suffix_basis_antigravity
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.raw_evidence_antigravity import encode_vector_matrix_antigravity
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import capture_exp463_authority_antigravity
from experiments.exp465_ecg200_c8_external_validation_antigravity.scoring_antigravity import score_fold_methods_antigravity


@lru_cache(maxsize=1)
def _authoritative_records_antigravity():
    train = np.random.default_rng(465).normal(size=(69, 140)); train.setflags(write=False)
    authority = capture_exp463_authority_antigravity(); records = []
    for seed in (20260721, 20261721):
        for fold, query in enumerate(np.array_split(np.arange(69, dtype=np.int64), 5)):
            fit = np.asarray([index for index in range(69) if index not in set(query)], dtype=np.int64)
            records.append((seed, fold, score_fold_methods_antigravity(train, fit, query, seed=seed, fold_index=fold, authority=authority)))
    return tuple(records)


def passing_scores_antigravity(*, confirmation: int = 0, c8_period: float = 1.0):
    """Ten in-memory score records; no dataset files or labels are accessed."""
    if confirmation == 0 and c8_period == 1.0:
        return _authoritative_records_antigravity()
    records = []
    for seed in (20260721, 20261721):
        for fold in range(5):
            methods = ("C0", "C6", "C8")
            size = 14 if fold < 4 else 13
            normal = (0.0,) * size
            changed = (1.0,) * size
            original = {"C0": normal, "C6": changed, "C8": normal}
            primary = {"C0": {"level_shift": normal, "gradual_drift": normal, "amplitude_scaling": normal, "period_change": normal}, "C6": {"level_shift": changed, "gradual_drift": changed, "amplitude_scaling": changed, "period_change": changed}, "C8": {"level_shift": changed, "gradual_drift": changed, "amplitude_scaling": changed, "period_change": changed}}
            if c8_period < 1.0:
                primary["C8"]["period_change"] = tuple(value * c8_period for value in normal)
            counts = {"C0": {"normal": 0, "level_shift": 0, "gradual_drift": 0, "amplitude_scaling": 0, "period_change": 0}, "C6": {"normal": size, "level_shift": size, "gradual_drift": size, "amplitude_scaling": size, "period_change": size}, "C8": {"normal": confirmation, "level_shift": size, "gradual_drift": size, "amplitude_scaling": size, "period_change": size}}
            cosines = {method: {name: ((0.0,) * size if name != "normal" or not confirmation else (1.0,) + (0.0,) * (size - 1)) for name in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")} for method in methods}
            basis = centered_suffix_basis_antigravity()
            matrix = np.repeat(basis[None, :], size, axis=0)
            blob = encode_vector_matrix_antigravity(matrix)
            zero_blob = encode_vector_matrix_antigravity(np.zeros_like(matrix))
            empty = MappingProxyType({})
            raw_scopes = MappingProxyType({name: blob for name in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")})
            c8_scopes = MappingProxyType({name: (zero_blob if name == "normal" else blob) for name in raw_scopes})
            query_amplitudes = MappingProxyType({"C0": empty, "C6": MappingProxyType({name: (1.0,) * size for name in raw_scopes}), "C8": MappingProxyType({name: ((0.0,) * size if name == "normal" else (1.0,) * size) for name in raw_scopes})})
            cosines["C6"] = {name: (1.0,) * size for name in raw_scopes}; cosines["C8"] = {name: ((1.0,) * size if name != "normal" else ((1.0,) + (0.0,) * (size - 1) if confirmation else (0.0,) * size)) for name in raw_scopes}
            records.append((seed, fold, SimpleNamespace(methods=methods, original=MappingProxyType(original), primary=MappingProxyType({method: MappingProxyType(values) for method, values in primary.items()}), confirmation_counts=MappingProxyType({method: MappingProxyType(values) for method, values in counts.items()}), query_cosines=MappingProxyType({method: MappingProxyType(values) for method, values in cosines.items()}), fit_amplitudes=MappingProxyType({"C0": (), "C6": (1.0,) * size, "C8": (1.0,) * size}), query_amplitudes=query_amplitudes, raw_evidence_kind=MappingProxyType({"C0": "", "C6": "raw_centered", "C8": "raw_centered"}), raw_fit_vector_blobs=MappingProxyType({"C0": "", "C6": blob, "C8": blob}), raw_query_vector_blobs=MappingProxyType({"C0": empty, "C6": raw_scopes, "C8": c8_scopes}), feature_dimensions=MappingProxyType({method: 3 for method in methods}), provenance=MappingProxyType({method: "fixture" for method in methods}))))
    return tuple(records)


class QualificationAntigravityTest(unittest.TestCase):
    def test_recomputes_c8_ten_of_ten_from_raw_scores_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        result = qualify_methods_antigravity(passing_scores_antigravity())
        self.assertEqual(len(result.method_rows), 30)
        self.assertEqual(len(result.rows), 10)
        self.assertLess(sum(row.passed for row in result.rows), 10)
        self.assertEqual(dict(result.seed_normal_confirmations), {20260721: 0, 20261721: 0})
        self.assertEqual(dict(result.seed_normal_wilson_upper), {20260721: 0.05273725818905172, 20261721: 0.05273725818905172})

    def test_confirmation_evidence_cannot_be_forged_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        with self.assertRaises(ValueError):
            qualify_methods_antigravity(passing_scores_antigravity(confirmation=1))

    def test_rejects_seed_oof_total_other_than_sixty_nine_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        records = list(passing_scores_antigravity())
        records.pop()
        with self.assertRaises(ValueError):
            qualify_methods_antigravity(tuple(records))

    def test_replay_length_guard_rejects_sixty_eight_and_seventy_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import _validate_exact_seed_oof_lengths_antigravity, qualify_methods_antigravity
        result = qualify_methods_antigravity(passing_scores_antigravity())
        for delta in (-1, 1):
            rows = list(result.method_rows)
            target = rows[0]
            rows[0] = SimpleNamespace(method=target.method, seed=target.seed, normal_scores=target.normal_scores[:delta] if delta < 0 else target.normal_scores + (0.0,))
            with self.assertRaises(ValueError): _validate_exact_seed_oof_lengths_antigravity(tuple(rows))
