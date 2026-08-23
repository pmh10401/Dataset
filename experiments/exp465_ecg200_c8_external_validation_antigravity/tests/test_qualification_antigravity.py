from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
import unittest


def passing_scores_antigravity(*, confirmation: int = 0, c8_period: float = 1.0):
    """Ten in-memory score records; no dataset files or labels are accessed."""
    records = []
    for seed in (20260721, 20261721):
        for fold in range(5):
            methods = ("C0", "C6", "C8")
            original = {method: (0.0, 0.1, 0.2) for method in methods}
            primary = {method: {"level_shift": (1.0, 1.1, 1.2), "gradual_drift": (1.0, 1.1, 1.2), "amplitude_scaling": (1.0, 1.1, 1.2), "period_change": (1.0, 1.1, 1.2)} for method in methods}
            if c8_period < 1.0:
                primary["C8"]["period_change"] = (0.0, 0.05, 0.1)
            counts = {method: {"normal": confirmation if method == "C8" else 0, "level_shift": 0, "gradual_drift": 0, "amplitude_scaling": 0, "period_change": 0} for method in methods}
            cosines = {method: {name: ((0.0, 0.0, 0.0) if name != "normal" or not confirmation else (1.0, 0.0, 0.0)) for name in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")} for method in methods}
            records.append((seed, fold, SimpleNamespace(methods=methods, original=MappingProxyType(original), primary=MappingProxyType({method: MappingProxyType(values) for method, values in primary.items()}), confirmation_counts=MappingProxyType({method: MappingProxyType(values) for method, values in counts.items()}), query_cosines=MappingProxyType({method: MappingProxyType(values) for method, values in cosines.items()}), feature_dimensions=MappingProxyType({method: 3 for method in methods}), provenance=MappingProxyType({method: "fixture" for method in methods}))))
    return tuple(records)


class QualificationAntigravityTest(unittest.TestCase):
    def test_recomputes_c8_ten_of_ten_from_raw_scores_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        result = qualify_methods_antigravity(passing_scores_antigravity())
        self.assertEqual(len(result.method_rows), 30)
        self.assertEqual([row.passed for row in result.rows], [True] * 10)
        self.assertEqual(dict(result.seed_normal_confirmations), {20260721: 0, 20261721: 0})

    def test_confirmation_evidence_cannot_be_forged_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        with self.assertRaises(ValueError):
            qualify_methods_antigravity(passing_scores_antigravity(confirmation=1))
