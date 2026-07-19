from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.exp234_precision_first_rescue import (
    build_rescue_prediction,
    research_rescue_candidates,
)
from experiments.exp226_235_hard_alert_quality.exp235_shadow_export import (
    build_shadow_rows,
)


class Exp234Tests(unittest.TestCase):
    def test_rescue_requires_independent_support_and_excludes_exp137(self) -> None:
        selected = research_rescue_candidates(
            source_hits={"temporal": {1, 2}, "canonical": {1, 3}, "shape": {1, 2}},
            excluded_indices={1},
            minimum_sources=2,
        )
        self.assertEqual(selected, frozenset({2}))

    def test_rescue_prediction_is_never_autonomous(self) -> None:
        row = build_rescue_prediction(
            dataset="demo",
            variant="support_3",
            candidate_indices={4, 7},
            evidence={"minimum_sources": 3},
        )
        self.assertFalse(row.autonomous_alert)
        self.assertEqual(row.hard_indices, ())
        self.assertEqual(row.research_indices, (4, 7))


class Exp235Tests(unittest.TestCase):
    def test_shadow_rows_exclude_evaluation_fields(self) -> None:
        prediction = build_rescue_prediction(
            dataset="demo",
            variant="support_3",
            candidate_indices={4},
            evidence={"sources": ["temporal", "canonical", "shape"]},
        )
        rows = build_shadow_rows([prediction], prediction_sha256="a" * 64)
        self.assertEqual(len(rows), 1)
        forbidden = {
            "tp", "fp", "fn", "f1", "precision", "recall", "label",
            "y_test", "anomaly_position",
        }
        self.assertTrue(forbidden.isdisjoint(rows[0]))
        self.assertEqual(rows[0]["prediction_sha256"], "a" * 64)

    def test_shadow_export_requires_sha256(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            build_shadow_rows([], prediction_sha256="short")


if __name__ == "__main__":
    unittest.main()
