"""Behavioral contracts for the immutable paper rerun identity and provenance."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from paper.equipment_anomaly_detection.config_paper import FROZEN_CONFIG, LINEAGE_LOCK_PATH, PAPER_ROOT
from paper.equipment_anomaly_detection.lineage_paper import (
    validate_codemap_triplet,
    validate_lineage_lock,
)


class ConfigLineageTests(unittest.TestCase):
    def test_frozen_identity_and_counts(self) -> None:
        """A changed public rerun identity or aggregate changes the frozen experiment."""
        self.assertEqual(FROZEN_CONFIG.run_identity, "paper-proposal-clean-rerun-20260812")
        self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_CONFIG.test_row_count, 195900)
        self.assertEqual(FROZEN_CONFIG.ablation_alert_counts, (2218, 2248, 2244, 2222))

    def test_lineage_has_no_prediction_coordinate_or_label_input(self) -> None:
        """Old alert coordinates and TEST labels must not enter the rerun path."""
        nodes = validate_lineage_lock(LINEAGE_LOCK_PATH)
        self.assertEqual(
            {node.node_id for node in nodes},
            {
                "rocket_exp40", "exp55_best", "exp56_best", "exp84_family_guard",
                "exp84_count_cap", "rocket_block_b", "canonical_normal_state",
                "exp234_three_source", "minirocket_1024_seed_intersection",
                "minirocket_4096_seed_intersection", "raw_exact_normal_guard",
                "low_frequency_two_depth", "hard_empty_budget_30", "train_omission_stability",
            },
        )
        self.assertFalse(any(node.input_kind == "prediction_coordinates" for node in nodes))
        self.assertTrue(all(node.label_access is False for node in nodes))

    def test_origin_validation_is_static_and_never_executes_a_legacy_module(self) -> None:
        """Provenance checking must not trigger legacy import-time effects or dependencies."""
        legacy_origin = "run_experiment_149_150_virtual_run_conformal_policy"
        sys.modules.pop(legacy_origin, None)
        validate_lineage_lock(LINEAGE_LOCK_PATH)
        self.assertNotIn(legacy_origin, sys.modules)

    def test_lineage_rejects_nonfinite_json_and_invalid_field_types(self) -> None:
        """Malformed lineage cannot silently become a provenance dependency."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lineage_paper.json"
            path.write_text("[NaN]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_lineage_lock(path)

            payload = json.loads(LINEAGE_LOCK_PATH.read_text(encoding="utf-8"))
            payload[0]["label_access"] = "false"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "types"):
                validate_lineage_lock(path)

            payload = json.loads(LINEAGE_LOCK_PATH.read_text(encoding="utf-8"))
            payload[0]["source_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                validate_lineage_lock(path)

    def test_paper_codemap_triplet_is_complete_and_hash_bound(self) -> None:
        """A missing or modified codemap artifact must invalidate the complete triplet."""
        codemap = validate_codemap_triplet(PAPER_ROOT)
        self.assertEqual(
            set(codemap.required_fields),
            {"callers", "effects", "tests", "source_origins"},
        )

        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "paper"
            copied_root.mkdir()
            for source in (
                "config_paper.py",
                "lineage_lock_paper.json",
                "codemap_paper.html",
                "codemap_paper.json",
                "codemap_paper.lock",
            ):
                (copied_root / source).write_bytes((PAPER_ROOT / source).read_bytes())
            (copied_root / "codemap_paper.html").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_codemap_triplet(copied_root)

    def test_all_new_static_files_use_approved_paper_naming(self) -> None:
        """The approved standalone namespace never introduces an antigravity filename."""
        names = tuple(path.name for path in PAPER_ROOT.rglob("*") if path.is_file())
        self.assertTrue(names)
        self.assertTrue(all("paper" in name for name in names))
        self.assertTrue(all("antigravity" not in name for name in names))


if __name__ == "__main__":
    unittest.main()
