"""Regression coverage for Task 5's create-only progress authority."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ProgressAntigravityTests(unittest.TestCase):
    def test_create_only_leaf_commit_requires_canonical_re_read_antigravity(self) -> None:
        """Break caught: accepting a leaf before its durable canonical reread."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            verified_prefix_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            payload = {"sequence": 0, "candidate_name": "Crop_A", "indices": [1], "contains_test_labels": False}
            digest = commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads((root / "000_Crop_A.json").read_text()), payload)
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json")), 1)
            with self.assertRaises(FileExistsError):
                commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)

    def test_verified_prefix_stops_at_predecessor_drift_antigravity(self) -> None:
        """Break caught: trusting progress past a missing or altered predecessor leaf."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            verified_prefix_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            commit_candidate_leaf_antigravity(root, "000_Crop_A.json", {"sequence": 0})
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json")), 1)
            (root / "000_Crop_A.json").write_text('{"sequence":0}\n')
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json")), 1)

