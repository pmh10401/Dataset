"""Canonical and create-only artifact behavior."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


class ArtifactAntigravityTests(unittest.TestCase):
    def test_canonical_json_is_sorted_and_rejects_nonfinite_antigravity(self) -> None:
        # Break caught: identical semantic evidence produces variable bytes or NaN bytes.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.artifacts_antigravity import (
            canonical_json_bytes_antigravity,
        )

        self.assertEqual(canonical_json_bytes_antigravity({"z": 1, "a": [True, None]}), b'{"a":[true,null],"z":1}\n')
        with self.assertRaises(ValueError):
            canonical_json_bytes_antigravity({"f1": float("nan")})

    def test_write_create_only_refuses_existing_target_antigravity(self) -> None:
        # Break caught: rerunning an evaluation overwrites prior evidence.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.artifacts_antigravity import (
            write_create_only_antigravity,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result_antigravity.json"
            write_create_only_antigravity(path, b"first\n")
            self.assertEqual(path.read_bytes(), b"first\n")
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")


if __name__ == "__main__":
    unittest.main()
