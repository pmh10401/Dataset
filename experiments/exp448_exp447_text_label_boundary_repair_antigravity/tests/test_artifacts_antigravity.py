"""Canonical and create-only Exp448 artifact behavior."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ArtifactAntigravityTests(unittest.TestCase):
    def test_canonical_json_is_sorted_and_rejects_nonfinite_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.artifacts_antigravity import (
            canonical_json_bytes_antigravity,
        )

        self.assertEqual(canonical_json_bytes_antigravity({"z": 1, "a": [True, None]}), b'{"a":[true,null],"z":1}\n')
        with self.assertRaises(ValueError):
            canonical_json_bytes_antigravity({"f1": float("nan")})

    def test_write_create_only_refuses_existing_target_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.artifacts_antigravity import (
            write_create_only_antigravity,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result_antigravity.json"
            write_create_only_antigravity(path, b"first\n")
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_create_only_rejects_symlink_and_zero_progress_write_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.artifacts_antigravity import (
            write_create_only_antigravity,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "destination_antigravity.json"
            link = root / "link_antigravity.json"
            destination.write_bytes(b"preserve\n")
            link.symlink_to(destination)
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(link, b"forbidden\n")
            partial = root / "partial_antigravity.json"
            with patch("os.write", return_value=0), self.assertRaises(OSError):
                write_create_only_antigravity(partial, b"partial\n")
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(partial, b"retry\n")


if __name__ == "__main__":
    unittest.main()
