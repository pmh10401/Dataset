"""Synthetic descriptor-pin tests for the Exp448 publication boundary."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest


class PublicationIdentityAntigravityTests(unittest.TestCase):
    def test_pinned_leaf_rejects_post_read_replacement_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import publication_identity_antigravity as publication

        with tempfile.TemporaryDirectory(prefix="exp448_identity_antigravity_") as temporary:
            root = Path(temporary)
            target = root / "authority_antigravity.json"
            target.write_bytes(b"first\n")
            directory_fd = os.open(root, publication._DIRECTORY_FLAGS_ANTIGRAVITY)
            try:
                pinned = publication._pin_leaf_antigravity(directory_fd, target.name)
                try:
                    self.assertEqual(publication._read_pinned_leaf_antigravity(pinned), b"first\n")
                    replacement = root / "replacement_antigravity.json"
                    replacement.write_bytes(b"second\n")
                    os.replace(replacement, target)
                    with self.assertRaises(ValueError):
                        publication._assert_pinned_leaf_antigravity(directory_fd, pinned)
                finally:
                    os.close(pinned.descriptor)
            finally:
                os.close(directory_fd)

    def test_capture_rejects_extra_or_symlinked_leaf_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import publication_identity_antigravity as publication

        with tempfile.TemporaryDirectory(prefix="exp448_identity_antigravity_") as temporary:
            root = Path(temporary)
            (root / "safe_antigravity.json").write_bytes(b"{}\n")
            descriptor = os.open(root, publication._DIRECTORY_FLAGS_ANTIGRAVITY)
            try:
                publication._exact_names_antigravity(descriptor, ("safe_antigravity.json",))
                (root / "extra_antigravity.json").write_bytes(b"{}\n")
                with self.assertRaises(ValueError):
                    publication._exact_names_antigravity(descriptor, ("safe_antigravity.json",))
            finally:
                os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
