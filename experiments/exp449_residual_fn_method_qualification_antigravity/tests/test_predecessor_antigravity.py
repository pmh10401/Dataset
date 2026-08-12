"""Task 3 descriptor-authority regression tests."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest


class PredecessorAntigravityTests(unittest.TestCase):
    def test_real_exp448_authority_has_frozen_target_facts_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import capture_exp448_authority_antigravity

        authority = capture_exp448_authority_antigravity()
        self.assertEqual(authority.population, 1117)
        self.assertEqual(authority.target_dataset_count, 48)
        self.assertEqual(authority.target_family_fn, (246, 200, 88, 75, 63, 56, 56, 53))
        self.assertEqual((authority.baseline_confusion.tp, authority.baseline_confusion.fp, authority.baseline_confusion.fn, authority.baseline_confusion.tn), (1930, 314, 1988, 191668))
        self.assertEqual(len(authority.target_baseline_rows), 48)

    def test_root_and_leaf_links_are_rejected_before_parse_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import predecessor_antigravity as predecessor

        with tempfile.TemporaryDirectory(prefix="exp449_predecessor_antigravity_") as temporary:
            directory = Path(temporary)
            leaf = directory / "leaf_antigravity.json"
            leaf.write_text("{}\n", encoding="utf-8")
            linked = directory / "linked_root_antigravity"
            linked.symlink_to(directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "directory"):
                predecessor.capture_exp448_authority_antigravity(root=linked)
        with tempfile.TemporaryDirectory(prefix="exp449_hardlink_antigravity_") as temporary:
            copied = Path(temporary) / "root_antigravity"
            copied.mkdir()
            for name in predecessor._LEAVES_ANTIGRAVITY:
                shutil.copyfile(predecessor._EXP448_ROOT_ANTIGRAVITY / name, copied / name)
            target = copied / "bootstrap_antigravity.json"
            replacement = Path(temporary) / "bootstrap_hardlink_antigravity.json"
            os.link(target, replacement)
            self.assertGreater(target.stat().st_nlink, 1)
            with self.assertRaisesRegex(ValueError, "regular inode"):
                predecessor.capture_exp448_authority_antigravity(root=copied)

    def test_wrong_inventory_is_rejected_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import capture_exp448_authority_antigravity

        with tempfile.TemporaryDirectory(prefix="exp449_predecessor_antigravity_") as temporary:
            with self.assertRaisesRegex(ValueError, "inventory"):
                capture_exp448_authority_antigravity(root=Path(temporary))

    def test_jsonl_rejects_blank_and_noncanonical_lines_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import _strict_jsonl_antigravity

        for payload in (b'{"b":1,"a":2}\n', b'{"a":1}\n\n'):
            with self.assertRaises(ValueError):
                _strict_jsonl_antigravity(payload, "synthetic")


if __name__ == "__main__":
    unittest.main()
