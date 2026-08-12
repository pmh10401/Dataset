"""Task 3 descriptor-authority regression tests."""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


class PredecessorAntigravityTests(unittest.TestCase):
    def _copy_exp448_antigravity(self, temporary: str, predecessor):  # type: ignore[no-untyped-def]
        root = Path(temporary) / "root_antigravity"
        root.mkdir()
        for name in predecessor._LEAVES_ANTIGRAVITY:
            shutil.copyfile(predecessor._EXP448_ROOT_ANTIGRAVITY / name, root / name)
        return root
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

    def test_same_size_leaf_overwrite_after_read_is_rejected_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import predecessor_antigravity as predecessor

        with tempfile.TemporaryDirectory(prefix="exp449_overwrite_antigravity_") as temporary:
            root = self._copy_exp448_antigravity(temporary, predecessor)
            original = predecessor._validate_antigravity
            def overwrite(*args):  # type: ignore[no-untyped-def]
                target = root / "bootstrap_antigravity.json"
                payload = target.read_bytes()
                target.write_bytes(payload[:-1] + (b" " if payload[-1:] == b"\n" else b"\n"))
                return original(*args)
            with patch.object(predecessor, "_validate_antigravity", side_effect=overwrite), self.assertRaisesRegex(ValueError, "leaf changed"):
                predecessor.capture_exp448_authority_antigravity(root=root)

    def test_root_namespace_replacement_after_read_is_rejected_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import predecessor_antigravity as predecessor

        with tempfile.TemporaryDirectory(prefix="exp449_namespace_antigravity_") as temporary:
            root = self._copy_exp448_antigravity(temporary, predecessor)
            replacement = Path(temporary) / "replaced_antigravity"
            original = predecessor._validate_antigravity
            def replace(*args):  # type: ignore[no-untyped-def]
                os.replace(root, replacement)
                root.mkdir()
                return original(*args)
            with patch.object(predecessor, "_validate_antigravity", side_effect=replace), self.assertRaisesRegex(ValueError, "directory changed"):
                predecessor.capture_exp448_authority_antigravity(root=root)

    def test_rebound_artifact_map_reaches_dataset_order_prefix_and_conservation_guards_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import predecessor_antigravity as predecessor
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity, canonical_jsonl_bytes_antigravity

        def mutate(case: str) -> None:
            with tempfile.TemporaryDirectory(prefix="exp449_semantic_antigravity_") as temporary:
                root = self._copy_exp448_antigravity(temporary, predecessor)
                metrics_path = root / "dataset_metrics_antigravity.jsonl"
                metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
                if case == "order":
                    metrics[0]["dataset_order"] = 1
                elif case == "prefix":
                    metrics[148]["dataset"] = "WrongFamily_normal_1"
                else:
                    metrics[0]["hard_baseline"]["tp"] += 1
                metrics_path.write_bytes(canonical_jsonl_bytes_antigravity(metrics))
                map_path = root / "artifact_sha256_antigravity.json"
                artifact_map = json.loads(map_path.read_text(encoding="utf-8"))
                artifact_map[metrics_path.name] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
                map_path.write_bytes(canonical_json_bytes_antigravity(artifact_map))
                with patch.object(predecessor, "_ARTIFACT_SHA256_ANTIGRAVITY", hashlib.sha256(map_path.read_bytes()).hexdigest()):
                    with self.assertRaises(ValueError):
                        predecessor.capture_exp448_authority_antigravity(root=root)
        for case in ("order", "prefix", "conservation"):
            with self.subTest(case=case):
                mutate(case)


if __name__ == "__main__":
    unittest.main()
