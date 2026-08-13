from __future__ import annotations
import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from .test_evaluation_antigravity import SHA, fixture_antigravity

class ReplayAntigravityTest(unittest.TestCase):
    def test_source_free_replay_equals_evaluation_and_rejects_mutation_inventory_links_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"evaluation_antigravity"; authority,deps,_ = fixture_antigravity(root)
            result = _evaluate_with_dependencies_antigravity(authority, SHA, deps)
            with patch("experiments.exp451_rdst_local_gap_qualification_antigravity.sources_antigravity.open_ordered_label_session_antigravity", side_effect=AssertionError), patch("experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity.verify_exp448_predecessor_antigravity", side_effect=AssertionError):
                self.assertEqual(inspect_replay_antigravity(root), result)
            (root/"extra_antigravity.json").write_text("{}")
            with self.assertRaises(ValueError): inspect_replay_antigravity(root)
            (root/"extra_antigravity.json").unlink()
            leaf=root/"datasets_antigravity.json"; payload=leaf.read_bytes(); leaf.write_bytes(payload.replace(b'"tp":1', b'"tp":2', 1))
            with self.assertRaises(ValueError): inspect_replay_antigravity(root)

    def test_rejects_symlink_and_hardlink_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            manifest=root/"manifest_antigravity.json"; backup=root/"manifest_copy_antigravity.json"; os.link(manifest,backup)
            with self.assertRaises(ValueError): inspect_replay_antigravity(root)

if __name__ == "__main__": unittest.main()
