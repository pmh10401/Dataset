"""Closed reviewed-map and readiness tests for the Exp449 lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ReadinessAntigravityTests(unittest.TestCase):
    def test_map_generator_emits_per_path_nontriplet_fingerprints_antigravity(self) -> None:
        """Break caught: a module-level digest hides drift in one represented codemap file."""
        from scratch.build_codemap_exp449_antigravity import build_map_antigravity

        graph, lock, _html = build_map_antigravity("3f60be54fa551be4d2695de154477d48b522ad9c", "2026-08-13T00:00:00+00:00")
        self.assertIn("tracked_fingerprints", lock)
        self.assertNotIn("docs/codemap/codemap_exp449_antigravity.json", lock["tracked_fingerprints"])
        self.assertEqual(graph["scope"], ["Task 5 official outcome is implemented and immutable: readiness and zero-label predict succeeded once; evaluation opened one 48-query/72,300-row label session and retained staging evidence before replay failed permanently; inspect/retry are prohibited and dashboard_access=blocked remains presentation-only."])

    def test_reviewed_map_rejects_dirty_parent_path_and_fingerprint_drift_antigravity(self) -> None:
        """Break caught: a stale or incomplete same-commit review can authorize readiness."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import readiness_antigravity as readiness

        commit, parent = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory(prefix="exp449_review_map_antigravity_") as temporary:
            root = Path(temporary)
            map_dir = root / "docs/codemap"; map_dir.mkdir(parents=True)
            tracked = ["AGENTS.md", "docs/codemap/other_antigravity.json"]
            for name in tracked:
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(name, encoding="utf-8")
            triplet = sorted(readiness._MAP_TRIPLET_ANTIGRAVITY)
            for name in triplet:
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text("{}", encoding="utf-8")
            lock = {
                "schema": "exp449_codemap_same_commit_v1", "source_parent_commit": parent,
                "working_tree_dirty": False, "represented_paths": sorted(tracked + triplet),
                "excluded_paths": triplet,
                "tracked_fingerprints": readiness.fingerprint_tracked_paths_antigravity(root, tracked),
            }
            (map_dir / "codemap_exp449_antigravity.lock").write_text(json.dumps(lock), encoding="utf-8")
            (map_dir / "codemap_exp449_antigravity.json").write_text(json.dumps({"source_parent_commit": parent}), encoding="utf-8")
            (map_dir / "codemap_exp449_antigravity.html").write_text("<!doctype html><title>map</title>", encoding="utf-8")
            responses = {("rev-parse", "HEAD"): commit, ("rev-parse", "HEAD^"): parent, ("status", "--porcelain=v1"): "", ("diff", "--name-only", "HEAD^"): "\n".join(sorted(tracked + triplet)), ("ls-files", "-z"): "\0".join(tracked + triplet) + "\0"}
            with patch.object(readiness, "_git_antigravity", side_effect=lambda *args, **_kwargs: responses[args]):
                readiness.validate_reviewed_map_antigravity(commit, repository=root)
                responses[("status", "--porcelain=v1")] = " M evidence"
                with self.assertRaisesRegex(ValueError, "clean"):
                    readiness.validate_reviewed_map_antigravity(commit, repository=root)
                responses[("status", "--porcelain=v1")] = ""
                lock["source_parent_commit"] = "c" * 40
                (map_dir / "codemap_exp449_antigravity.lock").write_text(json.dumps(lock), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source parent"):
                    readiness.validate_reviewed_map_antigravity(commit, repository=root)
                lock["source_parent_commit"] = parent
                lock["represented_paths"] = triplet
                (map_dir / "codemap_exp449_antigravity.lock").write_text(json.dumps(lock), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "represented"):
                    readiness.validate_reviewed_map_antigravity(commit, repository=root)
                lock["represented_paths"] = sorted(tracked + triplet)
                lock["tracked_fingerprints"] = {"AGENTS.md": "0" * 64}
                (map_dir / "codemap_exp449_antigravity.lock").write_text(json.dumps(lock), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    readiness.validate_reviewed_map_antigravity(commit, repository=root)

    def test_readiness_validates_review_first_and_last_and_stops_before_audit_antigravity(self) -> None:
        """Break caught: readiness completes after review drift or audits before its terminal receipt."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle
        from experiments.exp449_residual_fn_method_qualification_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory(prefix="exp449_readiness_antigravity_") as temporary:
            parent = Path(temporary); events: list[str] = []
            def reviewed(value: str) -> None:
                self.assertEqual(value, commit); events.append("review")
            result = readiness.run_readiness_with_dependencies_antigravity(
                commit, reviewed, receipt_dir=parent / "receipts", audit_path=parent / "audit.json",
                prediction_root=parent / "prediction", evaluation_root=parent / "evaluation",
            )
            self.assertEqual(result["reviewed_commit"], commit)
            self.assertEqual(events, ["review", "review"])
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(parent / "receipts").state, "S1")
            self.assertFalse((parent / "prediction").exists())
            self.assertFalse((parent / "evaluation").exists())
            occupied_parent = parent / "occupied"; occupied_parent.mkdir()
            audit = occupied_parent / "audit.json"; audit.write_bytes(b"foreign\n")
            with self.assertRaisesRegex(ValueError, "audit destination is occupied"):
                readiness.run_readiness_with_dependencies_antigravity(
                    commit, reviewed, receipt_dir=occupied_parent / "receipts", audit_path=audit,
                    prediction_root=occupied_parent / "prediction", evaluation_root=occupied_parent / "evaluation",
                )
            self.assertEqual(audit.read_bytes(), b"foreign\n")
            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(occupied_parent / "receipts").stopped)


if __name__ == "__main__":
    unittest.main()
