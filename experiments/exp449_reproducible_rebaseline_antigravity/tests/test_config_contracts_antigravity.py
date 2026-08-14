"""Exp449 version-boundary tests with an immutable Exp448 byte inventory."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP448_DESIGN_COMMIT = "26830622"
EXP448_FROZEN_PATHS = (
    "experiments/exp448_exp447_text_label_boundary_repair_antigravity",
    "docs/codemap/codemap_exp448_antigravity.html",
    "docs/codemap/codemap_exp448_antigravity.json",
    "docs/codemap/codemap_exp448_antigravity.lock",
    "experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/"
    "test_paper_r4_full_route_parity_recovery_antigravity.py",
)


def _git_stdout(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _blob_inventory_at(ref: str) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for line in _git_stdout("ls-tree", "-r", "--full-tree", ref, "--", *EXP448_FROZEN_PATHS).splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, blob_hash = metadata.split()
        if object_type == "blob":
            inventory[path] = blob_hash
    return dict(sorted(inventory.items()))


def exp448_frozen_blob_inventory_at_design_commit(ref: str) -> dict[str, str]:
    return _blob_inventory_at(ref)


def current_exp448_blob_inventory() -> dict[str, str]:
    tracked_paths = set(
        _git_stdout("ls-files", "--cached", "--", *EXP448_FROZEN_PATHS).splitlines()
    )
    untracked_paths = set(
        _git_stdout("ls-files", "--others", "--exclude-standard", "--", *EXP448_FROZEN_PATHS).splitlines()
    )
    inventory: dict[str, str] = {}
    for path in sorted(tracked_paths | untracked_paths):
        candidate = REPO_ROOT / path
        if candidate.is_file():
            inventory[path] = _git_stdout("hash-object", str(candidate)).strip()
    return inventory


class Exp449VersionBoundaryTests(unittest.TestCase):
    def test_exp449_state_and_triplet_exist_without_mutating_exp448(self) -> None:
        state = json.loads((REPO_ROOT / "RESEARCH_STATE.json").read_text())
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["status"], "design_approved_implementation_planned")
        for suffix in ("html", "json", "lock"):
            self.assertTrue(
                (REPO_ROOT / f"docs/codemap/codemap_exp449_antigravity.{suffix}").is_file()
            )
        frozen = exp448_frozen_blob_inventory_at_design_commit(EXP448_DESIGN_COMMIT)
        self.assertTrue(frozen)
        self.assertEqual(current_exp448_blob_inventory(), frozen)
