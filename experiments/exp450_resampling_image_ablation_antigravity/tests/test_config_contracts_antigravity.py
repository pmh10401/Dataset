"""Frozen identity and public-contract tests for Exp450."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    FROZEN_EXP450_CONFIG,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
)
from scratch.generate_exp450_codemap_antigravity import (
    REPOSITORY_ROOT,
    _repository_tracked_paths_antigravity,
    ast_callers_antigravity,
    build_codemap_payload_antigravity,
    tracked_existing_paths_antigravity,
)


class Exp450ConfigTests(unittest.TestCase):
    def test_four_conditions_are_exact_factorial(self) -> None:
        """A swapped mode or condition ordering must reject the approved ablation."""
        self.assertEqual(
            tuple((condition.condition_id, condition.length_mode, condition.image_mode)
                  for condition in FROZEN_EXP450_CONFIG.conditions),
            (
                ("A", "linear", "linear"),
                ("B", "polyphase_pchip", "linear"),
                ("C", "linear", "area"),
                ("D", "polyphase_pchip", "area"),
            ),
        )

    def test_population_statistics_and_roots_are_frozen(self) -> None:
        """A population or runtime-namespace drift must fail before execution."""
        self.assertEqual(FROZEN_EXP450_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP450_CONFIG.test_row_count, 195900)
        self.assertEqual(len(FROZEN_EXP450_CONFIG.smoke_datasets), 23)
        self.assertTrue(all("antigravity" in root.name for root in FROZEN_EXP450_CONFIG.roots))

    def test_condition_rejects_a_nonapproved_factor_pair(self) -> None:
        """Condition D with a linear image must never represent the ablation."""
        with self.assertRaisesRegex(ValueError, "approved factorial"):
            Exp450Condition("D", "polyphase_pchip", "linear")

    def test_backend_record_preserves_cpu_worker_provenance(self) -> None:
        """A backend decision must expose its frozen execution provenance."""
        decision = Exp450BackendDecision("mps", "2.13.0", True, False, 12, 11)
        self.assertEqual(
            decision.artifact_record(),
            {
                "tensor_backend": "mps",
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": False,
                "logical_cpu_count": 12,
                "cpu_workers": 11,
                "inner_numerical_threads": 1,
            },
        )
        with self.assertRaises(FrozenInstanceError):
            decision.cpu_workers = 1  # type: ignore[misc]

    def test_codemap_recognizes_retained_tests_paper_coverage(self) -> None:
        """A retained module's tests_paper caller must appear in its coverage list."""
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
        imaging = next(
            node for node in payload["nodes"]
            if node["path"] == "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py"
        )
        self.assertIn(
            "paper/equipment_anomaly_detection/tests_paper/test_hard_sources_paper.py",
            imaging["tests"],
        )

    def test_codemap_excludes_untracked_sources_but_retains_tracked_exp450_paths(self) -> None:
        """A filesystem scan must not put an untracked source into the triplet."""
        expected = (
            REPOSITORY_ROOT
            / "experiments"
            / "exp450_resampling_image_ablation_antigravity"
            / "config_antigravity.py"
        )
        namespace = expected.parent
        with TemporaryDirectory(prefix="codemap_untracked_antigravity_", dir=namespace) as temporary:
            untracked = Path(temporary) / "untracked_source_antigravity.py"
            untracked.write_text(
                "from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity "
                "import adaptive_average_1d_antigravity\n",
                encoding="utf-8",
            )
            paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
            payload = build_codemap_payload_antigravity(
                paths, ast_callers_antigravity(paths)
            )
        self.assertIn(expected, paths)
        self.assertNotIn(untracked, paths)
        untracked_relative = untracked.relative_to(REPOSITORY_ROOT).as_posix()
        self.assertNotIn(
            untracked_relative,
            {edge["caller"] for edge in payload["edges"]},
        )

    def test_codemap_discovers_index_tracked_new_source_but_excludes_untracked_peer(self) -> None:
        """New staged sources belong in the map; untracked peers never do."""
        with TemporaryDirectory(prefix="codemap_index_antigravity_") as temporary:
            repository = Path(temporary)

            def git(*arguments: str) -> None:
                subprocess.run(
                    ("git", *arguments),
                    cwd=repository,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            git("init")
            baseline = repository / "baseline_antigravity.py"
            baseline.write_text('"""Baseline."""\n', encoding="utf-8")
            git("add", baseline.name)
            git("-c", "user.name=Exp450", "-c", "user.email=exp450@example.invalid", "commit", "-m", "baseline")
            index_tracked = repository / "index_tracked_antigravity.py"
            index_tracked.write_text('"""Index-tracked fixture."""\n', encoding="utf-8")
            git("add", index_tracked.name)
            untracked = repository / "untracked_peer_antigravity.py"
            untracked.write_text('"""Untracked fixture."""\n', encoding="utf-8")
            paths = _repository_tracked_paths_antigravity(repository)
        self.assertIn(index_tracked.resolve(), paths)
        self.assertNotIn(untracked.resolve(), paths)


if __name__ == "__main__":
    unittest.main()
