from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
EXP457_PACKAGE_ANTIGRAVITY = (
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity"
)
EXP457_BASE_COMMIT_ANTIGRAVITY = "6e8e500d19afcb17907ef53ad1527b5781c6d007"


class ControlledForkAntigravityTest(unittest.TestCase):
    def test_exp457_frozen_constants_remain_unchanged_antigravity(self) -> None:
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.experiment, "Exp457")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.alpha, 0.005)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds, (20260721, 20261721))
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.primary_types,
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max, 0.05)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.pooled_sensitivity_min, 0.70)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.per_type_sensitivity_min, 0.50)

    def test_exp457_package_bytes_match_the_controlled_fork_base_antigravity(self) -> None:
        current_paths = subprocess.run(
            ["git", "ls-files", EXP457_PACKAGE_ANTIGRAVITY],
            cwd=ROOT_ANTIGRAVITY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        base_paths = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", EXP457_BASE_COMMIT_ANTIGRAVITY, EXP457_PACKAGE_ANTIGRAVITY],
            cwd=ROOT_ANTIGRAVITY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(current_paths, base_paths)

        for relative in current_paths:
            with self.subTest(relative=relative):
                current_bytes = (ROOT_ANTIGRAVITY / relative).read_bytes()
                base_bytes = subprocess.run(
                    ["git", "show", f"{EXP457_BASE_COMMIT_ANTIGRAVITY}:{relative}"],
                    cwd=ROOT_ANTIGRAVITY,
                    check=True,
                    capture_output=True,
                ).stdout
                self.assertEqual(
                    hashlib.sha256(current_bytes).hexdigest(),
                    hashlib.sha256(base_bytes).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
