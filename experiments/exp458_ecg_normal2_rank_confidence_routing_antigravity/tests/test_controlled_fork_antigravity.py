from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
EXP457_PACKAGE_ANTIGRAVITY = (
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity"
)
EXP458_PACKAGE_ANTIGRAVITY = (
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity"
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

    def test_task3_source_and_qualification_preserve_the_exp457_science_boundary_antigravity(self) -> None:
        """Break caught: Task 3 silently changes folds, Hydra wiring, perturbations, or seed order."""
        package = ROOT_ANTIGRAVITY / EXP458_PACKAGE_ANTIGRAVITY
        source_path = package / "sources_antigravity.py"
        qualification_path = package / "qualification_antigravity.py"
        self.assertTrue(source_path.is_file())
        self.assertTrue(qualification_path.is_file())

        source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        source_imports = {alias.name for node in ast.walk(source_tree) if isinstance(node, ast.Import) for alias in node.names}
        source_imports |= {node.module or "" for node in ast.walk(source_tree) if isinstance(node, ast.ImportFrom)}
        self.assertIn("sqlite3", source_imports)
        self.assertNotIn("requests", source_imports)
        self.assertNotIn("urllib", source_imports)
        self.assertNotIn("router_antigravity", source_imports)

        qualification_tree = ast.parse(qualification_path.read_text(encoding="utf-8"))
        qualification_imports = {
            alias.name for node in ast.walk(qualification_tree) if isinstance(node, ast.Import) for alias in node.names
        }
        qualification_imports |= {
            node.module or "" for node in ast.walk(qualification_tree) if isinstance(node, ast.ImportFrom)
        }
        self.assertIn(
            "experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity",
            qualification_imports,
        )
        self.assertIn(
            "experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity",
            qualification_imports,
        )
        self.assertNotIn("requests", qualification_imports)
        self.assertNotIn("urllib", qualification_imports)
        self.assertNotIn("router_antigravity", qualification_imports)
        self.assertNotIn("evaluation_antigravity", qualification_imports)
        self.assertNotIn("mlx", qualification_imports)
        self.assertNotIn("torch", qualification_imports)

        constants = {
            target.id: ast.literal_eval(node.value)
            for node in qualification_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"_EXPECTED_SEEDS_ANTIGRAVITY", "_EXPECTED_PRIMARY_TYPES_ANTIGRAVITY"}
        }
        self.assertEqual(constants["_EXPECTED_SEEDS_ANTIGRAVITY"], (20260721, 20261721))
        self.assertEqual(
            constants["_EXPECTED_PRIMARY_TYPES_ANTIGRAVITY"],
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )


if __name__ == "__main__":
    unittest.main()
