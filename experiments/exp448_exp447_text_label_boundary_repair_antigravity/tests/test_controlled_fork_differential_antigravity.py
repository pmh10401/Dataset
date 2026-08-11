"""Task2 differential proof for the four present controlled-fork modules only."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


_ROOT_ANTIGRAVITY = Path(__file__).parents[3]
_EXP447_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "experiments" / "exp447_exp446_postfreeze_performance_evaluation_antigravity"
_EXP448_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "experiments" / "exp448_exp447_text_label_boundary_repair_antigravity"


class ControlledForkDifferentialAntigravityTests(unittest.TestCase):
    def test_profile_changes_only_allowed_identity_keys_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.config_antigravity import FROZEN_CONFIG_ANTIGRAVITY as exp447
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY as exp448, compare_present_profile_antigravity,
        )

        comparison = compare_present_profile_antigravity(exp447, exp448)
        self.assertTrue(comparison.equal, comparison)
        self.assertEqual(comparison.changed_keys, frozenset({"experiment_id", "artifact_schema", "run_identity", "run_root"}))

    def test_paired_ast_allows_only_identity_delta_for_present_modules_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import (
            normalized_pair_equal_antigravity,
        )

        for name in ("config", "contracts", "artifacts", "bootstrap"):
            with self.subTest(name=name):
                self.assertTrue(normalized_pair_equal_antigravity(
                    (_EXP447_ANTIGRAVITY / f"{name}_antigravity.py").read_text(),
                    (_EXP448_ANTIGRAVITY / f"{name}_antigravity.py").read_text(),
                ))

    def test_paired_ast_rejects_unallowlisted_constant_delta_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import (
            normalized_pair_equal_antigravity,
        )

        source = (_EXP447_ANTIGRAVITY / "config_antigravity.py").read_text()
        changed = source.replace("195900", "195901", 1)
        self.assertFalse(normalized_pair_equal_antigravity(source, changed))

    def test_task2_import_firewall_has_no_runtime_dependencies_antigravity(self) -> None:
        forbidden = ("sqlite3", "evaluation_antigravity", "requests", "urllib", "dashboard", "cli_antigravity")
        for name in ("config", "contracts", "artifacts", "bootstrap"):
            source = (_EXP448_ANTIGRAVITY / f"{name}_antigravity.py").read_text()
            with self.subTest(name=name):
                imports = {
                    imported
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for imported in (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                }
                self.assertFalse(any(item == forbidden_name or item.startswith(f"{forbidden_name}.") for item in imports for forbidden_name in forbidden))


if __name__ == "__main__":
    unittest.main()
