"""Result-focused controlled-fork guard for Exp453 Task 2."""

from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path
import unittest

from experiments.exp452_exp449_index_order_repair_antigravity.config_antigravity import FROZEN_CONFIG_ANTIGRAVITY as EXP452_CONFIG_ANTIGRAVITY


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
EXP452_ANTIGRAVITY = ROOT_ANTIGRAVITY / "experiments/exp452_exp449_index_order_repair_antigravity"
EXP453_ANTIGRAVITY = ROOT_ANTIGRAVITY / "experiments/exp453_exp452_fixed_interpreter_binding_repair_antigravity"
ALLOWED_CONFIG_KEYS_ANTIGRAVITY = frozenset({
    "experiment", "schema", "kind", "run_id", "fixed_interpreter", "fixed_executable", "fixed_prefix", "runtime_versions",
    "evaluation_root", "receipt_directory", "failure_audit", "evaluation_leaves",
})


class _IdentityNormalizerAntigravity(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if type(node.value) is str:
            node.value = node.value.replace("Exp452", "ExpX").replace("Exp453", "ExpX").replace("exp452", "expx").replace("exp453", "expx")
        return node


def normalized_ast_antigravity(path: Path) -> str:
    tree = _IdentityNormalizerAntigravity().visit(ast.parse(path.read_text(encoding="utf-8")))
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


def scientific_profile_antigravity(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        value = dict(value)  # type: ignore[arg-type]
    return {key: copy.deepcopy(item) for key, item in value.items() if key not in ALLOWED_CONFIG_KEYS_ANTIGRAVITY}


class ControlledForkDifferentialAntigravityTest(unittest.TestCase):
    def test_scientific_config_is_identical_after_named_runtime_fields_are_removed_antigravity(self) -> None:
        """Break caught: runtime repair silently retunes an Exp452 scientific setting."""
        try:
            module = importlib.import_module("experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.config_antigravity")
        except ModuleNotFoundError:
            self.fail("Exp453 frozen scientific config is not implemented")
        successor = module.FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(scientific_profile_antigravity(EXP452_CONFIG_ANTIGRAVITY), scientific_profile_antigravity(successor))
        changed = scientific_profile_antigravity(successor)
        changed["crossfit_folds"] = 6
        self.assertNotEqual(scientific_profile_antigravity(EXP452_CONFIG_ANTIGRAVITY), changed)

    def test_contract_artifact_and_bootstrap_asts_are_identity_only_antigravity(self) -> None:
        """Break caught: validation, canonical bytes, or bootstrap logic differs."""
        for module in ("contracts", "artifacts", "bootstrap"):
            with self.subTest(module=module):
                self.assertEqual(normalized_ast_antigravity(EXP452_ANTIGRAVITY / f"{module}_antigravity.py"), normalized_ast_antigravity(EXP453_ANTIGRAVITY / f"{module}_antigravity.py"))

    def test_dashboard_projection_ast_changes_identity_only_antigravity(self) -> None:
        """Break caught: result projection changes aggregate qualification math."""
        self.assertEqual(normalized_ast_antigravity(EXP452_ANTIGRAVITY / "dashboard_export_antigravity.py"), normalized_ast_antigravity(EXP453_ANTIGRAVITY / "dashboard_export_antigravity.py"))

    def test_foundation_imports_do_not_reach_labels_or_evaluator_antigravity(self) -> None:
        """Break caught: Task 2 import accidentally begins scientific evaluation."""
        forbidden = {"sqlite3", "sources_antigravity", "lifecycle_antigravity", "readiness_antigravity", "evaluation_antigravity", "replay_antigravity", "cli_antigravity", "mlx", "torch"}
        for module in ("config", "contracts", "artifacts", "bootstrap", "dashboard_export"):
            tree = ast.parse((EXP453_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8"))
            imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            with self.subTest(module=module):
                self.assertFalse(any(name == blocked or name.endswith(f".{blocked}") for name in imported for blocked in forbidden))


if __name__ == "__main__":
    unittest.main()
