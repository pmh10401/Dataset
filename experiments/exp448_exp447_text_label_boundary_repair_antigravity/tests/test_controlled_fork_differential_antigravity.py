"""Controlled-fork differential proof through the Task3 source boundary."""
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

    def test_task3_boundary_nodes_are_the_only_new_source_authority_nodes_antigravity(self) -> None:
        allowed = {
            "publication_identity": {"capture_exp446_publication_antigravity"},
            "sources": {"verify_exp447_failure_authority_antigravity", "require_absent_exp447_root_antigravity", "convert_test_label_row_antigravity"},
            "readiness": {"verify_instances_schema_metadata_antigravity"},
        }
        forbidden = {"evaluation_antigravity", "replay_antigravity", "cli_antigravity", "requests", "urllib"}
        for module, expected in allowed.items():
            source = (_EXP448_ANTIGRAVITY / f"{module}_antigravity.py").read_text()
            tree = ast.parse(source)
            names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            with self.subTest(module=module):
                self.assertTrue(expected <= names)
                imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
                self.assertFalse(imports & forbidden)

    def test_task3_ast_delta_rejects_non_boundary_source_change_antigravity(self) -> None:
        """Only audit/root/metadata/converter syntax may differ in Task3 pairs."""
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import normalized_pair_equal_antigravity

        publication_447 = (_EXP447_ANTIGRAVITY / "publication_identity_antigravity.py").read_text()
        publication_448 = (_EXP448_ANTIGRAVITY / "publication_identity_antigravity.py").read_text()
        self.assertTrue(normalized_pair_equal_antigravity(publication_447, publication_448))

        predecessor = (_EXP447_ANTIGRAVITY / "sources_antigravity.py").read_text()
        successor = (_EXP448_ANTIGRAVITY / "sources_antigravity.py").read_text()
        self.assertIn("verify_exp447_failure_authority_antigravity()", successor)
        self.assertIn("convert_test_label_row_antigravity(index, raw_label)", successor)
        normalized = successor.replace(
            '"""Open labels only after the authenticated audit, source captures, and Exp404 authority."""',
            '"""Open the one-use TEST-label session only after byte/root-identical captures."""',
        ).replace(
            "identity: PublicationIdentityAntigravity, independently_confirmed: PublicationIdentityAntigravity | None, hard: HardPredictionsAntigravity | None",
            "identity: PublicationIdentityAntigravity, independently_confirmed: PublicationIdentityAntigravity | None",
        ).replace(
            "    _assert_exp447_byte_guard_antigravity()\n    verify_exp447_failure_authority_antigravity()\n    require_absent_exp447_root_antigravity()\n",
            "",
        ).replace(
            "            or identity is independently_confirmed or identity != independently_confirmed):\n        raise ValueError(\"two distinct byte/root-identical Exp446 captures are required before labels\")\n    if not isinstance(hard, HardPredictionsAntigravity) or hard.sha256 != FROZEN_CONFIG_ANTIGRAVITY.exp404_hard_prediction_sha256 or len(hard.coordinates) != 2218:\n        raise ValueError(\"verified Exp404 Hard authority is required before labels\")",
            "            or identity != independently_confirmed):\n        raise ValueError(\"two byte/root-identical Exp446 captures are required before labels\")",
        ).replace(
            "            converted = tuple(convert_test_label_row_antigravity(index, raw_label) for index, raw_label in rows)\n            if tuple(index for index, _ in converted) != tuple(range(len(converted))):\n                raise ValueError(\"TEST indices are not contiguous\")\n            next_index += 1\n            return converted",
            "            if any(type(index) is not int or type(label) is not int or label not in (0, 1) for index, label in rows):\n                raise ValueError(\"TEST label row is invalid\")\n            if tuple(index for index, _ in rows) != tuple(range(len(rows))):\n                raise ValueError(\"TEST indices are not contiguous\")\n            next_index += 1\n            return rows",
        ).replace(
            "    if fresh_first is fresh_second or any(capture is identity or capture is independently_confirmed or capture != identity for capture in (fresh_first, fresh_second)):",
            "    if fresh_first is fresh_second or any(capture != identity for capture in (fresh_first, fresh_second)):",
        )
        def strip_boundary_nodes(source: str) -> str:
            tree = ast.parse(source)
            allowed = {"_assert_exp447_byte_guard_antigravity", "require_absent_exp447_root_antigravity", "verify_exp447_failure_authority_antigravity", "convert_test_label_row_antigravity"}
            tree.body = [node for node in tree.body if getattr(node, "name", None) not in allowed and not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id.startswith(("_EXP447_", "_AUDIT_", "_REPOSITORY_")) for target in node.targets))]
            for node in tree.body:
                if isinstance(node, ast.Import):
                    node.names = [alias for alias in node.names if alias.name not in {"json", "subprocess"}]
            tree.body = [node for node in tree.body if not isinstance(node, ast.Import) or node.names]
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
                tree.body.pop(0)
            tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets))]
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(strip_boundary_nodes(predecessor), strip_boundary_nodes(normalized))
        self.assertNotEqual(strip_boundary_nodes(predecessor), strip_boundary_nodes(normalized.replace("ORDER BY i.instance_index", "ORDER BY i.label")))


if __name__ == "__main__":
    unittest.main()
