"""Controlled-fork differential guard for the Task 2 foundation boundary."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


_ROOT_ANTIGRAVITY = Path(__file__).parents[3]
_EXP449_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "experiments" / "exp449_residual_fn_method_qualification_antigravity"
_EXP452_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "experiments" / "exp452_exp449_index_order_repair_antigravity"
_CONFIG_ONLY_NODES_ANTIGRAVITY = frozenset({
    "DATASET_COUNT_ANTIGRAVITY",
    "PREDICTION_COUNT_ANTIGRAVITY",
    "FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY",
    "EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY",
    "EXP449_PREDICTION_ROOT_ANTIGRAVITY",
    "EXP452_EVALUATION_ROOT_ANTIGRAVITY",
    "EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY",
    "EXP452_FAILURE_AUDIT_ANTIGRAVITY",
    "INDEX_ORDER_RULE_ANTIGRAVITY",
})
_CONFIG_EXTRA_KEYS_ANTIGRAVITY = frozenset({
    "predecessor_experiment",
    "predecessor_prediction_aggregate_sha256",
    "predecessor_prediction_root",
    "evaluation_root",
    "receipt_directory",
    "failure_audit",
    "target_dataset_count",
    "target_prediction_count",
    "target_family_method_pair_count",
    "index_order_rule",
})
_CONFIG_IDENTITY_KEYS_ANTIGRAVITY = frozenset({"experiment", "schema", "kind", "run_id"})


class _IdentityNormalizerAntigravity(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if type(node.value) is str:
            node.value = (
                node.value
                .replace("Exp449", "ExpExperiment")
                .replace("Exp452", "ExpExperiment")
                .replace("exp449", "expExperiment")
                .replace("exp452", "expExperiment")
            )
        return node


def _assignment_names_antigravity(node: ast.stmt) -> set[str]:
    if not isinstance(node, ast.Assign):
        return set()
    return {target.id for target in node.targets if isinstance(target, ast.Name)}


def _normalize_config_antigravity(source: str, *, successor: bool) -> str:
    tree = ast.parse(source)
    body: list[ast.stmt] = []
    for node in tree.body:
        if successor and _assignment_names_antigravity(node) & _CONFIG_ONLY_NODES_ANTIGRAVITY:
            continue
        if isinstance(node, ast.Assign) and "FROZEN_CONFIG_ANTIGRAVITY" in _assignment_names_antigravity(node):
            call = node.value
            if not isinstance(call, ast.Call) or not call.args or not isinstance(call.args[0], ast.Dict):
                raise AssertionError("FROZEN_CONFIG_ANTIGRAVITY must remain a literal MappingProxyType dictionary")
            mapping = call.args[0]
            pairs = list(zip(mapping.keys, mapping.values, strict=True))
            kept: list[tuple[ast.expr | None, ast.expr]] = []
            for key, value in pairs:
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    raise AssertionError("frozen config keys must remain literal exact strings")
                if successor and key.value in _CONFIG_EXTRA_KEYS_ANTIGRAVITY:
                    continue
                if key.value in _CONFIG_IDENTITY_KEYS_ANTIGRAVITY:
                    value = ast.Constant(value="<IDENTITY>")
                kept.append((key, value))
            mapping.keys = [key for key, _value in kept]
            mapping.values = [value for _key, value in kept]
        body.append(node)
    tree.body = body
    normalized = _IdentityNormalizerAntigravity().visit(tree)
    return ast.dump(ast.fix_missing_locations(normalized), include_attributes=False)


def _normalize_identity_only_antigravity(source: str) -> str:
    tree = _IdentityNormalizerAntigravity().visit(ast.parse(source))
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


def _function_ast_antigravity(source: str, name: str) -> str:
    tree = ast.parse(source)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if function is None:
        raise AssertionError(f"missing controlled function: {name}")
    normalized = _IdentityNormalizerAntigravity().visit(function)
    return ast.dump(ast.fix_missing_locations(normalized), include_attributes=False)


class ControlledForkDifferentialAntigravityTest(unittest.TestCase):
    def test_config_ast_allows_only_named_identity_path_predecessor_and_index_nodes_antigravity(self) -> None:
        """Break caught: changing a scientific config node outside the explicit allowlist."""
        before = (_EXP449_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        self.assertEqual(_normalize_config_antigravity(before, successor=False), _normalize_config_antigravity(after, successor=True))

    def test_contract_artifact_and_bootstrap_asts_are_scientifically_identical_antigravity(self) -> None:
        """Break caught: changing validation, canonical IO, or bootstrap syntax."""
        for module in ("contracts", "artifacts", "bootstrap"):
            before = (_EXP449_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8")
            after = (_EXP452_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8")
            with self.subTest(module=module):
                self.assertEqual(_normalize_identity_only_antigravity(before), _normalize_identity_only_antigravity(after))

    def test_dashboard_aggregate_validation_ast_changes_identity_only_antigravity(self) -> None:
        """Break caught: weakening frozen aggregate math under the dashboard sanitizer."""
        before = (_EXP449_ANTIGRAVITY / "dashboard_export_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "dashboard_export_antigravity.py").read_text(encoding="utf-8")
        self.assertEqual(
            _function_ast_antigravity(before, "public_result_antigravity"),
            _function_ast_antigravity(after, "public_result_antigravity"),
        )

    def test_differential_rejects_unallowlisted_scientific_change_antigravity(self) -> None:
        """Break caught: a permissive normalizer accepting a retuned cross-fit fold count."""
        before = (_EXP449_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        changed = after.replace('"crossfit_folds": 5', '"crossfit_folds": 6', 1)
        self.assertNotEqual(_normalize_config_antigravity(before, successor=False), _normalize_config_antigravity(changed, successor=True))

    def test_task2_import_firewall_excludes_lifecycle_labels_network_and_accelerators_antigravity(self) -> None:
        """Break caught: foundation import opening Task 3+, network, dashboard-write, or GPU dependencies."""
        forbidden = {
            "sqlite3", "predecessor_antigravity", "sources_antigravity", "lifecycle_antigravity",
            "readiness_antigravity", "evaluation_antigravity", "replay_antigravity", "cli_antigravity",
            "requests", "urllib", "http", "mlx", "torch",
        }
        for module in ("config", "contracts", "artifacts", "bootstrap", "dashboard_export"):
            source = (_EXP452_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8")
            imported = {
                name
                for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for name in (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
            }
            with self.subTest(module=module):
                self.assertFalse(any(name == blocked or name.startswith(f"{blocked}.") for name in imported for blocked in forbidden))


if __name__ == "__main__":
    unittest.main()
