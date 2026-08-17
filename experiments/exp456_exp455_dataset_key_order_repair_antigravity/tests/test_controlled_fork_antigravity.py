from __future__ import annotations

import ast
from dataclasses import fields
import importlib
from pathlib import Path
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
EXP455_ANTIGRAVITY = ROOT_ANTIGRAVITY / "experiments/exp455_exp454_conservative_label_free_routing_antigravity"
EXP456_ANTIGRAVITY = ROOT_ANTIGRAVITY / "experiments/exp456_exp455_dataset_key_order_repair_antigravity"


class _NormalizeForkAntigravity(ast.NodeTransformer):
    _allowed = {
        "Exp455": "Exp456", "Exp455ConfigAntigravity": "Exp456ConfigAntigravity",
        "conservative_label_free_routing": "exp455_dataset_key_order_repair",
        "exp455-exp454-conservative-label-free-routing-antigravity-20260817": "exp456-exp455-dataset-key-order-repair-antigravity-20260817",
        "/Users/mac/Documents/Dataset/exp455_exp454_conservative_label_free_routing_antigravity_20260817": "/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817",
        "experiments/exp455_exp454_conservative_label_free_routing_antigravity": "experiments/exp456_exp455_dataset_key_order_repair_antigravity",
        "docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html": "docs/superpowers/specs/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity-design.html",
        "ae391bdaa9b34b8d603722262887143e0bdf8284bdcbf0fa1b059d4841c0369a": "79c6ad3e8bfad6f7f9b397fc1d20b448ebc656f198de5d449ff897a92de98247",
    }

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str) and node.value in self._allowed:
            return ast.copy_location(ast.Constant(self._allowed[node.value]), node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "Exp455ConfigAntigravity":
            return ast.copy_location(ast.Name(id="Exp456ConfigAntigravity", ctx=node.ctx), node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if node.name == "Exp455ConfigAntigravity":
            node.name = "Exp456ConfigAntigravity"
        return self.generic_visit(node)


def normalized_module_antigravity(path: Path) -> str:
    """Comparison helper: only approved Exp456 identity/document bindings normalize."""
    tree = _NormalizeForkAntigravity().visit(ast.parse(path.read_text(encoding="utf-8")))
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


class ControlledForkAntigravityTest(unittest.TestCase):
    def test_five_foundation_modules_are_controlled_exp455_fork_antigravity(self) -> None:
        """Break caught: a copied formula, target order, router objective, threshold, or tie break drifts."""
        for name in ("config", "contracts", "authority", "features", "router"):
            with self.subTest(module=name):
                self.assertEqual(
                    normalized_module_antigravity(EXP455_ANTIGRAVITY / f"{name}_antigravity.py"),
                    normalized_module_antigravity(EXP456_ANTIGRAVITY / f"{name}_antigravity.py"),
                )

    def test_fork_preserves_scientific_payload_and_observable_foundation_behavior_antigravity(self) -> None:
        """Break caught: a foundation differs from Exp455 despite retaining a superficially similar AST."""
        config455 = importlib.import_module("experiments.exp455_exp454_conservative_label_free_routing_antigravity.config_antigravity").FROZEN_CONFIG_ANTIGRAVITY
        config456 = importlib.import_module("experiments.exp456_exp455_dataset_key_order_repair_antigravity.config_antigravity").FROZEN_CONFIG_ANTIGRAVITY
        scientific_fields = (
            "candidates", "target_family_counts", "target_dataset_count", "baseline_dataset_count",
            "global_fn_must_be_below", "global_fp_max", "global_f1_min", "target_family_fn_delta_max",
            "exp454_sha256", "official_python", "router_inputs_label_free", "mlx_assessment_required",
        )
        self.assertEqual(tuple(getattr(config455, name) for name in scientific_fields), tuple(getattr(config456, name) for name in scientific_fields))

        contracts455 = importlib.import_module("experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity")
        contracts456 = importlib.import_module("experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity")
        self.assertEqual(
            tuple(field.name for field in fields(contracts455.DatasetRoutingFeaturesAntigravity)),
            tuple(field.name for field in fields(contracts456.DatasetRoutingFeaturesAntigravity)),
        )
        authority455 = importlib.import_module("experiments.exp455_exp454_conservative_label_free_routing_antigravity.authority_antigravity")
        authority456 = importlib.import_module("experiments.exp456_exp455_dataset_key_order_repair_antigravity.authority_antigravity")
        self.assertEqual(authority455._indices_antigravity([3, 1], "fixture"), authority456._indices_antigravity([3, 1], "fixture"))
        router455 = importlib.import_module("experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity")
        router456 = importlib.import_module("experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity")
        self.assertEqual(
            tuple((rule.feature_name, rule.threshold_twice, rule.candidate_when) for rule in router455.threshold_rules_antigravity("target_length", (3, 5, 10))),
            tuple((rule.feature_name, rule.threshold_twice, rule.candidate_when) for rule in router456.threshold_rules_antigravity("target_length", (3, 5, 10))),
        )
