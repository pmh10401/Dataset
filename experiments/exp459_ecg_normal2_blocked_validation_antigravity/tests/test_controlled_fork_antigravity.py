from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
EXP457_PACKAGE_ANTIGRAVITY = (
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity"
)
EXP458_PACKAGE_ANTIGRAVITY = (
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity"
)
EXP459_PACKAGE_ANTIGRAVITY = (
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity"
)


class ControlledForkAntigravityTest(unittest.TestCase):
    def test_exp459_config_changes_only_identity_predecessor_and_fold_planner_antigravity(
        self,
    ) -> None:
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY as EXP457_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY as EXP458_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY as EXP459_CONFIG_ANTIGRAVITY,
        )

        self.assertEqual(
            (
                EXP459_CONFIG_ANTIGRAVITY.runtime_executable,
                EXP459_CONFIG_ANTIGRAVITY.runtime_prefix,
                EXP459_CONFIG_ANTIGRAVITY.runtime_python_version,
                EXP459_CONFIG_ANTIGRAVITY.runtime_numpy_version,
                EXP459_CONFIG_ANTIGRAVITY.runtime_aeon_version,
                EXP459_CONFIG_ANTIGRAVITY.hydra_seeds,
                EXP459_CONFIG_ANTIGRAVITY.folds,
                EXP459_CONFIG_ANTIGRAVITY.primary_types,
                EXP459_CONFIG_ANTIGRAVITY.per_type_auc_min,
                EXP459_CONFIG_ANTIGRAVITY.pooled_auc_min,
                EXP459_CONFIG_ANTIGRAVITY.normal_wilson_upper_max,
                EXP459_CONFIG_ANTIGRAVITY.structure_gate,
                EXP459_CONFIG_ANTIGRAVITY.finite_required,
                EXP459_CONFIG_ANTIGRAVITY.test_metrics_may_inform_routing,
                EXP459_CONFIG_ANTIGRAVITY.mlx_applicable,
            ),
            (
                EXP458_CONFIG_ANTIGRAVITY.runtime_executable,
                EXP458_CONFIG_ANTIGRAVITY.runtime_prefix,
                EXP458_CONFIG_ANTIGRAVITY.runtime_python_version,
                EXP458_CONFIG_ANTIGRAVITY.runtime_numpy_version,
                EXP458_CONFIG_ANTIGRAVITY.runtime_aeon_version,
                EXP458_CONFIG_ANTIGRAVITY.hydra_seeds,
                EXP458_CONFIG_ANTIGRAVITY.folds,
                EXP458_CONFIG_ANTIGRAVITY.primary_types,
                EXP458_CONFIG_ANTIGRAVITY.per_type_auc_min,
                EXP458_CONFIG_ANTIGRAVITY.pooled_auc_min,
                EXP458_CONFIG_ANTIGRAVITY.normal_wilson_upper_max,
                EXP458_CONFIG_ANTIGRAVITY.structure_gate,
                EXP458_CONFIG_ANTIGRAVITY.finite_required,
                EXP458_CONFIG_ANTIGRAVITY.test_metrics_may_inform_routing,
                EXP458_CONFIG_ANTIGRAVITY.mlx_applicable,
            ),
        )
        self.assertEqual(EXP459_CONFIG_ANTIGRAVITY.experiment, "Exp459")
        self.assertEqual(EXP459_CONFIG_ANTIGRAVITY.predecessor_experiment, "Exp458")
        self.assertEqual(
            (
                EXP459_CONFIG_ANTIGRAVITY.candidate,
                EXP459_CONFIG_ANTIGRAVITY.alpha,
                EXP459_CONFIG_ANTIGRAVITY.knn_k,
                EXP459_CONFIG_ANTIGRAVITY.kernels,
                EXP459_CONFIG_ANTIGRAVITY.groups,
            ),
            (
                EXP457_CONFIG_ANTIGRAVITY.candidate,
                EXP457_CONFIG_ANTIGRAVITY.alpha,
                EXP457_CONFIG_ANTIGRAVITY.knn_k,
                EXP457_CONFIG_ANTIGRAVITY.kernels,
                EXP457_CONFIG_ANTIGRAVITY.groups,
            ),
        )
        self.assertEqual(
            EXP459_CONFIG_ANTIGRAVITY.query_ranges,
            ((0, 282), (283, 565), (566, 848), (849, 1130), (1131, 1412)),
        )

    def test_exp459_modules_avoid_forbidden_runtime_expansions_and_bind_science_to_exp458_antigravity(
        self,
    ) -> None:
        package = ROOT_ANTIGRAVITY / EXP459_PACKAGE_ANTIGRAVITY
        contracts_path = package / "contracts_antigravity.py"
        blocked_folds_path = package / "blocked_folds_antigravity.py"

        for path in (contracts_path, blocked_folds_path):
            self.assertTrue(path.is_file(), path)

        contracts_tree = ast.parse(contracts_path.read_text(encoding="utf-8"))
        contracts_imports = {
            alias.name
            for node in ast.walk(contracts_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        contracts_imports |= {
            node.module or ""
            for node in ast.walk(contracts_tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("requests", contracts_imports)
        self.assertNotIn("urllib", contracts_imports)
        self.assertNotIn("torch", contracts_imports)
        self.assertNotIn("mlx", contracts_imports)
        self.assertNotIn("router_antigravity", contracts_imports)
        self.assertIn(
            "experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity",
            contracts_imports,
        )
        self.assertIn(
            "experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity",
            contracts_imports,
        )

        blocked_tree = ast.parse(blocked_folds_path.read_text(encoding="utf-8"))
        blocked_calls = {
            ast.unparse(node.func)
            for node in ast.walk(blocked_tree)
            if isinstance(node, ast.Call)
        }
        self.assertIn("np.arange", blocked_calls)
        self.assertIn("np.array_split", blocked_calls)
        self.assertNotIn("np.random.default_rng", blocked_calls)
        self.assertNotIn("np.random.shuffle", blocked_calls)

        contracts_source = contracts_path.read_text(encoding="utf-8")
        self.assertIn("_WILSON_95_Z_ANTIGRAVITY", contracts_source)
        self.assertIn("mann_whitney_auc_antigravity", contracts_source)

        exp458_contracts_tree = ast.parse(
            (
                ROOT_ANTIGRAVITY
                / EXP458_PACKAGE_ANTIGRAVITY
                / "contracts_antigravity.py"
            ).read_text(encoding="utf-8")
        )
        exp459_contracts_tree = contracts_tree
        exp458_wilson = next(
            node
            for node in exp458_contracts_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_WILSON_95_Z_ANTIGRAVITY"
                for target in node.targets
            )
        )
        exp459_wilson = next(
            node
            for node in exp459_contracts_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_WILSON_95_Z_ANTIGRAVITY"
                for target in node.targets
            )
        )
        self.assertEqual(ast.unparse(exp459_wilson.value), ast.unparse(exp458_wilson.value))


if __name__ == "__main__":
    unittest.main()
