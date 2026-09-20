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


def _function_node_antigravity(tree: ast.Module, name: str) -> ast.FunctionDef:
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if function is None:
        raise AssertionError(f"missing function: {name}")
    return function


def _fold_count_guard_target_antigravity(function: ast.FunctionDef) -> int | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        compare = node.test
        if (
            not isinstance(compare.left, ast.Call)
            or not isinstance(compare.left.func, ast.Name)
            or compare.left.func.id != "len"
            or len(compare.left.args) != 1
            or not isinstance(compare.left.args[0], ast.Name)
            or compare.left.args[0].id != "folds"
            or len(compare.ops) != 1
            or not isinstance(compare.ops[0], ast.NotEq)
            or len(compare.comparators) != 1
        ):
            continue
        target = compare.comparators[0]
        if isinstance(target, ast.Constant) and type(target.value) is int:
            return target.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "folds"
            and isinstance(target.value, ast.Name)
            and target.value.id == "FROZEN_CONFIG_ANTIGRAVITY"
        ):
            from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.config_antigravity import (
                FROZEN_CONFIG_ANTIGRAVITY,
            )

            return FROZEN_CONFIG_ANTIGRAVITY.folds
    return None


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

        scorer_function = _function_node_antigravity(
            qualification_tree,
            "_score_fold_antigravity",
        )
        call = next(
            (
                node.value
                for node in scorer_function.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "fold" for target in node.targets)
                and isinstance(node.value, ast.Call)
            ),
            None,
        )
        self.assertIsNotNone(call)
        assert isinstance(call, ast.Call)
        self.assertIsInstance(call.func, ast.Name)
        self.assertEqual(call.func.id, "_exp457_score_fold_antigravity")
        self.assertEqual(
            [arg.id for arg in call.args if isinstance(arg, ast.Name)],
            ["train", "fit_indices", "query_indices"],
        )
        self.assertEqual(
            [(keyword.arg, getattr(keyword.value, "id", None)) for keyword in call.keywords],
            [("seed", "seed"), ("fold_index", "fold_index")],
        )

        import_aliases = [
            (alias.name, alias.asname)
            for node in qualification_tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity"
            for alias in node.names
        ]
        self.assertIn(
            ("_score_fold_antigravity", "_exp457_score_fold_antigravity"),
            import_aliases,
        )

        qualify_function = _function_node_antigravity(
            qualification_tree,
            "_qualify_normal2_with_dependencies_antigravity",
        )
        folds_assignment = next(
            (
                node
                for node in ast.walk(qualify_function)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "folds" for target in node.targets)
            ),
            None,
        )
        self.assertIsNotNone(folds_assignment)
        assert isinstance(folds_assignment, ast.Assign)
        self.assertIsInstance(folds_assignment.value, ast.Call)
        self.assertIsInstance(folds_assignment.value.func, ast.Name)
        self.assertEqual(folds_assignment.value.func.id, "tuple")
        self.assertEqual(len(folds_assignment.value.args), 1)
        inner_call = folds_assignment.value.args[0]
        self.assertIsInstance(inner_call, ast.Call)
        self.assertIsInstance(inner_call.func, ast.Name)
        self.assertEqual(inner_call.func.id, "crossfit_plan_antigravity")
        self.assertEqual(len(inner_call.args), 1)
        self.assertIsInstance(inner_call.args[0], ast.Call)
        self.assertIsInstance(inner_call.args[0].func, ast.Name)
        self.assertEqual(inner_call.args[0].func.id, "len")
        self.assertEqual(
            [arg.id for arg in inner_call.args[0].args if isinstance(arg, ast.Name)],
            ["values"],
        )
        self.assertEqual(
            [(keyword.arg, getattr(keyword.value, "id", None)) for keyword in inner_call.keywords],
            [("seed", "seed")],
        )

        self.assertEqual(_fold_count_guard_target_antigravity(qualify_function), 5)

        mutated_tree = ast.parse(qualification_path.read_text(encoding="utf-8"))
        mutated_function = _function_node_antigravity(
            mutated_tree,
            "_qualify_normal2_with_dependencies_antigravity",
        )
        mutated_guard = next(
            (
                node
                for node in ast.walk(mutated_function)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Call)
                and isinstance(node.test.left.func, ast.Name)
                and node.test.left.func.id == "len"
                and len(node.test.left.args) == 1
                and isinstance(node.test.left.args[0], ast.Name)
                and node.test.left.args[0].id == "folds"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.NotEq)
                and len(node.test.comparators) == 1
            ),
            None,
        )
        self.assertIsNotNone(mutated_guard)
        assert isinstance(mutated_guard, ast.If)
        mutated_guard.test.comparators[0] = ast.Constant(value=4)
        self.assertEqual(_fold_count_guard_target_antigravity(mutated_function), 4)
        self.assertNotEqual(_fold_count_guard_target_antigravity(mutated_function), 5)


if __name__ == "__main__":
    unittest.main()
