"""TDD coverage for Exp441's pure project-F1 and paired bootstrap boundary."""
from __future__ import annotations

import ast
import math
from unittest.mock import patch
import unittest

import numpy as np

from experiments.exp441_hard_empty_single_shot_recovery_antigravity import bootstrap_antigravity as bootstrap


class _SpyGenerator:
    """Deterministic replacement used to prove one paired draw per repetition."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.indices = np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.tile(np.asarray([1, 2], dtype=np.int64), 558),
        ))

    def integers(self, *args: object, **kwargs: object) -> np.ndarray:
        self.calls.append((args, kwargs))
        return self.indices.copy()


def _call_name_antigravity(node: ast.expr) -> str:
    """Return a semantic call target without evaluating source code."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name_antigravity(node.value)
        return f"{prefix}.{node.attr}" if prefix != "<expression>" else f"<expression>.{node.attr}"
    return "<expression>"


def _import_signature_antigravity(node: ast.Import | ast.ImportFrom) -> tuple[object, ...]:
    """Normalize one import for an exact executable allowlist comparison."""
    aliases = tuple((alias.name, alias.asname) for alias in node.names)
    if isinstance(node, ast.Import):
        return "import", 0, None, aliases
    return "from", node.level, node.module, aliases


class Exp441BootstrapAntigravityTests(unittest.TestCase):
    def test_project_f1_uses_precision_recall_harmonic_path(self) -> None:
        """A one-ULP hand-derived case catches direct 2TP arithmetic substitution."""
        confusion = (7, 999_999_996, 10_004, 0)
        precision = 7.0 / (7.0 + 999_999_996.0)
        recall = 7.0 / (7.0 + 10_004.0)
        expected = (2.0 * precision * recall) / (precision + recall)
        result = bootstrap.project_f1_antigravity(confusion)
        self.assertIs(type(result), float)
        self.assertEqual(result, expected)

    def test_project_f1_zero_denominator_is_zero(self) -> None:
        """No predicted or actual positives yields the frozen zero F1 convention."""
        self.assertEqual(bootstrap.project_f1_antigravity((0, 0, 0, 0)), 0.0)
        self.assertEqual(bootstrap.project_f1_antigravity((0, 3, 4, 0)), 0.0)

    def test_project_f1_rejects_bool_negative_nonfinite_and_wrong_length(self) -> None:
        """Invalid confusion values must not enter retrospective evidence."""
        invalid = (
            (True, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, math.inf, 0),
            (0, 0, math.nan, 0),
            (0, 0, 0),
            (0, 0, 0, 0, 0),
        )
        for confusion in invalid:
            with self.subTest(confusion=confusion):
                with self.assertRaises(ValueError):
                    bootstrap.project_f1_antigravity(confusion)

    def test_require_numpy_version_rejects_any_unfrozen_version(self) -> None:
        """Replay must fail closed when NumPy is not exactly the preregistered release."""
        self.assertIsNone(bootstrap.require_numpy_2_4_6_antigravity())
        with patch.object(bootstrap.np, "__version__", "2.4.5"):
            with self.assertRaisesRegex(RuntimeError, "2.4.6"):
                bootstrap.require_numpy_2_4_6_antigravity()

    def test_paired_bootstrap_replays_exact_generator_and_linear_quantiles(self) -> None:
        """Using one row, first macro, or float32 aggregation changes the frozen literals."""
        rows = np.zeros((1117, 9), dtype=np.float64)
        rows[0, :4] = (1.0, 0.0, 1.0, 0.0)
        rows[0, 4:8] = (1.0, 0.0, 1.0, 0.0)
        rows[0, 8] = 0.125
        rows[1, :4] = (2.0 ** -10 + 2.0 ** -40, 2.0 ** -11, 2.0 ** -12, 0.0)
        rows[1, 4:8] = (2.0 ** -10 + 2.0 ** -39, 2.0 ** -12, 2.0 ** -13, 0.0)
        rows[1, 8] = 0.5 + 2.0 ** -40
        rows[2, :4] = (2.0 ** -11, 2.0 ** -10 + 2.0 ** -40, 2.0 ** -13, 0.0)
        rows[2, 4:8] = (2.0 ** -10, 2.0 ** -11 + 2.0 ** -39, 2.0 ** -14, 0.0)
        rows[2, 8] = 0.5 + 2.0 ** -39
        spy = _SpyGenerator()
        quantile_calls: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
        original_quantile = bootstrap.np.quantile

        def quantile_spy(values: np.ndarray, q: np.ndarray, **kwargs: object) -> np.ndarray:
            quantile_calls.append((values.copy(), q.copy(), dict(kwargs)))
            return original_quantile(values, q, **kwargs)

        expected_bit_generator = np.random.PCG64(44120260808)
        with patch.object(bootstrap.np.random, "Generator", return_value=spy) as generator_ctor:
            with patch.object(bootstrap.np, "quantile", side_effect=quantile_spy):
                result = bootstrap.paired_dataset_bootstrap_antigravity(rows)

        generator_ctor.assert_called_once()
        bit_generator = generator_ctor.call_args.args[0]
        self.assertIsInstance(bit_generator, np.random.PCG64)
        self.assertEqual(bit_generator.state, expected_bit_generator.state)
        np.testing.assert_array_equal(spy.indices[:7], np.asarray([0, 1, 2, 1, 2, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(np.bincount(spy.indices), np.asarray([1, 558, 558]))
        self.assertEqual(len(spy.calls), 20_000)
        expected_kwargs = {
            "low": 0,
            "high": 1117,
            "size": 1117,
            "dtype": np.int64,
            "endpoint": False,
        }
        for args, kwargs in spy.calls:
            self.assertEqual(args, ())
            self.assertEqual(kwargs, expected_kwargs)

        # Independently precomputed from the exact binary row tally 1/558/558.
        expected_micro = float.fromhex("0x1.787a03e842630p-4")
        expected_macro = float.fromhex("0x1.ffa7fe2aa6e20p-2")
        self.assertEqual(len(quantile_calls), 2)
        for values, quantiles, kwargs in quantile_calls:
            self.assertEqual(values.dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(quantiles, np.asarray([0.025, 0.975], dtype=np.float64))
            self.assertEqual(kwargs, {"method": "linear"})
        np.testing.assert_array_equal(
            quantile_calls[0][0], np.full(20_000, expected_micro, dtype=np.float64),
        )
        np.testing.assert_array_equal(
            quantile_calls[1][0], np.full(20_000, expected_macro, dtype=np.float64),
        )
        self.assertEqual(result.micro_ci, (expected_micro, expected_micro))
        self.assertEqual(result.macro_ci, (expected_macro, expected_macro))
        for interval in (result.micro_ci, result.macro_ci):
            for endpoint in interval:
                self.assertIs(type(endpoint), float)

    def test_bootstrap_rejects_shape_and_nonfinite_rows(self) -> None:
        """Only the exact 1,117-by-9 finite row contract is accepted."""
        with self.assertRaisesRegex(ValueError, "bootstrap rows differ"):
            bootstrap.paired_dataset_bootstrap_antigravity(np.zeros((1116, 9), dtype=np.float64))
        invalid = np.zeros((1117, 9), dtype=np.float64)
        invalid[4, 8] = math.nan
        with self.assertRaisesRegex(ValueError, "bootstrap rows differ"):
            bootstrap.paired_dataset_bootstrap_antigravity(invalid)

    def test_bootstrap_module_allows_only_exact_pure_ast_capabilities(self) -> None:
        """I/O, readers, dynamic imports, accelerators, and process pools cannot enter."""
        source = bootstrap.__loader__.get_source(bootstrap.__name__)
        self.assertIsNotNone(source)
        tree = ast.parse(source or "")
        imports = [
            _import_signature_antigravity(node)
            for node in sorted(
                (
                    item
                    for item in ast.walk(tree)
                    if isinstance(item, (ast.Import, ast.ImportFrom))
                ),
                key=lambda item: (item.lineno, item.col_offset),
            )
        ]
        self.assertEqual(imports, [
            ("from", 0, "__future__", (("annotations", None),)),
            ("from", 0, "collections.abc", (("Sequence", None),)),
            ("import", 0, None, (("math", None),)),
            ("import", 0, None, (("numpy", "np"),)),
            ("from", 1, "contracts_antigravity", (("BootstrapResultAntigravity", None),)),
        ])
        self.assertEqual(
            [
                node.name
                for node in sorted(
                    (
                        item
                        for item in ast.walk(tree)
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                    key=lambda item: (item.lineno, item.col_offset),
                )
            ],
            [
                "project_f1_antigravity",
                "require_numpy_2_4_6_antigravity",
                "paired_dataset_bootstrap_antigravity",
            ],
        )
        module_assignments = [
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        self.assertEqual(module_assignments, ["__all__"])

        call_names = {
            _call_name_antigravity(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        self.assertEqual(call_names, {
            "<expression>.all",
            "<expression>.mean",
            "<expression>.sum",
            "BootstrapResultAntigravity",
            "RuntimeError",
            "ValueError",
            "any",
            "float",
            "isinstance",
            "len",
            "math.isfinite",
            "np.asarray",
            "np.empty",
            "np.isfinite",
            "np.quantile",
            "np.random.Generator",
            "np.random.PCG64",
            "project_f1_antigravity",
            "range",
            "require_numpy_2_4_6_antigravity",
            "rng.integers",
            "tuple",
        })
        identifiers = {
            node.id.casefold()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        } | {
            node.attr.casefold()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        prohibited_exact = {
            "__import__", "compile", "database", "eval", "evaluator", "exec",
            "importlib", "input", "labels", "location", "mlx", "mps",
            "multiprocessing", "open", "pool", "process", "pytorch", "reader",
            "sqlite", "sqlite3", "torch", "writer",
        }
        prohibited_patterns = (
            "anomaly_location", "dataset_reader", "evaluate", "label_reader",
            "load_anomaly", "load_dataset", "load_label", "read_anomaly",
            "read_dataset", "read_label",
        )
        violations = {
            identifier
            for identifier in identifiers
            if identifier in prohibited_exact
            or any(pattern in identifier for pattern in prohibited_patterns)
        }
        self.assertEqual(violations, set())


if __name__ == "__main__":
    unittest.main()
