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
        self._next_value = 0

    def integers(self, *args: object, **kwargs: object) -> np.ndarray:
        self.calls.append((args, kwargs))
        value = self._next_value
        self._next_value = 1 - self._next_value
        return np.full(1117, value, dtype=np.int64)


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
        """A hand-derived alternating draw proves paired baseline/union row sampling."""
        rows = np.zeros((1117, 9), dtype=np.float64)
        rows[0, :4] = (1, 0, 1, 0)
        rows[0, 4:8] = (1, 0, 1, 0)
        rows[0, 8] = 0.25
        rows[1, :4] = (0, 0, 1, 1)
        rows[1, 4:8] = (1, 0, 0, 1)
        rows[1, 8] = 0.75
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

        self.assertEqual(len(quantile_calls), 2)
        for _, quantiles, kwargs in quantile_calls:
            np.testing.assert_array_equal(quantiles, np.asarray([0.025, 0.975], dtype=np.float64))
            self.assertEqual(kwargs, {"method": "linear"})
        self.assertEqual(result.micro_ci, (0.0, 1.0))
        self.assertEqual(result.macro_ci, (0.25, 0.75))
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

    def test_bootstrap_module_has_no_evaluator_or_sqlite_dependency(self) -> None:
        """The pure calculation boundary must remain label-free and database-free."""
        source = bootstrap.__loader__.get_source(bootstrap.__name__)
        self.assertIsNotNone(source)
        tree = ast.parse(source or "")
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if node.module
        )
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("evaluation_antigravity", source or "")


if __name__ == "__main__":
    unittest.main()
