from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PREDICTION_FILES = [
    ROOT / "experiments/exp265_shadow_validation/contracts.py",
    ROOT / "experiments/exp265_shadow_validation/routing.py",
    ROOT / "experiments/exp265_shadow_validation/spool.py",
    ROOT / "experiments/exp265_shadow_validation/exporter.py",
    ROOT / "scratch/run_exp265_shadow_exporter.py",
]
FORBIDDEN_ARGUMENTS = {
    "y_test", "label", "labels", "label_path", "actual_anomaly_positions",
    "anomaly_positions", "test_precision", "test_f1", "test_recall", "test_metric",
}


class StructureTest(unittest.TestCase):
    def test_prediction_boundary_has_no_forbidden_arguments(self) -> None:
        violations: list[str] = []
        for path in PREDICTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = {argument.arg.lower() for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
                    forbidden = names & FORBIDDEN_ARGUMENTS
                    if forbidden:
                        violations.append(f"{path.name}:{node.lineno}:{node.name}:{sorted(forbidden)}")
        self.assertEqual(violations, [])

    def test_prediction_entrypoint_does_not_read_historical_database(self) -> None:
        source = (ROOT / "scratch/run_exp265_shadow_exporter.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("univariate_ts.db", source)
        self.assertNotIn("y_test", source)
        self.assertNotIn("historical_test", source)

    def test_routing_exposes_only_label_free_input(self) -> None:
        tree = ast.parse((ROOT / "experiments/exp265_shadow_validation/routing.py").read_text(encoding="utf-8"))
        route = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "route_shadow_run")
        self.assertEqual([argument.arg for argument in route.args.args], ["inputs"])


if __name__ == "__main__":
    unittest.main()
