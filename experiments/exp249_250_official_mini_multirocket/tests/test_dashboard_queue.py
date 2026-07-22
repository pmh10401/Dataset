from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DashboardQueueContractTest(unittest.TestCase):
    def test_queue_contains_exp249_then_exp250(self) -> None:
        queue = _load("exp249_queue", ROOT / "scratch" / "run_exp249_250_sequential.py")
        self.assertEqual([row["experiment"] for row in queue.build_queue()], [249, 250])
        self.assertTrue(all(row["workers"] == 7 for row in queue.build_queue()))

    def test_dashboard_catalog_contains_both_experiments(self) -> None:
        dashboard = _load("exp249_dashboard", ROOT / "serve_rank_dashboard.py")
        for experiment, source in (
            (249, "official_minirocket"),
            (250, "official_multirocket"),
        ):
            item = dashboard.EXPERIMENTS[f"experiment_{experiment}_{source}"]
            self.assertFalse(item["comparison_eligible"])
            self.assertEqual(item["autonomous_metric_scope"], "frozen_exp236_unchanged")


if __name__ == "__main__":
    unittest.main()
