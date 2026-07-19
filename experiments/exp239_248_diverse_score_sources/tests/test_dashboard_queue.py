from __future__ import annotations

import unittest

from scratch.run_exp239_248_sequential import _experiment_id, build_queue


class DashboardQueueTest(unittest.TestCase):
    def test_queue_order_is_239_through_248(self)->None:
        self.assertEqual([item["experiment"] for item in build_queue()],list(range(239,249)))

    def test_queue_uses_operational_names(self)->None:
        for item in build_queue():self.assertNotEqual(item["source"],f"experiment_{item['experiment']}")

    def test_canonical_queue_ids_match_dashboard_catalog(self)->None:
        self.assertEqual(
            _experiment_id(239),
            "experiment_239_multiscale_wavelet_energy_deviation",
        )


if __name__=="__main__":unittest.main()
