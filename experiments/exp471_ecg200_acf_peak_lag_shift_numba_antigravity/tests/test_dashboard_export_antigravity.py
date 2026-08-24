from __future__ import annotations

import tempfile
import unittest

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.tests.test_replay_antigravity import (
    published_result_root_antigravity,
)


class DashboardExportTestAntigravity(unittest.TestCase):
    def test_html_checks_only_completed_replay_and_shows_exp471_evidence_antigravity(self) -> None:
        """Break caught: result HTML claims private/official completion or omits ACF evidence."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(published_result_root_antigravity(directory))
        html = export_dashboard_antigravity(inspected)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("결과 replay 검증", html)
        self.assertIn('name="replay" checked disabled', html)
        for name in ("private", "official", "test", "fn", "routing"):
            self.assertIn(f'name="{name}" disabled', html)
            self.assertNotIn(f'name="{name}" checked', html)
        self.assertIn("awaiting_private_checkpoint", html)
        for token in ("C0", "E12", "C12", "peak", "shift"):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
