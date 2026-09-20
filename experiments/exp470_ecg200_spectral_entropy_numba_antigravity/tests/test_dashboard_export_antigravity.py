from __future__ import annotations

import tempfile
import unittest

from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.tests.test_replay_antigravity import ResultReplayTestAntigravity


class DashboardExportTestAntigravity(unittest.TestCase):
    def test_html_is_portable_escaped_and_checks_only_replay_antigravity(self) -> None:
        """Break caught: result HTML claims private/official completion or renders raw text unsafely."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.replay_antigravity import inspect_result_antigravity

        helper = ResultReplayTestAntigravity()
        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(helper._published_antigravity(directory))
        html = export_dashboard_antigravity(inspected)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("결과 replay 검증", html)
        self.assertIn('name="replay" checked', html)
        self.assertIn('name="replay" checked disabled', html)
        self.assertIn('name="private"', html)
        self.assertIn('name="official"', html)
        self.assertIn('name="test"', html)
        self.assertIn('name="fn"', html)
        self.assertIn('name="routing"', html)
        for name in ("private", "official", "test", "fn", "routing"):
            self.assertIn(f'name="{name}" disabled', html)
            self.assertNotIn(f'name="{name}" checked', html)
        self.assertNotIn('name="private" checked', html)
        self.assertNotIn('name="official" checked', html)
        self.assertIn("awaiting_private_checkpoint", html)
        self.assertIn("C0", html)
        self.assertIn("C11", html)


if __name__ == "__main__":
    unittest.main()
