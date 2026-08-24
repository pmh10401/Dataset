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

    def test_html_shows_concrete_fit_shift_values_antigravity(self) -> None:
        """Break caught: result HTML names peak/shift without retained numeric evidence."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(published_result_root_antigravity(directory))
        html = export_dashboard_antigravity(inspected)
        first_record = next(iter(inspected.records.values()))
        expected_shifts = "[" + ", ".join(str(int(value)) for value in first_record.fit_shifts) + "]"
        self.assertIn(f"fit shifts: {expected_shifts}", html)

    def test_html_shows_concrete_peak_absence_count_antigravity(self) -> None:
        """Break caught: result HTML omits how many ACF peaks were actually absent."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(published_result_root_antigravity(directory))
        html = export_dashboard_antigravity(inspected)
        expected_absent = sum(
            int((~record.fit_peak_presence).sum())
            for record in inspected.records.values()
        )
        self.assertIn(f"peak 부재 감지 수: {expected_absent}", html)

    def test_export_leaves_no_module_level_qualification_state_antigravity(self) -> None:
        """Break caught: export hides mutable qualification in module globals."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(published_result_root_antigravity(directory))
        import importlib
        module = importlib.import_module("experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.dashboard_export_antigravity")
        module.export_dashboard_antigravity(inspected)
        self.assertFalse(hasattr(module, "qualification"))


if __name__ == "__main__":
    unittest.main()
