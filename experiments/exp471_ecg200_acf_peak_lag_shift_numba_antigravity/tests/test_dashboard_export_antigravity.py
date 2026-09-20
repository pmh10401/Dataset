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

    def test_html_shows_aggregate_fit_and_query_shift_distributions_antigravity(self) -> None:
        """Break caught: result HTML shows one fold instead of aggregate shift distributions."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(published_result_root_antigravity(directory))
        html = export_dashboard_antigravity(inspected)
        scopes = ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        fit_values = [int(value) for record in inspected.records.values() for value in record.fit_shifts]
        query_values = [int(value) for record in inspected.records.values() for scope in scopes for value in record.query_shifts[scope]]
        histogram = lambda values: ", ".join(f"{value}:{values.count(value)}" for value in sorted(set(values)))
        self.assertIn(f"fit shift 분포: {histogram(fit_values)}", html)
        self.assertIn(f"query shift 분포: {histogram(query_values)}", html)

    def test_html_shows_fit_and_query_peak_absence_rows_antigravity(self) -> None:
        """Break caught: result HTML omits fit or query rows with an absent ACF peak."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_result_antigravity(published_result_root_antigravity(directory))
        html = export_dashboard_antigravity(inspected)
        scopes = ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        fit_absent = sum(
            int((~record.fit_peak_presence).any(axis=1).sum())
            for record in inspected.records.values()
        )
        query_absent = sum(
            int((~record.query_peak_presence[scope]).any(axis=1).sum())
            for record in inspected.records.values()
            for scope in scopes
        )
        self.assertIn(f"fit peak 없음 행: {fit_absent}", html)
        self.assertIn(f"query peak 없음 행: {query_absent}", html)

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
