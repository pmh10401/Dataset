"""Regression coverage for Task 5's local-only status projection."""
from __future__ import annotations

import unittest


class DashboardExportAntigravityTests(unittest.TestCase):
    def test_public_result_requires_complete_deterministic_qualification_projection_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import public_result_antigravity

        global_f1 = (2.0 * 1931) / (2 * 1931 + 314 + 1987)
        candidates = [{"family": family, "method": method, "target_fn_delta": -1, "gates": {"global_fn": True, "global_fp": True, "global_f1": True, "target_fn_delta": True}, "qualified": True, "global": {"tp": 1931, "fp": 314, "fn": 1987, "tn": 191668, "f1": global_f1}} for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY for method in methods]
        payload = {"schema": "exp449.public-method-qualification.v1", "experiment": "Exp449", "baseline": {"tp": 1930, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935086011}, "candidates": candidates, "families": [family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY], "status": "synthetic_complete", "repeatable": True, "qualified_candidate_count": 16, "routing_research_eligible": True, "runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300}}
        self.assertEqual(public_result_antigravity(payload), payload)
    def test_public_result_exposes_aggregate_only_allowlist_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import public_result_antigravity

        from experiments.exp449_residual_fn_method_qualification_antigravity.config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY
        global_f1 = (2.0 * 1931) / (2 * 1931 + 314 + 1987)
        payload = {"schema": "exp449.public-method-qualification.v1", "experiment": "Exp449", "baseline": {"tp": 1930, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935086011}, "candidates": [{"family": family, "method": method, "target_fn_delta": -1, "gates": {"global_fn": True, "global_fp": True, "global_f1": True, "target_fn_delta": True}, "qualified": True, "global": {"tp": 1931, "fp": 314, "fn": 1987, "tn": 191668, "f1": global_f1}} for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY for method in methods], "families": [family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY], "status": "synthetic_complete", "repeatable": True, "qualified_candidate_count": 16, "routing_research_eligible": True, "runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300}}
        self.assertEqual(public_result_antigravity(payload), payload)

    def test_public_result_rejects_forbidden_or_unallowlisted_nested_fields_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import public_result_antigravity

        valid = {"schema": "exp449.public-method-qualification.v1", "experiment": "Exp449", "baseline": {"tp": 1930, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935}, "candidates": [], "families": ["Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves"], "status": "synthetic_complete", "runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300}}
        for mutation in ({"dataset": "Crop_normal_1"}, {"candidates": [{"family": "Crop", "raw_labels": [1]}]}, {"runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300, "path": "/tmp"}}):
            candidate = dict(valid); candidate.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): public_result_antigravity(candidate)
    def test_public_status_returns_exact_twelve_allowlisted_fields_antigravity(self) -> None:
        """Break caught: leaking implementation details or omitting a local status field."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import (  # noqa: PLC0415
            STATUS_FIELDS_ANTIGRAVITY,
            public_status_antigravity,
        )

        source = {
            "schema": "exp449", "experiment_id": "Exp449", "phase": "prediction", "status": "running",
            "reviewed_commit": "a" * 40, "target_family_count": 8, "candidate_pair_count": 16,
            "dataset_unit_total": 96, "dataset_unit_completed": 0, "method_family": "Crop",
            "backend": "cpu", "label_access_started": False,
        }
        self.assertEqual(tuple(public_status_antigravity(source)), STATUS_FIELDS_ANTIGRAVITY)
        self.assertEqual(public_status_antigravity(source), source)

    def test_public_status_rejects_sensitive_fields_antigravity(self) -> None:
        """Break caught: recursively allowing dataset, filesystem, hash, log, or database data."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import public_status_antigravity  # noqa: PLC0415

        valid = {
            "schema": "exp449", "experiment_id": "Exp449", "phase": "prediction", "status": "running",
            "reviewed_commit": "a" * 40, "target_family_count": 8, "candidate_pair_count": 16,
            "dataset_unit_total": 96, "dataset_unit_completed": 0, "method_family": "Crop",
            "backend": "cpu", "label_access_started": False,
        }
        for key in ("dataset", "instance_index", "sha256", "path", "log", "database"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                public_status_antigravity({**valid, key: "forbidden"})

    def test_public_status_rejects_non_allowlisted_fields_antigravity(self) -> None:
        """Break caught: silently projecting a snapshot that carries undeclared fields."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import public_status_antigravity  # noqa: PLC0415
        valid = {"schema": "exp449", "experiment_id": "Exp449", "phase": "prediction", "status": "running", "reviewed_commit": "a" * 40, "target_family_count": 8, "candidate_pair_count": 16, "dataset_unit_total": 96, "dataset_unit_completed": 0, "method_family": "Crop", "backend": "cpu", "label_access_started": False}
        with self.assertRaises(ValueError):
            public_status_antigravity({**valid, "worker_elapsed_ms": 1})

    def test_public_status_rejects_nested_sensitive_values_and_wrong_types_antigravity(self) -> None:
        """Break caught: carrying nested sensitive material or non-scalar status values through projection."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.dashboard_export_antigravity import public_status_antigravity  # noqa: PLC0415
        valid = {"schema": "exp449", "experiment_id": "Exp449", "phase": "prediction", "status": "running", "reviewed_commit": "a" * 40, "target_family_count": 8, "candidate_pair_count": 16, "dataset_unit_total": 96, "dataset_unit_completed": 0, "method_family": "Crop", "backend": "cpu", "label_access_started": False}
        for field, value in (("method_family", {"nested": {"database": "forbidden"}}), ("backend", ["cpu"])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                public_status_antigravity({**valid, field: value})
