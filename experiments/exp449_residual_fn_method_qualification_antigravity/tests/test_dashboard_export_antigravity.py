"""Regression coverage for Task 5's local-only status projection."""
from __future__ import annotations

import unittest


class DashboardExportAntigravityTests(unittest.TestCase):
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
