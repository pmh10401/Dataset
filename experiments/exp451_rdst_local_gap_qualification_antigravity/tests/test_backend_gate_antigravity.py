from __future__ import annotations

import pickle
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp451_rdst_local_gap_qualification_antigravity.backend_gate_antigravity import (
    BackendDecisionAntigravity,
    BackendProbeAntigravity,
    backend_name_from_decision_antigravity,
    build_local_gap_backend_antigravity,
    decide_backend_antigravity,
    installed_backend_probe_antigravity,
)


class BackendGateAntigravityTest(unittest.TestCase):
    def probe_antigravity(self, **changes: object) -> BackendProbeAntigravity:
        values: dict[str, object] = {
            "name": "mlx", "available": True, "finite": True, "candidate_indices_match": True,
            "maximum_normalized_error": 0.0, "inclusive_cpu_seconds": 1.1,
            "inclusive_accelerator_seconds": 1.0, "versions": (("mlx", "fixture"),), "device": "fixture",
            "reason": "", "implementation_qualified": False,
        }
        values.update(changes)
        return BackendProbeAntigravity(**values)  # type: ignore[arg-type]

    def test_unqualified_passing_accelerator_probe_falls_back_to_factory_supported_cpu_antigravity(self) -> None:
        decision = decide_backend_antigravity(self.probe_antigravity())
        self.assertEqual(decision.selected_name, "cpu")
        self.assertIn("implementation_unqualified", decision.fallback_reason)
        self.assertIsNotNone(build_local_gap_backend_antigravity(backend_name_from_decision_antigravity(decision)))

    def test_exact_gate_boundary_rejects_every_failed_authority_antigravity(self) -> None:
        for change in ({"finite": False}, {"candidate_indices_match": False}, {"maximum_normalized_error": 1.00001e-5}, {"inclusive_cpu_seconds": 1.099}):
            with self.subTest(change=change):
                rejected = decide_backend_antigravity(self.probe_antigravity(**change))
                self.assertEqual(rejected.selected_name, "cpu")
                self.assertTrue(rejected.fallback_reason)

    def test_name_is_picklable_and_factory_returns_fresh_cpu_models_antigravity(self) -> None:
        self.assertEqual(pickle.loads(pickle.dumps("cpu")), "cpu")
        first = build_local_gap_backend_antigravity("cpu")
        second = build_local_gap_backend_antigravity("cpu")
        self.assertIsNot(first, second)
        reference = np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=float)
        first.fit(reference)
        self.assertTrue(np.isfinite(first.score(reference)).all())
        with self.assertRaises(ValueError):
            build_local_gap_backend_antigravity(np.str_("cpu"))

    def test_malformed_and_boolean_probe_fields_reject_antigravity(self) -> None:
        for change in ({"available": 1}, {"finite": 1}, {"maximum_normalized_error": float("nan")}, {"maximum_normalized_error": -0.1}, {"inclusive_cpu_seconds": True}, {"name": np.str_("mlx")}, {"versions": ()}, {"device": ""}, {"implementation_qualified": 1}):
            values = {"name": "mlx", "available": True, "finite": True, "candidate_indices_match": True, "maximum_normalized_error": 0.0, "inclusive_cpu_seconds": 1.1, "inclusive_accelerator_seconds": 1.0, "versions": (("mlx", "fixture"),), "device": "fixture", "reason": "", "implementation_qualified": False}
            values.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                BackendProbeAntigravity(**values)  # type: ignore[arg-type]

    def test_decision_rejects_direct_accelerator_construction_without_supported_qualified_probe_antigravity(self) -> None:
        with self.assertRaises(ValueError):
            BackendDecisionAntigravity("mlx", (self.probe_antigravity(),), "")
        self.assertEqual(decide_backend_antigravity(self.probe_antigravity(implementation_qualified=True)).selected_name, "cpu")

    def test_decision_requires_the_exact_single_probe_and_derived_fallback_reason_antigravity(self) -> None:
        probe = self.probe_antigravity()
        expected = decide_backend_antigravity(probe)
        for selected_name, probes, reason in (
            ("cpu", (), "reason"),
            ("cpu", (probe, probe), "reason"),
            ("cpu", (self.probe_antigravity(name="mps"),), expected.fallback_reason),
            ("cpu", (probe,), "arbitrary"),
        ):
            with self.subTest(probes=probes, reason=reason), self.assertRaises(ValueError):
                BackendDecisionAntigravity(selected_name, probes, reason)

    def test_installed_accelerator_probe_records_runtime_and_remains_fail_closed_antigravity(self) -> None:
        probe = installed_backend_probe_antigravity("mps")
        self.assertEqual(probe.name, "mps")
        self.assertTrue(probe.versions)
        self.assertTrue(probe.device)
        self.assertGreater(probe.inclusive_cpu_seconds, 0.0)
        self.assertGreater(probe.inclusive_accelerator_seconds, 0.0)
        self.assertEqual(decide_backend_antigravity(probe).selected_name, "cpu")

    def test_optional_runtime_failure_returns_closed_cpu_evidence_antigravity(self) -> None:
        import experiments.exp451_rdst_local_gap_qualification_antigravity.backend_gate_antigravity as module
        with patch.object(module.importlib.util, "find_spec", side_effect=RuntimeError("broken optional runtime")):
            probe = module.installed_backend_probe_antigravity("mlx")
        self.assertFalse(probe.available)
        self.assertIn("optional_runtime_error:RuntimeError", probe.reason)
        self.assertEqual(module.decide_backend_antigravity(probe).selected_name, "cpu")


if __name__ == "__main__":
    unittest.main()
