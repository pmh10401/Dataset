"""Subprocess audit of the frozen official aeon RDST surface."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import json
import subprocess

import experiments.exp451_rdst_local_gap_qualification_antigravity.method_audit_antigravity as audit_module

from experiments.exp451_rdst_local_gap_qualification_antigravity.method_audit_antigravity import (
    RdstAuditResultAntigravity,
    audit_rdst_antigravity,
)


class MethodAuditAntigravityTest(unittest.TestCase):
    def test_scientific_runtime_checks_exact_versions_then_audits_once_antigravity(self):
        from unittest.mock import patch
        from experiments.exp451_rdst_local_gap_qualification_antigravity import method_audit_antigravity as module
        expected=object()
        versions={"aeon":"1.4.0","numpy":"2.3.5","scikit-learn":"1.7.2"}
        with patch.object(module.sys,"executable",module.FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.fixed_interpreter), patch.object(module.sys,"version_info",(3,14,6)), patch.object(module.importlib.metadata,"version",side_effect=versions.__getitem__), patch.object(module,"audit_rdst_antigravity",return_value=expected) as audit:
            self.assertIs(module.validate_scientific_runtime_antigravity(),expected); audit.assert_called_once_with()
        with patch.object(module.sys,"executable","/wrong"), patch.object(module,"audit_rdst_antigravity") as audit, self.assertRaises(ValueError): module.validate_scientific_runtime_antigravity()
        audit.assert_not_called()
    def valid_response_antigravity(self) -> dict[str, object]:
        return {
            "schema": "exp451.rdst_audit.response.v1", "aeon_version": "1.4.0",
            "public_class_path": "aeon.transformations.collection.shapelet_based.RandomDilatedShapeletTransform",
            "implementation_class_path": "aeon.transformations.collection.shapelet_based._dilated_shapelet_transform.RandomDilatedShapeletTransform",
            "constructor_signature": "(max_shapelets: int = 10000, shapelet_lengths: List[int] | numpy.ndarray | None = None, proba_normalization: float = 0.8, threshold_percentiles: List[float] | numpy.ndarray | None = None, alpha_similarity: float = 0.5, use_prime_dilations: bool = False, random_state: int | None = None, n_jobs: int = 1)",
            "fit_signature": "(self, X, y=None)", "fit_transform_signature": "(self, X, y=None)",
            "source_sha256": "900397653dd89bb4f04b33182d87e0f682555568f7f50690ef9a78ba946334f2",
            "seed": 20260721, "max_shapelets": 1024, "n_jobs": 1, "y_is_none": True,
            "two_dimensional": True, "finite": True, "deterministic": True,
            "retained_shapelets": 2, "feature_dimension": 6,
            "first_output_sha256": "a" * 64, "second_output_sha256": "a" * 64,
        }

    def completed_antigravity(self, payload: object, *, stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
        stdout = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n" if type(payload) is dict else str(payload)
        return subprocess.CompletedProcess(["python"], returncode, stdout=stdout, stderr=stderr)

    def test_fixed_interpreter_audit_binds_source_signature_and_label_free_fixture_antigravity(self) -> None:
        result = audit_rdst_antigravity()
        self.assertIs(type(result), RdstAuditResultAntigravity)
        self.assertEqual(result.aeon_version, "1.4.0")
        self.assertEqual(result.public_class_path, "aeon.transformations.collection.shapelet_based.RandomDilatedShapeletTransform")
        self.assertEqual(result.implementation_class_path, "aeon.transformations.collection.shapelet_based._dilated_shapelet_transform.RandomDilatedShapeletTransform")
        self.assertEqual(result.source_sha256, "900397653dd89bb4f04b33182d87e0f682555568f7f50690ef9a78ba946334f2")
        self.assertEqual(result.constructor_signature, "(max_shapelets: int = 10000, shapelet_lengths: List[int] | numpy.ndarray | None = None, proba_normalization: float = 0.8, threshold_percentiles: List[float] | numpy.ndarray | None = None, alpha_similarity: float = 0.5, use_prime_dilations: bool = False, random_state: int | None = None, n_jobs: int = 1)")
        self.assertEqual(result.seed, 20260721)
        self.assertEqual(result.max_shapelets, 1024)
        self.assertTrue(result.y_is_none)
        self.assertTrue(result.deterministic)
        self.assertGreater(result.retained_shapelets, 0)
        self.assertEqual(result.feature_dimension, 3 * result.retained_shapelets)

    def test_audit_rejects_noncanonical_and_adversarial_subprocess_responses_antigravity(self) -> None:
        valid = self.valid_response_antigravity()
        cases: list[tuple[str, object, str, int]] = [
            ("nonzero", valid, "", 1), ("stderr", valid, "unexpected", 0),
            ("duplicate", '{"schema":"exp451.rdst_audit.response.v1","schema":"exp451.rdst_audit.response.v1"}', "", 0),
            ("whitespace", json.dumps(valid, sort_keys=True) + "\n", "", 0),
            ("key_order", json.dumps(dict(reversed(tuple(valid.items()))), separators=(",", ":")) + "\n", "", 0),
            ("extra", {**valid, "extra": 1}, "", 0),
            ("missing", {key: value for key, value in valid.items() if key != "finite"}, "", 0),
            ("wrong_type", {**valid, "seed": True}, "", 0),
            ("wrong_public_path", {**valid, "public_class_path": "wrong.path"}, "", 0),
            ("wrong_signature", {**valid, "fit_signature": "wrong"}, "", 0),
            ("false_boolean", {**valid, "finite": False}, "", 0),
        ]
        for name, payload, stderr, returncode in cases:
            with self.subTest(name=name):
                completed = self.completed_antigravity(payload, stderr=stderr, returncode=returncode)
                with patch.object(audit_module.subprocess, "run", return_value=completed), self.assertRaises((RuntimeError, ValueError)):
                    audit_rdst_antigravity()

    def test_audit_rejects_timeout_and_mutable_request_alias_antigravity(self) -> None:
        self.assertFalse(hasattr(audit_module, "_REQUEST_ANTIGRAVITY"))
        request = audit_module._audit_request_antigravity()
        request["seed"] = 1
        self.assertEqual(audit_module._audit_request_antigravity()["seed"], 20260721)
        with patch.object(audit_module.subprocess, "run", side_effect=subprocess.TimeoutExpired(["python"], 1)), self.assertRaisesRegex(RuntimeError, "timed out"):
            audit_rdst_antigravity()

    def test_audit_binding_uses_observed_class_path_antigravity(self) -> None:
        changed = self.valid_response_antigravity()
        changed["implementation_class_path"] = "aeon.transformations.collection.shapelet_based.changed.RandomDilatedShapeletTransform"
        with patch.object(audit_module.subprocess, "run", return_value=self.completed_antigravity(changed)), self.assertRaisesRegex(ValueError, "binding"):
            audit_rdst_antigravity()

    def test_exact_warning_policy_rejects_appended_and_wrong_categories_antigravity(self) -> None:
        allowed = audit_module._warning_is_allowed_antigravity(audit_module._KNOWN_AEON_WARNING_MESSAGE_ANTIGRAVITY, UserWarning)
        self.assertTrue(allowed)
        self.assertFalse(audit_module._warning_is_allowed_antigravity(audit_module._KNOWN_AEON_WARNING_MESSAGE_ANTIGRAVITY + " appended", UserWarning))
        self.assertFalse(audit_module._warning_is_allowed_antigravity(audit_module._KNOWN_AEON_WARNING_MESSAGE_ANTIGRAVITY, RuntimeWarning))


if __name__ == "__main__":
    unittest.main()
