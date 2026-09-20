"""Behavior contracts for the approved Exp468 NumPy fallback decision."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import unittest

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.acceleration_antigravity import (
    AccelerationAssessmentAntigravity,
    BackendAssessmentAntigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.backend_selection_antigravity import (
    BackendDecisionAntigravity,
    finalize_backend_decision_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
BENCHMARK_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/exp468_acceleration_benchmark_antigravity.json"
)
PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY = (
    "938d83b0319346270a6cc32b7b6ba3efb88122dc1bf431863d8030f4539fb63b"
)


def _backend_from_payload_antigravity(
    payload: dict[str, object],
) -> BackendAssessmentAntigravity:
    return BackendAssessmentAntigravity(
        **{
            field.name: payload[field.name]
            for field in dataclasses.fields(BackendAssessmentAntigravity)
        },
    )


def _artifact_assessment_antigravity() -> tuple[
    dict[str, object], AccelerationAssessmentAntigravity,
]:
    payload = json.loads(BENCHMARK_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    workload = payload["workload"]
    assert isinstance(workload, dict)
    backend_payloads = payload["backends"]
    assert isinstance(backend_payloads, dict)
    assessment = AccelerationAssessmentAntigravity(
        required_speedup=payload["timing_protocol"]["required_speedup"],
        workload_counts={
            key: workload[key]
            for key in ("fit_raw", "query_raw", "transformed_rows", "frames")
        },
        backends={
            name: _backend_from_payload_antigravity(backend)
            for name, backend in backend_payloads.items()
        },
        selected_backend=payload["selected_backend"],
        gate_passed=payload["gate_passed"],
    )
    return payload, assessment


def _passing_assessment_antigravity(
    failed: AccelerationAssessmentAntigravity,
) -> AccelerationAssessmentAntigravity:
    numba = dataclasses.replace(
        failed.backends["numba"],
        parity_passed=True,
        deterministic=True,
        speedup=1.1,
        eligible=True,
        fallback_reason=None,
    )
    return AccelerationAssessmentAntigravity(
        required_speedup=failed.required_speedup,
        workload_counts=failed.workload_counts,
        backends={**failed.backends, "numba": numba},
        selected_backend="numba",
        gate_passed=True,
    )


class BackendSelectionAntigravityTest(unittest.TestCase):
    def test_approved_artifact_finalizes_numpy_without_rewriting_gate_antigravity(
        self,
    ) -> None:
        payload, assessment = _artifact_assessment_antigravity()
        authorization = payload["fallback_authorization"]
        approval_path = ROOT_ANTIGRAVITY / authorization["approval_document_path"]
        approval_sha256 = hashlib.sha256(approval_path.read_bytes()).hexdigest()

        decision = finalize_backend_decision_antigravity(
            assessment,
            fallback_approval_sha256=approval_sha256,
        )

        self.assertEqual(payload["status"], "TASK_3_COMPLETE_NUMPY_FALLBACK_AUTHORIZED")
        self.assertIs(payload["gate_passed"], False)
        self.assertEqual(payload["selected_backend"], "numpy")
        self.assertIs(payload["fallback_authorized"], True)
        self.assertIs(authorization["authorized"], True)
        self.assertEqual(
            authorization["pre_approval_benchmark_sha256"],
            PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY,
        )
        self.assertEqual(authorization["approval_document_sha256"], approval_sha256)
        self.assertEqual(decision.selected_backend, "numpy")
        self.assertIs(decision.acceleration_gate_passed, False)
        self.assertIs(decision.fallback_authorized, True)
        self.assertEqual(decision.approval_evidence_sha256, approval_sha256)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decision.selected_backend = "numba"

    def test_failed_assessment_without_approval_cannot_finalize_antigravity(
        self,
    ) -> None:
        _, assessment = _artifact_assessment_antigravity()

        with self.assertRaisesRegex(PermissionError, "approval"):
            finalize_backend_decision_antigravity(assessment)

    def test_gate_pass_requires_eligible_parity_determinism_and_speed_antigravity(
        self,
    ) -> None:
        _, failed = _artifact_assessment_antigravity()
        passing = _passing_assessment_antigravity(failed)

        decision = finalize_backend_decision_antigravity(passing)
        self.assertIsInstance(decision, BackendDecisionAntigravity)
        self.assertEqual(decision.selected_backend, "numba")
        self.assertIs(decision.acceleration_gate_passed, True)
        self.assertIs(decision.fallback_authorized, False)
        self.assertIsNone(decision.approval_evidence_sha256)

        invalid_values = (
            ("eligible", False),
            ("parity_passed", False),
            ("deterministic", False),
            ("speedup", 1.0999999999999999),
            ("speedup", float("nan")),
        )
        for field, value in invalid_values:
            with self.subTest(field=field):
                candidate = dataclasses.replace(
                    passing.backends["numba"],
                    **{field: value},
                )
                invalid = dataclasses.replace(
                    passing,
                    backends={**passing.backends, "numba": candidate},
                )
                with self.assertRaisesRegex(ValueError, field):
                    finalize_backend_decision_antigravity(invalid)

    def test_failed_fallback_requires_every_candidate_failure_reason_antigravity(
        self,
    ) -> None:
        _, failed = _artifact_assessment_antigravity()
        invalid_numba = dataclasses.replace(
            failed.backends["numba"],
            fallback_reason="  ",
        )
        invalid = dataclasses.replace(
            failed,
            backends={**failed.backends, "numba": invalid_numba},
        )

        with self.assertRaisesRegex(ValueError, "failure reason"):
            finalize_backend_decision_antigravity(
                invalid,
                fallback_approval_sha256="a" * 64,
            )

    def test_failed_fallback_requires_lowercase_sha256_evidence_antigravity(
        self,
    ) -> None:
        _, failed = _artifact_assessment_antigravity()

        for malformed in ("", "a" * 63, "A" * 64, "g" * 64):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                    finalize_backend_decision_antigravity(
                        failed,
                        fallback_approval_sha256=malformed,
                    )


if __name__ == "__main__":
    unittest.main()
