from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    LengthGroup,
    ProcessedGroup,
    RawSeries,
)
from experiments.exp425_selective_recovery.contracts import (
    EligibilityAssessment,
    Exp425RunProvenance,
    FoldIdentityCounts,
    GroupEligibilityEvidence,
    ImportedRecordEnvelope,
    NativeRecordEnvelope,
    SourceProvenGroupEvidence,
)
from experiments.exp425_selective_recovery import checkpoints


class EligibilityContractTests(unittest.TestCase):
    def make_evidence(
        self,
        *,
        status: str = "eligible",
        processed_status: str = "diagnostic_only",
        folds: tuple[FoldIdentityCounts, ...] | None = None,
        final_fit: int | None = None,
        final_validation: int | None = None,
    ) -> GroupEligibilityEvidence:
        if folds is None:
            folds = (FoldIdentityCounts(10, 5, 9, 5),) * 5
        reason = None if status == "eligible" else "oof_identity_insufficient"
        return GroupEligibilityEvidence(
            dataset="Synthetic_normal_1",
            length=64,
            processed_status=processed_status,
            train_identity_sha256="b" * 64,
            fold_contract_sha256="a" * 64,
            unique_identity_count=15,
            folds=folds,
            final_fit=final_fit,
            final_validation=final_validation,
            status=status,
            reason=reason,
        )

    def test_fold_counts_reject_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            FoldIdentityCounts(1, 1, -1, 0)

    def test_evidence_rejects_non_sha256_fold_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "fold_contract_sha256"):
            GroupEligibilityEvidence(
                dataset="Synthetic_normal_1",
                length=64,
                processed_status="diagnostic_only",
                train_identity_sha256="b" * 64,
                fold_contract_sha256="A" * 64,
                unique_identity_count=15,
                folds=(FoldIdentityCounts(10, 5, 9, 5),) * 5,
                final_fit=None,
                final_validation=None,
                status="eligible",
                reason=None,
            )

    def test_insufficient_evidence_requires_a_failing_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "failing split"):
            self.make_evidence(
                status="oof_identity_insufficient",
                folds=(FoldIdentityCounts(10, 5, 9, 5),) * 5,
            )

    def test_eligible_evidence_rejects_small_early_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "eligible split"):
            self.make_evidence(
                status="eligible",
                folds=(FoldIdentityCounts(10, 5, 9, 1),) * 5,
            )

    def test_preprocessing_abstained_evidence_requires_empty_fold_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "preprocessing_abstained"):
            self.make_evidence(
                status="preprocessing_abstained",
                processed_status="zero_scale_abstain",
            )

    def test_candidate_eligible_requires_final_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_fit"):
            self.make_evidence(processed_status="candidate_eligible")

    def test_diagnostic_only_rejects_final_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "diagnostic_only"):
            self.make_evidence(final_fit=10, final_validation=5)

    def test_contracts_are_canonically_immutable(self) -> None:
        counts = FoldIdentityCounts(10, 5, 9, 5)
        evidence = self.make_evidence(folds=[counts] * 5)  # type: ignore[arg-type]
        with self.assertRaises(FrozenInstanceError):
            counts.oof_fit = 0  # type: ignore[misc]
        self.assertIsInstance(evidence.folds, tuple)
        with self.assertRaises(FrozenInstanceError):
            evidence.status = "oof_identity_insufficient"  # type: ignore[misc]
        assessment = EligibilityAssessment(groups=(), evidence={})
        with self.assertRaises(TypeError):
            assessment.evidence[64] = evidence  # type: ignore[index]

    def test_assessment_rejects_unsorted_or_duplicate_group_lengths(self) -> None:
        def abstained_group(length: int) -> ProcessedGroup:
            return ProcessedGroup(
                group=LengthGroup(length=length, series=(RawSeries(0, (1.0,) * length),)),
                raw_center=0.0,
                raw_scale=0.0,
                difference_center=0.0,
                difference_scale=0.0,
                status="zero_scale_abstain",
            )

        def abstained_evidence(length: int) -> GroupEligibilityEvidence:
            return GroupEligibilityEvidence(
                dataset="Synthetic_normal_1",
                length=length,
                processed_status="zero_scale_abstain",
                train_identity_sha256="b" * 64,
                fold_contract_sha256="a" * 64,
                unique_identity_count=1,
                folds=(),
                final_fit=None,
                final_validation=None,
                status="preprocessing_abstained",
                reason="zero_scale_abstain",
            )

        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            EligibilityAssessment(
                groups=(abstained_group(65), abstained_group(64)),
                evidence={64: abstained_evidence(64), 65: abstained_evidence(65)},
            )


class CheckpointPayloadContractTests(unittest.TestCase):
    def test_nested_caller_mutation_cannot_change_native_preimage_or_deepcopy(self) -> None:
        """Removing recursive detachment would let caller-owned nested payloads change a record."""
        digests = {name: sha256(name.encode()).hexdigest() for name in (
            "train_identity_sha256", "eligibility_sha256", "processed_groups_sha256", "model_evidence_sha256",
            "frozen_train_contract_sha256", "scoring_evidence_sha256", "result_evidence_sha256", "backend_audit_sha256",
        )}
        context = "c" * 64
        trace_body = {"schema": "exp425.native-trace.v1", "run_identity": "run", "dataset": "dataset", "run_context_sha256": context, **digests, "device": "mps"}
        trace = sha256(canonical_json_bytes(trace_body)).hexdigest()
        result = {"effective_result": {"status": "abstained", "result_evidence_sha256": digests["result_evidence_sha256"], "nested": ["stable"]}, "native_trace_body": trace_body}
        group = {"eligibility_sha256": digests["eligibility_sha256"], "processed_groups_sha256": digests["processed_groups_sha256"], "nested": {"value": "stable"}}
        model = {"train_identity_sha256": digests["train_identity_sha256"], "model_evidence_sha256": digests["model_evidence_sha256"], "frozen_train_contract_sha256": digests["frozen_train_contract_sha256"]}
        calibration = {"scoring_evidence_sha256": digests["scoring_evidence_sha256"]}
        body = {"schema_version": "exp425.dataset.v1", "dataset": "dataset", "origin": "exp425_native", "run_context_sha256": context, "result_payload": result, "group_evidence": group, "trace_sha256": trace, "model_hashes": model, "calibration_hashes": calibration, "backend_audit_sha256": digests["backend_audit_sha256"], "elapsed_seconds": 0.0, "reason": None}
        record = NativeRecordEnvelope("dataset", result, group, trace_sha256=trace, record_payload_sha256=sha256(canonical_json_bytes(body)).hexdigest(), model_hashes=model, calibration_hashes=calibration, elapsed_seconds=0.0, reason=None, **digests)
        before = canonical_json_bytes(checkpoints._native_record_body(context, record))
        result["effective_result"]["nested"].append("mutated")
        group["nested"]["value"] = "mutated"
        trace_body["device"] = "cpu"
        self.assertEqual(canonical_json_bytes(checkpoints._native_record_body(context, record)), before)
        self.assertEqual(canonical_json_bytes(checkpoints._native_record_body(context, deepcopy(record))), before)

    def test_imported_evidence_detaches_nested_source_payload(self) -> None:
        """Removing recursive detachment would change persisted import evidence after construction."""
        source = {"groups": [{"identity": "stable"}]}
        evidence = SourceProvenGroupEvidence(source, sha256(canonical_json_bytes(source)).hexdigest())
        record = ImportedRecordEnvelope("dataset", {"status": "abstained", "nested": ["stable"]}, evidence, *("a" * 64,) * 5, 0.0, None)
        before = canonical_json_bytes(checkpoints._import_record_body("b" * 64, record))
        source["groups"][0]["identity"] = "mutated"
        self.assertEqual(canonical_json_bytes(checkpoints._import_record_body("b" * 64, record)), before)


if __name__ == "__main__":
    unittest.main()
