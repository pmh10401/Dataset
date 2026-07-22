from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.exp239_248_diverse_score_sources.foundation import stable_json_hash
from experiments.exp267_273_zero_support_fn_recovery import qualification as module
from experiments.exp267_273_zero_support_fn_recovery.config import (
    CORRELATION_COLLAPSE_ABS_SPEARMAN,
    CORRELATION_MIN_ALIGNED_TRAIN_OOF,
)
from experiments.exp267_273_zero_support_fn_recovery.qualification import (
    MembershipHashError,
    QualificationEvidence,
    collapse_correlated_families,
    collapse_votes,
    evaluate_frozen_qualification,
    qualify_sources,
)
from experiments.exp267_273_zero_support_fn_recovery.train_risk import (
    array_sha256,
    audit_train_risk,
    leave_one_out_upper_pvalues,
    risk_row_sha256,
    summarize_train_risk,
    wilson_upper,
)


class TrainRiskTests(unittest.TestCase):
    def test_wilson_upper_bound_matches_known_case(self) -> None:
        self.assertAlmostEqual(wilson_upper(0, 100, confidence=0.95), 0.0263427, places=6)

    def test_query_exclusion_and_tie_formula(self) -> None:
        pvalues = leave_one_out_upper_pvalues(np.asarray([2.0, 2.0, 1.0]))
        np.testing.assert_allclose(pvalues, [2.0 / 3.0, 2.0 / 3.0, 1.0])

    def test_train_risk_rejects_test_or_evaluation_inputs(self) -> None:
        with self.assertRaises(ValueError):
            audit_train_risk([{
                "dataset": "d", "source": "scattering", "variant": "primary", "n_train": 3,
                "train_oof": np.asarray([1.0, 2.0, 3.0]), "test_scores": np.asarray([9.0]),
            }])

    def test_nonfinite_eligible_row_is_not_filtered_into_a_pass(self) -> None:
        rows = _risk_rows()
        rows[0] = {**rows[0], "status": "nonfinite_train_oof", "wilson_upper_95": None}
        rows[0]["risk_row_sha256"] = risk_row_sha256(rows[0])
        summary = summarize_train_risk(rows)
        median = next(row for row in summary if row["variant"] == "seed_median")
        self.assertFalse(median["all_required_valid"])
        self.assertEqual(median["invalid_eligible_datasets"], 1)


class QualificationTests(unittest.TestCase):
    def test_oof_coverage_allows_only_fully_abstained_missing_variant(self) -> None:
        risk = {
            "active": {"coverage_complete": True, "eligible_datasets": 2, "abstained_datasets": 0},
            "small_train": {
                "coverage_complete": True,
                "eligible_datasets": 0,
                "abstained_datasets": 2,
                "invalid_required_datasets": 0,
                "undocumented_abstentions": 0,
            },
        }
        module._assert_oof_variant_coverage(
            {"active", "small_train"}, {"active"}, risk, 2,
        )
        with self.assertRaises(ValueError):
            module._assert_oof_variant_coverage(
                {"active", "small_train"}, {"active"},
                {**risk, "small_train": {**risk["small_train"], "abstained_datasets": 1}},
                2,
            )

    def test_certified_oof_vector_accepts_empty_abstention_and_both_seed_axes(self) -> None:
        self.assertEqual(len(module._one_vector(np.empty((2, 0)), 4)), 0)
        np.testing.assert_allclose(
            module._one_vector(np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), 3),
            [2.0, 3.0, 4.0],
        )
        np.testing.assert_allclose(
            module._one_vector(np.asarray([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]]), 3),
            [2.0, 3.0, 4.0],
        )

    def test_frozen_correlation_contract(self) -> None:
        self.assertEqual(CORRELATION_COLLAPSE_ABS_SPEARMAN, 0.90)
        self.assertEqual(CORRELATION_MIN_ALIGNED_TRAIN_OOF, 20)

    def test_variants_of_one_family_cannot_double_vote(self) -> None:
        votes = collapse_votes({"tcn_seed_median": {3}, "tcn_seed_intersection": {3}})
        self.assertEqual(votes[3], {"normal_forecast"})

    def test_high_correlation_cross_family_collapses_connected_component(self) -> None:
        collapsed = collapse_correlated_families({
            "scattering": np.arange(20, dtype=float),
            "signature": np.arange(20, dtype=float) * 2.0,
        })
        self.assertEqual(len(collapsed["groups"]), 1)

    def test_low_overlap_marks_independence_unverified(self) -> None:
        collapsed = collapse_correlated_families({
            "scattering": np.arange(19, dtype=float),
            "signature": np.arange(19, dtype=float),
        })
        self.assertTrue(all(not item["independence_verified"] for item in collapsed["groups"]))

    def test_missing_reproducibility_proof_qualifies_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            result = qualify_sources(
                prediction, manifest, output_dir=root, evidence=_evidence(),
                expected_datasets=("d",), verify_predictions=False,
            )
            self.assertFalse(result["reproducibility_verified"])
            self.assertEqual(result["qualified_groups"], [])

    def test_verified_evidence_and_reproducibility_can_qualify_one_family(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            evidence = _evidence()
            proof = _write_reproducibility(root, prediction, evidence)
            result = qualify_sources(
                prediction, manifest, output_dir=root, evidence=evidence,
                expected_datasets=("d",), reproducibility_manifest_path=proof,
                verify_predictions=False,
            )
            self.assertEqual(len(result["qualified_groups"]), 1)
            self.assertEqual(
                result["qualified_groups"][0]["members"],
                ["tcn_seed_intersection", "tcn_seed_median"],
            )

    def test_nonfinite_required_row_blocks_membership(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            evidence = _evidence(invalid_first=True)
            proof = _write_reproducibility(root, prediction, evidence)
            result = qualify_sources(
                prediction, manifest, output_dir=root, evidence=evidence,
                expected_datasets=("d",), reproducibility_manifest_path=proof,
                verify_predictions=False,
            )
            self.assertEqual(result["qualified_groups"], [])
            self.assertEqual(result["risk_by_variant"]["tcn_seed_median"]["invalid_required_datasets"], 1)

    def test_tampered_evidence_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            evidence = _evidence()
            evidence.risk_rows[0]["normal_candidate_count"] = 99
            with self.assertRaises(ValueError):
                qualify_sources(
                    prediction, manifest, output_dir=root, evidence=evidence,
                    expected_datasets=("d",), verify_predictions=False,
                )

    def test_high_correlation_with_existing_certified_family_blocks_new_vote(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            evidence = _evidence(existing_matches_new=True)
            proof = _write_reproducibility(root, prediction, evidence)
            result = qualify_sources(
                prediction, manifest, output_dir=root, evidence=evidence,
                expected_datasets=("d",), reproducibility_manifest_path=proof,
                verify_predictions=False,
            )
            self.assertEqual(result["qualified_groups"], [])
            self.assertTrue(result["existing_certified_independence"]["collapsed_into_existing"])

    def test_membership_hash_and_lane_verification_precede_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            evidence = _evidence()
            proof = _write_reproducibility(root, prediction, evidence)
            membership = qualify_sources(
                prediction, manifest, output_dir=root, evidence=evidence,
                expected_datasets=("d",), reproducibility_manifest_path=proof,
                verify_predictions=False,
            )
            membership_path = root / "exp272_source_membership.json"
            membership_path.write_text(json.dumps({**membership, "qualified_groups": []}), encoding="utf-8")
            with patch.object(module, "_load_test_truth", side_effect=AssertionError("labels read early")):
                with self.assertRaises(MembershipHashError):
                    evaluate_frozen_qualification(
                        prediction, manifest, membership_path,
                        expected_datasets=("d",), verify_predictions=False,
                    )

    def test_post_freeze_metrics_use_verified_immutable_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            evidence = _evidence()
            proof = _write_reproducibility(root, prediction, evidence)
            qualify_sources(
                prediction, manifest, output_dir=root, evidence=evidence,
                expected_datasets=("d",), reproducibility_manifest_path=proof,
                verify_predictions=False,
            )
            verified = module._VerifiedExistingLanes(
                lanes={"d": {"hard": frozenset({1}), "review": frozenset({2})}},
                n_test={"d": 3}, input_sha256={"fixture": "pinned"}, _guard=module._LANE_GUARD,
            )
            with patch.object(module, "_load_verified_existing_lanes", return_value=verified):
                result = evaluate_frozen_qualification(
                    prediction, manifest, root / "exp272_source_membership.json",
                    expected_datasets=("d",), test_truth={"d": {0, 1, 2}},
                    verify_predictions=False,
                )
            self.assertEqual(result["autonomous"]["hard_tp"], 1)
            self.assertEqual(result["human_assisted_review"]["review_tp"], 1)
            self.assertNotIn("combined_f1", result["autonomous"])


def _write_prediction_fixture(root: Path) -> tuple[Path, Path]:
    rows = []
    for variant in ("seed_median", "seed_intersection"):
        rows.append({
            "dataset": "d", "source": "tcn_forecast", "variant": variant, "alpha": 0.005,
            "candidate_indices": [0], "evidence_family": "normal_forecast",
            "evidence_identity": "tcn_forecast", "contains_test_labels": False,
            "config_hash": "config", "run_contract_sha256": f"contract-{variant}",
        })
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    prediction = root / "prediction.jsonl"
    prediction.write_bytes(payload)
    manifest = root / "prediction_manifest.json"
    manifest.write_text(json.dumps({"prediction_sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8")
    return prediction, manifest


def _risk_rows() -> list[dict[str, object]]:
    rows = []
    for variant in ("seed_median", "seed_intersection"):
        row = {
            "dataset": "d", "source": "tcn_forecast", "variant": variant, "alpha": 0.005,
            "n_train": 1000, "train_tier": "ge_500", "eligible": True, "abstained": False,
            "abstention_reason": None, "coverage_complete": True, "status": "ok",
            "query_count": 1000, "reference_count": 999, "normal_candidate_count": 0,
            "normal_candidate_rate": 0.0, "wilson_upper_95": wilson_upper(0, 1000),
            "calibration_reference": "query_excluded_train_oof",
            "cache_sha256": "cache", "run_contract_sha256": f"contract-{variant}",
            "train_oof_sha256": "bound-oof",
        }
        row["risk_row_sha256"] = risk_row_sha256(row)
        rows.append(row)
    return rows


def _evidence(*, invalid_first: bool = False, existing_matches_new: bool = False) -> QualificationEvidence:
    rng = np.random.default_rng(20260721)
    base = rng.normal(size=1000)
    new = {
        "tcn_seed_median": {"d": base.copy()},
        "tcn_seed_intersection": {"d": base.copy()},
    }
    existing = {
        f"exp266_{name}": {"d": rng.permutation(base)}
        for name in ("kernel", "statistical", "structural", "change_point")
    }
    if existing_matches_new:
        existing["exp266_kernel"] = {"d": base.copy()}
    rows = _risk_rows()
    if invalid_first:
        rows[0]["status"] = "nonfinite_train_oof"
        rows[0]["wilson_upper_95"] = None
        rows[0]["risk_row_sha256"] = risk_row_sha256(rows[0])
    cache = {"tcn_forecast:d": "cache"}
    contracts = {
        "tcn_seed_median": "contract-seed_median",
        "tcn_seed_intersection": "contract-seed_intersection",
    }
    evidence_hash = module._evidence_hash(rows, new, existing, cache, contracts)
    return QualificationEvidence(
        risk_rows=tuple(rows), train_oof_by_variant=new, existing_train_oof=existing,
        cache_sha256=cache, run_contract_sha256=contracts,
        evidence_sha256=evidence_hash, _guard=module._EVIDENCE_GUARD,
    )


def _write_reproducibility(root: Path, prediction: Path, evidence: QualificationEvidence) -> Path:
    prediction_hash = hashlib.sha256(prediction.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "reproducible": True,
        "prediction_sha256_run_1": prediction_hash,
        "prediction_sha256_run_2": prediction_hash,
        "evidence_sha256_run_1": evidence.evidence_sha256,
        "evidence_sha256_run_2": evidence.evidence_sha256,
        "cache_sha256_run_1": dict(sorted(evidence.cache_sha256.items())),
        "cache_sha256_run_2": dict(sorted(evidence.cache_sha256.items())),
    }
    payload["reproducibility_manifest_sha256"] = stable_json_hash(payload)
    path = root / "reproducibility.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
