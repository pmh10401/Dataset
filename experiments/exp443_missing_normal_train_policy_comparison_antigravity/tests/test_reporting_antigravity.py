"""Behavior-first tests for the Exp443 label-free publication boundary."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
    CheckpointBindingAntigravity,
    Exp443CheckpointStoreAntigravity,
    ProcessIdentityAntigravity,
    ordered_work_units_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import ProgressWriterAntigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
    canonical_prediction_bytes_antigravity,
    route_policy_antigravity,
)


POLICIES = tuple(FROZEN_CONFIG.policies)
DATASETS = tuple(f"dataset_{index:04d}" for index in range(FROZEN_CONFIG.dataset_count))
AFFECTED = DATASETS[: FROZEN_CONFIG.affected_dataset_count]


def _fixture_snapshot() -> SimpleNamespace:
    hashes = {
        "database": FROZEN_CONFIG.database_sha256,
        "exp249_prediction": "a" * 64,
        "exp249_prediction_manifest": "b" * 64,
        "exp249_artifact_manifest": "c" * 64,
        "exp249_cache_manifest": "d" * 64,
        "exp249_audit": "e" * 64,
        "exp291_routing": "f" * 64,
        "exp291_manifest": "0" * 64,
        "exp414_repair_audit": "1" * 64,
        "exp414_manifest": "2" * 64,
        "exp404_hard": "3" * 64,
        "exp404_manifest": "4" * 64,
    }
    lanes = {
        dataset: {"hard": (), "standard": (), "priority": (), "promotion": ()}
        for dataset in DATASETS
    }
    candidates = {dataset: {1024: (), 4096: ()} for dataset in DATASETS}
    hard_coordinates: set[tuple[str, int]] = set()
    promotion_coordinates: set[tuple[str, int]] = set()
    # Build the same frozen routing cardinalities used by the real authority:
    # 2,116 inherited Hard, 137 Standard promotions, 35 exact-normal removals.
    for dataset_index, dataset in enumerate(DATASETS):
        remaining = 2116 - len(hard_coordinates)
        if remaining > 0:
            count = min(5, remaining)
            lanes[dataset]["hard"] = tuple(range(count))
            hard_coordinates.update((dataset, index) for index in range(count))
    for dataset_index, dataset in enumerate(DATASETS):
        remaining = 137 - len(promotion_coordinates)
        if remaining <= 0:
            break
        available = tuple(index for index in range(5) if (dataset, index) not in hard_coordinates)
        count = min(len(available), remaining)
        if count:
            chosen = available[:count]
            lanes[dataset]["standard"] = chosen
            candidates[dataset][1024] = chosen
            promotion_coordinates.update((dataset, index) for index in chosen)
    pre_exact = hard_coordinates | promotion_coordinates
    exact_normal = frozenset(sorted(hard_coordinates)[:35])
    final_hard = frozenset(pre_exact - exact_normal)
    return SimpleNamespace(
        datasets=DATASETS,
        affected_datasets=AFFECTED,
        affected_families=("FamilyA", "FamilyB", "FamilyC", "FamilyD"),
        affected_rows=FROZEN_CONFIG.affected_normal_train_rows,
        nonfinite_scalars=FROZEN_CONFIG.nonfinite_scalar_count,
        all_missing_rows=FROZEN_CONFIG.all_missing_row_count,
        automatic_hard_count=2218,
        exact_normal_count=35,
        standard_count=806,
        priority_count=31,
        routing_algebra=(2116, 137, 35, 2218),
        exp236_lanes=lanes,
        exp249_candidate_maps=candidates,
        test_row_count_by_dataset={dataset: 5 for dataset in DATASETS},
        target_length_by_dataset={dataset: 8 for dataset in DATASETS},
        hard_coordinates_exp291=final_hard,
        hard_coordinates_exp404=final_hard,
        source_hashes=hashes,
        affected_datasets_sha256="5" * 64,
        label_access_count=0,
    )


def _result(policy: str, dataset: str) -> PolicyDatasetResultAntigravity:
    scores = tuple(
        KernelSeedScoresAntigravity(kernel, seed, np.zeros(5, dtype="<f8"), np.zeros(5, dtype="<f8"), ())
        for kernel in (1024, 4096)
        for seed in FROZEN_CONFIG.seeds
    )
    return PolicyDatasetResultAntigravity(
        policy, dataset, 8, 5, 0, 5, 0, 5, scores,
        {1024: (), 4096: ()}, (), "scored",
    )


@dataclass
class _Fixture:
    root: Path
    attempt: Path
    snapshot: SimpleNamespace
    reviewed_commit: str
    temporary: tempfile.TemporaryDirectory[str]


def _write_fixture() -> _Fixture:
    temporary = tempfile.TemporaryDirectory(prefix="exp443_task9_reporting_antigravity_")
    root = Path(temporary.name)
    attempt = root / "attempt_0000_antigravity"
    attempt.mkdir(mode=0o700)
    snapshot = _fixture_snapshot()
    reviewed_commit = "a" * 40
    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.readiness_antigravity import _source_evidence_antigravity
    source_snapshot_sha256 = sha256(canonical_json_bytes_antigravity(_source_evidence_antigravity(snapshot))).hexdigest()
    units = ordered_work_units_antigravity(AFFECTED)
    binding = CheckpointBindingAntigravity(
        FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(),
        FROZEN_CONFIG.environment_lock_sha256,
        source_snapshot_sha256,
        FROZEN_CONFIG.database_sha256,
        reviewed_commit,
    )
    process = ProcessIdentityAntigravity(4242, "fixture-start")
    store = Exp443CheckpointStoreAntigravity.create_antigravity(attempt / "checkpoint_antigravity", units, binding, process)
    writer = ProgressWriterAntigravity(
        attempt, total=len(units), checkpoint_path=attempt / "checkpoint_antigravity" / "state_antigravity.json", process=process,
    )
    writer.begin_units_antigravity(units)
    results: dict[int, PolicyDatasetResultAntigravity] = {}
    for unit in units:
        result = _result(unit.policy, unit.dataset)
        results[unit.sequence] = result
        store.commit_result_antigravity(unit.sequence, result)
        writer.accept_unit_antigravity(result)
    writer.terminal_antigravity("complete", None)

    control_evidence = [
        {
            "dataset": dataset,
            "kernel": kernel,
            "seed": seed,
            "train_shape": [5],
            "test_shape": [5],
            "archive_train_shape": [5],
            "archive_test_shape": [5],
            "finite": True,
            "train_max_abs_difference": 0.0,
            "test_max_abs_difference": 0.0,
            "max_abs_difference": 0.0,
            "candidate_equal": True,
            "archive_candidates": [],
            "policy_candidates": [],
        }
        for dataset in AFFECTED
        for kernel in (1024, 4096)
        for seed in FROZEN_CONFIG.seeds
    ]
    control = {
        "status": "control_reproduction_passed", "passed": True, "reason": None,
        "score_evidence": control_evidence,
        "routing_identity": {
            "rows": 1117, "promotions": 137, "pre_exact": 2253,
            "exact_normal_removals": 35, "final_hard": 2218,
            "promotions_coordinate_equal": True, "pre_exact_coordinate_equal": True,
            "exact_normal_removals_coordinate_equal": True,
            "exp291_coordinate_equal": True, "exp404_coordinate_equal": True,
            "recomputed_rows": 19, "imported_rows": 1098,
        },
        "environment_identity": {
            "python_version": FROZEN_CONFIG.python_version,
            "aeon_version": FROZEN_CONFIG.aeon_version,
            "numpy_version": FROZEN_CONFIG.numpy_version,
            "scikit_learn_version": FROZEN_CONFIG.scikit_learn_version,
            "environment_lock_sha256": FROZEN_CONFIG.environment_lock_sha256,
            "outer_workers": 11, "inner_threads": 1,
            "acceleration_decision": FROZEN_CONFIG.acceleration_decision,
            "acceleration_reason": "official_aeon_minirocket_cpu_transformer_no_mlx_rewrite",
        },
        "label_access_count": 0, "max_abs_difference": 0.0,
        "candidate_mismatch_count": 0,
    }
    (attempt / "control_gate_antigravity.json").write_bytes(canonical_json_bytes_antigravity(control))
    parity = {
        "schema": "exp443.worker_parity.v1", "experiment_id": "Exp443", "reviewed_commit": reviewed_commit,
        "policy": POLICIES[0], "unit_count": 19, "workers": 11, "logical_cpu_count": 12,
        "inner_threads": 1, "label_access_count": 0, "semantic_equal": True,
        "sequential_payload_sha256": "7" * 64, "parallel_payload_sha256": "7" * 64,
        "control_status": "control_reproduction_passed", "control_max_abs_difference": 0.0,
        "prediction_bytes_sha256": "8" * 64,
    }

    runtime_by_policy: dict[str, bytes] = {}
    for policy in POLICIES:
        policy_results = {dataset: results[sequence] for sequence, unit in enumerate(units) if unit.policy == policy for dataset in (unit.dataset,)}
        runtime_prediction = route_policy_antigravity(snapshot, policy_results, policy)
        runtime_by_policy[policy] = canonical_prediction_bytes_antigravity(runtime_prediction)
        (attempt / f"prediction_{policy}_antigravity.json").write_bytes(runtime_by_policy[policy])
    parity["prediction_bytes_sha256"] = sha256(runtime_by_policy[POLICIES[0]]).hexdigest()
    (attempt / "worker_parity_antigravity.json").write_bytes(canonical_json_bytes_antigravity(parity))

    return _Fixture(root, attempt, snapshot, reviewed_commit, temporary)


class ReportingAntigravityTests(unittest.TestCase):
    def test_complete_checkpoint_inspection_does_not_rebind_process_or_write_state(self) -> None:
        fixture = _write_fixture()
        try:
            checkpoint = fixture.attempt / "checkpoint_antigravity"
            before = {
                path.relative_to(checkpoint).as_posix(): path.read_bytes()
                for path in sorted(checkpoint.rglob("*")) if path.is_file()
            }
            inspected = Exp443CheckpointStoreAntigravity.inspect_complete_antigravity(checkpoint)
            after = {
                path.relative_to(checkpoint).as_posix(): path.read_bytes()
                for path in sorted(checkpoint.rglob("*")) if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(len(inspected.results), 57)
            self.assertEqual(inspected.process, ProcessIdentityAntigravity(4242, "fixture-start"))
        finally:
            fixture.temporary.cleanup()

    def test_public_api_exposes_publish_inspect_and_read_only_complete_checkpoint_seam(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        self.assertTrue(callable(reporting_antigravity.publish_predictions_antigravity))
        self.assertTrue(callable(reporting_antigravity.inspect_publication_antigravity))
        self.assertTrue(callable(Exp443CheckpointStoreAntigravity.inspect_complete_antigravity))

    def test_fixture_publication_has_exact_twelve_files_three_1117_row_predictions_and_replays(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        fixture = _write_fixture()
        try:
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot):
                frozen = reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
                replay = reporting_antigravity.inspect_publication_antigravity(fixture.attempt)
            publication = fixture.attempt / "publication_antigravity"
            self.assertEqual(set(item.name for item in publication.iterdir()), set(reporting_antigravity.PUBLICATION_NAMES_ANTIGRAVITY))
            for policy in POLICIES:
                name = f"prediction_{policy}_antigravity.jsonl"
                self.assertEqual(len((publication / name).read_bytes().splitlines()), 1117)
            self.assertEqual(frozen, replay)
            self.assertEqual(frozen.label_access_count, 0)
            manifest = json.loads((publication / "manifest_antigravity.json").read_text())
            status = json.loads((publication / "status_antigravity.json").read_text())
            artifact_map = json.loads((publication / "artifact_sha256_antigravity.json").read_text())
            self.assertEqual(manifest["runtime_prediction_sha256"][POLICIES[0]], sha256((fixture.attempt / f"prediction_{POLICIES[0]}_antigravity.json").read_bytes()).hexdigest())
            self.assertEqual(set(manifest["leaf_sha256"]), set(reporting_antigravity.PUBLICATION_NAMES_ANTIGRAVITY[:9]))
            self.assertEqual(status["manifest_sha256"], sha256((publication / "manifest_antigravity.json").read_bytes()).hexdigest())
            self.assertEqual(status["stage"], "complete_review_only")
            self.assertEqual(artifact_map["self_excluded"], "artifact_sha256_antigravity.json")
            self.assertEqual(len(artifact_map["files"]), 11)
            runtime = json.loads((fixture.attempt / f"prediction_{POLICIES[0]}_antigravity.json").read_text())
            published_rows = [json.loads(line) for line in (publication / f"prediction_{POLICIES[0]}_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual(published_rows, runtime["rows"])
        finally:
            fixture.temporary.cleanup()

    def test_incomplete_checkpoint_is_rejected_without_publication(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        fixture = _write_fixture()
        try:
            state_path = fixture.attempt / "checkpoint_antigravity" / "state_antigravity.json"
            state = json.loads(state_path.read_text())
            state["status"] = "interrupted_resumable"
            state_path.write_bytes(canonical_json_bytes_antigravity(state))
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), self.assertRaisesRegex(ValueError, "complete"):
                reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
            self.assertFalse((fixture.attempt / "publication_antigravity").exists())
        finally:
            fixture.temporary.cleanup()

    def test_failed_control_and_runtime_prediction_drift_are_rejected(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        fixture = _write_fixture()
        try:
            control_path = fixture.attempt / "control_gate_antigravity.json"
            control = json.loads(control_path.read_text()); control["passed"] = False; control["status"] = "control_reproduction_failed"
            control_path.write_bytes(canonical_json_bytes_antigravity(control))
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), self.assertRaisesRegex(ValueError, "control"):
                reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
            control["passed"] = True; control["status"] = "control_reproduction_passed"
            control_path.write_bytes(canonical_json_bytes_antigravity(control))
            runtime_path = fixture.attempt / f"prediction_{POLICIES[0]}_antigravity.json"
            runtime_path.write_bytes(canonical_json_bytes_antigravity({"rows": []}))
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), self.assertRaisesRegex(ValueError, "prediction"):
                reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
        finally:
            fixture.temporary.cleanup()

    def test_control_routing_count_and_label_payload_must_match_frozen_evidence(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        fixture = _write_fixture()
        try:
            control_path = fixture.attempt / "control_gate_antigravity.json"
            control = json.loads(control_path.read_text())
            control["routing_identity"]["final_hard"] = 2217
            control_path.write_bytes(canonical_json_bytes_antigravity(control))
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), self.assertRaisesRegex(ValueError, "control"):
                reporting_antigravity.publish_predictions_antigravity(fixture.attempt)

            control["routing_identity"]["final_hard"] = 2218
            control_path.write_bytes(canonical_json_bytes_antigravity(control))
            parity_path = fixture.attempt / "worker_parity_antigravity.json"
            parity = json.loads(parity_path.read_text())
            parity["label"] = 1
            parity_path.write_bytes(canonical_json_bytes_antigravity(parity))
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), self.assertRaisesRegex(ValueError, "label|evaluation"):
                reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
        finally:
            fixture.temporary.cleanup()

    def test_artifact_map_tamper_is_rejected_on_inspection(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        fixture = _write_fixture()
        try:
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot):
                reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
                artifact = fixture.attempt / "publication_antigravity" / "artifact_sha256_antigravity.json"
                payload = json.loads(artifact.read_text()); payload["files"]["status_antigravity.json"] = "0" * 64
                artifact.write_bytes(canonical_json_bytes_antigravity(payload))
                with self.assertRaisesRegex(ValueError, "artifact"):
                    reporting_antigravity.inspect_publication_antigravity(fixture.attempt)
        finally:
            fixture.temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
