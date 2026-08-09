"""Behavior-first tests for the Exp443 label-free publication boundary."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
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


def _fixture_promotion_indices(dataset: str) -> tuple[int, ...]:
    """Cheap deterministic reproduction of the fixture's 137 promotions."""
    hard: set[tuple[str, int]] = set()
    remaining = 2116
    for candidate in DATASETS:
        if remaining <= 0:
            break
        count = min(5, remaining)
        hard.update((candidate, index) for index in range(count))
        remaining -= count
    promotions: set[tuple[str, int]] = set()
    remaining = 137
    for candidate in DATASETS:
        if remaining <= 0:
            break
        available = tuple(index for index in range(5) if (candidate, index) not in hard)
        chosen = available[: min(len(available), remaining)]
        promotions.update((candidate, index) for index in chosen)
        remaining -= len(chosen)
    return tuple(index for candidate, index in sorted(promotions) if candidate == dataset)


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
            lanes[dataset]["promotion"] = chosen
            candidates[dataset][1024] = chosen
            promotion_coordinates.update((dataset, index) for index in chosen)
    pre_exact = hard_coordinates | promotion_coordinates
    exact_normal = frozenset(sorted(hard_coordinates)[:35])
    final_hard = frozenset(pre_exact - exact_normal)
    # The real control recomputation reads the frozen score-cache authority.
    # Keep the fixture cache descriptor-shaped and immutable while using a
    # large TRAIN reference so alpha=0.005 can attain the expected candidates.
    cache_inventory = {}
    for dataset in AFFECTED:
        train_by_seed = np.zeros((2, 1000), dtype="<f8")
        train_by_seed_4096 = np.zeros((2, 1000), dtype="<f8")
        test_by_seed_1024 = np.zeros((2, 5), dtype="<f8")
        test_by_seed_1024[:, tuple(candidates[dataset][1024])] = 1.0
        test_by_seed_4096 = np.zeros((2, 5), dtype="<f8")
        for array in (train_by_seed, train_by_seed_4096, test_by_seed_1024, test_by_seed_4096):
            array.setflags(write=False)
        cache_inventory[dataset] = {
            "cache_file": Path(f"{dataset}_cache_antigravity.npz"),
            "metadata_file": Path(f"{dataset}_metadata_antigravity.json"),
            "cache_sha256": "6" * 64,
            "metadata_sha256": "7" * 64,
            "config_hash": "8" * 64,
            "n_train": 1000,
            "n_test": 5,
            "target_length": 8,
            "abstained": False,
            "arrays": {
                "kernels_1024__train_by_seed": train_by_seed,
                "kernels_1024__test_by_seed": test_by_seed_1024,
                "kernels_4096__train_by_seed": train_by_seed_4096,
                "kernels_4096__test_by_seed": test_by_seed_4096,
            },
        }
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
        cache_inventory=cache_inventory,
        test_row_count_by_dataset={dataset: 5 for dataset in DATASETS},
        target_length_by_dataset={dataset: 8 for dataset in DATASETS},
        hard_coordinates_exp291=final_hard,
        hard_coordinates_exp404=final_hard,
        source_hashes=hashes,
        affected_datasets_sha256="5" * 64,
        label_access_count=0,
    )


def _result(policy: str, dataset: str) -> PolicyDatasetResultAntigravity:
    # Match the fixture cache authority's alpha-attainable candidate sets for
    # Policy A while retaining deterministic no-candidate alternatives.
    promotion = _fixture_promotion_indices(dataset)
    oof = np.zeros(1000, dtype="<f8")
    test_1024 = np.zeros(5, dtype="<f8")
    test_1024[list(promotion)] = 1.0
    oof.setflags(write=False)
    test_1024.setflags(write=False)
    test_4096 = np.zeros(5, dtype="<f8")
    test_4096.setflags(write=False)
    scores = tuple(
        KernelSeedScoresAntigravity(
            kernel,
            seed,
            oof,
            test_1024 if kernel == 1024 else test_4096,
            promotion if kernel == 1024 else (),
        )
        for kernel in (1024, 4096)
        for seed in FROZEN_CONFIG.seeds
    )
    return PolicyDatasetResultAntigravity(
        policy, dataset, 8, 1000, 0, 1000, 0, 5, scores,
        {1024: promotion, 4096: ()}, (), "scored",
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

    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import (
        compare_policy_a_to_archive_antigravity,
    )
    control_object = compare_policy_a_to_archive_antigravity(
        snapshot,
        {
            unit.dataset: results[unit.sequence]
            for unit in units
            if unit.policy == POLICIES[0]
        },
    )
    control = {
        "status": control_object.status,
        "passed": control_object.passed,
        "reason": control_object.reason,
        "score_evidence": [dict(item) for item in control_object.score_evidence],
        "routing_identity": dict(control_object.routing_identity),
        "environment_identity": dict(control_object.environment_identity),
        "label_access_count": control_object.label_access_count,
        "max_abs_difference": control_object.max_abs_difference,
        "candidate_mismatch_count": control_object.candidate_mismatch_count,
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


def _replace_complete_checkpoint(fixture: _Fixture, units) -> None:
    """Write a self-consistent terminal checkpoint for authority-boundary tests."""
    checkpoint = fixture.attempt / "checkpoint_antigravity"
    shutil.rmtree(checkpoint)
    binding = CheckpointBindingAntigravity(
        FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(),
        FROZEN_CONFIG.environment_lock_sha256,
        sha256(canonical_json_bytes_antigravity(
            __import__(
                "experiments.exp443_missing_normal_train_policy_comparison_antigravity.readiness_antigravity",
                fromlist=["_source_evidence_antigravity"],
            )._source_evidence_antigravity(fixture.snapshot)
        )).hexdigest(),
        FROZEN_CONFIG.database_sha256,
        fixture.reviewed_commit,
    )
    process = ProcessIdentityAntigravity(4242, "fixture-start")
    store = Exp443CheckpointStoreAntigravity.create_antigravity(checkpoint, tuple(units), binding, process)
    for unit in units:
        store.commit_result_antigravity(unit.sequence, _result(unit.policy, unit.dataset))


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

    def test_fabricated_control_is_rejected_when_recomputed_control_disagrees(self) -> None:
        """Publication must bind Task 6 control bytes to a fresh checkpoint replay.

        A passing control file is only a self-report.  The publisher must call
        the canonical archive comparison with the inspected Policy A result
        map and reject when that recomputation is not the same pass evidence.
        """
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import (
            ControlGateResultAntigravity,
        )

        fixture = _write_fixture()
        try:
            # Keep the fixture's stored control internally passing while making
            # the patched canonical comparison fail.  This catches an
            # implementation that trusts only the JSON's self-reported flags.
            recomputed = ControlGateResultAntigravity(
                "control_reproduction_failed",
                False,
                "archive evidence does not match checkpoint",
                (),
                {},
                {},
                0,
                0.0,
                0,
            )
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), \
                 patch.object(
                     reporting_antigravity,
                     "compare_policy_a_to_archive_antigravity",
                     return_value=recomputed,
                     create=True,
                 ) as compare:
                with self.assertRaisesRegex(ValueError, "control"):
                    reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
            compare.assert_called_once()
            snapshot_arg, results_arg = compare.call_args.args[:2]
            self.assertIs(snapshot_arg, fixture.snapshot)
            self.assertEqual(set(results_arg), set(AFFECTED))
        finally:
            fixture.temporary.cleanup()

    def test_completed_inspection_rejects_56_58_missing_policy_and_wrong_dataset_units(self) -> None:
        """Terminal inspection must retain the exact 57-unit identity/order."""
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import WorkUnitAntigravity

        units_56 = tuple(
            WorkUnitAntigravity(sequence, policy, dataset)
            for sequence, (policy, dataset) in enumerate(
                ( (policy, dataset) for policy, count in zip(POLICIES, (19, 19, 18)) for dataset in AFFECTED[:count] )
            )
        )
        units_58 = tuple(
            WorkUnitAntigravity(sequence, policy, dataset)
            for sequence, (policy, dataset) in enumerate(
                ( (policy, dataset) for policy, count in zip(POLICIES, (19, 19, 20)) for dataset in DATASETS[:count] )
            )
        )
        units_missing_policy = tuple(
            WorkUnitAntigravity(sequence, policy, dataset)
            for sequence, (policy, dataset) in enumerate(
                (
                    (policy, dataset)
                    for policy, names in ((POLICIES[0], AFFECTED), (POLICIES[1], DATASETS[:38]))
                    for dataset in names
                )
            )
        )
        units_wrong_dataset = tuple(
            WorkUnitAntigravity(sequence, policy, dataset)
            for sequence, (policy, dataset) in enumerate(
                (
                    (policy, "dataset_zzzz" if policy == POLICIES[1] and dataset == AFFECTED[-1] else dataset)
                    for policy in POLICIES
                    for dataset in AFFECTED
                )
            )
        )

        for label, units in (
            ("56 units", units_56),
            ("58 units", units_58),
            ("missing policy", units_missing_policy),
            ("wrong dataset", units_wrong_dataset),
        ):
            with self.subTest(label=label):
                fixture = _write_fixture()
                try:
                    _replace_complete_checkpoint(fixture, units)
                    with self.assertRaisesRegex(ValueError, "complete|unit|57|policy|dataset"):
                        Exp443CheckpointStoreAntigravity.inspect_complete_antigravity(
                            fixture.attempt / "checkpoint_antigravity"
                        )
                finally:
                    fixture.temporary.cleanup()

    def test_progress_pid_schema_path_and_start_identity_are_bound(self) -> None:
        """Publication must reject progress/PID shape and ownership drift."""
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        mutations = (
            (
                "progress extra field",
                "progress_antigravity.json",
                lambda payload: payload.__setitem__("unexpected", 1),
                "progress",
            ),
            (
                "PID extra field",
                "pid_antigravity.json",
                lambda payload: payload.__setitem__("unexpected", 1),
                "PID",
            ),
            (
                "checkpoint path drift",
                "progress_antigravity.json",
                lambda payload: payload.__setitem__("checkpoint", "other_checkpoint_antigravity/state_antigravity.json"),
                "progress",
            ),
            (
                "process start identity drift",
                "pid_antigravity.json",
                lambda payload: payload.__setitem__("start_identity", "different-process-start"),
                "PID|process|identity",
            ),
        )
        for label, filename, mutation, message in mutations:
            with self.subTest(label=label):
                fixture = _write_fixture()
                try:
                    path = fixture.attempt / filename
                    payload = json.loads(path.read_text())
                    mutation(payload)
                    path.write_bytes(canonical_json_bytes_antigravity(payload))
                    with patch.object(
                        reporting_antigravity,
                        "verify_frozen_sources_antigravity",
                        return_value=fixture.snapshot,
                    ), self.assertRaisesRegex(ValueError, message):
                        reporting_antigravity.inspect_publication_antigravity(fixture.attempt)
                finally:
                    fixture.temporary.cleanup()

    def test_publication_failure_before_first_write_preserves_failed_staging_evidence(self) -> None:
        self._assert_failed_staging_preserved("before_file_write", destination_committed=False)

    def test_publication_failure_after_one_file_preserves_failed_staging_evidence(self) -> None:
        self._assert_failed_staging_preserved("after_first_file", destination_committed=False)

    def test_publication_failure_before_rename_preserves_failed_staging_evidence(self) -> None:
        self._assert_failed_staging_preserved("before_rename", destination_committed=False)

    def test_publication_failure_after_rename_verification_preserves_failed_staging_evidence(self) -> None:
        self._assert_failed_staging_preserved("after_rename_verification", destination_committed=True)

    def _assert_failed_staging_preserved(self, phase: str, *, destination_committed: bool) -> None:
        """Every publication phase retains one private failed audit sibling."""
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import reporting_antigravity

        def inject(observed: str) -> None:
            if observed == phase:
                raise RuntimeError(f"publication injected failure at {phase}")

        fixture = _write_fixture()
        try:
            with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot), \
                 patch.object(reporting_antigravity, "_publication_phase_antigravity", side_effect=inject, create=True), \
                 self.assertRaisesRegex(RuntimeError, phase):
                    reporting_antigravity.publish_predictions_antigravity(fixture.attempt)
            failed = sorted(
                path
                for path in fixture.attempt.iterdir()
                if path.name.startswith("_failed_") and path.name.endswith("_antigravity")
            )
            self.assertEqual(len(failed), 1)
            failure_path = failed[0] / "failure_antigravity.json"
            self.assertTrue(failure_path.is_file())
            failure = json.loads(failure_path.read_text())
            self.assertEqual(failure.get("status"), "failed")
            self.assertTrue(failure.get("error"))
            self.assertEqual(failure.get("phase"), phase)
            self.assertEqual(failure.get("destination_committed"), destination_committed)
            publication = fixture.attempt / "publication_antigravity"
            self.assertEqual(publication.exists(), destination_committed)
            if destination_committed:
                with patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=fixture.snapshot):
                    self.assertEqual(
                        reporting_antigravity.inspect_publication_antigravity(fixture.attempt).label_access_count,
                        0,
                    )
        finally:
            fixture.temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
