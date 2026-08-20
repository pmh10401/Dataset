"""RED/GREEN contracts for Exp450 local scheduling and score-root freezing."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from concurrent.futures import Future

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    RouteLanes,
    SourceExpansion,
)
from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    _completed_paper,
)

from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
    append_dataset_condition_failure_antigravity,
    verify_dataset_condition_inventory_antigravity,
    write_dataset_condition_success_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
    Exp450ConditionScore,
    Exp450DatasetConditionResult,
    Exp450RunContext,
    canonical_json_bytes,
)
import experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity as runner_antigravity
from experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity import (
    Exp450Incomplete,
    build_execution_schedule_antigravity,
    compose_all_conditions_antigravity,
    freeze_all_conditions_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.verifier_antigravity import (
    verify_exp450_score_root_antigravity,
)
from scratch.generate_exp450_codemap_antigravity import (
    REPOSITORY_ROOT,
    ast_callers_antigravity,
    build_codemap_payload_antigravity,
    tracked_existing_paths_antigravity,
)


CONDITIONS = (
    Exp450Condition("A", "linear", "linear"),
    Exp450Condition("B", "polyphase_pchip", "linear"),
    Exp450Condition("C", "linear", "area"),
    Exp450Condition("D", "polyphase_pchip", "area"),
)
BACKEND_CPU = Exp450BackendDecision("cpu", "test", False, False, 2, 1)
BACKEND_MPS = Exp450BackendDecision("mps", "test", True, False, 3, 2)


def _raw(dataset: str) -> RawDataset:
    train = tuple(np.asarray([index, index + 0.25, index + 0.5], dtype="<f4") for index in range(5))
    test = tuple(np.asarray([index + 0.75, index + 1.0, index + 1.25], dtype="<f4") for index in range(2))
    return RawDataset(
        dataset, 3, tuple(range(5)), tuple(row.tobytes() for row in train), train,
        tuple(range(2)), tuple(row.tobytes() for row in test), test,
    )


def _catalog(raws: tuple[RawDataset, ...]) -> dict[str, object]:
    entries = []
    for number, raw in enumerate(raws):
        def identity(split: str, indices: tuple[int, ...], blobs: tuple[bytes, ...]) -> dict[str, object]:
            bare = {
                "dataset": raw.dataset, "dataset_id": number, "series_length": raw.metadata_length,
                "split": split, "indices": list(indices),
                "values_blob_sha256": [sha256(blob).hexdigest() for blob in blobs],
            }
            return {**bare, "row_digest": sha256(canonical_json_bytes(bare)).hexdigest(), "count": len(blobs)}
        entries.append({
            "dataset": raw.dataset, "dataset_id": number, "series_length": raw.metadata_length,
            "train": identity("TRAIN", raw.train_indices, raw.train_blobs),
            "test": identity("TEST", raw.test_indices, raw.test_blobs),
        })
    bare = {"schema": "exp450-runner-fixture-antigravity.v1", "mode": "smoke", "database": {"fixture": "test"}, "datasets": entries}
    return {**bare, "catalog_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}


def _context(root: Path, catalog: dict[str, object]) -> Exp450RunContext:
    return Exp450RunContext("smoke", root, "a" * 64, catalog["catalog_sha256"], {"runner_antigravity.py": "b" * 64}, BACKEND_CPU, "c" * 64)


def _result(raw: RawDataset, condition: Exp450Condition) -> Exp450DatasetConditionResult:
    sources = HardSourceBundle(EXPECTED_HARD_SOURCES, {
        name: _completed_paper(name, np.asarray([0., 1., 2., 3., 4.]), np.asarray([1.5, 4.5]))
        for name in EXPECTED_HARD_SOURCES
    })
    lanes = RouteLanes(frozenset(), frozenset(), frozenset())
    route = HardRouteResult(lanes, frozenset(), frozenset(), frozenset(), lanes,
                            SourceExpansion(frozenset(), frozenset()), lanes, frozenset(), lanes,
                            frozenset(), frozenset(), len(raw.test_rows))
    return Exp450DatasetConditionResult(raw.dataset, condition.condition_id, sources, route)


def _score(raw: RawDataset, condition: Exp450Condition) -> Exp450ConditionScore:
    policies = PolicyBundle(tuple(
        PolicyRoute(name, {raw.dataset: frozenset({0})},
                    {raw.dataset: np.asarray([0.5, 0.2], dtype=np.float64)},
                    {raw.dataset: np.asarray([0.5, 0.2], dtype=np.float64)})
        for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
    ))
    result = _result(raw, condition)
    return Exp450ConditionScore(condition, {raw.dataset: raw}, {raw.dataset: result.sources},
                                {raw.dataset: result.route}, object(), object(), policies, BACKEND_CPU, {})


def _rebind_score_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    if "shard_sha256" in value:
        bare = {key: item for key, item in value.items() if key != "shard_sha256"}
        value["shard_sha256"] = sha256(canonical_json_bytes(bare)).hexdigest()
    if "manifest_sha256" in value:
        bare = {key: item for key, item in value.items() if key != "manifest_sha256"}
        value["manifest_sha256"] = sha256(canonical_json_bytes(bare)).hexdigest()
    path.write_bytes(canonical_json_bytes(value))


class Exp450RunnerVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="exp450_runner_verifier_antigravity_")
        self.root = Path(self.temporary.name) / "run_antigravity"
        self.raw = _raw("fixture_antigravity")
        self.catalog = _catalog((self.raw,))
        self.context = _context(self.root, self.catalog)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mps_schedule_runs_ab_cpu_then_cd_one_at_a_time(self) -> None:
        """Running C/D in the CPU pool would permit overlapping MPS work."""
        schedule = build_execution_schedule_antigravity(self.catalog, BACKEND_MPS)
        self.assertEqual(schedule.cpu_conditions, ("A", "B"))
        self.assertEqual(schedule.mps_conditions, ("C", "D"))
        self.assertEqual(schedule.maximum_concurrent_mps_jobs, 1)
        self.assertEqual(schedule.cpu_workers, 2)

    def test_global_stages_wait_for_all_dataset_condition_successes(self) -> None:
        """Calling Task6 with one missing checkpoint would change global selection."""
        results = {
            (self.raw.dataset, condition.condition_id): _result(self.raw, condition)
            for condition in CONDITIONS[:-1]
        }
        with patch.object(runner_antigravity, "score_low_frequency", side_effect=AssertionError("global stage called")):
            with self.assertRaisesRegex(Exp450Incomplete, "pending=1"):
                compose_all_conditions_antigravity(results, {self.raw.dataset: self.raw}, BACKEND_CPU, CONDITIONS)

    def test_global_preflight_rejects_extra_or_mismatched_results_before_scoring(self) -> None:
        """An extra or falsely keyed local result cannot enter global selection."""
        results = {(self.raw.dataset, condition.condition_id): _result(self.raw, condition) for condition in CONDITIONS}
        results[("extra_antigravity", "A")] = _result(self.raw, CONDITIONS[0])
        with patch.object(runner_antigravity, "score_low_frequency", side_effect=AssertionError("global stage called")):
            with self.assertRaises(Exp450Incomplete):
                compose_all_conditions_antigravity(results, {self.raw.dataset: self.raw}, BACKEND_CPU, CONDITIONS)

    def test_freeze_preflight_rejects_a_miskeyed_condition_before_any_root_write(self) -> None:
        """Writing A--C before discovering a bad D would leave a partial freeze."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        scores = {condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}
        scores["D"] = _score(self.raw, CONDITIONS[0])
        with self.assertRaises(Exp450Incomplete):
            freeze_all_conditions_antigravity(scores, self.root, self.context, self.catalog)
        self.assertFalse((self.root / "score_roots_antigravity").exists())

    def test_local_schedule_rejects_a_backend_that_differs_from_its_context(self) -> None:
        """Scoring under one backend but checkpointing another would break provenance."""
        with patch.object(runner_antigravity, "verify_dataset_condition_inventory_antigravity", side_effect=AssertionError("inventory read")):
            with self.assertRaisesRegex(ValueError, "backend"):
                runner_antigravity.execute_local_schedule_antigravity(
                    self.root, self.context, self.catalog, {self.raw.dataset: self.raw}, BACKEND_MPS
                )

    def test_mps_execution_closes_cpu_pool_before_sequential_cd(self) -> None:
        """Starting C/D before the pool closes would overlap MPS-heavy scoring."""
        events: list[str] = []
        mps_context = Exp450RunContext("smoke", self.root, "a" * 64, self.catalog["catalog_sha256"], {"runner_antigravity.py": "b" * 64}, BACKEND_MPS, "c" * 64)

        class FakePool:
            def __init__(self, *, max_workers: int) -> None:
                self.max_workers = max_workers
                events.append(f"pool={max_workers}")

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                events.append("pool_closed")

            def submit(self, function, *args):
                future = Future()
                try:
                    future.set_result(function(*args))
                except Exception as error:
                    future.set_exception(error)
                return future

        def score(values: RawDataset, condition: Exp450Condition, _backend: Exp450BackendDecision):
            events.append(condition.condition_id)
            return _result(values, condition)

        with (
            patch.object(runner_antigravity, "ProcessPoolExecutor", FakePool),
            patch.object(runner_antigravity, "run_dataset_local_exp450_antigravity", side_effect=score),
        ):
            completed = runner_antigravity.execute_local_schedule_antigravity(
                self.root, mps_context, self.catalog, {self.raw.dataset: self.raw}, BACKEND_MPS
            )
        self.assertEqual(tuple(completed), tuple((self.raw.dataset, condition.condition_id) for condition in CONDITIONS))
        self.assertEqual(events, ["pool=2", "A", "B", "pool_closed", "C", "D"])

    def test_resume_finishes_never_attempted_wave_one_before_wave_two_retry(self) -> None:
        """A never-attempted failure must be recorded as wave one, never wave two."""
        append_dataset_condition_failure_antigravity(self.root, self.context, self.catalog, self.raw, CONDITIONS[0], 1, RuntimeError("interrupted"))
        events: list[str] = []
        class FakePool:
            def __init__(self, *, max_workers: int) -> None: pass
            def __enter__(self): return self
            def __exit__(self, *_args) -> None: pass
            def submit(self, function, *args):
                future = Future()
                try: future.set_result(function(*args))
                except Exception as error: future.set_exception(error)
                return future
        def score(values, condition, _backend):
            events.append(condition.condition_id)
            if condition.condition_id == "B": raise RuntimeError("still failing")
            return _result(values, condition)
        with patch.object(runner_antigravity, "ProcessPoolExecutor", FakePool), patch.object(runner_antigravity, "run_dataset_local_exp450_antigravity", side_effect=score):
            with self.assertRaises(Exp450Incomplete):
                runner_antigravity.execute_local_schedule_antigravity(self.root, self.context, self.catalog, {self.raw.dataset: self.raw}, BACKEND_CPU)
        inventory = verify_dataset_condition_inventory_antigravity(self.root, self.context, self.catalog, {self.raw.dataset: self.raw})
        self.assertEqual([(item.condition_id, item.wave) for item in inventory.failures], [("A", 1), ("B", 1), ("B", 2)])
        self.assertEqual(events, ["B", "C", "D", "A", "B"])

    def test_four_manifests_verify_before_label_module_can_import(self) -> None:
        """A frozen root must retain four exact policy vectors without label access."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        sys.modules.pop("paper.equipment_anomaly_detection.label_access_paper", None)
        scores = {condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}
        manifests = freeze_all_conditions_antigravity(scores, self.root, self.context, self.catalog)
        self.assertEqual(tuple(manifests), ("A", "B", "C", "D"))
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)
        for condition_id, manifest in manifests.items():
            self.assertEqual(
                verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id=condition_id),
                manifest,
            )

    def test_verifier_rejects_score_alert_disagreement(self) -> None:
        """Changing an alert independently of its thresholded score must fail verification."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        freeze_all_conditions_antigravity({condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}, self.root, self.context, self.catalog)
        shard = next((self.root / "score_roots_antigravity" / "A_antigravity").glob("*_antigravity/condition_score_shard_antigravity.json"))
        shard.write_bytes(shard.read_bytes().replace(b'"alerts":[0]', b'"alerts":[]'))
        with self.assertRaisesRegex(ValueError, "score root"):
            verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id="A")

    def test_verifier_rejects_a_checkpoint_hash_change_after_freeze(self) -> None:
        """A score root must not certify a local checkpoint whose bound bytes changed."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        freeze_all_conditions_antigravity({condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}, self.root, self.context, self.catalog)
        checkpoint = next((self.root / "checkpoints_antigravity" / "A_antigravity").glob("*_antigravity/dataset_condition_success_antigravity.json"))
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "score root"):
            verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id="A")

    def test_verifier_rejects_an_unlisted_score_root_leaf(self) -> None:
        """Ignoring an extra score leaf would make the frozen inventory ambiguous."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        freeze_all_conditions_antigravity({condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}, self.root, self.context, self.catalog)
        (self.root / "score_roots_antigravity" / "A_antigravity" / "unexpected_antigravity").mkdir()
        with self.assertRaisesRegex(ValueError, "score root"):
            verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id="A")

    def test_verifier_rejects_fully_rebound_invalid_checkpoint_payload(self) -> None:
        """Rehashing a syntactically valid empty checkpoint payload cannot certify science."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        freeze_all_conditions_antigravity({condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}, self.root, self.context, self.catalog)
        checkpoint = next((self.root / "checkpoints_antigravity" / "A_antigravity").glob("*_antigravity/dataset_condition_success_antigravity.json"))
        def corrupt_checkpoint(value):
            value["payload"] = {}
            value["payload_sha256"] = sha256(canonical_json_bytes({})).hexdigest()
        _rebind_score_json(checkpoint, corrupt_checkpoint)
        manifest = self.root / "score_roots_antigravity" / "A_antigravity" / "condition_score_manifest_antigravity.json"
        def rebind_manifest(value):
            value["checkpoint_hashes"][0]["sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
        _rebind_score_json(manifest, rebind_manifest)
        with self.assertRaisesRegex(ValueError, "score root"):
            verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id="A")

    def test_verifier_rejects_fully_rebound_wrong_condition_mode(self) -> None:
        """Rebinding every score-root hash cannot change frozen A's factor pair."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        freeze_all_conditions_antigravity({condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}, self.root, self.context, self.catalog)
        shard = next((self.root / "score_roots_antigravity" / "A_antigravity").glob("*_antigravity/condition_score_shard_antigravity.json"))
        _rebind_score_json(shard, lambda value: value["condition"].update({"image_mode": "area"}))
        manifest = self.root / "score_roots_antigravity" / "A_antigravity" / "condition_score_manifest_antigravity.json"
        def rebind(value):
            value["condition"]["image_mode"] = "area"
            value["shards"][0]["sha256"] = sha256(shard.read_bytes()).hexdigest()
        _rebind_score_json(manifest, rebind)
        with self.assertRaisesRegex(ValueError, "score root"):
            verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id="A")

    def test_verifier_rejects_fully_rebound_boolean_or_string_scores(self) -> None:
        """JSON booleans and strings must not coerce into continuous float64 evidence."""
        for condition in CONDITIONS:
            write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, _result(self.raw, condition))
        freeze_all_conditions_antigravity({condition.condition_id: _score(self.raw, condition) for condition in CONDITIONS}, self.root, self.context, self.catalog)
        shard = next((self.root / "score_roots_antigravity" / "A_antigravity").glob("*_antigravity/condition_score_shard_antigravity.json"))
        _rebind_score_json(shard, lambda value: value["policies"][0].update({"scores": [True, "0.2"]}))
        manifest = self.root / "score_roots_antigravity" / "A_antigravity" / "condition_score_manifest_antigravity.json"
        _rebind_score_json(manifest, lambda value: value["shards"][0].update({"sha256": sha256(shard.read_bytes()).hexdigest()}))
        with self.assertRaisesRegex(ValueError, "score root"):
            verify_exp450_score_root_antigravity(self.root, context=self.context, catalog=self.catalog, condition_id="A")

    def test_codemap_includes_runner_direct_paper_import_edges(self) -> None:
        """Omitting a runner's direct paper imports makes its dependency map false."""
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
        caller = "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py"
        edges = {(edge["caller"], edge["callee"]) for edge in payload["edges"]}
        self.assertIn((caller, "paper/equipment_anomaly_detection/routing_paper.py"), edges)
        self.assertIn((caller, "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py"), edges)

    def test_fresh_verifier_import_does_not_load_scoring_source_route_or_label_modules(self) -> None:
        """Adding a science import to the verifier would violate the post-freeze firewall."""
        command = (
            "import sys; import experiments.exp450_resampling_image_ablation_antigravity.verifier_antigravity; "
            "blocked=('scoring_antigravity','hard_route_paper','hard_sources_paper','label_access_paper'); "
            "raise SystemExit(any(any(name.endswith(item) for item in blocked) for name in sys.modules))"
        )
        completed = subprocess.run([sys.executable, "-c", command], cwd=Path.cwd(), check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
