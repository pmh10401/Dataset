from __future__ import annotations

import importlib.util
import inspect
import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel import (
    config as exp419_config,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    BenchmarkSelectionProof,
    DatasetCost,
    RuntimeBenchmark,
    RuntimeTrial,
    WorkerCompletion,
    benchmark_selection_proof_payload,
)
from experiments.exp419_deterministic_cpu_parallel.coordinator import select_runtime


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = REPO_ROOT / "scratch/benchmark_exp419_deterministic_cpu_parallel.py"
OFFICIAL_SCRIPT = REPO_ROOT / "scratch/run_exp419_deterministic_cpu_parallel.py"
BALANCED_DATASETS = ("a", "b", "c", "d", "e", "f")
BALANCED_COSTS = (
    DatasetCost("a", 4, "candidate_eligible"),
    DatasetCost("b", 2, "candidate_eligible"),
    DatasetCost("c", 1, "candidate_eligible"),
    DatasetCost("d", 1, "candidate_eligible"),
    DatasetCost("e", 1, "candidate_eligible"),
    DatasetCost("f", 1, "candidate_eligible"),
)
INVALID_DIAGNOSTIC_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp419_deterministic_cpu_parallel_benchmark"
)
FAILED_BALANCED_BENCHMARK_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp419_deterministic_cpu_parallel_benchmark_balanced"
)
CORRECTED_BENCHMARK_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp419_deterministic_cpu_parallel_benchmark_balanced_runtime_recovery"
)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _selection_proof(
    datasets: tuple[str, ...] = BALANCED_DATASETS,
    **changes,
) -> BenchmarkSelectionProof:
    if datasets == BALANCED_DATASETS:
        values = {
            "anchor_datasets": ("a", "c", "d", "e", "f"),
            "added_datasets": ("b",),
            "selected_datasets": BALANCED_DATASETS,
            "proxy_headroom_gate": 2.5,
            "total_cost": 10,
            "shard_loads": {
                2: (5, 5),
                4: (4, 2, 2, 2),
                6: (4, 2, 1, 1, 1, 1),
            },
            "proxy_headrooms": {2: 2.0, 4: 2.5, 6: 2.5},
        }
    else:
        values = {
            "anchor_datasets": datasets[:-1],
            "added_datasets": datasets[-1:],
            "selected_datasets": datasets,
            "proxy_headroom_gate": 2.5,
            "total_cost": 100,
            "shard_loads": {
                2: (50, 50),
                4: (40, 20, 20, 20),
                6: (40, 12, 12, 12, 12, 12),
            },
            "proxy_headrooms": {2: 2.0, 4: 2.5, 6: 2.5},
        }
    values.update(changes)
    return BenchmarkSelectionProof(**values)


def _passing_benchmark(
    datasets: tuple[str, ...] = BALANCED_DATASETS,
    selection_proof: BenchmarkSelectionProof | None = None,
) -> RuntimeBenchmark:
    trials = (
        RuntimeTrial(1, 12.0, 100, True, True, "", datasets, "1" * 64),
        RuntimeTrial(2, 5.0, 200, True, True, "", datasets, "2" * 64),
        RuntimeTrial(4, 4.0, 300, True, True, "", datasets, "3" * 64),
        RuntimeTrial(6, 6.0, 400, True, True, "", datasets, "4" * 64),
    )
    return select_runtime(
        trials,
        selection_proof or _selection_proof(datasets),
    )


def _passing_measured_benchmark(
    module,
    datasets: tuple[str, ...] = BALANCED_DATASETS,
    source_manifest_sha256: str = "c" * 64,
    selection_proof: BenchmarkSelectionProof | None = None,
):
    measurements = []
    trials = []
    for worker_count, elapsed in ((1, 12.0), (2, 5.0), (4, 4.0), (6, 6.0)):
        workers = tuple(
            module.WorkerUtilization(
                shard_index=index,
                pid=1000 + index,
                elapsed_seconds=elapsed * 0.8,
                wall_occupancy_fraction=0.8,
            )
            for index in range(worker_count)
        )
        measurement = module.TrialRuntimeMeasurement(
            worker_count=worker_count,
            wall_seconds=elapsed,
            process_cpu_seconds=elapsed * (worker_count + 1) * 0.5,
            process_cpu_utilization=0.5,
            worker_utilization=workers,
            scientific_evidence_sha256=f"{worker_count}" * 64,
            source_manifest_sha256=source_manifest_sha256,
        )
        measurements.append(measurement)
        trials.append(
            RuntimeTrial(
                worker_count,
                elapsed,
                worker_count * 100,
                True,
                True,
                "",
                datasets,
                measurement.evidence_sha256,
            )
        )
    return (
        select_runtime(
            tuple(trials),
            selection_proof or _selection_proof(datasets),
        ),
        tuple(measurements),
    )


class OfficialOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script("exp419_official_cli", OFFICIAL_SCRIPT)

    def _stages(self, events: list[str], fail_at: str | None = None):
        module = self.module

        def stage(name: str, result):
            def invoke(*_args):
                events.append(name)
                if name == fail_at:
                    raise RuntimeError(name)
                return result

            return invoke

        return module.OfficialStages(
            clean_git=stage("clean_git", "a" * 40),
            label_guard=stage("label_guard", None),
            freeze_inputs=stage("freeze_inputs", object()),
            verify_benchmark=stage("verify_benchmark", _passing_benchmark()),
            open_roots=stage("open_roots", object()),
            run_resume=stage("run_resume", object()),
            revalidate=stage("revalidate", None),
            render=stage("render", {"x": b"x"}),
            verify=stage("verify", None),
            publish=stage("publish", None),
        )

    def test_official_stages_execute_in_exact_fail_stop_order(self) -> None:
        events: list[str] = []
        self.assertEqual(self.module.execute_official(self._stages(events)), 0)
        self.assertEqual(
            events,
            [
                "clean_git",
                "label_guard",
                "freeze_inputs",
                "verify_benchmark",
                "open_roots",
                "run_resume",
                "revalidate",
                "render",
                "verify",
                "publish",
            ],
        )

    def test_every_failed_boundary_prevents_all_later_stages(self) -> None:
        order = [
            "clean_git",
            "label_guard",
            "freeze_inputs",
            "verify_benchmark",
            "open_roots",
            "run_resume",
            "revalidate",
            "render",
            "verify",
            "publish",
        ]
        for index, failed in enumerate(order):
            with self.subTest(failed=failed):
                events: list[str] = []
                with self.assertRaisesRegex(RuntimeError, failed):
                    self.module.execute_official(self._stages(events, failed))
                self.assertEqual(events, order[: index + 1])


class CliContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = _load_script("exp419_benchmark_cli", BENCHMARK_SCRIPT)
        cls.official = _load_script("exp419_official_cli_contract", OFFICIAL_SCRIPT)

    @contextmanager
    def _exact_layout(self, temporary: str):
        parent = Path(temporary).resolve()
        root = (
            parent
            / "exp419_deterministic_cpu_parallel_benchmark_balanced"
        )
        root.mkdir()
        evidence = root / FROZEN_CONFIG.benchmark_evidence_filename
        trials = tuple(
            root / f"workers-{count}"
            for count in FROZEN_CONFIG.worker_candidates
        )
        frozen = replace(
            FROZEN_CONFIG,
            corrected_benchmark_root=str(root),
        )
        with patch.object(exp419_config, "FROZEN_CONFIG", frozen):
            yield root, evidence, trials

    def test_benchmark_payload_round_trips_exact_selection_proof(self) -> None:
        benchmark = _passing_benchmark()
        payload = self.benchmark.benchmark_payload(benchmark)
        self.assertEqual(
            payload["selection_proof"],
            benchmark_selection_proof_payload(benchmark.selection_proof),
        )
        self.assertEqual(
            self.benchmark.benchmark_from_payload(payload),
            benchmark,
        )
        mutations = (
            lambda proof: proof["anchor_datasets"].__setitem__(0, "z"),
            lambda proof: proof["added_datasets"].__setitem__(0, "z"),
            lambda proof: proof["selected_datasets"].reverse(),
            lambda proof: proof.update({"proxy_headroom_gate": 2.0}),
            lambda proof: proof.update({"total_cost": 11}),
            lambda proof: proof["shard_loads"].__setitem__(
                "4", [4, 2, 1, 3]
            ),
            lambda proof: proof["proxy_headrooms"].__setitem__("4", 2.6),
            lambda proof: proof.update({"sha256": "0" * 64}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                mutated = deepcopy(payload)
                mutation(mutated["selection_proof"])
                with self.assertRaises(ValueError):
                    self.benchmark.benchmark_from_payload(mutated)

        different_proof = _selection_proof(
            anchor_datasets=("b", "c", "d", "e", "f"),
            added_datasets=("a",),
        )
        copied_digest = self.benchmark.benchmark_payload(
            _passing_benchmark(selection_proof=different_proof)
        )
        copied_digest["sha256"] = payload["sha256"]
        with self.assertRaisesRegex(ValueError, "digest"):
            self.benchmark.benchmark_from_payload(copied_digest)

    def test_bounded_runtime_smoke_uses_two_train_batches_without_test_loader(
        self,
    ) -> None:
        proof = _selection_proof()
        frozen = SimpleNamespace(
            costs=BALANCED_COSTS,
            bootstrap=object(),
            context=SimpleNamespace(sha256="a" * 64),
        )
        observed: dict[str, object] = {}

        def bounded_workers(specs, bootstrap, *, batches_per_seed):
            observed["bootstrap"] = bootstrap
            observed["batches_per_seed"] = batches_per_seed
            return tuple(
                WorkerCompletion(
                    shard_index=spec.shard.shard_index,
                    pid=1000 + spec.shard.shard_index,
                    completed_datasets=spec.shard.datasets,
                    record_digests={
                        dataset: sha256(dataset.encode()).hexdigest()
                        for dataset in spec.shard.datasets
                    },
                    elapsed_seconds=0.5,
                    peak_rss_bytes=1024,
                    thread_environment={
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                    },
                    status="completed",
                    error="",
                )
                for spec in specs
            )

        with (
            patch.object(
                self.benchmark,
                "select_balanced_benchmark_datasets",
                return_value=proof,
            ),
            patch.object(
                self.benchmark,
                "spawn_bounded_train_smoke_workers",
                side_effect=bounded_workers,
            ),
        ):
            measurement = self.benchmark.run_bounded_train_runtime_smoke(
                frozen, 4
            )

        self.assertEqual(measurement.worker_count, 4)
        self.assertEqual(measurement.seed_workers, 3)
        self.assertEqual(measurement.batches_per_seed, 2)
        self.assertFalse(measurement.test_values_loaded)
        self.assertEqual(observed["bootstrap"], frozen.bootstrap)
        self.assertEqual(observed["batches_per_seed"], 2)
        self.assertNotIn(
            "load_raw_test_values",
            inspect.getsource(self.benchmark.run_bounded_train_runtime_smoke),
        )

    def test_benchmark_parser_is_structural_and_writer_checks_selection(
        self,
    ) -> None:
        benchmark, measurements = _passing_measured_benchmark(self.benchmark)
        selected_index = next(
            index
            for index, trial in enumerate(benchmark.trials)
            if trial.worker_count == benchmark.selected_worker_count
        )
        selected = benchmark.trials[selected_index]
        invalid_selected = RuntimeTrial(
            selected.worker_count,
            selected.elapsed_seconds,
            selected.peak_rss_bytes,
            False,
            selected.finite,
            selected.fallback,
            selected.completed_datasets,
            selected.evidence_sha256,
        )
        forged_trials = list(benchmark.trials)
        forged_trials[selected_index] = invalid_selected
        forged = RuntimeBenchmark(
            tuple(forged_trials),
            benchmark.selected_worker_count,
            benchmark.status,
            benchmark.benchmark_datasets,
            benchmark.selection_proof,
        )
        payload = self.benchmark.benchmark_payload(forged)
        self.assertEqual(
            self.benchmark.benchmark_from_payload(payload),
            forged,
        )

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark.json"
            with self.assertRaisesRegex(ValueError, "selection"):
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=forged,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=tuple(
                        CORRECTED_BENCHMARK_ROOT / f"workers-{count}"
                        for count in FROZEN_CONFIG.worker_candidates
                    ),
                    runtime_measurements=measurements,
                )
            self.assertFalse(path.exists())

    def test_parsers_expose_only_frozen_path_arguments(self) -> None:
        forbidden = {
            "workers",
            "worker_count",
            "label",
            "labels",
            "threshold",
            "seed",
            "horizon",
            "budget",
            "approval",
        }
        benchmark_dests = {
            action.dest for action in self.benchmark.build_parser()._actions
        }
        official_dests = {
            action.dest for action in self.official.build_parser()._actions
        }
        self.assertTrue(forbidden.isdisjoint(benchmark_dests))
        self.assertTrue(forbidden.isdisjoint(official_dests))
        self.assertEqual(
            official_dests - {"help"},
            {
                "database",
                "exp291_prediction",
                "exp291_manifest",
                "exp292_prediction",
                "exp292_manifest",
                "benchmark_evidence",
                "checkpoint_dir",
                "output_dir",
            },
        )

    def test_official_help_documents_but_does_not_accept_approval(self) -> None:
        help_text = self.official.build_parser().format_help()
        self.assertIn("Exp419 1,117개 전체 label-free 병렬 실행 승인", help_text)
        with self.assertRaises(SystemExit):
            self.official.build_parser().parse_args(["--approval", "yes"])

    def test_runbook_freezes_corrected_root_and_retains_diagnostic_root(
        self,
    ) -> None:
        runbook = (REPO_ROOT / "docs/EXP419_RUNBOOK_KO.md").read_text()
        invalid_root = str(INVALID_DIAGNOSTIC_ROOT)
        failed_balanced_root = str(FAILED_BALANCED_BENCHMARK_ROOT)
        corrected_root = str(CORRECTED_BENCHMARK_ROOT)
        corrected_evidence = (
            corrected_root + "/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"
        )
        self.assertIn(invalid_root, runbook)
        self.assertIn(failed_balanced_root, runbook)
        self.assertIn("진단 전용", runbook)
        self.assertIn("재사용", runbook)
        self.assertNotIn(f'--benchmark-root "{invalid_root}"', runbook)
        self.assertNotIn(
            '--benchmark-evidence "'
            + invalid_root
            + '/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"',
            runbook,
        )
        self.assertNotIn(f'--benchmark-root "{failed_balanced_root}"', runbook)
        self.assertNotIn(
            '--benchmark-evidence "'
            + failed_balanced_root
            + '/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"',
            runbook,
        )
        self.assertIn(f'--benchmark-root "{corrected_root}"', runbook)
        self.assertIn(f'--benchmark-evidence "{corrected_evidence}"', runbook)
        for frozen_preflight_literal in (
            "EXPECTED_ANCHORS = (",
            "EXPECTED_ADDITIONS = (",
            "EXPECTED_SELECTED = (",
            "EXPECTED_TOTAL_COST = 888_487_623",
            "EXPECTED_LOADS_4 = (353424384, 178469604, 178343436, 178250199)",
            "EXPECTED_LOADS_6 = (",
            "00ac443347f97c267ffdfce82185c4631ca802e97f1685057976194ba2e52522",
            "abs_tol=1e-12",
            "assert not root.exists()",
            "benchmark_selection_proof_from_payload(",
        ):
            with self.subTest(literal=frozen_preflight_literal):
                self.assertIn(frozen_preflight_literal, runbook)
        self.assertIn(
            'root / f"workers-{count}" for count in (1, 2, 4, 6)',
            runbook,
        )

    def test_runbook_records_exit_code_with_a_zsh_safe_variable(self) -> None:
        runbook = (REPO_ROOT / "docs/EXP419_RUNBOOK_KO.md").read_text()
        self.assertIn('exit_code="$?"', runbook)
        self.assertIn('printf \'%s\\n\' "$exit_code" > "$RUNEXIT"', runbook)
        self.assertIn('exit "$exit_code"', runbook)
        self.assertNotIn('status="$?"', runbook)

    def test_dirty_git_blocks_benchmark_and_official_before_other_work(self) -> None:
        for module in (self.benchmark, self.official):
            with self.subTest(module=module.__name__):
                calls: list[str] = []

                def dirty():
                    calls.append("clean")
                    raise ValueError("dirty")

                stages = module.make_test_stages(clean_git=dirty)
                with self.assertRaisesRegex(ValueError, "dirty"):
                    module.execute_for_test(stages)
                self.assertEqual(calls, ["clean"])

    def test_benchmark_revalidation_verifies_the_exact_frozen_inputs(
        self,
    ) -> None:
        verifier = getattr(
            self.benchmark, "verify_input_snapshots", None
        )
        self.assertTrue(
            callable(verifier),
            "benchmark runtime must bind the frozen input snapshot verifier",
        )
        paths = (
            Path("/immutable/database.sqlite"),
            Path("/immutable/exp291-prediction.jsonl"),
            Path("/immutable/exp291-manifest.json"),
            Path("/immutable/exp292-prediction.jsonl"),
            Path("/immutable/exp292-manifest.json"),
        )
        args = SimpleNamespace(
            database=paths[0],
            exp291_prediction=paths[1],
            exp291_manifest=paths[2],
            exp292_prediction=paths[3],
            exp292_manifest=paths[4],
        )
        observed: list[tuple[Path, ...]] = []

        with (
            patch.object(
                self.benchmark,
                "freeze_benchmark_inputs",
                return_value=object(),
            ),
            patch.object(
                self.benchmark, "require_clean_git", return_value="a" * 40
            ),
            patch.object(
                self.benchmark,
                "assert_exp419_label_isolation",
                return_value="b" * 64,
            ),
            patch.object(
                self.benchmark,
                "verify_input_snapshots",
                side_effect=lambda *actual: observed.append(actual),
            ),
        ):
            stages = self.benchmark._build_stages(args)
            stages.freeze_inputs("a" * 40, "b" * 64)
            stages.revalidate()

        self.assertEqual(observed, [paths])

    def test_real_parsers_do_no_filesystem_work_before_clean_and_label_guards(self) -> None:
        common = [
            "--database", "/does/not/exist/database",
            "--exp291-prediction", "/does/not/exist/291-prediction",
            "--exp291-manifest", "/does/not/exist/291-manifest",
            "--exp292-prediction", "/does/not/exist/292-prediction",
            "--exp292-manifest", "/does/not/exist/292-manifest",
        ]
        cases = (
            (
                self.benchmark,
                common + ["--benchmark-root", "/does/not/exist/benchmark"],
            ),
            (
                self.official,
                common
                + [
                    "--benchmark-evidence", "/does/not/exist/evidence",
                    "--checkpoint-dir", "/does/not/exist/checkpoint",
                    "--output-dir", "/does/not/exist/output",
                ],
            ),
        )
        for module, argv in cases:
            with self.subTest(module=module.__name__):
                events: list[str] = []

                def clean():
                    events.append("clean")
                    return "a" * 40

                def label():
                    events.append("label")
                    raise ValueError("label boundary")

                with patch.object(module, "require_clean_git", side_effect=clean), \
                     patch.object(
                         module,
                         "assert_exp419_label_isolation",
                         side_effect=label,
                     ), self.assertRaisesRegex(ValueError, "label boundary"):
                    module.main(argv)
                self.assertEqual(events, ["clean", "label"])

    def test_output_and_checkpoint_roots_must_be_new_disjoint_and_unaliased(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output = root / "output"
            self.official.require_disjoint_roots(checkpoint, output)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                self.official.require_disjoint_roots(
                    checkpoint, checkpoint / "output"
                )
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "new"):
                self.official.require_disjoint_roots(checkpoint, output)
            alias = root / "alias"
            alias.symlink_to(checkpoint, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "disjoint|alias"):
                self.official.require_disjoint_roots(checkpoint, alias)

    def test_benchmark_evidence_is_digest_context_and_selection_bound(self) -> None:
        benchmark, measurements = _passing_measured_benchmark(self.benchmark)
        costs = BALANCED_COSTS
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                observed = self.official.verify_benchmark_evidence(
                    path,
                    expected_git_commit="a" * 40,
                    expected_context_sha256="b" * 64,
                    expected_source_manifest_sha256="c" * 64,
                    expected_costs=costs,
                )
                self.assertEqual(observed, benchmark)

                stale_provenance = (
                    ("d" * 40, "b" * 64, "c" * 64),
                    ("a" * 40, "d" * 64, "c" * 64),
                    ("a" * 40, "b" * 64, "d" * 64),
                )
                for git_commit, context_sha256, source_manifest in stale_provenance:
                    with self.subTest(
                        git_commit=git_commit,
                        context_sha256=context_sha256,
                        source_manifest=source_manifest,
                    ), self.assertRaisesRegex(
                        ValueError, "manifest|provenance"
                    ):
                        self.official.verify_benchmark_evidence(
                            path,
                            expected_git_commit=git_commit,
                            expected_context_sha256=context_sha256,
                            expected_source_manifest_sha256=source_manifest,
                            expected_costs=costs,
                        )

                manifest_forged = json.loads(path.read_text())
                manifest_forged["source_manifest_sha256"] = "d" * 64
                path.write_text(json.dumps(manifest_forged))
                with self.assertRaisesRegex(ValueError, "manifest|provenance"):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=costs,
                    )

                path.unlink()
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                forged = json.loads(path.read_text())
                forged["benchmark"]["selected_worker_count"] = 6
                path.write_text(json.dumps(forged))
                with self.assertRaisesRegex(
                    ValueError, "digest|selection|benchmark"
                ):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=costs,
                    )

    def test_official_rejects_internally_consistent_arbitrary_benchmark_subset(self) -> None:
        benchmark, measurements = _passing_measured_benchmark(
            self.benchmark, ("a", "b")
        )
        costs = BALANCED_COSTS
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                with self.assertRaisesRegex(
                    ValueError, "proof|balance|subset|datasets"
                ):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=costs,
                    )

    def test_official_rejects_two_worker_only_qualified_selection(self) -> None:
        measurements = []
        trials = []
        for worker_count, elapsed in (
            (1, 100.0),
            (2, 49.0),
            (4, 60.0),
            (6, 70.0),
        ):
            workers = tuple(
                self.benchmark.WorkerUtilization(
                    shard_index=index,
                    pid=1000 + index,
                    elapsed_seconds=elapsed * 0.8,
                    wall_occupancy_fraction=0.8,
                )
                for index in range(worker_count)
            )
            measurement = self.benchmark.TrialRuntimeMeasurement(
                worker_count=worker_count,
                wall_seconds=elapsed,
                process_cpu_seconds=elapsed * (worker_count + 1) * 0.5,
                process_cpu_utilization=0.5,
                worker_utilization=workers,
                scientific_evidence_sha256=f"{worker_count}" * 64,
                source_manifest_sha256="c" * 64,
            )
            measurements.append(measurement)
            trials.append(
                RuntimeTrial(
                    worker_count,
                    elapsed,
                    worker_count * 100,
                    True,
                    True,
                    "",
                    BALANCED_DATASETS,
                    measurement.evidence_sha256,
                )
            )
        benchmark = RuntimeBenchmark(
            tuple(trials),
            2,
            "qualified_parallel_runtime",
            BALANCED_DATASETS,
            _selection_proof(),
        )
        payload = {
            "artifact_kind": "exp419_non_official_runtime_benchmark",
            "benchmark": self.benchmark.benchmark_payload(benchmark),
            "context_sha256": "b" * 64,
            "git_commit": "a" * 40,
            "runtime_measurements": [
                self.benchmark.runtime_measurement_payload(measurement)
                for measurement in measurements
            ],
            "schema_version": 1,
            "source_manifest_sha256": "c" * 64,
            "trial_roots": [],
            "worker_candidates": list(FROZEN_CONFIG.worker_candidates),
        }
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                payload["trial_roots"] = [
                    str(root) for root in trial_roots
                ]
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    ValueError, "4|6|eligible|authorize|selection"
                ):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )

    def test_official_recomputes_exact_proof_and_rejects_old_or_mutated_balance(
        self,
    ) -> None:
        old_names = tuple(f"old-{index:02d}" for index in range(15))
        old_proof = _selection_proof(old_names)
        wrong_partition = _selection_proof(
            anchor_datasets=("b", "c", "d", "e", "f"),
            added_datasets=("a",),
        )
        wrong_load_order = _selection_proof(
            shard_loads={
                2: (5, 5),
                4: (2, 4, 2, 2),
                6: (4, 2, 1, 1, 1, 1),
            },
        )
        attacks = (
            (
                _passing_measured_benchmark(
                    self.benchmark,
                    old_names,
                    selection_proof=old_proof,
                ),
                BALANCED_COSTS,
            ),
            (
                _passing_measured_benchmark(self.benchmark),
                (
                    DatasetCost("a", 4, "candidate_eligible"),
                    DatasetCost("b", 3, "candidate_eligible"),
                    *BALANCED_COSTS[2:],
                ),
            ),
            (
                _passing_measured_benchmark(
                    self.benchmark,
                    selection_proof=wrong_partition,
                ),
                BALANCED_COSTS,
            ),
            (
                _passing_measured_benchmark(
                    self.benchmark,
                    selection_proof=wrong_load_order,
                ),
                BALANCED_COSTS,
            ),
        )
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                for index, ((benchmark, measurements), costs) in enumerate(attacks):
                    with self.subTest(index=index):
                        self.benchmark.write_benchmark_evidence(
                            path,
                            benchmark=benchmark,
                            git_commit="a" * 40,
                            context_sha256="b" * 64,
                            source_manifest_sha256="c" * 64,
                            trial_roots=trial_roots,
                            runtime_measurements=measurements,
                        )
                        with self.assertRaisesRegex(
                            ValueError, "proof|balance|subset|datasets"
                        ):
                            self.official.verify_benchmark_evidence(
                                path,
                                expected_git_commit="a" * 40,
                                expected_context_sha256="b" * 64,
                                expected_source_manifest_sha256="c" * 64,
                                expected_costs=costs,
                            )
                        path.unlink()

    def test_official_compares_expected_proof_before_runtime_selection(
        self,
    ) -> None:
        wrong_proof = _selection_proof(
            anchor_datasets=("b", "c", "d", "e", "f"),
            added_datasets=("a",),
        )
        benchmark, measurements = _passing_measured_benchmark(
            self.benchmark,
            selection_proof=wrong_proof,
        )
        selection_events: list[str] = []

        def selection_reached(*_args):
            selection_events.append("selection")
            raise AssertionError("runtime selection reached before proof check")

        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                with (
                    patch.dict(
                        self.official.benchmark_from_payload.__globals__,
                        {"select_runtime": selection_reached},
                    ),
                    patch.object(
                        self.official,
                        "select_runtime",
                        side_effect=selection_reached,
                        create=True,
                    ),
                    self.assertRaisesRegex(ValueError, "selection proof"),
                ):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )
        self.assertEqual(selection_events, [])

    def test_official_rejects_terminated_diagnostic_roots_and_accepts_corrected_root(
        self,
    ) -> None:
        benchmark, measurements = _passing_measured_benchmark(self.benchmark)
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                for root in (
                    INVALID_DIAGNOSTIC_ROOT,
                    INVALID_DIAGNOSTIC_ROOT / "workers-1",
                ):
                    with self.subTest(root=root):
                        self.benchmark.write_benchmark_evidence(
                            path,
                            benchmark=benchmark,
                            git_commit="a" * 40,
                            context_sha256="b" * 64,
                            source_manifest_sha256="c" * 64,
                            trial_roots=tuple(
                                (
                                    root
                                    if count
                                    == FROZEN_CONFIG.worker_candidates[0]
                                    else root / f"trial-{count}"
                                )
                                for count in FROZEN_CONFIG.worker_candidates
                            ),
                            runtime_measurements=measurements,
                        )
                        with self.assertRaisesRegex(
                            ValueError, "exact|ordered|root"
                        ):
                            self.official.verify_benchmark_evidence(
                                path,
                                expected_git_commit="a" * 40,
                                expected_context_sha256="b" * 64,
                                expected_source_manifest_sha256="c" * 64,
                                expected_costs=BALANCED_COSTS,
                            )
                        path.unlink()

                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                self.assertEqual(
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    ),
                    benchmark,
                )

    def test_official_rejects_old_manifest_bound_trials_under_current_top_level(self) -> None:
        benchmark, measurements = _passing_measured_benchmark(
            self.benchmark, source_manifest_sha256="d" * 64
        )
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="d" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                payload = json.loads(path.read_bytes())
                payload["source_manifest_sha256"] = "c" * 64
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    ValueError, "manifest|measurement"
                ):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )

    def test_runtime_measurements_are_finite_consistent_and_evidence_bound(self) -> None:
        benchmark, measurements = _passing_measured_benchmark(self.benchmark)
        for trial, measurement in zip(
            benchmark.trials, measurements, strict=True
        ):
            self.assertEqual(trial.evidence_sha256, measurement.evidence_sha256)
            self.assertTrue(
                isclose(
                    measurement.process_cpu_utilization,
                    measurement.process_cpu_seconds
                    / (trial.elapsed_seconds * (trial.worker_count + 1)),
                )
            )
        with TemporaryDirectory() as temporary:
            with self._exact_layout(temporary) as (
                _root,
                path,
                trial_roots,
            ):
                self.benchmark.write_benchmark_evidence(
                    path,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trial_roots,
                    runtime_measurements=measurements,
                )
                payload = json.loads(path.read_bytes())
                payload["runtime_measurements"][0][
                    "process_cpu_seconds"
                ] = -1.0
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    ValueError, "CPU|cpu|finite|nonnegative"
                ):
                    self.official.verify_benchmark_evidence(
                        path,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )

    def test_benchmark_requires_fresh_disjoint_candidate_roots(self) -> None:
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = (
                parent
                / "exp419_deterministic_cpu_parallel_benchmark_balanced"
            )
            frozen = replace(
                FROZEN_CONFIG,
                corrected_benchmark_root=str(root),
            )
            with patch.object(exp419_config, "FROZEN_CONFIG", frozen):
                roots = self.benchmark.prepare_trial_roots(root)
                self.assertEqual(
                    tuple(roots),
                    FROZEN_CONFIG.worker_candidates,
                )
                self.assertEqual(
                    tuple(roots.values()),
                    tuple(
                        root / f"workers-{count}"
                        for count in FROZEN_CONFIG.worker_candidates
                    ),
                )
                with self.assertRaisesRegex(ValueError, "new"):
                    self.benchmark.prepare_trial_roots(root)

    def test_benchmark_root_allowlist_rejects_arbitrary_and_symlink_aliases(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = (
                parent
                / "exp419_deterministic_cpu_parallel_benchmark_balanced"
            )
            frozen = replace(
                FROZEN_CONFIG,
                corrected_benchmark_root=str(root),
            )
            with patch.object(exp419_config, "FROZEN_CONFIG", frozen):
                arbitrary = parent / "arbitrary-benchmark"
                with self.assertRaisesRegex(
                    ValueError, "exact|canonical|allowlist"
                ):
                    self.benchmark.prepare_trial_roots(arbitrary)
                self.assertFalse(arbitrary.exists())

                target_parent = parent / "target"
                target_parent.mkdir()
                alias_parent = parent / "alias"
                alias_parent.symlink_to(
                    target_parent, target_is_directory=True
                )
                alias_root = alias_parent / root.name
                with self.assertRaisesRegex(
                    ValueError, "exact|canonical|allowlist"
                ):
                    self.benchmark.prepare_trial_roots(alias_root)
                self.assertFalse((target_parent / root.name).exists())

    def test_official_requires_exact_evidence_and_ordered_canonical_trial_roots(
        self,
    ) -> None:
        benchmark, measurements = _passing_measured_benchmark(self.benchmark)
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = (
                parent
                / "exp419_deterministic_cpu_parallel_benchmark_balanced"
            )
            root.mkdir()
            evidence = root / "EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"
            trials = tuple(
                root / f"workers-{count}"
                for count in FROZEN_CONFIG.worker_candidates
            )
            frozen = replace(
                FROZEN_CONFIG,
                corrected_benchmark_root=str(root),
            )
            with patch.object(exp419_config, "FROZEN_CONFIG", frozen):
                self.benchmark.write_benchmark_evidence(
                    evidence,
                    benchmark=benchmark,
                    git_commit="a" * 40,
                    context_sha256="b" * 64,
                    source_manifest_sha256="c" * 64,
                    trial_roots=trials,
                    runtime_measurements=measurements,
                )
                self.assertEqual(
                    self.official.verify_benchmark_evidence(
                        evidence,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    ),
                    benchmark,
                )

                copied = parent / evidence.name
                copied.write_bytes(evidence.read_bytes())
                with self.assertRaisesRegex(
                    ValueError, "evidence|exact|canonical"
                ):
                    self.official.verify_benchmark_evidence(
                        copied,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )

                payload = json.loads(evidence.read_bytes())
                payload["trial_roots"] = [
                    str(parent / "arbitrary" / f"workers-{count}")
                    for count in FROZEN_CONFIG.worker_candidates
                ]
                evidence.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    ValueError, "trial|exact|ordered|canonical"
                ):
                    self.official.verify_benchmark_evidence(
                        evidence,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )

                payload["trial_roots"] = [str(path) for path in trials]
                evidence.write_text(json.dumps(payload))
                target = parent / "worker-alias-target"
                target.mkdir()
                trials[0].symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(
                    ValueError, "trial|exact|ordered|canonical"
                ):
                    self.official.verify_benchmark_evidence(
                        evidence,
                        expected_git_commit="a" * 40,
                        expected_context_sha256="b" * 64,
                        expected_source_manifest_sha256="c" * 64,
                        expected_costs=BALANCED_COSTS,
                    )

    def test_train_metadata_rows_build_exact_costs_without_loading_series(self) -> None:
        costs = self.benchmark.costs_from_metadata_rows(
            (
                ("a", 160, 49),
                ("a", 320, 50),
                ("b", 256, 100),
            ),
            expected_datasets=("a", "b"),
        )
        self.assertEqual(
            tuple((cost.dataset, cost.cost, cost.stratum) for cost in costs),
            (
                ("a", 121_960, "diagnostic_oof"),
                ("b", 518_400, "candidate_eligible"),
            ),
        )

    def test_train_metadata_ignores_rows_outside_frozen_population(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.planner import (
            select_benchmark_datasets,
        )

        expected_rows = (
            ("a", 160, 49),
            ("a", 320, 50),
            ("b", 256, 100),
        )
        expected = self.benchmark.costs_from_metadata_rows(
            expected_rows, expected_datasets=("a", "b")
        )
        observed = self.benchmark.costs_from_metadata_rows(
            (
                ("CornellWhaleChallenge", 1, 0),
                *expected_rows,
                ("Wafer_normal_1", 4_000_000, 1_000_000),
            ),
            expected_datasets=("a", "b"),
        )
        self.assertEqual(observed, expected)
        self.assertEqual(
            select_benchmark_datasets(observed),
            select_benchmark_datasets(expected),
        )

    def test_train_metadata_requires_valid_unique_complete_frozen_population(self) -> None:
        rows = (("a", 160, 49), ("b", 256, 100))
        invalid_expected = (
            ("a", "a"),
            ("a", ""),
            ("a", 1),
            (),
        )
        for expected in invalid_expected:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ValueError, "expected|population|dataset"
            ):
                self.benchmark.costs_from_metadata_rows(
                    rows, expected_datasets=expected
                )
        with self.assertRaisesRegex(ValueError, "cover|missing|population"):
            self.benchmark.costs_from_metadata_rows(
                (("a", 160, 49),),
                expected_datasets=("a", "b"),
            )

    def test_train_metadata_query_remains_train_normal_only(self) -> None:
        source = inspect.getsource(self.benchmark._load_train_metadata_costs)
        self.assertIn("WHERE i.split = 'TRAIN' AND i.label = '0'", source)


class SourceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script("exp419_benchmark_static", BENCHMARK_SCRIPT)

    def _fixture(
        self,
        root: Path,
        source: str = "VALUE = 1\n",
    ) -> tuple[Path, Path]:
        package = root / "experiments/exp419_deterministic_cpu_parallel"
        scratch = root / "scratch"
        package.mkdir(parents=True)
        scratch.mkdir()
        runtime = package / "__init__.py"
        runtime.write_text(source)
        (scratch / "benchmark_exp419_deterministic_cpu_parallel.py").write_text(
            "VALUE = 2\n"
        )
        (scratch / "run_exp419_deterministic_cpu_parallel.py").write_text(
            "VALUE = 3\n"
        )
        return package / "SOURCE_MANIFEST.json", runtime

    def test_checked_in_manifest_matches_exact_runtime_source_set(self) -> None:
        digest = self.module.assert_exp419_label_isolation()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        manifest = json.loads(
            (
                REPO_ROOT
                / "experiments/exp419_deterministic_cpu_parallel"
                / "SOURCE_MANIFEST.json"
            ).read_bytes()
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["sources"]), 12)
        self.assertEqual(
            [entry["path"] for entry in manifest["sources"]],
            sorted(entry["path"] for entry in manifest["sources"]),
        )

    def test_source_change_and_manifest_tamper_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, runtime = self._fixture(root)
            self.module.propose_source_manifest(root, output_path=manifest)
            self.module.verify_source_manifest(root, manifest)
            original = runtime.read_text()
            runtime.write_text(original + "# changed\n")
            with self.assertRaisesRegex(ValueError, "digest|source"):
                self.module.verify_source_manifest(root, manifest)
            runtime.write_text(original)
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "manifest|JSON|canonical"):
                self.module.verify_source_manifest(root, manifest)

    def test_manifest_proposal_differs_until_explicit_regeneration(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, runtime = self._fixture(root)
            self.module.propose_source_manifest(root, output_path=manifest)
            stale = manifest.read_bytes()
            runtime.write_text(runtime.read_text() + "# reviewed change\n")

            proposed = self.module.propose_source_manifest(root)
            self.assertNotEqual(proposed, stale)
            self.assertEqual(manifest.read_bytes(), stale)

            self.module.propose_source_manifest(root, output_path=manifest)
            self.assertEqual(manifest.read_bytes(), proposed)
            self.module.verify_source_manifest(root, manifest)

    def test_manifest_rejects_missing_extra_duplicate_and_traversal_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _runtime = self._fixture(root)
            self.module.propose_source_manifest(root, output_path=manifest)
            original = json.loads(manifest.read_bytes())
            mutations = []
            missing = json.loads(json.dumps(original))
            missing["sources"].pop()
            mutations.append(missing)
            duplicate = json.loads(json.dumps(original))
            duplicate["sources"].append(dict(duplicate["sources"][0]))
            mutations.append(duplicate)
            traversal = json.loads(json.dumps(original))
            traversal["sources"][0]["path"] = "../outside.py"
            mutations.append(traversal)
            for mutation in mutations:
                manifest.write_text(json.dumps(mutation))
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ValueError, "manifest|source|path|duplicate"
                ):
                    self.module.verify_source_manifest(root, manifest)
            self.module.propose_source_manifest(root, output_path=manifest)
            extra = (
                root
                / "experiments/exp419_deterministic_cpu_parallel/extra.py"
            )
            extra.write_text("VALUE = 4\n")
            with self.assertRaisesRegex(ValueError, "extra|set|source"):
                self.module.verify_source_manifest(root, manifest)

    def test_manifest_rejects_symlink_and_nonregular_source(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, runtime = self._fixture(root)
            self.module.propose_source_manifest(root, output_path=manifest)
            target = root / "target.py"
            target.write_text(runtime.read_text())
            runtime.unlink()
            runtime.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink|regular|source"):
                self.module.verify_source_manifest(root, manifest)
            runtime.unlink()
            runtime.mkdir()
            with self.assertRaisesRegex(ValueError, "regular|source"):
                self.module.verify_source_manifest(root, manifest)

    def test_direct_explicit_label_access_is_denied_before_manifest_acceptance(self) -> None:
        snippets = (
            "from secret.labels import load_labels\n",
            "source.load_test_labels()\n",
            'ARTIFACT = "frozen_test_labels.bin"\n',
            'QUERY = "SELECT i.labels_blob FROM instances i"\n',
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _runtime = self._fixture(root, snippet)
                with self.assertRaisesRegex(ValueError, "label"):
                    self.module.propose_source_manifest(
                        root, output_path=manifest
                    )

    def test_guard_literal_exemption_rejects_executable_and_label_payloads(self) -> None:
        snippets = (
            "_FORBIDDEN_KEYS = source.load_test_labels()\n",
            '_FORBIDDEN_KEYS = "frozen_test_labels.bin"\n',
            "_FORBIDDEN_KEYS = {source.load_test_labels}\n",
            "_FORBIDDEN_KEYS = {value for value in ('safe',)}\n",
            '_FORBIDDEN_KEYS = "frozen_" + "test_labels.bin"\n',
            '_FORBIDDEN_KEYS = f"frozen_{\'test_labels.bin\'}"\n',
            "_FORBIDDEN_KEYS = {'safe_but_unapproved'}\n",
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _runtime = self._fixture(root)
                reporting = (
                    root
                    / "experiments/exp419_deterministic_cpu_parallel/reporting.py"
                )
                reporting.write_text(snippet)
                with self.assertRaisesRegex(ValueError, "guard|label"):
                    self.module.propose_source_manifest(
                        root, output_path=manifest
                    )

    def test_dynamic_alias_is_governed_by_frozen_bytes_not_dataflow(self) -> None:
        snippets = (
            "name = input()\nreader = getattr(source, name)\nreader()\n",
            "target = input()\nPath(target).read_bytes()\n",
            "opener = open\ntarget = input()\nopener(target)\n",
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, runtime = self._fixture(root, snippet)
                self.module.propose_source_manifest(root, output_path=manifest)
                self.module.verify_source_manifest(root, manifest)
                runtime.write_text(snippet + "# unreviewed byte\n")
                with self.assertRaisesRegex(ValueError, "digest|source"):
                    self.module.verify_source_manifest(root, manifest)

    def test_manifest_proposal_never_writes_without_explicit_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _runtime = self._fixture(root)
            payload = self.module.propose_source_manifest(root)
            self.assertIsInstance(payload, bytes)
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
