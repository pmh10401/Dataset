from __future__ import annotations

import importlib.util
import json
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    RuntimeBenchmark,
    RuntimeTrial,
)
from experiments.exp419_deterministic_cpu_parallel.coordinator import select_runtime


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = REPO_ROOT / "scratch/benchmark_exp419_deterministic_cpu_parallel.py"
OFFICIAL_SCRIPT = REPO_ROOT / "scratch/run_exp419_deterministic_cpu_parallel.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _passing_benchmark(datasets: tuple[str, ...] = ("a",)) -> RuntimeBenchmark:
    trials = (
        RuntimeTrial(1, 12.0, 100, True, True, "", datasets, "1" * 64),
        RuntimeTrial(2, 5.0, 200, True, True, "", datasets, "2" * 64),
        RuntimeTrial(4, 4.0, 300, True, True, "", datasets, "3" * 64),
        RuntimeTrial(6, 6.0, 400, True, True, "", datasets, "4" * 64),
    )
    return select_runtime(trials)


def _passing_measured_benchmark(module, datasets: tuple[str, ...] = ("a",)):
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
                datasets,
                measurement.evidence_sha256,
            )
        )
    return select_runtime(tuple(trials)), tuple(measurements)


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
        benchmark, measurements = _passing_measured_benchmark(
            self.benchmark, ("a", "b")
        )
        costs = (
            DatasetCost("a", 1, "preprocessing_only"),
            DatasetCost("b", 2, "diagnostic_oof"),
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark.json"
            self.benchmark.write_benchmark_evidence(
                path,
                benchmark=benchmark,
                git_commit="a" * 40,
                context_sha256="b" * 64,
                source_manifest_sha256="c" * 64,
                trial_roots=tuple(
                    Path(temporary) / f"workers-{count}"
                    for count in FROZEN_CONFIG.worker_candidates
                ),
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

            with self.assertRaisesRegex(ValueError, "manifest|provenance"):
                self.official.verify_benchmark_evidence(
                    path,
                    expected_git_commit="a" * 40,
                    expected_context_sha256="b" * 64,
                    expected_source_manifest_sha256="d" * 64,
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
                trial_roots=tuple(
                    Path(temporary) / f"workers-{count}"
                    for count in FROZEN_CONFIG.worker_candidates
                ),
                runtime_measurements=measurements,
            )
            forged = json.loads(path.read_text())
            forged["benchmark"]["selected_worker_count"] = 6
            path.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "digest|selection|benchmark"):
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
        costs = (
            DatasetCost("a", 1, "preprocessing_only"),
            DatasetCost("b", 2, "diagnostic_oof"),
            DatasetCost("c", 3, "candidate_eligible"),
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark.json"
            self.benchmark.write_benchmark_evidence(
                path,
                benchmark=benchmark,
                git_commit="a" * 40,
                context_sha256="b" * 64,
                source_manifest_sha256="c" * 64,
                trial_roots=tuple(
                    Path(temporary) / f"workers-{count}"
                    for count in FROZEN_CONFIG.worker_candidates
                ),
                runtime_measurements=measurements,
            )
            with self.assertRaisesRegex(ValueError, "subset|datasets"):
                self.official.verify_benchmark_evidence(
                    path,
                    expected_git_commit="a" * 40,
                    expected_context_sha256="b" * 64,
                    expected_source_manifest_sha256="c" * 64,
                    expected_costs=costs,
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
            path = Path(temporary) / "benchmark.json"
            self.benchmark.write_benchmark_evidence(
                path,
                benchmark=benchmark,
                git_commit="a" * 40,
                context_sha256="b" * 64,
                source_manifest_sha256="c" * 64,
                trial_roots=tuple(
                    Path(temporary) / f"workers-{count}"
                    for count in FROZEN_CONFIG.worker_candidates
                ),
                runtime_measurements=measurements,
            )
            payload = json.loads(path.read_bytes())
            payload["runtime_measurements"][0]["process_cpu_seconds"] = -1.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "CPU|cpu|finite|nonnegative"):
                self.official.verify_benchmark_evidence(
                    path,
                    expected_git_commit="a" * 40,
                    expected_context_sha256="b" * 64,
                    expected_source_manifest_sha256="c" * 64,
                    expected_costs=(
                        DatasetCost("a", 1, "preprocessing_only"),
                    ),
                )

    def test_benchmark_requires_fresh_disjoint_candidate_roots(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            roots = self.benchmark.prepare_trial_roots(root)
            self.assertEqual(
                tuple(roots),
                FROZEN_CONFIG.worker_candidates,
            )
            self.assertEqual(
                len({path.resolve() for path in roots.values()}),
                len(FROZEN_CONFIG.worker_candidates),
            )
            self.assertTrue(
                all(path.parent == root.resolve() for path in roots.values())
            )
            with self.assertRaisesRegex(ValueError, "new"):
                self.benchmark.prepare_trial_roots(root)

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
        self.assertEqual(len(manifest["sources"]), 10)
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
