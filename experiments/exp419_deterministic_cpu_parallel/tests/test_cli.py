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
                expected_costs=costs,
            )
            self.assertEqual(observed, benchmark)

            forged = json.loads(path.read_text())
            forged["benchmark"]["selected_worker_count"] = 6
            path.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "digest|selection|benchmark"):
                self.official.verify_benchmark_evidence(
                    path,
                    expected_git_commit="a" * 40,
                    expected_context_sha256="b" * 64,
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


class StaticIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script("exp419_benchmark_static", BENCHMARK_SCRIPT)

    def test_exp419_sources_have_no_test_label_access_path(self) -> None:
        paths = tuple(
            sorted(
                (REPO_ROOT / "experiments/exp419_deterministic_cpu_parallel").glob(
                    "*.py"
                )
            )
        ) + (BENCHMARK_SCRIPT, OFFICIAL_SCRIPT)
        self.module.assert_source_label_isolation(
            {str(path): path.read_text() for path in paths}
        )

    def test_adversarial_label_access_snippets_fail_closed(self) -> None:
        snippets = {
            "aliased-import.py":
                "from secret import load_labels as values_loader\nvalues_loader()\n",
            "attribute-loader.py":
                "import secret as source\nsource.load_test_labels()\n",
            "stored-attribute-loader.py":
                "import secret as source\nreader = source.load_labels\nreader()\n",
            "concatenated-path.py":
                'LABEL_PATH = "/private/" + "labels.bin"\nPath(LABEL_PATH).read_bytes()\n',
            "formatted-path.py":
                'path = f"/private/{\'test\'}_{\'labels\'}.npy"\nopen(path, "rb")\n',
            "obfuscated-sql.py":
                'QUERY = "SEL" + "ECT i." + "label\\nFR" + "OM instances"\n',
            "quoted-sql.py":
                'QUERY = """SELECT i.[label], i.`labels_blob`\nFROM instances"""\n',
            "frozen-artifact.py":
                'ARTIFACT = "frozen_" + "test_labels"\n',
            "fake-allowlist-artifact.py":
                '_MY_ALLOWLIST = "test_labels.bin"\n',
            "getattr-loader.py":
                'loader = getattr(source, "load_" + "labels")\nloader()\n',
            "dynamic-label-path.py":
                "label_path = input()\nPath(label_path).read_bytes()\n",
            "dynamic-callable-getattr.py":
                "name = input()\nreader = getattr(source, name)\nreader()\n",
            "aliased-filesystem-sink.py":
                "opener = open\ndata_path = input()\nopener(data_path)\n",
            str(BENCHMARK_SCRIPT): (
                "def prepare_trial_roots(root):\n"
                "    label_path = input()\n"
                "    return Path(root / label_path).read_bytes()\n"
            ),
            "method-alias.py":
                "reader = Path.read_bytes\ntarget = input()\nreader(target)\n",
            "destructured-sinks.py":
                "opener, reader = open, Path.read_text\n",
            "container-sink.py":
                "operations = [open, Path.read_bytes]\n",
            "container-getattr.py": (
                "name = input()\n"
                "reader = getattr(source, name)\n"
                "operations = [reader]\n"
                "operations[0]()\n"
            ),
            "attribute-getattr.py": (
                "name = input()\n"
                "reader = getattr(source, name)\n"
                "holder.reader = reader\n"
            ),
        }
        for name, source in snippets.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "label"
            ):
                self.module.assert_source_label_isolation({name: source})

    def test_guard_denylist_and_normal_train_predicate_are_allowed(self) -> None:
        sources = {
            "guard.py": (
                '_FORBIDDEN_KEYS = {"label", "labels_blob"}\n'
                'QUERY = """SELECT i.instance_index, i.values_blob\n'
                "FROM instances i WHERE i.split='TRAIN' AND i.label='0'\"\"\"\n"
            )
        }
        self.module._verify_sensitive_inventory(
            sources, self.module.propose_sensitive_inventory(sources)
        )

    def test_dynamic_nonfilesystem_getattr_requires_explicit_inventory(self) -> None:
        sources = {
            "benign.py": (
                "name = input()\n"
                "value = getattr(source, name)\n"
                "result = (value is None, name.casefold())\n"
            )
        }
        with self.assertRaisesRegex(ValueError, "label isolation inventory"):
            self.module.assert_source_label_isolation(sources)
        self.module._verify_sensitive_inventory(
            sources, self.module.propose_sensitive_inventory(sources)
        )

    def test_inventory_binds_upstream_scope_and_occurrence_count(self) -> None:
        original = {
            "scope.py": (
                "from pathlib import Path\n"
                "def read(root):\n"
                "    target = root / 'worker-1'\n"
                "    return Path(target).resolve()\n"
            )
        }
        approved = self.module.propose_sensitive_inventory(original)
        self.module._verify_sensitive_inventory(original, approved)
        attacks = (
            {
                "scope.py": (
                    "from pathlib import Path\n"
                    "def read(root):\n"
                    "    target = input()\n"
                    "    return Path(target).resolve()\n"
                )
            },
            {
                "scope.py": (
                    "from pathlib import Path\n"
                    "def read(root):\n"
                    "    target = root / 'worker-1'\n"
                    "    Path(target).resolve()\n"
                    "    return Path(target).resolve()\n"
                )
            },
        )
        for attack in attacks:
            with self.subTest(source=attack), self.assertRaisesRegex(
                ValueError, "label isolation inventory"
            ):
                self.module._verify_sensitive_inventory(attack, approved)

    def test_inventory_rejects_stale_missing_and_duplicate_entries(self) -> None:
        sources = {
            "inventory.py": "from pathlib import Path\nPath('safe').resolve()\n"
        }
        approved = self.module.propose_sensitive_inventory(sources)
        mutations = (
            approved[:-1],
            approved + approved[:1],
            approved + ((
                "stale.py", "<module>", "call:path", "0" * 64,
                "1" * 64, "body[0]",
            ),),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, "label isolation inventory"
            ):
                self.module._verify_sensitive_inventory(sources, mutation)

    def test_uninventoried_sensitive_syntax_and_reflection_fail(self) -> None:
        snippets = {
            "lambda.py": "reader = lambda: open('safe')\n",
            "decorator.py": "@getattr(source, 'decorate')\ndef run(): pass\n",
            "default.py": "def run(value=Path('safe')): pass\n",
            "sql.py": "connection.execute(input())\n",
            "import.py": "__import__(input())\n",
            "reflection.py": "operator.attrgetter(input())\n",
            "eval.py": "eval(input())\n",
            "exec.py": "exec(input())\n",
        }
        for name, source in snippets.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "label"
            ):
                self.module.assert_source_label_isolation({name: source})

    def test_guard_helper_filesystem_call_is_not_exempt(self) -> None:
        source = (
            "def assert_source_label_isolation(sources):\n"
            "    return open(input()).read()\n"
        )
        with self.assertRaisesRegex(ValueError, "label isolation inventory"):
            self.module.assert_source_label_isolation({"guard.py": source})


if __name__ == "__main__":
    unittest.main()
