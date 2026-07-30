from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel.contracts import (
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

    def test_official_main_reaches_clean_git_before_root_validation(self) -> None:
        events: list[str] = []

        def dirty():
            events.append("clean")
            raise ValueError("dirty")

        stages = self.official.make_test_stages(clean_git=dirty)
        parser = SimpleNamespace(
            parse_args=lambda _argv: SimpleNamespace(
                checkpoint_dir=Path("/checkpoint"),
                output_dir=Path("/output"),
            )
        )
        with patch.object(self.official, "build_parser", return_value=parser), \
             patch.object(self.official, "_build_stages", return_value=stages), \
             patch.object(
                 self.official,
                 "require_disjoint_roots",
                 side_effect=lambda *_args: events.append("roots"),
             ), self.assertRaisesRegex(ValueError, "dirty"):
            self.official.main([])
        self.assertEqual(events, ["clean"])

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
        benchmark = _passing_benchmark(("a", "b"))
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
            )
            observed = self.official.verify_benchmark_evidence(
                path,
                expected_git_commit="a" * 40,
                expected_context_sha256="b" * 64,
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
    def test_exp419_sources_have_no_test_label_access_path(self) -> None:
        paths = tuple(
            sorted(
                (REPO_ROOT / "experiments/exp419_deterministic_cpu_parallel").glob(
                    "*.py"
                )
            )
        ) + (BENCHMARK_SCRIPT, OFFICIAL_SCRIPT)
        forbidden_artifacts = {
            "test_labels.npy",
            "test_labels.json",
            "frozen_test_labels",
            "labels_blob",
        }
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            forbidden_key_nodes: set[ast.AST] = set()
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "_FORBIDDEN_KEYS"
                        for target in node.targets
                    )
                ):
                    forbidden_key_nodes.update(ast.walk(node.value))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported = []
                    if isinstance(node, ast.Import):
                        imported = [alias.name for alias in node.names]
                    else:
                        imported = [
                            f"{node.module or ''}.{alias.name}"
                            for alias in node.names
                        ]
                    self.assertFalse(
                        any(
                            "label" in name.lower()
                            and "assert_label_free_sql_contract" not in name
                            for name in imported
                        ),
                        f"forbidden label loader import in {path}",
                    )
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    lowered = node.value.lower()
                    if node not in forbidden_key_nodes:
                        self.assertFalse(
                            any(name in lowered for name in forbidden_artifacts),
                            f"forbidden frozen label artifact in {path}",
                        )
                    if "select " in lowered and " from " in lowered:
                        projection = lowered.split(" from ", 1)[0]
                        self.assertNotIn("label", projection)


if __name__ == "__main__":
    unittest.main()
