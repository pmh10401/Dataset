from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp419_deterministic_cpu_parallel import config
from experiments.exp419_deterministic_cpu_parallel.config import (
    FROZEN_CONFIG,
    corrected_benchmark_evidence_path,
    corrected_trial_roots,
    require_exact_benchmark_evidence_path,
    require_exact_benchmark_root,
    require_exact_trial_roots,
    seed_workers_for_outer_workers,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG as EXP417_FROZEN_CONFIG


class Exp419ConfigTests(unittest.TestCase):
    def test_exact_runtime_contract_is_frozen(self):
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp419")
        self.assertEqual(FROZEN_CONFIG.worker_candidates, (1, 2, 4, 6))
        self.assertEqual(FROZEN_CONFIG.start_method, "spawn")
        self.assertEqual(FROZEN_CONFIG.speedup_gate, 2.0)
        self.assertEqual(FROZEN_CONFIG.memory_limit_bytes, 27_487_790_694)
        self.assertEqual(FROZEN_CONFIG.torch_threads, 1)
        self.assertEqual(FROZEN_CONFIG.benchmark_proxy_headroom_gate, 2.5)
        self.assertEqual(FROZEN_CONFIG.benchmark_proxy_headroom_workers, (4, 6))
        self.assertEqual(
            FROZEN_CONFIG.corrected_benchmark_root,
            "/Users/mac/Documents/Dataset/"
            "exp419_deterministic_cpu_parallel_benchmark_balanced",
        )
        self.assertEqual(
            FROZEN_CONFIG.benchmark_evidence_filename,
            "EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json",
        )

    def test_modified_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "worker_candidates"):
            replace(FROZEN_CONFIG, worker_candidates=(1, 8)).validate()
        with self.assertRaisesRegex(ValueError, "benchmark_proxy_headroom_gate"):
            replace(FROZEN_CONFIG, benchmark_proxy_headroom_gate=2.0).validate()
        with self.assertRaisesRegex(ValueError, "corrected_benchmark_root"):
            replace(
                FROZEN_CONFIG, corrected_benchmark_root="/tmp/benchmark"
            ).validate()

    def test_outer_and_seed_worker_layout_is_exact_and_within_cpu_budget(self):
        self.assertEqual(
            FROZEN_CONFIG.seed_workers_by_outer_worker,
            ((1, 1), (2, 3), (4, 3), (6, 2)),
        )
        for outer_workers, seed_workers in (
            (1, 1), (2, 3), (4, 3), (6, 2)
        ):
            with self.subTest(outer_workers=outer_workers):
                self.assertEqual(
                    seed_workers_for_outer_workers(outer_workers), seed_workers
                )
                self.assertLessEqual(outer_workers * seed_workers, 12)
        with self.assertRaisesRegex(ValueError, "outer worker"):
            seed_workers_for_outer_workers(3)

    def test_exp417_scientific_config_database_and_lanes_are_bound(self):
        required_fields = {
            "exp417_config",
            "database_sha256",
            "expected_hard",
            "expected_standard",
            "expected_priority",
            "expected_automatic_normal",
            "exp291_prediction_sha256",
            "exp291_manifest_sha256",
            "exp292_prediction_sha256",
            "exp292_manifest_sha256",
        }
        self.assertTrue(required_fields <= set(FROZEN_CONFIG.__dataclass_fields__))
        self.assertIs(FROZEN_CONFIG.exp417_config, EXP417_FROZEN_CONFIG)
        self.assertEqual(FROZEN_CONFIG.database_sha256, EXP417_FROZEN_CONFIG.database_sha256)
        self.assertEqual(FROZEN_CONFIG.expected_hard, 2218)
        self.assertEqual(FROZEN_CONFIG.expected_standard, 806)
        self.assertEqual(FROZEN_CONFIG.expected_priority, 31)
        self.assertEqual(FROZEN_CONFIG.expected_automatic_normal, 35)
        self.assertEqual(FROZEN_CONFIG.exp291_prediction_sha256, EXP417_FROZEN_CONFIG.exp291_prediction_sha256)
        self.assertEqual(FROZEN_CONFIG.exp291_manifest_sha256, EXP417_FROZEN_CONFIG.exp291_manifest_sha256)
        self.assertEqual(FROZEN_CONFIG.exp292_prediction_sha256, EXP417_FROZEN_CONFIG.exp292_prediction_sha256)
        self.assertEqual(FROZEN_CONFIG.exp292_manifest_sha256, EXP417_FROZEN_CONFIG.exp292_manifest_sha256)

    def test_modified_exp417_database_binding_is_rejected(self):
        self.assertIn("database_sha256", FROZEN_CONFIG.__dataclass_fields__)
        with self.assertRaisesRegex(ValueError, "database_sha256"):
            replace(FROZEN_CONFIG, database_sha256="0" * 64).validate()

    def test_exact_benchmark_filesystem_contract_rejects_aliases_and_reordering(
        self,
    ):
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "exp419_deterministic_cpu_parallel_benchmark_balanced"
            frozen = replace(
                FROZEN_CONFIG,
                corrected_benchmark_root=str(root),
            )
            with patch.object(config, "FROZEN_CONFIG", frozen):
                evidence = root / FROZEN_CONFIG.benchmark_evidence_filename
                trials = tuple(
                    root / f"workers-{count}"
                    for count in FROZEN_CONFIG.worker_candidates
                )
                self.assertEqual(corrected_benchmark_evidence_path(), evidence)
                self.assertEqual(corrected_trial_roots(), trials)
                self.assertEqual(require_exact_benchmark_root(root), root)
                self.assertEqual(
                    require_exact_benchmark_evidence_path(evidence), evidence
                )
                self.assertEqual(require_exact_trial_roots(trials), trials)

                arbitrary = parent / "arbitrary"
                with self.assertRaisesRegex(ValueError, "exact|canonical"):
                    require_exact_benchmark_root(arbitrary)
                with self.assertRaisesRegex(ValueError, "exact|canonical"):
                    require_exact_benchmark_evidence_path(
                        arbitrary / evidence.name
                    )
                with self.assertRaisesRegex(ValueError, "exact|ordered"):
                    require_exact_trial_roots(tuple(reversed(trials)))

                target_parent = parent / "target"
                target_parent.mkdir()
                alias_parent = parent / "alias"
                alias_parent.symlink_to(target_parent, target_is_directory=True)
                alias_root = alias_parent / root.name
                with self.assertRaisesRegex(ValueError, "exact|canonical"):
                    require_exact_benchmark_root(alias_root)
