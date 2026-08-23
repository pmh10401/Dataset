"""Frozen identity and contract tests for Exp451 verified accelerated resume."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
)
from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import (
    APPROVED_DESIGN_PATH_ANTIGRAVITY,
    APPROVED_PLAN_PATH_ANTIGRAVITY,
    EVALUATION_ROOT,
    EXP450_OFFICIAL_ROOT,
    EXP450_SOURCE_CATALOG_SHA256,
    EXP450_SOURCE_COMMIT,
    EXP450_SOURCE_CONTEXT_SHA256,
    EXP450_SOURCE_ROOT,
    EXP450_SOURCE_TRANSFORMS_SHA256,
    EXP451_FINAL_COUNTS_BY_CONDITION,
    EXP451_MIGRATED_COORDINATES,
    EXP451_MIGRATED_COUNTS_BY_CONDITION,
    EXP451_REMAINING_COORDINATES,
    EXP451_TOTAL_COORDINATES,
    FROZEN_EXP451_CONFIG,
    OFFICIAL_ROOT,
    REPORT_ROOT,
    SMOKE_ROOT,
    FrozenExp451Config,
)
from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
    Exp451BackendDecision,
    Exp451Condition,
    Exp451MigrationEntry,
    Exp451MigrationManifest,
    Exp451RunContext,
    canonical_json_bytes,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenPaperConfig


DUMMY_DIGEST = "a" * 64
DUMMY_DIGEST_2 = "b" * 64
DUMMY_DIGEST_3 = "c" * 64


def _build_valid_4234_entries() -> list[Exp451MigrationEntry]:
    entries: list[Exp451MigrationEntry] = []
    # A: 1117 datasets
    for i in range(1117):
        ds = f"ds_a_{i:04d}"
        entries.append(
            Exp451MigrationEntry(
                dataset=ds,
                condition_id="A",
                source_relative_path=f"checkpoints/A/{ds}.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path=f"checkpoints/A/{ds}.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="imported_exp450_mps",
            )
        )
    # B: 1117 datasets
    for i in range(1117):
        ds = f"ds_b_{i:04d}"
        entries.append(
            Exp451MigrationEntry(
                dataset=ds,
                condition_id="B",
                source_relative_path=f"checkpoints/B/{ds}.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path=f"checkpoints/B/{ds}.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="imported_exp450_mps",
            )
        )
    # C: 1000 datasets
    for i in range(1000):
        ds = f"ds_c_{i:04d}"
        entries.append(
            Exp451MigrationEntry(
                dataset=ds,
                condition_id="C",
                source_relative_path=f"checkpoints/C/{ds}.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path=f"checkpoints/C/{ds}.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="imported_exp450_mps",
            )
        )
    # D: 1000 datasets
    for i in range(1000):
        ds = f"ds_d_{i:04d}"
        entries.append(
            Exp451MigrationEntry(
                dataset=ds,
                condition_id="D",
                source_relative_path=f"checkpoints/D/{ds}.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path=f"checkpoints/D/{ds}.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="imported_exp450_mps",
            )
        )
    return entries


class Exp451ConfigTests(unittest.TestCase):
    def test_exp451_config_binds_isolated_roots_and_unchanged_population(self) -> None:
        self.assertEqual(FROZEN_EXP451_CONFIG.experiment_id, "Exp451")
        self.assertEqual(FROZEN_EXP451_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP451_CONFIG.test_row_count, 195900)
        self.assertTrue(all("exp451" in str(root).lower() for root in FROZEN_EXP451_CONFIG.roots))
        self.assertTrue(all("antigravity" in root.name for root in FROZEN_EXP451_CONFIG.roots))

    def test_four_conditions_are_exact_factorial(self) -> None:
        self.assertEqual(
            tuple(
                (condition.condition_id, condition.length_mode, condition.image_mode)
                for condition in FROZEN_EXP451_CONFIG.conditions
            ),
            (
                ("A", "linear", "linear"),
                ("B", "polyphase_pchip", "linear"),
                ("C", "linear", "area"),
                ("D", "polyphase_pchip", "area"),
            ),
        )

    def test_source_constants_match_exp450_authority(self) -> None:
        self.assertEqual(
            FROZEN_EXP451_CONFIG.source_context_sha256,
            "5d5b44aec0e738ae61d9dafd85d9eaea1448baba49705c4d042c5ed1d6886f83",
        )
        self.assertEqual(
            FROZEN_EXP451_CONFIG.source_transforms_sha256,
            "28b5fcf731fc7ed31546939bbbaa932e38770f3f6e77864fc353328ddaf059c7",
        )
        self.assertEqual(
            FROZEN_EXP451_CONFIG.source_catalog_sha256,
            "99aeceb59e98cac160ddc87b252a9be7aca087ba78627847bfbe4005f4b39c4b",
        )
        self.assertEqual(FROZEN_EXP451_CONFIG.source_commit, "eac0d3b4")
        self.assertEqual(
            FROZEN_EXP451_CONFIG.source_root,
            Path("/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_official_antigravity_20260820"),
        )
        self.assertEqual(EXP450_OFFICIAL_ROOT, EXP450_SOURCE_ROOT)
        self.assertEqual(EXP450_SOURCE_CONTEXT_SHA256, FROZEN_EXP451_CONFIG.source_context_sha256)
        self.assertEqual(EXP450_SOURCE_TRANSFORMS_SHA256, FROZEN_EXP451_CONFIG.source_transforms_sha256)
        self.assertEqual(EXP450_SOURCE_CATALOG_SHA256, FROZEN_EXP451_CONFIG.source_catalog_sha256)
        self.assertEqual(EXP450_SOURCE_COMMIT, "eac0d3b4")

    def test_roots_and_inventory_constants_are_frozen(self) -> None:
        self.assertEqual(
            SMOKE_ROOT,
            Path("/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_smoke_antigravity_20260823"),
        )
        self.assertEqual(
            OFFICIAL_ROOT,
            Path("/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_official_antigravity_20260823"),
        )
        self.assertEqual(
            EVALUATION_ROOT,
            Path("/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_evaluation_antigravity_20260823"),
        )
        self.assertEqual(
            REPORT_ROOT,
            Path("/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_report_antigravity_20260823"),
        )
        self.assertEqual(EXP451_TOTAL_COORDINATES, 4468)
        self.assertEqual(EXP451_MIGRATED_COORDINATES, 4234)
        self.assertEqual(EXP451_REMAINING_COORDINATES, 234)
        self.assertEqual(EXP451_MIGRATED_COUNTS_BY_CONDITION, {"A": 1117, "B": 1117, "C": 1000, "D": 1000})
        self.assertEqual(EXP451_FINAL_COUNTS_BY_CONDITION, {"A": 1117, "B": 1117, "C": 1117, "D": 1117})
        self.assertEqual(
            APPROVED_DESIGN_PATH_ANTIGRAVITY,
            Path("docs/superpowers/specs/2026-08-23-exp451-verified-accelerated-resume-antigravity-design.md"),
        )
        self.assertEqual(
            APPROVED_PLAN_PATH_ANTIGRAVITY,
            Path("docs/superpowers/plans/2026-08-23-exp451-verified-accelerated-resume-antigravity-plan.md"),
        )

    def test_frozen_config_rejects_invalid_attributes(self) -> None:
        paper = FrozenPaperConfig()
        with self.assertRaisesRegex(ValueError, "identity"):
            FrozenExp451Config(
                experiment_id="Exp450",
                database_path=Path(paper.database_path),
                database_sha256=paper.database_sha256,
                dataset_count=1117,
                test_row_count=195900,
                smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
                policy_id="hard_plus_stable",
                conditions=FROZEN_EXP451_CONFIG.conditions,
                roots=FROZEN_EXP451_CONFIG.roots,
                approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
                source_root=EXP450_OFFICIAL_ROOT,
                source_context_sha256=EXP450_SOURCE_CONTEXT_SHA256,
                source_transforms_sha256=EXP450_SOURCE_TRANSFORMS_SHA256,
                source_catalog_sha256=EXP450_SOURCE_CATALOG_SHA256,
                source_commit=EXP450_SOURCE_COMMIT,
            )

        with self.assertRaisesRegex(ValueError, "population"):
            FrozenExp451Config(
                experiment_id="Exp451",
                database_path=Path(paper.database_path),
                database_sha256=paper.database_sha256,
                dataset_count=1000,
                test_row_count=195900,
                smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
                policy_id="hard_plus_stable",
                conditions=FROZEN_EXP451_CONFIG.conditions,
                roots=FROZEN_EXP451_CONFIG.roots,
                approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
                source_root=EXP450_OFFICIAL_ROOT,
                source_context_sha256=EXP450_SOURCE_CONTEXT_SHA256,
                source_transforms_sha256=EXP450_SOURCE_TRANSFORMS_SHA256,
                source_catalog_sha256=EXP450_SOURCE_CATALOG_SHA256,
                source_commit=EXP450_SOURCE_COMMIT,
            )

        with self.assertRaisesRegex(ValueError, "policy"):
            FrozenExp451Config(
                experiment_id="Exp451",
                database_path=Path(paper.database_path),
                database_sha256=paper.database_sha256,
                dataset_count=1117,
                test_row_count=195900,
                smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
                policy_id="invalid_policy",
                conditions=FROZEN_EXP451_CONFIG.conditions,
                roots=FROZEN_EXP451_CONFIG.roots,
                approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
                source_root=EXP450_OFFICIAL_ROOT,
                source_context_sha256=EXP450_SOURCE_CONTEXT_SHA256,
                source_transforms_sha256=EXP450_SOURCE_TRANSFORMS_SHA256,
                source_catalog_sha256=EXP450_SOURCE_CATALOG_SHA256,
                source_commit=EXP450_SOURCE_COMMIT,
            )


class Exp451ContractsTests(unittest.TestCase):
    def test_canonical_json_bytes_is_deterministic_and_rejects_nonfinite(self) -> None:
        encoded = canonical_json_bytes({"b": 2, "a": 1})
        self.assertEqual(encoded, b'{"a":1,"b":2}\n')
        with self.assertRaisesRegex(ValueError, "canonical finite JSON"):
            canonical_json_bytes({"nan": float("nan")})

    def test_condition_rejects_nonapproved_factors(self) -> None:
        with self.assertRaisesRegex(ValueError, "approved factorial"):
            Exp451Condition("D", "polyphase_pchip", "linear")
        with self.assertRaisesRegex(ValueError, "approved factorial"):
            Exp451Condition("E", "linear", "linear")

    def test_backend_decision_preserves_provenance_and_immutability(self) -> None:
        decision = Exp451BackendDecision(
            tensor_backend="mps",
            torch_version="2.13.0",
            mps_available=True,
            mlx_available=False,
            logical_cpu_count=12,
            cpu_workers=11,
            inner_numerical_threads=1,
            tensor_eligible=True,
            transfer_compilation_overhead="bounded_device_transfer_and_cpu_pool_fallback",
            deterministic_numerical_parity="finite_scores_identical_candidate_indices_within_tolerance",
            canonical_cpu_fallback="pytorch_cpu_native_adaptive_avg_pool",
            benchmark_requirement="mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
        )
        self.assertEqual(
            decision.artifact_record(),
            {
                "tensor_backend": "mps",
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": False,
                "logical_cpu_count": 12,
                "cpu_workers": 11,
                "inner_numerical_threads": 1,
                "tensor_eligible": True,
                "transfer_compilation_overhead": "bounded_device_transfer_and_cpu_pool_fallback",
                "deterministic_numerical_parity": "finite_scores_identical_candidate_indices_within_tolerance",
                "canonical_cpu_fallback": "pytorch_cpu_native_adaptive_avg_pool",
                "benchmark_requirement": "mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
            },
        )
        self.assertTrue(decision.tensor_eligibility)
        self.assertEqual(
            decision.parity_numerical_requirements,
            "finite_scores_identical_candidate_indices_within_tolerance",
        )
        self.assertEqual(
            decision.benchmark_required,
            "mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
        )
        with self.assertRaises(FrozenInstanceError):
            decision.cpu_workers = 1  # type: ignore[misc]

        with self.assertRaisesRegex(ValueError, "tensor backend"):
            Exp451BackendDecision(
                tensor_backend="cpu",
                torch_version="2.13.0",
                mps_available=True,
                mlx_available=False,
                logical_cpu_count=12,
                cpu_workers=11,
            )

    def test_run_context_requires_absolute_root_valid_digests_and_closure(self) -> None:
        backend = Exp451BackendDecision(
            tensor_backend="mps",
            torch_version="2.13.0",
            mps_available=True,
            mlx_available=False,
            logical_cpu_count=12,
            cpu_workers=11,
        )
        context = Exp451RunContext(
            mode="full",
            run_root=OFFICIAL_ROOT,
            database_sha256=DUMMY_DIGEST,
            catalog_sha256=DUMMY_DIGEST_2,
            code_closure={"transforms_antigravity.py": DUMMY_DIGEST_3},
            backend=backend,
            config_sha256=DUMMY_DIGEST,
        )
        self.assertEqual(len(context.sha256), 64)
        self.assertEqual(context.mode, "full")
        self.assertEqual(context.run_root, OFFICIAL_ROOT)

        with self.assertRaisesRegex(ValueError, "run mode"):
            Exp451RunContext(
                mode="invalid",
                run_root=OFFICIAL_ROOT,
                database_sha256=DUMMY_DIGEST,
                catalog_sha256=DUMMY_DIGEST_2,
                code_closure={"a.py": DUMMY_DIGEST_3},
                backend=backend,
                config_sha256=DUMMY_DIGEST,
            )

        with self.assertRaisesRegex(ValueError, "run root must be absolute"):
            Exp451RunContext(
                mode="full",
                run_root=Path("relative/path"),
                database_sha256=DUMMY_DIGEST,
                catalog_sha256=DUMMY_DIGEST_2,
                code_closure={"a.py": DUMMY_DIGEST_3},
                backend=backend,
                config_sha256=DUMMY_DIGEST,
            )

    def test_migration_entry_validates_fields_and_formats_artifact_record(self) -> None:
        entry = Exp451MigrationEntry(
            dataset="dataset_001",
            condition_id="A",
            source_relative_path="checkpoints/A/dataset_001.json",
            source_envelope_sha256=DUMMY_DIGEST,
            payload_sha256=DUMMY_DIGEST_2,
            target_relative_path="checkpoints/A/dataset_001.json",
            target_envelope_sha256=DUMMY_DIGEST_3,
            execution_origin="imported_exp450_mps",
        )
        self.assertEqual(
            entry.artifact_record(),
            {
                "dataset": "dataset_001",
                "condition_id": "A",
                "source_relative_path": "checkpoints/A/dataset_001.json",
                "source_envelope_sha256": DUMMY_DIGEST,
                "payload_sha256": DUMMY_DIGEST_2,
                "target_relative_path": "checkpoints/A/dataset_001.json",
                "target_envelope_sha256": DUMMY_DIGEST_3,
                "execution_origin": "imported_exp450_mps",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            entry.dataset = "other"  # type: ignore[misc]

        with self.assertRaisesRegex(ValueError, "condition_id"):
            Exp451MigrationEntry(
                dataset="dataset_001",
                condition_id="Z",
                source_relative_path="checkpoints/A/dataset_001.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path="checkpoints/A/dataset_001.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="imported_exp450_mps",
            )

        with self.assertRaisesRegex(ValueError, "digest"):
            Exp451MigrationEntry(
                dataset="dataset_001",
                condition_id="A",
                source_relative_path="checkpoints/A/dataset_001.json",
                source_envelope_sha256="bad_digest",
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path="checkpoints/A/dataset_001.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="imported_exp450_mps",
            )

    def test_migration_entry_rejects_non_imported_origin(self) -> None:
        with self.assertRaisesRegex(ValueError, "execution_origin"):
            Exp451MigrationEntry(
                dataset="dataset_001",
                condition_id="A",
                source_relative_path="checkpoints/A/dataset_001.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path="checkpoints/A/dataset_001.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="exp451_native_cpu_pool_fallback",
            )

        with self.assertRaisesRegex(ValueError, "execution_origin"):
            Exp451MigrationEntry(
                dataset="dataset_001",
                condition_id="A",
                source_relative_path="checkpoints/A/dataset_001.json",
                source_envelope_sha256=DUMMY_DIGEST,
                payload_sha256=DUMMY_DIGEST_2,
                target_relative_path="checkpoints/A/dataset_001.json",
                target_envelope_sha256=DUMMY_DIGEST_3,
                execution_origin="other_origin",
            )

    def test_migration_entry_rejects_unsafe_relative_paths(self) -> None:
        unsafe_paths = [
            "/absolute/path/dataset.json",
            "\\absolute\\path\\dataset.json",
            "",
            "   ",
            ".",
            "..",
            "./dataset.json",
            "../dataset.json",
            "checkpoints/./dataset.json",
            "checkpoints/../dataset.json",
            "checkpoints/A/../../escape.json",
        ]
        for unsafe in unsafe_paths:
            with self.subTest(source_path=unsafe):
                with self.assertRaisesRegex(ValueError, "source_relative_path"):
                    Exp451MigrationEntry(
                        dataset="dataset_001",
                        condition_id="A",
                        source_relative_path=unsafe,
                        source_envelope_sha256=DUMMY_DIGEST,
                        payload_sha256=DUMMY_DIGEST_2,
                        target_relative_path="checkpoints/A/dataset_001.json",
                        target_envelope_sha256=DUMMY_DIGEST_3,
                        execution_origin="imported_exp450_mps",
                    )
            with self.subTest(target_path=unsafe):
                with self.assertRaisesRegex(ValueError, "target_relative_path"):
                    Exp451MigrationEntry(
                        dataset="dataset_001",
                        condition_id="A",
                        source_relative_path="checkpoints/A/dataset_001.json",
                        source_envelope_sha256=DUMMY_DIGEST,
                        payload_sha256=DUMMY_DIGEST_2,
                        target_relative_path=unsafe,
                        target_envelope_sha256=DUMMY_DIGEST_3,
                        execution_origin="imported_exp450_mps",
                    )

    def test_migration_manifest_requires_exact_unique_inventory(self) -> None:
        source_context = DUMMY_DIGEST

        # Empty entries must fail
        with self.assertRaises(ValueError):
            Exp451MigrationManifest(source_context_sha256=source_context, entries=())

        # Construct full valid 4,234 entries
        valid_entries = _build_valid_4234_entries()
        self.assertEqual(len(valid_entries), 4234)

        # Valid manifest creation must succeed
        manifest = Exp451MigrationManifest(
            source_context_sha256=source_context,
            entries=tuple(valid_entries),
        )
        self.assertEqual(manifest.total_count, 4234)
        self.assertEqual(
            manifest.counts_by_condition,
            {"A": 1117, "B": 1117, "C": 1000, "D": 1000},
        )
        self.assertEqual(len(manifest.sha256), 64)
        self.assertEqual(len(manifest.entry_map), 4234)

        record = manifest.artifact_record()
        self.assertEqual(record["source_context_sha256"], source_context)
        self.assertEqual(record["total_entries"], 4234)
        self.assertEqual(len(record["entries"]), 4234)

        # Duplicate coordinate must fail
        dup_entries = list(valid_entries)
        # Replace last entry with a duplicate of the first entry
        dup_entries[-1] = Exp451MigrationEntry(
            dataset=valid_entries[0].dataset,
            condition_id=valid_entries[0].condition_id,
            source_relative_path="checkpoints/A/ds_a_0000.json",
            source_envelope_sha256=DUMMY_DIGEST,
            payload_sha256=DUMMY_DIGEST_2,
            target_relative_path="checkpoints/A/ds_a_0000.json",
            target_envelope_sha256=DUMMY_DIGEST_3,
            execution_origin="imported_exp450_mps",
        )
        with self.assertRaisesRegex(ValueError, "unique coordinates"):
            Exp451MigrationManifest(
                source_context_sha256=source_context,
                entries=tuple(dup_entries),
            )

        # Incomplete entries (4233 entries) must fail
        with self.assertRaisesRegex(ValueError, "do not match required"):
            Exp451MigrationManifest(
                source_context_sha256=source_context,
                entries=tuple(valid_entries[:-1]),
            )


if __name__ == "__main__":
    unittest.main()
