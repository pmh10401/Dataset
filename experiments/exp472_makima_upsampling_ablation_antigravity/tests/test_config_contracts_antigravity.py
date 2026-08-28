"""Behavior tests for the frozen Exp472 identity and artifact contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import unittest

import numpy as np

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    FROZEN_EXP450_CONFIG,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    EXP471_ROOT_ANTIGRAVITY,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import (
    BOOTSTRAP_DRAWS_ANTIGRAVITY,
    BOOTSTRAP_SEED_ANTIGRAVITY,
    CONDITION_E_ANTIGRAVITY,
    EXP471_REUSE_ROOT_ANTIGRAVITY,
    EXP472_EVALUATION_ROOT_ANTIGRAVITY,
    EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP472_REPORT_ROOT_ANTIGRAVITY,
    EXP472_ROOT_ANTIGRAVITY,
    EXP472_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP472_CONFIG,
    MLX_FALLBACK_REASON_ANTIGRAVITY,
    assess_exp472_backend_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    Exp472RunContextAntigravity,
    Exp472ScoreManifestAntigravity,
    Exp472ScoreShardAntigravity,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenPaperConfig


_DIGEST_A_ANTIGRAVITY = "a" * 64
_DIGEST_B_ANTIGRAVITY = "b" * 64
_DIGEST_C_ANTIGRAVITY = "c" * 64
_DIGEST_D_ANTIGRAVITY = "d" * 64


def backend_antigravity(
    *, mps_available: bool = True
) -> Exp472BackendDecisionAntigravity:
    return Exp472BackendDecisionAntigravity(
        pytorch_available=True,
        pytorch_version="2.13.0",
        scipy_available=True,
        scipy_version="1.17.1",
        mlx_available=False,
        mlx_version="unavailable",
        mlx_metal_available=False,
        mlx_metal_version="unavailable",
        mps_available=mps_available,
        tensor_backend="mps" if mps_available else "cpu",
        logical_cpu_count=12,
        cpu_workers=11,
        inner_numerical_threads=1,
        mlx_fallback_reason=MLX_FALLBACK_REASON_ANTIGRAVITY,
    )


class Exp472ConfigContractsTests(unittest.TestCase):
    def test_condition_e_is_the_only_new_condition(self) -> None:
        """Changing any factor must be rejected at the frozen E boundary."""
        self.assertEqual(
            (
                CONDITION_E_ANTIGRAVITY.condition_id,
                CONDITION_E_ANTIGRAVITY.downsampling,
                CONDITION_E_ANTIGRAVITY.upsampling,
                CONDITION_E_ANTIGRAVITY.image_mode,
            ),
            ("E", "polyphase", "makima", "area"),
        )
        with self.assertRaisesRegex(ValueError, "approved design"):
            replace(CONDITION_E_ANTIGRAVITY, upsampling="pchip")

    def test_population_roots_and_seed_are_frozen(self) -> None:
        """Population, root, and resampling changes must not enter Exp472."""
        self.assertEqual(FROZEN_EXP472_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP472_CONFIG.test_row_count, 195900)
        self.assertEqual(len(FROZEN_EXP472_CONFIG.smoke_datasets), 23)
        self.assertEqual(FROZEN_EXP472_CONFIG.bootstrap_seed, 47220260828)
        self.assertEqual(FROZEN_EXP472_CONFIG.bootstrap_draws, 20_000)
        self.assertTrue(
            all("antigravity" in part.name for part in FROZEN_EXP472_CONFIG.roots)
        )

    def test_bootstrap_interval_is_frozen_to_95_percentile(self) -> None:
        """A non-percentile method or non-95% level must fail paired analysis."""
        self.assertEqual(
            (
                getattr(FROZEN_EXP472_CONFIG, "bootstrap_interval_method", None),
                getattr(FROZEN_EXP472_CONFIG, "bootstrap_confidence_level", None),
            ),
            ("percentile", 0.95),
        )
        with self.assertRaisesRegex(ValueError, "paired-analysis policy"):
            replace(FROZEN_EXP472_CONFIG, bootstrap_confidence_level=0.90)

    def test_policy_tie_tolerance_and_database_digest_are_literal_frozen(self) -> None:
        """Policy, tie, or database drift must not borrow a mutable peer value."""
        self.assertEqual(FROZEN_EXP472_CONFIG.policy_id, "hard_plus_stable")
        self.assertEqual(FROZEN_EXP472_CONFIG.tie_tolerance, 1e-12)
        self.assertEqual(
            FROZEN_EXP472_CONFIG.database_sha256,
            "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
        )

    def test_exp450_population_paper_identity_and_exp471_root_are_reused(self) -> None:
        """A copied or reordered input identity must fail the reuse contract."""
        paper = FrozenPaperConfig()
        self.assertEqual(
            FROZEN_EXP472_CONFIG.smoke_datasets,
            FROZEN_EXP450_CONFIG.smoke_datasets,
        )
        self.assertEqual(
            (
                FROZEN_EXP472_CONFIG.database_path,
                FROZEN_EXP472_CONFIG.database_sha256,
            ),
            (Path(paper.database_path), paper.database_sha256),
        )
        self.assertEqual(EXP471_REUSE_ROOT_ANTIGRAVITY, EXP471_ROOT_ANTIGRAVITY)
        self.assertEqual(
            FROZEN_EXP472_CONFIG.roots,
            (
                EXP472_SMOKE_ROOT_ANTIGRAVITY,
                EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
                EXP472_EVALUATION_ROOT_ANTIGRAVITY,
                EXP472_REPORT_ROOT_ANTIGRAVITY,
            ),
        )
        self.assertTrue(
            all(root.parent == EXP472_ROOT_ANTIGRAVITY for root in FROZEN_EXP472_CONFIG.roots)
        )
        self.assertEqual(
            (BOOTSTRAP_SEED_ANTIGRAVITY, BOOTSTRAP_DRAWS_ANTIGRAVITY),
            (47220260828, 20_000),
        )

    def test_backend_record_freezes_availability_selection_and_worker_policy(self) -> None:
        """Backend or worker-policy drift must be rejected before score creation."""
        decision = backend_antigravity()
        record = decision.artifact_record()
        self.assertEqual(
            set(record),
            {
                "pytorch_available",
                "pytorch_version",
                "scipy_available",
                "scipy_version",
                "mlx_available",
                "mlx_version",
                "mlx_metal_available",
                "mlx_metal_version",
                "mps_available",
                "tensor_backend",
                "logical_cpu_count",
                "cpu_workers",
                "inner_numerical_threads",
                "mlx_fallback_reason",
            },
        )
        self.assertEqual(record["tensor_backend"], "mps")
        self.assertEqual(record["cpu_workers"], 11)
        self.assertEqual(record["inner_numerical_threads"], 1)
        self.assertEqual(record["mlx_fallback_reason"], MLX_FALLBACK_REASON_ANTIGRAVITY)
        with self.assertRaisesRegex(ValueError, "tensor backend"):
            replace(decision, tensor_backend="cpu")
        with self.assertRaisesRegex(ValueError, "worker policy"):
            replace(decision, cpu_workers=12)

    def test_backend_assessment_records_real_readiness_without_changing_policy(self) -> None:
        """An omitted runtime probe must not yield an incomplete readiness record."""
        decision = assess_exp472_backend_antigravity()
        self.assertTrue(decision.scipy_available)
        self.assertNotEqual(decision.scipy_version, "unavailable")
        self.assertEqual(decision.logical_cpu_count, os.cpu_count())
        self.assertEqual(decision.cpu_workers, max(1, os.cpu_count() - 1))
        self.assertEqual(
            decision.tensor_backend,
            "mps" if decision.mps_available else "cpu",
        )
        self.assertEqual(
            decision.mlx_fallback_reason,
            MLX_FALLBACK_REASON_ANTIGRAVITY,
        )

    def test_score_shard_requires_finite_float64_and_exact_threshold_alerts(self) -> None:
        """A dtype, finite-value, or threshold mutation must fail publication input."""
        scores = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
        identity = {"test_count": 3, "indices_sha256": _DIGEST_A_ANTIGRAVITY}
        diagnostic = {"upsampling_invoked": True, "overshoot": 0.0}
        shard = Exp472ScoreShardAntigravity(
            dataset="toy",
            test_identity=identity,
            scores=scores,
            alerts=frozenset({1, 2}),
            upsampled_rows=2,
            diagnostic=diagnostic,
        )
        scores[0] = 0.75
        identity["test_count"] = 0
        diagnostic["overshoot"] = 3.0
        np.testing.assert_array_equal(
            shard.scores, np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
        )
        self.assertEqual(shard.test_identity["test_count"], 3)
        self.assertEqual(shard.diagnostic["overshoot"], 0.0)
        self.assertFalse(shard.scores.flags.writeable)
        with self.assertRaises(ValueError):
            shard.scores[0] = 1.0
        with self.assertRaisesRegex(ValueError, "score shard differs"):
            replace(shard, scores=np.asarray([0.1, 0.5], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "score threshold differs"):
            replace(shard, alerts=frozenset({2}))
        with self.assertRaisesRegex(ValueError, "score shard differs"):
            replace(shard, scores=np.asarray([0.1, np.inf, 0.9], dtype=np.float64))

    def test_score_shard_allows_train_plus_test_upsampled_row_count(self) -> None:
        """A dataset-wide diagnostic count may exceed the TEST score length."""
        shard = Exp472ScoreShardAntigravity(
            dataset="toy",
            test_identity={"test_count": 2},
            scores=np.asarray([0.1, 0.9], dtype=np.float64),
            alerts=frozenset({1}),
            upsampled_rows=5,
            diagnostic={"scope": "train_and_test"},
        )
        self.assertEqual(shard.upsampled_rows, 5)

    def test_run_context_is_root_bound_deeply_immutable_and_digest_stable(self) -> None:
        """A root or code-closure mutation must change or fail the run identity."""
        closure = {"module_antigravity.py": _DIGEST_A_ANTIGRAVITY}
        context = Exp472RunContextAntigravity(
            mode="smoke",
            run_root=EXP472_SMOKE_ROOT_ANTIGRAVITY,
            database_sha256=_DIGEST_B_ANTIGRAVITY,
            catalog_sha256=_DIGEST_C_ANTIGRAVITY,
            code_closure=closure,
            backend=backend_antigravity(mps_available=False),
            config_sha256=_DIGEST_D_ANTIGRAVITY,
        )
        digest = context.sha256
        closure["module_antigravity.py"] = "0" * 64
        self.assertEqual(context.code_closure["module_antigravity.py"], _DIGEST_A_ANTIGRAVITY)
        self.assertEqual(context.sha256, digest)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(TypeError):
            context.code_closure["other_antigravity.py"] = _DIGEST_A_ANTIGRAVITY
        with self.assertRaisesRegex(ValueError, "run root"):
            replace(context, run_root=EXP472_OFFICIAL_ROOT_ANTIGRAVITY)
        with self.assertRaises(FrozenInstanceError):
            context.mode = "full"

    def test_terminal_manifest_requires_exact_ordered_mode_coverage(self) -> None:
        """Missing, duplicate, or unbound shards must not form a terminal manifest."""
        datasets = FROZEN_EXP472_CONFIG.smoke_datasets
        shards = {dataset: _DIGEST_A_ANTIGRAVITY for dataset in datasets}
        manifest = Exp472ScoreManifestAntigravity(
            mode="smoke",
            context_sha256=_DIGEST_B_ANTIGRAVITY,
            dataset_order=datasets,
            shard_sha256=shards,
            manifest_sha256=_DIGEST_C_ANTIGRAVITY,
        )
        self.assertEqual(manifest.dataset_order, datasets)
        self.assertEqual(tuple(manifest.shard_sha256), tuple(sorted(datasets)))
        with self.assertRaises(TypeError):
            manifest.shard_sha256[datasets[0]] = _DIGEST_D_ANTIGRAVITY
        with self.assertRaisesRegex(ValueError, "complete unique coverage"):
            replace(manifest, dataset_order=datasets[:-1])

    def test_smoke_manifest_rejects_reordered_exp450_inventory(self) -> None:
        """Twenty-three valid names in the wrong order must not form authority."""
        datasets = tuple(reversed(FROZEN_EXP450_CONFIG.smoke_datasets))
        shards = {dataset: _DIGEST_A_ANTIGRAVITY for dataset in datasets}
        with self.assertRaisesRegex(ValueError, "frozen Exp450 smoke inventory"):
            Exp472ScoreManifestAntigravity(
                mode="smoke",
                context_sha256=_DIGEST_B_ANTIGRAVITY,
                dataset_order=datasets,
                shard_sha256=shards,
                manifest_sha256=_DIGEST_C_ANTIGRAVITY,
            )


if __name__ == "__main__":
    unittest.main()
