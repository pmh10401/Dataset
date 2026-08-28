"""Lifecycle contracts for durable, label-free Exp472 score execution."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp472_makima_upsampling_ablation_antigravity.artifacts_antigravity import (
    append_failure_exp472_antigravity,
    append_progress_exp472_antigravity,
    publish_manifest_exp472_antigravity,
    publish_score_shard_exp472_antigravity,
    score_shard_record_exp472_antigravity,
    status_exp472_antigravity,
    verify_manifest_exp472_antigravity,
    verify_score_shard_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import (
    EXP472_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP472_CONFIG,
    MLX_FALLBACK_REASON_ANTIGRAVITY,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    Exp472RunContextAntigravity,
    Exp472ScoreShardAntigravity,
    canonical_json_bytes_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.runner_antigravity import (
    pending_datasets_exp472_antigravity,
    run_score_wave_exp472_antigravity,
)


_DIGEST_A_ANTIGRAVITY = "a" * 64
_DIGEST_B_ANTIGRAVITY = "b" * 64
_DIGEST_C_ANTIGRAVITY = "c" * 64
_DIGEST_D_ANTIGRAVITY = "d" * 64


def _backend_antigravity(*, mps: bool = False) -> Exp472BackendDecisionAntigravity:
    return Exp472BackendDecisionAntigravity(
        pytorch_available=True,
        pytorch_version="2.13.0",
        scipy_available=True,
        scipy_version="1.17.1",
        mlx_available=False,
        mlx_version="unavailable",
        mlx_metal_available=False,
        mlx_metal_version="unavailable",
        mps_available=mps,
        tensor_backend="mps" if mps else "cpu",
        logical_cpu_count=2,
        cpu_workers=1,
        inner_numerical_threads=1,
        mlx_fallback_reason=MLX_FALLBACK_REASON_ANTIGRAVITY,
    )


def _context_antigravity(*, mps: bool = False) -> Exp472RunContextAntigravity:
    return Exp472RunContextAntigravity(
        mode="smoke",
        run_root=EXP472_SMOKE_ROOT_ANTIGRAVITY,
        database_sha256=_DIGEST_A_ANTIGRAVITY,
        catalog_sha256=_DIGEST_B_ANTIGRAVITY,
        code_closure={"scoring_antigravity.py": _DIGEST_C_ANTIGRAVITY},
        backend=_backend_antigravity(mps=mps),
        config_sha256=_DIGEST_D_ANTIGRAVITY,
    )


def _shard_antigravity(dataset: str = "A", score: float = 0.9) -> Exp472ScoreShardAntigravity:
    return Exp472ScoreShardAntigravity(
        dataset=dataset,
        test_identity={"dataset": dataset, "test_count": 1, "indices_sha256": _DIGEST_A_ANTIGRAVITY},
        scores=np.asarray([score], dtype=np.float64),
        alerts=frozenset({0}) if score >= 0.5 else frozenset(),
        upsampled_rows=1,
        diagnostic={"shape": "fixture"},
    )


class Exp472ArtifactsRunnerTests(unittest.TestCase):
    """Each test uses a private temporary root, never an Exp472 runtime root."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp472_task4_antigravity_")
        self.root = Path(self.temporary.name)
        self.context = _context_antigravity()
        self.catalog = ({"dataset": "A"}, {"dataset": "B"}, {"dataset": "C"}, {"dataset": "D"})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_create_only_shard_is_idempotent_for_same_bytes_and_rejects_drift(self) -> None:
        """Replacing a completed coordinate with different scores must fail."""
        shard = _shard_antigravity()
        first = publish_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0], shard)
        second = publish_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0], shard)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with self.assertRaisesRegex(ValueError, "different bytes"):
            publish_score_shard_exp472_antigravity(
                self.root, self.context, self.catalog[0], replace(shard, scores=np.asarray([0.1], dtype=np.float64), alerts=frozenset())
            )

    def test_shard_verifier_reopens_and_rejects_identity_or_digest_drift(self) -> None:
        """A changed catalog identity or self-digest must invalidate a shard."""
        path = publish_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0], _shard_antigravity())
        verified = verify_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0])
        self.assertEqual(verified["dataset"], "A")
        record = json.loads(path.read_text())
        record["dataset"] = "tampered"
        path.write_bytes(canonical_json_bytes_exp472_antigravity(record))
        with self.assertRaisesRegex(ValueError, "identity|digest"):
            verify_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0])

    def test_score_record_binds_context_closure_condition_and_float64_payload(self) -> None:
        """Omitting a provenance binding or changing a score digest must fail verification."""
        record = score_shard_record_exp472_antigravity(self.context, _shard_antigravity())
        self.assertEqual(record["schema"], "exp472-score-shard-antigravity.v1")
        self.assertEqual(record["context_sha256"], self.context.sha256)
        self.assertEqual(record["code_closure"], {"scoring_antigravity.py": _DIGEST_C_ANTIGRAVITY})
        self.assertEqual(record["condition"], {"id": "E", "downsampling": "polyphase", "upsampling": "makima", "image_mode": "area"})
        self.assertEqual(record["score_dtype"], "float64")
        bare = {key: value for key, value in record.items() if key != "shard_sha256"}
        self.assertEqual(record["shard_sha256"], sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest())

    def test_manifest_requires_verified_complete_ordered_smoke_inventory_and_no_failures(self) -> None:
        """A partial or failed lifecycle must not create a terminal manifest."""
        smoke_catalog = tuple(
            {"dataset": dataset} for dataset in FROZEN_EXP472_CONFIG.smoke_datasets
        )
        for entry in smoke_catalog:
            publish_score_shard_exp472_antigravity(self.root, self.context, entry, _shard_antigravity(entry["dataset"]))
        with self.assertRaisesRegex(ValueError, "zero failures"):
            publish_manifest_exp472_antigravity(self.root, self.context, smoke_catalog, failures={"A"})
        manifest = publish_manifest_exp472_antigravity(self.root, self.context, smoke_catalog)
        verified = verify_manifest_exp472_antigravity(self.root, self.context, smoke_catalog)
        self.assertEqual(manifest.read_bytes(), canonical_json_bytes_exp472_antigravity(verified))
        self.assertEqual(verified["dataset_order"], list(FROZEN_EXP472_CONFIG.smoke_datasets))

    def test_pending_coordinates_are_catalog_ordered_and_include_only_missing_or_failed(self) -> None:
        """A completed coordinate must not rerun; failed B must be retried before missing D."""
        pending = pending_datasets_exp472_antigravity(self.catalog, completed={"A", "C"}, failures={"B"})
        self.assertEqual(pending, ("B", "D"))

    def test_progress_and_failure_records_are_append_only_and_status_has_exact_fields(self) -> None:
        """Replacing append-only JSONL status history would erase crash evidence."""
        append_progress_exp472_antigravity(self.root, completed=1, failed=0, remaining=3, elapsed_seconds=10.0, dataset="A", wave=1, status="completed")
        append_failure_exp472_antigravity(self.root, self.context, dataset="B", wave=1, error=RuntimeError("fixture failure"))
        append_progress_exp472_antigravity(self.root, completed=1, failed=1, remaining=3, elapsed_seconds=20.0, dataset="B", wave=1, status="failed")
        status = status_exp472_antigravity(self.root, total=4, now_monotonic=40.0)
        self.assertEqual(set(status), {"schema", "completed", "failed", "remaining", "total", "rate_per_second", "eta_seconds", "manifest_complete", "evaluation_complete", "report_complete", "latest_progress"})
        self.assertEqual((status["completed"], status["failed"], status["remaining"], status["total"]), (1, 1, 3, 4))
        self.assertEqual(len((self.root / "progress_antigravity.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len((self.root / "failures_antigravity.jsonl").read_text().splitlines()), 1)

    def test_runner_uses_exactly_two_waves_and_continues_after_failures(self) -> None:
        """A first-wave failure must be recorded while later datasets and one retry continue."""
        attempts: dict[str, int] = {}

        def score_one(dataset: str) -> Exp472ScoreShardAntigravity:
            attempts[dataset] = attempts.get(dataset, 0) + 1
            if dataset == "B" and attempts[dataset] == 1:
                raise RuntimeError("transient")
            return _shard_antigravity(dataset)

        result = run_score_wave_exp472_antigravity(self.root, _context_antigravity(mps=True), self.catalog, score_one=score_one)
        self.assertEqual(result["waves"], 2)
        self.assertEqual(attempts, {"A": 1, "B": 2, "C": 1, "D": 1})
        self.assertEqual(result["failures"], ())
        self.assertFalse((self.root / "exp472_score_manifest_antigravity.json").exists())

    def test_runner_never_retries_after_wave_two_or_publishes_terminal_manifest_with_failure(self) -> None:
        """A permanent error must stop at its bounded retry and leave no terminal authority."""
        def always_fails(dataset: str) -> Exp472ScoreShardAntigravity:
            if dataset == "B":
                raise RuntimeError("permanent")
            return _shard_antigravity(dataset)

        result = run_score_wave_exp472_antigravity(self.root, _context_antigravity(mps=True), self.catalog, score_one=always_fails)
        self.assertEqual(result["waves"], 2)
        self.assertEqual(result["failures"], ("B",))
        self.assertFalse((self.root / "exp472_score_manifest_antigravity.json").exists())


if __name__ == "__main__":
    unittest.main()
