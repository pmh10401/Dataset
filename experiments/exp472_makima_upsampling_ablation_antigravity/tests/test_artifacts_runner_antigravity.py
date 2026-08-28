"""Lifecycle contracts for durable, label-free Exp472 score execution."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp472_makima_upsampling_ablation_antigravity.artifacts_antigravity import (
    Exp472CatalogEntryAntigravity,
    append_failure_exp472_antigravity,
    append_progress_exp472_antigravity,
    catalog_sha256_exp472_antigravity,
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
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import (
    Exp472AlignmentTraceAntigravity,
    align_rows_exp472_antigravity,
)


_DIGEST_A_ANTIGRAVITY = "a" * 64
_DIGEST_B_ANTIGRAVITY = "b" * 64
_DIGEST_C_ANTIGRAVITY = "c" * 64
_DIGEST_D_ANTIGRAVITY = "d" * 64


def _backend_antigravity(*, mps: bool = False, logical_cpu_count: int = 2) -> Exp472BackendDecisionAntigravity:
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
        logical_cpu_count=logical_cpu_count,
        cpu_workers=max(1, logical_cpu_count - 1),
        inner_numerical_threads=1,
        mlx_fallback_reason=MLX_FALLBACK_REASON_ANTIGRAVITY,
    )


def _context_antigravity(
    catalog: tuple[Exp472CatalogEntryAntigravity, ...] | None = None,
    *, mps: bool = False, logical_cpu_count: int = 2,
) -> Exp472RunContextAntigravity:
    entries = _catalog_antigravity() if catalog is None else catalog
    return Exp472RunContextAntigravity(
        mode="smoke",
        run_root=EXP472_SMOKE_ROOT_ANTIGRAVITY,
        database_sha256=_DIGEST_A_ANTIGRAVITY,
        catalog_sha256=catalog_sha256_exp472_antigravity(entries),
        code_closure={"scoring_antigravity.py": _DIGEST_C_ANTIGRAVITY},
        backend=_backend_antigravity(mps=mps, logical_cpu_count=logical_cpu_count),
        config_sha256=_DIGEST_D_ANTIGRAVITY,
    )


def _shard_antigravity(dataset: str = "A", score: float = 0.9) -> Exp472ScoreShardAntigravity:
    return Exp472ScoreShardAntigravity(
        dataset=dataset,
        test_identity={"dataset": dataset, "test_count": 1, "indices_sha256": _DIGEST_A_ANTIGRAVITY},
        scores=np.asarray([score], dtype=np.float64),
        alerts=frozenset({0}) if score >= 0.5 else frozenset(),
        upsampled_rows=1,
        diagnostic=_diagnostic_antigravity(dataset, "fixture"),
    )


def _catalog_antigravity(names: tuple[str, ...] = ("A", "B", "C", "D")) -> tuple[Exp472CatalogEntryAntigravity, ...]:
    return tuple(
        Exp472CatalogEntryAntigravity(
            dataset=dataset,
            test_identity={"dataset": dataset, "test_count": 1, "indices_sha256": _DIGEST_A_ANTIGRAVITY},
        )
        for dataset in names
    )


def _diagnostic_antigravity(dataset: str, shape: str) -> dict[str, object]:
    del shape
    input_evidence = {
        "dtype": "float32", "shape": [1], "row_boundaries": [2],
        "row_bytes_sha256": [_DIGEST_A_ANTIGRAVITY],
    }
    output_evidence = {
        "dtype": "float32", "shape": [1, 3], "row_boundaries": [3],
        "row_bytes_sha256": [_DIGEST_B_ANTIGRAVITY],
    }
    invocation = {
        "ordinal": 0, "phase_tag": "hard_sources", "source_tag": "TRAIN",
        "target_len": 3, "row_count": 1, "input_evidence": input_evidence,
        "input_sha256": sha256(canonical_json_bytes_exp472_antigravity(input_evidence)).hexdigest(),
        "output_evidence": output_evidence,
        "output_sha256": sha256(canonical_json_bytes_exp472_antigravity(output_evidence)).hexdigest(),
        "singleton_rows": 0, "equal_rows": 0, "polyphase_downsample_rows": 0,
        "makima_rows": 1, "makima_row_indices": [0], "overshoot": 0.0,
        "undershoot": 0.0, "source_extrema": 0, "output_extrema": 0,
        "added_extrema": 0,
    }
    invocation["invocation_sha256"] = sha256(
        canonical_json_bytes_exp472_antigravity(invocation)
    ).hexdigest()
    aggregate = {
        "invocation_count": 1, "row_invocation_count": 1,
        "makima_row_invocation_count": 1, "upsampled_rows": 1,
    }
    bare = {
        "schema": "exp472-shape-diagnostic-antigravity.v3", "dataset": dataset,
        "trace_state": "sealed", "alignment_status": "completed", **aggregate,
        "invocations": [invocation],
        "invocation_order_sha256": sha256(canonical_json_bytes_exp472_antigravity([("hard_sources", "TRAIN")])).hexdigest(),
        "invocation_aggregate_sha256": sha256(canonical_json_bytes_exp472_antigravity(aggregate)).hexdigest(),
    }
    return {**bare, "diagnostic_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}


def _cpu_score_succeeds_antigravity(dataset: str) -> Exp472ScoreShardAntigravity:
    shard = _shard_antigravity(dataset)
    return replace(shard, diagnostic=_diagnostic_antigravity(dataset, f"cpu-{os.environ.get('OMP_NUM_THREADS')}"))


def _cpu_score_raises_antigravity(dataset: str) -> Exp472ScoreShardAntigravity:
    raise RuntimeError(f"cpu fixture failure for {dataset}")


class Exp472ArtifactsRunnerTests(unittest.TestCase):
    """Each test uses a private temporary root, never an Exp472 runtime root."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp472_task4_antigravity_")
        self.root = Path(self.temporary.name)
        self.catalog = _catalog_antigravity()
        self.context = _context_antigravity(self.catalog)

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

    def test_shard_roundtrip_preserves_nested_json_identity_and_rejects_tampering(self) -> None:
        """JSON lists reconstructed as immutable tuples are canonically equivalent, not unequal."""
        identity = {
            "dataset": "A",
            "test_count": 1,
            "indices": [0],
            "nested": {"blobs": ["a" * 64, "b" * 64]},
        }
        entry = Exp472CatalogEntryAntigravity("A", identity)
        context = _context_antigravity((entry,))
        shard = Exp472ScoreShardAntigravity(
            dataset="A",
            test_identity={
                "dataset": "A",
                "test_count": 1,
                "indices": (0,),
                "nested": {"blobs": ("a" * 64, "b" * 64)},
            },
            scores=np.asarray([0.9], dtype=np.float64),
            alerts=frozenset({0}),
            upsampled_rows=1,
            diagnostic=_diagnostic_antigravity("A", "nested"),
        )
        path = publish_score_shard_exp472_antigravity(self.root, context, entry, shard)
        self.assertEqual(verify_score_shard_exp472_antigravity(self.root, context, entry)["test_identity"], identity)
        changed = json.loads(path.read_text())
        changed["test_identity"]["nested"]["blobs"][1] = "c" * 64
        bare = {key: value for key, value in changed.items() if key != "shard_sha256"}
        changed["shard_sha256"] = sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()
        path.write_bytes(canonical_json_bytes_exp472_antigravity(changed))
        with self.assertRaisesRegex(ValueError, "TEST identity"):
            verify_score_shard_exp472_antigravity(self.root, context, entry)

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
        smoke_catalog = _catalog_antigravity(FROZEN_EXP472_CONFIG.smoke_datasets)
        smoke_context = _context_antigravity(smoke_catalog)
        for entry in smoke_catalog:
            publish_score_shard_exp472_antigravity(self.root, smoke_context, entry, _shard_antigravity(entry.dataset))
        with self.assertRaisesRegex(ValueError, "zero failures"):
            publish_manifest_exp472_antigravity(self.root, smoke_context, smoke_catalog, failures={"A"})
        manifest = publish_manifest_exp472_antigravity(self.root, smoke_context, smoke_catalog)
        verified = verify_manifest_exp472_antigravity(self.root, smoke_context, smoke_catalog)
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

        result = run_score_wave_exp472_antigravity(self.root, _context_antigravity(self.catalog, mps=True), self.catalog, score_one=score_one)
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

        result = run_score_wave_exp472_antigravity(self.root, _context_antigravity(self.catalog, mps=True), self.catalog, score_one=always_fails)
        self.assertEqual(result["waves"], 2)
        self.assertEqual(result["failures"], ("B",))
        self.assertFalse((self.root / "exp472_score_manifest_antigravity.json").exists())

    def test_catalog_boundary_requires_identity_and_exact_ordered_digest(self) -> None:
        """Dataset-only, reordered, or TEST-identity drifted catalogs must not resume work."""
        with self.assertRaisesRegex(ValueError, "test_identity"):
            run_score_wave_exp472_antigravity(
                self.root, self.context, ({"dataset": "A"},), score_one=_cpu_score_succeeds_antigravity
            )
        with self.assertRaisesRegex(ValueError, "catalog digest"):
            run_score_wave_exp472_antigravity(
                self.root, self.context, tuple(reversed(self.catalog)), score_one=_cpu_score_succeeds_antigravity
            )
        drifted = (replace(self.catalog[0], test_identity={"dataset": "A", "test_count": 2, "indices_sha256": _DIGEST_A_ANTIGRAVITY}), *self.catalog[1:])
        with self.assertRaisesRegex(ValueError, "catalog digest"):
            publish_manifest_exp472_antigravity(self.root, self.context, drifted)

    def test_verifier_rejects_all_payload_and_test_identity_drift_after_recomputed_digest(self) -> None:
        """A valid-looking rewritten envelope must still fail every frozen shard invariant."""
        path = publish_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0], _shard_antigravity())
        mutations = (
            ("schema", "other"), ("score_dtype", "float32"), ("scores", [0.1]),
            ("alerts", []), ("diagnostic", {"shape": "drift"}),
            ("test_identity", {"dataset": "A", "test_count": 2, "indices_sha256": _DIGEST_A_ANTIGRAVITY}),
        )
        original = json.loads(path.read_text())
        for key, value in mutations:
            with self.subTest(field=key):
                changed = dict(original)
                changed[key] = value
                bare = {name: item for name, item in changed.items() if name != "shard_sha256"}
                changed["shard_sha256"] = sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()
                path.write_bytes(canonical_json_bytes_exp472_antigravity(changed))
                with self.assertRaises(ValueError):
                    verify_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0])
        path.write_bytes(canonical_json_bytes_exp472_antigravity(original))

    def test_v3_trace_validator_rejects_nested_recomputed_tampering(self) -> None:
        """A recomputed outer digest cannot make a malformed trace authoritative."""
        rows = (np.asarray([0.0, 1.0], dtype=np.float32),)
        trace = Exp472AlignmentTraceAntigravity("A")
        output = align_rows_exp472_antigravity(rows, 3)
        trace.observe(rows, 3, output, "hard_sources", "TRAIN")
        trace.observe(rows, 3, output, "hard_sources", "TEST")
        shard = replace(
            _shard_antigravity(), upsampled_rows=2, diagnostic=trace.seal()
        )
        path = publish_score_shard_exp472_antigravity(
            self.root, self.context, self.catalog[0], shard
        )
        original = json.loads(path.read_text())
        mutations = (
            ("ordinal", 2),
            ("source_tag", "TRAIN"),
            ("upsampled_rows", 0),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(original))
                if field == "upsampled_rows":
                    changed["upsampled_rows"] = value
                else:
                    changed["diagnostic"]["invocations"][1][field] = value
                    invocation = changed["diagnostic"]["invocations"][1]
                    invocation["invocation_sha256"] = sha256(canonical_json_bytes_exp472_antigravity({key: item for key, item in invocation.items() if key != "invocation_sha256"})).hexdigest()
                    diagnostic = changed["diagnostic"]
                    diagnostic["diagnostic_sha256"] = sha256(canonical_json_bytes_exp472_antigravity({key: item for key, item in diagnostic.items() if key != "diagnostic_sha256"})).hexdigest()
                bare = {key: item for key, item in changed.items() if key != "shard_sha256"}
                changed["shard_sha256"] = sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()
                path.write_bytes(canonical_json_bytes_exp472_antigravity(changed))
                with self.assertRaisesRegex(ValueError, "diagnostic|upsampled"):
                    verify_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0])
        path.write_bytes(canonical_json_bytes_exp472_antigravity(original))

    def test_torn_existing_shard_is_rejected_without_overwrite(self) -> None:
        """A crash-truncated create-only path must preserve evidence instead of overwriting it."""
        shard = _shard_antigravity()
        path = publish_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0], shard)
        path.write_bytes(b'{"schema":')
        with self.assertRaisesRegex(ValueError, "JSON|noncanonical"):
            publish_score_shard_exp472_antigravity(self.root, self.context, self.catalog[0], shard)
        self.assertEqual(path.read_bytes(), b'{"schema":')

    def test_second_run_skips_verified_shards_and_smoke_run_publishes_reopenable_manifest(self) -> None:
        """Resume must score only absent shards, then produce one idempotent 23-entry authority."""
        smoke_catalog = _catalog_antigravity(FROZEN_EXP472_CONFIG.smoke_datasets)
        context = _context_antigravity(smoke_catalog, mps=True)
        first = run_score_wave_exp472_antigravity(self.root, context, smoke_catalog, score_one=_cpu_score_succeeds_antigravity)
        self.assertTrue(first["manifest_published"])
        manifest = self.root / "exp472_score_manifest_antigravity.json"
        original = manifest.read_bytes()
        second = run_score_wave_exp472_antigravity(self.root, context, smoke_catalog, score_one=_cpu_score_raises_antigravity)
        self.assertTrue(second["manifest_published"])
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(verify_manifest_exp472_antigravity(self.root, context, smoke_catalog)["dataset_order"], list(FROZEN_EXP472_CONFIG.smoke_datasets))
        manifest.write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "different bytes"):
            publish_manifest_exp472_antigravity(self.root, context, smoke_catalog)

    def test_cpu_scheduler_requires_pickle_safe_callback_and_records_cpu_errors(self) -> None:
        """Nested callbacks fail per coordinate; picklable workers stay bounded and serial internally."""
        context = _context_antigravity(self.catalog, logical_cpu_count=3)
        def unpicklable(dataset: str) -> Exp472ScoreShardAntigravity:
            return _shard_antigravity(dataset)
        rejected = run_score_wave_exp472_antigravity(self.root, context, self.catalog, score_one=unpicklable)
        self.assertEqual(rejected["waves"], 2)
        self.assertEqual(rejected["failures"], ("A", "B", "C", "D"))
        self.assertEqual(rejected["workers"], context.backend.cpu_workers)
        self.assertIn("pickle-safe", (self.root / "failures_antigravity.jsonl").read_text())
        with tempfile.TemporaryDirectory(prefix="exp472_cpu_antigravity_") as other:
            cpu_root = Path(other)
            succeeded = run_score_wave_exp472_antigravity(cpu_root, context, self.catalog, score_one=_cpu_score_succeeds_antigravity)
            self.assertEqual(succeeded["waves"], 2)
            self.assertEqual(succeeded["failures"], ())
            self.assertEqual(succeeded["workers"], context.backend.cpu_workers)
            for entry in self.catalog:
                self.assertEqual(verify_score_shard_exp472_antigravity(cpu_root, context, entry)["diagnostic"]["dataset"], entry.dataset)
        with tempfile.TemporaryDirectory(prefix="exp472_cpu_error_antigravity_") as other:
            failed = run_score_wave_exp472_antigravity(Path(other), context, self.catalog, score_one=_cpu_score_raises_antigravity)
            self.assertEqual(failed["failures"], ("A", "B", "C", "D"))
            self.assertIn("cpu fixture failure", (Path(other) / "failures_antigravity.jsonl").read_text())

    def test_status_uses_quiet_time_and_rejects_symlink_manifest(self) -> None:
        """ETA must age through quiet time and a manifest symlink must never report complete."""
        empty = status_exp472_antigravity(self.root, total=4, now_monotonic=50.0)
        self.assertEqual((empty["rate_per_second"], empty["eta_seconds"]), (0.0, None))
        append_progress_exp472_antigravity(self.root, completed=1, failed=0, remaining=3, elapsed_seconds=10.0, observed_monotonic=100.0, dataset="A", wave=1, status="completed")
        status = status_exp472_antigravity(self.root, total=4, now_monotonic=110.0)
        self.assertEqual(status["rate_per_second"], 0.05)
        self.assertEqual(status["eta_seconds"], 60.0)
        target = self.root / "not_manifest_antigravity.json"
        target.write_text("{}\n")
        (self.root / "exp472_score_manifest_antigravity.json").symlink_to(target)
        self.assertFalse(status_exp472_antigravity(self.root, total=4, now_monotonic=110.0)["manifest_complete"])

    def test_completed_coordinates_win_over_stale_failure_set(self) -> None:
        """A verified completion must not be requeued merely because old failure evidence exists."""
        self.assertEqual(
            pending_datasets_exp472_antigravity(self.catalog, completed={"A", "C"}, failures={"A", "B"}),
            ("B", "D"),
        )


if __name__ == "__main__":
    unittest.main()
