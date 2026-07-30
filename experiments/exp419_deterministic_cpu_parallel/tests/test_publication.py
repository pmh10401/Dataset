from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp417_causal_multihorizon_shadow.reporting import (
    TrustedCheckpointProvenance,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    BenchmarkSelectionProof,
)
from experiments.exp419_deterministic_cpu_parallel.tests.test_reporting import (
    EXACT_FILES,
    fixture_inputs,
)


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            render_exp419_payloads,
        )

        cls.parallel_run, cls.benchmark, cls.plan, cls.trusted = fixture_inputs()
        cls.payloads = dict(
            render_exp419_payloads(
                cls.parallel_run, cls.benchmark, cls.plan, cls.trusted
            )
        )

    def test_publish_requires_complete_fresh_checkpoint_and_exact_bundle(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.publication import (
            publish_exp419,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = SimpleNamespace(root=root / "canonical-checkpoint")
            checkpoint.root.mkdir()
            output = root / "official"
            callbacks: list[str] = []
            with patch.object(
                TrustedCheckpointProvenance,
                "from_store",
                side_effect=(self.trusted, self.trusted),
            ) as snapshots:
                hashes = publish_exp419(
                    self.payloads,
                    output,
                    checkpoints=checkpoint,
                    run=self.parallel_run,
                    benchmark=self.benchmark,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                    plan=self.plan,
                    pre_publish_verify=lambda: callbacks.append("fresh"),
                )

            self.assertEqual(snapshots.call_count, 2)
            self.assertEqual(callbacks, ["fresh"])
            self.assertEqual(tuple(sorted(path.name for path in output.iterdir())),
                             tuple(sorted(EXACT_FILES)))
            self.assertEqual(set(hashes), set(EXACT_FILES))

    def test_publish_requires_independent_expected_selection_proof(
        self,
    ) -> None:
        from experiments.exp419_deterministic_cpu_parallel.publication import (
            publish_exp419,
        )

        datasets = self.benchmark.benchmark_datasets
        alternate = BenchmarkSelectionProof(
            anchor_datasets=(datasets[0], datasets[2]),
            added_datasets=(datasets[1],),
            selected_datasets=datasets,
            proxy_headroom_gate=self.benchmark.selection_proof.proxy_headroom_gate,
            total_cost=self.benchmark.selection_proof.total_cost,
            shard_loads=self.benchmark.selection_proof.shard_loads,
            proxy_headrooms=self.benchmark.selection_proof.proxy_headrooms,
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "official"
            with self.assertRaisesRegex(ValueError, "selection proof"):
                publish_exp419(
                    self.payloads,
                    output,
                    checkpoints=SimpleNamespace(
                        root=Path(temporary) / "checkpoint"
                    ),
                    run=self.parallel_run,
                    benchmark=self.benchmark,
                    expected_selection_proof=alternate,
                    plan=self.plan,
                    pre_publish_verify=lambda: None,
                )
            self.assertFalse(output.exists())

    def test_publish_fails_closed_on_stale_checkpoint_extra_name_and_overlap(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.publication import (
            publish_exp419,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = SimpleNamespace(root=root / "canonical-checkpoint")
            checkpoint.root.mkdir()
            stale = TrustedCheckpointProvenance(
                "f" * 64,
                self.trusted.result_evidence_sha256,
                self.trusted.checkpoint_payload_sha256,
            )
            with patch.object(
                TrustedCheckpointProvenance,
                "from_store",
                side_effect=(self.trusted, stale),
            ), self.assertRaisesRegex(ValueError, "changed"):
                publish_exp419(
                    self.payloads,
                    root / "stale",
                    checkpoints=checkpoint,
                    run=self.parallel_run,
                    benchmark=self.benchmark,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                    plan=self.plan,
                    pre_publish_verify=lambda: None,
                )
            self.assertFalse((root / "stale").exists())

            extra = {**self.payloads, "extra.bin": b"extra\n"}
            with patch.object(
                TrustedCheckpointProvenance,
                "from_store",
                return_value=self.trusted,
            ), self.assertRaises(ValueError):
                publish_exp419(
                    extra,
                    root / "extra",
                    checkpoints=checkpoint,
                    run=self.parallel_run,
                    benchmark=self.benchmark,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                    plan=self.plan,
                    pre_publish_verify=lambda: None,
                )
            with patch.object(
                TrustedCheckpointProvenance,
                "from_store",
                return_value=self.trusted,
            ), self.assertRaises(ValueError):
                publish_exp419(
                    self.payloads,
                    checkpoint.root / "nested-output",
                    checkpoints=checkpoint,
                    run=self.parallel_run,
                    benchmark=self.benchmark,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                    plan=self.plan,
                    pre_publish_verify=lambda: None,
                )
