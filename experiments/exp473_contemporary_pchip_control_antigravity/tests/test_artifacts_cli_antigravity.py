"""Durable Exp473 shard, manifest, status, and verification contracts."""

from __future__ import annotations

from functools import partial
from hashlib import sha256
import io
import json
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp473_contemporary_pchip_control_antigravity.artifacts_antigravity import (
    publish_manifest_exp473_antigravity,
    publish_run_context_exp473_antigravity,
    publish_score_shard_exp473_antigravity,
    verify_manifest_exp473_antigravity,
    verify_run_context_exp473_antigravity,
    verify_score_shard_exp473_antigravity,
)
from experiments.exp473_contemporary_pchip_control_antigravity.cli_antigravity import (
    main_antigravity,
)
from experiments.exp473_contemporary_pchip_control_antigravity.contracts_antigravity import (
    Exp473ScoreShardAntigravity,
)
import experiments.exp473_contemporary_pchip_control_antigravity.runner_antigravity as runner_mod
from experiments.exp473_contemporary_pchip_control_antigravity.runner_antigravity import (
    run_score_wave_exp473_antigravity,
)


def _shard(dataset: str) -> Exp473ScoreShardAntigravity:
    scores = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
    return Exp473ScoreShardAntigravity(
        dataset=dataset,
        test_identity={"count": 3, "indices": [0, 1, 2]},
        scores=scores,
        alerts=frozenset({1, 2}),
    )


def _runner_score_antigravity(dataset: str) -> Exp473ScoreShardAntigravity:
    return _shard(dataset)


def _runner_score_with_payload_antigravity(
    payload: bytes, dataset: str
) -> Exp473ScoreShardAntigravity:
    if not payload.startswith(b"EXP473_SUBMIT_"):
        raise ValueError("Exp473 submit marker differs")
    return _shard(dataset)


class _FakeRuntimeAntigravity:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.run_calls = 0
        self.verify_calls = 0

    def run_exp473_antigravity(self) -> dict[str, object]:
        self.run_calls += 1
        return {"completed": 23 if self.mode == "smoke" else 1117, "failed": 0}

    def verify_exp473_antigravity(self) -> dict[str, object]:
        self.verify_calls += 1
        return {"manifest_sha256": "c" * 64}


class Exp473ArtifactsCliTestsAntigravity(unittest.TestCase):
    def test_run_context_is_create_only_and_self_digesting(self) -> None:
        with TemporaryDirectory(prefix="exp473_context_antigravity_") as directory:
            root = Path(directory)
            bare = {
                "schema": "exp473-run-context-antigravity.v1",
                "experiment": "Exp473",
                "mode": "smoke",
                "backend": {"cpu_workers": 11},
            }
            record = publish_run_context_exp473_antigravity(root, bare)
            self.assertEqual(
                verify_run_context_exp473_antigravity(root), record
            )
            self.assertEqual(len(record["context_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                publish_run_context_exp473_antigravity(root, bare)

    def test_score_shard_requires_float64_finite_and_exact_threshold_alerts(self) -> None:
        with self.assertRaises(ValueError):
            Exp473ScoreShardAntigravity(
                "fixture",
                {"count": 1, "indices": [0]},
                np.asarray([0.75], dtype=np.float32),
                frozenset({0}),
            )
        with self.assertRaises(ValueError):
            Exp473ScoreShardAntigravity(
                "fixture",
                {"count": 1, "indices": [0]},
                np.asarray([0.75], dtype=np.float64),
                frozenset(),
            )

    def test_create_only_shards_and_terminal_manifest_reopen_read_only(self) -> None:
        with TemporaryDirectory(prefix="exp473_artifacts_antigravity_") as directory:
            root = Path(directory)
            catalog = ("alpha", "beta")
            context_sha256 = "a" * 64
            for dataset in catalog:
                publish_score_shard_exp473_antigravity(
                    root, context_sha256, _shard(dataset)
                )
            manifest = publish_manifest_exp473_antigravity(
                root, context_sha256, catalog
            )
            self.assertEqual(manifest["completed"], 2)
            self.assertEqual(
                verify_manifest_exp473_antigravity(root, context_sha256, catalog),
                manifest,
            )
            for dataset in catalog:
                reopened = verify_score_shard_exp473_antigravity(
                    root, context_sha256, dataset
                )
                self.assertEqual(reopened.dataset, dataset)
                np.testing.assert_array_equal(reopened.scores, _shard(dataset).scores)
            with self.assertRaises(FileExistsError):
                publish_score_shard_exp473_antigravity(
                    root, context_sha256, _shard("alpha")
                )

    def test_status_is_read_only_and_closed_to_root_overrides(self) -> None:
        stream = io.StringIO()
        self.assertEqual(main_antigravity(("status", "--mode", "smoke"), stdout=stream), 0)
        record = json.loads(stream.getvalue())
        self.assertEqual(record["experiment"], "Exp473")
        self.assertEqual(record["mode"], "smoke")
        self.assertTrue(record["read_only"])
        with self.assertRaises(ValueError):
            main_antigravity(("status", "--root", "/tmp/other"), stdout=io.StringIO())

    def test_two_wave_runner_publishes_only_a_complete_manifest(self) -> None:
        with TemporaryDirectory(prefix="exp473_runner_antigravity_") as directory:
            root = Path(directory)
            result = run_score_wave_exp473_antigravity(
                root,
                "b" * 64,
                ("alpha", "beta"),
                score_one=_runner_score_antigravity,
                cpu_workers=1,
                tensor_backend="cpu",
            )
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(result["manifest_published"])
            verify_manifest_exp473_antigravity(
                root, "b" * 64, ("alpha", "beta")
            )
            reopened = run_score_wave_exp473_antigravity(
                root,
                "b" * 64,
                ("alpha", "beta"),
                score_one=_runner_score_antigravity,
                cpu_workers=1,
                tensor_backend="cpu",
            )
            self.assertEqual(reopened["completed"], 2)
            self.assertEqual(reopened["failed"], 0)
            self.assertTrue(reopened["manifest_published"])

    def test_mps_wave_persists_first_shard_before_second_dataset_starts(self) -> None:
        with TemporaryDirectory(prefix="exp473_stream_antigravity_") as directory:
            root = Path(directory)
            first_shard = (
                root
                / "score_shards_antigravity"
                / f"{sha256(b'alpha').hexdigest()}_antigravity"
                / "score_shard_antigravity.json"
            )
            seen: list[bool] = []

            def score_one(dataset: str) -> Exp473ScoreShardAntigravity:
                if dataset == "beta":
                    seen.append(first_shard.is_file())
                return _shard(dataset)

            result = run_score_wave_exp473_antigravity(
                root,
                "c" * 64,
                ("alpha", "beta"),
                score_one=score_one,
                cpu_workers=1,
                tensor_backend="mps",
            )
            self.assertEqual(seen, [True])
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(first_shard.is_file())

    def test_parent_compares_test_identity_without_pickling_it_into_the_callback(
        self,
    ) -> None:
        marker = "EXP473_ID_PARENT_" + os.urandom(16).hex()
        with TemporaryDirectory(prefix="exp473_identity_antigravity_") as directory:
            result = run_score_wave_exp473_antigravity(
                Path(directory),
                "e" * 64,
                ("alpha", "beta"),
                score_one=_runner_score_antigravity,
                cpu_workers=1,
                tensor_backend="mps",
                expected_identities={
                    "alpha": {"count": 3, "indices": [0, 1, 2]},
                    "beta": {"count": 3, "indices": [0, 1, 2], "marker": marker},
                },
            )
        payload = pickle.dumps(_runner_score_antigravity)
        self.assertNotIn(marker.encode("ascii"), payload)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["manifest_published"])

    def test_cpu_submit_packet_excludes_captured_callback_payload(self) -> None:
        marker = b"EXP473_SUBMIT_" + os.urandom(32 * 1024)
        packets: list[bytes] = []
        score_one = partial(_runner_score_with_payload_antigravity, marker)

        class _ImmediateFuture:
            def __init__(self, payload: object) -> None:
                self._payload = payload

            def result(self) -> object:
                if isinstance(self._payload, BaseException):
                    raise self._payload
                return self._payload

        class _RecordingPool:
            def __init__(self, *args: object, **kwargs: object) -> None:
                initializer = kwargs.get("initializer")
                initargs = kwargs.get("initargs", ())
                if callable(initializer):
                    initializer(*initargs)

            def __enter__(self) -> "_RecordingPool":
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

            def submit(self, fn: object, *args: object) -> _ImmediateFuture:
                packets.append(pickle.dumps((fn, args)))
                try:
                    return _ImmediateFuture(fn(*args))
                except BaseException as error:
                    return _ImmediateFuture(error)

        with (
            TemporaryDirectory(prefix="exp473_submit_antigravity_") as directory,
            patch.object(runner_mod, "ProcessPoolExecutor", _RecordingPool),
            patch.object(runner_mod, "wait", lambda fs, return_when=None: (set(fs), set())),
        ):
            result = run_score_wave_exp473_antigravity(
                Path(directory),
                "d" * 64,
                ("alpha", "beta"),
                score_one=score_one,
                cpu_workers=1,
                tensor_backend="cpu",
            )
        self.assertEqual(result["completed"], 2)
        self.assertTrue(packets)
        for packet in packets:
            self.assertNotIn(marker, packet)
            self.assertLess(len(packet), len(marker))

    def test_closed_smoke_run_and_verify_dispatch_only_typed_runtime(self) -> None:
        smoke = _FakeRuntimeAntigravity("smoke")
        output = io.StringIO()
        self.assertEqual(main_antigravity(("smoke",), stdout=output, runtime=smoke), 0)
        self.assertEqual(smoke.run_calls, 1)
        self.assertEqual(json.loads(output.getvalue())["command"], "smoke")
        verified = io.StringIO()
        self.assertEqual(
            main_antigravity(
                ("verify", "--mode", "smoke"),
                stdout=verified,
                runtime=smoke,
            ),
            0,
        )
        self.assertEqual(smoke.verify_calls, 1)
        full = _FakeRuntimeAntigravity("full")
        self.assertEqual(
            main_antigravity(("run",), stdout=io.StringIO(), runtime=full), 0
        )
        with self.assertRaises(ValueError):
            main_antigravity(("run",), stdout=io.StringIO(), runtime=smoke)


if __name__ == "__main__":
    unittest.main()
