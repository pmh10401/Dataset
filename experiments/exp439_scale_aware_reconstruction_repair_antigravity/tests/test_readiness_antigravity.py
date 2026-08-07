from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import BackendDecision
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.readiness_antigravity import (
    ReadinessDependenciesAntigravity,
    _create_readiness_antigravity,
)


_EXP437_PARENT = (
    "/Users/mac/Documents/Dataset/"
    "exp437_metadata_roundtrip_antigravity_20260807"
)


def _exp437_snapshot(*, terminal=False, **changes):
    payload = {
        "run_identity": "exp437-metadata-roundtrip-antigravity-20260807",
        "run_parent": _EXP437_PARENT,
        "state": "running",
        "pid": 43210,
        "pid_active": True,
        "pid_os_live": True,
        "backend": "cpu",
        "completed": 938,
        "checkpoint_completed": 939,
        "total": 1117,
        "error": 0,
        "heartbeat": "2026-08-07T04:50:00Z",
        "heartbeat_age_seconds": 1.0,
        "publication_exists": False,
        "evaluation_exists": False,
        "concurrency_authorized": True,
    }
    if terminal:
        payload.update(
            state="checkpoint_complete",
            pid_active=False,
            pid_os_live=False,
            completed=1117,
            checkpoint_completed=1117,
            publication_exists=True,
        )
    payload.update(changes)
    return payload


class _Context:
    datasets = ("A", "B")
    test_identity_manifest = {"A": ((0, 16),), "B": ((0, 16),)}
    code_commit = "a" * 40
    database_sha256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"


class _Session:
    context = _Context()
    label_access_count = 0

    def load_train(self, dataset):
        raise AssertionError("readiness must not load TRAIN values")

    def load_test_values(self, dataset, train):
        raise AssertionError("readiness must not load TEST values")


@contextmanager
def _session_factory(commit):
    yield _Session()


class Exp439ReadinessAntigravityTests(unittest.TestCase):
    def test_readiness_owns_absent_parent_and_freezes_all_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sources = {}
            for name in ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"):
                path = base / f"{name}_antigravity.bin"
                path.write_bytes(name.encode())
                sources[name] = path
            run_parent = base / "run_parent_antigravity"
            dependencies = ReadinessDependenciesAntigravity(
                clean_head=lambda: "a" * 40,
                sha256_file=lambda path: "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c" if path == sources["database"] else (path.name[0] * 64),
                exp437_terminal=lambda: _exp437_snapshot(terminal=True),
                input_session_factory=_session_factory,
                assess_backends=lambda: (BackendDecision("cpu", None, (), "synthetic fallback"), {"cpu": {"version": "synthetic"}}),
                focused_tests=lambda: {"returncode": 0, "tests": 68},
            )
            payload = _create_readiness_antigravity(
                run_parent=run_parent,
                source_paths=sources,
                dependencies=dependencies,
                expected_datasets=2,
            )
            artifact = run_parent / "readiness_antigravity.json"
            self.assertEqual(json.loads(artifact.read_text()), payload)
            self.assertEqual(payload["datasets"], ["A", "B"])
            self.assertEqual(payload["test_values_opened"], 0)
            self.assertEqual(payload["test_labels_opened"], 0)
            self.assertEqual(payload["exp437_execution"]["state"], "checkpoint_complete")
            self.assertEqual(payload["exp437_execution_mode"], "terminal")
            self.assertFalse(payload["concurrent_with_exp437"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                _create_readiness_antigravity(
                    run_parent=run_parent,
                    source_paths=sources,
                    dependencies=dependencies,
                    expected_datasets=2,
                )

    def test_fresh_live_cpu_exp437_is_accepted_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = {name: base / f"{name}_antigravity.bin" for name in ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest")}
            for path in paths.values():
                path.write_bytes(b"x")
            parent = base / "run_parent_antigravity"
            dependencies = ReadinessDependenciesAntigravity(
                clean_head=lambda: "a" * 40,
                sha256_file=lambda path: "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c" if path == paths["database"] else "a" * 64,
                exp437_terminal=lambda: _exp437_snapshot(),
                input_session_factory=_session_factory,
                assess_backends=lambda: (BackendDecision("cpu", None, (), "fallback"), {"cpu": {"version": "synthetic"}}),
                focused_tests=lambda: {"returncode": 0},
            )
            payload = _create_readiness_antigravity(
                run_parent=parent,
                source_paths=paths,
                dependencies=dependencies,
                expected_datasets=2,
            )
            self.assertEqual(payload["exp437_execution_mode"], "concurrent")
            self.assertTrue(payload["concurrent_with_exp437"])
            self.assertEqual(payload["exp437_execution"]["pid"], 43210)

    def test_unsafe_live_exp437_fails_before_creating_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = {name: base / f"{name}_antigravity.bin" for name in ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest")}
            for path in paths.values():
                path.write_bytes(b"x")
            parent = base / "run_parent_antigravity"
            dependencies = ReadinessDependenciesAntigravity(
                clean_head=lambda: "a" * 40,
                sha256_file=lambda path: "a" * 64,
                exp437_terminal=lambda: _exp437_snapshot(pid_os_live=False),
                input_session_factory=_session_factory,
                assess_backends=lambda: (BackendDecision("cpu", None, (), "fallback"), {}),
                focused_tests=lambda: {"returncode": 0},
            )
            with self.assertRaisesRegex(ValueError, "Exp437"):
                _create_readiness_antigravity(
                    run_parent=parent,
                    source_paths=paths,
                    dependencies=dependencies,
                    expected_datasets=2,
                )
            self.assertFalse(parent.exists())


if __name__ == "__main__":
    unittest.main()
