from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp438_frequency_complexity_evidence_antigravity.contracts_antigravity import BackendDecision
from experiments.exp438_frequency_complexity_evidence_antigravity.readiness_antigravity import (
    ReadinessDependenciesAntigravity,
    _create_readiness_antigravity,
)


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


class Exp438ReadinessAntigravityTests(unittest.TestCase):
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
                exp437_terminal=lambda: {"state": "complete", "pid_active": False},
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
            self.assertEqual(payload["exp437_terminal"]["state"], "complete")
            with self.assertRaisesRegex(ValueError, "already exists"):
                _create_readiness_antigravity(
                    run_parent=run_parent,
                    source_paths=sources,
                    dependencies=dependencies,
                    expected_datasets=2,
                )

    def test_nonterminal_exp437_fails_before_creating_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = {name: base / f"{name}_antigravity.bin" for name in ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest")}
            for path in paths.values():
                path.write_bytes(b"x")
            parent = base / "run_parent_antigravity"
            dependencies = ReadinessDependenciesAntigravity(
                clean_head=lambda: "a" * 40,
                sha256_file=lambda path: "a" * 64,
                exp437_terminal=lambda: {"state": "running", "pid_active": True},
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
