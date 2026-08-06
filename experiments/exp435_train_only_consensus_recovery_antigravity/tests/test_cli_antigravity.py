"""TDD checks for the fixed Exp435 command boundary."""

from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from dataclasses import asdict
from hashlib import sha256

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp435_train_only_consensus_recovery_antigravity.cli_antigravity import (
    _CliDependencies,
    _write_new,
    build_parser,
    main,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.config_antigravity import FROZEN_CONFIG


class Exp435CliTests(unittest.TestCase):
    def _valid_readiness(self, commit: str) -> dict[str, object]:
        lanes = {
            "exp291_prediction": FROZEN_CONFIG.exp291_prediction_sha256,
            "exp291_manifest": FROZEN_CONFIG.exp291_manifest_sha256,
            "exp292_prediction": FROZEN_CONFIG.exp292_prediction_sha256,
            "exp292_manifest": FROZEN_CONFIG.exp292_manifest_sha256,
        }
        return {
            "schema": "exp435.readiness.v1",
            "state": "ready",
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "run_identity": "exp435-train-only-consensus-recovery-antigravity-20260806",
            "git_commit": commit,
            "config_sha256": sha256(canonical_json_bytes(asdict(FROZEN_CONFIG))).hexdigest(),
            "input_sha256": {"database": FROZEN_CONFIG.database_sha256, **lanes},
            "database_sha256": FROZEN_CONFIG.database_sha256,
            "lane_artifact_sha256": lanes,
            "backend": "cpu",
            "backend_decision": {"backend": "cpu"},
            "runtime_versions": {"python": "test"},
            "scientific_parity": True,
            "output_root_absent": True,
            "checkpoint_root_absent": True,
            "evaluation_root_absent": True,
            "test_values_opened": 0,
            "test_labels_opened": 0,
            "label_access_count": 0,
            "focused_tests": {"returncode": 0},
        }

    def _dependencies(self, parent: Path, calls: list[str]) -> _CliDependencies:
        output = parent / "output_antigravity"
        checkpoint = parent / "checkpoint_antigravity"
        readiness = parent / "readiness_antigravity.json"

        def readiness_call() -> dict[str, object]:
            calls.append("readiness")
            return {
                "schema": "exp435.readiness.v1",
                "state": "ready",
                "git_commit": "a" * 40,
                "backend": "cpu",
                "output_root_absent": True,
                "checkpoint_root_absent": True,
            }

        def execute(mode: str) -> dict[str, object]:
            calls.append(mode)
            return {
                "state": "complete_review_only",
                "completed_datasets": 1117,
                "failed_datasets": 0,
            }

        return _CliDependencies(
            output=output,
            checkpoint=checkpoint,
            readiness=readiness,
            current_head=lambda: "a" * 40,
            readiness_call=readiness_call,
            execute=execute,
            inspect=lambda: {"state": "complete_review_only"},
            evaluate=lambda: {"state": "evaluated"},
        )

    def _main(self, argv: list[str], dependencies: _CliDependencies):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv, _dependencies=dependencies)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_parser_has_exactly_five_modes(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        for name in ("readiness", "new", "resume", "inspect", "evaluate"):
            self.assertIn(name, help_text)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["new", "--output-root", "/tmp/alternate"])

    def test_readiness_uses_only_fixed_file_and_no_alternate_path(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            code, _stdout, stderr = self._main(
                ["readiness", "--readiness-output", str(parent / "other.json")],
                dependencies,
            )
            self.assertEqual(code, 2)
            self.assertIn("unrecognized", stderr)
            self.assertEqual(calls, [])

    def test_new_rechecks_clean_commit_before_execute(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            dependencies.readiness.write_text(
                json.dumps({"schema": "exp435.readiness.v1", "state": "ready", "git_commit": "a" * 40}),
                encoding="utf-8",
            )
            code, _stdout, stderr = self._main(["new"], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("readiness", stderr)
            self.assertEqual(calls, [])

    def test_readiness_action_is_the_single_writer(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "new_parent"
            output = parent / "output_antigravity"
            checkpoint = parent / "checkpoint_antigravity"
            readiness = parent / "readiness_antigravity.json"
            calls: list[str] = []
            payload = self._valid_readiness("a" * 40)

            def readiness_call() -> dict[str, object]:
                calls.append("readiness")
                parent.mkdir()
                _write_new(readiness, canonical_json_bytes(payload))
                return payload

            dependencies = _CliDependencies(
                output=output,
                checkpoint=checkpoint,
                readiness=readiness,
                current_head=lambda: "a" * 40,
                readiness_call=readiness_call,
                execute=lambda mode: {"state": "complete_review_only"},
                inspect=lambda: {"state": "complete_review_only"},
                evaluate=lambda: {"state": "evaluated"},
            )
            code, stdout, stderr = self._main(["readiness"], dependencies)
            self.assertEqual(code, 0, stderr)
            self.assertEqual(calls, ["readiness"])
            self.assertEqual(readiness.read_bytes(), canonical_json_bytes(payload))
            self.assertEqual(json.loads(stdout), payload)

    def test_prediction_modes_do_not_import_evaluation(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            with patch.dict("sys.modules", {"experiments.exp435_train_only_consensus_recovery_antigravity.evaluation_antigravity": None}):
                code, _stdout, stderr = self._main(["inspect"], dependencies)
            self.assertEqual(code, 2)
            self.assertNotIn("evaluation", stderr.lower())


if __name__ == "__main__":
    unittest.main()
