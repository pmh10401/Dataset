from pathlib import Path
import subprocess
import sys
import unittest

from experiments.exp438_frequency_complexity_evidence_antigravity.cli_antigravity import (
    CliDependenciesAntigravity,
    _validate_exp437_replay_antigravity,
    build_parser_antigravity,
    main_with_dependencies_antigravity,
)


def _exp437_snapshot_antigravity(**changes):
    payload = {
        "run_identity": "exp437-metadata-roundtrip-antigravity-20260807",
        "run_parent": "/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807",
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
    payload.update(changes)
    return payload


class Exp438CliAntigravityTests(unittest.TestCase):
    def test_replay_accepts_monotonic_live_recovery_pid(self) -> None:
        bound = _exp437_snapshot_antigravity()
        payload = {
            "exp437_execution": bound,
            "exp437_execution_mode": "concurrent",
            "concurrent_with_exp437": True,
        }
        current = _exp437_snapshot_antigravity(
            pid=54321, completed=950, checkpoint_completed=951
        )
        self.assertEqual(
            _validate_exp437_replay_antigravity(payload, current), "concurrent"
        )

    def test_replay_rejects_regression_or_tampered_mode(self) -> None:
        bound = _exp437_snapshot_antigravity()
        base = {
            "exp437_execution": bound,
            "exp437_execution_mode": "concurrent",
            "concurrent_with_exp437": True,
        }
        cases = (
            (base, _exp437_snapshot_antigravity(completed=937)),
            ({**base, "exp437_execution_mode": "terminal"}, bound),
            ({**base, "concurrent_with_exp437": False}, bound),
            (base, _exp437_snapshot_antigravity(pid_os_live=False)),
            (base, _exp437_snapshot_antigravity(backend="mps")),
        )
        for payload, current in cases:
            with self.subTest(payload=payload, current=current), self.assertRaises(ValueError):
                _validate_exp437_replay_antigravity(payload, current)

    def test_parser_exposes_exactly_five_commands_and_no_overrides(self) -> None:
        parser = build_parser_antigravity()
        for command in ("readiness", "new", "resume", "inspect", "evaluate"):
            self.assertEqual(parser.parse_args([command]).command, command)
        for forbidden in (
            ["new", "--workers", "1"],
            ["new", "--backend", "cpu"],
            ["new", "--threshold", "0.5"],
            ["evaluate", "--alpha", "0.1"],
        ):
            with self.assertRaises(SystemExit):
                parser.parse_args(forbidden)

    def test_dispatch_calls_only_the_selected_dependency(self) -> None:
        evaluation_module = (
            "experiments.exp438_frequency_complexity_evidence_antigravity."
            "evaluation_antigravity"
        )
        previously_loaded = sys.modules.pop(evaluation_module, None)
        called = []
        dependencies = CliDependenciesAntigravity(
            **{
                command: (lambda command=command: called.append(command))
                for command in ("readiness", "new", "resume", "inspect", "evaluate")
            }
        )
        try:
            self.assertEqual(main_with_dependencies_antigravity(["inspect"], dependencies), 0)
            self.assertEqual(called, ["inspect"])
            self.assertNotIn(evaluation_module, sys.modules)
        finally:
            if previously_loaded is not None:
                sys.modules[evaluation_module] = previously_loaded

    def test_wrapper_help_works_outside_repository_cwd(self) -> None:
        root = Path(__file__).resolve().parents[3]
        wrapper = root / "scratch/run_exp438_frequency_complexity_evidence_antigravity.py"
        completed = subprocess.run(
            [sys.executable, str(wrapper), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("readiness", completed.stdout)
        self.assertIn("evaluate", completed.stdout)


if __name__ == "__main__":
    unittest.main()
