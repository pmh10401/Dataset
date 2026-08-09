"""RED/GREEN tests for the isolated Exp444 environment contract."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity import environment_antigravity as environment_module_antigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.environment_antigravity import (
    EnvironmentEvidenceAntigravity,
    verify_environment_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG, inner_thread_environment_names_antigravity


class Exp444EnvironmentAntigravityTests(unittest.TestCase):
    def test_environment_evidence_api_is_present(self) -> None:
        self.assertTrue(callable(verify_environment_antigravity))
        self.assertTrue(hasattr(EnvironmentEvidenceAntigravity, "from_payload"))

    def test_environment_evidence_round_trip_and_seven_thread_controls(self) -> None:
        evidence = EnvironmentEvidenceAntigravity(
            "3.14.6", "synthetic-platform", "arm64", 12, 11, 1,
            {"llvmlite": "0.48.0", "mlx-metal": "0.32.0", "mlx": "0.32.0", "numba": "0.66.0", "numpy": "2.4.6", "pip": "26.1.2"},
            FROZEN_CONFIG.environment_lock_sha256,
            {name: "1" for name in inner_thread_environment_names_antigravity()},
            (), FROZEN_CONFIG.acceleration_decision,
            "synthetic pre-run parity gate evidence", None, "0.32.0", "0.66.0", None, None,
        )
        self.assertEqual(EnvironmentEvidenceAntigravity.from_payload(evidence.to_payload()), evidence)
        with self.assertRaises(ValueError):
            EnvironmentEvidenceAntigravity(
                "3.14.6", "synthetic-platform", "arm64", 12, 11, 1,
                {"llvmlite": "0.48.0", "mlx-metal": "0.32.0", "mlx": "0.32.0", "numba": "0.66.0", "numpy": "2.4.6", "pip": "26.1.2"},
                FROZEN_CONFIG.environment_lock_sha256,
                {name: ("2" if name == "NUMBA_NUM_THREADS" else "1") for name in inner_thread_environment_names_antigravity()},
                (), FROZEN_CONFIG.acceleration_decision,
                "synthetic pre-run parity gate evidence", None, "0.32.0", "0.66.0", None, None,
            )

    def test_complete_freeze_rejects_missing_wrong_transitive_and_extra_distributions(self) -> None:
        lock_path = Path(FROZEN_CONFIG.environment_lock_path)
        expected = lock_path.read_bytes()
        environment_module_antigravity._validate_complete_freeze_bytes_antigravity(expected)
        cases = {
            "missing_transitive": expected.replace(b"llvmlite==0.48.0\n", b""),
            "wrong_transitive": expected.replace(b"mlx-metal==0.32.0", b"mlx-metal==0.31.0"),
            "wrong_direct": expected.replace(b"numba==0.66.0", b"numba==0.65.0"),
            "extra_distribution": expected + b"setuptools==99.0.0\n",
        }
        for name, actual in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "complete pip freeze"):
                environment_module_antigravity._validate_complete_freeze_bytes_antigravity(actual)

    def test_environment_evidence_requires_exact_complete_lock_inventory(self) -> None:
        complete = {
            "llvmlite": "0.48.0", "mlx-metal": "0.32.0", "mlx": "0.32.0",
            "numba": "0.66.0", "numpy": "2.4.6", "pip": "26.1.2",
        }
        common = (
            "3.14.6", "synthetic-platform", "arm64", 12, 11, 1,
            FROZEN_CONFIG.environment_lock_sha256,
            {name: "1" for name in inner_thread_environment_names_antigravity()},
            (), FROZEN_CONFIG.acceleration_decision, "synthetic gate",
        )
        evidence = EnvironmentEvidenceAntigravity(
            common[0], common[1], common[2], common[3], common[4], common[5],
            complete, common[6], common[7], common[8], common[9], common[10],
            None, "0.32.0", "0.66.0", None, None,
        )
        self.assertEqual(dict(evidence.package_versions), complete)
        for name, inventory in (
            ("missing_transitive", {key: value for key, value in complete.items() if key != "llvmlite"}),
            ("wrong_transitive", {**complete, "mlx-metal": "0.31.0"}),
            ("extra_distribution", {**complete, "setuptools": "99.0.0"}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                EnvironmentEvidenceAntigravity(
                    common[0], common[1], common[2], common[3], common[4], common[5],
                    inventory, common[6], common[7], common[8], common[9], common[10],
                    None, "0.32.0", "0.66.0", None, None,
                )

    def test_nonempty_bootstrap_lock_rejects_before_any_external_mutation(self) -> None:
        script = Path("scripts/create_exp444_environment_antigravity.sh").resolve()
        with tempfile.TemporaryDirectory(prefix="exp444_bootstrap_antigravity_") as root_name:
            root = Path(root_name)
            lock = root / "nonempty_lock_antigravity.txt"
            lock.write_text("numba==0.66.0\n", encoding="utf-8")
            marker = root / "python_invoked_antigravity.txt"
            fake_python = root / "fake_python_antigravity.sh"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                "printf invoked > \"${EXP444_FAKE_MARKER_ANTIGRAVITY}\"\n"
                "if [[ \"${1:-}\" == '-c' ]]; then printf '3.14.6\\n'; exit 0; fi\n"
                "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'venv' ]]; then "
                "mkdir -p \"$3/bin\"; cp \"$0\" \"$3/bin/python\"; chmod 700 \"$3/bin/python\"; exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o700)
            sentinel_venv = root / "absent_parent_antigravity" / "sentinel_venv_antigravity"
            shell = (
                "source \"$1\"; LOCK_REQUIREMENTS=\"$2\"; VENV_ROOT=\"$3\"; "
                "PYTHON_BIN=\"$4\"; if main_antigravity --bootstrap-lock-antigravity; "
                "then exit 0; else exit $?; fi"
            )
            completed = subprocess.run(
                ["bash", "-c", shell, "exp444_bootstrap_test_antigravity", str(script), str(lock), str(sentinel_venv), str(fake_python)],
                env={**os.environ, "EXP444_FAKE_MARKER_ANTIGRAVITY": str(marker)},
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("bootstrap is allowed only while the tracked lock is empty", completed.stderr)
            self.assertFalse(marker.exists(), "bootstrap rejection must happen before interpreter invocation")
            self.assertFalse(sentinel_venv.parent.exists(), "bootstrap rejection must not create the venv parent")


if __name__ == "__main__":
    unittest.main()
