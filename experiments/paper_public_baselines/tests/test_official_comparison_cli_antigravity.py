from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "scratch/run_official_public_baseline_comparison_antigravity.py"


def _load_launcher():
    specification = importlib.util.spec_from_file_location(
        "official_comparison_launcher_antigravity", SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("launcher specification is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class OfficialComparisonCliTests(unittest.TestCase):
    def test_verifier_uses_the_canonical_evaluation_artifact_name(self) -> None:
        module = _load_launcher()

        self.assertEqual(
            module.EVALUATION_ARTIFACT_NAME_ANTIGRAVITY,
            "artifact_sha256.json",
        )

    def test_parser_accepts_only_closed_official_commands(self) -> None:
        module = _load_launcher()
        parser = module.build_parser_antigravity()

        for command in ("prepare-deep-view", "evaluate", "combine", "verify"):
            with self.subTest(command=command):
                self.assertEqual(parser.parse_args([command]).command, command)
        with self.assertRaises(SystemExit):
            parser.parse_args(["score"])

    def test_import_is_label_and_scorer_free(self) -> None:
        code = (
            "import importlib.util,sys;"
            f"p={str(SCRIPT)!r};"
            "s=importlib.util.spec_from_file_location('official_cli',p);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "bad=[n for n in sys.modules if n.endswith('.label_access') "
            "or n.endswith('.methods.deep_svdd') or n.endswith('.methods.drocc')];"
            "assert not bad,bad"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
