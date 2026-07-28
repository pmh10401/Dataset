from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stderr
from io import StringIO
import unittest

from scratch.run_exp411_near_repeat_qualification import main


class Exp411CliTests(unittest.TestCase):
    def test_benchmark_mode_requires_new_report_path(self) -> None:
        with TemporaryDirectory() as raw:
            database = Path(raw) / "database.db"
            database.write_bytes(b"db")
            stderr = StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main([
                    "--acceleration-benchmark",
                    "--db-path",
                    str(database),
                ])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn(
                "--acceleration-benchmark requires --benchmark-report",
                stderr.getvalue(),
            )

    def test_full_mode_requires_checkpoint_and_benchmark_paths(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            manifest = root / "manifest.json"
            database.write_bytes(b"db")
            manifest.write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                main([
                    "--full-qualification",
                    "--db-path",
                    str(database),
                    "--population-manifest",
                    str(manifest),
                    "--output-dir",
                    str(root / "output"),
                ])

            self.assertEqual(raised.exception.code, 2)

    def test_smoke_mode_rejects_checkpoint_path(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)

            with self.assertRaises(SystemExit) as raised:
                main([
                    "--representative-smoke",
                    "--checkpoint-dir",
                    str(root / "checkpoint"),
                    "--output-dir",
                    str(root / "output"),
                ])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
