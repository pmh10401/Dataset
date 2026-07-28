from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp414_missing_aware_near_repeat.contracts import (
    MissingInventory,
)
from scratch.run_exp414_missing_aware_near_repeat import (
    _atomic_json_write,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class Exp414CliTests(unittest.TestCase):
    def test_modes_are_mutually_exclusive(self) -> None:
        stderr = StringIO()
        with (
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["--input-audit", "--representative-smoke"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_full_mode_requires_every_real_input_and_new_output(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            manifest = root / "manifest.json"
            database.write_bytes(b"db")
            manifest.write_text("{}", encoding="utf-8")
            stderr = StringIO()
            with (
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--full-qualification",
                        "--db-path",
                        str(database),
                        "--population-manifest",
                        str(manifest),
                        "--output-dir",
                        str(root / "output"),
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        for required in (
            "--exp411-checkpoint-dir",
            "--checkpoint-dir",
            "--benchmark-report",
        ):
            self.assertIn(required, stderr.getvalue())

    def test_smoke_mode_forbids_every_real_path(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            file_path = root / "input"
            file_path.write_bytes(b"x")
            real_arguments = (
                ("--db-path", file_path),
                ("--population-manifest", file_path),
                ("--exp411-checkpoint-dir", root),
                ("--checkpoint-dir", root / "checkpoint"),
                ("--benchmark-report", file_path),
            )
            for flag, path in real_arguments:
                with (
                    self.subTest(flag=flag),
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    main(
                        [
                            "--representative-smoke",
                            flag,
                            str(path),
                            "--output-dir",
                            str(root / f"output-{flag[2:]}"),
                        ]
                    )

    def test_benchmark_refuses_an_existing_report(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            report = root / "benchmark.json"
            database.write_bytes(b"db")
            report.write_text("{}", encoding="utf-8")
            with (
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--acceleration-benchmark",
                        "--db-path",
                        str(database),
                        "--benchmark-report",
                        str(report),
                    ]
                )
        self.assertEqual(raised.exception.code, 2)

    def test_input_audit_prints_canonical_inventory_without_writing(self) -> None:
        inventory = MissingInventory(
            datasets=1117,
            affected_datasets=19,
            affected_runs=164,
            nonfinite_values=983,
            all_missing_runs=0,
            affected_datasets_sha256="c" * 64,
        )
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            manifest = root / "manifest.json"
            database.write_bytes(b"db")
            manifest.write_text("{}", encoding="utf-8")
            stdout = StringIO()
            with (
                patch(
                    "scratch.run_exp414_missing_aware_near_repeat."
                    "_load_population_manifest",
                    return_value=("Alpha_normal_0",),
                ),
                patch(
                    "scratch.run_exp414_missing_aware_near_repeat."
                    "scan_missing_inventory",
                    return_value=inventory,
                ),
                patch(
                    "scratch.run_exp414_missing_aware_near_repeat."
                    "_require_inventory"
                ) as require_inventory,
                redirect_stdout(stdout),
            ):
                result = main(
                    [
                        "--input-audit",
                        "--db-path",
                        str(database),
                        "--population-manifest",
                        str(manifest),
                    ]
                )
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"database.db", "manifest.json"},
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            stdout.getvalue().encode("ascii"),
            canonical_json_bytes(asdict(inventory)) + b"\n",
        )
        require_inventory.assert_called_once_with(inventory)

    def test_benchmark_writes_one_new_canonical_json_file(self) -> None:
        payload = {"schema_version": 1, "status": "passed"}
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            report = root / "benchmark.json"
            database.write_bytes(b"db")
            stdout = StringIO()
            with (
                patch(
                    "scratch.run_exp414_missing_aware_near_repeat."
                    "run_fixed_train_benchmark",
                    return_value=payload,
                ),
                redirect_stdout(stdout),
            ):
                result = main(
                    [
                        "--acceleration-benchmark",
                        "--db-path",
                        str(database),
                        "--benchmark-report",
                        str(report),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(report.read_text()), payload)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"database.db", "benchmark.json"},
            )

    def test_benchmark_race_never_clobbers_a_report_that_appears(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            report = root / "benchmark.json"
            real_link = os.link

            def publish_competitor(source: object, destination: object) -> None:
                Path(destination).write_bytes(b"competitor\n")
                real_link(source, destination)

            with (
                patch(
                    "scratch.run_exp414_missing_aware_near_repeat.os.link",
                    side_effect=publish_competitor,
                ),
                self.assertRaises(FileExistsError),
            ):
                _atomic_json_write(report, {"status": "ours"})
            self.assertEqual(report.read_bytes(), b"competitor\n")
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"benchmark.json"},
            )

    def test_smoke_requires_an_output_below_a_temporary_root(self) -> None:
        output = REPO_ROOT / ".exp414-nontemporary-smoke-output"
        self.assertFalse(output.exists())
        with (
            patch(
                "scratch.run_exp414_missing_aware_near_repeat."
                "run_representative_smoke"
            ) as run_smoke,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main(
                [
                    "--representative-smoke",
                    "--output-dir",
                    str(output),
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        run_smoke.assert_not_called()

    def test_full_mode_rejects_nested_execution_roots(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            manifest = root / "manifest.json"
            report = root / "benchmark.json"
            source_checkpoint = root / "exp411"
            database.write_bytes(b"db")
            manifest.write_text("{}", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")
            source_checkpoint.mkdir()
            nested_cases = (
                (
                    source_checkpoint / "exp414",
                    root / "publication-a",
                ),
                (
                    root / "exp414-b",
                    root / "exp414-b" / "publication",
                ),
            )
            for target_checkpoint, output in nested_cases:
                with (
                    self.subTest(
                        target_checkpoint=target_checkpoint,
                        output=output,
                    ),
                    patch(
                        "scratch.run_exp414_missing_aware_near_repeat."
                        "run_full_qualification"
                    ) as run_full,
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(
                        [
                            "--full-qualification",
                            "--db-path",
                            str(database),
                            "--population-manifest",
                            str(manifest),
                            "--exp411-checkpoint-dir",
                            str(source_checkpoint),
                            "--checkpoint-dir",
                            str(target_checkpoint),
                            "--benchmark-report",
                            str(report),
                            "--output-dir",
                            str(output),
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)
                run_full.assert_not_called()

    def test_full_mode_delegates_all_six_fixed_inputs_once(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "database.db"
            manifest = root / "manifest.json"
            source_checkpoint = root / "exp411"
            target_checkpoint = root / "exp414"
            report = root / "benchmark.json"
            output = root / "publication"
            database.write_bytes(b"db")
            manifest.write_text("{}", encoding="utf-8")
            source_checkpoint.mkdir()
            report.write_text("{}", encoding="utf-8")
            with (
                patch(
                    "scratch.run_exp414_missing_aware_near_repeat."
                    "run_full_qualification",
                    return_value={"artifact": "d" * 64},
                ) as run_full,
                redirect_stdout(StringIO()),
            ):
                result = main(
                    [
                        "--full-qualification",
                        "--db-path",
                        str(database),
                        "--population-manifest",
                        str(manifest),
                        "--exp411-checkpoint-dir",
                        str(source_checkpoint),
                        "--checkpoint-dir",
                        str(target_checkpoint),
                        "--benchmark-report",
                        str(report),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            run_full.assert_called_once_with(
                database.resolve(),
                manifest.resolve(),
                output.resolve(),
                exp411_checkpoint_dir=source_checkpoint.resolve(),
                checkpoint_dir=target_checkpoint.resolve(),
                benchmark_report=report.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
