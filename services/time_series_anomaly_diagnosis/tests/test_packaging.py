from hashlib import sha256
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = SERVICE_ROOT / "packaging"
SPEC = PACKAGING_ROOT / "service_app.spec"
REQUIREMENTS = PACKAGING_ROOT / "requirements-build.txt"


class PackagingTest(unittest.TestCase):
    def test_package_includes_service_assets_and_excludes_databases(self) -> None:
        spec = SPEC.read_text(encoding="utf-8")

        self.assertIn("service_root = Path(SPEC).resolve().parent.parent", spec)
        self.assertIn("service_launcher.py", spec)
        self.assertIn('service_root / "web"', spec)
        self.assertIn('service_root / "resources"', spec)
        self.assertNotIn("univariate_ts.db", spec)
        self.assertNotIn("analysis_history.sqlite3", spec)
        for hidden_import in (
            "aeon.transformations.collection.convolution_based._minirocket",
            "aeon.transformations.collection.convolution_based._multirocket",
            "aeon.transformations.collection.convolution_based._hydra",
        ):
            self.assertIn(hidden_import, spec)
        self.assertIn("exclude_binaries=True", spec)
        self.assertIn("COLLECT(", spec)
        self.assertIn("BUNDLE(", spec)
        self.assertIn("local.research.time-series-anomaly-diagnosis", spec)
        self.assertNotIn('excludes=["torch"]', spec)

    def test_direct_build_versions_are_exactly_pinned(self) -> None:
        self.assertEqual(
            REQUIREMENTS.read_text(encoding="utf-8").splitlines(),
            [
                "numpy==2.3.5",
                "numba==0.63.1",
                "aeon==1.4.0",
                "scikit-learn==1.7.2",
                "pyinstaller==6.22.2",
                "pyinstaller-hooks-contrib==2026.7",
                "torch==2.13.0",
            ],
        )

    def test_build_command_targets_current_os_spec(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            compose_build_command,
        )

        output = Path("dist/package")
        command = compose_build_command(
            output,
            clean=True,
            python_executable=Path("/opt/diagnosis/python"),
        )

        self.assertEqual(
            command,
            [
                "/opt/diagnosis/python",
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--distpath",
                str(output.resolve()),
                "--workpath",
                str((output.parent / ".package-build").resolve()),
                "--clean",
                str(SPEC),
            ],
        )
        self.assertNotIn("--target-os", command)

    def test_runtime_manifest_records_reproducibility_inputs(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            build_runtime_manifest,
        )

        command = ["python", "-m", "PyInstaller", "service_app.spec"]
        manifest = build_runtime_manifest(command, "2026-09-08T12:00:00+00:00")

        self.assertEqual(
            manifest,
            {
                "schema": "time_series_anomaly_diagnosis_runtime_manifest_v1",
                "built_at_utc": "2026-09-08T12:00:00+00:00",
                "python_version": platform.python_version(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "direct_dependencies": {
                    "numpy": "2.3.5",
                    "numba": "0.63.1",
                    "aeon": "1.4.0",
                    "scikit-learn": "1.7.2",
                    "pyinstaller": "6.22.2",
                    "pyinstaller-hooks-contrib": "2026.7",
                    "torch": "2.13.0",
                },
                "model_manifest_sha256": sha256(
                    (SERVICE_ROOT / "resources/model_manifest.json").read_bytes()
                ).hexdigest(),
                "build_command": command,
            },
        )

    def test_finalize_distribution_creates_empty_external_data_directory(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            finalize_distribution,
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "TimeSeriesAnomalyDiagnosis.app").mkdir()

            final_directory = finalize_distribution(
                output,
                "Darwin",
                ["python", "-m", "PyInstaller"],
                "2026-09-08T12:00:00+00:00",
            )

            self.assertEqual(final_directory, output.resolve())
            self.assertEqual(list((output / "data").iterdir()), [])
            manifest = json.loads((output / "runtime_manifest.json").read_text("utf-8"))
            self.assertEqual(
                manifest["schema"],
                "time_series_anomaly_diagnosis_runtime_manifest_v1",
            )

    def test_finalize_distribution_rejects_bundled_database(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            finalize_distribution,
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "TimeSeriesAnomalyDiagnosis.app").mkdir()
            (output / "accidental.sqlite3").write_bytes(b"not a database")

            with self.assertRaisesRegex(ValueError, "database"):
                finalize_distribution(
                    output,
                    "Darwin",
                    ["python", "-m", "PyInstaller"],
                    "2026-09-08T12:00:00+00:00",
                )

    def test_finalize_distribution_reuses_empty_data_directory(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            finalize_distribution,
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "TimeSeriesAnomalyDiagnosis.app").mkdir()
            (output / "data").mkdir()

            finalize_distribution(
                output,
                "Darwin",
                ["python", "-m", "PyInstaller"],
                "2026-09-08T12:00:00+00:00",
            )

            self.assertEqual(list((output / "data").iterdir()), [])

    def test_finalize_distribution_rejects_nonempty_data_directory(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            finalize_distribution,
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "TimeSeriesAnomalyDiagnosis.app").mkdir()
            data_directory = output / "data"
            data_directory.mkdir()
            (data_directory / "keep.txt").write_text("administrator data", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                finalize_distribution(
                    output,
                    "Darwin",
                    ["python", "-m", "PyInstaller"],
                    "2026-09-08T12:00:00+00:00",
                )

    def test_finalize_distribution_locates_windows_one_folder(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import (
            finalize_distribution,
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            distribution = output / "TimeSeriesAnomalyDiagnosis"
            distribution.mkdir()
            (distribution / "TimeSeriesAnomalyDiagnosis.exe").write_bytes(b"executable")

            final_directory = finalize_distribution(
                output,
                "Windows",
                ["python", "-m", "PyInstaller"],
                "2026-09-08T12:00:00+00:00",
            )

            self.assertEqual(final_directory, distribution.resolve())
            self.assertEqual(list((distribution / "data").iterdir()), [])

    def test_main_runs_current_os_build_and_finalizes_output(self) -> None:
        from services.time_series_anomaly_diagnosis.packaging.build_package import main

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "package"
            commands: list[list[str]] = []

            def fake_run(command: list[str], **options: object) -> None:
                self.assertEqual(options, {"check": True, "cwd": SERVICE_ROOT.parents[1]})
                commands.append(command)
                (output / "TimeSeriesAnomalyDiagnosis.app").mkdir(parents=True)

            final_directory = main(
                ["--clean", "--output", str(output)],
                run_command=fake_run,
                system_name="Darwin",
            )

            self.assertEqual(final_directory, output.resolve())
            self.assertEqual(len(commands), 1)
            self.assertIn("--clean", commands[0])
            self.assertEqual(list((output / "data").iterdir()), [])
            self.assertTrue((output / "runtime_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
