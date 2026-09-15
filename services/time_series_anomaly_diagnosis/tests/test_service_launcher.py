import builtins
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from services.time_series_anomaly_diagnosis import service_launcher
from services.time_series_anomaly_diagnosis.service_launcher import (
    resolve_data_directory,
    resolve_resources_directory,
)
from services.time_series_anomaly_diagnosis.tests.test_dataset_importer import (
    make_source_database,
    read_labels,
)


class ServiceLauncherTest(unittest.TestCase):
    def test_default_data_directory_is_next_to_windows_executable(self) -> None:
        actual = resolve_data_directory(None, Path("C:/Diagnosis/Diagnosis.exe"), "win32")
        self.assertEqual(actual, Path("C:/Diagnosis/data"))

    def test_default_data_directory_is_next_to_mac_app(self) -> None:
        executable = Path("/Applications/Diagnosis.app/Contents/MacOS/Diagnosis")
        self.assertEqual(
            resolve_data_directory(None, executable, "darwin"),
            Path("/Applications/data"),
        )

    def test_explicit_data_directory_is_resolved(self) -> None:
        with TemporaryDirectory() as directory:
            expected = Path(directory).resolve()
            self.assertEqual(resolve_data_directory(Path(directory), Path("unused"), "linux"), expected)

    def test_frozen_resources_are_loaded_from_packaged_service_path(self) -> None:
        frozen_root = Path("/Applications/Diagnosis.app/Contents/Frameworks")

        self.assertEqual(
            resolve_resources_directory(None, Path("unused"), frozen_root),
            frozen_root / "services/time_series_anomaly_diagnosis/resources",
        )

    def test_source_resources_are_next_to_launcher(self) -> None:
        launcher = Path("/checkout/services/time_series_anomaly_diagnosis/service_launcher.py")

        self.assertEqual(
            resolve_resources_directory(None, launcher, None),
            launcher.parent / "resources",
        )

    def test_explicit_resources_override_frozen_location(self) -> None:
        with TemporaryDirectory() as directory:
            expected = Path(directory).resolve()
            self.assertEqual(
                resolve_resources_directory(
                    Path(directory),
                    Path("unused"),
                    Path("/frozen"),
                ),
                expected,
            )

    def test_main_prepares_numba_cache_before_import_and_closes_cleanly(self) -> None:
        with TemporaryDirectory() as directory:
            data = Path(directory).resolve()
            (data / "univariate_ts.db").touch()
            resources = data / "resources"
            resources.mkdir()
            for name in (
                "model_manifest.json",
                "routing_catalog.json",
                "recovery_catalog.json",
                "review_evidence.json",
            ):
                (resources / name).touch()

            stopped = Event()
            handlers = {}
            server = MagicMock(server_port=7777)
            server.service_status = {"team_url": "http://diagnosis.internal:7777/"}
            server.shutdown.side_effect = stopped.set
            create_runtime_server = MagicMock(return_value=server)
            real_import = builtins.__import__

            def import_after_cache(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "services.time_series_anomaly_diagnosis.application_server":
                    self.assertEqual(os.environ["NUMBA_CACHE_DIR"], str(data / "cache" / "numba"))
                    return SimpleNamespace(create_runtime_server=create_runtime_server)
                return real_import(name, globals, locals, fromlist, level)

            def save_handler(kind, handler):
                handlers[kind] = handler

            def serve():
                handlers[service_launcher.signal.SIGINT](service_launcher.signal.SIGINT, None)
                self.assertTrue(stopped.wait(timeout=2.0))

            server.serve_forever.side_effect = serve
            output = StringIO()
            with (
                patch.object(builtins, "__import__", side_effect=import_after_cache),
                patch.object(service_launcher.signal, "signal", side_effect=save_handler),
                patch.object(service_launcher.webbrowser, "open") as open_browser,
                redirect_stdout(output),
            ):
                service_launcher.main(
                    [
                        "--port", "0",
                        "--public-host", "diagnosis.internal",
                        "--data-dir", str(data),
                        "--resources-directory", str(resources),
                    ]
                )

        create_runtime_server.assert_called_once_with(
            "0.0.0.0",
            0,
            "diagnosis.internal",
            data / "univariate_ts.db",
            data / "analysis_history.sqlite3",
            resources,
        )
        open_browser.assert_called_once_with("http://127.0.0.1:7777/")
        self.assertIn("http://diagnosis.internal:7777/", output.getvalue())
        server.server_close.assert_called_once_with()

    def test_missing_source_database_fails_before_browser_opens(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.object(service_launcher.webbrowser, "open") as open_browser:
                with self.assertRaisesRegex(FileNotFoundError, "univariate_ts.db"):
                    service_launcher.main(["--data-dir", directory])
        open_browser.assert_not_called()

    def test_module_import_installs_database_with_default_or_explicit_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite3"
            make_source_database(source, train_labels=("7", "7", "0"), test_labels=("7", "0"))
            original = source.read_bytes()
            for explicit_report in (False, True):
                with self.subTest(explicit_report=explicit_report):
                    data = root / str(explicit_report)
                    report = root / "custom.html" if explicit_report else data / "database_import_report.html"
                    command = [
                        sys.executable, "-m", service_launcher.__name__,
                        "--data-dir", str(data), "--import-database", str(source),
                    ]
                    if explicit_report:
                        command.extend(["--import-report", str(report)])
                    completed = subprocess.run(
                        command, capture_output=True, text=True, timeout=20,
                        cwd=Path(__file__).resolve().parents[3],
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(read_labels(data / "univariate_ts.db", "TRAIN"), ("0", "0", "1"))
                    self.assertEqual(read_labels(data / "univariate_ts.db", "TEST"), ("0", "1"))
                    self.assertIn("Demo", report.read_text(encoding="utf-8"))
                    self.assertFalse((data / "analysis_history.sqlite3").exists())
            self.assertEqual(source.read_bytes(), original)

    def test_import_exits_before_runtime_import_and_browser_open(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite3"
            make_source_database(source)
            real_import = builtins.__import__

            def reject_runtime(name, globals=None, locals=None, fromlist=(), level=0):
                self.assertNotEqual(name, "services.time_series_anomaly_diagnosis.application_server")
                return real_import(name, globals, locals, fromlist, level)

            with (
                patch.object(builtins, "__import__", side_effect=reject_runtime),
                patch.object(service_launcher.webbrowser, "open") as open_browser,
                redirect_stdout(StringIO()),
            ):
                service_launcher.main([
                    "--data-dir", str(root / "data"), "--import-database", str(source),
                ])
            open_browser.assert_not_called()

    def test_import_report_requires_import_database(self) -> None:
        output = StringIO()
        with redirect_stderr(output), self.assertRaises(SystemExit) as raised:
            service_launcher.main(["--import-report", "report.html"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--import-report requires --import-database", output.getvalue())


if __name__ == "__main__":
    unittest.main()
