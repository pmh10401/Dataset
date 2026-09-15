from argparse import ArgumentParser
import os
from pathlib import Path
import signal
import socket
import sys
from threading import Thread
import webbrowser

from services.time_series_anomaly_diagnosis.dataset_importer import install_database


def resolve_data_directory(
    explicit: Path | None,
    executable: Path,
    platform: str,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if platform == "darwin" and executable.parent.name == "MacOS":
        return executable.parents[3] / "data"
    return executable.parent / "data"


def resolve_resources_directory(
    explicit: Path | None,
    module_file: Path,
    frozen_root: Path | None,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if frozen_root is not None:
        return frozen_root / "services" / "time_series_anomaly_diagnosis" / "resources"
    return module_file.parent / "resources"


def main(arguments: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = ArgumentParser(description="Time Series Anomaly Diagnosis")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8485)
    parser.add_argument("--public-host", default=socket.gethostname())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--import-database", type=Path)
    parser.add_argument("--import-report", type=Path)
    parser.add_argument("--resources-directory", type=Path)
    args = parser.parse_args(arguments)
    if args.import_report is not None and args.import_database is None:
        parser.error("--import-report requires --import-database")
    data_directory = resolve_data_directory(args.data_dir, Path(sys.executable), sys.platform)
    if args.import_database is not None:
        report = args.import_report or data_directory / "database_import_report.html"
        summary = install_database(args.import_database, data_directory, report)
        print(f"등록 성공: {len(summary.datasets)}개 데이터셋; 보고서: {report}")
        return
    frozen_root = getattr(sys, "_MEIPASS", None)
    resources_directory = resolve_resources_directory(
        args.resources_directory,
        Path(__file__),
        Path(frozen_root) if frozen_root else None,
    )
    source_database = data_directory / "univariate_ts.db"
    if not source_database.is_file():
        raise FileNotFoundError(f"source database not found: {source_database}")
    cache_directory = data_directory / "cache" / "numba"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_directory)

    from services.time_series_anomaly_diagnosis.application_server import (
        create_runtime_server,
    )

    server = create_runtime_server(
        args.host,
        args.port,
        args.public_host,
        source_database,
        data_directory / "analysis_history.sqlite3",
        resources_directory,
    )
    print(server.service_status["team_url"], flush=True)
    webbrowser.open(f"http://127.0.0.1:{server.server_port}/")

    def stop_server(*_ignored: object) -> None:
        Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
