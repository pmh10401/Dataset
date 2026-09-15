from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys


PACKAGING_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = PACKAGING_ROOT.parent
SPEC = PACKAGING_ROOT / "service_app.spec"
REQUIREMENTS = PACKAGING_ROOT / "requirements-build.txt"
APPLICATION_NAME = "TimeSeriesAnomalyDiagnosis"
DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def direct_dependencies() -> dict[str, str]:
    return dict(
        line.split("==", 1)
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line
    )


def compose_build_command(
    output: Path,
    clean: bool,
    python_executable: Path = Path(sys.executable),
) -> list[str]:
    output = output.resolve()
    command = [
        str(python_executable),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(output),
        "--workpath",
        str((output.parent / f".{output.name}-build").resolve()),
    ]
    if clean:
        command.append("--clean")
    return [*command, str(SPEC)]


def build_runtime_manifest(
    build_command: list[str],
    built_at_utc: str | None = None,
) -> dict[str, object]:
    return {
        "schema": "time_series_anomaly_diagnosis_runtime_manifest_v1",
        "built_at_utc": built_at_utc or datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "direct_dependencies": direct_dependencies(),
        "model_manifest_sha256": sha256(
            (SERVICE_ROOT / "resources/model_manifest.json").read_bytes()
        ).hexdigest(),
        "build_command": build_command,
    }


def distribution_directory(output: Path, system_name: str) -> Path:
    if system_name == "Darwin":
        artifact = output / f"{APPLICATION_NAME}.app"
        final_directory = output
    elif system_name == "Windows":
        final_directory = output / APPLICATION_NAME
        artifact = final_directory / f"{APPLICATION_NAME}.exe"
    else:
        raise ValueError(f"unsupported packaging platform: {system_name}")
    if not artifact.exists():
        raise FileNotFoundError(f"expected package artifact not found: {artifact}")
    return final_directory


def finalize_distribution(
    output: Path,
    system_name: str,
    build_command: list[str],
    built_at_utc: str | None = None,
) -> Path:
    final_directory = distribution_directory(output.resolve(), system_name)
    if any(
        path.is_file() and path.suffix.lower() in DATABASE_SUFFIXES
        for path in final_directory.rglob("*")
    ):
        raise ValueError("database files must remain outside the packaged application")

    data_directory = final_directory / "data"
    if data_directory.exists() and (
        not data_directory.is_dir() or any(data_directory.iterdir())
    ):
        raise ValueError("package data directory must be empty")
    data_directory.mkdir(exist_ok=True)
    (final_directory / "runtime_manifest.json").write_text(
        json.dumps(
            build_runtime_manifest(build_command, built_at_utc),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return final_directory


def main(
    arguments: list[str] | None = None,
    run_command=subprocess.run,
    system_name: str | None = None,
) -> Path:
    parser = ArgumentParser(description="Build the anomaly diagnosis desktop package")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/time_series_anomaly_diagnosis"),
    )
    options = parser.parse_args(arguments)
    command = compose_build_command(options.output, options.clean)
    run_command(command, check=True, cwd=SERVICE_ROOT.parents[1])
    return finalize_distribution(
        options.output,
        system_name or platform.system(),
        command,
    )


if __name__ == "__main__":
    main()
