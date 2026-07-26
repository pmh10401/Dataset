from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


HISTORICAL_COMMIT = "930b8ce8927d18a0101c647d26e21c2cd20be800"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
ORIGINAL_ROOT = DATA_ROOT / "exp356_359_operational_confidence_tiers"
ORIGINAL_CONTRACT = ORIGINAL_ROOT / "00_exp356_baseline_contract.json"
ORIGINAL_EVIDENCE = ORIGINAL_ROOT / "exp357_frozen_evidence.jsonl"
ORIGINAL_MANIFEST = ORIGINAL_ROOT / "exp357_prediction_manifest.json"
OUTPUT_ROOT = DATA_ROOT / "exp386_388_existing_evidence_alpha_sweep"
ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):.3f}".replace(".", "p")


def alpha_output_root(alpha: float, root: Path = OUTPUT_ROOT) -> Path:
    return Path(root) / f"alpha_{alpha_tag(alpha)}"


def _source_hash_contract() -> dict[str, str]:
    contract = json.loads(ORIGINAL_CONTRACT.read_text(encoding="utf-8"))
    values = contract["full_run_gate"]["source_sha256"]
    if not isinstance(values, dict) or not values:
        raise ValueError("original source SHA256 contract is missing")
    return {str(path): str(digest) for path, digest in values.items()}


def extract_and_verify_historical_runtime(destination: Path) -> dict[str, str]:
    source_hashes = _source_hash_contract()
    paths = sorted(source_hashes)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", HISTORICAL_COMMIT, "--", *paths],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")
    mismatches = {
        relative: {
            "expected": expected,
            "actual": sha256_file(Path(destination) / relative),
        }
        for relative, expected in source_hashes.items()
        if not (Path(destination) / relative).is_file()
        or sha256_file(Path(destination) / relative) != expected
    }
    if mismatches:
        raise ValueError(f"historical source SHA256 mismatch: {mismatches}")
    return source_hashes


def run_historical_alpha(
    alpha: float,
    *,
    output_root: Path = OUTPUT_ROOT,
    workers: int = 7,
    progress_offset: int = 0,
    progress_total: int = 1117,
    dataset_limit: int | None = None,
) -> dict[str, object]:
    if float(alpha) not in ALPHAS:
        raise ValueError(f"alpha is not preregistered: {alpha}")
    target = alpha_output_root(alpha, output_root)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp386_legacy_") as temporary:
        runtime_root = Path(temporary)
        source_hashes = extract_and_verify_historical_runtime(runtime_root)
        code = """
import importlib
from pathlib import Path
from experiments.exp356_359_operational_confidence_tiers import config
config.EVIDENCE_ALPHA = __ALPHA__
config.OUTPUT_ROOT = Path(__OUTPUT__)
pipeline = importlib.import_module(
    "experiments.exp356_359_operational_confidence_tiers.pipeline"
)
pipeline.verify_full_run_gate = lambda: {
    "historical_source_commit": __COMMIT__,
    "verified_by_outer_runtime": True,
}
pipeline.run_phase(
    "357-predict",
    output_root=Path(__OUTPUT__),
    workers=__WORKERS__,
    dataset_limit=__DATASET_LIMIT__,
)
pipeline.run_phase("358-all", output_root=Path(__OUTPUT__))
pipeline.run_phase("359-all", output_root=Path(__OUTPUT__))
""".replace("__ALPHA__", repr(float(alpha))).replace(
            "__OUTPUT__", repr(str(target))
        ).replace("__COMMIT__", repr(HISTORICAL_COMMIT)).replace(
            "__WORKERS__", repr(int(workers))
        ).replace(
            "__DATASET_LIMIT__", repr(dataset_limit)
        )
        log_path = target / "historical_replay.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=runtime_root,
                env={
                    **os.environ,
                    "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                    "PYTHONPATH": str(runtime_root),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            last_done = -1
            while process.poll() is None:
                status_path = target / "exp357_status.json"
                if status_path.is_file():
                    try:
                        status = json.loads(status_path.read_text())
                        done = int(status.get("datasets_done") or 0)
                        if done != last_done:
                            print(
                                f"Progress: [{progress_offset + done}/{progress_total}] "
                                f"rows={progress_offset + done} "
                                f"dataset={status.get('current_dataset') or 'alpha_' + alpha_tag(alpha)} "
                                f"errors={1 if status.get('error') else 0}",
                                flush=True,
                            )
                            last_done = done
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                time.sleep(2.0)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"historical alpha replay failed: alpha={alpha}, log={log_path}"
            )
    prediction = target / "exp357_frozen_evidence.jsonl"
    manifest = target / "exp357_prediction_manifest.json"
    if not prediction.is_file() or not manifest.is_file():
        raise ValueError("historical replay did not produce Exp357 predictions")
    return {
        "alpha": float(alpha),
        "alpha_tag": alpha_tag(alpha),
        "output_root": str(target),
        "prediction_sha256": sha256_file(prediction),
        "manifest_sha256": sha256_file(manifest),
        "historical_commit": HISTORICAL_COMMIT,
        "verified_source_files": len(source_hashes),
        "dataset_limit": dataset_limit,
        "log": str(log_path),
    }
