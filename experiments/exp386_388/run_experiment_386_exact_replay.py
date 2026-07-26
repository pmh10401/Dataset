from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp386_388.historical_runtime import (
    ORIGINAL_EVIDENCE,
    ORIGINAL_MANIFEST,
    OUTPUT_ROOT,
    alpha_output_root,
    run_historical_alpha,
    sha256_file,
)


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = run_historical_alpha(0.005)
    target = alpha_output_root(0.005)
    actual = sha256_file(target / "exp357_frozen_evidence.jsonl")
    expected = sha256_file(ORIGINAL_EVIDENCE)
    manifest_expected = json.loads(ORIGINAL_MANIFEST.read_text())["prediction_sha256"]
    exact = actual == expected == manifest_expected
    report = {
        "experiment_id": "experiment_386_exact_existing_evidence_replay",
        "alpha": 0.005,
        "exact_prediction_match": exact,
        "expected_prediction_sha256": expected,
        "actual_prediction_sha256": actual,
        "runtime": result,
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT_ROOT / "exp386_exact_replay_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "exp386_status.json").write_text(
        json.dumps(
            {
                "id": "experiment_386_exact_existing_evidence_replay",
                "status": "completed" if exact else "failed",
                "phase": "completed" if exact else "exact_replay_mismatch",
                "datasets_done": 1117,
                "datasets_total": 1117,
                "progress_percent": 100.0,
                "errors": 0 if exact else 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("Progress: [1117/1117] rows=1117 dataset=exact_replay errors=" + ("0" if exact else "1"))
    if not exact:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
