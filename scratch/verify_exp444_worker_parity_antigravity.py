#!/usr/bin/env python3
"""Label-free one-worker/11-worker parity smoke for Exp444 Task 7."""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
import sys

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"

_REPO_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_ANTIGRAVITY))

from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp444_train_only_stability_precision_gate_antigravity.runtime_antigravity import (
    canonical_results_bytes_antigravity,
    ordered_parallel_map_antigravity,
    production_worker_count_antigravity,
    sequential_reference_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.sources_antigravity import (
    load_candidate_values_antigravity,
    verify_frozen_sources_antigravity,
)


def main() -> int:
    snapshot = verify_frozen_sources_antigravity()
    try:
        inputs = load_candidate_values_antigravity(snapshot)
        if snapshot.label_access_count != 0 or snapshot.session is None or snapshot.session.label_access_count != 0:
            raise RuntimeError("label access was observed")
        sequential = sequential_reference_antigravity(inputs)
        parallel = ordered_parallel_map_antigravity(inputs, workers=production_worker_count_antigravity())
        sequential_bytes = canonical_results_bytes_antigravity(sequential)
        parallel_bytes = canonical_results_bytes_antigravity(parallel)
        if sequential_bytes != parallel_bytes:
            raise RuntimeError("worker parity mismatch")
        payload = {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "candidate_count": len(inputs),
            "workers": production_worker_count_antigravity(),
            "sequential_sha256": sha256(sequential_bytes).hexdigest(),
            "parallel_sha256": sha256(parallel_bytes).hexdigest(),
            "canonical_byte_equal": True,
            "label_access_count": snapshot.label_access_count,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    finally:
        if snapshot.session is not None:
            snapshot.session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
