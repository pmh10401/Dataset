from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .config import EXPERIMENT, EXPERIMENT_ID, SOURCE
from .inputs import FrozenDatasetInputs, load_frozen_inputs


def _sorted(values: set[int]) -> list[int]:
    return sorted(map(int, values))


def route_dataset(inputs: FrozenDatasetInputs) -> dict[str, object]:
    promoted = inputs.standard & (inputs.mini1024 | inputs.mini4096)
    both = inputs.mini1024 & inputs.mini4096
    independent = inputs.ordinal | inputs.complexity | inputs.fourier | inputs.masked
    grade_s = promoted & both & independent
    grade_a = (promoted & both) - grade_s
    grade_b = promoted - both
    final_hard = inputs.hard | promoted
    final_standard = inputs.standard - promoted
    if (final_hard & final_standard) or (final_hard & inputs.priority) or (final_standard & inputs.priority):
        raise ValueError(f"final lane overlap: dataset={inputs.dataset}")
    if not promoted <= inputs.standard:
        raise ValueError(f"promotion outside baseline Standard: dataset={inputs.dataset}")
    if (grade_s | grade_a | grade_b) != promoted:
        raise ValueError(f"confidence coverage mismatch: dataset={inputs.dataset}")
    return {
        "experiment_id": EXPERIMENT_ID, "experiment": EXPERIMENT, "source": SOURCE,
        "dataset": inputs.dataset, "n_test": inputs.n_test,
        "baseline_hard_indices": _sorted(inputs.hard),
        "baseline_standard_indices": _sorted(inputs.standard),
        "baseline_priority_indices": _sorted(inputs.priority),
        "promoted_indices": _sorted(promoted),
        "final_hard_indices": _sorted(final_hard),
        "final_standard_indices": _sorted(final_standard),
        "final_priority_indices": _sorted(inputs.priority),
        "grade_s_indices": _sorted(grade_s), "grade_a_indices": _sorted(grade_a),
        "grade_b_indices": _sorted(grade_b),
        "mini1024_confirmation_indices": _sorted(promoted & inputs.mini1024),
        "mini4096_confirmation_indices": _sorted(promoted & inputs.mini4096),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "selected_from_prior_test_review": True, "retrospective": True,
        "deployment_eligible": False,
    }


def predict_all(output_root: Path, names: tuple[str, ...] | None = None) -> tuple[Path, dict[str, object]]:
    inputs = load_frozen_inputs()
    selected = tuple(sorted(inputs)) if names is None else tuple(names)
    rows = [route_dataset(inputs[name]) for name in selected]
    rows.sort(key=lambda row: str(row["dataset"]))
    data = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "frozen_predictions.jsonl"
    with tempfile.NamedTemporaryFile(dir=output_root, delete=False) as handle:
        temporary = Path(handle.name); handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    manifest = {
        "experiment_id": EXPERIMENT_ID, "datasets": len(rows),
        "prediction_sha256": hashlib.sha256(data).hexdigest(),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "selected_from_prior_test_review": True, "retrospective": True,
        "deployment_eligible": False,
    }
    (output_root / "prediction_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return path, manifest

