from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp411_near_repeat_qualification.contracts import TrainRun
from experiments.exp411_near_repeat_qualification.runner import (
    dataset_train_identity_sha256,
)

from .contracts import DatasetRepairAudit, RunRepairAudit


def repair_train_blob(
    dataset: str,
    instance_index: int,
    payload: bytes,
) -> tuple[TrainRun | None, RunRepairAudit]:
    """Repair one normal TRAIN ``<f4`` payload without changing its length."""
    if not isinstance(payload, bytes) or len(payload) % 4:
        raise ValueError("Exp414 TRAIN blob width is invalid")

    values = np.frombuffer(payload, dtype="<f4")
    finite = np.isfinite(values)
    missing_positions = [int(position) for position in np.flatnonzero(~finite)]
    raw_values_sha256 = sha256(payload).hexdigest()
    audit_base = {
        "dataset": str(dataset),
        "instance_index": int(instance_index),
        "raw_values_sha256": raw_values_sha256,
        "length": len(values),
        "nan_count": int(np.isnan(values).sum()),
        "posinf_count": int(np.isposinf(values).sum()),
        "neginf_count": int(np.isneginf(values).sum()),
        "missing_positions_sha256": sha256(
            canonical_json_bytes(missing_positions)
        ).hexdigest(),
    }

    if finite.all():
        repaired_payload = payload
        repair_method = "unchanged"
    elif not finite.any():
        return None, RunRepairAudit(
            repaired_values_sha256=None,
            repair_method="all_missing_abstain",
            **audit_base,
        )
    else:
        positions = np.arange(len(values), dtype=np.float64)
        repaired = values.astype(np.float64, copy=True)
        repaired[~finite] = np.interp(
            positions[~finite],
            positions[finite],
            values[finite].astype(np.float64),
        )
        repaired_payload = np.asarray(repaired, dtype="<f4").tobytes()
        repair_method = "linear_and_edge_nearest"

    repaired_values_sha256 = sha256(repaired_payload).hexdigest()
    repaired_values = np.frombuffer(repaired_payload, dtype="<f4")
    return (
        TrainRun(
            dataset=str(dataset),
            instance_index=int(instance_index),
            values=tuple(float(value) for value in repaired_values),
            values_sha256=repaired_values_sha256,
        ),
        RunRepairAudit(
            repaired_values_sha256=repaired_values_sha256,
            repair_method=repair_method,
            **audit_base,
        ),
    )


def build_dataset_repair_audit(
    dataset: str,
    runs: Sequence[TrainRun],
    audits: Sequence[RunRepairAudit],
) -> DatasetRepairAudit:
    """Bind raw and repaired TRAIN identities to a dataset repair decision."""
    audit_rows = tuple(audits)
    repaired_runs = tuple(runs)
    if not audit_rows:
        raise ValueError("Exp414 dataset repair audit is empty")
    if any(row.dataset != dataset for row in audit_rows) or any(
        run.dataset != dataset for run in repaired_runs
    ):
        raise ValueError("Exp414 dataset repair audit dataset mismatch")
    if len({row.instance_index for row in audit_rows}) != len(audit_rows):
        raise ValueError("Exp414 dataset repair audit indices are duplicated")
    if len({run.instance_index for run in repaired_runs}) != len(repaired_runs):
        raise ValueError("Exp414 repaired TRAIN indices are duplicated")

    all_missing_indices = {
        row.instance_index
        for row in audit_rows
        if row.repair_method == "all_missing_abstain"
    }
    expected_repaired_indices = {
        row.instance_index for row in audit_rows
    } - all_missing_indices
    if {run.instance_index for run in repaired_runs} != expected_repaired_indices:
        raise ValueError("Exp414 repaired TRAIN audit coverage is invalid")
    if any(not np.isfinite(run.values).all() for run in repaired_runs):
        raise ValueError("Exp414 repaired TRAIN values are non-finite")

    raw_identity_payload = [
        {
            "dataset": row.dataset,
            "instance_index": int(row.instance_index),
            "values_sha256": row.raw_values_sha256,
        }
        for row in sorted(audit_rows, key=lambda row: row.instance_index)
    ]
    raw_train_identity_sha256 = sha256(
        canonical_json_bytes(raw_identity_payload)
    ).hexdigest()
    if all_missing_indices:
        repaired_train_identity_sha256 = None
        status = "missing_support_abstain"
    else:
        repaired_train_identity_sha256 = dataset_train_identity_sha256(
            tuple(sorted(repaired_runs, key=lambda run: run.instance_index))
        )
        status = "ready"
    return DatasetRepairAudit(
        dataset=dataset,
        runs=audit_rows,
        raw_train_identity_sha256=raw_train_identity_sha256,
        repaired_train_identity_sha256=repaired_train_identity_sha256,
        status=status,
    )


__all__ = ["build_dataset_repair_audit", "repair_train_blob"]
