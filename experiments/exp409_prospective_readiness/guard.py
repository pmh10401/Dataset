"""Fail-closed label and evaluation boundary for Exp409 prediction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
import sqlite3

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import DatasetBundle
from experiments.exp226_235_hard_alert_quality.preprocessing import (
    sanitize_series,
)
from experiments.exp403_408.prediction import find_forbidden_fields


FORBIDDEN_INPUT_NAMES = frozenset(
    {
        "exp408_variant_metrics.csv",
        "exp408_variant_candidate_evaluation.csv",
        "exp403_408_frozen_test_labels.csv",
    }
)
_TRUTH_TOKENS = frozenset(
    {
        "actual",
        "anomaly",
        "anomalies",
        "ground",
        "label",
        "labels",
        "target",
        "targets",
        "truth",
        "ytest",
    }
)
_EVALUATION_TOKENS = frozenset(
    {
        "accuracy",
        "auc",
        "f1",
        "fn",
        "fp",
        "metric",
        "metrics",
        "precision",
        "recall",
        "tp",
    }
)


def validate_label_blind_payload(
    value: object, path: str = "prediction"
) -> None:
    forbidden = find_forbidden_fields(value, path)
    if forbidden:
        raise ValueError(
            f"forbidden Exp409 prediction field: {forbidden[0]}"
        )


def reject_forbidden_inputs(paths: Sequence[Path]) -> None:
    for raw in paths:
        path = Path(raw)
        lowered_parts = tuple(part.lower() for part in path.parts)
        if path.name.lower() in FORBIDDEN_INPUT_NAMES:
            raise ValueError(f"Exp409 evaluation input is forbidden: {path.name}")
        if (
            "exp403_408_feature_ml_validation" in lowered_parts
            and "checkpoints" in lowered_parts
        ):
            raise ValueError("Exp403-408 checkpoint input is forbidden")


class LabelBlindSqlTrace:
    """Collect SQL statements and reject label-derived TEST access."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def __call__(self, statement: str) -> None:
        self.statements.append(str(statement))

    def assert_clean(self) -> None:
        for statement in self.statements:
            normalized = " ".join(statement.lower().split())
            tokens = set(re.findall(r"[a-z0-9]+", normalized))
            if "order by" in normalized and (
                tokens & (_TRUTH_TOKENS | _EVALUATION_TOKENS)
            ):
                raise ValueError("Exp409 SQL trace used TEST-derived ordering")
            if tokens & _TRUTH_TOKENS:
                raise ValueError("Exp409 SQL trace accessed TEST label truth")
            if "test" in tokens and tokens & _EVALUATION_TOKENS:
                raise ValueError("Exp409 SQL trace accessed TEST evaluation")


def load_values_bundle_label_blind(
    dataset: str,
    path: Path,
    *,
    trace: LabelBlindSqlTrace,
) -> DatasetBundle:
    """Load TRAIN/TEST values through a traced read-only connection."""
    resolved = Path(path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.set_trace_callback(trace)
    try:
        train = _load_split_values(connection, dataset, "TRAIN")
        test = _load_split_values(connection, dataset, "TEST")
    finally:
        connection.close()
    trace.assert_clean()
    return DatasetBundle(
        dataset=str(dataset),
        train_normal=train,
        test_values=test,
    )


def _load_split_values(
    connection: sqlite3.Connection,
    dataset: str,
    split: str,
) -> tuple[np.ndarray, ...]:
    rows = connection.execute(
        """
        SELECT i.values_blob
        FROM instances AS i
        JOIN datasets AS d ON d.id = i.dataset_id
        WHERE d.name = ? AND i.split = ?
        ORDER BY i.instance_index
        """,
        (str(dataset), str(split)),
    ).fetchall()
    if not rows:
        raise ValueError(f"no {split} rows for dataset={dataset}")
    return tuple(
        sanitize_series(np.frombuffer(row[0], dtype=np.float32))
        for row in rows
    )
