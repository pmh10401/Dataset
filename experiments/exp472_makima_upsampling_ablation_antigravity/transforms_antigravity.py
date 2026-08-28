"""Exact SciPy Makima alignment and label-free shape evidence for Exp472."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math

import numpy as np
import scipy.interpolate
import scipy.signal

from paper.equipment_anomaly_detection.contracts_paper import RawDataset


_SHAPE_SCHEMA_ANTIGRAVITY = "exp472-shape-diagnostic-antigravity.v2"
_TRACE_ORDER_ANTIGRAVITY = (("hard_sources", "TRAIN"), ("hard_sources", "TEST"), ("minirocket", "TRAIN"), ("minirocket", "TEST"), ("legacy_exp40", "TRAIN"), ("legacy_exp40", "TEST"), ("legacy_route", "TRAIN"), ("legacy_route", "TEST"))


def _canonical_json_bytes_exp472_antigravity(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _finite_row_exp472_antigravity(values: np.ndarray) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32).reshape(-1)
    if row.size == 0 or not np.isfinite(row).all():
        raise ValueError("Exp472 resampling requires finite nonempty input and positive target")
    return row


def _extrema_count_exp472_antigravity(values: np.ndarray) -> int:
    differences = np.diff(values)
    signs = np.sign(differences)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))


def resample_series_exp472_antigravity(values: np.ndarray, target_len: int) -> np.ndarray:
    """Return one finite exact-length row, changing Exp450 only for Makima upsampling."""
    row = _finite_row_exp472_antigravity(values)
    if target_len <= 0:
        raise ValueError("Exp472 resampling requires finite nonempty input and positive target")
    if row.size == target_len:
        return row.astype(np.float32, copy=False)
    if row.size == 1:
        return np.full(target_len, row[0], dtype=np.float32)
    if target_len < row.size:
        divisor = math.gcd(row.size, target_len)
        result = scipy.signal.resample_poly(
            row, target_len // divisor, row.size // divisor
        )
    else:
        source_x = np.linspace(0.0, 1.0, row.size, dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        result = scipy.interpolate.Akima1DInterpolator(
            source_x, row, method="makima", extrapolate=False
        )(target_x)
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (target_len,) or not np.isfinite(output).all():
        raise ValueError("Exp472 resampler output differs")
    if target_len > row.size and (output[0] != row[0] or output[-1] != row[-1]):
        raise ValueError("Exp472 Makima endpoints differ")
    return output


def shape_diagnostic_exp472_antigravity(
    source: np.ndarray, output: np.ndarray
) -> Mapping[str, object]:
    """Describe finite interpolation shape without clipping or selecting on it."""
    source_row = _finite_row_exp472_antigravity(source)
    output_row = _finite_row_exp472_antigravity(output)
    source_min = float(np.min(source_row))
    source_max = float(np.max(source_row))
    output_min = float(np.min(output_row))
    output_max = float(np.max(output_row))
    source_extrema = _extrema_count_exp472_antigravity(source_row)
    output_extrema = _extrema_count_exp472_antigravity(output_row)
    return {
        "source_min": source_min,
        "source_max": source_max,
        "output_min": output_min,
        "output_max": output_max,
        "overshoot": float(max(0.0, output_max - source_max)),
        "undershoot": float(max(0.0, source_min - output_min)),
        "source_extrema": source_extrema,
        "output_extrema": output_extrema,
        "added_extrema": max(0, output_extrema - source_extrema),
        "source_sha256": sha256(source_row.tobytes()).hexdigest(),
        "output_sha256": sha256(output_row.tobytes()).hexdigest(),
    }


@dataclass
class Exp472AlignmentTraceAntigravity:
    """Seals diagnostics from the actual Exp450-selected alignment calls."""
    dataset: str
    _records: list[dict[str, object]] = field(default_factory=list)
    _state: str = field(default="open", init=False)
    _diagnostic: Mapping[str, object] | None = field(default=None, init=False)
    _expected: tuple[tuple[str, str], ...] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("Exp472 trace dataset differs")

    def _require_open_antigravity(self) -> None:
        if self._state == "poisoned":
            raise ValueError("Exp472 trace is poisoned")
        if self._state != "open":
            raise ValueError("Exp472 trace is not open")

    def begin_scoring(self, expected: Sequence[tuple[str, str]]) -> None:
        """Accept Exp450's exact gate-derived plan before it can call an aligner."""
        self._require_open_antigravity()
        plan = tuple(expected)
        if any(
            type(pair) is not tuple or len(pair) != 2
            or pair not in _TRACE_ORDER_ANTIGRAVITY
            for pair in plan
        ) or any(
            _TRACE_ORDER_ANTIGRAVITY.index(plan[index])
            >= _TRACE_ORDER_ANTIGRAVITY.index(plan[index + 1])
            for index in range(len(plan) - 1)
        ):
            self._state = "poisoned"
            raise ValueError("Exp472 trace plan differs")
        self._expected = plan
        self._state = "scoring"

    def poison(self) -> None:
        """Make any failed score invocation permanently non-publishable."""
        if self._state != "sealed":
            self._state = "poisoned"

    def _require_scoring_antigravity(self) -> None:
        if self._state == "poisoned":
            raise ValueError("Exp472 trace is poisoned")
        if self._state != "scoring":
            raise ValueError("Exp472 trace is not scoring")

    @staticmethod
    def _row_evidence_antigravity(
        rows: Sequence[np.ndarray],
        shape: Sequence[int],
    ) -> dict[str, object]:
        offsets = [0]
        for row in rows:
            offsets.append(offsets[-1] + int(row.size))
        return {
            "dtype": "<f4",
            "order": "C",
            "shape": list(shape),
            "row_boundaries": offsets,
            "row_bytes_sha256": [sha256(row.tobytes()).hexdigest() for row in rows],
        }

    @staticmethod
    def _actual_input_row_antigravity(value: object) -> np.ndarray:
        row = np.asarray(value)
        if (
            row.dtype != np.dtype("<f4") or row.ndim != 1
            or not row.flags.c_contiguous or row.size == 0 or not np.isfinite(row).all()
        ):
            raise ValueError("Exp472 trace input differs")
        return row

    def observe(self, rows: Sequence[np.ndarray], target_len: int, output: np.ndarray, phase_tag: str, source_tag: str) -> None:
        """Passively bind exactly one already-computed Exp450 alignment result."""
        self._require_scoring_antigravity()
        try:
            if (phase_tag, source_tag) not in _TRACE_ORDER_ANTIGRAVITY:
                raise ValueError("Exp472 trace tag differs")
            if self._records and _TRACE_ORDER_ANTIGRAVITY.index((phase_tag, source_tag)) <= _TRACE_ORDER_ANTIGRAVITY.index((self._records[-1]["phase_tag"], self._records[-1]["source_tag"])):
                raise ValueError("Exp472 trace order differs")
            if type(target_len) is not int or target_len <= 0:
                raise ValueError("Exp472 trace target differs")
            materialized = tuple(self._actual_input_row_antigravity(row) for row in rows)
            array = np.asarray(output)
            if array.dtype != np.dtype("float32") or array.shape != (len(materialized), target_len) or not np.isfinite(array).all():
                raise ValueError("Exp472 trace output differs")
            output_rows = tuple(array[index] for index in range(array.shape[0]))
            input_evidence = self._row_evidence_antigravity(materialized, (len(materialized),))
            output_evidence = self._row_evidence_antigravity(output_rows, array.shape)
            row_details = [shape_diagnostic_exp472_antigravity(row, array[index]) for index, row in enumerate(materialized)]
            makima_pairs = [(index, item) for index, (row, item) in enumerate(zip(materialized, row_details, strict=True)) if 1 < row.size < target_len]
            bare = {"ordinal": len(self._records), "phase_tag": phase_tag, "source_tag": source_tag, "target_len": target_len, "row_count": len(materialized), "input_evidence": input_evidence, "input_sha256": sha256(_canonical_json_bytes_exp472_antigravity(input_evidence)).hexdigest(), "output_evidence": output_evidence, "output_sha256": sha256(_canonical_json_bytes_exp472_antigravity(output_evidence)).hexdigest(), "singleton_rows": sum(row.size == 1 for row in materialized), "equal_rows": sum(row.size == target_len for row in materialized), "polyphase_downsample_rows": sum(row.size > target_len for row in materialized), "makima_rows": len(makima_pairs), "makima_row_indices": [index for index, _item in makima_pairs], "overshoot": float(sum(item["overshoot"] for _index, item in makima_pairs)), "undershoot": float(sum(item["undershoot"] for _index, item in makima_pairs)), "source_extrema": sum(item["source_extrema"] for _index, item in makima_pairs), "output_extrema": sum(item["output_extrema"] for _index, item in makima_pairs), "added_extrema": sum(item["added_extrema"] for _index, item in makima_pairs)}
            self._records.append({**bare, "invocation_sha256": sha256(_canonical_json_bytes_exp472_antigravity(bare)).hexdigest()})
        except Exception:
            self.poison()
            raise

    def mark_score_completed(self) -> None:
        """Close the observer boundary only after the exact Exp450 score returns."""
        self._require_scoring_antigravity()
        observed = tuple(
            (record["phase_tag"], record["source_tag"])
            for record in self._records
        )
        if observed != self._expected:
            self.poison()
            raise ValueError("Exp472 trace calls differ")
        self._state = "score_completed"

    def seal(self) -> Mapping[str, object]:
        if self._state == "poisoned":
            raise ValueError("Exp472 trace is poisoned")
        if self._state == "scoring":
            raise ValueError("Exp472 trace score is incomplete")
        if self._state != "score_completed":
            raise ValueError("Exp472 trace is not open")
        records = tuple(self._records)
        slots = {(r["source_tag"], index) for r in records for index in r["makima_row_indices"]}
        aggregate = {"invocation_count": len(records), "row_invocation_count": sum(int(r["row_count"]) for r in records), "makima_row_invocation_count": sum(int(r["makima_rows"]) for r in records), "upsampled_rows": len(slots)}
        bare = {"schema": "exp472-shape-diagnostic-antigravity.v3", "dataset": self.dataset, "trace_state": "sealed", "alignment_status": "completed", **aggregate, "invocations": records, "invocation_order_sha256": sha256(_canonical_json_bytes_exp472_antigravity([(r["phase_tag"], r["source_tag"]) for r in records])).hexdigest(), "invocation_aggregate_sha256": sha256(_canonical_json_bytes_exp472_antigravity(aggregate)).hexdigest()}
        self._state = "sealed"
        self._diagnostic = {**bare, "diagnostic_sha256": sha256(_canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}
        return self._diagnostic

    def sealed_diagnostic(self) -> Mapping[str, object]:
        """Return only a trace sealed by the exact completed score invocation."""
        if self._state != "sealed" or self._diagnostic is None:
            raise ValueError("Exp472 trace is not sealed")
        return self._diagnostic


def align_rows_exp472_antigravity(
    rows: Sequence[np.ndarray],
    target_len: int,
    diagnostics: MutableSequence[Mapping[str, object]] | None = None,
) -> np.ndarray:
    """Align rows in source order and optionally append one diagnostic per row."""
    if target_len <= 0:
        raise ValueError("Exp472 resampling requires finite nonempty input and positive target")
    materialized_rows = tuple(rows)
    if not materialized_rows:
        return np.empty((0, target_len), dtype=np.float32)
    output_rows = []
    for row in materialized_rows:
        output = resample_series_exp472_antigravity(row, target_len)
        output_rows.append(output)
        if diagnostics is not None:
            diagnostics.append(shape_diagnostic_exp472_antigravity(row, output))
    return np.stack(output_rows).astype(np.float32, copy=False)


def _validated_raw_exp472_antigravity(values: RawDataset) -> RawDataset:
    if (
        not isinstance(values, RawDataset)
        or type(values.dataset) is not str
        or not values.dataset
        or type(values.metadata_length) is not int
        or values.metadata_length <= 0
    ):
        raise ValueError("Exp472 requires a valid label-free raw dataset")
    for indices, blobs, rows in (
        (values.train_indices, values.train_blobs, values.train_rows),
        (values.test_indices, values.test_blobs, values.test_rows),
    ):
        if (
            len(indices) != len(blobs)
            or len(blobs) != len(rows)
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise ValueError("Exp472 raw dataset identity differs")
        for blob, row in zip(blobs, rows, strict=True):
            array = np.asarray(row)
            if (
                not isinstance(blob, bytes)
                or array.dtype != np.dtype("<f4")
                or array.ndim != 1
                or array.size == 0
                or array.tobytes() != blob
            ):
                raise ValueError("Exp472 raw dataset identity differs")
    return values


def _has_only_finite_rows_exp472_antigravity(raw: RawDataset) -> bool:
    """Mirror Exp450's dataset-wide abstention gate without invoking alignment."""
    return all(
        np.isfinite(np.asarray(row, dtype=np.float32)).all()
        for row in (*raw.train_rows, *raw.test_rows)
    )


def _empty_shape_split_exp472_antigravity(row_count: int) -> Mapping[str, object]:
    """Record an uninvoked split while retaining its raw cardinality."""
    if type(row_count) is not int or row_count < 0:
        raise ValueError("Exp472 shape row count differs")
    return {
        "row_count": row_count,
        "aligned_row_count": 0,
        "upsampled_rows": 0,
        "overshoot": 0.0,
        "undershoot": 0.0,
        "source_extrema": 0,
        "output_extrema": 0,
        "added_extrema": 0,
        "rows_sha256": sha256(_canonical_json_bytes_exp472_antigravity([])).hexdigest(),
    }


def _split_shape_diagnostic_exp472_antigravity(
    rows: Sequence[np.ndarray], target_len: int
) -> Mapping[str, object]:
    diagnostics: list[Mapping[str, object]] = []
    for row in rows:
        output = resample_series_exp472_antigravity(row, target_len)
        diagnostics.append(shape_diagnostic_exp472_antigravity(row, output))
    if not diagnostics:
        return _empty_shape_split_exp472_antigravity(0)
    return {
        "row_count": len(diagnostics),
        "aligned_row_count": len(diagnostics),
        "upsampled_rows": sum(1 for row in rows if 1 < np.asarray(row).size < target_len),
        "overshoot": float(sum(float(item["overshoot"]) for item in diagnostics)),
        "undershoot": float(sum(float(item["undershoot"]) for item in diagnostics)),
        "source_extrema": sum(int(item["source_extrema"]) for item in diagnostics),
        "output_extrema": sum(int(item["output_extrema"]) for item in diagnostics),
        "added_extrema": sum(int(item["added_extrema"]) for item in diagnostics),
        "rows_sha256": sha256(
            _canonical_json_bytes_exp472_antigravity(diagnostics)
        ).hexdigest(),
    }


def count_upsampled_rows_exp472_antigravity(values: RawDataset) -> int:
    """Count only rows for which condition E actually invokes Makima."""
    raw = _validated_raw_exp472_antigravity(values)
    if not _has_only_finite_rows_exp472_antigravity(raw):
        return 0
    return sum(
        1
        for row in (*raw.train_rows, *raw.test_rows)
        if 1 < np.asarray(row).size < raw.metadata_length
    )


def dataset_shape_diagnostic_exp472_antigravity(values: RawDataset) -> Mapping[str, object]:
    """Aggregate all train/test shape diagnostics without reading labels."""
    raw = _validated_raw_exp472_antigravity(values)
    finite = _has_only_finite_rows_exp472_antigravity(raw)
    if finite:
        train = _split_shape_diagnostic_exp472_antigravity(
            raw.train_rows, raw.metadata_length
        )
        test = _split_shape_diagnostic_exp472_antigravity(raw.test_rows, raw.metadata_length)
        alignment_status, alignment_reason = "completed", None
    else:
        train = _empty_shape_split_exp472_antigravity(len(raw.train_rows))
        test = _empty_shape_split_exp472_antigravity(len(raw.test_rows))
        alignment_status, alignment_reason = "not_invoked", "nonfinite_input"
    bare = {
        "schema": _SHAPE_SCHEMA_ANTIGRAVITY,
        "dataset": raw.dataset,
        "target_len": raw.metadata_length,
        "row_count": int(train["row_count"]) + int(test["row_count"]),
        "aligned_row_count": int(train["aligned_row_count"]) + int(test["aligned_row_count"]),
        "upsampled_rows": int(train["upsampled_rows"]) + int(test["upsampled_rows"]),
        "alignment_status": alignment_status,
        "alignment_reason": alignment_reason,
        "train": train,
        "test": test,
    }
    return {
        **bare,
        "diagnostic_sha256": sha256(
            _canonical_json_bytes_exp472_antigravity(bare)
        ).hexdigest(),
    }


def test_identity_exp472_antigravity(values: RawDataset) -> Mapping[str, object]:
    """Return a deterministic TEST-only value identity for the E score shard."""
    raw = _validated_raw_exp472_antigravity(values)
    bare = {
        "dataset": raw.dataset,
        "series_length": raw.metadata_length,
        "split": "TEST",
        "indices": list(raw.test_indices),
        "values_blob_sha256": [
            sha256(blob).hexdigest() for blob in raw.test_blobs
        ],
    }
    return {
        **bare,
        "row_digest": sha256(_canonical_json_bytes_exp472_antigravity(bare)).hexdigest(),
        "count": len(raw.test_rows),
    }


__all__ = [
    "Exp472AlignmentTraceAntigravity",
    "align_rows_exp472_antigravity",
    "count_upsampled_rows_exp472_antigravity",
    "dataset_shape_diagnostic_exp472_antigravity",
    "resample_series_exp472_antigravity",
    "shape_diagnostic_exp472_antigravity",
    "test_identity_exp472_antigravity",
]
