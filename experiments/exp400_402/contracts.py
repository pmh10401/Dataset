from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1
COLLECTION_MODE = "retrospective_replay"
POLICY_VERSION = "exp397-review-ranking-replay-v1"
ALPHAS = (0.005, 0.01, 0.02, 0.05)
ALPHA_KEYS = tuple(str(alpha) for alpha in ALPHAS)
VISIBLE_VARIANTS = ("exp292_baseline", "tier_preserving_grammar")
CONTROL_VARIANTS = ("additive_grammar_family", "grammar_only_control")
EXPECTED_DATASETS = 1117
EXPECTED_STANDARD_CANDIDATES = 806
EXPECTED_AUTOMATIC_HARD = 2218
EXPECTED_PRIORITY_REVIEW = 31

FORBIDDEN_FIELDS = frozenset(
    {
        "y_test",
        "true_label",
        "is_true_anomaly",
        "actual_anomaly_positions",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "combined_f1",
    }
)

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "collection_mode",
        "policy_version",
        "run_id",
        "dataset",
        "n_test",
        "automatic_hard_count",
        "priority_review_count",
        "standard_review_count",
        "candidates",
        "contains_test_labels",
        "contains_evaluation_metrics",
        "automatic_policy_changed",
        "retrospective",
        "prospective",
        "prediction_sha256",
    }
)

CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_index",
        "exp292_tier",
        "exp292_rank",
        "tier_preserving_rank",
        "control_rank",
        "grammar_support_count",
        "p_values",
    }
)


def _canonical_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def prediction_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prediction_rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    data = "".join(canonical_json(row) + "\n" for row in rows)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _scan_forbidden(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_FIELDS or lowered.endswith(
                ("_tp", "_fp", "_fn", "_f1", "_precision", "_recall")
            ):
                raise ValueError(f"forbidden field: {path}.{key}")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def _exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    path: str,
) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"unknown field: {path}.{sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing field: {path}.{sorted(missing)[0]}")


def _integer(value: object, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _finite(value: object, path: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{path} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _alpha_integer_map(value: object, path: str, minimum: int) -> dict[str, int]:
    if type(value) is not dict or set(value) != set(ALPHA_KEYS):
        raise ValueError(f"{path} must contain exactly the four alpha keys")
    return {
        key: _integer(value[key], f"{path}.{key}", minimum=minimum)
        for key in ALPHA_KEYS
    }


def _validate_candidate(
    raw: object,
    *,
    run_id: str,
    n_test: int,
) -> dict[str, object]:
    if type(raw) is not dict:
        raise ValueError("candidate must be an object")
    _exact_fields(raw, CANDIDATE_FIELDS, "candidate")
    index = _integer(raw["candidate_index"], "candidate_index")
    if index >= n_test:
        raise ValueError("candidate index outside TEST range")
    if raw["candidate_id"] != f"{run_id}:{index}":
        raise ValueError("candidate_id mismatch")
    tier = _integer(raw["exp292_tier"], "exp292_tier", minimum=1)
    if tier not in {1, 2, 3}:
        raise ValueError("exp292_tier must be 1, 2, or 3")
    _integer(raw["exp292_rank"], "exp292_rank", minimum=1)
    _alpha_integer_map(
        raw["tier_preserving_rank"],
        "tier_preserving_rank alpha",
        minimum=1,
    )
    control_rank = raw["control_rank"]
    if type(control_rank) is not dict or set(control_rank) != set(CONTROL_VARIANTS):
        raise ValueError("control_rank must contain both research controls")
    for variant in CONTROL_VARIANTS:
        _alpha_integer_map(
            control_rank[variant],
            f"control_rank.{variant} alpha",
            minimum=1,
        )
    support = _alpha_integer_map(
        raw["grammar_support_count"],
        "grammar_support_count alpha",
        minimum=0,
    )
    if any(value > 3 for value in support.values()):
        raise ValueError("grammar support count must be between 0 and 3")
    p_values = raw["p_values"]
    expected_scores = {"transition", "dwell", "recovery", "composite"}
    if type(p_values) is not dict or set(p_values) != expected_scores:
        raise ValueError("p_values must contain the four grammar scores")
    for name in expected_scores:
        value = _finite(p_values[name], f"p_values.{name}")
        if value < 0.0 or value > 1.0:
            raise ValueError("p-value must be between 0 and 1")
    return dict(raw)


def validate_shadow_row(raw: Mapping[str, object]) -> dict[str, object]:
    if type(raw) is not dict:
        raise ValueError("prediction row must be an object")
    _scan_forbidden(raw)
    _exact_fields(raw, TOP_LEVEL_FIELDS, "prediction")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if raw["collection_mode"] != COLLECTION_MODE:
        raise ValueError("unsupported collection_mode")
    if raw["policy_version"] != POLICY_VERSION:
        raise ValueError("unsupported policy_version")
    run_id = raw["run_id"]
    dataset = raw["dataset"]
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset must be a non-empty string")
    n_test = _integer(raw["n_test"], "n_test", minimum=1)
    _integer(raw["automatic_hard_count"], "automatic_hard_count")
    _integer(raw["priority_review_count"], "priority_review_count")
    standard_count = _integer(
        raw["standard_review_count"],
        "standard_review_count",
    )
    candidates = raw["candidates"]
    if type(candidates) is not list:
        raise ValueError("candidates must be a list")
    validated = [
        _validate_candidate(candidate, run_id=run_id, n_test=n_test)
        for candidate in candidates
    ]
    indices = [int(candidate["candidate_index"]) for candidate in validated]
    if len(indices) != len(set(indices)):
        raise ValueError("candidate index duplicated")
    if len(validated) != standard_count:
        raise ValueError("standard review candidate count mismatch")
    flags = {
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_policy_changed": False,
        "retrospective": True,
        "prospective": False,
    }
    for field, expected in flags.items():
        if raw[field] is not expected:
            raise ValueError(f"{field} must be {expected}")
    stated_hash = raw["prediction_sha256"]
    if type(stated_hash) is not str or len(stated_hash) != 64:
        raise ValueError("prediction SHA256 is invalid")
    unsigned: dict[str, Any] = dict(raw)
    unsigned.pop("prediction_sha256")
    if prediction_sha256(unsigned) != stated_hash:
        raise ValueError("prediction SHA256 mismatch")
    return dict(raw)
