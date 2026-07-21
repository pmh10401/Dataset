"""Label-free Exp272 recovery-review routing and prediction freeze."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.exp239_248_diverse_score_sources.foundation import stable_json_hash

from .contracts import RecoveryLanes
from .qualification import _read_prediction_rows, _verify_membership, _variant_key


EXP266_PREDICTION = Path(
    "/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow/03_frozen_predictions.jsonl"
)
EXP266_MANIFEST = EXP266_PREDICTION.with_name("04_prediction_manifest.json")
EXP266_PREDICTION_SHA256 = "2f85a4a2c80570be6471b19f27878e392a5e55a6a116ed00372b5c1bb95ad246"
EXP266_MANIFEST_SHA256 = "6cdc2ac153409e4f9dec0dc78c7abe138a36fb1a7a2dded265305118eaaebf4e"

_BASELINE_FIELDS = (
    "hard_indices", "shadow_indices", "priority_indices", "standard_indices",
)
_RECOVERY_FIELDS = (
    "recovery_shadow_indices", "recovery_priority_indices", "data_quality_review_indices",
)
_FORBIDDEN = frozenset({
    "y_test", "label", "labels", "test_labels", "tp", "fp", "fn", "precision",
    "recall", "f1", "combined_f1", "true_indices",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _indices(values: Iterable[int], *, n_test: int, field: str) -> set[int]:
    result = set(values)
    if any(type(index) is not int or index < 0 or index >= n_test for index in result):
        raise ValueError(f"{field} contains an index outside TEST range")
    return result


def route_recovery_review(
    *,
    existing: RecoveryLanes,
    source_candidates: Mapping[str, Iterable[int]],
    quality_flags: Iterable[int],
    n_test: int,
) -> RecoveryLanes:
    """Preserve baseline lanes and route only outside-lane independent support."""
    if type(n_test) is not int or n_test < 0:
        raise ValueError("n_test must be a non-negative integer")
    baseline = set()
    for field in _BASELINE_FIELDS:
        baseline.update(_indices(getattr(existing, field), n_test=n_test, field=field))
    flags = _indices(quality_flags, n_test=n_test, field="quality_flags")
    support: dict[int, set[str]] = defaultdict(set)
    for group, values in source_candidates.items():
        if not isinstance(group, str) or not group:
            raise ValueError("evidence group names must be non-empty strings")
        for index in _indices(values, n_test=n_test, field=f"source_candidates[{group}]"):
            if index not in baseline:
                support[index].add(group)
    quality = {index for index in support if index in flags}
    recovery_shadow = {
        index for index, groups in support.items() if len(groups) >= 3 and index not in quality
    }
    recovery_priority = {
        index for index, groups in support.items() if len(groups) == 2 and index not in quality
    }
    return RecoveryLanes(
        hard_indices=existing.hard_indices,
        shadow_indices=existing.shadow_indices,
        priority_indices=existing.priority_indices,
        standard_indices=existing.standard_indices,
        recovery_shadow_indices=tuple(sorted(recovery_shadow)),
        recovery_priority_indices=tuple(sorted(recovery_priority)),
        data_quality_review_indices=tuple(sorted(quality)),
    )


def _validate_boundary(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _FORBIDDEN:
                raise ValueError(f"prediction contains evaluation field: {key}")
            if normalized in {"contains_test_labels", "contains_evaluation_metrics"} and child is not False:
                raise ValueError(f"prediction boundary flag must be false: {key}")
            _validate_boundary(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_boundary(child)


def _validate_row(row: Mapping[str, object]) -> None:
    _validate_boundary(row)
    dataset = row.get("dataset")
    n_test = row.get("n_test")
    n_train = row.get("n_train")
    if (
        not isinstance(dataset, str) or not dataset
        or type(n_test) is not int or n_test < 0
        or type(n_train) is not int or n_train < 0
    ):
        raise ValueError("prediction row requires dataset, n_train, and n_test")
    lanes: dict[str, set[int]] = {}
    for field in (*_BASELINE_FIELDS, *_RECOVERY_FIELDS):
        values = row.get(field)
        if not isinstance(values, list) or values != sorted(values) or len(values) != len(set(values)):
            raise ValueError(f"{field} must be a sorted unique list")
        lanes[field] = _indices(values, n_test=n_test, field=field)
    all_fields = (*_BASELINE_FIELDS, *_RECOVERY_FIELDS)
    for offset, field in enumerate(all_fields):
        for other in all_fields[offset + 1 :]:
            if lanes[field] & lanes[other]:
                raise ValueError(f"lane overlap: {field} and {other}")
    support_raw = row.get("support_by_index")
    if not isinstance(support_raw, dict):
        raise ValueError("support_by_index must be an object")
    support: dict[int, list[str]] = {}
    for raw_index, groups in support_raw.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError("support index is not an integer") from error
        _indices((index,), n_test=n_test, field="support_by_index")
        if not isinstance(groups, list) or groups != sorted(groups) or len(groups) != len(set(groups)):
            raise ValueError("support families must be sorted and unique")
        support[index] = groups
    quality_flags_raw = row.get("quality_flag_indices")
    if not isinstance(quality_flags_raw, list):
        raise ValueError("quality_flag_indices must be a list")
    quality_flags = _indices(quality_flags_raw, n_test=n_test, field="quality_flag_indices")
    expected_quality = {index for index in support if index in quality_flags}
    if lanes["data_quality_review_indices"] != expected_quality:
        raise ValueError("Data Quality Review routing mismatch")
    expected_shadow = {
        index for index, groups in support.items()
        if len(groups) >= 3 and index not in expected_quality
    }
    expected_priority = {
        index for index, groups in support.items()
        if len(groups) == 2 and index not in expected_quality
    }
    if lanes["recovery_shadow_indices"] != expected_shadow:
        raise ValueError("Recovery Shadow routing mismatch")
    if lanes["recovery_priority_indices"] != expected_priority:
        raise ValueError("Recovery Priority routing mismatch")
    one_source = row.get("one_source_sensitivity_indices")
    if not isinstance(one_source, list):
        raise ValueError("one_source_sensitivity_indices must be a list")
    one = _indices(one_source, n_test=n_test, field="one_source_sensitivity_indices")
    if one != {
        index for index, groups in support.items()
        if len(groups) == 1 and index not in expected_quality
    }:
        raise ValueError("one-source sensitivity mapping mismatch")
    if row.get("review_only") is not True or row.get("retrospective") is not True:
        raise ValueError("Exp272 rows must be retrospective review-only predictions")


def freeze_recovery_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    expected_datasets: Iterable[str],
    membership_sha256: str,
    input_sha256: Mapping[str, str],
) -> tuple[Path, Path]:
    """Write canonical label-free Exp272 rows and a self-consistent manifest."""
    expected = tuple(sorted(str(value) for value in expected_datasets))
    materialized = [dict(row) for row in rows]
    for row in materialized:
        _validate_row(row)
        if row.get("input_sha256") != dict(sorted(input_sha256.items())):
            raise ValueError("prediction row input hashes differ from freeze contract")
    names = [str(row["dataset"]) for row in materialized]
    if len(names) != len(set(names)) or tuple(sorted(names)) != expected:
        raise ValueError("Exp272 prediction population is incomplete or duplicated")
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in sorted(materialized, key=lambda item: str(item["dataset"])))
    prediction_hash = hashlib.sha256(payload).hexdigest()
    root = Path(output_dir)
    prediction_path = root / "exp272_frozen_predictions.jsonl"
    manifest_path = root / "exp272_prediction_manifest.json"
    _atomic_bytes(prediction_path, payload)
    manifest = {
        "schema_version": 1,
        "experiment_id": "experiment_272_recovery_review_selector",
        "datasets": len(expected),
        "dataset_list_sha256": stable_json_hash(expected),
        "rows": len(materialized),
        "prediction_sha256": prediction_hash,
        "membership_sha256": membership_sha256,
        "input_sha256": dict(sorted(input_sha256.items())),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "review_only": True,
        "retrospective": True,
    }
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    _atomic_bytes(manifest_path, _canonical_bytes(manifest) + b"\n")
    return prediction_path, manifest_path


def verify_recovery_prediction_hash(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_datasets: Iterable[str],
) -> bool:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Exp272 manifest must be an object")
    if manifest.get("manifest_sha256") != stable_json_hash({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }):
        raise ValueError("Exp272 manifest self-hash mismatch")
    payload = Path(prediction_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("prediction_sha256"):
        raise ValueError("Exp272 prediction SHA256 mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    expected = tuple(sorted(str(value) for value in expected_datasets))
    if tuple(str(row.get("dataset")) for row in rows) != expected:
        raise ValueError("Exp272 prediction dataset order or coverage mismatch")
    if (
        manifest.get("datasets") != len(expected)
        or manifest.get("rows") != len(rows)
        or manifest.get("dataset_list_sha256") != stable_json_hash(expected)
    ):
        raise ValueError("Exp272 manifest population mismatch")
    for row in rows:
        _validate_row(row)
    return True


def _load_pinned_exp266() -> tuple[object, dict[str, dict[str, Any]], dict[str, str]]:
    if _sha256(EXP266_PREDICTION) != EXP266_PREDICTION_SHA256:
        raise ValueError("pinned Exp266 prediction SHA256 mismatch")
    if _sha256(EXP266_MANIFEST) != EXP266_MANIFEST_SHA256:
        raise ValueError("pinned Exp266 manifest SHA256 mismatch")
    manifest = json.loads(EXP266_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("prediction_sha256") != EXP266_PREDICTION_SHA256:
        raise ValueError("pinned Exp266 manifest points to another prediction")
    rows = [json.loads(line) for line in EXP266_PREDICTION.read_text(encoding="utf-8").splitlines() if line]
    by_dataset = {str(row.get("dataset")): row for row in rows}
    if len(rows) != 1117 or len(by_dataset) != 1117:
        raise ValueError("pinned Exp266 prediction population mismatch")
    from experiments.exp266_fn_recovery_shadow.inputs import load_frozen_inputs

    with tempfile.TemporaryDirectory(prefix="exp272-exp266-audit-") as raw:
        frozen = load_frozen_inputs(output_root=Path(raw))
    if set(by_dataset) != set(frozen.population):
        raise ValueError("Exp266 row and frozen-input populations differ")
    return frozen, by_dataset, {
        "exp266_prediction": EXP266_PREDICTION_SHA256,
        "exp266_manifest": EXP266_MANIFEST_SHA256,
    }


def build_and_freeze_verified_recovery_predictions(
    new_prediction_path: Path,
    new_prediction_manifest_path: Path,
    membership_path: Path,
    *,
    cache_root: Path,
    output_dir: Path,
    source_runtime: Mapping[str, Mapping[str, object]] | None = None,
    expected_datasets: Iterable[str] | None = None,
) -> tuple[Path, Path]:
    """Build all 1,117 Exp272 rows from pinned existing and verified new evidence."""
    frozen, exp266_rows, input_hashes = _load_pinned_exp266()
    expected = (
        tuple(frozen.population)
        if expected_datasets is None
        else tuple(sorted(str(value) for value in expected_datasets))
    )
    if not set(expected).issubset(frozen.population):
        raise ValueError("requested recovery population is outside frozen Exp266")
    from .source_pipeline import verify_prediction_hash

    verify_prediction_hash(
        new_prediction_path, new_prediction_manifest_path,
        expected_datasets=expected, cache_root=cache_root, source_runtime=source_runtime,
    )
    new_rows, prediction_hash, prediction_manifest_hash = _read_prediction_rows(
        new_prediction_path, new_prediction_manifest_path, expected_datasets=expected,
    )
    membership = _verify_membership(membership_path, prediction_hash=prediction_hash)
    qualified_groups = membership.get("qualified_groups")
    if not isinstance(qualified_groups, list):
        raise ValueError("Exp272 source membership groups are invalid")
    group_by_member: dict[str, str] = {}
    for group in qualified_groups:
        if not isinstance(group, Mapping):
            raise ValueError("Exp272 source membership group is invalid")
        label = str(group["group"])
        for member in group["members"]:
            key = str(member)
            if key in group_by_member:
                raise ValueError("source variant belongs to multiple qualified groups")
            group_by_member[key] = label
    rows_by_dataset: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in new_rows:
        rows_by_dataset[str(row["dataset"])].append(row)
    output_rows: list[dict[str, object]] = []
    for dataset in expected:
        item = frozen.datasets[dataset]
        old_row = exp266_rows[dataset]
        baseline_expected = {
            "hard_indices": sorted(item.hard), "shadow_indices": sorted(item.shadow),
            "priority_indices": sorted(item.priority), "standard_indices": sorted(item.standard),
        }
        if any(old_row.get(field) != values for field, values in baseline_expected.items()):
            raise ValueError(f"pinned Exp266 baseline lane mismatch: dataset={dataset}")
        expected_support = {
            family: sorted(indices) for family, indices in item.evidence.items()
        }
        expected_availability = {
            family: family in item.available_groups for family in item.evidence
        }
        if (
            old_row.get("support_sources") != expected_support
            or old_row.get("source_availability") != expected_availability
            or old_row.get("quality_flag_indices") != sorted(item.quality_flags)
        ):
            raise ValueError(f"pinned Exp266 support contract mismatch: dataset={dataset}")
        source_candidates: dict[str, set[int]] = {
            family: set(indices)
            for family, indices in item.evidence.items()
            if family in item.available_groups
        }
        new_group_candidates: dict[str, set[int]] = defaultdict(set)
        eligibility: dict[str, bool] = {
            family: family in item.available_groups for family in item.evidence
        }
        for row in rows_by_dataset[dataset]:
            variant = _variant_key(row["source"], row["variant"])
            eligibility[variant] = bool(row.get("eligibility", False))
            group = group_by_member.get(variant)
            if group is not None:
                new_group_candidates[group].update(int(index) for index in row["candidate_indices"])
        for group, indices in new_group_candidates.items():
            if group in source_candidates:
                raise ValueError("new evidence group collides with an existing family")
            source_candidates[group] = indices
        existing = RecoveryLanes(
            hard_indices=tuple(item.hard), shadow_indices=tuple(item.shadow),
            priority_indices=tuple(item.priority), standard_indices=tuple(item.standard),
            recovery_shadow_indices=(), recovery_priority_indices=(),
        )
        routed = route_recovery_review(
            existing=existing, source_candidates=source_candidates,
            quality_flags=item.quality_flags, n_test=item.n_test,
        )
        baseline = set(item.hard | item.shadow | item.priority | item.standard)
        support: dict[int, list[str]] = defaultdict(list)
        for group, indices in source_candidates.items():
            for index in indices - baseline:
                support[index].append(group)
        support = {index: sorted(groups) for index, groups in sorted(support.items())}
        one_source = sorted(
            index for index, groups in support.items()
            if len(groups) == 1 and index not in item.quality_flags
        )
        config_hash = stable_json_hash({
            "routing_order": ["existing", "data_quality", "support_ge_3", "support_eq_2", "no_alert"],
            "recovery_shadow_min_groups": 3,
            "recovery_priority_groups": 2,
            "membership_sha256": membership["membership_sha256"],
        })
        row_input_hashes = {
            **input_hashes,
            "new_prediction": prediction_hash,
            "new_prediction_manifest": prediction_manifest_hash,
            "membership": str(membership["membership_sha256"]),
        }
        output_rows.append({
            "dataset": dataset, "n_train": item.n_train, "n_test": item.n_test,
            "hard_indices": list(routed.hard_indices),
            "shadow_indices": list(routed.shadow_indices),
            "priority_indices": list(routed.priority_indices),
            "standard_indices": list(routed.standard_indices),
            "recovery_shadow_indices": list(routed.recovery_shadow_indices),
            "recovery_priority_indices": list(routed.recovery_priority_indices),
            "data_quality_review_indices": list(routed.data_quality_review_indices),
            "quality_flag_indices": sorted(item.quality_flags),
            "one_source_sensitivity_indices": one_source,
            "support_by_index": {str(index): groups for index, groups in support.items()},
            "source_eligibility": dict(sorted(eligibility.items())),
            "input_sha256": dict(sorted(row_input_hashes.items())),
            "config_sha256": config_hash,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "review_only": True,
            "retrospective": True,
        })
    return freeze_recovery_predictions(
        output_rows, output_dir=output_dir, expected_datasets=expected,
        membership_sha256=str(membership["membership_sha256"]),
        input_sha256={
            **input_hashes, "new_prediction": prediction_hash,
            "new_prediction_manifest": prediction_manifest_hash,
            "membership": str(membership["membership_sha256"]),
        },
    )


__all__ = [
    "build_and_freeze_verified_recovery_predictions", "freeze_recovery_predictions",
    "route_recovery_review", "verify_recovery_prediction_hash",
]
