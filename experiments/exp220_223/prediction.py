from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from experiments.exp220_223.policy import load_verified_policy, sha256_file
from experiments.exp220_223.storage import load_batch_instances, load_batch_metadata


FORBIDDEN_PREDICTION_FIELDS = {
    "label",
    "labels",
    "y_test",
    "test_labels",
    "test_position",
    "test_length",
    "tp",
    "fp",
    "fn",
    "f1",
    "precision",
    "recall",
    "outcome",
    "outcomes",
}
EXP137_FIELDS = ("exp137_hard", "exp137_standard", "exp137_priority")
SOURCE_NAMES = (
    "temporal_random_kernel",
    "spectral_image",
    "texture_image",
    "interval_statistical",
)


def _validate_prediction_row(row: Mapping[str, object]) -> None:
    forbidden = FORBIDDEN_PREDICTION_FIELDS & {str(key).lower() for key in row}
    if forbidden:
        raise ValueError(f"evaluation field in prediction: {sorted(forbidden)}")
    common = {
        "batch_id",
        "instance_id",
        "profile_id",
        "policy_hash",
        *EXP137_FIELDS,
    }
    if "exp224_shadow" in row:
        required = common | {
            "temporal_p",
            "canonical_p",
            "selected_alpha",
            "exp137_excluded",
            "exp224_shadow",
        }
    else:
        required = common | {
            "exp190_high",
            "exp190_standard",
            "exp195_shadow",
            "source_support_count",
            "canonical_knn_p",
            "exp195_zero_train_guard",
        }
    missing = required - set(row)
    if missing:
        raise ValueError(f"prediction fields missing: {sorted(missing)}")


def route_instance(
    *,
    batch_id: str,
    instance_id: str,
    profile_id: str,
    policy_hash: str,
    evidence: Mapping[str, object],
    conservative_alpha: float,
) -> dict[str, object]:
    exp137 = [bool(evidence.get(field, False)) for field in EXP137_FIELDS]
    if sum(exp137) > 1:
        raise ValueError(f"Exp137 lanes overlap: instance_id={instance_id}")
    source_pvalues = evidence.get("source_pvalues")
    if not isinstance(source_pvalues, Mapping):
        raise ValueError(f"source_pvalues missing: instance_id={instance_id}")
    missing_sources = set(SOURCE_NAMES) - set(source_pvalues)
    if missing_sources:
        raise ValueError(
            f"source p-values missing: instance_id={instance_id} sources={sorted(missing_sources)}"
        )
    pvalues = {name: float(source_pvalues[name]) for name in SOURCE_NAMES}
    canonical_p = float(evidence["canonical_knn_p"])
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pvalues.values()):
        raise ValueError(f"invalid source p-value: instance_id={instance_id}")
    if not math.isfinite(canonical_p) or not 0.0 <= canonical_p <= 1.0:
        raise ValueError(f"invalid canonical p-value: instance_id={instance_id}")
    support = sum(value <= conservative_alpha for value in pvalues.values())
    excluded = any(exp137)
    conservative = support >= 2 and not excluded
    high = conservative and canonical_p <= 0.005
    standard = conservative and not high
    guard = bool(evidence["exp195_zero_train_guard"])
    row: dict[str, object] = {
        "batch_id": batch_id,
        "instance_id": instance_id,
        "profile_id": profile_id,
        "policy_hash": policy_hash,
        "exp137_hard": exp137[0],
        "exp137_standard": exp137[1],
        "exp137_priority": exp137[2],
        "exp190_high": high,
        "exp190_standard": standard,
        "exp195_shadow": high and guard,
        "source_support_count": support,
        "source_pvalues": pvalues,
        "canonical_knn_p": canonical_p,
        "exp195_zero_train_guard": guard,
        "autonomous_promotion": False,
    }
    _validate_prediction_row(row)
    return row


def freeze_shadow_predictions(
    *,
    rows: Iterable[Mapping[str, object]],
    batch_id: str,
    policy_sha256: str,
    source_sha256: str,
    output_dir: Path,
) -> dict[str, object]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row.get("instance_id", "")), str(row.get("profile_id", ""))),
    )
    seen: set[str] = set()
    for row in ordered:
        _validate_prediction_row(row)
        if row["batch_id"] != batch_id or row["policy_hash"] != policy_sha256:
            raise ValueError("prediction batch/policy contract mismatch")
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate prediction instance_id: {instance_id}")
        seen.add(instance_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{batch_id}.shadow_predictions.jsonl"
    manifest_path = output_dir / f"{batch_id}.prediction_manifest.json"
    if prediction_path.exists() or manifest_path.exists():
        raise FileExistsError(f"frozen prediction already exists: batch_id={batch_id}")
    with prediction_path.open("x", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    manifest: dict[str, object] = {
        "experiment_id": "experiment_222_prospective_shadow_run",
        "batch_id": batch_id,
        "policy_sha256": policy_sha256,
        "source_sha256": source_sha256,
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_rows": len(ordered),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "prediction_path": str(prediction_path),
        "manifest_path": str(manifest_path),
    }


def verify_prediction_manifest(
    prediction_path: Path, manifest_path: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    prediction_path = Path(prediction_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual_hash = sha256_file(prediction_path)
    if actual_hash != manifest.get("prediction_sha256"):
        raise ValueError(
            "prediction hash mismatch: "
            f"expected={manifest.get('prediction_sha256')} actual={actual_hash}"
        )
    rows: list[dict[str, object]] = []
    with prediction_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid prediction JSONL line={line_number}") from exc
            _validate_prediction_row(row)
            rows.append(row)
    if len(rows) != int(manifest.get("prediction_rows", -1)):
        raise ValueError("prediction row count mismatch")
    ids = [str(row["instance_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prediction instance_id")
    if any(row["batch_id"] != manifest["batch_id"] for row in rows):
        raise ValueError("prediction batch mismatch")
    if any(row["policy_hash"] != manifest["policy_sha256"] for row in rows):
        raise ValueError("prediction policy mismatch")
    return manifest, rows


def predict_batch(
    *,
    db_path: Path,
    batch_id: str,
    policy_path: Path,
    output_dir: Path,
    evidence_provider: Callable[
        [list[dict[str, object]], Mapping[str, object]],
        Mapping[str, Mapping[str, object]],
    ],
) -> dict[str, object]:
    policy = load_verified_policy(policy_path, verify_artifacts=True)
    batch = load_batch_metadata(db_path, batch_id)
    if batch["policy_sha256"] != policy["policy_sha256"]:
        raise ValueError("batch policy hash does not match frozen policy")
    instances = load_batch_instances(db_path, batch_id)
    evidence = evidence_provider(instances, policy)
    instance_ids = {str(row["instance_id"]) for row in instances}
    if set(evidence) != instance_ids:
        raise ValueError("evidence coverage does not match batch instances")
    conservative_alpha = float(policy.get("metadata", {}).get("conservative_alpha", 0.01))
    rows = [
        route_instance(
            batch_id=batch_id,
            instance_id=str(instance["instance_id"]),
            profile_id=str(instance["profile_id"]),
            policy_hash=str(policy["policy_sha256"]),
            evidence=evidence[str(instance["instance_id"])],
            conservative_alpha=conservative_alpha,
        )
        for instance in instances
    ]
    return freeze_shadow_predictions(
        rows=rows,
        batch_id=batch_id,
        policy_sha256=str(policy["policy_sha256"]),
        source_sha256=batch["source_sha256"],
        output_dir=output_dir,
    )
