"""Create-only, digest-bound score artifacts for Exp473."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from .contracts_antigravity import (
    Exp473ScoreShardAntigravity,
    canonical_json_bytes_exp473_antigravity,
)


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")


def _digest_antigravity(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_ANTIGRAVITY.fullmatch(value) is None:
        raise ValueError(f"Exp473 {field} must be a lowercase SHA-256")
    return value


def _shard_path_antigravity(root: Path, dataset: str) -> Path:
    token = sha256(dataset.encode("utf-8")).hexdigest()
    return (
        Path(root)
        / "score_shards_antigravity"
        / f"{token}_antigravity"
        / "score_shard_antigravity.json"
    )


def _write_create_only_antigravity(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()


def publish_run_context_exp473_antigravity(
    root: Path, bare_context: dict[str, object]
) -> dict[str, object]:
    bare = dict(bare_context)
    if (
        bare.get("schema") != "exp473-run-context-antigravity.v1"
        or bare.get("experiment") != "Exp473"
        or bare.get("mode") not in {"smoke", "full"}
        or "context_sha256" in bare
    ):
        raise ValueError("Exp473 run context identity differs")
    record = {
        **bare,
        "context_sha256": sha256(
            canonical_json_bytes_exp473_antigravity(bare)
        ).hexdigest(),
    }
    _write_create_only_antigravity(
        Path(root) / "run_context_antigravity.json",
        canonical_json_bytes_exp473_antigravity(record),
    )
    return record


def verify_run_context_exp473_antigravity(root: Path) -> dict[str, object]:
    path = Path(root) / "run_context_antigravity.json"
    raw = path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp473 run context is invalid") from error
    if not isinstance(record, dict) or raw != canonical_json_bytes_exp473_antigravity(record):
        raise ValueError("Exp473 run context is noncanonical")
    bare = dict(record)
    digest = bare.pop("context_sha256", None)
    if (
        bare.get("schema") != "exp473-run-context-antigravity.v1"
        or bare.get("experiment") != "Exp473"
        or bare.get("mode") not in {"smoke", "full"}
        or digest != sha256(canonical_json_bytes_exp473_antigravity(bare)).hexdigest()
    ):
        raise ValueError("Exp473 run context identity differs")
    return record


def _record_from_shard_antigravity(
    context_sha256: str, shard: Exp473ScoreShardAntigravity
) -> dict[str, object]:
    bare: dict[str, object] = {
        "schema": "exp473-d-prime-score-shard-antigravity.v1",
        "experiment": "Exp473",
        "control_id": "D_prime",
        "base_condition_id": "D",
        "length_mode": "polyphase_pchip",
        "image_mode": "area",
        "policy_id": "hard_plus_stable",
        "score_threshold": 0.5,
        "context_sha256": _digest_antigravity(context_sha256, "context digest"),
        "dataset": shard.dataset,
        "test_identity": dict(shard.test_identity),
        "scores": [float(value) for value in shard.scores],
        "alerts": sorted(shard.alerts),
    }
    return {
        **bare,
        "shard_sha256": sha256(
            canonical_json_bytes_exp473_antigravity(bare)
        ).hexdigest(),
    }


def publish_score_shard_exp473_antigravity(
    root: Path,
    context_sha256: str,
    shard: Exp473ScoreShardAntigravity,
) -> dict[str, object]:
    if not isinstance(shard, Exp473ScoreShardAntigravity):
        raise ValueError("Exp473 score shard type differs")
    record = _record_from_shard_antigravity(context_sha256, shard)
    _write_create_only_antigravity(
        _shard_path_antigravity(Path(root), shard.dataset),
        canonical_json_bytes_exp473_antigravity(record),
    )
    return record


def verify_score_shard_exp473_antigravity(
    root: Path, context_sha256: str, dataset: str
) -> Exp473ScoreShardAntigravity:
    expected_context = _digest_antigravity(context_sha256, "context digest")
    if type(dataset) is not str or not dataset:
        raise ValueError("Exp473 dataset identity is invalid")
    path = _shard_path_antigravity(Path(root), dataset)
    raw = path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp473 score shard is invalid") from error
    if not isinstance(record, dict) or raw != canonical_json_bytes_exp473_antigravity(record):
        raise ValueError("Exp473 score shard is noncanonical")
    bare = dict(record)
    shard_sha256 = bare.pop("shard_sha256", None)
    if (
        record.get("schema") != "exp473-d-prime-score-shard-antigravity.v1"
        or record.get("experiment") != "Exp473"
        or record.get("control_id") != "D_prime"
        or record.get("base_condition_id") != "D"
        or record.get("length_mode") != "polyphase_pchip"
        or record.get("image_mode") != "area"
        or record.get("policy_id") != "hard_plus_stable"
        or record.get("score_threshold") != 0.5
        or record.get("context_sha256") != expected_context
        or record.get("dataset") != dataset
        or shard_sha256
        != sha256(canonical_json_bytes_exp473_antigravity(bare)).hexdigest()
    ):
        raise ValueError("Exp473 score shard identity differs")
    try:
        return Exp473ScoreShardAntigravity(
            dataset,
            record["test_identity"],
            np.asarray(record["scores"], dtype=np.float64),
            frozenset(record["alerts"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp473 score shard payload differs") from error


def publish_manifest_exp473_antigravity(
    root: Path, context_sha256: str, catalog: tuple[str, ...]
) -> dict[str, object]:
    context = _digest_antigravity(context_sha256, "context digest")
    names = tuple(catalog)
    if not names or len(names) != len(set(names)) or any(
        type(name) is not str or not name for name in names
    ):
        raise ValueError("Exp473 catalog differs")
    shard_digests: dict[str, str] = {}
    for dataset in names:
        verify_score_shard_exp473_antigravity(root, context, dataset)
        path = _shard_path_antigravity(Path(root), dataset)
        record = json.loads(path.read_text(encoding="utf-8"))
        shard_digests[dataset] = record["shard_sha256"]
    bare: dict[str, object] = {
        "schema": "exp473-d-prime-score-manifest-antigravity.v1",
        "experiment": "Exp473",
        "control_id": "D_prime",
        "context_sha256": context,
        "catalog": list(names),
        "completed": len(names),
        "failed": 0,
        "remaining": 0,
        "shard_sha256_by_dataset": shard_digests,
    }
    record = {
        **bare,
        "manifest_sha256": sha256(
            canonical_json_bytes_exp473_antigravity(bare)
        ).hexdigest(),
    }
    _write_create_only_antigravity(
        Path(root) / "score_manifest_antigravity.json",
        canonical_json_bytes_exp473_antigravity(record),
    )
    return record


def verify_manifest_exp473_antigravity(
    root: Path, context_sha256: str, catalog: tuple[str, ...]
) -> dict[str, object]:
    path = Path(root) / "score_manifest_antigravity.json"
    raw = path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp473 score manifest is invalid") from error
    if not isinstance(record, dict) or raw != canonical_json_bytes_exp473_antigravity(record):
        raise ValueError("Exp473 score manifest is noncanonical")
    bare = dict(record)
    digest = bare.pop("manifest_sha256", None)
    names = tuple(catalog)
    if (
        record.get("schema") != "exp473-d-prime-score-manifest-antigravity.v1"
        or record.get("experiment") != "Exp473"
        or record.get("control_id") != "D_prime"
        or record.get("context_sha256") != context_sha256
        or record.get("catalog") != list(names)
        or record.get("completed") != len(names)
        or record.get("failed") != 0
        or record.get("remaining") != 0
        or digest != sha256(canonical_json_bytes_exp473_antigravity(bare)).hexdigest()
    ):
        raise ValueError("Exp473 score manifest identity differs")
    expected_digests = record.get("shard_sha256_by_dataset")
    if not isinstance(expected_digests, dict) or tuple(expected_digests) != names:
        raise ValueError("Exp473 score manifest coverage differs")
    for dataset in names:
        verify_score_shard_exp473_antigravity(root, context_sha256, dataset)
        shard_record = json.loads(
            _shard_path_antigravity(Path(root), dataset).read_text(encoding="utf-8")
        )
        if expected_digests.get(dataset) != shard_record.get("shard_sha256"):
            raise ValueError("Exp473 score manifest shard binding differs")
    return record


__all__ = [
    "publish_manifest_exp473_antigravity",
    "publish_run_context_exp473_antigravity",
    "publish_score_shard_exp473_antigravity",
    "verify_manifest_exp473_antigravity",
    "verify_run_context_exp473_antigravity",
    "verify_score_shard_exp473_antigravity",
]
