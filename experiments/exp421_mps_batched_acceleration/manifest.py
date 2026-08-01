"""Canonical, TRAIN-only provenance manifest for the Exp421 diagnostic.

The manifest is deliberately a boundary layer.  It serializes immutable
CPU/MPS evidence and the complete hybrid policy bundle, hashes only canonical
JSON, and writes into a fresh attempt root without clobbering an existing
artifact.  No series values, TEST identity, labels, locations, checkpoints,
or prediction payloads are accepted here.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np

from .config import Exp421Config, FROZEN_CONFIG, frozen_config_sha256
from .contracts import (
    BackendMeasurement,
    BackendPolicy,
    BatchKey,
    Exp421Manifest,
    MpsAssessment,
    ParityReport,
    _policy_rows_equivalent,
)
from .policy import BackendPolicyBundle


_TOP_LEVEL_FIELDS = frozenset(
    {
        "experiment_id",
        "state",
        "config",
        "candidate_batch_sizes",
        "cached_candidate_indices",
        "measurements",
        "parity",
        "policy",
        "policy_bundle",
        "provenance",
        "source_digests",
        "source_manifest_git_commit",
        "source_manifest_sha256",
        "attempt_root",
    }
)
_CONFIG_FIELDS = frozenset(
    {"config_sha256", "candidate_batch_sizes", "frozen_model_seeds", "frozen_perturbation_seeds"}
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "batch_size",
        "backend",
        "transfer_seconds",
        "compute_seconds",
        "synchronize_seconds",
        "total_seconds",
        "finite",
        "candidate_coordinates_sha256",
    }
)
_PARITY_FIELDS = frozenset(
    {
        "batch_key",
        "cpu_candidate_indices_by_alpha",
        "mps_candidate_indices_by_alpha",
        "max_normalized_error",
        "spearman",
    }
)
_BATCH_KEY_FIELDS = frozenset({"batch_size", "model_seed", "perturbation_seed"})
_ASSESSMENT_FIELDS = frozenset(
    {
        "batch_key",
        "uses_mps",
        "fallback_reason",
        "mps_available",
        "device",
        "speedup",
    }
)
_POLICY_ROW_FIELDS = frozenset({"backend", "assessment"})
_POLICY_BUNDLE_FIELDS = frozenset({"policies", "reasons", "fail_closed"})
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_FORBIDDEN_COMPONENT_RE = re.compile(
    r"(?:^|[-_.])(?:TEST|LABEL|GROUND[_-]?TRUTH|LOCATION)(?:$|[-_.])",
    re.IGNORECASE,
)
_OFFICIAL_EXP419_MARKERS = (
    "exp419_deterministic_cpu_parallel",
    "exp419_deterministic_cpu_parallel.checkpoint",
    "exp419_prediction",
    "exp419_checkpoint",
)
_SOURCE_MANIFEST_FIELDS = frozenset({"artifact_kind", "schema_version", "git_commit", "sources"})
_SOURCE_ENTRY_FIELDS = frozenset({"path", "sha256"})
_PRIVATE_ARTIFACT_SUFFIXES = frozenset(
    {
        ".db",
        ".sqlite",
        ".sqlite3",
        ".jsonl",
        ".npz",
        ".npy",
        ".pt",
        ".pth",
        ".ckpt",
        ".checkpoint",
        ".csv",
    }
)


def _repository_root() -> Path:
    # manifest.py -> exp421 package -> experiments -> worktree root
    return Path(__file__).resolve().parents[2]


def _source_manifest_path() -> Path:
    return _repository_root() / "SOURCE_MANIFEST.json"


def _allowed_source_path(path: Path) -> bool:
    root = _repository_root()
    allowed_roots = (
        root / "experiments" / "exp421_mps_batched_acceleration",
        root / "docs",
        root / "scratch",
    )
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    if not any(resolved == allowed or allowed in resolved.parents for allowed in allowed_roots):
        return False
    if resolved.suffix.lower() in _PRIVATE_ARTIFACT_SUFFIXES:
        return False
    components = {component.lower() for component in resolved.parts}
    if components.intersection({"dataset", "private", "checkpoints", "predictions", "outputs"}):
        return False
    return True


def _git_tree_paths(commit: str) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=_repository_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {line for line in result.stdout.splitlines() if line}


def _load_source_manifest() -> tuple[str, dict[str, str], str]:
    """Read and independently verify the committed source trust root.

    The source manifest itself is intentionally not one of ``sources``; its
    complete bytes are hashed separately in the Exp421 payload, avoiding a
    recursive self-digest while still detecting source-manifest tampering.
    """
    path = _source_manifest_path()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SOURCE_MANIFEST.json is unreadable or invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != _SOURCE_MANIFEST_FIELDS:
        raise ValueError("SOURCE_MANIFEST.json has an invalid schema")
    if payload["artifact_kind"] != "exp421_train_only_acceleration_source_manifest":
        raise ValueError("SOURCE_MANIFEST.json artifact kind is invalid")
    if payload["schema_version"] != 1:
        raise ValueError("SOURCE_MANIFEST.json schema version is invalid")
    commit = _git_digest(payload["git_commit"])
    entries = payload["sources"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("SOURCE_MANIFEST.json sources must be a nonempty list")
    source_digests: dict[str, str] = {}
    git_paths = _git_tree_paths(commit)
    if not git_paths:
        raise ValueError("SOURCE_MANIFEST git commit is unavailable")
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _SOURCE_ENTRY_FIELDS:
            raise ValueError("SOURCE_MANIFEST.json source entry schema is invalid")
        relative = entry["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("SOURCE_MANIFEST.json source paths must be relative")
        resolved = (_repository_root() / relative).resolve(strict=False)
        if not _allowed_source_path(resolved):
            raise ValueError("SOURCE_MANIFEST.json contains an out-of-scope source path")
        digest = _sha256_digest(entry["sha256"], f"SOURCE_MANIFEST.json[{relative}]")
        if relative in source_digests:
            raise ValueError("SOURCE_MANIFEST.json contains duplicate source paths")
        source_digests[relative] = digest
        if relative not in git_paths:
            raise ValueError(f"SOURCE_MANIFEST git commit does not contain {relative}")
        actual = _source_file_digest(str(resolved))
        if actual != digest:
            raise ValueError(f"SOURCE_MANIFEST source digest drift: {relative}")
    return commit, dict(sorted(source_digests.items())), sha256(raw).hexdigest()


def _is_official_exp419_path(value: str) -> bool:
    components = [component for component in re.split(r"[/\\]+", value.lower()) if component]
    return any(
        component == "exp419"
        or component.startswith("exp419_")
        or component.startswith("exp419-")
        or component in _OFFICIAL_EXP419_MARKERS
        for component in components
    )


def _sha256_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_digest(value: object) -> str:
    if not isinstance(value, str) or not _GIT_RE.fullmatch(value):
        raise ValueError("git_commit must be a lowercase Git object digest")
    return value


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    if not isinstance(payload, Mapping):
        raise ValueError("manifest payload must be a mapping")
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest payload is not canonical JSON") from exc


def canonical_manifest_sha256(payload: Mapping[str, object] | Exp421Manifest) -> str:
    """Hash exactly the canonical UTF-8 JSON representation of a payload."""
    if isinstance(payload, Exp421Manifest):
        payload = _manifest_to_payload(payload)
    return sha256(_canonical_bytes(payload)).hexdigest()


def _path_text(path: object, name: str, *, allow_package_tests: bool = False) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{name} must be a nonempty path")
    value = path.strip()
    lowered = value.lower()
    if _is_official_exp419_path(value) or any(marker in lowered for marker in _OFFICIAL_EXP419_MARKERS):
        raise ValueError(f"{name} must not target official Exp419 artifacts")
    # Keep lower-case ``tests/test_*.py`` source files possible while
    # rejecting explicit artifact declarations such as ``TEST_labels.json``.
    components = re.split(r"[/\\]+", value)
    forbidden = []
    for component in components:
        if allow_package_tests and component.lower() == "tests":
            continue
        if allow_package_tests and component.startswith("test_"):
            continue
        forbidden.append(component)
    if any(_FORBIDDEN_COMPONENT_RE.search(component) for component in forbidden):
        raise ValueError(f"{name} contains a forbidden TEST/label/location token")
    return value


def _attempt_root(path: object) -> str:
    value = _path_text(path, "attempt_root")
    resolved = Path(value).expanduser().resolve(strict=False)
    if _is_official_exp419_path(str(resolved)):
        raise ValueError("attempt_root must be outside official Exp419 roots")
    return str(resolved)


def _source_file_digest(path: str) -> str:
    try:
        data = Path(path).expanduser().read_bytes()
    except OSError as exc:
        raise ValueError(f"source file is not readable: {path}") from exc
    return sha256(data).hexdigest()


def _validate_source_digests(source_digests: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(source_digests, Mapping) or not source_digests:
        raise ValueError("source_digests must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for raw_path, digest in source_digests.items():
        path_text = str(raw_path) if isinstance(raw_path, str) else ""
        allow_package_tests = "experiments/exp421_mps_batched_acceleration/" in path_text.replace("\\", "/")
        path = _path_text(raw_path, "source digest path", allow_package_tests=allow_package_tests)
        declared_path = Path(path).expanduser()
        resolved = (
            declared_path
            if declared_path.is_absolute()
            else _repository_root() / declared_path
        ).resolve(strict=False)
        if not _allowed_source_path(resolved):
            raise ValueError("source digest path must be inside an allowed Exp421/docs/scratch root")
        normalized[path] = _sha256_digest(digest, f"source_digests[{path}]")
        actual = _source_file_digest(str(resolved))
        if actual != normalized[path]:
            raise ValueError(f"source digest drift: {path}")
    _, committed_sources, _ = _load_source_manifest()
    if dict(sorted(normalized.items())) != committed_sources:
        raise ValueError("source_digests must exactly match SOURCE_MANIFEST.json")
    return dict(sorted(normalized.items()))


def _validate_provenance_boundary(provenance: Mapping[str, object]) -> None:
    """Reject label/location/TEST declarations before contract construction."""
    for name, value in provenance.items():
        if not isinstance(value, str):
            continue
        if name in {"source", "runtime", "device", "device_name", "torch_version"}:
            if re.search(r"\btest\b|label|ground[_-]?truth|location", value, re.IGNORECASE):
                raise ValueError(f"provenance {name} contains a forbidden boundary token")


def _source_manifest_metadata() -> tuple[str, dict[str, str], str]:
    """Return the independently validated source commit, entries, and bytes hash."""
    return _load_source_manifest()


def _serialize_batch_key(key: BatchKey) -> dict[str, int]:
    if not isinstance(key, BatchKey):
        raise ValueError("batch key is invalid")
    return {
        "batch_size": key.batch_size,
        "model_seed": key.model_seed,
        "perturbation_seed": key.perturbation_seed,
    }


def _parse_batch_key(payload: object) -> BatchKey:
    if not isinstance(payload, Mapping) or set(payload) != _BATCH_KEY_FIELDS:
        raise ValueError("batch key payload has an invalid schema")
    try:
        return BatchKey(
            batch_size=payload["batch_size"],
            model_seed=payload["model_seed"],
            perturbation_seed=payload["perturbation_seed"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("batch key payload is not frozen") from exc


def _serialize_measurement(measurement: BackendMeasurement | None) -> dict[str, object] | None:
    if measurement is None:
        return None
    if not isinstance(measurement, BackendMeasurement):
        raise ValueError("measurement is invalid")
    return {
        "batch_size": measurement.batch_size,
        "backend": measurement.backend,
        "transfer_seconds": measurement.transfer_seconds,
        "compute_seconds": measurement.compute_seconds,
        "synchronize_seconds": measurement.synchronize_seconds,
        "total_seconds": measurement.total_seconds,
        "finite": measurement.finite,
        "candidate_coordinates_sha256": measurement.candidate_coordinates_sha256,
    }


def _parse_measurement(payload: object, *, expected_batch_size: int, expected_backend: str) -> BackendMeasurement:
    if not isinstance(payload, Mapping) or set(payload) != _MEASUREMENT_FIELDS:
        raise ValueError("measurement payload has an invalid schema")
    if payload.get("batch_size") != expected_batch_size or payload.get("backend") != expected_backend:
        raise ValueError("measurement payload does not bind to its batch/backend")
    try:
        return BackendMeasurement(
            batch_size=payload["batch_size"],
            backend=payload["backend"],
            transfer_seconds=payload["transfer_seconds"],
            compute_seconds=payload["compute_seconds"],
            synchronize_seconds=payload["synchronize_seconds"],
            total_seconds=payload["total_seconds"],
            finite=payload["finite"],
            candidate_coordinates_sha256=payload["candidate_coordinates_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("measurement payload is invalid") from exc


def _serialize_parity(parity: ParityReport | None) -> dict[str, object] | None:
    if parity is None:
        return None
    if not isinstance(parity, ParityReport):
        raise ValueError("parity is invalid")
    def serialize_indices(values: Mapping[float, np.ndarray]) -> dict[str, list[int]]:
        return {str(alpha): values[alpha].tolist() for alpha in values}

    return {
        "batch_key": _serialize_batch_key(parity.batch_key),
        "cpu_candidate_indices_by_alpha": serialize_indices(parity.cpu_candidate_indices_by_alpha),
        "mps_candidate_indices_by_alpha": serialize_indices(parity.mps_candidate_indices_by_alpha),
        "max_normalized_error": parity.max_normalized_error,
        "spearman": parity.spearman,
    }


def _parse_parity(payload: object, *, expected_batch_key: BatchKey) -> ParityReport:
    if not isinstance(payload, Mapping) or set(payload) != _PARITY_FIELDS:
        raise ValueError("parity payload has an invalid schema")
    key = _parse_batch_key(payload["batch_key"])
    if key != expected_batch_key:
        raise ValueError("parity payload does not bind to its batch key")

    def parse_indices(values: object) -> dict[float, np.ndarray]:
        if not isinstance(values, Mapping):
            raise ValueError("parity indices must be a mapping")
        result: dict[float, np.ndarray] = {}
        for alpha_text, indices in values.items():
            try:
                alpha = float(alpha_text)
            except (TypeError, ValueError) as exc:
                raise ValueError("parity alpha is invalid") from exc
            if not isinstance(indices, list) or any(type(item) is not int for item in indices):
                raise ValueError("parity indices must be integer lists")
            result[alpha] = np.asarray(indices, dtype=np.int64)
        return result

    try:
        return ParityReport(
            batch_key=key,
            cpu_candidate_indices_by_alpha=parse_indices(payload["cpu_candidate_indices_by_alpha"]),
            mps_candidate_indices_by_alpha=parse_indices(payload["mps_candidate_indices_by_alpha"]),
            max_normalized_error=payload["max_normalized_error"],
            spearman=payload["spearman"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("parity payload is invalid") from exc


def _serialize_assessment(assessment: MpsAssessment) -> dict[str, object]:
    if not isinstance(assessment, MpsAssessment):
        raise ValueError("assessment is invalid")
    return {
        "batch_key": _serialize_batch_key(assessment.batch_key),
        "uses_mps": assessment.uses_mps,
        "fallback_reason": assessment.fallback_reason,
        "mps_available": assessment.mps_available,
        "device": assessment.device,
        "speedup": assessment.speedup,
    }


def _parse_assessment(
    payload: object,
    *,
    expected_batch_size: int,
    cpu_measurement: BackendMeasurement,
    mps_measurement: BackendMeasurement | None,
    parity: ParityReport | None,
) -> MpsAssessment:
    if not isinstance(payload, Mapping) or set(payload) != _ASSESSMENT_FIELDS:
        raise ValueError("assessment payload has an invalid schema")
    key = _parse_batch_key(payload["batch_key"])
    if key.batch_size != expected_batch_size:
        raise ValueError("assessment payload does not bind to its batch")
    try:
        return MpsAssessment(
            batch_key=key,
            cpu_measurement=cpu_measurement,
            mps_measurement=mps_measurement,
            parity_report=parity,
            speedup=payload["speedup"],
            uses_mps=payload["uses_mps"],
            fallback_reason=payload["fallback_reason"],
            mps_available=payload["mps_available"],
            device=payload["device"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("assessment payload is invalid") from exc


def _serialize_policy(policy: BackendPolicy) -> dict[str, object]:
    if not isinstance(policy, BackendPolicy):
        raise ValueError("policy is invalid")
    return {
        "backend": policy.backend,
        "assessment": _serialize_assessment(policy.assessment),
    }


def _parse_policy(
    payload: object,
    *,
    expected_batch_size: int,
    cpu_measurement: BackendMeasurement,
    mps_measurement: BackendMeasurement | None,
    parity: ParityReport | None,
) -> BackendPolicy:
    if not isinstance(payload, Mapping) or set(payload) != _POLICY_ROW_FIELDS:
        raise ValueError("policy row has an invalid schema")
    assessment = _parse_assessment(
        payload["assessment"],
        expected_batch_size=expected_batch_size,
        cpu_measurement=cpu_measurement,
        mps_measurement=mps_measurement,
        parity=parity,
    )
    try:
        return BackendPolicy(backend=payload["backend"], assessment=assessment)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy row is invalid") from exc


def _mapping_rows(
    bundle: BackendPolicyBundle,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    measurements: dict[str, dict[str, object]] = {}
    parities: dict[str, dict[str, object] | None] = {}
    policies: dict[str, object] = {}
    for batch_size in FROZEN_CONFIG.candidate_batch_sizes:
        assessment = bundle.policies[batch_size].assessment
        measurements[str(batch_size)] = {
            "cpu": _serialize_measurement(assessment.cpu_measurement),
            "mps": _serialize_measurement(assessment.mps_measurement),
        }
        parities[str(batch_size)] = _serialize_parity(assessment.parity_report)
        policies[str(batch_size)] = _serialize_policy(bundle.policies[batch_size])
    return measurements, parities, {
        "policies": policies,
        "reasons": {str(key): value for key, value in bundle.reasons.items()},
        "fail_closed": bundle.fail_closed,
    }


def _manifest_to_payload(manifest: Exp421Manifest) -> dict[str, object]:
    """Serialize a fully populated Exp421Manifest into JSON-ready values."""
    if not isinstance(manifest, Exp421Manifest):
        raise ValueError("manifest must be an Exp421Manifest")
    if not isinstance(manifest.policy_bundle, BackendPolicyBundle):
        raise ValueError("manifest must retain the complete BackendPolicyBundle")
    if manifest.measurements is None or manifest.parities is None:
        raise ValueError("manifest must retain complete measurement and parity evidence")
    if manifest.source_digests is None or manifest.attempt_root is None:
        raise ValueError("manifest must retain source digests and a fresh attempt root")
    source_digests = dict(sorted(manifest.source_digests.items()))
    source_manifest_commit, committed_sources, source_manifest_sha256 = _source_manifest_metadata()
    if source_digests != committed_sources:
        raise ValueError("manifest source digests do not match SOURCE_MANIFEST.json")
    measurements, parities, policy_bundle = _mapping_rows(manifest.policy_bundle)
    # Validate that the explicit evidence fields have not diverged from the
    # policy's assessment rows before serializing either representation.
    for batch_size in FROZEN_CONFIG.candidate_batch_sizes:
        expected = manifest.policy_bundle.policies[batch_size].assessment
        row = manifest.measurements[batch_size]
        if row["cpu"] != expected.cpu_measurement or row["mps"] != expected.mps_measurement:
            raise ValueError("measurement evidence does not match policy bundle")
        if manifest.parities[batch_size] != expected.parity_report:
            raise ValueError("parity evidence does not match policy bundle")
    return {
        "experiment_id": manifest.experiment_id,
        "state": manifest.state,
        "config": {
            "config_sha256": frozen_config_sha256(manifest.config),
            "candidate_batch_sizes": list(manifest.config.candidate_batch_sizes),
            "frozen_model_seeds": list(manifest.config.frozen_model_seeds),
            "frozen_perturbation_seeds": list(manifest.config.frozen_perturbation_seeds),
        },
        "candidate_batch_sizes": list(manifest.config.candidate_batch_sizes),
        "cached_candidate_indices": manifest.cached_candidate_indices.tolist(),
        "measurements": measurements,
        "parity": parities,
        "policy": _serialize_policy(manifest.policy),
        "policy_bundle": policy_bundle,
        "provenance": dict(manifest.provenance),
        "source_digests": source_digests,
        "source_manifest_git_commit": source_manifest_commit,
        "source_manifest_sha256": source_manifest_sha256,
        "attempt_root": manifest.attempt_root,
    }


def build_manifest(
    *,
    policy_bundle: BackendPolicyBundle,
    cached_candidate_indices: np.ndarray,
    provenance: Mapping[str, object],
    source_digests: Mapping[str, str],
    attempt_root: str | Path,
    state: str = "draft",
    config: Exp421Config = FROZEN_CONFIG,
    policy: BackendPolicy | None = None,
) -> Exp421Manifest:
    """Construct and validate a complete immutable Exp421 manifest."""
    if type(config) is not Exp421Config:
        raise ValueError("config must be an Exp421Config")
    config.validate()
    if config != FROZEN_CONFIG:
        raise ValueError("Exp421 manifest requires the exact frozen config")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    if not isinstance(policy_bundle, BackendPolicyBundle):
        raise ValueError("policy_bundle must be a BackendPolicyBundle")
    if tuple(policy_bundle.policies) != config.candidate_batch_sizes:
        raise ValueError("policy_bundle must cover every frozen batch size")
    compatibility_policy = policy_bundle.policies[1] if policy is None else policy
    if not isinstance(compatibility_policy, BackendPolicy):
        raise ValueError("policy must be a BackendPolicy")
    if not _policy_rows_equivalent(compatibility_policy, policy_bundle.policies[1]):
        raise ValueError("policy must reference the batch-one bundle row")
    if not isinstance(cached_candidate_indices, np.ndarray):
        raise ValueError("cached_candidate_indices must be an int64 array")
    cached = np.array(cached_candidate_indices, dtype=np.int64, copy=True)
    cached.setflags(write=False)
    _validate_provenance_boundary(provenance)
    normalized_sources = _validate_source_digests(source_digests)
    normalized_root = _attempt_root(str(attempt_root))
    root_path = Path(normalized_root)
    if root_path.exists() and (not root_path.is_dir() or any(root_path.iterdir())):
        raise FileExistsError("attempt_root must be fresh and empty")
    # The complete evidence is derived from the bundle, not from any caller
    # supplied subset, so every batch row remains auditable.
    measurements: dict[int, Mapping[str, BackendMeasurement | None]] = {}
    parities: dict[int, ParityReport | None] = {}
    for batch_size in config.candidate_batch_sizes:
        assessment = policy_bundle.policies[batch_size].assessment
        measurements[batch_size] = {
            "cpu": assessment.cpu_measurement,
            "mps": assessment.mps_measurement,
        }
        parities[batch_size] = assessment.parity_report
    manifest = Exp421Manifest(
        state=state,
        policy=compatibility_policy,
        cached_candidate_indices=cached,
        provenance=dict(provenance),
        config=config,
        experiment_id=config.experiment_id,
        policy_bundle=policy_bundle,
        measurements=measurements,
        parities=parities,
        source_digests=normalized_sources,
        attempt_root=normalized_root,
    )
    payload = _manifest_to_payload(manifest)
    validate_manifest(payload)
    return manifest


def _validate_top_level(payload: Mapping[str, object]) -> None:
    if set(payload) != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - set(payload))
        extra = sorted(set(payload) - _TOP_LEVEL_FIELDS)
        raise ValueError(f"manifest fields mismatch; missing={missing}, extra={extra}")


def _parse_candidate_indices(payload: object) -> np.ndarray:
    if not isinstance(payload, list) or any(type(item) is not int for item in payload):
        raise ValueError("cached_candidate_indices must be an integer list")
    if payload and (any(item < 0 for item in payload) or any(a >= b for a, b in zip(payload, payload[1:]))):
        raise ValueError("cached_candidate_indices must be sorted unique nonnegative indices")
    value = np.asarray(payload, dtype=np.int64)
    value.setflags(write=False)
    return value


def _parse_config(payload: object, config: Exp421Config) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _CONFIG_FIELDS:
        raise ValueError("config payload has an invalid schema")
    expected = {
        "config_sha256": frozen_config_sha256(config),
        "candidate_batch_sizes": list(config.candidate_batch_sizes),
        "frozen_model_seeds": list(config.frozen_model_seeds),
        "frozen_perturbation_seeds": list(config.frozen_perturbation_seeds),
    }
    if dict(payload) != expected:
        raise ValueError("config payload does not bind to the frozen Exp421 config")


def _parse_measurement_rows(payload: object) -> dict[int, dict[str, BackendMeasurement | None]]:
    expected = tuple(FROZEN_CONFIG.candidate_batch_sizes)
    if not isinstance(payload, Mapping) or set(payload) != {str(size) for size in expected}:
        raise ValueError("measurements must cover the exact frozen batch schedule")
    rows: dict[int, dict[str, BackendMeasurement | None]] = {}
    for batch_size in expected:
        row_payload = payload[str(batch_size)]
        if not isinstance(row_payload, Mapping) or set(row_payload) != {"cpu", "mps"}:
            raise ValueError("measurement row schema is invalid")
        cpu = _parse_measurement(row_payload["cpu"], expected_batch_size=batch_size, expected_backend="cpu")
        mps_payload = row_payload["mps"]
        mps = None if mps_payload is None else _parse_measurement(
            mps_payload, expected_batch_size=batch_size, expected_backend="mps"
        )
        rows[batch_size] = {"cpu": cpu, "mps": mps}
    return rows


def _parse_parity_rows(payload: object) -> dict[int, ParityReport | None]:
    expected = tuple(FROZEN_CONFIG.candidate_batch_sizes)
    if not isinstance(payload, Mapping) or set(payload) != {str(size) for size in expected}:
        raise ValueError("parity must cover the exact frozen batch schedule")
    rows: dict[int, ParityReport | None] = {}
    for batch_size in expected:
        item = payload[str(batch_size)]
        if item is None:
            rows[batch_size] = None
            continue
        # The batch key is parsed by _parse_parity itself and validated here.
        key_payload = item.get("batch_key") if isinstance(item, Mapping) else None
        key = _parse_batch_key(key_payload)
        if key.batch_size != batch_size:
            raise ValueError("parity row does not bind to its batch")
        rows[batch_size] = _parse_parity(item, expected_batch_key=key)
    return rows


def _parse_policy_bundle(
    payload: object,
    *,
    measurements: Mapping[int, Mapping[str, BackendMeasurement | None]],
    parities: Mapping[int, ParityReport | None],
) -> BackendPolicyBundle:
    if not isinstance(payload, Mapping) or set(payload) != _POLICY_BUNDLE_FIELDS:
        raise ValueError("policy_bundle payload has an invalid schema")
    policies_payload = payload["policies"]
    reasons_payload = payload["reasons"]
    expected = tuple(FROZEN_CONFIG.candidate_batch_sizes)
    if not isinstance(policies_payload, Mapping) or set(policies_payload) != {str(size) for size in expected}:
        raise ValueError("policy_bundle policies have incomplete batch coverage")
    if not isinstance(reasons_payload, Mapping) or set(reasons_payload) != {str(size) for size in expected}:
        raise ValueError("policy_bundle reasons have incomplete batch coverage")
    policies: dict[int, BackendPolicy] = {}
    reasons: dict[int, str] = {}
    for batch_size in expected:
        policies[batch_size] = _parse_policy(
            policies_payload[str(batch_size)],
            expected_batch_size=batch_size,
            cpu_measurement=measurements[batch_size]["cpu"],
            mps_measurement=measurements[batch_size]["mps"],
            parity=parities[batch_size],
        )
        reason = reasons_payload[str(batch_size)]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("policy fallback reasons must be nonempty strings")
        reasons[batch_size] = reason
    try:
        return BackendPolicyBundle(
            policies=policies,
            reasons=reasons,
            fail_closed=payload["fail_closed"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("policy_bundle payload is invalid") from exc


def validate_manifest(payload: Mapping[str, object]) -> Exp421Manifest:
    """Validate a decoded manifest and reconstruct all immutable evidence."""
    if not isinstance(payload, Mapping):
        raise ValueError("manifest payload must be a mapping")
    _validate_top_level(payload)
    if payload["experiment_id"] != FROZEN_CONFIG.experiment_id:
        raise ValueError("manifest experiment_id is invalid")
    if payload["state"] not in {"draft", "passed", "failed"}:
        raise ValueError("manifest state is invalid")
    _parse_config(payload["config"], FROZEN_CONFIG)
    if payload["candidate_batch_sizes"] != list(FROZEN_CONFIG.candidate_batch_sizes):
        raise ValueError("candidate batch schedule is not frozen")
    cached = _parse_candidate_indices(payload["cached_candidate_indices"])
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    _validate_provenance_boundary(provenance)
    source_digests = _validate_source_digests(payload["source_digests"])
    expected_source_commit, committed_sources, expected_source_manifest = _source_manifest_metadata()
    if source_digests != committed_sources:
        raise ValueError("source declarations do not match SOURCE_MANIFEST.json")
    if payload["source_manifest_git_commit"] != expected_source_commit:
        raise ValueError("source_manifest_git_commit does not match SOURCE_MANIFEST.json")
    if payload["source_manifest_sha256"] != expected_source_manifest:
        raise ValueError("source_manifest_sha256 does not match SOURCE_MANIFEST.json bytes")
    attempt_root = _attempt_root(payload["attempt_root"])
    if attempt_root != payload["attempt_root"]:
        raise ValueError("attempt_root must use a canonical absolute path")
    measurements = _parse_measurement_rows(payload["measurements"])
    parities = _parse_parity_rows(payload["parity"])
    bundle = _parse_policy_bundle(
        payload["policy_bundle"],
        measurements=measurements,
        parities=parities,
    )
    singular = _parse_policy(
        payload["policy"],
        expected_batch_size=1,
        cpu_measurement=measurements[1]["cpu"],
        mps_measurement=measurements[1]["mps"],
        parity=parities[1],
    )
    if singular.backend != bundle.policies[1].backend:
        raise ValueError("singular compatibility policy does not match bundle batch one")
    # Constructing the immutable contract rechecks provenance, MLX coherence,
    # exact source fields, and all cached evidence before returning anything.
    manifest = Exp421Manifest(
        state=payload["state"],
        policy=singular,
        cached_candidate_indices=cached,
        provenance=dict(provenance),
        config=FROZEN_CONFIG,
        experiment_id=FROZEN_CONFIG.experiment_id,
        policy_bundle=bundle,
        measurements=measurements,
        parities=parities,
        source_digests=source_digests,
        attempt_root=attempt_root,
    )
    normalized = _manifest_to_payload(manifest)
    if canonical_manifest_sha256(normalized) != canonical_manifest_sha256(payload):
        raise ValueError("manifest payload is not canonical or has inconsistent evidence")
    return manifest


def write_manifest(path: str | Path, manifest: Exp421Manifest | Mapping[str, object]) -> Exp421Manifest:
    """Atomically write a validated manifest into its fresh attempt root."""
    target = Path(path).expanduser()
    if target.exists():
        raise FileExistsError(f"manifest target already exists: {target}")
    if isinstance(manifest, Exp421Manifest):
        payload = _manifest_to_payload(manifest)
    elif isinstance(manifest, Mapping):
        validated = validate_manifest(manifest)
        payload = _manifest_to_payload(validated)
        manifest = validated
    else:
        raise ValueError("manifest must be an Exp421Manifest or payload mapping")
    validated = validate_manifest(payload)
    root = Path(validated.attempt_root)
    if target.resolve(strict=False).parent != root.resolve(strict=False):
        raise ValueError("manifest target must be inside its declared attempt_root")
    if any(marker in str(target.resolve(strict=False)).lower() for marker in _OFFICIAL_EXP419_MARKERS):
        raise ValueError("manifest target must be outside official Exp419 roots")
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError("attempt_root must be fresh and empty")
    else:
        root.mkdir(parents=True, exist_ok=False)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            data = _canonical_bytes(payload)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # The target was checked before creating the temporary file.  replace
        # is atomic on the same filesystem; a subsequent call is rejected by
        # the initial existence check and cannot silently clobber evidence.
        os.replace(temp_name, target)
        temp_name = None
        with target.open("rb") as handle:
            decoded = json.loads(handle.read().decode("utf-8"))
        return validate_manifest(decoded)
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "build_manifest",
    "canonical_manifest_sha256",
    "validate_manifest",
    "write_manifest",
]
