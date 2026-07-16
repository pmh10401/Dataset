from __future__ import annotations

import hashlib
import csv
import json
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Mapping


POLICY_VERSION = "exp220-shadow-policy-v1"


def runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "scikit-learn", "torch", "pycatch22", "aeon"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _policy_hash(payload_without_hash: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload_without_hash)).hexdigest()


def freeze_policy(
    path: Path,
    *,
    artifacts: Mapping[str, Path],
    profiles: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    artifact_rows: dict[str, dict[str, object]] = {}
    for name, source in sorted(artifacts.items()):
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        artifact_rows[str(name)] = {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }

    body: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "experiments": {
            "baseline": "137",
            "review": "190",
            "shadow_hard": "195",
        },
        "seed": 20260718,
        "exp190": {
            "conservative_alpha_source": "exp177_risk_target_0.005",
            "canonical_knn_alpha": 0.005,
            "high": "exp178_conservative AND canonical_knn_005",
            "standard": "exp178_conservative MINUS canonical_knn_005",
            "exploratory_enabled": False,
        },
        "exp195": {
            "source": "exp190_high",
            "guard": "crossfit_train_normal_candidate_count == 0",
            "autonomous_alert": False,
        },
        "prediction_contract": {
            "contains_test_labels": False,
            "contains_test_position": False,
            "contains_test_length_policy": False,
            "contains_family_performance": False,
            "unregistered_profile": "onboarding_required",
        },
        "profiles": {str(key): dict(value) for key, value in sorted(profiles.items())},
        "artifacts": artifact_rows,
        "metadata": dict(metadata or {}),
    }
    frozen = {**body, "policy_sha256": _policy_hash(body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return frozen


def load_verified_policy(path: Path, *, verify_artifacts: bool = False) -> dict[str, object]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("policy_sha256", ""))
    body = {key: value for key, value in payload.items() if key != "policy_sha256"}
    actual = _policy_hash(body)
    if not expected or actual != expected:
        raise ValueError(
            f"policy hash mismatch: expected={expected or '<missing>'} actual={actual}"
        )
    if verify_artifacts:
        for name, row in payload.get("artifacts", {}).items():
            artifact_path = Path(row["path"])
            if not artifact_path.is_file():
                raise ValueError(f"artifact missing: {name} path={artifact_path}")
            artifact_sha = sha256_file(artifact_path)
            if artifact_sha != row["sha256"]:
                raise ValueError(
                    "artifact hash mismatch: "
                    f"name={name} expected={row['sha256']} actual={artifact_sha}"
                )
    return payload


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def default_artifact_paths(
    repo_root: Path, data_root: Path, database_path: Path
) -> dict[str, Path]:
    repo_root = Path(repo_root)
    data_root = Path(data_root)
    exp177_root = data_root / "exp177_185_unattended"
    exp186_root = data_root / "exp186_200_unattended"
    return {
        "univariate_train_database": Path(database_path),
        "exp137_frozen_results": repo_root
        / "results/exp137/experiment_137_operational_triage_results.csv",
        "exp137_routing_code": repo_root / "run_experiment_137_operational_triage.py",
        "exp153_score_manifest": repo_root
        / "outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv",
        "exp153_raw_scoring_bridge": repo_root
        / "experiments/exp153_157/exp153_157_bridge.py",
        "exp177_policy": exp177_root / "exp177_frozen_policy.json",
        "exp178_predictions": exp177_root / "exp178_frozen_predictions.jsonl",
        "exp178_manifest": exp177_root / "exp178_prediction_manifest.json",
        "exp179_feature_manifest": exp177_root / "exp179_canonical_feature_manifest.csv",
        "exp180_score_manifest": exp177_root / "exp180_canonical_score_manifest.csv",
        "exp190_predictions": exp186_root / "exp190_frozen_predictions.jsonl",
        "exp190_manifest": exp186_root / "exp190_prediction_manifest.json",
        "exp195_predictions": exp186_root / "exp195_frozen_predictions.jsonl",
        "exp195_manifest": exp186_root / "exp195_prediction_manifest.json",
        "exp195_train_guard": exp186_root / "train_guard_precompute.json",
        "exp177_core_code": repo_root / "experiments/exp177_185/core.py",
        "exp186_core_code": repo_root / "experiments/exp186_200/core.py",
        "exp220_policy_code": repo_root / "experiments/exp220_223/policy.py",
        "exp220_scoring_code": repo_root / "experiments/exp220_223/scoring.py",
        "exp220_prediction_code": repo_root / "experiments/exp220_223/prediction.py",
        "exp220_replay_code": repo_root / "experiments/exp220_223/replay.py",
        "exp221_storage_code": repo_root / "experiments/exp220_223/storage.py",
        "exp223_evaluation_code": repo_root / "experiments/exp220_223/evaluation.py",
        "exp220_223_cli_code": repo_root / "experiments/exp220_223/run_exp220_223.py",
    }


def freeze_default_policy(
    *,
    repo_root: Path,
    data_root: Path,
    database_path: Path,
    output_path: Path,
) -> dict[str, object]:
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve()
    exp177_root = data_root / "exp177_185_unattended"
    exp186_root = data_root / "exp186_200_unattended"
    baseline_path = repo_root / "results/exp137/experiment_137_operational_triage_results.csv"
    baseline = {
        row["dataset_name"]: row
        for row in _csv_rows(baseline_path)
        if int(row["train_normal_count"]) >= 20
    }
    if len(baseline) != 639:
        raise ValueError(f"eligible profile mismatch: {len(baseline)} != 639")
    feature_manifest_path = exp177_root / "exp179_canonical_feature_manifest.csv"
    feature_manifest = {
        row["dataset_name"]: row for row in _csv_rows(feature_manifest_path)
    }
    precompute_path = exp186_root / "train_guard_precompute.json"
    precompute = json.loads(precompute_path.read_text(encoding="utf-8"))["results"]
    if set(baseline) != set(feature_manifest) or set(baseline) != set(precompute):
        raise ValueError("profile coverage differs across frozen inputs")
    exp177_policy_path = exp177_root / "exp177_frozen_policy.json"
    exp177_policy = json.loads(exp177_policy_path.read_text(encoding="utf-8"))
    conservative_alpha = None
    for variant in exp177_policy["policy_variants"]:
        if float(variant["risk_target"]) == 0.005:
            conservative_alpha = float(variant["selected_alpha"])
            break
    if conservative_alpha is None:
        raise ValueError("Exp177 conservative alpha is missing")

    profiles = {
        profile_id: {
            "n_train": int(baseline[profile_id]["train_normal_count"]),
            "train_target_len": int(feature_manifest[profile_id]["train_target_len"]),
            "exp195_zero_train_guard": (
                sum(
                    int(value)
                    for value in precompute[profile_id]["candidate_counts_195"]
                )
                == 0
            ),
        }
        for profile_id in sorted(baseline)
    }
    artifacts = default_artifact_paths(repo_root, data_root, database_path)
    return freeze_policy(
        output_path,
        artifacts=artifacts,
        profiles=profiles,
        metadata={
            "eligible_profile_count": 639,
            "conservative_alpha": conservative_alpha,
            "score_rtol": 1e-6,
            "score_atol": 1e-8,
            "raw_query_scorer": "exp153 four-source crossfit + exp179 canonical KNN",
            "exp137_upstream_adapter_required": True,
            "runtime_versions": runtime_versions(),
            "end_to_end_train_only_provenance_validated": False,
            "shadow_execution_gate": "raw_replay_report_required",
        },
    )
