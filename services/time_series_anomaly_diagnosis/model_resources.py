from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .multi_evidence_review import DatasetReviewEvidence, parse_review_evidence_resource

ALLOWED_METHODS = frozenset({"minirocket_local_gap", "hydra_local_gap", "multirocket_local_gap"})


@dataclass(frozen=True)
class ModelResources:
    route_by_dataset: dict[str, str]
    recovery_datasets: frozenset[str]
    review_evidence: dict[str, DatasetReviewEvidence]


def verified_json(path: Path, expected_sha256: str) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"verified resource missing: {path.name}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"verified resource changed: {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"verified resource is not an object: {path.name}")
    return value


def parse_model_resources(
    route_payload: dict[str, object],
    recovery_payload: dict[str, object],
    review_payload: dict[str, object],
) -> ModelResources:
    evaluation = route_payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("route payload missing evaluation object")
    decisions = evaluation.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("route payload missing decisions list")

    route_by_dataset: dict[str, str] = {}
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("route decision entry is not an object")
        dataset = row.get("dataset")
        method = row.get("method")
        if not isinstance(dataset, str) or not isinstance(method, str):
            raise ValueError("route decision entry has invalid dataset or method type")
        route_by_dataset[dataset] = method

    if len(decisions) != 1117 or len(route_by_dataset) != 1117:
        raise ValueError("route catalog must contain 1117 unique datasets")
    if set(route_by_dataset.values()) - ALLOWED_METHODS:
        raise ValueError("route catalog contains an unknown method")

    recovery_rows = recovery_payload.get("rows")
    if not isinstance(recovery_rows, list):
        raise ValueError("recovery payload missing rows list")

    if len(recovery_rows) != 155:
        raise ValueError("recovery catalog must contain 155 rows")

    recovery_dataset_names: list[str] = []
    for row in recovery_rows:
        if not isinstance(row, dict) or row.get("status") != "ok":
            raise ValueError("recovery catalog row is invalid or status is not ok")
        dataset = row.get("dataset")
        if not isinstance(dataset, str):
            raise ValueError("recovery dataset name is not a string")
        recovery_dataset_names.append(dataset)

    recovery_datasets = frozenset(recovery_dataset_names)
    if len(recovery_datasets) != 155 or not recovery_datasets <= route_by_dataset.keys():
        raise ValueError("recovery datasets must be unique routed datasets")

    review_evidence = parse_review_evidence_resource(review_payload)
    if len(review_evidence) != 1117:
        raise ValueError("review evidence must contain 1117 unique datasets")

    return ModelResources(
        route_by_dataset=route_by_dataset,
        recovery_datasets=recovery_datasets,
        review_evidence=review_evidence,
    )


def load_model_resources(resource_directory: Path) -> ModelResources:
    directory = Path(resource_directory)
    manifest = json.loads((directory / "model_manifest.json").read_text("utf-8"))
    if manifest.get("schema") != "service_model_resources_v2":
        raise ValueError("model resource manifest schema mismatch")
    route_payload = verified_json(
        directory / "routing_catalog.json", manifest["files"]["routing_catalog.json"]
    )
    recovery_payload = verified_json(
        directory / "recovery_catalog.json", manifest["files"]["recovery_catalog.json"]
    )
    review_payload = verified_json(
        directory / "review_evidence.json", manifest["files"]["review_evidence.json"]
    )
    return parse_model_resources(route_payload, recovery_payload, review_payload)
